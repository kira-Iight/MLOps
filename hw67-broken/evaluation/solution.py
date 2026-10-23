"""Реализация двух функций: score и quality_gate.

Контракт и примеры: EVALUATION.md и tests/test_eval.py.
"""

import json
import math


def _parse_answer(raw):
    """Строгий JSON: единственное поле answer, строка.

    Возвращает строку-answer или None, если формат невалиден. Совпадение
    с expected и вхождение в choices проверяются в score отдельно.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"answer"}:
        return None
    if not isinstance(parsed["answer"], str):
        return None
    return parsed["answer"]


def _new_stats():
    return {"n": 0, "correct": 0, "valid": 0, "errors": 0,
            "accuracy": 0.0, "format_rate": 0.0}


def _add(stats, valid, correct, error):
    stats["n"] += 1
    stats["valid"] += 1 if valid else 0
    stats["correct"] += 1 if correct else 0
    stats["errors"] += 1 if error is not None else 0


def _finalize(stats):
    n = stats["n"]
    stats["accuracy"] = stats["correct"] / n if n else 0.0
    stats["format_rate"] = stats["valid"] / n if n else 0.0
    return stats


def score(rows, outputs):
    """Посчитать overall, slices и per_item по ответам.

    rows: [{id, group_id, input, expected, choices, slices, source}, ...]
    outputs: [{id, raw, error, latency_seconds}, ...]

    Пустой набор, дубли id и несовпадение множеств id — ValueError.
    Ошибка генерации/пустой/невалидный ответ остаётся в знаменателе:
    n = len(rows), correct/valid — счётчики, accuracy = correct/n.
    """
    if not rows:
        raise ValueError("Пустой набор rows")
    if not outputs:
        raise ValueError("Пустой набор outputs")

    row_ids = [r["id"] for r in rows]
    out_ids = [o["id"] for o in outputs]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Дублирующиеся id в rows")
    if len(set(out_ids)) != len(out_ids):
        raise ValueError("Дублирующиеся id в outputs")
    if set(row_ids) != set(out_ids):
        raise ValueError("id в rows и outputs не совпадают")

    by_id = {o["id"]: o for o in outputs}
    overall = _new_stats()
    slices: dict[str, dict] = {}
    per_item = []

    for row in rows:
        out = by_id[row["id"]]
        raw = out.get("raw", "")
        error = out.get("error")
        answer = _parse_answer(raw)
        valid = answer is not None and answer in row["choices"]
        correct = valid and answer == row["expected"]

        _add(overall, valid, correct, error)
        for s in row["slices"]:
            slices.setdefault(s, _new_stats())
            _add(slices[s], valid, correct, error)

        per_item.append({
            "id": row["id"],
            "raw": raw,
            "parsed": answer,
            "valid": valid,
            "correct": correct,
            "error": error,
            "slices": list(row["slices"]),
        })

    for s in slices.values():
        _finalize(s)
    _finalize(overall)
    return {"overall": overall, "slices": slices, "per_item": per_item}


def _finite(*values):
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Нечисловая или NaN метрика: {v!r}")


def _check_report(report, name):
    if not isinstance(report, dict):
        raise ValueError(f"{name}: отчёт не словарь")
    if report.get("status") != "complete":
        raise ValueError(f"{name}: status != complete")
    if not report.get("model"):
        raise ValueError(f"{name}: не указана модель")
    overall = report.get("overall")
    per_item = report.get("per_item")
    slices = report.get("slices")
    if not isinstance(overall, dict) or not isinstance(per_item, list) or not isinstance(slices, dict):
        raise ValueError(f"{name}: нет overall/per_item/slices")

    n = overall.get("n")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name}: n <= 0")
    if len(per_item) != n:
        raise ValueError(f"{name}: len(per_item) != n")
    if overall.get("errors", 0) != 0:
        raise ValueError(f"{name}: есть ошибки генерации")

    correct = sum(1 for p in per_item if p["correct"])
    valid = sum(1 for p in per_item if p["valid"])
    if overall.get("correct") != correct or overall.get("valid") != valid:
        raise ValueError(f"{name}: overall не соответствует per_item")
    _finite(overall.get("accuracy"), overall.get("format_rate"))
    if abs(overall["accuracy"] - correct / n) > 1e-9:
        raise ValueError(f"{name}: accuracy != correct/n")
    if abs(overall["format_rate"] - valid / n) > 1e-9:
        raise ValueError(f"{name}: format_rate != valid/n")

    if len(slices) < 3:
        raise ValueError(f"{name}: меньше трёх срезов")
    for sname, stats in slices.items():
        sn = stats.get("n")
        if not isinstance(sn, int) or sn <= 0:
            raise ValueError(f"{name}/{sname}: пустой срез")
        rows_in_slice = [p for p in per_item if sname in p["slices"]]
        if len(rows_in_slice) != sn:
            raise ValueError(f"{name}/{sname}: n среза не совпадает с per_item")
        s_correct = sum(1 for p in rows_in_slice if p["correct"])
        s_valid = sum(1 for p in rows_in_slice if p["valid"])
        if stats.get("correct") != s_correct or stats.get("valid") != s_valid:
            raise ValueError(f"{name}/{sname}: counts не соответствуют per_item")
        _finite(stats.get("accuracy"), stats.get("format_rate"))
        if abs(stats["accuracy"] - s_correct / sn) > 1e-9:
            raise ValueError(f"{name}/{sname}: accuracy != correct/n")
        if abs(stats["format_rate"] - s_valid / sn) > 1e-9:
            raise ValueError(f"{name}/{sname}: format_rate != valid/n")
    return per_item, slices


def quality_gate(baseline, candidate, critical_slices):
    """Правило допуска в champion.

    PASS: {passed: True, ...}. Любая несовместимость/регрессия/missing
    /NaN/ошибка генерации → ValueError с причиной. Порог регрессии — ноль.
    """
    if not critical_slices:
        raise ValueError("Нет критических срезов")

    for key in ("protocol_sha256", "golden_sha256", "baseline_spec_sha256"):
        if not baseline.get(key) or not candidate.get(key):
            raise ValueError(f"{key}: отсутствует")
        if baseline[key] != candidate[key]:
            raise ValueError(f"{key}: не совпадает")

    b_items, b_slices = _check_report(baseline, "baseline")
    c_items, c_slices = _check_report(candidate, "candidate")

    if [p["id"] for p in b_items] != [p["id"] for p in c_items]:
        raise ValueError("id per_item baseline и candidate не совпадают")
    if baseline["overall"]["n"] < 50:
        raise ValueError("golden: минимум 50 примеров")

    b_o, c_o = baseline["overall"], candidate["overall"]
    if c_o["accuracy"] < b_o["accuracy"]:
        raise ValueError(
            f"accuracy регрессия: {c_o['accuracy']:.4f} < {b_o['accuracy']:.4f}"
        )
    if c_o["format_rate"] < b_o["format_rate"]:
        raise ValueError(
            f"format_rate регрессия: {c_o['format_rate']:.4f} < {b_o['format_rate']:.4f}"
        )

    if set(b_slices) != set(c_slices):
        raise ValueError("Набор срезов baseline и candidate не совпадает")
    for sname in b_slices:
        if b_slices[sname]["n"] != c_slices[sname]["n"]:
            raise ValueError(f"Срез {sname}: разный n")

    for sname in critical_slices:
        if sname not in b_slices or sname not in c_slices:
            raise ValueError(f"Критический срез {sname} отсутствует")
        if c_slices[sname]["accuracy"] < b_slices[sname]["accuracy"]:
            raise ValueError(
                f"Срез {sname}: регрессия "
                f"{c_slices[sname]['accuracy']:.4f} < {b_slices[sname]['accuracy']:.4f}"
            )

    beh = candidate.get("behavioral")
    if not isinstance(beh, dict) or not isinstance(beh.get("overall"), dict):
        raise ValueError("behavioral отсутствует в candidate")
    b_o_beh = beh["overall"]
    beh_n = b_o_beh.get("n", 0)
    beh_correct = b_o_beh.get("correct", 0)
    if beh_n < 3:
        raise ValueError(f"behavioral: минимум 3 кейса, есть {beh_n}")
    if beh_correct != beh_n:
        raise ValueError(f"behavioral: пройдено {beh_correct}/{beh_n}")

    return {
        "passed": True,
        "baseline": {
            "accuracy": b_o["accuracy"],
            "format_rate": b_o["format_rate"],
        },
        "candidate": {
            "accuracy": c_o["accuracy"],
            "format_rate": c_o["format_rate"],
        },
        "critical_slices": list(critical_slices),
        "behavioral": {"n": beh_n, "correct": beh_correct},
    }
