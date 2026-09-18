"""Стадия collect: новостной датасет → data/raw.jsonl.

Поддерживает несколько источников (BBC News, ag_news). Версия датасета
(v1/v2) определяется в params.yaml — sources[version].

Контракт выхода: JSONL со строками {"id", "topic", "messages": [system, user, assistant]}.
"""

import hashlib
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.config import load_params


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера."""
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def load_source(source: str) -> list[dict]:
    """Загрузить один источник и вернуть плоский список строк."""
    rows = []
    if source == "Kessel/bbcnews":
        ds = load_dataset(source, split="train")
        for idx, row in enumerate(ds):
            rows.append({
                "prefix": "bbc",
                "idx": idx,
                "topic": row["Category"],
                "text": row["Text"],
            })
    elif source == "fancyzhx/ag_news":
        ds = load_dataset(source, split="train")
        label_names = ds.features["label"].names
        for idx, row in enumerate(ds):
            rows.append({
                "prefix": "ag",
                "idx": idx,
                "topic": label_names[row["label"]].lower(),
                "text": row["text"],
            })
    else:
        raise SystemExit(f"неизвестный источник: {source!r}")
    return rows


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    version = cfg["version"]
    n_rows = cfg["n_rows"]
    variants = cfg["system_prompts"]

    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")

    sources = cfg["sources"][version]
    print(f"Загрузка {len(sources)} источника(ов) для версии {version}: {sources}")

    all_rows: list[dict] = []
    for source in sources:
        all_rows.extend(load_source(source))
    print(f"Всего загружено {len(all_rows)} строк из источников")

    limit = n_rows if n_rows else len(all_rows)

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = 0
    prompts_used: set[str] = set()

    with out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            if written >= limit:
                break

            example_id = f"{row['prefix']}_{row['idx']:05d}"
            topic = row["topic"]
            text = row["text"].strip()

            topics = cfg.get("topics")
            if topics is not None and topic not in topics:
                continue

            min_chars = params["clean"]["min_user_chars"]
            max_chars = params["clean"]["max_user_chars"]
            if not (min_chars <= len(text) <= max_chars):
                continue

            prompt = pick_prompt(example_id, variants)
            prompts_used.add(prompt)

            record = {
                "id": example_id,
                "topic": topic,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": f"Категория: {topic}"},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    metrics = {
        "version": version,
        "sources": sources,
        "rows_scanned": len(all_rows),
        "rows_written": written,
        "system_prompt_variants": len(prompts_used),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: версия {version}, записано {written} строк, "
        f"вариантов инструкции {len(prompts_used)}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()