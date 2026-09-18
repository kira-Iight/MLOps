"""Стадия split: разбиение на train/val/test."""

import json
import random
import time
from pathlib import Path
from src.groups import make_group_key

from src.config import load_params
from src.contamination import report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def row_split(count: int, ratios: dict[str, float], seed: int) -> list[str]:
    """Раздать строкам метки сплита в заданных долях."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    labels = [""] * count
    start = 0
    names = list(ratios)
    for i, name in enumerate(names):
        stop = count if i == len(names) - 1 else start + round(count * ratios[name])
        for pos in order[start:stop]:
            labels[pos] = name
        start = stop
    return labels


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    sizes: dict[str, int] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        sizes[key] = sizes.get(key, 0) + 1

    # Групповой сплит: строки одной группы не разъезжаются по train/test.
    # Ключ группы формируется одним модулем — src.groups.make_group_key —
    # чтобы split и contamination мерили одно и то же.
    granularity = cfg.get("group_granularity", 1)
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        key = make_group_key(ex, granularity)
        groups.setdefault(key, []).append(ex)


    keys = list(groups.keys())
    random.Random(cfg["seed"]).shuffle(keys)

    names = list(cfg["ratios"])
    buckets: dict[str, list[Example]] = {name: [] for name in names}
    start = 0
    for i, name in enumerate(names):
        stop = len(keys) if i == len(names) - 1 else start + round(len(keys) * cfg["ratios"][name])
        for key in keys[start:stop]:
            buckets[name].extend(groups[key])
        start = stop

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
        granularity=granularity,
    )

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(groups),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(groups)}, {metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()
