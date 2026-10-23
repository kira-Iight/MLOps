"""Вход стадии train — выход стадии tokenize из ДЗ 4.

Формат файла: torch.save словаря {"examples": [...], "pad_token_id", ...},
где у каждого примера input_ids / attention_mask / labels — списки int,
а labels = -100 на промпте. Здесь ничего не токенизируется заново:
маску и шаблон уже проверили в ДЗ 4.
"""

import random
from pathlib import Path

import torch

LABEL_PAD_ID = -100


def load_split(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"Нет {p}.\n"
            "Вход стадии train — выход стадии tokenize из ДЗ 4 (data/tokenized/*.pt).\n"
            "Укажите пути к своим файлам в params.yaml, секция data."
        )
    blob = torch.load(p, weights_only=True)
    if not blob.get("examples"):
        raise SystemExit(f"{p}: в файле нет примеров")
    return blob


def pad_batch(features: list[dict], pad_id: int, max_len: int | None = None) -> dict[str, torch.Tensor]:
    """Динамический паддинг слева, labels на паддинге -100.

    max_len — потолок длины для памяти: логиты на весь словарь растут
    линейно по длине, при seq=999 на MPS с 8 ГБ backward не влезает.
    """
    if max_len is not None:
        # ИСПРАВЛЕНО: ответ ассистента стоит в КОНЦЕ последовательности
        # (после "Определи категорию: ..."), а обрезка [:max_len] режет
        # именно конец — ответ исчезает, все labels становятся -100.
        # Берём последние max_len токенов: теряем начало новостного текста,
        # но сохраняем инструкцию и ответ.
        trimmed = []
        for f in features:
            cut = {
                "input_ids": f["input_ids"][-max_len:],
                "attention_mask": f["attention_mask"][-max_len:],
                "labels": f["labels"][-max_len:],
            }
            # Если после обрезки ответ всё равно потерян — пропускаем.
            if any(l != LABEL_PAD_ID for l in cut["labels"]):
                trimmed.append(cut)
        features = trimmed
        if not features:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
            }
    width = max(len(f["input_ids"]) for f in features)

    def left(seq, value):
        return [value] * (width - len(seq)) + list(seq)

    return {
        "input_ids": torch.tensor([left(f["input_ids"], pad_id) for f in features]),
        "attention_mask": torch.tensor([left(f["attention_mask"], 0) for f in features]),
        "labels": torch.tensor([left(f["labels"], LABEL_PAD_ID) for f in features]),
    }


def batches(examples: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int,
            max_len: int | None = None):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield pad_batch([examples[j] for j in order[i:i + batch_size]], pad_id, max_len)
