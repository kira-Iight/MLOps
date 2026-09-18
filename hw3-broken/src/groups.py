"""Формирование ключа группы для сплита и проверки контаминации.

Один модуль на два места специально: если ключ группы разъедется, сплит
и проверка контаминации начнут мерить разные вещи, и проверка станет
зелёной при реальном пересечении.
"""

import hashlib

from src.schema import Example
from src.textnorm import normalize_group


def make_group_key(example: Example, granularity: int = 1) -> str:
    """Ключ группы строки.

    При granularity == 1 — тема целиком (одна группа на тему).
    При granularity > 1 — тема дробится на N подгрупп по хэшу id.
    Это нужно для датасетов с малым числом тем (BBC News — 5 категорий),
    иначе пропорции 80/10/10 не делятся и val оказывается пустым.
    """
    base = normalize_group(example.topic)
    if granularity <= 1:
        return base
    digest = int(hashlib.md5(example.id.encode("utf-8")).hexdigest(), 16)
    return f"{base}__{digest % granularity}"
