"""Генерация golden/behavioral/training-index из датасета ДЗ 3.

Golden берётся из test-сплита (не train!), чтобы не было утечки.
Все три файла — JSONL с полями по контракту EVALUATION.md.
"""
import json
import random
from pathlib import Path

HW3 = Path.home() / "Downloads/hw1-broken/hw3-broken/data"
OUT = Path("data/eval")
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def to_eval_row(record):
    text = record["messages"][1]["content"]
    topic = record["topic"]
    choices = ["business", "entertainment", "politics", "sport", "tech",
               "world", "sports", "sci/tech"]
    known = set(choices)
    if topic not in known:
        return None
    slices = [f"topic_{topic}", "all", "news"]
    slices.append("short" if len(text) < 200 else "long")
    return {
        "id": record["id"],
        "group_id": record["id"].rsplit("_", 1)[0] if "_" in record["id"] else record["id"],
        "input": f"Текст новости:\n{text[:1500]}\n\nОпредели категорию: business, entertainment, politics, sport, tech, world, sports, sci/tech.",
        "expected": topic,
        "choices": choices,
        "slices": slices,
        "source": "ДЗ 3 test-сплит BBC News + ag_news",
    }


# training-index: все train-вопросы
train_records = read_jsonl(HW3 / "train.jsonl")
train_index = [
    {"id": r["id"], "group_id": r["topic"], "input": r["messages"][1]["content"]}
    for r in train_records
]
(OUT / "training-index.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in train_index),
    encoding="utf-8",
)
print(f"training-index: {len(train_index)} строк")

# golden: из test-сплита
test_records = read_jsonl(HW3 / "test.jsonl")
random.seed(42)
random.shuffle(test_records)

golden = []
for r in test_records:
    row = to_eval_row(r)
    if row:
        golden.append(row)
    if len(golden) >= 60:
        break

(OUT / "golden.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in golden),
    encoding="utf-8",
)
print(f"golden: {len(golden)} строк")

# behavioral: 3 кейса
behavioral = [
    {
        "id": "behavior-format",
        "group_id": "behavior-format",
        "input": "Текст новости:\nThe company reported record quarterly earnings, driven by strong cloud revenue.\n\nОпредели категорию: business, entertainment, politics, sport, tech, world, sports, sci/tech. Ответ строго JSON.",
        "expected": "business",
        "choices": ["business", "entertainment", "politics", "sport", "tech", "world", "sports", "sci/tech"],
        "slices": ["behavior"],
        "type": "format",
        "source": "behavioral: строгий формат",
    },
    {
        "id": "behavior-paraphrase",
        "group_id": "behavior-paraphrase",
        "input": "Ниже новостной отрывок. Классифицируй его по одной из тем: business, entertainment, politics, sport, tech, world, sports, sci/tech.\n\nОтрывок: The team won the championship final in a dramatic match.",
        "expected": "sports",
        "choices": ["business", "entertainment", "politics", "sport", "tech", "world", "sports", "sci/tech"],
        "slices": ["behavior"],
        "type": "paraphrase",
        "source": "behavioral: перефраз инструкции",
    },
    {
        "id": "behavior-refusal",
        "group_id": "behavior-refusal",
        "input": "Сочини стихотворение про осень.",
        "expected": "REFUSE",
        "choices": ["business", "entertainment", "politics", "sport", "tech", "world", "sports", "sci/tech", "REFUSE"],
        "slices": ["behavior"],
        "type": "refusal",
        "source": "behavioral: запрос вне темы",
    },
]
(OUT / "behavioral.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in behavioral),
    encoding="utf-8",
)
print(f"behavioral: {len(behavioral)} строк")