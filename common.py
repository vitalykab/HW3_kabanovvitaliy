"""Shared data protocol and output formats. Run scripts from the project root."""

import csv
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import f1_score

R = Path("results")
R.mkdir(exist_ok=True)
LABELS = ["negative", "neutral", "positive"]


def data(split):
    return load_dataset(
        "json",
        data_files=str(Path("data") / f"{split}.jsonl"),
        split="train",
        cache_dir=".cache/datasets",
    )


def protocol():
    return json.loads(Path("data/protocol.json").read_text())


def select_ids(dataset, ids):
    positions = {key: i for i, key in enumerate(dataset["id"])}
    return dataset.select([positions[key] for key in ids])


def macro(y, predictions):
    return float(
        f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)
    )


def save(name, obj):
    (R / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def read(name):
    return json.loads((R / name).read_text())


def table(name, rows):
    with (R / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
