from statistics import median
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert import reference
from common import data, device, protocol, read, save, select_ids, table


def sync(target):
    if target == "mps":
        torch.mps.synchronize()
    elif target == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def main():
    target = device()
    val = data("validation")
    p = protocol()
    tok = AutoTokenizer.from_pretrained("models/best")
    model = (
        AutoModelForSequenceClassification.from_pretrained("models/best")
        .to(target)
        .eval()
    )
    before = read("before_reload.json")
    row = select_ids(val, [p["reload_validation_id"]])[0]
    after = reference(model, tok, row, target)
    diff = float(
        np.max(np.abs(np.array(before["probabilities"]) - after["probabilities"]))
    )
    assert (
        after["device"] == before["device"]
        and after["label"] == before["label"]
        and diff <= 1e-5
    )
    save("reload.json", dict(before=before, after=after, max_abs_diff=diff))
    texts = list(select_ids(val, p["latency_validation_ids"])["text"])
    timings = []
    for size in [1, 2, 4, 8, 16]:
        batches = [
            tok(
                texts[i : i + size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(target)
            for i in range(0, len(texts), size)
        ]
        seconds = []
        for run in range(4):
            sync(target)
            start = perf_counter()
            for batch in batches:
                model(**batch)
            sync(target)
            if run:
                seconds.append(perf_counter() - start)
        timings.append(
            dict(
                batch_size=size,
                device=target,
                n=len(texts),
                run1=seconds[0],
                run2=seconds[1],
                run3=seconds[2],
                median_seconds=median(seconds),
            )
        )
    table("latency.csv", timings)


if __name__ == "__main__":
    main()
