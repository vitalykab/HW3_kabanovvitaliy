"""Save the inputs needed to verify truncation and dynamic padding in answers.md."""

import json
from pathlib import Path

from transformers import AutoTokenizer
from common import data, save


def main():
    tok = AutoTokenizer.from_pretrained("models/best")
    train = data("train")
    short = []
    long = None
    for row in train:
        ids = tok(row["text"], verbose=False)["input_ids"]
        if (
            3 <= len(ids) <= 10
            and all(len(ids) != length for _, length in short)
            and len(short) < 2
        ):
            short.append((row, len(ids)))
        if long is None and 133 <= len(ids) <= 140:
            long = (row, ids)
        if long and len(short) == 2:
            break
    row, ids = long
    limited = tok(row["text"], truncation=True, max_length=128)["input_ids"]
    encoded = tok(
        [r["text"] for r, _ in short], padding=True, truncation=True, max_length=128
    )
    before = tok.convert_ids_to_tokens(ids)
    after = tok.convert_ids_to_tokens(limited)
    # Full text is available locally after reconstructing the dataset.
    Path("data/truncation_example.json").write_text(
        json.dumps(
            dict(
                id=row["id"],
                original=row["text"],
                before_tokens=before,
                after_tokens=after,
                after_text=tok.decode(limited, skip_special_tokens=True),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    save(
        "token_examples.json",
        dict(
            long=dict(
                id=row["id"],
                n_before=len(before),
                n_after=len(after),
                before_tail=before[-13:],
                after_tail=after[-8:],
            ),
            padding=[
                dict(id=r["id"], attention_mask=mask)
                for (r, _), mask in zip(short, encoded["attention_mask"])
            ],
        ),
    )


if __name__ == "__main__":
    main()
