"""Train, validate and save the best checkpoint. Test is evaluated in a later process."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from common import LABELS, R, data, device, macro, protocol, save, select_ids


def tokenize(dataset, tokenizer):
    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=128)
        enc["labels"] = [LABELS.index(label) for label in batch["label"]]
        return enc

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def loader(dataset, tokenizer, batch=32, shuffle=False):
    return DataLoader(
        tokenize(dataset, tokenizer),
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


@torch.inference_mode()
def predict(model, batches, target):
    model.eval()
    labels = []
    for batch in batches:
        inputs = {k: v.to(target) for k, v in batch.items() if k != "labels"}
        labels.extend(model(**inputs).logits.argmax(-1).tolist())
    return [LABELS[i] for i in labels]


@torch.inference_mode()
def reference(model, tokenizer, row, target):
    model.eval()
    sample = tokenizer(
        row["text"], return_tensors="pt", max_length=128, truncation=True
    ).to(target)
    probs = model(**sample).logits.softmax(-1)[0].tolist()
    return dict(
        id=row["id"],
        label=LABELS[probs.index(max(probs))],
        probabilities=probs,
        device=target,
    )


def main():
    set_seed(42)
    target = device()
    tok = AutoTokenizer.from_pretrained("models/bert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/bert",
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={v: i for i, v in enumerate(LABELS)},
    ).to(target)
    train, val = data("train"), data("validation")
    batches, validation = loader(train, tok, shuffle=True), loader(val, tok)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    log_dir = R / "tensorboard"
    log_dir.mkdir(exist_ok=True)
    run_dir = TemporaryDirectory(dir=log_dir)
    writer = SummaryWriter(log_dir=run_dir.name)
    best, history, step = -1.0, [], 0
    for epoch in (1, 2):
        model.train()
        for batch in batches:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{k: v.to(target) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            writer.add_scalar("train/loss", loss.item(), step)
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        predictions = predict(model, validation, target)
        score = macro(val["label"], predictions)
        history.append(dict(epoch=epoch, macro_f1=score))
        writer.add_scalar("validation/macro_f1", score, step)
        for row, pred in zip(val, predictions):
            if row["label"] != pred:
                writer.add_text(
                    "validation/error",
                    f'{row["id"]}, true={row["label"]}, predicted={pred}',
                    step,
                )
                break
        if score > best:
            best, best_epoch = score, epoch
            model.save_pretrained("models/best")
            tok.save_pretrained("models/best")
            row = select_ids(val, [protocol()["reload_validation_id"]])[0]
            save("before_reload.json", reference(model, tok, row, target))
        writer.flush()
    writer.close()
    (event_file,) = Path(run_dir.name).glob("events.out.tfevents.*")
    event_file.replace(log_dir / "events.out.tfevents.rubert")
    run_dir.cleanup()
    save(
        "bert_selection.json",
        dict(history=history, best_epoch=best_epoch, device=target),
    )


if __name__ == "__main__":
    main()
