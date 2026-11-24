import itertools
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import LABELS, data, macro, save, table


def normalized(text):
    return " ".join((text or "").lower().replace("ё", "е").split())


def threshold_stats(labels, probability, threshold):
    negative = np.asarray(labels) == "negative"
    flag = probability >= threshold
    tp = int((flag & negative).sum())
    fp = int((flag & ~negative).sum())
    fn = int((~flag & negative).sum())
    return dict(
        threshold=threshold,
        sent=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )


def choose_threshold(rows):
    active = [r for r in rows if r["sent"]]
    eligible = [r for r in active if r["precision"] >= 0.8]
    if eligible:
        return max(eligible, key=lambda r: (r["recall"], r["threshold"]))
    return max(
        active,
        key=lambda r: (r["precision"], r["recall"], r["threshold"]),
        default=None,
    )


def main():
    splits = {s: data(s) for s in ("train", "validation", "test")}
    ids, texts, audit, overlaps = {}, {}, {}, {}
    for split, ds in splits.items():
        ids[split] = set(ds["id"])
        norm = [normalized(t) for t in ds["text"]]
        texts[split] = set(norm)
        audit[split] = dict(
            n=len(ds),
            labels=dict(Counter(ds["label"])),
            empty=norm.count(""),
            repeated_ids=len(ds) - len(ids[split]),
            repeated_texts=len(ds) - len(texts[split]),
        )
    for a, b in itertools.combinations(splits, 2):
        overlaps[f"{a}/{b}"] = dict(
            ids=len(ids[a] & ids[b]), texts=len(texts[a] & texts[b])
        )
    train, val = splits["train"], splits["validation"]
    lengths = {}
    for label in LABELS:
        counts = [len(r["text"].split()) for r in train if r["label"] == label]
        lengths[label] = dict(
            zip(["median", "p95"], np.percentile(counts, [50, 95]).tolist())
        )
    save("data_audit.json", dict(splits=audit, overlaps=overlaps, train_words=lengths))
    models, scores = {}, {}
    for kind, ngrams in [("word", (1, 2)), ("char", (3, 5))]:
        model = make_pipeline(
            TfidfVectorizer(
                analyzer=kind,
                ngram_range=ngrams,
                min_df=2,
                max_features=60000,
                sublinear_tf=True,
            ),
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
        )
        model.fit(train["text"], train["label"])
        scores[kind] = macro(val["label"], model.predict(val["text"]))
        models[kind] = model
    winner = max(scores, key=scores.get)
    model = models[winner]
    probability = model.predict_proba(val["text"])[
        :, list(model.classes_).index("negative")
    ]
    rows = [
        threshold_stats(val["label"], probability, i / 100) for i in range(5, 100, 5)
    ]
    table("threshold_validation.csv", rows)
    save(
        "baseline_selection.json",
        dict(
            validation_macro_f1=scores,
            selected=winner,
            threshold=choose_threshold(rows),
        ),
    )
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/tfidf.joblib")


if __name__ == "__main__":
    main()
