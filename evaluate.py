import joblib
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import threshold_stats
from bert import loader, predict
from common import LABELS, data, device, macro, protocol, read, save, table
from generator import metrics


def main():
    baseline, bert, llm = (
        read(name + "_selection.json") for name in ("baseline", "bert", "llm")
    )
    test = data("test")
    linear = joblib.load("models/tfidf.joblib")
    prob = linear.predict_proba(test["text"])
    tf_pred = linear.classes_[prob.argmax(axis=1)].tolist()
    negative = prob[:, list(linear.classes_).index("negative")]
    target = device()
    tokenizer = AutoTokenizer.from_pretrained("models/best")
    model = AutoModelForSequenceClassification.from_pretrained("models/best").to(target)
    bert_pred = predict(model, loader(test, tokenizer), target)
    full = [
        dict(id=i, true=y, tfidf=p, bert=b, p_negative=float(n))
        for i, y, p, b, n in zip(
            test["id"], test["label"], tf_pred, bert_pred, negative
        )
    ]
    table("test_predictions.csv", full)
    lookup = {r["id"]: r for r in full}
    raw = read("llm_test.json")
    assert [r["id"] for r in raw] == protocol()["generator_test_ids"]
    common = []
    for r in raw:
        item = lookup[r["id"]]
        assert r["true"] == item["true"]
        common.append(dict(**item, llm=r["prediction"]))
    table("common_predictions.csv", common)
    results = []

    def add(
        split,
        method,
        n,
        score,
        selected="",
        valid_f1="",
        valid_share="",
        per=("", "", ""),
        precision="",
        recall="",
    ):
        results.append(
            dict(
                split=split,
                method=method,
                n=n,
                macro_f1=score,
                selected=selected,
                negative_f1=per[0],
                neutral_f1=per[1],
                positive_f1=per[2],
                valid_macro_f1=valid_f1,
                valid_share=valid_share,
                precision=precision,
                recall=recall,
            )
        )

    for kind, score in baseline["validation_macro_f1"].items():
        add("validation", "tfidf_" + kind, 8578, score, kind == baseline["selected"])
    for h in bert["history"]:
        add(
            "validation",
            f'bert_epoch_{h["epoch"]}',
            8578,
            h["macro_f1"],
            h["epoch"] == bert["best_epoch"],
        )
    for mode, m in llm["metrics"].items():
        add(
            "validation_30",
            mode,
            30,
            m["macro_all"],
            mode == llm["selected"],
            m["macro_valid"],
            m["valid_share"],
        )
    matrices = {}
    for method, p in [("tfidf", tf_pred), ("bert", bert_pred)]:
        add(
            "test",
            method,
            len(test),
            macro(test["label"], p),
            per=f1_score(
                test["label"], p, labels=LABELS, average=None, zero_division=0
            ),
        )
        matrices[method] = confusion_matrix(test["label"], p, labels=LABELS).tolist()
        add(
            "test_150",
            method,
            len(common),
            macro([r["true"] for r in common], [r[method] for r in common]),
        )
    m = metrics(raw)
    add(
        "test_150",
        llm["selected"],
        len(raw),
        m["macro_all"],
        valid_f1=m["macro_valid"],
        valid_share=m["valid_share"],
    )
    chosen = baseline["threshold"]
    routing = (
        threshold_stats(test["label"], negative, chosen["threshold"])
        if chosen
        else None
    )
    if routing:
        add(
            "test",
            "manual_review",
            len(test),
            "",
            precision=routing["precision"],
            recall=routing["recall"],
        )
    save("test_routing.json", routing)
    save(
        "confusion_matrices.json",
        dict(order=LABELS, rows="true", columns="predicted", matrices=matrices),
    )
    table("metrics.csv", results)


if __name__ == "__main__":
    main()
