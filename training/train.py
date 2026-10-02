"""
Train the Latin-script language model.

    python training/train.py

Steps
  1. (re)generate the augmented dataset from data/seed/*.txt
  2. train baseline models (Multinomial Naive Bayes, Linear SVM) for comparison
  3. train the final model (char+word TF-IDF + Logistic Regression)
  4. evaluate on the held-out test set, save metrics + confusion matrix
  5. save the model to models/latin_model.joblib
"""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from data.generate_dataset import main as generate_dataset  # noqa: E402
from detector.model import MODEL_PATH, build_pipeline, preprocess_many, save_model  # noqa: E402


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def plot_confusion(cm, labels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping confusion-matrix image")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - Latin-script model (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    print("== 1. Generating dataset ==")
    generate_dataset()
    Xtr_raw, ytr = read_csv(ROOT / "data/processed/train.csv")
    Xte_raw, yte = read_csv(ROOT / "data/processed/test.csv")
    Xtr, Xte = preprocess_many(Xtr_raw), preprocess_many(Xte_raw)

    print("\n== 2. Baselines ==")
    baselines = {
        "Naive Bayes (char 1-4)": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4))),
            ("clf", MultinomialNB(alpha=0.1))]),
        "Linear SVM (char 1-4)": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)),
            ("clf", LinearSVC(C=1.0))]),
    }
    results = {}
    for name, pipe in baselines.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {"accuracy": round(accuracy_score(yte, pred), 4),
                         "macro_f1": round(f1_score(yte, pred, average="macro"), 4)}
        print(f"{name:28s} acc={results[name]['accuracy']:.4f}  macroF1={results[name]['macro_f1']:.4f}")

    print("\n== 3. Final model: char(1-5)+word(1-2) TF-IDF + Logistic Regression ==")
    t0 = time.time()
    model = build_pipeline()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    results["Logistic Regression (char 1-5 + word 1-2) [FINAL]"] = {
        "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
    print(f"accuracy={acc:.4f}  macroF1={f1:.4f}  (trained in {train_time:.1f}s)\n")
    report = classification_report(yte, pred, digits=4)
    print(report)

    labels = sorted(set(ytr))
    cm = confusion_matrix(yte, pred, labels=labels)
    (ROOT / "models").mkdir(exist_ok=True)
    plot_confusion(cm, labels, ROOT / "models/confusion_matrix.png")

    metrics = {
        "train_size": len(Xtr), "test_size": len(Xte), "labels": labels,
        "model_comparison": results,
        "classification_report": classification_report(yte, pred, digits=4, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    (ROOT / "models/metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "models/classification_report.txt").write_text(report)

    print("== 4. Retraining final model on train+test and saving ==")
    model = build_pipeline()
    model.fit(Xtr + Xte, ytr + yte)
    save_model(model)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
