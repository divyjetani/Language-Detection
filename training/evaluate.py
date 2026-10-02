"""
End-to-end evaluation of the full hybrid detector (script rules + ML + lexicon)
on hand-written sentences that do NOT appear in the training seeds.

    python training/evaluate.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector import LanguageDetector  # noqa: E402


def main():
    det = LanguageDetector()
    with open(ROOT / "data/eval_cases.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    per_lang = defaultdict(lambda: [0, 0])
    errors = []
    for r in rows:
        pred = det.detect(r["text"], detailed=False)["language"]
        ok = pred == r["expected"]
        per_lang[r["expected"]][0] += ok
        per_lang[r["expected"]][1] += 1
        if not ok:
            errors.append((r["text"], r["expected"], pred))

    correct = sum(v[0] for v in per_lang.values())
    print(f"End-to-end accuracy: {correct}/{len(rows)} = {correct / len(rows):.2%}\n")
    print(f"{'language':12s} correct/total")
    for lang, (c, t) in sorted(per_lang.items()):
        print(f"{lang:12s} {c}/{t}")
    if errors:
        print("\nMisclassified:")
        for text, exp, pred in errors:
            print(f"  {text!r}: expected {exp}, got {pred}")

    evaluate_genz(det)


def evaluate_genz(det):
    """Gen-Z layer: is the language still right, and is the Gen-Z style detected?"""
    with open(ROOT / "data/genz_eval.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lang_ok = genz_ok = 0
    tp = fp = fn = 0
    errors = []
    for r in rows:
        res = det.detect(r["text"], detailed=False)
        exp_genz = r["expected_genz"] == "1"
        lang_ok += res["language"] == r["expected_language"]
        genz_ok += res["is_genz"] == exp_genz
        tp += res["is_genz"] and exp_genz
        fp += res["is_genz"] and not exp_genz
        fn += (not res["is_genz"]) and exp_genz
        if res["language"] != r["expected_language"] or res["is_genz"] != exp_genz:
            errors.append((r["text"], r["expected_language"], exp_genz, res["language"], res["is_genz"]))
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    print(f"\nGen-Z test set ({n} sentences)")
    print(f"  language correct : {lang_ok}/{n} = {lang_ok / n:.2%}")
    print(f"  Gen-Z correct    : {genz_ok}/{n} = {genz_ok / n:.2%}  (precision {precision:.2f}, recall {recall:.2f})")
    for text, el, eg, pl, pg in errors:
        print(f"  {text!r}: expected {el}/genz={eg}, got {pl}/genz={pg}")


if __name__ == "__main__":
    main()
