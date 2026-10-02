"""
Build the training / test dataset from the hand-written seed sentences.

Why augmentation?  Romanized Hindi/Gujarati has no standard spelling:
"chhe / che / 6e", "nahi / nhi / nai", "kaise / kese". Real users also drop
vowels, repeat letters, mix cases and glue sentences together. We simulate
these variations so that the model learns robust character patterns
instead of memorising one spelling.

IMPORTANT: seeds are split into train/test BEFORE augmentation, so no
augmented copy of a test sentence ever appears in training (no leakage).

Usage:
    python data/generate_dataset.py
Outputs:
    data/processed/train.csv, data/processed/test.csv
"""

import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "seed"
OUT_DIR = ROOT / "processed"

LABELS = ["english", "hinglish", "gujlish", "spanish", "french", "german", "italian", "portuguese"]
TARGET_PER_CLASS = {"english": 2500, "hinglish": 2500, "gujlish": 2500}
DEFAULT_TARGET = 700
TEST_FRACTION = 0.2
RNG = random.Random(42)

# Common spelling variants seen in real chats
VARIANTS = {
    # Gujlish
    "che": ["chhe", "6e", "che", "chhe"], "chhe": ["che", "6e"], "chu": ["chhu", "6u"],
    "cho": ["chho", "6o"], "kem": ["km", "kemm"], "shu": ["su", "shun"], "su": ["shu"],
    "nathi": ["nthi", "nathi", "nai"], "majama": ["maja ma", "majja ma", "mazama"],
    "maja": ["majja", "maza"], "bahu": ["bau", "bhu"], "saru": ["saaru"], "tame": ["tme"],
    "mane": ["mne"], "kyare": ["kyare", "kyre"], "aavje": ["avje"], "aavu": ["avu"],
    "jaiye": ["jaie", "jaiee"], "pachi": ["pachhi", "pchi"], "karvu": ["krvu"],
    "aapde": ["apde"], "ketla": ["ktla"], "hatu": ["htu"],
    # Hinglish
    "hai": ["h", "he", "hai", "hai"], "hain": ["hai", "h"], "nahi": ["nhi", "nai", "nahin"],
    "kya": ["kyaa", "kia"], "kyun": ["kyu", "kyon"], "kaise": ["kese", "kaisay"],
    "raha": ["rha"], "rahe": ["rhe"], "rahi": ["rhi"], "kar": ["kr"], "bahut": ["bohot", "bhot", "bahot"],
    "accha": ["acha", "achha"], "mujhe": ["mjhe", "muje"], "main": ["mai", "me"],
    "hoon": ["hu", "hun"], "tum": ["tm"], "yaar": ["yr", "yaarr"], "theek": ["thik", "thk"],
    "pata": ["pta"], "abhi": ["abi"], "phir": ["fir"],
    # English (chat style)
    "you": ["u", "you"], "are": ["r", "are"], "please": ["pls", "plz"], "okay": ["ok", "k"],
    "thanks": ["thx", "thanks"], "because": ["coz", "bcoz"], "tomorrow": ["tmrw"],
}

EMOJI = ["", "", "", " 😂", " 🙏", " ❤️", " !!", "??", " ...", " :)"]


def load_seeds():
    data = {}
    for label in LABELS:
        path = SEED_DIR / f"{label}.txt"
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        data[label] = sorted(set(lines))
    return data


def spell_variant(sentence: str, p: float = 0.5) -> str:
    words = sentence.split()
    out = []
    for w in words:
        core = re.sub(r"[^\w']", "", w.lower())
        if core in VARIANTS and RNG.random() < p:
            out.append(w.lower().replace(core, RNG.choice(VARIANTS[core])))
        else:
            out.append(w)
    return " ".join(out)


def drop_vowels(sentence: str, p: float = 0.15) -> str:
    """'kar raha hai' -> 'kr rha hai' (typical SMS style)."""
    words = []
    for w in sentence.split():
        if len(w) > 3 and RNG.random() < p:
            w = w[0] + re.sub(r"[aeiou]", "", w[1:-1]) + w[-1]
        words.append(w)
    return " ".join(words)


def stretch(sentence: str, p: float = 0.1) -> str:
    """'bhai' -> 'bhaaai' (emphasis)."""
    words = []
    for w in sentence.split():
        if RNG.random() < p and re.search(r"[aeiou]", w):
            i = max(i for i, c in enumerate(w) if c in "aeiou")
            w = w[:i] + w[i] * RNG.randint(2, 4) + w[i + 1:]
        words.append(w)
    return " ".join(words)


def recase(sentence: str) -> str:
    r = RNG.random()
    if r < 0.15:
        return sentence.capitalize()
    if r < 0.2:
        return sentence.upper()
    if r < 0.25:
        return sentence.title()
    return sentence


def augment(label: str, seeds: list[str], n: int) -> list[str]:
    romanized = label in ("hinglish", "gujlish", "english")
    out = list(seeds)  # always keep originals
    while len(out) < n:
        r = RNG.random()
        if r < 0.35:
            s = RNG.choice(seeds) + " " + RNG.choice(seeds)          # longer text
        else:
            s = RNG.choice(seeds)
        if romanized:
            s = spell_variant(s)
            s = drop_vowels(s) if label != "english" else s
            s = stretch(s)
        s = recase(s) + RNG.choice(EMOJI)
        out.append(s)
    return out


def main():
    seeds = load_seeds()
    train_rows, test_rows = [], []
    for label, sents in seeds.items():
        sents = sents[:]
        RNG.shuffle(sents)
        k = max(3, int(len(sents) * TEST_FRACTION))
        test_seed, train_seed = sents[:k], sents[k:]
        target = TARGET_PER_CLASS.get(label, DEFAULT_TARGET)
        train = augment(label, train_seed, target)
        test = augment(label, test_seed, max(len(test_seed), int(target * TEST_FRACTION)))
        train_rows += [(t, label) for t in train]
        test_rows += [(t, label) for t in test]
        print(f"{label:12s} seeds={len(sents):4d}  train={len(train):5d}  test={len(test):4d}")

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in (("train.csv", train_rows), ("test.csv", test_rows)):
        RNG.shuffle(rows)
        with open(OUT_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
    print(f"\nWrote {len(train_rows)} train and {len(test_rows)} test rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
