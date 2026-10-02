"""
Text normalisation and tokenisation.

Social-media style text ("kemmm chooo??? 😂 @atri http://x.y") is cleaned so that
the model sees only linguistic content.
"""

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"[@#](\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "soooo" -> "soo"
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, strip URLs/e-mails/mentions/emojis, squeeze repeats.

    Digits are kept only inside words, because Gen-Z shortforms use them
    ("6e" = che, "km6" = kem cho, "gr8", "2day"); stand-alone numbers are dropped.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MENTION_RE.sub(r" \1 ", text)       # keep hashtag word, drop symbol
    text = text.lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M") or ch in " '’" or ch in "0123456789":
            out.append(ch)
        else:
            out.append(" ")                    # punctuation, emoji -> space
    text = "".join(out).replace("’", "'")
    text = REPEAT_RE.sub(r"\1\1", text)
    # drop stand-alone numbers ("call me at 5" -> "call me at"), keep "6e", "gr8"
    words = [w for w in text.split() if not w.strip("'").isdigit()]
    return " ".join(words)


def tokenize(text: str) -> list[str]:
    """Split normalised text into word tokens (keeps Indic vowel signs attached)."""
    return [t.strip("'") for t in text.split() if t.strip("'")]
