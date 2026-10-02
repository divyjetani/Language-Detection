"""
Unicode script analysis.

Every character is mapped to the writing system (script) it belongs to, using
Unicode block ranges. Script is the single strongest signal in language
identification: Gujarati script => Gujarati, Tamil script => Tamil, etc.
Only the Latin script (English, Hinglish, Gujlish, Spanish ...) and a few shared
scripts (Devanagari, Arabic, Cyrillic, Han) need further disambiguation.
"""

import unicodedata
from collections import Counter

# (start, end, script_name) - inclusive Unicode ranges
SCRIPT_RANGES = [
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"), (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"), (0xA8E0, 0xA8FF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"), (0x3130, 0x318F, "Hangul"), (0xAC00, 0xD7AF, "Hangul"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x4E00, 0x9FFF, "Han"), (0x3400, 0x4DBF, "Han"), (0xF900, 0xFAFF, "Han"),
]

# Scripts used by exactly one language (in our scope) -> direct mapping
SCRIPT_TO_LANGUAGE = {
    "Gujarati": "gujarati",
    "Bengali": "bengali",
    "Gurmukhi": "punjabi",
    "Odia": "odia",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Sinhala": "sinhala",
    "Thai": "thai",
    "Lao": "lao",
    "Tibetan": "tibetan",
    "Myanmar": "burmese",
    "Georgian": "georgian",
    "Armenian": "armenian",
    "Hebrew": "hebrew",
    "Greek": "greek",
    "Hangul": "korean",
    "Ethiopic": "amharic",
    "Khmer": "khmer",
}


def char_script(ch: str) -> str | None:
    """Return the script name of a single character, or None for non-letters."""
    cp = ord(ch)
    for start, end, name in SCRIPT_RANGES:
        if start <= cp <= end:
            # skip punctuation / digits that live inside script blocks
            cat = unicodedata.category(ch)
            if cat[0] in ("L", "M"):
                return name
            return None
    if unicodedata.category(ch)[0] == "L":
        return "Other"
    return None


def script_profile(text: str) -> dict[str, float]:
    """Fraction of letters belonging to each script, e.g. {'Latin': 0.8, 'Devanagari': 0.2}."""
    counts = Counter(s for s in (char_script(c) for c in text) if s)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in counts.most_common()}


def token_script(token: str) -> str | None:
    """Dominant script of a single token."""
    prof = script_profile(token)
    return max(prof, key=prof.get) if prof else None


# ---------------------------------------------------------------------------
# Disambiguation for scripts shared by several languages
# ---------------------------------------------------------------------------
URDU_ONLY = set("ٹڈڑںےۓھ")
PERSIAN_HINT = set("پچژگکی")
UKRAINIAN_ONLY = set("іїєґІЇЄҐ")
BULGARIAN_HINT = set("ъщ")  # weak hint, Russian also uses these
ASSAMESE_ONLY = set("ৰৱ")


def resolve_script_language(script: str, text: str) -> str:
    """Map a (possibly shared) script to the most likely language."""
    if script in SCRIPT_TO_LANGUAGE:
        if script == "Bengali" and any(c in ASSAMESE_ONLY for c in text):
            return "assamese"
        return SCRIPT_TO_LANGUAGE[script]
    if script in ("Hiragana", "Katakana"):
        return "japanese"
    if script == "Han":
        # Han + kana => Japanese; Han alone => Chinese
        if any(char_script(c) in ("Hiragana", "Katakana") for c in text):
            return "japanese"
        return "chinese"
    if script == "Arabic":
        if any(c in URDU_ONLY for c in text):
            return "urdu"
        if any(c in PERSIAN_HINT for c in text):
            return "persian"
        return "arabic"
    if script == "Cyrillic":
        if any(c in UKRAINIAN_ONLY for c in text):
            return "ukrainian"
        return "russian"
    return "unknown"
