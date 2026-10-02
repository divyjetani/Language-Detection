import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import LanguageDetector  # noqa: E402
from detector.normalizer import normalize, tokenize  # noqa: E402
from detector.scripts import script_profile  # noqa: E402


@pytest.fixture(scope="module")
def det():
    return LanguageDetector()


@pytest.mark.parametrize("text,expected", [
    ("kem cho maja ma ne?", "gujlish"),
    ("kem cho", "gujlish"),
    ("mane nathi khabar", "gujlish"),
    ("tame kya jao cho", "gujlish"),
    ("bhai aaje meeting che office ma", "gujlish"),
    ("kaise ho bhai", "hinglish"),
    ("mujhe nahi pata", "hinglish"),
    ("kya kar rahe ho", "hinglish"),
    ("kal meeting hai office mein", "hinglish"),
    ("how are you doing today", "english"),
    ("please send me the report", "english"),
    ("તમે કેમ છો?", "gujarati"),
    ("आप कैसे हैं?", "hindi"),
    ("मी घरी जात आहे", "marathi"),
    ("வணக்கம்", "tamil"),
    ("Привет как дела", "russian"),
    ("お元気ですか", "japanese"),
    ("你好世界", "chinese"),
    ("آپ کیسے ہیں", "urdu"),
    ("Bonjour, comment ça va?", "french"),
    ("Ich bin müde", "german"),
])
def test_language(det, text, expected):
    assert det.detect(text)["language"] == expected


def test_romanized_flags(det):
    r = det.detect("kem cho maja ma ne?")
    assert r["is_romanized"] and r["base_language"] == "gujarati" and r["script"] == "Latin"
    r = det.detect("kaise ho bhai")
    assert r["is_romanized"] and r["base_language"] == "hindi"
    r = det.detect("how are you")
    assert not r["is_romanized"]


def test_code_mixing(det):
    r = det.detect("hu tane call karu chu")
    assert r["is_code_mixed"]
    assert set(r["languages_in_text"]) == {"gu", "en"}
    assert r["code_mixing_index"] > 0
    assert not det.detect("mane nathi khabar")["is_code_mixed"]


def test_native_script_code_mixed(det):
    r = det.detect("मैं office जा रहा हूँ")
    assert r["language"] == "hindi" and r["is_code_mixed"]


def test_empty_and_noise(det):
    assert det.detect("12345 !!! 😂")["language"] == "unknown"
    assert det.detect("")["language"] == "unknown"


def test_robust_to_noise(det):
    assert det.detect("KEMMM CHOOO??? 😂😂 @atri http://x.com")["language"] == "gujlish"
    assert det.detect("kya kr rha h bhaiii")["language"] == "hinglish"


def test_batch(det):
    out = det.detect_batch(["kem cho", "kaise ho", "hello there"])
    assert [r["language"] for r in out] == ["gujlish", "hinglish", "english"]


def test_normalizer():
    assert normalize("Heyyyy!!! http://a.com 123") == "heyy"
    assert tokenize(normalize("તમે કેમ છો?")) == ["તમે", "કેમ", "છો"]


def test_script_profile():
    p = script_profile("abc અબક")
    assert p == {"Latin": 0.5, "Gujarati": 0.5}


# ----------------------------------------------------------------- Gen-Z layer
@pytest.mark.parametrize("text,language,display", [
    ("km 6o bhai, jalso padi gayo", "gujlish", "Gujarati (Gen Z)"),
    ("aaje party ma bhukka bolavi didha", "gujlish", "Gujarati (Gen Z)"),
    ("bhai ye lowkey op hai fr", "hinglish", "Hindi (Gen Z)"),
    ("kya scene hai bhai, chill maar", "hinglish", "Hindi (Gen Z)"),
    ("ngl this biryani is bussin fr", "english", "English (Gen Z)"),
    ("मैं literally dead हूँ bro, no cap", "hindi", "Hindi (Gen Z)"),
])
def test_genz_detected(det, text, language, display):
    r = det.detect(text)
    assert r["language"] == language
    assert r["is_genz"] and r["style"] == "genz"
    assert r["display_name"] == display
    assert r["genz_terms"] and all("meaning" in t for t in r["genz_terms"])


@pytest.mark.parametrize("text", [
    "kem cho maja ma ne?", "kaise ho bhai", "how are you doing today", "તમે કેમ છો?",
])
def test_standard_text_not_genz(det, text):
    r = det.detect(text)
    assert not r["is_genz"] and r["style"] == "standard"
    assert r["display_name"] == r["language_name"]


def test_digit_shortforms_kept(det):
    # "6e" / "6o" are Gujarati Gen-Z spellings of che / cho - digits must survive cleaning
    assert normalize("su kr 6e 123") == "su kr 6e"
    r = det.detect("su kr 6e bhai")
    assert r["language"] == "gujlish" and r["is_genz"]


def test_genz_slang_tagged_english_inside_gujlish(det):
    r = det.detect("tame kya jao cho ngl")
    tags = {t["token"]: t["lang"] for t in r["tokens"]}
    assert tags["ngl"] == "en" and tags["tame"] == "gu"
