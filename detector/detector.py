"""
LanguageDetector - the hybrid detection engine.

Pipeline
--------
1. Normalise + tokenise the text.
2. Script analysis (Unicode blocks) for every token.
3a. If the text is mainly in a native script (Gujarati, Devanagari, Tamil ...)
    -> rule-based script -> language mapping (+ marker words for Devanagari:
       Hindi vs Marathi vs Nepali).
3b. If the text is mainly in Latin script
    -> ML model (char + word n-gram TF-IDF, Logistic Regression) decides between
       English / Hinglish / Gujlish / Spanish / French / German / Italian / Portuguese
    -> blended with a lexicon vote when the model is unsure (short inputs).
4. Word-level tagging + Code-Mixing Index (CMI) to report how mixed the text is.
"""

from __future__ import annotations

import math
from functools import lru_cache

from . import genz
from . import lexicons as L
from .model import load_model
from .normalizer import normalize, tokenize
from .scripts import resolve_script_language, script_profile, token_script

LANGUAGE_INFO = {
    "english":    {"name": "English", "base": "english", "romanized": False},
    "hindi":      {"name": "Hindi", "base": "hindi", "romanized": False},
    "gujarati":   {"name": "Gujarati", "base": "gujarati", "romanized": False},
    "hinglish":   {"name": "Hinglish", "base": "hindi", "romanized": True},
    "gujlish":    {"name": "Gujlish", "base": "gujarati", "romanized": True},
    "marathi":    {"name": "Marathi", "base": "marathi", "romanized": False},
    "nepali":     {"name": "Nepali", "base": "nepali", "romanized": False},
    "spanish":    {"name": "Spanish", "base": "spanish", "romanized": False},
    "french":     {"name": "French", "base": "french", "romanized": False},
    "german":     {"name": "German", "base": "german", "romanized": False},
    "italian":    {"name": "Italian", "base": "italian", "romanized": False},
    "portuguese": {"name": "Portuguese", "base": "portuguese", "romanized": False},
    "bengali": {"name": "Bengali", "base": "bengali", "romanized": False},
    "assamese": {"name": "Assamese", "base": "assamese", "romanized": False},
    "punjabi": {"name": "Punjabi", "base": "punjabi", "romanized": False},
    "odia": {"name": "Odia", "base": "odia", "romanized": False},
    "tamil": {"name": "Tamil", "base": "tamil", "romanized": False},
    "telugu": {"name": "Telugu", "base": "telugu", "romanized": False},
    "kannada": {"name": "Kannada", "base": "kannada", "romanized": False},
    "malayalam": {"name": "Malayalam", "base": "malayalam", "romanized": False},
    "sinhala": {"name": "Sinhala", "base": "sinhala", "romanized": False},
    "urdu": {"name": "Urdu", "base": "urdu", "romanized": False},
    "arabic": {"name": "Arabic", "base": "arabic", "romanized": False},
    "persian": {"name": "Persian", "base": "persian", "romanized": False},
    "russian": {"name": "Russian", "base": "russian", "romanized": False},
    "ukrainian": {"name": "Ukrainian", "base": "ukrainian", "romanized": False},
    "greek": {"name": "Greek", "base": "greek", "romanized": False},
    "hebrew": {"name": "Hebrew", "base": "hebrew", "romanized": False},
    "chinese": {"name": "Chinese", "base": "chinese", "romanized": False},
    "japanese": {"name": "Japanese", "base": "japanese", "romanized": False},
    "korean": {"name": "Korean", "base": "korean", "romanized": False},
    "thai": {"name": "Thai", "base": "thai", "romanized": False},
    "lao": {"name": "Lao", "base": "lao", "romanized": False},
    "tibetan": {"name": "Tibetan", "base": "tibetan", "romanized": False},
    "burmese": {"name": "Burmese", "base": "burmese", "romanized": False},
    "khmer": {"name": "Khmer", "base": "khmer", "romanized": False},
    "georgian": {"name": "Georgian", "base": "georgian", "romanized": False},
    "armenian": {"name": "Armenian", "base": "armenian", "romanized": False},
    "amharic": {"name": "Amharic", "base": "amharic", "romanized": False},
    "unknown": {"name": "Unknown / other language", "base": "unknown", "romanized": False},
}

# word-level tag for each sentence-level label
TOKEN_TAG = {"english": "en", "hinglish": "hi", "gujlish": "gu", "hindi": "hi", "gujarati": "gu"}

MIN_CONFIDENCE = 0.40          # below this a Latin text is reported as "unknown"
LEXICON_BLEND_THRESHOLD = 0.75  # below this the lexicon vote is blended in


class LanguageDetector:
    def __init__(self, model=None):
        self.model = model if model is not None else load_model()
        self.classes = list(self.model.classes_)

    # ------------------------------------------------------------------ public
    def detect(self, text: str, detailed: bool = True) -> dict:
        clean = normalize(text)
        tokens = tokenize(clean)
        if not tokens:
            return self._result(text, "unknown", 0.0, {}, {}, [], "empty", note="No alphabetic text found.")

        profile = script_profile(clean)
        tok_scripts = [token_script(t) for t in tokens]
        latin_tokens = [t for t, s in zip(tokens, tok_scripts) if s == "Latin"]
        native_tokens = [t for t, s in zip(tokens, tok_scripts) if s and s != "Latin"]

        # Native script wins if it is more than a quarter of the words: in
        # "मैं literally dead हूँ bro" the English words are insertions into Hindi.
        if native_tokens and len(native_tokens) > 0.25 * len(tokens):
            res = self._detect_native(text, clean, tokens, tok_scripts, native_tokens, profile)
        else:
            res = self._detect_latin(text, latin_tokens, tokens, tok_scripts, profile)

        self._add_genz(res, tokens, clean)

        if not detailed:
            for k in ("tokens", "probabilities"):
                res.pop(k, None)
        return res

    @staticmethod
    def _add_genz(res: dict, tokens: list[str], clean: str):
        """Second layer: is the text written in Gen-Z style? (slang / shortforms)"""
        if res["language"] == "unknown":
            return
        g = genz.analyze(tokens, clean, res["base_language"])
        res.update(g)
        if g["is_genz"]:
            base_name = LANGUAGE_INFO.get(res["base_language"], {}).get("name", res["base_language"].title())
            res["style"] = "genz"
            res["display_name"] = f"{base_name} (Gen Z 😎)"          # e.g. "Gujarati (Gen Z)"

    def detect_batch(self, texts: list[str], detailed: bool = False) -> list[dict]:
        return [self.detect(t, detailed=detailed) for t in texts]

    # ------------------------------------------------------------ native path
    def _detect_native(self, raw, clean, tokens, tok_scripts, native_tokens, profile):
        native_scripts = {s: v for s, v in profile.items() if s != "Latin"}
        script = max(native_scripts, key=native_scripts.get)
        lang = resolve_script_language(script, clean)
        purity = native_scripts[script] / sum(native_scripts.values())
        confidence = 0.80 + 0.19 * purity
        probs = {lang: round(confidence, 4)}
        method = "script"

        if script == "Devanagari":
            lang, probs, confidence = self._devanagari(native_tokens)
            method = "script+lexicon"

        tags = []
        own_tag = TOKEN_TAG.get(lang, lang[:2])
        for t, s in zip(tokens, tok_scripts):
            tags.append({"token": t, "script": s, "lang": "en" if s == "Latin" else own_tag})
        return self._result(raw, lang, confidence, probs, profile, tags, method)

    def _devanagari(self, tokens):
        tset = set(tokens)
        scores = {
            "hindi": len(tset & L.HINDI_DEVANAGARI) + 0.5,   # prior: Hindi is most common
            "marathi": len(tset & L.MARATHI_DEVANAGARI),
            "nepali": len(tset & L.NEPALI_DEVANAGARI),
        }
        total = sum(scores.values())
        probs = {k: round(v / total, 4) for k, v in scores.items()}
        lang = max(probs, key=probs.get)
        conf = 0.6 + 0.39 * probs[lang]
        return lang, probs, conf

    # ------------------------------------------------------------- latin path
    def _detect_latin(self, raw, latin_tokens, tokens, tok_scripts, profile):
        latin_text = " ".join(latin_tokens)
        ml = dict(zip(self.classes, self.model.predict_proba([latin_text])[0]))
        lex = self._lexicon_vote(latin_tokens)
        top_ml = max(ml.values())

        probs = dict(ml)
        method = "ml"
        if lex and top_ml < LEXICON_BLEND_THRESHOLD:
            # weight lexicon more for very short inputs, where n-grams are weak
            w = 0.6 if len(latin_tokens) <= 3 else 0.35
            probs = {c: (1 - w) * ml.get(c, 0) + w * lex.get(c, 0) for c in self.classes}
            method = "ml+lexicon"

        lang = max(probs, key=probs.get)
        confidence = probs[lang]
        note = None
        if confidence < MIN_CONFIDENCE and not lex:
            note = f"Low confidence; closest match was '{lang}'."
            lang = "unknown"

        tags = self._tag_tokens(tokens, tok_scripts, lang)
        probs = {k: round(float(v), 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}
        return self._result(raw, lang, float(confidence), probs, profile, tags, method, note=note)

    def _lexicon_vote(self, tokens):
        """Soft vote from marker words: returns {} when no evidence."""
        en = sum(t in L.ENGLISH for t in tokens)
        hi = sum(t in L.HINDI_ROMAN for t in tokens)
        gu = sum(t in L.GUJARATI_ROMAN for t in tokens)
        if hi + gu == 0 and en == 0:
            return {}
        # Indic marker words are much more decisive than English function words
        raw = {"english": 0.6 * en, "hinglish": 1.5 * hi, "gujlish": 1.5 * gu}
        if hi + gu == 0:
            raw["english"] += 1.0
        exp = {k: math.exp(v) for k, v in raw.items()}
        s = sum(exp.values())
        return {k: v / s for k, v in exp.items()}

    # ----------------------------------------------------- word-level tagging
    def _tag_tokens(self, tokens, tok_scripts, sentence_lang):
        indic = TOKEN_TAG.get(sentence_lang)
        tags = []
        for t, s in zip(tokens, tok_scripts):
            if s != "Latin":
                lang = resolve_script_language(s, t) if s else "unknown"
                tag = TOKEN_TAG.get(lang, lang[:2])
            elif sentence_lang in ("hinglish", "gujlish"):
                # romanized Hindi and Gujarati share many short words ("hu", "kal",
                # "ma"), so inside an Indic sentence every non-English word takes
                # the sentence's language; only English words are split out.
                if t in L.GUJARATI_ROMAN or t in L.HINDI_ROMAN or t in L.SHARED_INDIC_ROMAN:
                    tag = indic
                elif t in L.ENGLISH or t in genz.ENGLISH_WORDS:
                    tag = "en"         # includes Gen-Z slang like "ngl", "rizz"
                else:
                    tag = indic        # unseen romanized word -> sentence language
            elif sentence_lang == "english":
                if t in L.ENGLISH:
                    tag = "en"
                elif t in L.GUJARATI_ROMAN:
                    tag = "gu"
                elif t in L.HINDI_ROMAN:
                    tag = "hi"
                else:
                    tag = "en"
            else:
                tag = sentence_lang[:2] if sentence_lang != "unknown" else "un"
            tags.append({"token": t, "script": s, "lang": tag})
        return tags

    @staticmethod
    def code_mixing_index(tags) -> float:
        """CMI (Das & Gambäck, 2014): 100 * (1 - max_lang_tokens / language_tokens)."""
        langs = [t["lang"] for t in tags if t["lang"] not in ("un", None)]
        if not langs:
            return 0.0
        counts = {}
        for l in langs:
            counts[l] = counts.get(l, 0) + 1
        return round(100 * (1 - max(counts.values()) / len(langs)), 2)

    # ------------------------------------------------------------------ utils
    def _result(self, raw, lang, confidence, probs, profile, tags, method, note=None):
        info = LANGUAGE_INFO.get(lang, {"name": lang.title(), "base": lang, "romanized": False})
        cmi = self.code_mixing_index(tags) if tags else 0.0
        mixed_langs = sorted({t["lang"] for t in tags if t["lang"] not in ("un",)})
        dominant_script = max(profile, key=profile.get) if profile else None
        out = {
            "text": raw,
            "language": lang,
            "language_name": info["name"],
            "base_language": info["base"],
            "is_romanized": info["romanized"],
            "script": dominant_script,
            "confidence": round(float(confidence), 4),
            "is_code_mixed": cmi > 0 and len(mixed_langs) > 1,
            "code_mixing_index": cmi,
            "languages_in_text": mixed_langs,
            "scripts": profile,
            "probabilities": probs,
            "tokens": tags,
            "method": method,
            # Gen-Z layer (filled in by _add_genz)
            "style": "standard",
            "display_name": info["name"],
            "is_genz": False,
            "genz_score": 0.0,
            "genz_terms": [],
        }
        if note:
            out["note"] = note
        return out


@lru_cache(maxsize=1)
def get_detector() -> LanguageDetector:
    """Shared singleton (model is loaded once)."""
    return LanguageDetector()


def detect(text: str, detailed: bool = True) -> dict:
    """Convenience function:  from detector import detect;  detect("kem cho")"""
    return get_detector().detect(text, detailed=detailed)
