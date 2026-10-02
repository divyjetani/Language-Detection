# System Architecture

## 1. Overview

The detector is a **hybrid system**: fast deterministic rules where they are
reliable (writing systems), and a statistical model where rules fail (text
written in English letters).

```
                         ┌──────────────────────────┐
  input text ───────────►│ 1. Normaliser            │  lower-case, remove URLs, @mentions,
                         │    (normalizer.py)       │  emojis, digits; "sooo" → "soo"
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ 2. Script analyser       │  Unicode block of every letter
                         │    (scripts.py)          │  → script share + script per word
                         └────────────┬─────────────┘
            native words > 25% of all words?
               ┌──────── yes ─────────┴───────── no ────────┐
               ▼                                            ▼
  ┌──────────────────────────┐              ┌───────────────────────────────┐
  │ 3a. Script rules         │              │ 3b. Latin-script ML model     │
  │  Gujarati  → gujarati    │              │  TF-IDF char 1-5 n-grams      │
  │  Tamil     → tamil  …    │              │  + TF-IDF word 1-2 n-grams    │
  │  Devanagari→ marker words│              │  → Logistic Regression        │
  │    (Hindi/Marathi/Nepali)│              │  8 classes: english, hinglish,│
  │  Arabic → Urdu/Persian/  │              │  gujlish, es, fr, de, it, pt  │
  │    Arabic (special chars)│              │                               │
  │  Han+Kana → Japanese     │              │  if max prob < 0.75:          │
  │  Cyrillic → Russian/Ukr. │              │   blend with lexicon vote     │
  └────────────┬─────────────┘              │  if max prob < 0.40 and no    │
               │                            │   lexicon hits → "unknown"    │
               │                            └──────────────┬────────────────┘
               └──────────────────┬────────────────────────┘
                                  ▼
                  ┌──────────────────────────────┐
                  │ 4. Word-level tagger + CMI   │  each word → en / hi / gu / …
                  └──────────────┬───────────────┘  Code-Mixing Index (0-100)
                                 ▼
                  ┌──────────────────────────────┐
                  │ 5. Gen-Z style layer         │  slang + shortform lexicon
                  │    (genz.py)                 │  → "Gujarati (Gen Z)" + meanings
                  └──────────────┬───────────────┘
                                 ▼
                            JSON result
```

## 2. Components

### 2.1 Normaliser — `detector/normalizer.py`
* Unicode NFC normalisation (so composed and decomposed Indic characters match).
* Removes URLs, e-mails; keeps the word of `#hashtags` / `@mentions`.
* Replaces digits, punctuation and emoji with spaces.
* Squeezes 3+ repeated characters to 2 (`"bhaaaai"` → `"bhaai"`).
* Keeps digits **inside** words (`6e`, `km6`, `gr8` are Gen-Z spellings) and drops
  stand-alone numbers.
* Tokeniser keeps Indic vowel signs (matras) attached to their word — Python's
  default `\w` would split "રહો" incorrectly.

### 2.2 Script analyser — `detector/scripts.py`
Maps each character to one of 30+ scripts using Unicode ranges
(e.g. Gujarati `U+0A80–U+0AFF`, Devanagari `U+0900–U+097F`).
Shared scripts are disambiguated with tell-tale characters:

| Script | Rule |
|---|---|
| Arabic | `ٹ ڈ ڑ ں ے ھ` → Urdu; `پ چ ژ گ ک ی` → Persian; else Arabic |
| Cyrillic | `і ї є ґ` → Ukrainian; else Russian |
| Han | with Hiragana/Katakana → Japanese; else Chinese |
| Bengali | `ৰ ৱ` → Assamese; else Bengali |
| Devanagari | marker-word vote (below) |

### 2.3 Devanagari disambiguation
Hindi, Marathi and Nepali share Devanagari. A vote counts high-frequency function
words unique to each (`है, था, और, मैं` vs `आहे, नाही, आणि, मी` vs
`छ, छन्, गर्नु, तपाईं`), with a small prior towards Hindi.

### 2.4 Latin-script ML model — `detector/model.py`
* **Features**
  * character n-grams (1–5) with word boundaries (`char_wb`), TF-IDF, sub-linear TF —
    captures spelling-independent cues: `chh`, `vu `, ` ma ` (Gujarati),
    `hai`, `rha`, `nhi` (Hindi), `ing`, `th` (English), `ção`, `eau` (pt / fr);
  * word unigrams + bigrams, TF-IDF — captures whole marker words / phrases.
* **Classifier**: multinomial Logistic Regression (`C=8`, balanced class weights).
  Chosen over Linear SVM because it outputs calibrated probabilities used for
  confidence, the unknown-language threshold and lexicon blending.

### 2.5 Lexicon vote — `detector/lexicons.py`
Curated lists of high-precision words:
* Gujlish markers: `che/chhe, cho, kem, shu, nathi, tame, mane, javu, karvu, aavje …`
* Hinglish markers: `hai, hain, kya, nahi, mujhe, raha, gaya, karna, kaise …`
* ~400 common English words.

Words common to both Hindi and Gujarati in romanized form (`bhai, kal, ma, na,
to, pan …`) are moved to a shared list and never used to tell them apart.
The vote is a softmax over weighted counts; it is mixed into the model's
probabilities only when the model is unsure (weight 0.6 for ≤3 words, else 0.35).

### 2.6 Word-level tagging and Code-Mixing Index
Every word gets a language code. In a Hinglish/Gujlish sentence, words found in
the English lexicon are tagged `en`, all other words take the sentence's Indic
language. The **Code-Mixing Index** (Das & Gambäck, 2014):

```
CMI = 100 × (1 − max_i(w_i) / n)
```
where `w_i` is the number of words in language *i* and `n` the number of
language-tagged words. `"bhai aaje meeting che office ma"` → 4 gu, 2 en →
CMI = 100 × (1 − 4/6) = 33.3.

### 2.7 Gen-Z style layer — `detector/genz.py`
Gen Z is a **style**, not a language, so it runs after language detection and
never changes the `language` label.

| Dictionary | Counted when | Examples |
|---|---|---|
| `ENGLISH`, `ENGLISH_PHRASES` | always (Gen-Z English slang is mixed into every language) | rizz, delulu, lowkey, ngl, fr, istg, no cap, it's giving |
| `HINDI`, `HINDI_PHRASES` | base language = Hindi | bawaal, jhakaas, pakau, kya scene hai, chill maar, kr / rha / h |
| `GUJARATI`, `GUJARATI_PHRASES` | base language = Gujarati | jalso padi gayo, bhukka bolavi didha, fadu, taddan, km 6o, su6, 6e |
| `SHARED` | base = Hindi or Gujarati | chhapri, kadak, bindaas, jugaad |

Each entry is `term: (meaning, weight)`. Phrases are matched first; words inside a
matched phrase are not counted again. If the summed weight is ≥ `GENZ_THRESHOLD`
(1.0), the result gets `style = "genz"` and
`display_name = "<Base language> (Gen Z)"`. English slang words are also tagged
`en` by the word tagger inside Hinglish/Gujlish sentences.

Script rule: if more than a quarter of the words are in a
native script, that script decides the language (so *"मैं literally dead हूँ, no
cap"* is Hindi (Gen Z), not English).

## 3. Data pipeline — `data/generate_dataset.py`
1. ~1,000 hand-written seed sentences in 8 classes (`data/seed/`).
2. **Split first** (80 / 20 per class) → test sentences never appear in training.
3. Augment each split separately, simulating real chat text:
   * spelling variants (`che→chhe/6e`, `nahi→nhi/nai`, `you→u`),
   * ~30 Gen-Z style seed sentences per English / Hinglish / Gujlish class,
   * vowel dropping (`kar raha→kr rha`), letter stretching (`bhaaai`),
   * sentence concatenation, casing changes, emoji / punctuation noise.
4. Output: 11,000 train rows, 2,200 test rows.

## 4. API layer — `api/main.py`
FastAPI app; the model is loaded once at start-up (lifespan hook) and shared.
Pydantic models validate input (length limits, batch size) and document the
response schema, which FastAPI turns into Swagger docs at `/docs`.

## 5. Robustness
* If `models/latin_model.joblib` is missing or was saved by an incompatible
  scikit-learn version, `load_model()` retrains automatically (deterministic, seed 42).
* Empty / emoji-only / numeric input returns `unknown` with method `empty`.
* Gibberish in Latin letters with no lexicon matches and low probability returns `unknown`.
