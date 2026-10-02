# Gen-Z update — what changed and why

Goal: when a user writes in Gen-Z slang or shortforms, show the language as
**Gujarati (Gen Z)**, **Hindi (Gen Z)** or **English (Gen Z)**, and explain the slang.

Design choice: Gen Z is a *style* of a language, not a new language. So the
`language` field stays `gujlish` / `hinglish` / `english` / `hindi` / `gujarati`,
and a Gen-Z layer is added on top. This keeps the language model accurate and
lets you add slang without retraining.

## New files
| File | What it does |
|---|---|
| `detector/genz.py` | The Gen-Z engine. Slang + shortform dictionaries (English ≈95, Hindi ≈40, Gujarati ≈30, shared 4), each as `term: (meaning, weight)`. `analyze()` scores the text and returns `is_genz`, `genz_score`, `genz_terms`. |
| `data/genz_eval.csv` | 33 test sentences (22 Gen-Z, 11 standard) to measure the Gen-Z layer. |
| `docs/screenshot_genz.png` | Screenshot of the new UI. |
| `CHANGES_GENZ.md` | This file. |

## Changed files
| File | Change |
|---|---|
| `detector/normalizer.py` | Keeps digits inside words (`6e`, `6o`, `km6`, `gr8`) — Gujarati Gen-Z writes "6" for *chh*. Stand-alone numbers are still removed. |
| `detector/detector.py` | 1) imports `genz`; 2) new `_add_genz()` runs after language detection and sets `style`, `display_name`, `is_genz`, `genz_score`, `genz_terms`; 3) Gen-Z English slang (`ngl`, `rizz`) is tagged `en` in Hinglish/Gujlish sentences; 4) native script now wins when it is >25% of words, so "मैं literally dead हूँ, no cap" → Hindi (Gen Z). |
| `api/main.py` | Response schema has the new fields (`GenZTerm` model). API description mentions Gen Z. |
| `frontend/index.html` | Headline shows `display_name` ("Gujarati (Gen Z)"), 🔥 Gen Z badge, new "Slang decoded" box with meanings, 3 Gen-Z example chips, batch table uses `display_name`. |
| `cli.py` | Prints the Gen-Z label and the slang meanings. |
| `data/seed/english.txt`, `hinglish.txt`, `gujlish.txt` | +30 Gen-Z style sentences each, so the language model handles slang-heavy text. |
| `models/*` | Retrained (accuracy 95.0 %, macro F1 0.958 on held-out sentences). |
| `training/evaluate.py` | Also evaluates the Gen-Z layer on `data/genz_eval.csv`. |
| `tests/test_detector.py`, `tests/test_api.py` | +13 Gen-Z tests (49 total, all passing). |
| `README.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/PROJECT_REPORT.md` | Documented the Gen-Z layer, new fields and updated results. |

## New API fields
```json
"style": "genz",
"display_name": "Gujarati (Gen Z)",
"is_genz": true,
"genz_score": 2.2,
"genz_terms": [{"term": "jalso padi gayo", "meaning": "had a blast"}]
```

## How the scoring works
* Every slang entry has a weight: 1.0 = clearly Gen Z on its own (`rizz`, `jalso`,
  `bawaal`, `no cap`); 0.3–0.6 = common chat shortform (`idk`, `h`, `kr`, `km`).
* English slang counts for every language; Hindi slang counts only for Hindi text,
  Gujarati slang only for Gujarati text.
* Total ≥ 1.0 → Gen Z. So one clear slang word is enough, but shortforms need to
  appear together ("kya kr rha h" = Gen Z, "nhi" alone ≠ Gen Z).

## Adding new slang (no retraining)
Open `detector/genz.py` and add a line to the right dictionary:
```python
GUJARATI = {
    ...
    "aavi ja": ("come on / bring it", 0.8),
}
```
Then restart the server.
