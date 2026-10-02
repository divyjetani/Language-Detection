# API Reference

Base URL (local): `http://localhost:8000`

Interactive docs are generated automatically:
* Swagger UI — `/docs` (try every endpoint from the browser)
* ReDoc — `/redoc`

All requests and responses use JSON (UTF-8). CORS is open, so the API can be
called from any front-end.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Server status |
| POST | `/api/detect` | Detect the language of one text |
| GET | `/api/detect?text=...` | Same, via query string |
| POST | `/api/detect/batch` | Detect up to 100 texts |
| GET | `/api/languages` | List every language the system can return |
| GET | `/api/model/info` | Training metrics of the ML model |
| GET | `/` | Web interface |

---

## POST `/api/detect`

### Request body
| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string | yes | — | 1–5000 characters |
| `detailed` | boolean | no | `true` | include `tokens` and `probabilities` |

```json
{ "text": "bhai aaje meeting che office ma", "detailed": true }
```

### Response
```json
{
  "text": "bhai aaje meeting che office ma",
  "language": "gujlish",
  "language_name": "Gujlish (Gujarati written in English letters)",
  "base_language": "gujarati",
  "is_romanized": true,
  "script": "Latin",
  "confidence": 0.9921,
  "is_code_mixed": true,
  "code_mixing_index": 33.33,
  "languages_in_text": ["en", "gu"],
  "scripts": { "Latin": 1.0 },
  "probabilities": { "gujlish": 0.9921, "english": 0.0031, "hinglish": 0.0029, "...": 0 },
  "tokens": [
    { "token": "bhai",    "script": "Latin", "lang": "gu" },
    { "token": "aaje",    "script": "Latin", "lang": "gu" },
    { "token": "meeting", "script": "Latin", "lang": "en" },
    { "token": "che",     "script": "Latin", "lang": "gu" },
    { "token": "office",  "script": "Latin", "lang": "en" },
    { "token": "ma",      "script": "Latin", "lang": "gu" }
  ],
  "method": "ml",
  "style": "standard",
  "display_name": "Gujlish (Gujarati written in English letters)",
  "is_genz": false,
  "genz_score": 0.0,
  "genz_terms": [],
  "processing_time_ms": 9.45
}
```

Gen-Z example — `{"text": "km 6o bhai, jalso padi gayo"}` returns (abridged):
```json
{
  "language": "gujlish",
  "language_name": "Gujlish (Gujarati written in English letters)",
  "base_language": "gujarati",
  "style": "genz",
  "display_name": "Gujarati (Gen Z)",
  "is_genz": true,
  "genz_score": 2.2,
  "genz_terms": [
    { "term": "jalso padi gayo", "meaning": "had a blast" },
    { "term": "6o", "meaning": "are (cho)" },
    { "term": "km", "meaning": "how (kem)" }
  ]
}
```

### Response fields
| Field | Meaning |
|---|---|
| `language` | Detected label: `english`, `hindi`, `gujarati`, `hinglish`, `gujlish`, `marathi`, `tamil`, `french`, … or `unknown` |
| `language_name` | Human-readable name |
| `base_language` | The real language behind the text (`gujlish` → `gujarati`, `hinglish` → `hindi`) |
| `is_romanized` | `true` when an Indian language is written in English letters |
| `script` | Dominant writing system (`Latin`, `Gujarati`, `Devanagari`, …) |
| `confidence` | 0–1 |
| `is_code_mixed` | `true` when words from more than one language appear |
| `code_mixing_index` | 0–100 (CMI, Das & Gambäck 2014). 0 = one language; higher = more mixed |
| `languages_in_text` | Word-level language codes present (`en`, `hi`, `gu`, …) |
| `scripts` | Share of letters in each script |
| `probabilities` | Class probabilities (Latin-script texts) or rule scores (native scripts) |
| `tokens` | Per-word language tags |
| `style` | `genz` or `standard` |
| `display_name` | Label to show users: `Gujarati (Gen Z)`, `Hindi (Gen Z)`, `English (Gen Z)` for Gen-Z text, otherwise the same as `language_name` |
| `is_genz` | `true` when Gen-Z slang/shortforms reach the threshold (score ≥ 1.0) |
| `genz_score` | Sum of the weights of matched slang terms |
| `genz_terms` | Matched slang with meanings: `[{"term": "ngl", "meaning": "not gonna lie"}]` |
| `method` | `script`, `script+lexicon`, `ml`, `ml+lexicon` or `empty` — which part of the system decided |
| `note` | Optional, e.g. low confidence explanation |
| `processing_time_ms` | Server time |

---

## GET `/api/detect`
```
GET /api/detect?text=kem%20cho&detailed=false
```
Same response as POST.

---

## POST `/api/detect/batch`

```json
{ "texts": ["kem cho", "kaise ho", "how are you", "હું ઘરે છું"], "detailed": false }
```
Response:
```json
{
  "count": 4,
  "summary": { "gujlish": 1, "hinglish": 1, "english": 1, "gujarati": 1 },
  "results": [ { "language": "gujlish", "...": "..." }, "..." ],
  "processing_time_ms": 21.7
}
```
Limits: 1–100 texts, each ≤ 5000 characters.

---

## GET `/api/languages`
```json
{ "count": 40, "languages": [ { "code": "gujlish", "name": "Gujlish (...)", "base": "gujarati", "romanized": true }, ... ] }
```

## GET `/api/model/info`
Returns model type, dataset sizes and the accuracy / F1 comparison of all trained models.

---

## Errors
| Status | When |
|---|---|
| 422 | Validation error (empty text, text too long, empty batch, >100 texts) |
| 413 | A batch item longer than 5000 characters |
| 404 | `/api/model/info` before the model has been trained |

---

## Client examples

**Python**
```python
import requests
r = requests.post("http://localhost:8000/api/detect", json={"text": "kem cho maja ma ne?"})
print(r.json()["language"])   # gujlish
```

**JavaScript**
```js
const res = await fetch("http://localhost:8000/api/detect", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "kaise ho bhai" })
});
console.log((await res.json()).language);   // hinglish
```

**cURL**
```bash
curl -X POST http://localhost:8000/api/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["kem cho", "kaise ho", "hello"]}'
```
