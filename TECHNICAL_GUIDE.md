# Language Detection - Technical Guide

This guide describes how the project is implemented, from its training data and
language-detection pipeline to the API, web interface, command line, and tests.
It complements the user-focused [README](./README.md), the [architecture
notes](./docs/ARCHITECTURE.md), and the [API reference](./docs/API.md).

## 1. What the application does

The project identifies text as English, Hindi, Gujarati, their Romanized forms
Hinglish and Gujlish, or one of the additional supported languages. It also:

- distinguishes native-script text from Latin-script text;
- identifies Romanized Hindi and Gujarati as Hinglish and Gujlish while
  preserving their base languages;
- provides a language tag for each token and estimates code mixing;
- optionally reports recognized Gen-Z slang and shortforms as a style layered
  on top of the language result;
- exposes the detector as a Python library, REST API, browser application, and
  command-line tool.

The implementation is hybrid. Unicode script rules handle much native-script
text; a trained classifier handles the more ambiguous Latin-script branch;
curated word lists supplement uncertain classifier results and provide
word-level labels and slang decoding.

## 2. Repository map

| Path | Responsibility |
|---|---|
| [detector/detector.py](./detector/detector.py) | Orchestrates normalization, script routing, classification, token tags, code-mixing metrics, and Gen-Z annotation. |
| [detector/normalizer.py](./detector/normalizer.py) | Unicode normalization, social-text cleanup, and tokenization. |
| [detector/scripts.py](./detector/scripts.py) | Unicode script profiling and script-to-language rules. |
| [detector/model.py](./detector/model.py) | TF-IDF / Logistic Regression pipeline construction, persistence, and lazy loading. |
| [detector/lexicons.py](./detector/lexicons.py) | Curated Romanized Hindi/Gujarati, shared Indic, English, and Devanagari marker vocabularies. |
| [detector/genz.py](./detector/genz.py) | Language-specific slang and shortform dictionaries, matching, meanings, and style score. |
| [data/seed/](./data/seed) | Hand-written text examples for the eight classifier labels. |
| [data/generate_dataset.py](./data/generate_dataset.py) | Seed splitting, deterministic augmentation, and CSV generation. |
| [training/train.py](./training/train.py) | Baseline/final model training, held-out evaluation, and artifact writing. |
| [training/evaluate.py](./training/evaluate.py) | End-to-end evaluation of script rules, model, lexicons, and Gen-Z detection. |
| [models/](./models) | Serialized classifier and reports/metrics produced by training. |
| [api/main.py](./api/main.py) | FastAPI app, request/response schemas, routes, and model initialization. |
| [frontend/index.html](./frontend/index.html) | Self-contained web UI (HTML, CSS, JavaScript) served by FastAPI. |
| [cli.py](./cli.py) | One-shot, JSON, file, and interactive command-line entry points. |
| [tests/](./tests) | Pytest checks for detection behavior and HTTP routes. |
| [requirements.txt](./requirements.txt) | Python runtime, API, training, and test dependencies. |
| [Dockerfile](./Dockerfile), [run.bat](./run.bat), [run.sh](./run.sh) | Container build and local startup paths. |

## 3. Detection request lifecycle

For each text, `LanguageDetector.detect()` in
[detector/detector.py](./detector/detector.py) performs these steps:

1. **Normalize and tokenize.** The original input is retained in the response;
   a cleaned copy is used by the classifier and token logic.
2. **Profile scripts.** Letters and combining marks are assigned Unicode
   writing systems, then script shares and the dominant script per token are
   calculated.
3. **Choose a classification route.** If more than one quarter of the tokens
   are non-Latin, the native-script route is used. Otherwise, Latin tokens are
   passed to the ML route. This permits English insertions in native Hindi text,
   for example, without routing the whole sentence to the English classifier.
4. **Tag tokens and calculate mixing.** Each token receives a language code;
   those tags are used to compute the Code-Mixing Index (CMI).
5. **Annotate style.** If the language is not `unknown`, the Gen-Z dictionaries
   are checked. This step does not replace the language classification.
6. **Apply output detail setting.** With `detailed=False`, `tokens` and
   `probabilities` are removed, but the sentence-level fields remain.

The detector returns method metadata (`empty`, `script`, `script+lexicon`,
`ml`, or `ml+lexicon`) to show which route produced the result.

### 3.1 Input normalization

[detector/normalizer.py](./detector/normalizer.py) uses Unicode NFC and then:

- removes URL and email strings;
- strips `@` / `#` markers while keeping their word text;
- lowercases text;
- keeps Unicode letters, combining marks, apostrophes, and digits, converting
  other characters (including punctuation and emoji) to spaces;
- reduces a run of three or more identical characters to two (for example,
  `soooo` becomes `soo`);
- drops numbers that form standalone tokens, while preserving digits inside
  tokens for chat spellings such as `6e`, `6o`, and `gr8`.

Tokenization splits the cleaned text on whitespace and trims apostrophes. It
does not apply Python's generic `\w` token pattern, which could split Indic
combining signs from their letters.

### 3.2 Script profiling and native-script route

[detector/scripts.py](./detector/scripts.py) identifies scripts by Unicode
ranges and checks Unicode character categories so punctuation/digits inside
those ranges do not count as letters. The implementation includes Latin,
Gujarati, Devanagari, Arabic, Cyrillic, Han, Kana, and other supported scripts.
The profile is the fraction of letters in each script, not the fraction of
tokens.

For scripts mapped directly to a language, the dominant non-Latin script
selects the result (for example Gujarati script maps to Gujarati). Shared
scripts receive lightweight disambiguation:

- **Devanagari:** counts Hindi, Marathi, and Nepali marker words, with a small
  Hindi prior.
- **Arabic:** checks distinctive Urdu characters first, then Persian hints;
  otherwise chooses Arabic.
- **Cyrillic:** Ukrainian-specific letters select Ukrainian; otherwise Russian.
- **Han:** any Hiragana/Katakana in the text selects Japanese; Han alone selects
  Chinese.
- **Bengali:** Assamese-specific characters select Assamese; otherwise Bengali.

Native-script confidence is a heuristic based on how much of the non-Latin
script profile belongs to the selected script. It is not a calibrated model
probability. For Devanagari, the marker-word vote supplies the class scores and
confidence instead.

### 3.3 Latin-script model

[detector/model.py](./detector/model.py) builds a scikit-learn `Pipeline` with
two parallel feature extractors and a Logistic Regression classifier:

- **Character TF-IDF:** `char_wb` n-grams of lengths 1-5, sublinear term
  frequency, minimum document frequency 2, and up to 120,000 features.
- **Word TF-IDF:** word unigrams and bigrams, sublinear term frequency, and a
  token pattern that includes Unicode word characters.
- **Classifier:** Logistic Regression with `C=8.0`, balanced class weights,
  and up to 3,000 iterations.

The eight trained labels are `english`, `hinglish`, `gujlish`, `spanish`,
`french`, `german`, `italian`, and `portuguese`. This model does not have a
separate trained class for every language in the API; most additional languages
are handled by native-script rules.

### 3.4 Lexicon blending and unknown results

[detector/lexicons.py](./detector/lexicons.py) contains Gujlish/Hinglish marker
words, words shared by Romanized Hindi and Gujarati, an English word list, and
Devanagari marker vocabularies. The shared list avoids treating ambiguous
short words as decisive Hindi-vs-Gujarati evidence.

For Latin text, the detector builds a softmax vote from English, Hinglish, and
Gujlish marker counts. That vote is blended with model probabilities only when
the best model probability is below `0.75` and there is lexicon evidence. The
lexicon weight is `0.60` for three or fewer Latin tokens and `0.35` otherwise.
If the winning blended probability is below `0.40` and there are no lexicon
hits, the detector returns `unknown` with a note naming the closest model
match. Empty or non-linguistic input returns `unknown` through the `empty`
route before model prediction.

### 3.5 Token labels and code mixing

For a Hinglish/Gujlish sentence, recognized English tokens are tagged `en`;
Indic marker/shared words and otherwise unseen Latin tokens take the
sentence-level Indic tag. This is a practical heuristic, not independent
token-by-token language inference. Native-script tokens are tagged using their
script, while Latin insertions are tagged `en`.

CMI is calculated over the non-unknown token tags:

```text
CMI = 100 * (1 - largest language token count / total tagged token count)
```

The `is_code_mixed` flag is true only when CMI is greater than zero and more
than one token language is represented. For a sentence with one language tag,
CMI is zero. `languages_in_text` is the sorted set of token language codes.

### 3.6 Gen-Z style and meanings

[detector/genz.py](./detector/genz.py) has separate word and phrase dictionaries
for English, Hindi, Gujarati, and shared Indic slang. Entries associate a
meaning and a weight. Phrase matching happens before individual-word matching,
so words consumed by a matched phrase are not counted twice. Applicable English
slang is considered alongside the base-language-specific vocabulary.

The weights are summed; a score of at least `1.0` sets `is_genz=true`,
`style="genz"`, and a display label such as `Gujarati (Gen Z)`. The underlying
`language` label remains unchanged (for example, `gujlish`). Detected terms and
their meanings are returned in `genz_terms`. Editing this dictionary does not
require retraining the language model.

## 4. Training and model artifacts

The seed corpus currently has eight text files, one per trained class, with
roughly 1,090 sentences in total. The exact source examples are under
[data/seed/](./data/seed). They are the source data; generated CSVs are outputs
of [data/generate_dataset.py](./data/generate_dataset.py).

The generation process:

1. Loads and deduplicates each class's seed sentences.
2. Shuffles using a fixed random seed (`42`) and splits the original seeds
   into train and test partitions (80/20, with at least three test seeds).
3. Augments each partition separately. For romanized English/Hinglish/Gujlish,
   changes may include known spelling variants, vowel deletion, letter
   stretching, recasing, sentence concatenation, and punctuation/emoji noise.
4. Writes shuffled `data/processed/train.csv` and `test.csv`. Because splitting
   occurs before augmentation, no augmented version of a test seed is copied
   into training.

`training/train.py` regenerates those CSVs, preprocesses the text through the
same normalizer, trains Naive Bayes and Linear SVM baselines, then fits the
final Logistic Regression pipeline. Metrics are computed on the generated
held-out test partition. It writes:

- `models/metrics.json` with labels, dataset sizes, model comparison,
  classification report, and confusion matrix values;
- `models/classification_report.txt` with the readable report;
- `models/confusion_matrix.png` when matplotlib is available;
- `models/latin_model.joblib`, fit one final time on the combined train and
  test rows after reporting held-out scores.

Consequently, the reported evaluation metrics come from the model trained
before the final combined-data refit, while the saved production model uses all
generated examples. `training/evaluate.py` separately tests the complete
detector on `data/eval_cases.csv` and Gen-Z behavior on `data/genz_eval.csv`.
These are small hand-authored checks; use held-out test metrics as the main
classifier measure.

### Model loading and automatic retraining

The serialized model is loaded by `load_model()` in
[detector/model.py](./detector/model.py). `get_detector()` caches one
`LanguageDetector` instance in-process. If the model file is absent or loading
raises an exception (including a scikit-learn compatibility warning promoted to
an exception), normal loading triggers training and then reloads the model.
Passing `auto_train=False` instead raises an error on a missing or unreadable
model. The model should only be loaded from a trusted source because joblib
serialization is Python pickle-based.

## 5. API and browser application

### API server

[api/main.py](./api/main.py) defines a FastAPI application and Pydantic models.
Its lifespan hook calls `get_detector()` at startup, so model load/retrain cost
is paid before serving requests rather than on the first detection request.
The service uses open CORS configuration and serves the single-page interface
at `/`.

| Route | Behavior |
|---|---|
| `GET /health` | Returns status and package version. |
| `POST /api/detect` | Detects one request-body text. `detailed` defaults to true. |
| `GET /api/detect?text=...` | Query-string equivalent of single-text detection. |
| `POST /api/detect/batch` | Detects 1-100 texts; each must be at most 5,000 characters. |
| `GET /api/languages` | Returns the language metadata table. |
| `GET /api/model/info` | Returns persisted training sizes, labels, and model comparisons. |
| `GET /docs`, `/redoc` | FastAPI-generated interactive API documentation. |

Pydantic enforces non-empty single text, a maximum 5,000-character single-text
length, and batch cardinality. The batch route explicitly returns HTTP 413 if
an item exceeds the text limit. Invalid schema inputs normally produce HTTP
422. `/api/model/info` returns HTTP 404 if `metrics.json` does not exist.
`detailed=false` omits tokens/probabilities on individual results. Batch
detection defaults to non-detailed results and includes a language-count
summary and aggregate processing time.

Responses include the original text, detected label/name, base language,
Romanized flag, script profile, confidence, CMI and code-mixing flag, token and
probability details (when requested), detection method, Gen-Z metadata, and
processing time. See [docs/API.md](./docs/API.md) for request/response examples
and field definitions.

### Browser UI

[frontend/index.html](./frontend/index.html) is a self-contained frontend with
no frontend package/build step. It has:

- a single-text form with sample prompts and Ctrl/Command+Enter support;
- a result view for the label, confidence meter, Romanized/mixed/Gen-Z badges,
  language share, top probabilities, and color-coded token labels;
- an expandable dialog for all tokens when the result is long;
- a batch tab which submits one non-empty line per item (up to 100) and renders
  up to six result rows;
- responsive behavior for narrow screens.

The UI calls relative `/api/detect` and `/api/detect/batch` paths, so it is
served from the same origin as the API. It renders user-provided text through
text content or an HTML-escaping helper before inserting it into markup.

## 6. Python library and CLI

The package exports `LanguageDetector`, `detect`, `get_detector`, and
`LANGUAGE_INFO` from [detector/__init__.py](./detector/__init__.py).
`detect(text, detailed=True)` is a convenience wrapper around the cached
detector. Use `LanguageDetector` directly when an instance is preferred or a
model object is injected for testing.

[cli.py](./cli.py) supports:

```text
python cli.py "kem cho maja ma ne?"
python cli.py "bro kal meeting hai office mein" --json
python cli.py --file sentences.txt
python cli.py
```

The file mode processes each non-blank line independently. Without arguments,
the CLI prompts until Ctrl+C/EOF. Normal mode prints a human-readable summary;
`--json` prints the full result as UTF-8 JSON.

## 7. Running, dependencies, and deployment

Python 3.10 or newer is expected. The dependencies in
[requirements.txt](./requirements.txt) cover scikit-learn/numpy/joblib for the
model, FastAPI/Uvicorn/Pydantic for the service, matplotlib for training
visualization, and pytest/httpx for tests.

- **Windows:** `run.bat` creates/activates `.venv`, installs requirements,
  trains if the model file is missing, then runs Uvicorn on port 8000.
- **macOS/Linux:** `run.sh` performs the analogous setup using Python 3 and a
  POSIX virtual environment.
- **Manual:** install dependencies, run `python training/train.py`, then
  `python -m uvicorn api.main:app --reload --port 8000`.
- **Docker:** [Dockerfile](./Dockerfile) uses `python:3.11-slim`, installs
  dependencies, trains the model during image build to match the installed
  scikit-learn version, exposes port 8000, and runs Uvicorn on `0.0.0.0:8000`.

The local startup scripts only train if the model file is absent. A model
already present but incompatible with the installed scikit-learn version is
handled by the detector's automatic load/retrain logic at API startup.

## 8. Tests and evaluation

Run the automated suite from the project root:

```bash
python -m pytest -v
```

The pytest suite covers representative Latin and native-script classification,
Romanized/base-language metadata, code-mixing, empty/noisy input, batch
detection, normalization and script profiles, Gen-Z dictionary behavior, API
validation and responses, language/model-info routes, and frontend serving.

For non-pytest quality measurements, run:

```bash
python training/evaluate.py
```

This reports end-to-end expected-label counts for the hand-written evaluation
CSVs and Gen-Z language/style checks. The project README records the current
reported held-out model scores and should be consulted for the latest checked-in
metrics.

## 9. Extending behavior

- **Improve classification data:** edit the relevant `data/seed/<label>.txt`
  and rerun `python training/train.py`.
- **Add a trained Latin class:** add the class's seed file and update `LABELS`
  and its augmentation target/logic if needed in
  [data/generate_dataset.py](./data/generate_dataset.py); then retrain. Also
  update the language metadata and any routing/tagging assumptions in
  [detector/detector.py](./detector/detector.py).
- **Add native-script support:** add or refine Unicode script ranges and
  script-language disambiguation in [detector/scripts.py](./detector/scripts.py),
  and add matching metadata in `LANGUAGE_INFO`.
- **Improve word tags or marker votes:** adjust the dictionaries and tagging
  rules in [detector/lexicons.py](./detector/lexicons.py) and
  [detector/detector.py](./detector/detector.py).
- **Add slang or meanings:** add weighted terms/phrases to
  [detector/genz.py](./detector/genz.py); model retraining is not required.
- **Change API limits/schema:** update the Pydantic models and route logic in
  [api/main.py](./api/main.py), keep [docs/API.md](./docs/API.md) in sync, and
  add/adjust API tests.
- **Change detection semantics:** extend [tests/test_detector.py](./tests/test_detector.py)
  with examples that cover both the intended new behavior and nearby
  regressions.

## 10. Known constraints

- Native-script identification is mostly a Unicode/script heuristic; it is not
  a full neural or statistical language model for every supported language.
- The trained Latin classifier only predicts its eight trained labels. Its
  `unknown` outcome is a threshold-based fallback, not a dedicated trained
  unknown class.
- Romanized spellings, slang, and code-mixed token assignments are inherently
  variable. Lexicons and synthetic augmentation help but cannot cover every
  dialect or spelling.
- Token-level language tags use dictionary membership and the sentence-level
  class for unknown words; they should be treated as informative estimates,
  not gold-standard linguistic annotations.
- Model confidence on the native-script route is rule-derived; classifier
  probabilities and rule confidence are not directly comparable.
- Retraining at first startup requires the seed data, training dependencies,
  and a writable project/model directory.
