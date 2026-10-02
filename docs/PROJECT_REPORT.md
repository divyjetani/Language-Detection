# Language Identification for English, Hindi, Gujarati and their Romanized, Code-Mixed Forms (Hinglish & Gujlish)

**Subject:** Natural Language Processing  
**Project:** Language Detection

---

## Abstract
Language identification (LID) is usually treated as solved for monolingual text
in standard scripts. It remains hard for Indian social-media text, where
Hindi and Gujarati are often typed in English letters (*"kem cho maja ma ne?"*,
*"kaise ho bhai?"*) and freely mixed with English words. Such romanized text
has no standard spelling and shares its alphabet with English, so the writing
system gives no clue. This project builds a hybrid LID system that combines
deterministic Unicode-script rules for native scripts with a character- and
word-n-gram Logistic Regression model for Latin-script text, supported by a
curated marker-word lexicon. It identifies English, Hindi, Gujarati, Hinglish
and Gujlish, plus 34 other languages, tags each word with its language, and
measures the degree of code-mixing with the Code-Mixing Index. On held-out
sentences never seen during training, the Latin-script model reaches 95.0 %
accuracy (F1: English 0.97, Gujlish 0.93, Hinglish 0.92). A second, rule-based
layer detects **Gen-Z style** (slang and shortforms such as *ngl, rizz, bawaal,
jalso, km 6o*) and labels text as e.g. *Gujarati (Gen Z)*. The system is
exposed through a REST API, a web interface and a command-line tool.

## 1. Introduction
India has hundreds of millions of internet users, and many of them write their
native languages in the Roman alphabet because standard keyboards default to
it. Search engines, chatbots, content moderation and translation all need to
know the language of a message before processing it. General-purpose tools such as
`langdetect` or Google's CLD3 are trained on standard-script text and have no
class for romanized Gujarati, so they cannot return "Gujlish" for
*"tame kya jao cho?"* and will force it into some other language.

**Objectives**
1. Detect English, Hindi and Gujarati in their native scripts.
2. Detect Hinglish and Gujlish (romanized Hindi and Gujarati) and report the real underlying language.
3. Identify other languages instead of forcing them into one of the five classes.
4. Provide word-level language tags and a code-mixing measure.
5. Deliver a usable system: Python library, REST API, web UI and CLI.

## 2. Literature review
* **Cavnar & Trenkle (1994)** introduced n-gram frequency profiles for text
  categorisation — the basis of most LID systems.
* **Lui & Baldwin (2012), *langid.py*** showed that a multinomial Naive Bayes
  model over byte n-grams generalises well across domains.
* **Google CLD3** uses a small neural network over character n-gram embeddings.
* **Das & Gambäck (2014)** studied code-mixing in Indian social media and proposed
  the **Code-Mixing Index (CMI)** used in this project.
* **FIRE shared tasks on Mixed Script Information Retrieval (2014–2016)**
  established word-level LID for romanized Hindi, Bengali, Gujarati and other
  Indian languages as a benchmark problem, and found that character n-grams
  plus dictionaries perform strongly.
* **Bhat et al. (2014)** and **Barman et al. (2014)** found that combining
  character n-gram classifiers with lexicon features improves word-level tagging
  of code-mixed text.

Gap addressed: most public tools ignore romanized Indian languages, and
romanized Gujarati in particular has very few resources compared with Hinglish.

## 3. Methodology

### 3.1 Problem decomposition
The key observation is that **script determines difficulty**:
* Text in Gujarati, Tamil, Bengali etc. script can be identified almost perfectly by
  Unicode ranges — no learning is needed.
* Devanagari (Hindi/Marathi/Nepali), Arabic (Arabic/Urdu/Persian) and Cyrillic
  need light disambiguation using distinctive words or characters.
* Latin-script text is the real challenge: English, Hinglish, Gujlish and
  European languages all use the same 26 letters.

So the system routes text by script, and uses machine learning only for the
Latin branch.

### 3.2 Dataset
No public Gujlish corpus of usable size exists, so a dataset was built for this
project:

| Class | Seed sentences |
|---|---|
| English | 207 |
| Hinglish | 275 |
| Gujlish | 272 |
| Spanish, French, German, Italian, Portuguese | 67 each |
| **Total** | **~1,090** |

About 30 sentences in each of English, Hinglish and Gujlish are written in
Gen-Z style, so the language model stays accurate on slang-heavy text.

Seeds cover everyday conversation (greetings, food, travel, work, college,
family, festivals) and include natural English insertions (*"meeting kal
savare che"*).

**Augmentation.** Romanized Indian text has no fixed spelling, so the generator
simulates real variation: spelling variants (*che / chhe / 6e*, *nahi / nhi*),
vowel dropping (*kar raha → kr rha*), letter stretching (*bhaaai*), sentence
concatenation, capitalisation and emoji noise. Seeds are **split into train and
test before augmentation**, so the test set measures generalisation to new
sentences, not recognition of re-spelled training sentences.
Final size: 11,000 training rows, 2,200 test rows.

### 3.3 Pre-processing
Unicode NFC normalisation, lower-casing, removal of URLs/e-mails/emojis/digits,
squeezing repeated characters, and an Indic-aware tokeniser that keeps
vowel signs attached to their consonants.

### 3.4 Features
* **Character n-grams (1–5, word-bounded), TF-IDF.** These capture
  sub-word patterns robust to spelling noise: Gujarati verb endings
  (*-vu, -vanu, -she, -ish*), the copula *che*, Hindi *hai / rha / nhi*,
  English *th / ing / tion*.
* **Word unigrams and bigrams, TF-IDF.** These capture function words and
  short phrases (*"kem cho"*, *"kya hai"*).

### 3.5 Models compared
| Model | Features |
|---|---|
| Multinomial Naive Bayes | char 1–4 |
| Linear SVM | char 1–4 |
| **Logistic Regression (final)** | char 1–5 + word 1–2 |

### 3.6 Lexicon component
High-precision marker lists for Gujlish (≈280 words), Hinglish (≈190 words) and
English (≈410 words). Words that are ambiguous between romanized Hindi and Gujarati
(*bhai, kal, ma, na, to, pan*) are excluded from the decision. When the model's top
probability is below 0.75, its distribution is blended with a softmax lexicon vote;
the lexicon receives more weight for inputs of three words or fewer, where
n-gram statistics are thin.

### 3.7 Code-mixing
Words are tagged *en / hi / gu*, and the Code-Mixing Index is computed as
`CMI = 100 × (1 − max_lang_words / tagged_words)`.

### 3.8 Gen-Z style detection
Gen-Z text is treated as a **register (style) of a language, not a separate
language**: *"ngl this biryani is bussin fr"* is still English, and
*"km 6o bhai, jalso padi gayo"* is still Gujlish. Making Gen-Z its own class would
mix up two different questions (which language? which style?) and would need far
more data. So style detection runs as a second layer after language detection:

1. A slang lexicon (`detector/genz.py`) has four parts: English Gen-Z slang and
   shortforms (≈95 entries: *rizz, delulu, lowkey, ngl, fr, istg, no cap, it's
   giving*), Hindi (≈40: *bawaal, jhakaas, kya scene hai, chill maar, kr rha h*),
   Gujarati (≈30: *jalso padi gayo, bhukka bolavi didha, fadu, km 6o, su6*), and
   slang shared by both (*chhapri, kadak, bindaas*).
2. English slang is counted for every language, because Indian Gen-Z mixes it
   into Hindi and Gujarati; Hindi or Gujarati slang is counted only when it matches
   the detected base language.
3. Each entry has a weight: 1.0 for unambiguous slang (*rizz*), 0.3–0.6 for
   common chat shortforms (*idk, h, kr*). Text is Gen-Z when the total reaches
   1.0, so one clear slang word is enough, but ordinary shortforms need to
   appear together. Words inside a matched phrase are not counted twice.
4. The output keeps the language label unchanged and adds `style`,
   `display_name` (*Gujarati (Gen Z)*), `genz_score` and the matched terms with
   their meanings.

The normaliser was changed to keep digits *inside* words, because Gujarati
Gen-Z spelling uses "6" for the *chh* sound (*6e = che, km6 = kem cho*).
Stand-alone numbers are still removed.

## 4. Results

### 4.1 Model comparison (held-out test set, 2,200 rows)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 89.3 % | 0.884 |
| Linear SVM | 94.3 % | 0.937 |
| **Logistic Regression** | **95.0 %** | **0.958** |

Logistic Regression gives the best accuracy and macro F1, and it also outputs
calibrated probabilities, which are needed for confidence scores, the
unknown-language threshold and lexicon blending.

### 4.2 Per-class results (Logistic Regression)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| English | 0.98 | 0.97 | 0.97 |
| Gujlish | 0.95 | 0.91 | 0.93 |
| Hinglish | 0.89 | 0.95 | 0.92 |
| French | 1.00 | 0.97 | 0.99 |
| German | 1.00 | 1.00 | 1.00 |
| Italian | 1.00 | 0.96 | 0.98 |
| Portuguese | 0.94 | 0.94 | 0.94 |
| Spanish | 0.94 | 0.94 | 0.94 |

### 4.3 Error analysis
* **Gujlish ↔ Hinglish** is the main confusion (44 Gujlish rows predicted as
  Hinglish, 22 the other way). The two share vocabulary (*bhai, kal, paisa,
  pagal*) and many short test strings contain only shared words.
  Gujarati-specific endings (*che, cho, nathi, -vu*) are usually present and
  decisive when they appear.
* **Spanish ↔ Portuguese** (8 rows each way) are confused because the languages
  are lexically close and their seed sets are small (67 sentences each).
* A handful of short European sentences are labelled Hinglish/Gujlish
  (e.g. 6 Italian rows), because short words like *ma, che, no* exist in both.

### 4.4 End-to-end system test
74 new sentences across 28 languages (native-script Hindi, Gujarati, Marathi,
Nepali, Tamil, Telugu, Urdu, Chinese, Japanese, etc., plus Hinglish, Gujlish
and English) were run through the full pipeline: **73 / 74 correct**. The one
error, *"bhai ek kaam kar ne"*, is Gujlish predicted as Hinglish; it contains
almost no Gujarati-only words. Because this set is small and hand-picked, the
held-out test-set figures in 4.1 are the more reliable estimate.

### 4.5 Gen-Z layer
On a separate set of 33 sentences not used in training (`data/genz_eval.csv`;
22 Gen-Z, 11 standard, across Gujlish, Hinglish, English, Hindi and Gujarati):

| Measure | Result |
|---|---|
| Language still correct | 33 / 33 |
| Gen-Z vs standard correct | 33 / 33 |
| False Gen-Z alarms on standard text | 0 |

The first run missed *"kya kr rhe ho aaj kl"*; the shortform *kl* (= *kal*) was
then added to the lexicon. Like 4.4, this set is small, so it shows that the
layer works rather than measuring its accuracy precisely.

### 4.6 Sample outputs
| Input | Output | Code-mixed (CMI) |
|---|---|---|
| kem cho maja ma ne? | Gujlish → Gujarati | no |
| bhai aaje meeting che office ma | Gujlish → Gujarati | yes (33.3) |
| bro kal meeting hai office mein | Hinglish → Hindi | yes |
| मैं office जा रहा हूँ | Hindi | yes |
| તમે ક્યાં રહો છો? | Gujarati | no |
| मी घरी जात आहे | Marathi | no |
| آپ کیسے ہیں | Urdu | no |
| asdkj qwpoe zxcmn | unknown | — |
| km 6o bhai, jalso padi gayo | **Gujarati (Gen Z)** (Gujlish) | — |
| bhai ye lowkey op hai fr | **Hindi (Gen Z)** (Hinglish) | yes |
| ngl this biryani is bussin fr | **English (Gen Z)** | — |

## 5. Implementation
* **Language:** Python 3.10+
* **Libraries:** scikit-learn (TF-IDF, Logistic Regression), FastAPI + Uvicorn (REST API),
  Pydantic (validation), Matplotlib (confusion matrix), pytest (49 automated tests).
* **Interfaces:** Python package `detector`, REST API with Swagger docs,
  responsive web UI, CLI.
* **Performance:** about 2 ms per text for the detector itself (≈10 ms per API call); training takes a few seconds.

## 6. Limitations
1. The dataset is self-built and conversational; formal or domain-specific
   romanized text (e.g. news headlines) is less well covered.
2. Very short, ambiguous inputs (*"bhai"*, *"kal"*) cannot be resolved between
   Hinglish and Gujlish — the information is simply not in the text.
3. Other romanized Indian languages (romanized Marathi, Bengali, Tamil) are not
   modelled separately; they will usually be labelled Hinglish or unknown.
4. Word-level tagging inside a Hinglish/Gujlish sentence separates English from the
   Indic language, but does not separate Hindi words from Gujarati words in the
   same sentence.
5. European languages beyond the five trained ones fall back to "unknown".
6. Slang changes fast. The Gen-Z lexicon is a hand-made list and has to be
   updated as new words appear; words that are also normal English (*bet, ate,
   mid, lit*) are given low weight so they don't trigger Gen-Z on their own.

## 7. Future work
* Collect real Gujlish/Hinglish data from social media and WhatsApp exports.
* Add romanized Marathi, Bengali, Tamil and Telugu classes.
* Train a word-level sequence model (CRF or BiLSTM-CRF) for finer code-mixing analysis.
* Fine-tune a multilingual transformer (e.g. MuRIL or IndicBERT) and compare with the n-gram model.
* Add transliteration: convert detected Gujlish into Gujarati script (*kem cho → કેમ છો*).

## 8. Conclusion
Routing text by script and using machine learning only where it is needed gives
a fast, explainable LID system. It handles the hard case of romanized,
code-mixed Hindi and Gujarati with about 92–93 % F1 on unseen sentences, and
covers 39 languages overall. A small, carefully augmented, leakage-free dataset
with character n-gram features was enough to tell Gujlish from Hinglish and
English reliably.

## References
1. Cavnar, W. B., & Trenkle, J. M. (1994). *N-Gram-Based Text Categorization.* SDAIR-94.
2. Lui, M., & Baldwin, T. (2012). *langid.py: An Off-the-shelf Language Identification Tool.* ACL System Demonstrations.
3. Das, A., & Gambäck, B. (2014). *Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text.* ICON.
4. Barman, U., Das, A., Wagner, J., & Foster, J. (2014). *Code Mixing: A Challenge for Language Identification in the Language of Social Media.* CodeSwitch Workshop, EMNLP.
5. Bhat, I. A., Mujadia, V., Tammewar, A., Bhat, R. A., & Shrivastava, M. (2014). *IIIT-H System Submission for FIRE 2014 Shared Task on Transliterated Search.* FIRE.
6. Sequiera, R., Choudhury, M., Gupta, P., Rosso, P., Kumar, S., Banerjee, S., Naskar, S. K., Bandyopadhyay, S., Chittaranjan, G., Das, A., & Chakma, K. (2015). *Overview of FIRE-2015 Shared Task on Mixed Script Information Retrieval.* FIRE.
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
8. The Unicode Consortium. *The Unicode Standard* — code charts for Indic scripts.
