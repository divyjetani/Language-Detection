"""
Machine-learning model for Latin-script text.

Features : TF-IDF of character n-grams (1-5, word-boundary aware)
           + TF-IDF of word unigrams/bigrams
Classifier: multinomial Logistic Regression (gives calibrated probabilities)

Character n-grams capture sub-word patterns that separate the languages even
with spelling noise: "chh", "6e", "vu", "jo " -> Gujlish; "hai", "rha", "nhi" ->
Hinglish; "th", "ing", "tion" -> English; "ção", "ñ", "eau" -> Portuguese,
Spanish, French.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .normalizer import normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "latin_model.joblib"


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2,
                                 sublinear_tf=True, max_features=120_000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
    ])
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def preprocess_many(texts):
    return [normalize(t) for t in texts]


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def _train_now():
    """Train the model on the spot (used on first run or after a library upgrade)."""
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from training.train import main as train_main
    print("[detector] Training model (one-time, ~10-30 seconds)...")
    train_main()


def load_model(path: Path = MODEL_PATH, auto_train: bool = True) -> Pipeline:
    """Load the trained model. If it is missing or was saved with an incompatible
    scikit-learn version, retrain it automatically."""
    import warnings
    path = Path(path)
    if path.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # version-mismatch warning -> retrain
                return joblib.load(path)
        except Exception as e:  # noqa: BLE001
            if not auto_train:
                raise
            print(f"[detector] Could not load saved model ({type(e).__name__}); retraining.")
    elif not auto_train:
        raise FileNotFoundError(f"Model not found at {path}. Run:  python training/train.py")
    _train_now()
    return joblib.load(path)
