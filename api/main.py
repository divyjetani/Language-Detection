"""
REST API for the Language Detection project (FastAPI).

Run:
    uvicorn api.main:app --reload --port 8000
Then open:
    http://localhost:8000          -> web UI
    http://localhost:8000/docs     -> interactive Swagger API docs
    http://localhost:8000/redoc    -> ReDoc API docs
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from detector import LANGUAGE_INFO, __version__, get_detector  # noqa: E402

MAX_TEXT_LEN = 5000
MAX_BATCH = 100

@asynccontextmanager
async def lifespan(_app):
    get_detector()   # load the model once at startup so the first request is fast
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Language Detection API",
    version=__version__,
    description=(
        "Detects **English, Hindi, Gujarati**, their romanized forms **Hinglish** and "
        "**Gujlish** (e.g. *kem cho maja ma ne?*), plus 34 other languages. "
        "Also returns word-level language tags, a Code-Mixing Index, and a **Gen-Z style** flag "
        "(e.g. *Gujarati (Gen Z)* for \"km 6o bhai, jalso padi gayo\") with the slang it found."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------- schemas
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN, examples=["kem cho maja ma ne?"])
    detailed: bool = Field(True, description="Include word-level tags and class probabilities")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             examples=[["kem cho", "kaise ho", "how are you", "આપ કેમ છો"]])
    detailed: bool = False


class TokenTag(BaseModel):
    token: str
    script: str | None
    lang: str


class GenZTerm(BaseModel):
    term: str
    meaning: str


class DetectResponse(BaseModel):
    text: str
    language: str
    language_name: str
    base_language: str
    is_romanized: bool
    script: str | None
    confidence: float
    is_code_mixed: bool
    code_mixing_index: float
    languages_in_text: list[str]
    scripts: dict[str, float]
    probabilities: dict[str, float] | None = None
    tokens: list[TokenTag] | None = None
    method: str
    # Gen-Z layer
    style: str = "standard"
    display_name: str | None = None
    is_genz: bool = False
    genz_score: float = 0.0
    genz_terms: list[GenZTerm] = []
    note: str | None = None
    processing_time_ms: float | None = None


# ----------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/languages", tags=["info"])
def languages():
    """All languages the system can return."""
    return {
        "count": len(LANGUAGE_INFO),
        "languages": [{"code": k, **v} for k, v in LANGUAGE_INFO.items()],
    }


@app.post("/api/detect", response_model=DetectResponse, response_model_exclude_none=True, tags=["detection"])
def detect_post(req: DetectRequest):
    """Detect the language of one text."""
    t0 = time.perf_counter()
    res = get_detector().detect(req.text, detailed=req.detailed)
    res["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return res


@app.get("/api/detect", response_model=DetectResponse, response_model_exclude_none=True, tags=["detection"])
def detect_get(text: str = Query(..., min_length=1, max_length=MAX_TEXT_LEN), detailed: bool = True):
    """Same as POST /api/detect, handy for quick browser testing: /api/detect?text=kem%20cho"""
    return detect_post(DetectRequest(text=text, detailed=detailed))


@app.post("/api/detect/batch", tags=["detection"])
def detect_batch(req: BatchRequest):
    """Detect the language of up to 100 texts at once."""
    for t in req.texts:
        if len(t) > MAX_TEXT_LEN:
            raise HTTPException(413, f"Each text must be at most {MAX_TEXT_LEN} characters")
    t0 = time.perf_counter()
    results = get_detector().detect_batch(req.texts, detailed=req.detailed)
    summary = {}
    for r in results:
        summary[r["language"]] = summary.get(r["language"], 0) + 1
    return {
        "count": len(results),
        "summary": summary,
        "results": results,
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/api/model/info", tags=["info"])
def model_info():
    """Training metrics of the Latin-script ML model."""
    path = ROOT / "models" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found - run training/train.py")
    m = json.loads(path.read_text())
    return {
        "model": "TF-IDF (char 1-5 + word 1-2 n-grams) + Logistic Regression",
        "train_size": m["train_size"],
        "test_size": m["test_size"],
        "labels": m["labels"],
        "model_comparison": m["model_comparison"],
    }
