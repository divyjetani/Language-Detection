import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_detect_post():
    r = client.post("/api/detect", json={"text": "kem cho maja ma ne?"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "gujlish"
    assert "tokens" in body and "probabilities" in body


def test_detect_get_not_detailed():
    r = client.get("/api/detect", params={"text": "kaise ho bhai", "detailed": False})
    body = r.json()
    assert body["language"] == "hinglish"
    assert "tokens" not in body


def test_batch():
    r = client.post("/api/detect/batch", json={"texts": ["kem cho", "kaise ho", "હું ઘરે છું"]})
    body = r.json()
    assert body["count"] == 3
    assert [x["language"] for x in body["results"]] == ["gujlish", "hinglish", "gujarati"]


def test_validation():
    assert client.post("/api/detect", json={"text": ""}).status_code == 422
    assert client.post("/api/detect/batch", json={"texts": []}).status_code == 422


def test_languages_and_model_info():
    assert client.get("/api/languages").json()["count"] >= 40
    assert "model_comparison" in client.get("/api/model/info").json()


def test_frontend():
    r = client.get("/")
    assert r.status_code == 200 and "Language Detector" in r.text


def test_genz_fields_in_api():
    body = client.post("/api/detect", json={"text": "km 6o bhai, jalso padi gayo"}).json()
    assert body["is_genz"] is True
    assert body["display_name"] == "Gujarati (Gen Z)"
    assert {"term", "meaning"} <= set(body["genz_terms"][0])
