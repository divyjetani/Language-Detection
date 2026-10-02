#!/usr/bin/env bash
# One-command setup + run (macOS / Linux)
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f "models/latin_model.joblib" ]; then
  python training/train.py
fi
echo ""
echo "  Web app : http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
