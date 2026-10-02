@echo off
REM One-command setup + run (Windows)
cd /d "%~dp0"
if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if not exist "models\latin_model.joblib" (
  python training\train.py
)
echo.
echo   Web app : http://localhost:8000
echo   API docs: http://localhost:8000/docs
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
