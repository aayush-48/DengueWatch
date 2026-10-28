@echo off
REM Starts the FastAPI backend and the Streamlit frontend (Windows). Run from the project root.
cd /d %~dp0
start "DengueWatch API" cmd /k python -m uvicorn backend.api:app --port 8000
timeout /t 4 >nul
python -m streamlit run app\streamlit_app.py
