#!/usr/bin/env bash
# Starts the FastAPI backend and the Streamlit frontend. Run from the project root.
set -e
cd "$(dirname "$0")"
python -m uvicorn backend.api:app --port 8000 &
API_PID=$!
trap "kill $API_PID" EXIT
sleep 3
python -m streamlit run app/streamlit_app.py
