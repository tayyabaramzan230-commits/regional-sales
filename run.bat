@echo off
REM One-step local launcher (Windows)
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
streamlit run app.py
