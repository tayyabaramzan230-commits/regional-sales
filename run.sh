#!/usr/bin/env bash
# One-step local launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
