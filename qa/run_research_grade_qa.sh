#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p qa

{
  echo "[QA] Started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[QA] Python: $(python3 --version)"
  echo "[QA] Step 1/2: Compile checks"
  python3 -m compileall optimai_pipeline.py streamlit_app.py
  echo "[QA] Step 2/2: Pytest suite"
  MPLBACKEND=Agg OPTIMAI_QA_FAST=1 python3 -m pytest -q tests --durations=10 --junitxml qa/pytest-report.xml
  echo "[QA] Completed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} 2>&1 | tee qa/last_qa_run.log
