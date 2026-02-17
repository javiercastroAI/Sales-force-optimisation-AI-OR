from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


# Keep smoke test deterministic and fast while still executing the full app workflow.
os.environ.setdefault("OPTIMAI_QA_FAST", "1")


def test_streamlit_app_renders_without_runtime_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=240)

    assert not at.exception
    assert at.title
    assert at.title[0].value == "OptimAI Field Sales Pilot"
