"""Return agent configuration constants."""

from __future__ import annotations

from pathlib import Path


MAX_REACT_ITERATIONS = 8
RETURN_WINDOW_DAYS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
