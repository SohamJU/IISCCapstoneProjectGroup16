"""Escalation agent configuration constants."""

from __future__ import annotations

from pathlib import Path


# ── ReAct loop ────────────────────────────────────────────────────────────
MAX_REACT_ITERATIONS = 8  # safety cap on reasoning loops
CONVERSATION_MEMORY_TURNS = 10  # recent turns to include in prompt

# ── Knowledge base / RAG support paths ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
