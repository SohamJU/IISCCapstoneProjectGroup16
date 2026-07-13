"""Product agent configuration constants."""

from __future__ import annotations

from pathlib import Path


# ── ReAct loop ────────────────────────────────────────────────────────────
MAX_REACT_ITERATIONS = 8  # safety cap on reasoning loops
CONVERSATION_MEMORY_TURNS = 10  # recent turns to include in prompt

# ── Twitter samples ───────────────────────────────────────────────────────
USE_TWITTER_SAMPLES = False  # toggle twitter placeholder in prompt
TOP_K_TWITTER_RESULTS = 3  # for future vector store retrieval

# ── Schema path ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCT_SCHEMA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "product_catalog.schema.json"
)

# ── Search / Retrieval ────────────────────────────────────────────────────
BM25_TOP_K = 15          # candidates retrieved from BM25 index
VECTOR_TOP_K = 15        # candidates retrieved from vector store
HYBRID_FINAL_TOP_K = 5   # final results returned to the agent after RRF fusion
