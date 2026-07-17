"""Return agent configuration constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RETURN_SCHEMA_PATHS = [
    PROJECT_ROOT / "data" / "synthetic" / "returns.schema.json",
    PROJECT_ROOT / "data" / "synthetic" / "orders.schema.json",
    PROJECT_ROOT / "data" / "synthetic" / "order_items.schema.json",
]
MAX_REACT_ITERATIONS = 8
