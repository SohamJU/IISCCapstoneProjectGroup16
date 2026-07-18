"""Order agent configuration constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORDER_SCHEMA_PATHS = [
    PROJECT_ROOT / "data" / "synthetic" / "orders.schema.json",
    PROJECT_ROOT / "data" / "synthetic" / "order_items.schema.json",
    PROJECT_ROOT / "data" / "synthetic" / "customers.schema.json",
]
MAX_REACT_ITERATIONS = 8
