"""Core POC smoke test for routing and orchestration.

Runs representative queries through SupportOrchestrator and prints routes.
This script is intentionally lightweight and focused on core flow validation.

Usage:
    python pipelines/04_core_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import SupportOrchestrator


QUERIES = [
    "Where is my order ORD-000001?",
    "I want to return my order item OI-0000001",
    "Recommend a cheaper alternative to what I bought last time",
    "Compare iPhone 15 and Samsung S24 camera specs",
    "I want to return my order and buy a replacement",
    "I need a human manager, this is unacceptable",
]


def main() -> None:
    orchestrator = SupportOrchestrator()
    session_id = "core-smoke"

    print("Core smoke test started")
    print("-" * 72)

    for i, query in enumerate(QUERIES, start=1):
        result = orchestrator.handle(query, session_id=session_id)
        routes = ",".join(result.routes) if result.routes else result.route
        print(f"{i}. query: {query}")
        print(f"   routes: {routes} | confidence={result.confidence:.2f}")
        print(f"   response preview: {result.response[:180].replace(chr(10), ' ')}")
        print("-")

    print("Core smoke test finished")


if __name__ == "__main__":
    main()
