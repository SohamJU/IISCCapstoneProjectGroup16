"""Run core POC checks and smoke flow in sequence.

Steps:
1. Readiness check
2. Build vector indexes
3. Core smoke test

Usage:
    python pipelines/06_run_core_poc.py
    python pipelines/06_run_core_poc.py --review-limit 2000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_step(args: list[str], label: str) -> None:
    print("=" * 72)
    print(label)
    print("=" * 72)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core POC flow")
    parser.add_argument(
        "--review-limit",
        type=int,
        default=5000,
        help="Review ingestion limit for vector indexing step",
    )
    parsed = parser.parse_args()

    python = sys.executable

    _run_step(
        [python, str(PROJECT_ROOT / "pipelines" / "05_core_readiness_check.py")],
        "STEP 1/3: Core readiness check",
    )
    _run_step(
        [
            python,
            str(PROJECT_ROOT / "pipelines" / "03_build_vector_indexes.py"),
            "--review-limit",
            str(parsed.review_limit),
        ],
        "STEP 2/3: Build vector indexes",
    )
    _run_step(
        [python, str(PROJECT_ROOT / "pipelines" / "04_core_smoke_test.py")],
        "STEP 3/3: Core smoke test",
    )

    print("\nCore POC run complete.")


if __name__ == "__main__":
    main()
