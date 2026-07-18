"""Build Pinecone vector indexes for policy and review retrieval.

Usage:
    python pipelines/03_build_vector_indexes.py
    python pipelines/03_build_vector_indexes.py --review-limit 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import build_vector_indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pinecone vector indexes")
    parser.add_argument(
        "--review-limit",
        type=int,
        default=5000,
        help="Maximum reviews to ingest into vector index",
    )
    args = parser.parse_args()

    stats = build_vector_indexes(review_limit=args.review_limit)
    print("Vector indexing complete:")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
