"""Download Amazon Reviews 2023 data from Hugging Face using Requests.

Downloads product metadata and customer reviews for configured categories
(Electronics, Appliances). Images and videos columns are excluded.

Usage:
    python -m src.data.pipeline.download_amazon_2023
    python -m src.data.pipeline.download_amazon_2023 --max-reviews 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.data import (
    AMAZON_2023_CATEGORIES,
    AMAZON_2023_MAX_REVIEWS,
    AMAZON_2023_RAW_DIR,
)

# Columns to drop from metadata (not needed for chatbot)
META_DROP_COLUMNS = ["images", "videos"]
# Columns to drop from reviews
REVIEW_DROP_COLUMNS = ["images"]


def download_jsonl_subset(
    url: str, output_path: Path, max_rows: int, drop_columns: list[str], desc: str
) -> Path:
    """Stream a JSONL file from a URL and save up to max_rows as Parquet."""
    if output_path.exists():
        print(f"  [skip] {desc} already exists: {output_path.name}")
        return output_path

    print(f"  Downloading {desc} (max {max_rows:,} rows) …")
    rows = []
    
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        for i, line in enumerate(r.iter_lines()):
            if i >= max_rows:
                break
            if line:
                try:
                    data = json.loads(line)
                    # Drop heavy columns
                    for col in drop_columns:
                        data.pop(col, None)
                    rows.append(data)
                except json.JSONDecodeError:
                    continue
            
            if (i + 1) % 50000 == 0:
                print(f"    Loaded {(i + 1):,} rows...")

    df = pd.DataFrame(rows)
    
    # Coerce known mixed-type columns to string to prevent Pyarrow conversion errors
    for col in ["price", "average_rating", "rating_number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    df.to_parquet(str(output_path), index=False)
    print(f"    Saved {len(df):,} rows -> {output_path.name}")
    return output_path


def download_metadata(category: str, max_products: int, force: bool = False) -> Path:
    """Download product metadata for a category, excluding images/videos."""
    output_path = AMAZON_2023_RAW_DIR / f"meta_{category}.parquet"
    if force and output_path.exists():
        output_path.unlink()
        
    AMAZON_2023_RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"
    return download_jsonl_subset(
        url=url,
        output_path=output_path,
        max_rows=max_products,
        drop_columns=META_DROP_COLUMNS,
        desc=f"metadata for {category}",
    )


def download_reviews(
    category: str,
    max_reviews: int = AMAZON_2023_MAX_REVIEWS,
    force: bool = False,
) -> Path:
    """Download customer reviews for a category, capped at *max_reviews*."""
    output_path = AMAZON_2023_RAW_DIR / f"reviews_{category}.parquet"
    if force and output_path.exists():
        output_path.unlink()

    AMAZON_2023_RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
    return download_jsonl_subset(
        url=url,
        output_path=output_path,
        max_rows=max_reviews,
        drop_columns=REVIEW_DROP_COLUMNS,
        desc=f"reviews for {category}",
    )


def download_all(
    max_reviews: int = AMAZON_2023_MAX_REVIEWS,
    streaming: bool = False,
    force: bool = False,
) -> None:
    """Download metadata and reviews for all configured categories."""
    
    # We download significantly more metadata to ensure we have a good overlap with reviews.
    max_products = max_reviews * 10
    
    for category in AMAZON_2023_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        download_metadata(category, max_products=max_products, force=force)
        download_reviews(category, max_reviews=max_reviews, force=force)

    print(f"\n✅ Download complete. Files at: {AMAZON_2023_RAW_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 data.")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=AMAZON_2023_MAX_REVIEWS,
        help=f"Max reviews per category (default: {AMAZON_2023_MAX_REVIEWS:,}).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="(Ignored) Legacy compatibility flag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_all(
        max_reviews=args.max_reviews,
        force=args.force,
    )
