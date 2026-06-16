"""Run structured and unstructured preprocessing workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.loaders import (
    load_electronics_reviews,
    load_twitter_support_conversations,
)
from src.data.preprocessing import (
    build_conversation_history,
    preprocess_customer_support_conversations,
    preprocess_product_catalog,
    preprocess_ratings,
    save_conversation_history,
    save_customer_support_conversations,
    save_processed_data,
    save_product_catalog,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run repository preprocessing steps.")
    parser.add_argument(
        "--dataset",
        choices=["twitter", "ratings", "product", "all"],
        default="twitter",
        help="Which preprocessing flow to run.",
    )
    parser.add_argument(
        "--twitter-path",
        type=Path,
        default=None,
        help="Optional path to the Twitter support CSV.",
    )
    parser.add_argument(
        "--product-path",
        type=Path,
        default=None,
        help="Optional path to a product catalog CSV.",
    )
    parser.add_argument(
        "--customer-only",
        action="store_true",
        help="Keep only inbound customer tweets in the processed Twitter dataset.",
    )
    return parser.parse_args()


def run_twitter_preprocessing(csv_path: Path | None, customer_only: bool) -> None:
    """Preprocess customer support conversations and save outputs."""

    conversations_df = load_twitter_support_conversations(file_path=csv_path)
    processed_conversations = preprocess_customer_support_conversations(
        conversations_df,
        customer_only=customer_only,
    )
    conversation_history = build_conversation_history(processed_conversations)

    conversations_path = save_customer_support_conversations(processed_conversations)
    history_path = save_conversation_history(conversation_history)

    print(f"Saved processed conversations to {_display_path(conversations_path)}")
    print(f"Saved conversation history to {_display_path(history_path)}")


def run_ratings_preprocessing() -> None:
    """Preprocess the structured ratings dataset and save the result."""

    ratings_df = load_electronics_reviews()
    processed_ratings = preprocess_ratings(ratings_df)
    save_processed_data(processed_ratings)


def run_product_preprocessing(csv_path: Path | None) -> None:
    """Preprocess a product catalog CSV and save the result."""

    if csv_path is None:
        raise ValueError("--product-path is required when dataset is 'product' or 'all'.")

    product_df = pd.read_csv(csv_path)
    processed_products = preprocess_product_catalog(product_df)
    output_path = save_product_catalog(processed_products)
    print(f"Saved processed product catalog to {_display_path(output_path)}")


def main() -> None:
    """Run the selected preprocessing flows."""

    args = parse_args()

    if args.dataset in {"twitter", "all"}:
        run_twitter_preprocessing(args.twitter_path, args.customer_only)

    if args.dataset in {"ratings", "all"}:
        run_ratings_preprocessing()

    if args.dataset in {"product", "all"}:
        run_product_preprocessing(args.product_path)


def _display_path(path: Path) -> str:
    """Render paths relative to the repository root when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
