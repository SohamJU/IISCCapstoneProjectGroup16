"""Configuration for unstructured data preprocessing artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_DATA_DIR = PROJECT_ROOT / "archive"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TWITTER_SUPPORT_DATA_FILENAME = "twcs.csv"
TWITTER_SUPPORT_DATA_DIR = ARCHIVE_DATA_DIR / "twcs"
TWITTER_SUPPORT_PROCESSED_FILENAME = "twitter_support_processed.csv"
TWITTER_SUPPORT_HISTORY_FILENAME = "twitter_support_conversation_history.csv"

PRODUCT_CATALOG_PROCESSED_FILENAME = "product_catalog_processed.csv"
