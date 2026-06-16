"""Project data configuration."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv() -> bool:
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file
load_dotenv()

# Shared directories
ARCHIVE_DATA_DIR = PROJECT_ROOT / "archive"
AMAZON_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AMAZON_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Amazon Kaggle data
AMAZON_KAGGLE_DATASET = "saurav9786/amazon-product-reviews"
AMAZON_RAW_DATA_FILENAME = "ratings_electronics.csv"
AMAZON_PROCESSED_DATA_FILENAME = "ratings_electronics_processed.csv"

# Twitter customer support Kaggle data
TWITTER_SUPPORT_DATA_FILENAME = "twcs.csv"
TWITTER_SUPPORT_DATA_DIR = ARCHIVE_DATA_DIR / "twcs"
TWITTER_SUPPORT_PROCESSED_FILENAME = "twitter_support_processed.csv"
TWITTER_SUPPORT_HISTORY_FILENAME = "twitter_support_conversation_history.csv"

# Product catalog document outputs
PRODUCT_CATALOG_PROCESSED_FILENAME = "product_catalog_processed.csv"

# PostgreSQL database (Aiven)
POSTGRESQL_AIVEN_PASSWORD = os.getenv("POSTGRESQL_AIVEN_PASSWORD")
POSTGRESQL_CONNECTION_STRING = (
    "postgresql://avnadmin:"
    f"{POSTGRESQL_AIVEN_PASSWORD}"
    "@pg-20bad560-myproject123-456.e.aivencloud.com:12548/defaultdb?sslmode=require"
)
POSTGRESQL_TABLE_NAME = "ratings_electronics"
