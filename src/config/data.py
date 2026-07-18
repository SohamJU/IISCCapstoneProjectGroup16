"""Database configuration"""

import os
from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.data import postgresql



# Load environment variables from .env file
load_dotenv()

# Amazon Kaggle data
AMAZON_KAGGLE_DATASET = "saurav9786/amazon-product-reviews"
AMAZON_RAW_DATA_FILENAME = "ratings_electronics.csv"
AMAZON_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AMAZON_PROCESSED_DATA_FILENAME = "ratings_electronics_processed.csv"
AMAZON_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# PostgreSQL database (Aiven) shared group endpoints
POSTGRESQL_HOST = os.getenv(
    "POSTGRESQL_HOST", "pg-20bad560-myproject123-456.e.aivencloud.com"
)
POSTGRESQL_PORT = os.getenv("POSTGRESQL_PORT", "12548")

POSTGRESQL_USER = os.getenv("POSTGRESQL_USER", "avnadmin")
POSTGRESQL_DB = os.getenv("POSTGRESQL_DB", "defaultdb")
# Upload Behavior (replace, append, skip, fail)
POSTGRESQL_UPLOAD_BEHAVIOR = "skip"

# Password remains in .env
POSTGRESQL_AIVEN_PASSWORD = os.getenv("POSTGRESQL_AIVEN_PASSWORD")
if POSTGRESQL_AIVEN_PASSWORD is None:
    raise ValueError("POSTGRESQL_AIVEN_PASSWORD environment variable not set")
POSTGRESQL_CONNECTION_STRING = f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_AIVEN_PASSWORD}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DB}?sslmode=require"
# ---------------------------------------------------------------------------
# Amazon Reviews 2023 (McAuley Lab / Hugging Face)
# ---------------------------------------------------------------------------
AMAZON_2023_CATEGORIES = ["Electronics", "Appliances"]
AMAZON_2023_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "amazon_2023"
AMAZON_2023_MAX_REVIEWS = 10_000  # cap per category for POC

# ---------------------------------------------------------------------------
# Processed data paths
# ---------------------------------------------------------------------------
PRODUCT_CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "product_catalog.csv"
REVIEWS_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.csv"

# ---------------------------------------------------------------------------
# Synthetic data paths
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
CUSTOMERS_PATH = SYNTHETIC_DIR / "customers.csv"
ORDERS_PATH = SYNTHETIC_DIR / "orders.csv"
ORDER_ITEMS_PATH = SYNTHETIC_DIR / "order_items.csv"
RETURNS_PATH = SYNTHETIC_DIR / "returns.csv"
CUSTOMER_QUERIES_PATH = SYNTHETIC_DIR / "customer_queries.csv"

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# ---------------------------------------------------------------------------
# Scale targets
# ---------------------------------------------------------------------------
NUM_CUSTOMERS = 2_000
NUM_ORDERS = 10_000
NUM_RETURNS = 2_000

# ---------------------------------------------------------------------------
# PostgreSQL Tables Mapping
# ---------------------------------------------------------------------------
POSTGRESQL_TABLES = {
    PRODUCT_CATALOG_PATH: "product_catalog",
    REVIEWS_PROCESSED_PATH: "reviews",
    CUSTOMERS_PATH: "customers",
    ORDERS_PATH: "orders",
    ORDER_ITEMS_PATH: "order_items",
    RETURNS_PATH: "returns",
    CUSTOMER_QUERIES_PATH: "customer_queries",
}
