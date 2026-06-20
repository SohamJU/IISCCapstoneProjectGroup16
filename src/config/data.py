"""Database configuration"""

import os
from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file
load_dotenv()

# Amazon Kaggle data
AMAZON_KAGGLE_DATASET = "saurav9786/amazon-product-reviews"
AMAZON_RAW_DATA_FILENAME = "ratings_electronics.csv"
AMAZON_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AMAZON_PROCESSED_DATA_FILENAME = "ratings_electronics_processed.csv"
AMAZON_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# PostgreSQL database (Aiven)
POSTGRESQL_AIVEN_PASSWORD = os.getenv("POSTGRESQL_AIVEN_PASSWORD")
POSTGRESQL_CONNECTION_STRING = f"postgresql://avnadmin:{POSTGRESQL_AIVEN_PASSWORD}@pg-20bad560-myproject123-456.e.aivencloud.com:12548/defaultdb?sslmode=require"
POSTGRESQL_TABLE_NAME = "ratings_electronics"