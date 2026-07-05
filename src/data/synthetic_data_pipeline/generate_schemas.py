"""Generate JSON schema files for generated datasets.

Usage:
    python -m src.data.pipeline.generate_schemas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.data import (
    PRODUCT_CATALOG_PATH,
    REVIEWS_PROCESSED_PATH,
    CUSTOMERS_PATH,
    ORDERS_PATH,
    ORDER_ITEMS_PATH,
    RETURNS_PATH,
    CUSTOMER_QUERIES_PATH,
)


COLUMN_DESCRIPTIONS = {
    "order_id": "Unique identifier for the order.",
    "customer_id": "Unique identifier for the customer.",
    "product_id": "Unique identifier for the product (ASIN).",
    "order_item_id": "Unique identifier for the order line item.",
    "return_id": "Unique identifier for the return request.",
    "query_id": "Unique identifier for the customer query.",
    "user_id": "Unique identifier for the user who wrote the review.",
    "status": "Current status of the order or return (e.g., delivered, pending, approved).",
    "payment_status": "Status of the payment.",
    "item_status": "Status of the individual order item.",
    "reason": "Reason for the return.",
    "quantity": "Number of units purchased in the order item.",
    "unit_price": "Price per unit of the product.",
    "total_amount": "Total amount paid for the order.",
    "refund_amount": "Amount refunded for a return.",
    "order_date": "Date the order was placed.",
    "est_delivery_date": "Estimated date of delivery.",
    "actual_delivery_date": "Actual date of delivery.",
    "request_date": "Date the return was requested.",
    "tracking_number": "Shipping tracking number.",
    "shipping_address": "Customer's shipping address.",
    "category": "Product category.",
    "title": "Name or title of the product.",
    "price": "Current price of the product in the catalog.",
    "rating": "Review rating (1-5 stars).",
    "text": "Text content of the review.",
    "timestamp": "Time the review was submitted.",
    "query_text": "Text content of the customer's support query.",
    "query_type": "Category of the customer's query.",
    "expected_resolution": "The expected action to resolve the query.",
    "all_intents": "List of all intents extracted or matched.",
    "average_rating": "Average rating of the product.",
    "batch_id": "Identifier for the batch processing.",
    "bought_together": "List of products frequently bought together.",
    "city": "City of the customer's address.",
    "country": "Country of the customer's address.",
    "description": "Detailed description of the product.",
    "email": "Customer's email address.",
    "features": "List of product features or specifications.",
    "first_name": "Customer's first name.",
    "helpful_vote": "Number of helpful votes received on a review.",
    "intent": "Primary intent of the customer query.",
    "is_bestseller": "Indicator if the product is a bestseller.",
    "language": "Language of the review or query.",
    "last_name": "Customer's last name.",
    "loyalty_tier": "Customer's loyalty program tier (e.g., Gold, Silver).",
    "main_category": "Main category the product belongs to.",
    "persona": "Assigned customer persona.",
    "preferred_categories": "List of categories the customer prefers.",
    "query": "The text of the customer query.",
    "rating_count": "Total number of ratings the product has received.",
    "review_id": "Unique identifier for the review.",
    "signup_date": "Date the customer registered.",
    "state": "State or province of the customer's address.",
    "store": "Store or brand name.",
    "sub_categories": "Sub-categories the product belongs to.",
    "verified_purchase": "Indicates if the review is from a verified purchase."
}


def infer_schema(df: pd.DataFrame, file_path: Path) -> dict:
    """Infer the schema and structural details of a pandas DataFrame."""
    schema = {
        "file_name": file_path.name,
        "num_rows": len(df),
        "columns": []
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        # Extract a few non-null sample values for context
        samples = df[col].dropna().head(3).tolist()
        
        # Convert non-serializable types if necessary
        clean_samples = []
        for s in samples:
            if isinstance(s, (pd.Timestamp, pd.Timedelta)):
                clean_samples.append(str(s))
            else:
                clean_samples.append(s)

        col_desc = COLUMN_DESCRIPTIONS.get(col, "No description available.")
        
        col_info = {
            "name": col,
            "type": col_type,
            "description": col_desc,
            "sample_values": clean_samples
        }
        
        # Check for limited unique values for any low-cardinality column
        if df[col].nunique() < 15:
            unique_vals = df[col].dropna().unique().tolist()
            # Convert non-serializable types if necessary
            clean_unique_vals = []
            for uv in unique_vals:
                if isinstance(uv, (pd.Timestamp, pd.Timedelta)):
                    clean_unique_vals.append(str(uv))
                else:
                    clean_unique_vals.append(uv)
            col_info["unique_values"] = clean_unique_vals

        schema["columns"].append(col_info)
        
    return schema


def run(force: bool = False) -> None:
    """Generate schema JSON files for all key datasets."""
    datasets = [
        PRODUCT_CATALOG_PATH,
        REVIEWS_PROCESSED_PATH,
        CUSTOMERS_PATH,
        ORDERS_PATH,
        ORDER_ITEMS_PATH,
        RETURNS_PATH,
        CUSTOMER_QUERIES_PATH,
    ]
    
    for path in datasets:
        if not path.exists():
            continue
            
        schema_path = path.with_suffix(".schema.json")
        if schema_path.exists() and not force:
            print(f"  [skip] Schema already exists: {schema_path.name}")
            continue
            
        print(f"  Generating schema for {path.name}...")
        
        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                print(f"  [warn] Unsupported file extension for schema generation: {path.name}")
                continue
                
            schema = infer_schema(df, path)
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
                
            print(f"  ✅ Saved schema → {schema_path.name}")
            
        except Exception as e:
            print(f"  ❌ Error generating schema for {path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset schemas.")
    parser.add_argument("--force", action="store_true", help="Force regenerate schemas.")
    args = parser.parse_args()
    
    run(force=args.force)
