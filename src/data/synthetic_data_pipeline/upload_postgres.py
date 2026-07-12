"""Upload generated datasets to PostgreSQL database.

Usage:
    python -m src.data.pipeline.upload_postgres
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.data import (
    POSTGRESQL_TABLES,
    POSTGRESQL_UPLOAD_BEHAVIOR,
    POSTGRESQL_AIVEN_PASSWORD
)
from src.data.postgresql import upload_dataframe_to_postgresql_db


def run(force: bool = False, behavior: str | None = None) -> None:
    """Iterate through defined tables and upload them to PostgreSQL."""
    
    upload_behavior = behavior or POSTGRESQL_UPLOAD_BEHAVIOR
    
    if upload_behavior.lower() == "skip" and not force:
        print("  [skip] PostgreSQL upload behavior is set to 'skip'.")
        return
        
    if not POSTGRESQL_AIVEN_PASSWORD:
        print("  ❌ [skip] PostgreSQL password not found in .env. Skipping upload.")
        return

    print(f"  Uploading datasets to PostgreSQL (behavior: {upload_behavior})...")

    for file_path, table_name in POSTGRESQL_TABLES.items():
        if not file_path.exists():
            print(f"  [warn] File not found for upload: {file_path.name}")
            continue
            
        print(f"  -> Uploading {file_path.name} to table '{table_name}'...")
        try:
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                print(f"  [warn] Unsupported file format for Postgres upload: {file_path.name}")
                continue

            upload_dataframe_to_postgresql_db(
                df=df, 
                table_name=table_name, 
                if_exists=upload_behavior
            )
            print(f"  ✅ Successfully uploaded {len(df)} rows to '{table_name}'.")
            
        except Exception as e:
            print(f"  ❌ Error uploading {table_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload datasets to PostgreSQL.")
    parser.add_argument("--force", action="store_true", help="Force upload even if set to skip.")
    parser.add_argument("--behavior", choices=["replace", "append", "skip"], help="Override default config behavior.")
    args = parser.parse_args()
    
    run(force=args.force, behavior=args.behavior)
