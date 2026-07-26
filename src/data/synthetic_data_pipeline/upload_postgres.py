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
    POSTGRESQL_CONNECTION_STRING,
    POSTGRESQL_TABLES,
    POSTGRESQL_UPLOAD_BEHAVIOR,
)
from src.data.postgresql import upload_dataframe_to_postgresql_db


def run(
    force: bool = False,
    behavior: str | None = None,
    confirm_replace: bool = False,
) -> None:
    """Iterate through defined tables and upload them to PostgreSQL.

    Args:
        force: Proceed even when the configured behavior is 'skip'.
        behavior: Override the configured behavior ('fail', 'append', 'replace').
        confirm_replace: Required to allow 'replace', which DROPS each table.
    """
    upload_behavior = (behavior or POSTGRESQL_UPLOAD_BEHAVIOR).lower()

    if upload_behavior == "skip":
        if not force:
            print("  [skip] PostgreSQL upload behavior is set to 'skip'.")
            return
        # Previously --force passed the literal string "skip" through to
        # pandas.to_sql, which only accepts fail/replace/append — so every
        # table raised. Resolve the sentinel to the safe default instead.
        print("  [force] behavior 'skip' resolved to 'fail' (non-destructive).")
        upload_behavior = "fail"

    if upload_behavior == "replace" and not confirm_replace:
        print(
            "  ❌ Refusing to run with behavior 'replace': this DROPS every table\n"
            "     listed in POSTGRESQL_TABLES on the SHARED team database.\n"
            "     Re-run with --confirm-replace if that is genuinely intended."
        )
        return

    # Check the credential actually used to connect. This previously tested
    # POSTGRESQL_AIVEN_PASSWORD, which is an empty string in this project —
    # the real credentials are embedded in POSTGRESQL_CONNECTION_STRING. The
    # guard was therefore always true, so the upload silently refused to run
    # for everyone while reporting a missing password.
    if not POSTGRESQL_CONNECTION_STRING:
        print(
            "  ❌ [skip] POSTGRESQL_CONNECTION_STRING not set in .env. "
            "Skipping upload."
        )
        return

    if upload_behavior == "replace":
        print(
            "  ⚠️  DESTRUCTIVE: dropping and recreating "
            f"{len(POSTGRESQL_TABLES)} tables on the shared database."
        )

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
                if_exists=upload_behavior,
                confirm_destructive=confirm_replace,
            )
            print(f"  ✅ Successfully uploaded {len(df)} rows to '{table_name}'.")
            
        except Exception as e:
            print(f"  ❌ Error uploading {table_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload datasets to PostgreSQL.")
    parser.add_argument(
        "--force", action="store_true", help="Force upload even if set to skip."
    )
    parser.add_argument(
        "--behavior",
        choices=["fail", "append", "replace", "skip"],
        help="Override default config behavior.",
    )
    parser.add_argument(
        "--confirm-replace",
        action="store_true",
        help="Required with --behavior replace. DROPS every target table on the "
        "shared team database.",
    )
    args = parser.parse_args()

    run(
        force=args.force,
        behavior=args.behavior,
        confirm_replace=args.confirm_replace,
    )
