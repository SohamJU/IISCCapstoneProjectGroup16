#!/usr/bin/env python3
"""
Migration script to initialize customer_sessions table in PostgreSQL.

Usage:
    python -m pipelines.setup_sessions_table
    or
    python pipelines/setup_sessions_table.py
"""

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.session_persistence import initialize_sessions_table


def main() -> None:
    """Initialize the customer_sessions table."""
    print("=" * 70)
    print("Customer Sessions Table Migration")
    print("=" * 70)
    
    print("\n[*] Initializing customer_sessions table...")
    initialize_sessions_table()
    
    print("\n[✓] Migration completed successfully!")
    print("\nTable Schema:")
    print("""
    Table: customer_sessions
    ├── id (SERIAL PRIMARY KEY)
    ├── customer_id (VARCHAR(255) NOT NULL)
    ├── session_id (VARCHAR(255) NOT NULL UNIQUE)
    ├── conversation_turns (JSONB NOT NULL)
    ├── created_at (TIMESTAMP NOT NULL)
    ├── updated_at (TIMESTAMP NOT NULL)
    ├── is_active (BOOLEAN NOT NULL)
    └── Indexes:
        ├── idx_customer_id (customer_id)
        ├── idx_created_at (created_at DESC)
        └── idx_is_active (is_active)
    """)
    print("\nYou can now use session_persistence functions to store/retrieve conversations.")


if __name__ == "__main__":
    main()
