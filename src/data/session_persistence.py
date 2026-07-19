"""PostgreSQL session persistence layer for customer conversations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import Json

from src.config.data import POSTGRESQL_CONNECTION_STRING
from src.memory.conversation_memory import ConversationMemory


def initialize_sessions_table() -> None:
    """
    Create the customer_sessions table in PostgreSQL if it doesn't exist.
    
    Schema:
    - id: UUID primary key (auto-generated)
    - customer_id: VARCHAR, foreign key reference to customers table
    - session_id: VARCHAR, unique session identifier
    - conversation_turns: JSONB, stores conversation turns as JSON array
    - created_at: TIMESTAMP, when the session was started
    - updated_at: TIMESTAMP, when the session was last updated
    - is_active: BOOLEAN, whether session is still ongoing
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        # create_table_sql = """
        # CREATE TABLE IF NOT EXISTS customer_sessions (
        #     id SERIAL PRIMARY KEY,
        #     customer_id VARCHAR(255) NOT NULL,
        #     session_id VARCHAR(255) NOT NULL UNIQUE,
        #     conversation_turns JSONB NOT NULL DEFAULT '[]'::jsonb,
        #     created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        #     updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        #     is_active BOOLEAN NOT NULL DEFAULT TRUE,
        #     INDEX idx_customer_id (customer_id),
        #     INDEX idx_created_at (created_at DESC),
        #     INDEX idx_is_active (is_active)
        # );
        # """
        create_table_sql = """
                -- 1. Create the base table
        CREATE TABLE IF NOT EXISTS customer_sessions (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL UNIQUE,
            conversation_turns JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        );

        -- 2. Create the standalone indexes
        CREATE INDEX IF NOT EXISTS idx_customer_sessions_customer_id 
            ON customer_sessions (customer_id);

        CREATE INDEX IF NOT EXISTS idx_customer_sessions_created_at 
            ON customer_sessions (created_at DESC);

        -- Optimizes lookups for active sessions (the most common query pattern)
        CREATE INDEX IF NOT EXISTS idx_customer_sessions_active_lookup 
            ON customer_sessions (customer_id) 
            WHERE is_active = TRUE;
        """
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        conn.close()
        print("[✓] customer_sessions table initialized successfully")
    except Exception as e:
        print(f"[✗] Error initializing sessions table: {e}")
        raise


def save_session_to_db(
    customer_id: str,
    session_id: str,
    conversation_memory: ConversationMemory,
) -> bool:
    """
    Save or update a conversation session in PostgreSQL.
    
    Args:
        customer_id: Unique identifier for the customer
        session_id: Unique identifier for this session
        conversation_memory: ConversationMemory object with turns
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        # Convert turns to JSON-serializable format
        turns_data = [
            {
                "role": turn.role,
                "text": turn.text,
                "metadata": turn.metadata or {},
            }
            for turn in conversation_memory.turns
        ]

        # Upsert: if session_id exists, update it; otherwise insert
        upsert_sql = """
        INSERT INTO customer_sessions 
            (customer_id, session_id, conversation_turns, updated_at)
        VALUES 
            (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (session_id)
        DO UPDATE SET
            conversation_turns = EXCLUDED.conversation_turns,
            updated_at = CURRENT_TIMESTAMP,
            is_active = TRUE
        RETURNING id;
        """

        cur.execute(upsert_sql, (customer_id, session_id, Json(turns_data)))
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return result is not None
    except Exception as e:
        print(f"[✗] Error saving session to database: {e}")
        return False


def get_customer_session_history(
    customer_id: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    """
    Retrieve the N most recent sessions for a customer from PostgreSQL.
    
    Args:
        customer_id: Unique identifier for the customer
        limit: Maximum number of sessions to retrieve (default: 5)
    
    Returns:
        List of session dictionaries with keys:
        - session_id: str
        - conversation_turns: list[dict]
        - created_at: str (ISO format)
        - updated_at: str (ISO format)
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        query_sql = """
        SELECT 
            session_id,
            conversation_turns,
            created_at::text,
            updated_at::text,
            is_active
        FROM customer_sessions
        WHERE customer_id = %s
        ORDER BY updated_at DESC
        LIMIT %s;
        """

        cur.execute(query_sql, (customer_id, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "conversation_turns": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "is_active": row[4],
            })

        return sessions
    except Exception as e:
        print(f"[✗] Error retrieving session history: {e}")
        return []


def load_session_from_db(session_id: str) -> Optional[ConversationMemory]:
    """
    Load a specific session from PostgreSQL and reconstruct the ConversationMemory.
    
    Args:
        session_id: Unique identifier for the session
    
    Returns:
        ConversationMemory object if found, None otherwise
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        query_sql = """
        SELECT conversation_turns, customer_id
        FROM customer_sessions
        WHERE session_id = %s;
        """

        cur.execute(query_sql, (session_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row is None:
            return None

        turns_data, customer_id = row
        
        # Reconstruct ConversationMemory from stored turns
        memory = ConversationMemory(session_id=session_id)
        for turn in turns_data:
            memory.add_turn(
                role=turn["role"],
                text=turn["text"],
                metadata=turn.get("metadata"),
            )

        return memory
    except Exception as e:
        print(f"[✗] Error loading session from database: {e}")
        return None


def close_session(session_id: str) -> bool:
    """
    Mark a session as inactive (closed) in the database.
    
    Args:
        session_id: Unique identifier for the session
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        update_sql = """
        UPDATE customer_sessions
        SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s;
        """

        cur.execute(update_sql, (session_id,))
        conn.commit()
        cur.close()
        conn.close()

        return True
    except Exception as e:
        print(f"[✗] Error closing session: {e}")
        return False


def get_session_count_for_customer(customer_id: str) -> int:
    """
    Get total number of sessions for a customer.
    
    Args:
        customer_id: Unique identifier for the customer
    
    Returns:
        Number of sessions for this customer
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        query_sql = """
        SELECT COUNT(*) FROM customer_sessions
        WHERE customer_id = %s;
        """

        cur.execute(query_sql, (customer_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        return result[0] if result else 0
    except Exception as e:
        print(f"[✗] Error getting session count: {e}")
        return 0


def delete_old_sessions_for_customer(
    customer_id: str,
    keep_count: int = 5,
) -> bool:
    """
    Delete sessions older than the most recent keep_count sessions for a customer.
    Implements the 5-session limit per customer.
    
    Args:
        customer_id: Unique identifier for the customer
        keep_count: Number of most recent sessions to keep (default: 5)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()

        # Delete sessions that are not in the top keep_count most recent
        delete_sql = """
        DELETE FROM customer_sessions
        WHERE customer_id = %s
        AND id NOT IN (
            SELECT id FROM customer_sessions
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
        );
        """

        cur.execute(delete_sql, (customer_id, customer_id, keep_count))
        conn.commit()
        deleted_count = cur.rowcount
        cur.close()
        conn.close()

        if deleted_count > 0:
            print(f"[✓] Deleted {deleted_count} old session(s) for customer {customer_id}")

        return True
    except Exception as e:
        print(f"[✗] Error deleting old sessions: {e}")
        return False
