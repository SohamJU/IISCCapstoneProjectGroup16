"""Enhanced session manager with PostgreSQL persistence for customer conversations."""

from __future__ import annotations

from typing import Optional

from src.memory.conversation_memory import ConversationMemory
from src.memory.session_manager import SessionManager as InMemorySessionManager
from src.data.session_persistence import (
    save_session_to_db,
    load_session_from_db,
    get_customer_session_history,
    close_session,
    delete_old_sessions_for_customer,
)


class PersistentSessionManager(InMemorySessionManager):
    """
    Enhanced SessionManager that persists conversations to PostgreSQL.
    
    This extends the in-memory SessionManager to automatically save sessions
    to the database, supporting customer history retrieval and session limits.
    """

    def __init__(self, persist_to_db: bool = True) -> None:
        """
        Initialize the persistent session manager.
        
        Args:
            persist_to_db: Whether to persist sessions to PostgreSQL (default: True)
        """
        super().__init__()
        self.persist_to_db = persist_to_db

    def get_or_create_session(self, session_id: str, customer_id: str | None = None) -> ConversationMemory:
        """
        Fetch an existing session or create a new one.
        
        If persist_to_db is True, attempts to load from database first,
        otherwise creates new in-memory session.
        
        Args:
            session_id: Unique identifier for the session
            customer_id: (Optional) Customer identifier for persistence
        
        Returns:
            ConversationMemory object for the session
        """
        # Check in-memory cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try to load from database if persistence is enabled
        if self.persist_to_db and customer_id:
            db_session = load_session_from_db(session_id)
            if db_session:
                self._sessions[session_id] = db_session
                return db_session

        # Create new session
        session = ConversationMemory(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def append_turn(
        self,
        session_id: str,
        *,
        role: str,
        text: str,
        metadata: dict[str, object] | None = None,
        customer_id: str | None = None,
    ) -> ConversationMemory:
        """
        Append a turn to a session and save to database.
        
        Args:
            session_id: Session identifier
            role: Role of the speaker (e.g., "user", "assistant")
            text: Message text
            metadata: Optional metadata for the turn
            customer_id: Optional customer ID for persistence
        
        Returns:
            Updated ConversationMemory object
        """
        session = super().append_turn(session_id, role=role, text=text, metadata=metadata)

        # Persist to database if enabled and customer_id provided
        if self.persist_to_db and customer_id:
            save_session_to_db(customer_id, session_id, session)

        return session

    def close_session(self, session_id: str, customer_id: str | None = None) -> None:
        """
        Close a session and mark it as inactive in the database.
        
        Args:
            session_id: Session identifier
            customer_id: Optional customer ID for cleanup
        """
        super().clear_session(session_id)

        # Mark as inactive in database if enabled
        if self.persist_to_db:
            close_session(session_id)

        # Clean up old sessions if customer_id provided
        if customer_id:
            delete_old_sessions_for_customer(customer_id, keep_count=5)

    def get_customer_history(
        self,
        customer_id: str,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """
        Retrieve recent conversation sessions for a customer.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of sessions to retrieve (default: 5)
        
        Returns:
            List of session summaries with conversation_turns, created_at, etc.
        """
        if not self.persist_to_db:
            return []

        return get_customer_session_history(customer_id, limit=limit)

    def get_last_session_for_customer(self, customer_id: str) -> Optional[dict[str, object]]:
        """
        Retrieve the most recent session for a customer.
        
        Args:
            customer_id: Customer identifier
        
        Returns:
            Most recent session summary or None if no sessions exist
        """
        sessions = self.get_customer_history(customer_id, limit=1)
        return sessions[0] if sessions else None
