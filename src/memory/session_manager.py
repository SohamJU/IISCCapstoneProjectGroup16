"""Session lifecycle management."""

from __future__ import annotations

from src.memory.conversation_memory import ConversationMemory


class SessionManager:
    """Maintains per-session conversation memory objects."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Fetch an existing memory object or create a new one."""

        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id=session_id)
        return self._sessions[session_id]

    def append_turn(
        self,
        session_id: str,
        *,
        role: str,
        text: str,
        metadata: dict | None = None,
    ) -> ConversationMemory:
        """Append a turn to a session and return the session memory."""

        session = self.get_or_create_session(session_id)
        session.add_turn(role=role, text=text, metadata=metadata)
        return session

    def clear_session(self, session_id: str) -> None:
        """Remove a session from memory."""

        self._sessions.pop(session_id, None)
