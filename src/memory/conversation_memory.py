"""Conversation memory management."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """A single utterance in a conversation thread."""

    role: str
    text: str
    metadata: dict[str, object] | None = None


@dataclass
class ConversationMemory:
    """Stores recent turns and produces compact context for downstream prompting."""

    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(
        self, role: str, text: str, metadata: dict[str, object] | None = None
    ) -> None:
        """Append one turn to the session history."""

        if not text or not text.strip():
            return
        self.turns.append(
            ConversationTurn(role=role, text=text.strip(), metadata=metadata or {})
        )

    def get_recent_turns(self, limit: int = 6) -> list[ConversationTurn]:
        """Return the latest turns for prompt assembly."""

        if limit <= 0:
            return []
        return self.turns[-limit:]

    def build_context_window(self, limit: int = 6) -> str:
        """Render a compact text history for contextual understanding."""

        recent_turns = self.get_recent_turns(limit=limit)
        return "\n".join(f"{turn.role}: {turn.text}" for turn in recent_turns)

    def clear(self) -> None:
        """Reset session memory."""

        self.turns.clear()
