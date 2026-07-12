from src.memory.conversation_memory import ConversationMemory
from src.memory.session_manager import SessionManager


def test_conversation_memory_builds_recent_context() -> None:
    memory = ConversationMemory(session_id="session-1")

    memory.add_turn("customer", "My order has not arrived.")
    memory.add_turn("support", "Can you share your order number?")
    memory.add_turn("customer", "It is ORD-123.")

    context = memory.build_context_window(limit=2)

    assert "support: Can you share your order number?" in context
    assert "customer: It is ORD-123." in context
    assert "My order has not arrived." not in context


def test_session_manager_reuses_existing_session() -> None:
    manager = SessionManager()

    first_session = manager.append_turn(
        "abc",
        role="customer",
        text="Need help with my refund.",
    )
    second_session = manager.get_or_create_session("abc")

    assert first_session is second_session
    assert len(second_session.turns) == 1
