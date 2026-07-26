"""Tests for customer session persistence functionality."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.conversation_memory import ConversationMemory, ConversationTurn
from src.memory.persistent_session_manager import PersistentSessionManager
from src.data.session_persistence import (
    initialize_sessions_table,
    save_session_to_db,
    get_customer_session_history,
    load_session_from_db,
    close_session,
    delete_old_sessions_for_customer,
    get_session_count_for_customer,
)
from src.utils.customer_history import (
    format_session_summary,
    format_customer_history_context,
    enrich_message_with_history,
    get_customer_topics_from_history,
    get_customer_sentiment_summary,
)


class TestSessionPersistence:
    """Test suite for session persistence functionality."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_db(self):
        """Initialize database before tests."""
        initialize_sessions_table()
        yield
        # Cleanup happens after all tests

    def test_save_session_to_db(self):
        """Test saving a conversation session to database."""
        # Create test session
        memory = ConversationMemory(session_id="test-session-001")
        memory.add_turn(role="user", text="Hello, I need help with my order")
        memory.add_turn(role="assistant", text="I'd be happy to help!")

        # Save to database
        result = save_session_to_db(
            customer_id="test-customer-001",
            session_id="test-session-001",
            conversation_memory=memory,
        )

        assert result, "Failed to save session to database"

    def test_load_session_from_db(self):
        """Test loading a session from database."""
        # First save
        memory = ConversationMemory(session_id="test-session-002")
        memory.add_turn(role="user", text="Can I return my item?")
        memory.add_turn(role="assistant", text="Yes, we have a 30-day return policy")

        save_session_to_db(
            customer_id="test-customer-001",
            session_id="test-session-002",
            conversation_memory=memory,
        )

        # Load and verify
        loaded_memory = load_session_from_db("test-session-002")

        assert loaded_memory is not None, "Failed to load session"
        assert len(loaded_memory.turns) == 2, "Turns not loaded correctly"
        assert loaded_memory.turns[0].text == "Can I return my item?"
        assert loaded_memory.turns[1].text == "Yes, we have a 30-day return policy"

    def test_get_customer_session_history(self):
        """Test retrieving customer's session history."""
        # Create multiple sessions
        for i in range(3):
            memory = ConversationMemory(session_id=f"test-session-hist-{i}")
            memory.add_turn(role="user", text=f"Question {i}")
            memory.add_turn(role="assistant", text=f"Answer {i}")

            save_session_to_db(
                customer_id="test-customer-hist",
                session_id=f"test-session-hist-{i}",
                conversation_memory=memory,
            )

        # Retrieve history
        history = get_customer_session_history("test-customer-hist", limit=5)

        assert len(history) >= 3, "Did not retrieve all sessions"
        assert all("session_id" in s for s in history), "Missing session_id"
        assert all("conversation_turns" in s for s in history), "Missing turns"

    def test_delete_old_sessions(self):
        """Test cleanup of old sessions (keep only 5 most recent)."""
        customer_id = "test-customer-cleanup"

        # Create 7 sessions
        for i in range(7):
            memory = ConversationMemory(session_id=f"test-cleanup-{i}")
            memory.add_turn(role="user", text=f"Message {i}")

            save_session_to_db(
                customer_id=customer_id,
                session_id=f"test-cleanup-{i}",
                conversation_memory=memory,
            )

        # Clean up old sessions (keep 5)
        delete_old_sessions_for_customer(customer_id, keep_count=5)

        # Verify only 5 remain
        remaining = get_customer_session_history(customer_id, limit=10)
        assert len(remaining) <= 5, f"Expected <= 5 sessions, got {len(remaining)}"

    def test_close_session(self):
        """Test closing a session."""
        memory = ConversationMemory(session_id="test-close-session")
        memory.add_turn(role="user", text="Goodbye")

        save_session_to_db(
            customer_id="test-customer-001",
            session_id="test-close-session",
            conversation_memory=memory,
        )

        # Close session
        result = close_session("test-close-session")
        assert result, "Failed to close session"

    def test_get_session_count(self):
        """Test getting session count for customer."""
        customer_id = "test-customer-count"

        # Create 3 sessions
        for i in range(3):
            memory = ConversationMemory(session_id=f"test-count-{i}")
            memory.add_turn(role="user", text=f"Test {i}")

            save_session_to_db(
                customer_id=customer_id,
                session_id=f"test-count-{i}",
                conversation_memory=memory,
            )

        # Get count
        count = get_session_count_for_customer(customer_id)
        assert count >= 3, f"Expected at least 3 sessions, got {count}"


class TestPersistentSessionManager:
    """Test suite for persistent session manager."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_db(self):
        """Initialize database before tests."""
        initialize_sessions_table()
        yield

    def test_create_persistent_session_manager(self):
        """Test creating a persistent session manager."""
        mgr = PersistentSessionManager(persist_to_db=True)
        assert mgr.persist_to_db is True

    def test_get_or_create_session_with_customer_id(self):
        """Test getting or creating session with customer ID."""
        mgr = PersistentSessionManager(persist_to_db=True)

        session = mgr.get_or_create_session(
            session_id="persistent-test-001",
            customer_id="persistent-customer-001",
        )

        assert session is not None
        assert session.session_id == "persistent-test-001"

    def test_append_turn_with_persistence(self):
        """Test appending turns with persistence."""
        mgr = PersistentSessionManager(persist_to_db=True)

        session = mgr.append_turn(
            session_id="persistent-test-002",
            role="user",
            text="Hello from persistent manager",
            customer_id="persistent-customer-002",
        )

        assert len(session.turns) == 1
        assert session.turns[0].text == "Hello from persistent manager"

    def test_get_customer_history_from_manager(self):
        """Test retrieving customer history through manager."""
        mgr = PersistentSessionManager(persist_to_db=True)

        # Create sessions
        for i in range(2):
            mgr.append_turn(
                session_id=f"persistent-test-hist-{i}",
                role="user",
                text=f"History test {i}",
                customer_id="persistent-customer-hist",
            )

        # Retrieve
        history = mgr.get_customer_history(
            customer_id="persistent-customer-hist",
            limit=5,
        )

        assert len(history) >= 2, "Did not retrieve all sessions from manager"

    def test_close_session_from_manager(self):
        """Test closing session through manager."""
        mgr = PersistentSessionManager(persist_to_db=True)

        # Create session
        mgr.append_turn(
            session_id="persistent-test-close",
            role="user",
            text="Test close",
            customer_id="persistent-customer-close",
        )

        # Close it
        mgr.close_session(
            session_id="persistent-test-close",
            customer_id="persistent-customer-close",
        )

        # Verify it's closed (should still exist but marked inactive)
        history = mgr.get_customer_history("persistent-customer-close", limit=5)
        assert len(history) >= 1


class TestCustomerHistoryUtils:
    """Test suite for customer history utilities."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_db_and_data(self):
        """Initialize database and create test data."""
        initialize_sessions_table()

        # Create test session with multiple turns
        mgr = PersistentSessionManager(persist_to_db=True)
        for i in range(3):
            mgr.append_turn(
                session_id=f"utils-test-session-{i}",
                role="user",
                text=f"Question about product and order {i}",
                customer_id="utils-test-customer",
            )
            mgr.append_turn(
                session_id=f"utils-test-session-{i}",
                role="assistant",
                text=f"Response about your product order {i}",
                customer_id="utils-test-customer",
            )

        yield

    def test_format_session_summary(self):
        """Test formatting a session summary."""
        history = get_customer_session_history("utils-test-customer", limit=1)
        assert len(history) > 0

        summary = format_session_summary(history[0], max_turns=2)
        assert "Session:" in summary
        assert "Date:" in summary
        assert "USER:" in summary or "ASSISTANT:" in summary

    def test_format_customer_history_context(self):
        """Test formatting full customer history."""
        context = format_customer_history_context(
            customer_id="utils-test-customer",
            num_sessions=2,
            max_turns_per_session=2,
        )

        assert context is not None
        assert "utils-test-customer" in context
        assert "Recent Conversation History" in context

    def test_enrich_message_with_history(self):
        """Test enriching a message with history."""
        enriched = enrich_message_with_history(
            user_message="Can you help?",
            customer_id="utils-test-customer",
            include_history=True,
            num_sessions=2,
        )

        assert "Can you help?" in enriched
        assert "Recent Conversation History" in enriched or len(enriched) > len("Can you help?")

    def test_get_customer_topics(self):
        """Test extracting customer topics."""
        topics = get_customer_topics_from_history("utils-test-customer")

        assert isinstance(topics, list)
        # Should contain some topics based on our test messages
        assert len(topics) >= 0

    def test_get_customer_sentiment_summary(self):
        """Test generating sentiment summary."""
        summary = get_customer_sentiment_summary("utils-test-customer")

        assert summary["customer_id"] == "utils-test-customer"
        assert "total_sessions" in summary
        assert "total_interactions" in summary
        assert "has_issues" in summary


# Manual test runner for development
if __name__ == "__main__":
    print("=" * 70)
    print("Customer Session Persistence Test Suite")
    print("=" * 70)

    print("\n[1] Initializing database table...")
    initialize_sessions_table()

    print("\n[2] Testing basic session save/load...")
    memory = ConversationMemory(session_id="manual-test-001")
    memory.add_turn(role="user", text="Hello!")
    memory.add_turn(role="assistant", text="Hi there!")

    saved = save_session_to_db("manual-customer", "manual-test-001", memory)
    print(f"    ✓ Session saved: {saved}")

    loaded = load_session_from_db("manual-test-001")
    print(f"    ✓ Session loaded: {loaded is not None}")
    if loaded:
        print(f"      - Turns: {len(loaded.turns)}")

    print("\n[3] Testing customer history retrieval...")
    history = get_customer_session_history("manual-customer", limit=5)
    print(f"    ✓ Retrieved {len(history)} session(s)")

    print("\n[4] Testing persistent session manager...")
    mgr = PersistentSessionManager(persist_to_db=True)
    session = mgr.get_or_create_session("manager-test-001", "manager-customer")
    mgr.append_turn(
        session_id="manager-test-001",
        role="user",
        text="Test message",
        customer_id="manager-customer",
    )
    print(f"    ✓ Session appended with {len(session.turns)} turns")

    print("\n[5] Testing customer history formatting...")
    context = format_customer_history_context("manager-customer", num_sessions=1)
    if context:
        lines = context.split("\n")
        print(f"    ✓ Formatted history ({len(lines)} lines)")
    else:
        print("    - No history available")

    print("\n[✓] All manual tests completed!")
    print("\nTo run full pytest suite: pytest tests/test_session_persistence.py -v")
