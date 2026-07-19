"""
Quick Start Guide: Customer Session Persistence

This guide shows how to integrate customer session persistence into your chatbot.
"""

# STEP 1: Initialize the database table
# ======================================
# Run this once to set up the PostgreSQL table:
#
#   python -m pipelines.setup_sessions_table
#
# Or manually in Python:
#
#   from src.data.session_persistence import initialize_sessions_table
#   initialize_sessions_table()

# STEP 2: Update your Gradio App
# ===============================

"""
File: app/gradio_app.py

Modified version to include customer session persistence:
"""

from src.agents.orchestrator.agent import SupportOrchestrator
from src.memory.persistent_session_manager import PersistentSessionManager
from src.utils.customer_history import enrich_message_with_history
import gradio as gr


# Initialize orchestrator with persistent session manager
ORCHESTRATOR = SupportOrchestrator()
ORCHESTRATOR._session_manager = PersistentSessionManager(persist_to_db=True)


def _respond(
    message: str,
    history: list[dict[str, str]] | None,
    session_id: str,
    customer_id: str,
) -> str:
    """Handle one chat turn with customer session persistence."""

    # Enrich message with customer's past conversation history
    enriched_message = enrich_message_with_history(
        user_message=message,
        customer_id=customer_id,
        include_history=True,
        num_sessions=2,  # Include context from last 2 sessions
        max_turns_per_session=3,  # Show last 3 turns per session
    )

    # Get or create session
    session = ORCHESTRATOR._session_manager.get_or_create_session(
        session_id=session_id,
        customer_id=customer_id,
    )

    # Handle request through orchestrator
    result = ORCHESTRATOR.handle(enriched_message, session_id=session_id)

    # Save conversation to persistent database
    ORCHESTRATOR._session_manager.append_turn(
        session_id=session_id,
        role="user",
        text=message,
        customer_id=customer_id,
    )

    ORCHESTRATOR._session_manager.append_turn(
        session_id=session_id,
        role="assistant",
        text=result.response,
        customer_id=customer_id,
    )

    routes_text = ",".join(result.routes) if result.routes else result.route
    return f"[routes={routes_text} confidence={result.confidence:.2f}]\n{result.response}"


def build_app() -> gr.Blocks:
    """Build Gradio app with customer session persistence."""
    with gr.Blocks(title="Agentic Customer Support") as demo:
        gr.Markdown("# Agentic Customer Support with Session History")
        gr.Markdown(
            "This chatbot remembers your conversation history across sessions. "
            "Provide your customer ID to receive personalized support based on your previous interactions."
        )

        # Customer ID input
        customer_id = gr.Textbox(
            value="customer-001",
            label="Customer ID",
            info="Your unique customer identifier (used to retrieve past conversations)",
        )

        # Session ID input
        session_id = gr.Textbox(
            value="session-001",
            label="Session ID",
            info="Current session identifier (auto-generated if needed)",
        )

        # Chat interface
        chat_kwargs = {
            "fn": _respond,
            "additional_inputs": [session_id, customer_id],
            "title": "Support Assistant",
            "description": "Ask about products, orders, returns, recommendations, or escalation. "
                          "Your conversation is saved for future reference.",
        }
        if "type" in gr.ChatInterface.__init__.__code__.co_varnames:
            chat_kwargs["type"] = "messages"

        gr.ChatInterface(**chat_kwargs)

        gr.Markdown(
            """
            ---
            ### How It Works
            - **Customer ID**: Identifies you across all sessions
            - **Session ID**: Unique ID for this conversation
            - **History**: Your last 2 sessions are automatically included in context
            - **Persistence**: This conversation is saved to our database
            - **Session Limit**: We keep your 5 most recent sessions
            """
        )

    return demo


if __name__ == "__main__":
    build_app().launch()


# STEP 3: Retrieve Customer History
# ==================================

from src.data.session_persistence import get_customer_session_history
from src.utils.customer_history import get_customer_sentiment_summary

# Get a customer's conversation history
def show_customer_history(customer_id: str):
    """Display a customer's conversation history."""
    sessions = get_customer_session_history(customer_id, limit=5)

    print(f"\n{'='*70}")
    print(f"Customer {customer_id} - Session History")
    print(f"{'='*70}\n")

    for i, session in enumerate(sessions, 1):
        session_id = session["session_id"]
        created_at = session["created_at"]
        turns = session["conversation_turns"]

        print(f"[Session {i}] {session_id}")
        print(f"  Created: {created_at}")
        print(f"  Turns: {len(turns)}")
        print()

        for turn in turns[-2:]:  # Show last 2 turns
            role = turn["role"].upper()
            text = turn["text"][:100] + "..." if len(turn["text"]) > 100 else turn["text"]
            print(f"    {role}: {text}")
        print()

    # Show sentiment summary
    summary = get_customer_sentiment_summary(customer_id)
    print(f"Customer Summary:")
    print(f"  - Total Sessions: {summary['total_sessions']}")
    print(f"  - Total Interactions: {summary['total_interactions']}")
    print(f"  - Has Issues: {summary['has_issues']}")
    print(f"  - Topics: {', '.join(summary['topics']) if summary['topics'] else 'None'}")
    print()


# Example usage:
# show_customer_history("customer-001")


# STEP 4: Close Sessions
# ======================

from src.memory.persistent_session_manager import PersistentSessionManager

def end_customer_session(session_id: str, customer_id: str):
    """End a session and clean up old sessions."""
    mgr = PersistentSessionManager(persist_to_db=True)
    mgr.close_session(session_id, customer_id)
    print(f"✓ Session {session_id} closed")
    print(f"✓ Cleaned up old sessions, keeping 5 most recent")


# Example usage:
# end_customer_session("session-001", "customer-001")


# STEP 5: Manual Session Management
# ==================================

from src.memory.persistent_session_manager import PersistentSessionManager

def manual_session_example():
    """Example of manual session management."""
    mgr = PersistentSessionManager(persist_to_db=True)

    customer_id = "customer-123"
    session_id = "session-2024-01-15-001"

    # Get or create session
    session = mgr.get_or_create_session(session_id, customer_id)

    # Add turns
    mgr.append_turn(
        session_id=session_id,
        role="user",
        text="Can I return my order?",
        customer_id=customer_id,
    )

    mgr.append_turn(
        session_id=session_id,
        role="assistant",
        text="Yes, we accept returns within 30 days of purchase.",
        customer_id=customer_id,
    )

    # Retrieve customer's history
    history = mgr.get_customer_history(customer_id, limit=5)
    print(f"Customer {customer_id} has {len(history)} recent session(s)")

    # Get last session
    last_session = mgr.get_last_session_for_customer(customer_id)
    if last_session:
        print(f"Last session: {last_session['session_id']}")
        print(f"  Created: {last_session['created_at']}")
        print(f"  Turns: {len(last_session['conversation_turns'])}")

    # Close session
    mgr.close_session(session_id, customer_id)
    print(f"Session closed and cleaned up")


# Example usage:
# manual_session_example()


# TROUBLESHOOTING
# ===============

"""
1. Database Connection Error
   Error: "could not connect to server"
   Solution:
   - Check POSTGRESQL_* environment variables in .env
   - Verify Aiven instance is running
   - Test connection: python -c "from src.data.session_persistence import initialize_sessions_table; initialize_sessions_table()"

2. Sessions Not Saving
   Solution:
   - Ensure PersistentSessionManager is initialized with persist_to_db=True
   - Verify customer_id is provided when appending turns
   - Check that table was created: initialize_sessions_table()

3. Session History Empty
   Solution:
   - Verify customer_id matches between save and retrieval
   - Check that sessions were saved to correct customer_id
   - Run: from src.data.session_persistence import get_customer_session_history
         get_customer_session_history("customer-id")

4. Need to Debug
   Solution:
   - Run test suite: pytest tests/test_session_persistence.py -v
   - Check manual tests: python tests/test_session_persistence.py
   - Enable logging in orchestrator
"""

# KEY CONCEPTS
# ============

"""
Customer ID vs Session ID:
- customer_id: Identifies a PERSON (persistent across all their sessions)
- session_id: Identifies a CONVERSATION (unique for each chat session)

Session Persistence:
- Automatic: Appending turns with customer_id saves to database
- Manual: Use save_session_to_db() for fine-grained control
- Retrieval: Use get_customer_history() to fetch recent sessions

Session Limits:
- Default: Keep 5 most recent sessions per customer
- Automatic cleanup: delete_old_sessions_for_customer() on session close
- Override: Pass keep_count parameter to customize limit

History Context:
- Automatic: enrich_message_with_history() adds context to LLM
- Manual: Use format_customer_history_context() to format history
- Topics: get_customer_topics_from_history() extracts key topics
- Sentiment: get_customer_sentiment_summary() analyzes interactions
"""
