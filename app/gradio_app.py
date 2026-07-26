"""Gradio interface for local end-to-end agent orchestration testing."""

from __future__ import annotations

import inspect
import os
import sys
from uuid import uuid4

import gradio as gr

# Ensure the project root is on sys.path so `src` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.agents.orchestrator import SupportOrchestrator
from src.data.postgresql import execute_sql_query

ORCHESTRATOR = SupportOrchestrator()


def get_all_customers() -> list[tuple[str, str]]:
    """Fetch all customer names and IDs for the dropdown."""
    try:
        results = execute_sql_query(
            "SELECT customer_id, first_name, last_name FROM customers "
            "ORDER BY first_name, last_name LIMIT 500;"
        )
        if isinstance(results, list):
            return [
                (
                    f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
                    or str(row["customer_id"]),
                    str(row["customer_id"]),
                )
                for row in results
                if row.get("customer_id")
            ]
    except Exception as exc:
        print(f"Error fetching customers from database: {exc}")
    return [("Guest User", "guest-user-1")]


def _new_session_id(customer_id: str | None) -> str:
    """Mint a fresh session ID scoped to a customer.

    The session box used to hold a fixed "gradio-session-1" while the customer
    dropdown changed underneath it, so two customers shared one conversation
    thread. The orchestrator now namespaces its graph thread by customer as
    well, but a visibly fresh ID keeps the UI honest about what is happening.
    """
    who = (customer_id or "guest").strip()[:12] or "guest"
    return f"{who}-{uuid4().hex[:6]}"


def _on_customer_change(customer_id: str) -> tuple[str, list]:
    """Reset the session ID and clear the transcript when the customer changes."""
    return _new_session_id(customer_id), []


def _format_trace(result) -> str:
    """Render the routing trace shown beneath each reply.

    The old UI printed ``[routes=... confidence=0.70]`` on every message. That
    0.70 was a hardcoded constant, so it conveyed nothing. The router now
    reports genuine confidence, and the tools each agent actually called are
    the most useful signal for debugging a bad answer during a demo.
    """
    parts = [f"**Route:** `{', '.join(result.routes) or result.route}`"]
    parts.append(f"**Confidence:** `{result.confidence:.2f}`")

    if result.tools_used:
        # Preserve call order but collapse consecutive repeats.
        seen: list[str] = []
        for tool in result.tools_used:
            if not seen or seen[-1] != tool:
                seen.append(tool)
        parts.append(f"**Tools:** `{', '.join(seen)}`")

    if result.warnings:
        parts.append(f"**Warnings:** {'; '.join(result.warnings)}")

    return " · ".join(parts)


def _respond(
    message: str,
    history: list[dict[str, str]] | None,
    session_id: str,
    customer_id: str,
    show_trace: bool,
) -> str:
    """Handle one chat turn using the orchestrator."""
    result = ORCHESTRATOR.handle(
        message,
        session_id=session_id or "gradio-default",
        customer_id=customer_id or None,
    )

    if not show_trace:
        return result.response

    return f"{result.response}\n\n---\n<sub>{_format_trace(result)}</sub>"


def build_app() -> gr.Blocks:
    """Build and return the Gradio app."""
    with gr.Blocks(title="Agentic Customer Support") as demo:
        gr.Markdown("# Agentic Customer Support")
        gr.Markdown(
            "A LangGraph supervisor routes each message to the Product, Order, "
            "Return, Recommendation, Escalation or Fallback specialist, then "
            "merges their answers into a single reply."
        )

        # Make degraded mode impossible to miss. Previously a bad API key
        # silently swapped every agent for canned strings with no UI signal.
        if ORCHESTRATOR.is_degraded:
            gr.Markdown(
                f"> ⚠️ **DEGRADED MODE** — {ORCHESTRATOR.degraded_reason}\n\n"
                "> Replies are canned fallbacks, not real agent output. "
                "Check `GROQ_API_KEY` and the database connection."
            )

        with gr.Row():
            customer_choices = get_all_customers()
            first_customer = (
                customer_choices[0][1] if customer_choices else "guest-user-1"
            )
            customer_id = gr.Dropdown(
                label="Customer",
                choices=customer_choices,
                value=first_customer,
                allow_custom_value=True,
                info="Selecting a customer lets agents use their real orders and history.",
            )
            session_id = gr.Textbox(
                value=_new_session_id(first_customer),
                label="Session ID",
                info="Auto-resets when you switch customer. Reuse an ID to continue a conversation.",
            )
            show_trace = gr.Checkbox(
                value=True,
                label="Show routing trace",
                info="Displays route, confidence and tools used.",
            )

        with gr.Row():
            new_session_btn = gr.Button("Start new session", size="sm")

        chat_kwargs: dict[str, object] = {
            "fn": _respond,
            "additional_inputs": [session_id, customer_id, show_trace],
            "title": "Support Assistant",
            "description": "Ask about products, orders, returns, recommendations, or escalation.",
            "examples": [
                ["Show me wireless headphones under $250"],
                ["What is your return policy?"],
                ["Where is my order ORD-000123?"],
                ["I want to return an item and order a replacement"],
                ["I need to speak to a human"],
            ],
        }
        if "type" in inspect.signature(gr.ChatInterface.__init__).parameters:
            chat_kwargs["type"] = "messages"

        chat = gr.ChatInterface(**chat_kwargs)

        # Switching customer must start a clean conversation: new session ID and
        # an empty transcript. Without this the visible history belonged to the
        # previous customer even though the agents had already moved on.
        transcript = getattr(chat, "chatbot", None)
        outputs = [session_id] + ([transcript] if transcript is not None else [])

        def _switch(customer: str):
            new_id, cleared = _on_customer_change(customer)
            return (new_id, cleared) if transcript is not None else new_id

        customer_id.change(fn=_switch, inputs=[customer_id], outputs=outputs)
        new_session_btn.click(fn=_switch, inputs=[customer_id], outputs=outputs)

    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
