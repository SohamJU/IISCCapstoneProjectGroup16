"""Gradio interface for local end-to-end agent orchestration testing."""

from __future__ import annotations

import os
import sys
import inspect

import gradio as gr

# Ensure the project root is on sys.path so `src` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# from src.agents.orchestrator import SupportOrchestrator
from src.agents.orchestrator.agent import SupportOrchestrator
from src.data.postgresql import execute_sql_query

ORCHESTRATOR = SupportOrchestrator()


def get_all_customers() -> list[tuple[str, str]]:
	"""Fetch all customer names and IDs for the dropdown."""
	try:
		# Fetch name components and ID from the customers table
		results = execute_sql_query("SELECT customer_id, first_name, last_name FROM customers ORDER BY first_name, last_name;")
		if isinstance(results, list):
			return [
				(f"{row.get('first_name', '')} {row.get('last_name', '')}".strip() or str(row['customer_id']), str(row['customer_id']))
				for row in results if row.get("customer_id")
			]
	except Exception as e:
		print(f"Error fetching customers from database: {e}")
	return [("Gradio User 1", "gradio-user-1")]


def _respond(message: str, history: list[dict[str, str]] | None, session_id: str, customer_id: str) -> str:
	"""Handle one chat turn using the orchestrator."""
	result = ORCHESTRATOR.handle(
		message, 
		session_id=session_id or "gradio-default",
		customer_id=customer_id
	)
	routes_text = ",".join(result.routes) if result.routes else result.route
	return f"[routes={routes_text} confidence={result.confidence:.2f}]\n{result.response}"


def build_app() -> gr.Blocks:
	"""Build and return the Gradio app."""
	with gr.Blocks(title="Agentic Customer Support") as demo:
		gr.Markdown("# Agentic Customer Support")
		gr.Markdown(
			"Product, Order, Return, Recommendation, and Escalation flows routed through a central router."
		)

		with gr.Row():
			customer_choices = get_all_customers()
			customer_id = gr.Dropdown(
				label="Customer Name",
				choices=customer_choices,
				value=customer_choices[0][1] if customer_choices else "gradio-user-1",
				allow_custom_value=True,
				info="Select an existing customer or enter a new one to retrieve history.",
			)
			session_id = gr.Textbox(
				value="gradio-session-1",
				label="Session ID",
				info="Use the same session ID to retain conversation memory.",
			)

		chat_kwargs: dict[str, object] = {
			"fn": _respond,
			"additional_inputs": [session_id, customer_id],
			"title": "Support Assistant",
			"description": "Ask about products, orders, returns, recommendations, or escalation.",
		}
		if "type" in inspect.signature(gr.ChatInterface.__init__).parameters:
			chat_kwargs["type"] = "messages"

		gr.ChatInterface(**chat_kwargs)

	return demo


if __name__ == "__main__":
	build_app().launch(share=True)
