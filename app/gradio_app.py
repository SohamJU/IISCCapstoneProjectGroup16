"""Gradio interface for local end-to-end agent orchestration testing."""

from __future__ import annotations

import inspect

import gradio as gr

# from src.agents.orchestrator import SupportOrchestrator
from src.agents.orchestrator.agent import SupportOrchestrator


ORCHESTRATOR = SupportOrchestrator()


def _respond(message: str, history: list[dict[str, str]] | None, session_id: str) -> str:
	"""Handle one chat turn using the orchestrator."""
	result = ORCHESTRATOR.handle(message, session_id=session_id or "gradio-default")
	routes_text = ",".join(result.routes) if result.routes else result.route
	return f"[routes={routes_text} confidence={result.confidence:.2f}]\n{result.response}"


def build_app() -> gr.Blocks:
	"""Build and return the Gradio app."""
	with gr.Blocks(title="Agentic Customer Support") as demo:
		gr.Markdown("# Agentic Customer Support")
		gr.Markdown(
			"Product, Order, Return, Recommendation, and Escalation flows routed through a central router."
		)

		session_id = gr.Textbox(
			value="gradio-user-1",
			label="Session ID",
			info="Use the same session ID to retain conversation memory.",
		)

		chat_kwargs: dict[str, object] = {
			"fn": _respond,
			"additional_inputs": [session_id],
			"title": "Support Assistant",
			"description": "Ask about products, orders, returns, recommendations, or escalation.",
		}
		if "type" in inspect.signature(gr.ChatInterface.__init__).parameters:
			chat_kwargs["type"] = "messages"

		gr.ChatInterface(**chat_kwargs)

	return demo


if __name__ == "__main__":
	build_app().launch()
