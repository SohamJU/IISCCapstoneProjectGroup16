"""Coordinates agent execution across specialist agents.

This module is designed to be imported from the project root. Avoid running
it in a non-interactive debug console if you expect input() prompts; prefer an
integrated terminal or direct shell execution.

Debug checklist:
- Confirm the environment is interactive (`python src/agents/orchestrator.py`)
- If running from VS Code, choose "Integrated Terminal" for console input
- If the agent appears stuck at "terminal_selection", it is likely waiting for
  input() in a non-interactive session.
- If the agent appears stuck, enable `debug=True` on Orchestrator and inspect
  the printed routing decision, target agent, and selected response
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.agents.order_agent import OrderAgent
from src.agents.return_agent import ReturnAgent
from src.agents.router_agent import RouteDecision, RouterAgent

try:
    from src.agents.product_agent import ProductRecommendationAgent
except ImportError as exc:  # pragma: no cover - environment dependent
    ProductRecommendationAgent = None  # type: ignore[assignment]
    _PRODUCT_AGENT_IMPORT_ERROR = exc


class _FallbackProductAgent:
    """Fallback product specialist used when the real product agent cannot load."""

    name = "product"

    def chat(self, user_message: str) -> str:
        return (
            "Product recommendations are temporarily unavailable. "
            "Please try again later or ask a different question."
        )


class EscalationAgent:
    """Fallback escalation agent for unknown or unsupported requests."""

    name = "escalation"

    def chat(self, user_message: str) -> str:
        return (
            "I’m forwarding this to a human-level specialist because your request "
            "needs extra attention. Please hold while I connect you with the right team."
        )


@dataclass(frozen=True)
class ChatResult:
    """Structured result returned from the orchestrator."""

    target_agent: str
    confidence: float
    reason: str
    answer: str
    follow_up: Optional[str]
    metadata: dict[str, Any]


class Orchestrator:
    """Route user requests to the right specialist agent and return a combined answer."""

    def __init__(
        self,
        session_id: str = "default",
        default_agent: str = "product",
        use_llm_fallback: bool = True,
        debug: bool = False,
    ) -> None:
        self.router = RouterAgent(
            session_id=session_id,
            default_agent=default_agent,
            use_llm_fallback=use_llm_fallback,
            debug=debug,
        )
        self.debug = debug
        self._agents: dict[str, Any] = {
            "order": OrderAgent(),
            "return": ReturnAgent(),
            "escalation": EscalationAgent(),
        }

        if ProductRecommendationAgent is not None:
            self._agents["product"] = ProductRecommendationAgent(session_id=session_id)
        else:
            self._agents["product"] = _FallbackProductAgent()
            if self.debug:
                print(
                    "[Orchestrator] ProductRecommendationAgent import failed:",
                    _PRODUCT_AGENT_IMPORT_ERROR,
                )

    def chat(self, user_message: str) -> ChatResult:
        """Route the user message, invoke the selected agent, and return the answer."""
        if self.debug:
            print("[Orchestrator] Starting chat flow")
            print(f"[Orchestrator] User input: {user_message!r}")

        decision = self.router.route(user_message)
        agent_name = decision.target_agent
        agent = self._agents.get(agent_name, self._agents["escalation"])

        if self.debug:
            print(f"[Orchestrator] Routed to {agent_name} with confidence {decision.confidence}")
            print(f"[Orchestrator] Routing reason: {decision.reason}")
            print(f"[Orchestrator] Selected agent class: {agent.__class__.__name__}")

        try:
            answer = agent.chat(user_message)
        except Exception as exc:
            if self.debug:
                print(f"[Orchestrator] Agent chat failed: {exc}")
            answer = (
                "Sorry, something went wrong while processing your request. "
                "Please try again or provide more details."
            )
            agent_name = "escalation"

        follow_up = self._build_follow_up(user_message, decision, answer)

        if self.debug:
            print(f"[Orchestrator] Answer: {answer!r}")
            print(f"[Orchestrator] Follow-up: {follow_up!r}")

        return ChatResult(
            target_agent=agent_name,
            confidence=decision.confidence,
            reason=decision.reason,
            answer=answer,
            follow_up=follow_up,
            metadata=decision.metadata,
        )

    def _build_follow_up(
        self,
        user_message: str,
        decision: RouteDecision,
        answer: str,
    ) -> Optional[str]:
        """Create a gentle follow-up question when the response can be improved."""
        if decision.confidence < 0.5:
            return (
                "I want to make sure I understood you correctly. "
                "Can you provide a bit more detail about what you need?"
            )

        if decision.target_agent == "product":
            return (
                "Would you like me to narrow this recommendation by price, brand, or use case?"
            )

        if decision.target_agent == "order":
            return (
                "Can you share your order ID or shipment details so I can help track it?"
            )

        if decision.target_agent == "return":
            return (
                "Do you want help with a refund, replacement, cancellation, or return policy?"
            )

        return None


def main() -> None:
    """Run a simple interactive chat loop using the orchestrator."""
    orchestrator = Orchestrator(debug=False)
    print("Welcome to the customer support assistant. Ask me anything about products, orders, or returns.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "Interactive mode requires a terminal. "
            "Please run this script from an integrated terminal or shell, not from a non-interactive console."
        )
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Goodbye. If you need more help, run this command again in an interactive terminal.")
            break

        if not user_input or user_input.lower() in {"exit", "quit", "bye"}:
            print("Assistant: Thanks for chatting. Have a great day!")
            break

        result = orchestrator.chat(user_input)
        print(f"\nAssistant ({result.target_agent}): {result.answer}")
        if result.follow_up:
            print(f"\nAssistant: {result.follow_up}")
        print("\n---\n")


if __name__ == "__main__":
    main()
 