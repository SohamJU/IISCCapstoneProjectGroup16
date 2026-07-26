"""Local CLI for end-to-end agent orchestration testing."""

import os
import sys

# Ensure the project root is on sys.path so `src` is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

# Previously absent, so running `python main.py` without an externally
# exported GROQ_API_KEY dropped the whole system into deterministic mode.
load_dotenv()

from src.agents.orchestrator import SupportOrchestrator


def main() -> None:
    """Run a local CLI loop for end-to-end agent orchestration testing."""
    orchestrator = SupportOrchestrator()
    session_id = os.getenv("SUPPORT_SESSION_ID", "local-cli")
    customer_id = os.getenv("SUPPORT_CUSTOMER_ID") or None

    print("Agentic Customer Support CLI")
    if orchestrator.is_degraded:
        print(f"!! DEGRADED MODE: {orchestrator.degraded_reason}")
        print("!! Replies are canned fallbacks, not real agent output.")
    if customer_id:
        print(f"Customer: {customer_id}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = orchestrator.handle(
            user_input, session_id=session_id, customer_id=customer_id
        )

        routes_text = ", ".join(result.routes) if result.routes else result.route
        trace = f"[routes={routes_text} confidence={result.confidence:.2f}"
        if result.tools_used:
            trace += f" tools={','.join(dict.fromkeys(result.tools_used))}"
        trace += "]"

        print(f"{trace}\nAssistant: {result.response}\n")


if __name__ == "__main__":
    main()
