import os
import sys

# Ensure the project root is on sys.path so `src` is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agents.orchestrator.agent import SupportOrchestrator


def main() -> None:
    """Run a local CLI loop for end-to-end agent orchestration testing."""
    orchestrator = SupportOrchestrator()
    session_id = "local-cli"

    print("Agentic Customer Support CLI")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = orchestrator.handle(user_input, session_id=session_id)
        routes_text = ",".join(result.routes) if result.routes else result.route
        print(
            f"[routes={routes_text} confidence={result.confidence:.2f}]\n"
            f"Assistant: {result.response}\n"
        )


if __name__ == "__main__":
    main()
