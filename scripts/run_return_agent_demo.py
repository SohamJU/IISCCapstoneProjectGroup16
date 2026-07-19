"""Small demo script for ProductReturnAgent.

Usage: python scripts/run_return_agent_demo.py
"""
from src.agents.return_agent import ProductReturnAgent


def main():
    agent = ProductReturnAgent()

    print("--- Policy Summary ---")
    print(agent.summarize(400))
    print()

    print("--- Handle Query Example ---")
    print(agent.handle_query("How do I return an item?"))
    print()

    print("--- Lookup Order Example ---")
    print(agent.lookup_order("ORD-000001"))
    print()

    print("--- RAG-like Query Example ---")
    print(agent.rag_query("restocking fee"))


if __name__ == "__main__":
    main()
