"""Product Recommendation Agent — LangGraph ReAct agent with memory.

Uses ``langgraph.prebuilt.create_react_agent`` for the ReAct tool-calling
loop and ``MemorySaver`` for per-session conversation persistence.

Usage::

    from src.agents.product_agent import ProductRecommendationAgent

    agent = ProductRecommendationAgent(use_twitter_samples=False)
    print(agent.chat("Show me laptops under $500"))
    print(agent.chat("Which of those has the best rating?"))
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agents.llm import get_llm
from src.agents.product_agent.config import (
    MAX_REACT_ITERATIONS,
    PRODUCT_SCHEMA_PATH,
    USE_TWITTER_SAMPLES,
)
from src.agents.product_agent.prompts import build_system_prompt
from src.agents.product_agent.tools import get_twitter_samples, query_products


class ProductRecommendationAgent:
    """LangGraph-based ReAct agent that recommends products from the catalog.

    Parameters
    ----------
    session_id : str
        Identifier for conversation memory. Each unique session_id
        maintains its own conversation history.
    use_twitter_samples : bool
        When ``True``, includes the ``get_twitter_samples`` tool and
        adds the Twitter context / SQL-priority instruction to the
        system prompt.  When ``False``, both are fully omitted.
    max_iterations : int
        Safety cap on ReAct reasoning cycles (recursion limit).
    """

    def __init__(
        self,
        session_id: str = "default",
        use_twitter_samples: bool = USE_TWITTER_SAMPLES,
        max_iterations: int = MAX_REACT_ITERATIONS,
        debug: bool = False,
    ) -> None:
        self.session_id = session_id
        self.use_twitter_samples = use_twitter_samples
        self.max_iterations = max_iterations
        self.debug = debug

        # Load schema
        self.schema = self._load_schema()

        # Build tool list — conditionally include twitter tool
        self._tools = [query_products]
        if self.use_twitter_samples:
            self._tools.append(get_twitter_samples)

        # Build system prompt — conditionally include twitter sections
        self._system_prompt = build_system_prompt(
            use_twitter=self.use_twitter_samples,
            schema=self.schema,
        )

        # Checkpointer for conversation memory
        self._checkpointer = MemorySaver()

        # Create the LangGraph ReAct agent
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

    # ── Public API ────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Conversation history is automatically maintained by the
        LangGraph checkpointer, keyed by ``session_id``.

        Parameters
        ----------
        user_message : str
            The user's natural-language query.

        Returns
        -------
        str
            The agent's final answer or a clarification question.
        """
        config = {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": self.max_iterations * 2,
        }

        if self.debug:
            print(f"\\n{'='*20} DEBUG: Agent Execution Started {'='*20}")
            final_content = "I wasn't able to generate a response. Please try again."
            for event in self._agent.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="values",
            ):
                msgs = event.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    last_msg.pretty_print()
                    final_content = last_msg.content
            print(f"{'='*20} DEBUG: Agent Execution Finished {'='*20}\\n")
            return final_content
        else:
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
            )

            # Extract the last AI message from the result
            messages = result.get("messages", [])
            if messages:
                return messages[-1].content
            return "I wasn't able to generate a response. Please try again."

    def reset_memory(self) -> None:
        """Clear conversation history for this session.

        Creates a fresh checkpointer, effectively resetting all
        stored conversation state.
        """
        self._checkpointer = MemorySaver()
        self._agent = create_react_agent(
            model=get_llm(),
            tools=self._tools,
            prompt=self._system_prompt,
            checkpointer=self._checkpointer,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_schema() -> dict[str, Any]:
        """Load the product catalog JSON schema from disk."""
        if not PRODUCT_SCHEMA_PATH.exists():
            return {"columns": [], "_note": "Schema file not found"}
        with open(PRODUCT_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
