"""Product Agent — stateless LangGraph ReAct agent.

Conversation state is owned by the supervisor graph
(:mod:`src.agents.graph`), not by this class. See
:class:`src.agents.base_agent.SpecialistAgent` for why.

Usage::

    from src.agents.product_agent import ProductAgent

    agent = ProductAgent()
    print(agent.chat("Show me laptops under $500"))
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

from src.agents.base_agent import SpecialistAgent
from src.agents.product_agent.config import (
    MAX_REACT_ITERATIONS,
    PRODUCT_SCHEMA_PATH,
    USE_TWITTER_SAMPLES,
)
from src.agents.product_agent.prompts import build_system_prompt
from src.agents.product_agent.tools import (
    get_product_details,
    list_product_categories,
    query_products,
    search_product_reviews,
    search_products,
)
from src.agents.shared_tools import lookup_support_policy


class ProductRecommendationAgent(SpecialistAgent):
    """ReAct agent that answers product questions from the catalog.

    Parameters
    ----------
    session_id : str
        Retained for API compatibility; the agent itself is stateless.
    use_twitter_samples : bool
        When ``True``, adds the Twitter tone-reference section to the prompt.
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
        self.use_twitter_samples = use_twitter_samples
        self.schema = self._load_schema()

        # Every bound tool's JSON schema is re-sent on EVERY ReAct step, and
        # Groq charges prompt tokens per call, so an unused tool is a recurring
        # tax. This agent's six tools cost ~1,300 tokens per step.
        tools = [
            search_products,
            get_product_details,
            list_product_categories,
            lookup_support_policy,
            query_products,
        ]

        # search_product_reviews needs Pinecone. Without a key it can only ever
        # return "not configured", so binding it burns ~200 tokens per step to
        # advertise a tool that cannot work — and tempts the model into calling
        # it and wasting a whole extra round-trip.
        if os.getenv("PINECONE_API_KEY"):
            tools.append(search_product_reviews)

        super().__init__(
            name="product",
            system_prompt=build_system_prompt(
                use_twitter=use_twitter_samples,
                schema=self.schema,
            ),
            tools=tools,
            max_iterations=max_iterations,
            session_id=session_id,
            debug=debug,
        )

    @staticmethod
    def _load_schema() -> dict[str, Any]:
        """Load the product catalog JSON schema from disk."""
        if not PRODUCT_SCHEMA_PATH.exists():
            return {"columns": [], "_note": "Schema file not found"}
        with open(PRODUCT_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            return cast(dict[str, Any], json.load(fh))


class ProductAgent(ProductRecommendationAgent):
    """Alias retaining ProductAgent naming for the current architecture."""
