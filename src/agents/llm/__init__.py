"""Shared LLM provider package."""

from src.agents.llm.llm_provider import get_llm, get_router_llm, get_synthesis_llm

__all__ = ["get_llm", "get_router_llm", "get_synthesis_llm"]
