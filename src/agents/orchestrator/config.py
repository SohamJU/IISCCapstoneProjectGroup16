"""Configuration constants for SupportOrchestrator."""

from __future__ import annotations

# Re-exported from the graph, where the threshold is actually applied. Kept
# here so existing `from src.agents.orchestrator import
# LOW_CONFIDENCE_THRESHOLD` imports keep working.
from src.agents.graph.nodes import LOW_CONFIDENCE_THRESHOLD

__all__ = ["LOW_CONFIDENCE_THRESHOLD"]
