from __future__ import annotations

from typing import Any

from src.agents.fallback_agent import FallbackAgent
from src.agents.orchestrator import LOW_CONFIDENCE_THRESHOLD, SupportOrchestrator
from src.agents.router import RouterAgent
from src.rag.pipeline import chunk_text


def test_router_keyword_order_route() -> None:
    router = RouterAgent(use_llm_fallback=False)
    assert router.route("Where is my order ORD-123?") == "order"


def test_router_keyword_return_route() -> None:
    router = RouterAgent(use_llm_fallback=False)
    assert router.route("I want a refund for a damaged item") == "return"


def test_router_keyword_escalation_route() -> None:
    router = RouterAgent(use_llm_fallback=False)
    assert router.route("I want to speak to a human manager now") == "escalation"


def test_fallback_agent_greeting_uses_humanized_response() -> None:
    agent = FallbackAgent()
    response = agent.chat("Hi")
    assert "Hi" in response
    assert "help" in response.lower()
    assert "products" in response.lower() or "orders" in response.lower()


def test_chunk_text_keeps_non_empty_chunks() -> None:
	text = " ".join(["alpha"] * 500)
	chunks = chunk_text(text, chunk_size=120, overlap=20)
	assert chunks
	assert all(len(c) > 0 for c in chunks)


class _StubAgent:
	def __init__(self, name: str) -> None:
		self.name = name
		self.last_message = ""

	def chat(self, user_message: str) -> str:
		self.last_message = user_message
		return f"{self.name}: {user_message}"


class _StubRouter:
	def __init__(
		self,
		route: str,
		confidence: float,
		*,
		routes: list[str] | None = None,
		confidences: dict[str, float] | None = None,
	) -> None:
		self.route = route
		self.confidence = confidence
		self.routes = routes or [route]
		self.confidences = confidences or {route: confidence}

	def classify(self, user_message: str) -> dict[str, Any]:
		return {
			"route": self.route,
			"confidence": self.confidence,
			"reason": "test",
		}

	def classify_multi(self, user_message: str) -> dict[str, Any]:
		return {
			"routes": self.routes,
			"confidences": self.confidences,
			"reason": "test",
		}

	def build_subtask_message(self, route: str, user_message: str) -> str:
		return f"subtask[{route}]::{user_message}"


def _stub_agents() -> dict[str, object]:
	return {
		"product": _StubAgent("product"),
		"order": _StubAgent("order"),
		"return": _StubAgent("return"),
		"recommendation": _StubAgent("recommendation"),
		"escalation": _StubAgent("escalation"),
		"fallback": _StubAgent("fallback"),
	}


def test_orchestrator_dispatches_to_selected_route() -> None:
    orchestrator = SupportOrchestrator(use_llm_router_fallback=False)
    orchestrator.router = _StubRouter(route="order", confidence=0.95)
    orchestrator._get_or_create_session_agents = lambda session_id: _stub_agents()  # type: ignore[method-assign]

    result = orchestrator.handle("Track my order", session_id="test-1")
    assert result.route == "order"
    assert "order:" in result.response


def test_orchestrator_unknown_route_falls_back() -> None:
    orchestrator = SupportOrchestrator(use_llm_router_fallback=False)
    orchestrator.router = _StubRouter(route="not-a-route", confidence=0.9)
    orchestrator._get_or_create_session_agents = lambda session_id: _stub_agents()  # type: ignore[method-assign]

    result = orchestrator.handle("Some unsupported intent", session_id="test-2")
    assert result.route == "fallback"
    assert "fallback:" in result.response


def test_orchestrator_low_confidence_escalates() -> None:
    orchestrator = SupportOrchestrator(use_llm_router_fallback=False)
    orchestrator.router = _StubRouter(route="product", confidence=LOW_CONFIDENCE_THRESHOLD - 0.01)
    orchestrator._get_or_create_session_agents = lambda session_id: _stub_agents()  # type: ignore[method-assign]

    result = orchestrator.handle("Tell me about this item", session_id="test-3")
    assert result.route == "escalation"
    assert "escalation:" in result.response


def test_router_multi_intent_keywords_detected() -> None:
    router = RouterAgent(use_llm_fallback=False)
    result = router.classify_multi("I want to return my last order and get a cheaper replacement")
    routes = result.get("routes", [])
    assert isinstance(routes, list)
    assert "return" in routes
    assert "order" in routes
    assert "recommendation" in routes


def test_orchestrator_executes_multi_intent_sequentially() -> None:
	orchestrator = SupportOrchestrator(use_llm_router_fallback=False)
	router_stub = _StubRouter(
		route="return",
		confidence=0.95,
		routes=["return", "recommendation"],
		confidences={"return": 0.95, "recommendation": 0.9},
	)
	orchestrator.router = router_stub
	agents = _stub_agents()
	orchestrator._get_or_create_session_agents = lambda session_id: agents  # type: ignore[method-assign]

	user_message = "I want to return my order and buy a cheaper alternative"
	result = orchestrator.handle(user_message, session_id="test-4")
	assert result.routes == ["return", "recommendation"]
	assert "Handled your request in 2 steps:" in result.response
	assert "Step 1 - return" in result.response
	assert "Step 2 - recommendation" in result.response

	assert isinstance(agents["return"], _StubAgent)
	assert isinstance(agents["recommendation"], _StubAgent)
	assert agents["return"].last_message == f"subtask[return]::{user_message}"
	assert agents["recommendation"].last_message == f"subtask[recommendation]::{user_message}"


def test_orchestrator_single_intent_returns_plain_agent_response() -> None:
	orchestrator = SupportOrchestrator(use_llm_router_fallback=False)
	orchestrator.router = _StubRouter(route="order", confidence=0.95)
	orchestrator._get_or_create_session_agents = lambda session_id: _stub_agents()  # type: ignore[method-assign]

	result = orchestrator.handle("Track my order", session_id="test-5")
	assert "Handled your request" not in result.response
	assert result.response.startswith("order:")


def test_orchestrator_deterministic_mode_runs_without_llm_agents() -> None:
	orchestrator = SupportOrchestrator(
		use_llm_router_fallback=False,
		deterministic_mode=True,
	)

	result = orchestrator.handle("Where is my order ORD-000123?", session_id="test-det-1")
	assert result.route == "order"
	assert "deterministic-mode" in result.response


def test_router_gracefully_disables_llm_fallback_on_init_error() -> None:
	# Simulate missing/invalid LLM by forcing init with fallback enabled;
	# Router should still be usable via keyword routing.
	router = RouterAgent(use_llm_fallback=True)
	route = router.route("I want to return my order")
	assert route == "return"

