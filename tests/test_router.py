from src.agents.router_agent import RouterAgent


def test_routes_product_queries_to_product_agent() -> None:
    router = RouterAgent()

    decision = router.route("Show me laptops under $500 with good reviews")

    assert decision.target_agent == "product"
    assert decision.confidence >= 0.6
    assert "product" in decision.reason.lower()


def test_routes_order_queries_to_order_agent() -> None:
    router = RouterAgent()

    decision = router.route("Where is my order and can you track my shipment?")

    assert decision.target_agent == "order"
    assert decision.confidence >= 0.6


def test_routes_return_queries_to_return_agent() -> None:
    router = RouterAgent()

    decision = router.route("I need a refund for my broken phone and a replacement")

    assert decision.target_agent == "return"
    assert decision.confidence >= 0.6
