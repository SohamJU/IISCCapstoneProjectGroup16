import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.product_agent.agent import ProductRecommendationAgent

agent = ProductRecommendationAgent(debug=True)
print(agent.chat("Can you recommend a good laptop for programming under $1000?"))