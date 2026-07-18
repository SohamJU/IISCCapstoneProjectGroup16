import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.order_agent.agent import OrderAgent

agent = OrderAgent(debug=False)
print(agent.chat("Can you help me track my order #ORD-000001?"))