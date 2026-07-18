import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.return_agent.agent import ReturnAgent

agent = ReturnAgent(debug=True)
print(agent.chat("can you return my order #ORD-000001?"))