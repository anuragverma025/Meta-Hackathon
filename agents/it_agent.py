from agents.base_agent import BaseAgent
from agents.policies import RulePolicy


class ITAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, RulePolicy(role="it"))