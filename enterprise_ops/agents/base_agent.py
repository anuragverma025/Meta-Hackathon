from contracts import ObservationSchema, ActionSchema


class BaseAgent:
    def __init__(self, agent_id: str, policy):
        self.agent_id = agent_id
        self.policy = policy

    def act(self, obs: ObservationSchema) -> ActionSchema:
        decision = self.policy.decide(obs, self.agent_id)

        # Ensure safe defaults
        if decision is None:
            decision = {}

        return ActionSchema(
            tool_call=decision.get("tool_call"),
            tool_params=decision.get("tool_params", {}),
            message_to=decision.get("message_to"),
            message_content=decision.get("message_content"),
            reasoning=decision.get("reasoning"),
        )