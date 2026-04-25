from contracts import ObservationSchema, ActionSchema, AGENT_IT_STRATEGIC
from agents.base_agent import BaseAgent


class ITStrategicAgent(BaseAgent):
    """
    Handles batch/routine tickets — priority 2 and 3.
    Focus: throughput, efficiency, batch processing.
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_IT_STRATEGIC, policy=None)
        self.batch_queue = []

    def act(self, obs: ObservationSchema) -> ActionSchema:
        normal = [t for t in obs.tickets if t.priority >= 2 and not t.resolved]

        if not normal:
            if obs.inbox:
                return ActionSchema(
                    message_to="manager_agent",
                    message_content=(
                        "STRATEGIC: Normal queue empty. Available for reallocation."
                    ),
                    reasoning="Queue empty, awaiting manager instructions",
                )
            return ActionSchema(
                tool_call="get_tickets",
                tool_params={"priority_filter": 2},
                reasoning="Scanning for normal priority tickets",
            )

        target = min(normal, key=lambda t: (t.priority, t.sla_steps_remaining))
        return ActionSchema(
            tool_call="resolve_ticket",
            tool_params={
                "ticket_id": target.id,
                "resolution_note": f"STRATEGIC: Batch resolution of {target.id}",
            },
            reasoning=f"Batch processing P{target.priority} ticket {target.id}",
        )
