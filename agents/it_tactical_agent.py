from contracts import ObservationSchema, ActionSchema, AGENT_IT_TACTICAL, AGENT_MANAGER
from agents.base_agent import BaseAgent


class ITTacticalAgent(BaseAgent):
    """
    Handles urgent/critical tickets — priority 1 and SLA-critical.
    Focus: speed, SLA compliance, emergency resolution.
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_IT_TACTICAL, policy=None)

    def act(self, obs: ObservationSchema) -> ActionSchema:
        critical = [t for t in obs.tickets if t.priority == 1 and not t.resolved]
        sla_critical = [t for t in obs.tickets if t.sla_steps_remaining <= 2 and not t.resolved]

        target = None
        if sla_critical:
            target = min(sla_critical, key=lambda t: t.sla_steps_remaining)
        elif critical:
            target = critical[0]

        if target:
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": f"TACTICAL: Emergency resolution of {target.id}",
                },
                reasoning=(
                    f"SLA breach imminent for {target.id} "
                    f"({target.sla_steps_remaining} steps remaining)"
                ),
            )

        if obs.inbox:
            return ActionSchema(
                message_to=AGENT_MANAGER if "manager_agent" in str(obs.inbox) else None,
                message_content="TACTICAL: All urgent tickets cleared. Standing by.",
                reasoning="Queue empty, informing manager",
            )

        return ActionSchema(
            tool_call="get_tickets",
            tool_params={"priority_filter": 1},
            reasoning="Scanning for new urgent tickets",
        )
