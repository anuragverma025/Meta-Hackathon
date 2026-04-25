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
        tickets = obs.tickets or []
        normal = [
            t for t in tickets
            if not t.resolved and t.priority >= 2
        ]

        # Retry logic — skip last failed ticket
        recent = obs.recent_history or []
        last_failed_ticket = None
        if recent:
            last = recent[-1]
            if not last.success and last.tool_call == "resolve_ticket":
                last_failed_ticket = last.tool_params.get("ticket_id")

        if normal:
            # Skip failed ticket, try next
            targets = [t for t in normal if t.id != last_failed_ticket]
            if not targets:
                targets = normal
            # Sort by SLA pressure then priority
            target = min(targets, key=lambda t: (t.sla_steps_remaining, t.priority))
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": "STRATEGIC: Backlog resolution",
                },
                reasoning=f"Clearing P{target.priority} backlog: {target.id}",
            )

        if obs.inbox:
            return ActionSchema(
                message_to="manager_agent",
                message_content="STRATEGIC: Normal queue empty. Available.",
                tool_call="get_project_status",
                tool_params={},
                reasoning="Queue empty - monitoring projects",
            )
        return ActionSchema(
            tool_call="get_tickets",
            tool_params={"priority_filter": 2},
            reasoning="Scanning for new normal priority tickets",
        )
