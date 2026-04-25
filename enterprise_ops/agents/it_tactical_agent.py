from contracts import ObservationSchema, ActionSchema, AGENT_IT_TACTICAL
from agents.base_agent import BaseAgent


class ITTacticalAgent(BaseAgent):
    """
    Handles urgent/critical tickets — priority 1 and SLA-critical.
    Focus: speed, SLA compliance, emergency resolution.
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_IT_TACTICAL, policy=None)

    def act(self, obs: ObservationSchema) -> ActionSchema:
        tickets = obs.tickets or []

        # Step 1-3: SLA rescue mode — eliminate breach risk first
        sla_critical = [
            t for t in tickets
            if not t.resolved and t.sla_steps_remaining <= 3
        ]
        p1_tickets = [
            t for t in tickets
            if not t.resolved and t.priority == 1
        ]

        # Check recent history for retry logic
        recent = obs.recent_history or []
        last_failed_ticket = None
        if recent:
            last = recent[-1]
            if not last.success and last.tool_call == "resolve_ticket":
                last_failed_ticket = last.tool_params.get("ticket_id")

        # SLA critical first — absolute priority
        if sla_critical:
            # Skip last failed ticket — try next one
            targets = [t for t in sla_critical if t.id != last_failed_ticket]
            if not targets:
                targets = sla_critical  # retry if no other option
            target = min(targets, key=lambda t: t.sla_steps_remaining)
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": f"TACTICAL RESCUE: SLA={target.sla_steps_remaining}",
                },
                reasoning=f"SLA breach in {target.sla_steps_remaining} steps",
            )

        # P1 tickets next
        if p1_tickets:
            targets = [t for t in p1_tickets if t.id != last_failed_ticket]
            if not targets:
                targets = p1_tickets
            target = targets[0]
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": "TACTICAL: P1 resolution",
                },
                reasoning=f"Resolving P1 ticket {target.id}",
            )

        if obs.step_number <= 4:
            return ActionSchema(
                message_to="manager_agent",
                message_content="TACTICAL: P1 queue clear. Standing by.",
                tool_call="get_project_status",
                tool_params={},
                reasoning="No urgent tickets - checking project status",
            )
        return ActionSchema(
            tool_call="get_project_status",
            tool_params={},
            reasoning="Monitoring project status after ticket resolution",
        )
