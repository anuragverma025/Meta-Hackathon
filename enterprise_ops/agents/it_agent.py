from contracts import ObservationSchema, ActionSchema, AGENT_IT
from agents.base_agent import BaseAgent


class ITAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_id=AGENT_IT, policy=None)

    def act(self, obs: ObservationSchema) -> ActionSchema:
        tickets = obs.tickets or []
        
        # Check action history for retry logic
        recent = obs.recent_history or []
        last_failed_ticket = None
        if recent:
            last = recent[-1]
            if not last.success and last.tool_call == "resolve_ticket":
                last_failed_ticket = last.tool_params.get("ticket_id")
        
        # Priority order: SLA breach risk → P1 → P2 → P3
        unresolved = [t for t in tickets if not t.resolved]
        
        sla_critical = [t for t in unresolved 
                       if t.sla_steps_remaining <= 2
                       and t.id != last_failed_ticket]
        if sla_critical:
            target = min(sla_critical, key=lambda t: t.sla_steps_remaining)
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={"ticket_id": target.id,
                            "resolution_note": f"SLA rescue: {target.id}"},
                reasoning=f"SLA breach risk: {target.sla_steps_remaining} steps"
            )
        
        p1 = [t for t in unresolved 
              if t.priority == 1 and t.id != last_failed_ticket]
        if p1:
            target = p1[0]
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={"ticket_id": target.id,
                            "resolution_note": f"P1 resolution: {target.id}"},
                reasoning=f"Resolving P1 ticket {target.id}"
            )
        
        normal = [t for t in unresolved if t.id != last_failed_ticket]
        if normal:
            target = min(normal, key=lambda t: (t.priority,
                                                t.sla_steps_remaining))
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={"ticket_id": target.id,
                            "resolution_note": f"Resolution: {target.id}"},
                reasoning=f"Resolving P{target.priority} ticket {target.id}"
            )
        
        return ActionSchema(
            tool_call="get_tickets",
            tool_params={},
            reasoning="Scanning for new tickets"
        )