from agents.utils import (
    get_urgent_ticket,
    get_stale_deal,
    get_negotiation_deal,
    get_ready_task,
)


class RulePolicy:
    def __init__(self, role: str):
        self.role = role

    def decide(self, obs, agent_id: str) -> dict:
        if self.role == "it":
            return self._it_logic(obs)

        if self.role == "finance":
            return self._finance_logic(obs, agent_id)

        if self.role == "project":
            return self._project_logic(obs, agent_id)

        if self.role == "manager":
            return self._manager_logic(obs)

        if self.role == "oversight":
            return self._oversight_logic(obs)

        return {}

    # ---------------- IT ----------------
    def _it_logic(self, obs):
        # Step 0: survey the queue and notify manager so the message bus is exercised
        if obs.step_number == 0:
            sla_critical = sum(1 for t in obs.tickets if not t.resolved and t.sla_steps_remaining <= 5)
            return {
                "tool_call": "get_tickets",
                "tool_params": {},
                "message_to": "manager_agent",
                "message_content": (
                    f"IT agent online at step {obs.step_number}. "
                    f"Tickets in queue: {len(obs.tickets)}. "
                    f"SLA-critical (<=5 steps): {sla_critical}."
                ),
                "reasoning": "Initial ticket survey + manager notification",
            }

        ticket = get_urgent_ticket(obs.tickets)
        if ticket:
            return {
                "tool_call": "resolve_ticket",
                "tool_params": {
                    "ticket_id": ticket.id,
                    "resolution_note": f"Resolved by IT agent at step {obs.step_number}",
                },
                "reasoning": f"Resolving urgent ticket {ticket.id}",
            }

        # Resolve any remaining unresolved ticket
        unresolved = sorted(
            [t for t in obs.tickets if not t.resolved],
            key=lambda t: (t.priority, t.sla_steps_remaining),
        )
        if unresolved:
            t = unresolved[0]
            return {
                "tool_call": "resolve_ticket",
                "tool_params": {
                    "ticket_id": t.id,
                    "resolution_note": f"Resolved by IT agent at step {obs.step_number}",
                },
                "reasoning": f"Resolving ticket {t.id}",
            }

        return {
            "tool_call": "get_tickets",
            "tool_params": {},
            "reasoning": "All tickets resolved — monitoring queue",
        }

    # ---------------- Finance ----------------
    def _finance_logic(self, obs, agent_id: str):
        deal = get_stale_deal(obs.active_deals)
        if deal:
            return {
                "tool_call": "approve_budget",
                "tool_params": {
                    "amount": 500.0,
                    "justification": f"Re-engagement budget for stale deal {deal.id}",
                    "requester_agent": agent_id,
                },
                "reasoning": f"Approving budget to re-engage stale deal {deal.id}",
            }

        deal = get_negotiation_deal(obs.active_deals)
        if deal:
            return {
                "tool_call": "approve_budget",
                "tool_params": {
                    "amount": 1000.0,
                    "justification": f"Closing budget for negotiation deal {deal.id}",
                    "requester_agent": agent_id,
                },
                "reasoning": f"Approving budget for deal {deal.id} in negotiation",
            }

        return {"reasoning": "No active deals requiring budget action"}

    # ---------------- Project ----------------
    def _project_logic(self, obs, agent_id: str):
        task = get_ready_task(obs.project_tasks)
        if task and obs.resource_pool and obs.resource_pool.compute_units > 0:
            return {
                "tool_call": "allocate_resource",
                "tool_params": {
                    "resource_type": "compute",
                    "amount": 1,
                    "requester_agent": agent_id,
                },
                "reasoning": f"Allocating compute for task {task.id}",
            }

        return {
            "tool_call": "get_project_status",
            "tool_params": {},
            "reasoning": "Monitoring project status",
        }

    # ---------------- Manager ----------------
    def _manager_logic(self, obs):
        if obs.resource_pool:
            if obs.resource_pool.engineers_available < 1:
                return {
                    "message_to": "it_agent",
                    "message_content": "No engineers available. Prioritize critical tickets.",
                    "reasoning": "Resource shortage",
                }

            if obs.resource_pool.budget_remaining < 100:
                return {
                    "message_to": "finance_agent",
                    "message_content": "Budget critically low. Review deals.",
                    "reasoning": "Budget issue",
                }

        if len(obs.project_tasks) > 5:
            return {
                "message_to": "broadcast",
                "message_content": "High workload detected. Prioritize execution.",
                "reasoning": "Too many tasks",
            }

        # Default: check project status (earns +1 task_completion reward)
        return {
            "tool_call": "get_project_status",
            "tool_params": {},
            "reasoning": "Monitoring all project tasks",
        }

    # ---------------- OVERSIGHT (ADVANCED) ----------------
    def _oversight_logic(self, obs):
        issues = []
        severity_score = 0

        critical_tickets = [
            t for t in obs.tickets
            if t.priority == 1 and not t.resolved
        ]
        near_breach = [t for t in critical_tickets if t.sla_steps_remaining <= 1]
        if near_breach:
            issues.append(f"{len(near_breach)} critical tickets near SLA breach")
            severity_score += 3

        stale_deals = [d for d in obs.active_deals if d.steps_since_contact > 4]
        if stale_deals:
            issues.append(f"{len(stale_deals)} stale deals")
            severity_score += 2

        blocked_tasks = [t for t in obs.project_tasks if t.status == "blocked"]
        if blocked_tasks:
            issues.append(f"{len(blocked_tasks)} blocked tasks")
            severity_score += 2

        if obs.resource_pool:
            if obs.resource_pool.engineers_available == 0:
                issues.append("No engineers available")
                severity_score += 3
            if obs.resource_pool.budget_remaining < 50:
                issues.append("Budget critically low")
                severity_score += 2

        if len(obs.project_tasks) > 8:
            issues.append("Anomalous spike in task volume")
            severity_score += 2

        if len(obs.tickets) > 8:
            issues.append("Anomalous spike in tickets")
            severity_score += 2

        if issues:
            return {
                "message_to": "manager_agent",
                "message_content": f"Oversight Alert: {' | '.join(issues)}",
                "reasoning": f"Oversight detected system risk (severity={severity_score})",
                "tool_call": None,
                "tool_params": {},
            }

        return {
            "reasoning": "System stable",
            "tool_params": {},
        }
