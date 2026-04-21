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
            return self._finance_logic(obs)

        if self.role == "project":
            return self._project_logic(obs)

        if self.role == "manager":
            return self._manager_logic(obs)

        if self.role == "oversight":
            return self._oversight_logic(obs)

        return {}

    # ---------------- IT ----------------
    def _it_logic(self, obs):
        ticket = get_urgent_ticket(obs.tickets)

        if ticket:
            return {
                "tool_call": "resolve_ticket",
                "tool_params": {"ticket_id": ticket.id},
                "reasoning": f"Resolving urgent ticket {ticket.id}",
            }

        return {"reasoning": "No urgent tickets"}

    # ---------------- Finance ----------------
    def _finance_logic(self, obs):
        deal = get_stale_deal(obs.active_deals)

        if deal:
            return {
                "tool_call": "contact_client",
                "tool_params": {"deal_id": deal.id},
                "reasoning": f"Following up stale deal {deal.id}",
            }

        deal = get_negotiation_deal(obs.active_deals)
        if deal:
            return {
                "tool_call": "contact_client",
                "tool_params": {"deal_id": deal.id},
                "reasoning": f"Prioritizing negotiation deal {deal.id}",
            }

        return {"reasoning": "No action needed"}

    # ---------------- Project ----------------
    def _project_logic(self, obs):
        task = get_ready_task(obs.project_tasks)

        if task:
            return {
                "tool_call": "start_task",
                "tool_params": {"task_id": task.id},
                "reasoning": f"Starting task {task.id}",
            }

        return {"reasoning": "No ready tasks"}

    # ---------------- Manager ----------------
    def _manager_logic(self, obs):
        if obs.resource_pool:
            if obs.resource_pool.engineers_available < 1:
                return {
                    "message_to": "it_agent",
                    "message_content": "⚠️ No engineers available. Prioritize critical tickets.",
                    "reasoning": "Resource shortage",
                }

            if obs.resource_pool.budget_remaining < 100:
                return {
                    "message_to": "finance_agent",
                    "message_content": "⚠️ Budget critically low. Review deals.",
                    "reasoning": "Budget issue",
                }

        if len(obs.project_tasks) > 5:
            return {
                "message_to": "broadcast",
                "message_content": "⚠️ High workload detected. Prioritize execution.",
                "reasoning": "Too many tasks",
            }

        return {"reasoning": "No coordination needed"}

    # ---------------- OVERSIGHT (ADVANCED) ----------------
    def _oversight_logic(self, obs):
        issues = []
        severity_score = 0

        # -------- Critical SLA Risk --------
        critical_tickets = [
            t for t in obs.tickets
            if t.priority == 1 and not t.resolved
        ]

        near_breach = [
            t for t in critical_tickets
            if t.sla_steps_remaining <= 1
        ]

        if near_breach:
            issues.append(f"{len(near_breach)} critical tickets near SLA breach")
            severity_score += 3

        # -------- Stale Deals --------
        stale_deals = [
            d for d in obs.active_deals
            if d.steps_since_contact > 4
        ]

        if stale_deals:
            issues.append(f"{len(stale_deals)} stale deals")
            severity_score += 2

        # -------- Blocked Tasks --------
        blocked_tasks = [
            t for t in obs.project_tasks
            if t.status == "blocked"
        ]

        if blocked_tasks:
            issues.append(f"{len(blocked_tasks)} blocked tasks")
            severity_score += 2

        # -------- Resource Stress --------
        if obs.resource_pool:
            if obs.resource_pool.engineers_available == 0:
                issues.append("No engineers available")
                severity_score += 3

            if obs.resource_pool.budget_remaining < 50:
                issues.append("Budget critically low")
                severity_score += 2

        # -------- Simple Anomaly Detection --------
        # Detect unusual spike in workload
        if len(obs.project_tasks) > 8:
            issues.append("Anomalous spike in task volume")
            severity_score += 2

        if len(obs.tickets) > 8:
            issues.append("Anomalous spike in tickets")
            severity_score += 2

        # -------- Decision --------
        if issues:
            priority = 1 if severity_score >= 5 else 2

            return {
                "message_to": "manager_agent",
                "message_content": f"🚨 Oversight Alert: {' | '.join(issues)}",
                "reasoning": f"Oversight detected system risk (severity={severity_score})",
                "tool_call": None,
                "tool_params": {},
            }

        return {
            "reasoning": "System stable",
            "tool_params": {},
        }