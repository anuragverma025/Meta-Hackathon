def prioritize_tickets(tickets):
    return sorted(
        tickets,
        key=lambda t: (t.priority, t.sla_steps_remaining)
    )


def get_urgent_ticket(tickets):
    for t in prioritize_tickets(tickets):
        if not t.resolved and (t.priority == 1 or t.sla_steps_remaining <= 1):
            return t
    return None


def get_stale_deal(deals):
    for d in deals:
        if d.steps_since_contact > 3:
            return d
    return None


def get_negotiation_deal(deals):
    for d in deals:
        if d.stage == "negotiation":
            return d
    return None


def get_ready_task(tasks):
    for task in tasks:
        if task.status != "pending":
            continue

        if all(dep not in [t.id for t in tasks if t.status != "completed"]
               for dep in task.depends_on):
            return task

    return None