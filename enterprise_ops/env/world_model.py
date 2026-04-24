"""
world_model.py — Persistent world state for EnterpriseOps Arena.

Owns the single source of truth for episode state. All mutations go through
apply_action() so every state change is causally logged to SQLite.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, Optional

from contracts import (
    ActionSchema,
    DealItem,
    ProjectTask,
    ResourcePool,
    TicketItem,
)


# ---------------------------------------------------------------------------
# DB schema DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_name   TEXT PRIMARY KEY,
    step_number     INTEGER NOT NULL,
    state_json      TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS causal_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    step_number     INTEGER NOT NULL,
    agent_id        TEXT NOT NULL,
    action_json     TEXT NOT NULL,
    changed_fields  TEXT NOT NULL,
    reason          TEXT NOT NULL,
    timestamp       REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Persistent world state backed by SQLite.

    All episode state lives here. Agents never mutate state directly —
    they go through apply_action() which writes a causal log entry and
    auto-snapshots every 5 steps.

    Args:
        db_path: Path to the SQLite file. Defaults to episodes.db in cwd.
        seed:    RNG seed for reproducibility.
    """

    SNAPSHOT_INTERVAL: int = 5

    def __init__(self, db_path: str = "episodes.db", seed: int = 42) -> None:
        self.db_path = db_path
        self.rng = random.Random(seed)
        self._step: int = 0
        self._schema_version: int = 1

        self._tickets: dict[str, TicketItem] = {}
        self._deals: dict[str, DealItem] = {}
        self._tasks: dict[str, ProjectTask] = {}
        self._resource_pool: ResourcePool = ResourcePool(
            engineers_available=0, budget_remaining=0.0, compute_units=0
        )
        self._budget_history: list[dict[str, Any]] = []
        self._pending_approvals: list[dict[str, Any]] = []

        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Initialisation from scenario
    # ------------------------------------------------------------------

    def load_scenario(self, scenario: dict[str, Any]) -> None:
        """Initialise world state from a scenario dict (loaded from YAML)."""
        self._step = 0
        self._schema_version = 1
        self._budget_history = []
        self._pending_approvals = []

        rp = scenario.get("resource_pool", {})
        self._resource_pool = ResourcePool(
            engineers_available=rp.get("engineers", 5),
            budget_remaining=float(rp.get("budget", 50000)),
            compute_units=rp.get("compute", 20),
        )

        self._tickets = {
            t["id"]: TicketItem(**t)
            for t in scenario.get("starting_tickets", [])
        }
        self._deals = {
            d["id"]: DealItem(**d)
            for d in scenario.get("starting_deals", [])
        }
        self._tasks = {
            t["id"]: ProjectTask(**t)
            for t in scenario.get("starting_tasks", [])
        }

        with self._conn() as conn:
            conn.execute("DELETE FROM snapshots")
            conn.execute("DELETE FROM causal_log")

        self._save_snapshot("snapshot_step_0")

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a deep copy of the full current world state."""
        return {
            "step": self._step,
            "schema_version": self._schema_version,
            "tickets": {k: v.model_dump() for k, v in self._tickets.items()},
            "deals": {k: v.model_dump() for k, v in self._deals.items()},
            "tasks": {k: v.model_dump() for k, v in self._tasks.items()},
            "resource_pool": self._resource_pool.model_dump(),
            "budget_history": deepcopy(self._budget_history),
            "pending_approvals": deepcopy(self._pending_approvals),
        }

    @property
    def step(self) -> int:
        return self._step

    @property
    def schema_version(self) -> int:
        return self._schema_version

    def get_tickets(self) -> list[TicketItem]:
        return list(self._tickets.values())

    def get_deals(self) -> list[DealItem]:
        return list(self._deals.values())

    def get_tasks(self) -> list[ProjectTask]:
        return list(self._tasks.values())

    def get_resource_pool(self) -> ResourcePool:
        return self._resource_pool.model_copy()

    def get_budget_history(self) -> list[dict[str, Any]]:
        return deepcopy(self._budget_history)

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        return deepcopy(self._pending_approvals)

    # ------------------------------------------------------------------
    # Public write interface
    # ------------------------------------------------------------------

    def apply_action(
        self,
        agent_id: str,
        action: ActionSchema,
        state_delta: dict[str, Any],
        reason: str,
    ) -> None:
        """
        Apply a validated state delta and write a causal log entry.

        Supported state_delta keys:
            ticket_update, deal_update, task_update, resource_update,
            add_budget_event, add_approval, resolve_approval
        """
        changed_fields: list[dict[str, Any]] = []

        if "ticket_update" in state_delta:
            changed_fields += self._apply_ticket_update(state_delta["ticket_update"])
        if "deal_update" in state_delta:
            changed_fields += self._apply_deal_update(state_delta["deal_update"])
        if "task_update" in state_delta:
            changed_fields += self._apply_task_update(state_delta["task_update"])
        if "resource_update" in state_delta:
            changed_fields += self._apply_resource_update(state_delta["resource_update"])
        if "add_budget_event" in state_delta:
            entry = state_delta["add_budget_event"]
            self._budget_history.append(entry)
            changed_fields.append({"field": "budget_history", "before": None, "after": entry})
        if "add_approval" in state_delta:
            approval = state_delta["add_approval"]
            self._pending_approvals.append(approval)
            changed_fields.append({"field": "pending_approvals", "before": None, "after": approval})
        if "resolve_approval" in state_delta:
            approval_id = state_delta["resolve_approval"]["id"]
            self._pending_approvals = [
                a for a in self._pending_approvals if a.get("id") != approval_id
            ]
            changed_fields.append({"field": "pending_approvals.resolved", "before": approval_id, "after": None})

        self._write_causal_log(agent_id, action, changed_fields, reason)

    def advance_step(self) -> None:
        """
        Increment step counter, age time-sensitive fields, and
        auto-snapshot every SNAPSHOT_INTERVAL steps.
        """
        self._step += 1

        for ticket in self._tickets.values():
            if not ticket.resolved and ticket.sla_steps_remaining > 0:
                object.__setattr__(ticket, "sla_steps_remaining", ticket.sla_steps_remaining - 1)

        for deal in self._deals.values():
            if deal.stage not in ("closed_won", "closed_lost"):
                object.__setattr__(deal, "steps_since_contact", deal.steps_since_contact + 1)

        for task in self._tasks.values():
            if task.status not in ("completed", "failed") and task.deadline_steps > 0:
                object.__setattr__(task, "deadline_steps", task.deadline_steps - 1)

        if self._step % self.SNAPSHOT_INTERVAL == 0:
            self._save_snapshot(f"snapshot_step_{self._step}")

    def set_schema_version(self, version: int) -> None:
        self._schema_version = version

    # ------------------------------------------------------------------
    # Snapshot & rollback
    # ------------------------------------------------------------------

    def _save_snapshot(self, name: str) -> None:
        state_json = json.dumps(self.get_state())
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (snapshot_name, step_number, state_json, created_at) VALUES (?, ?, ?, ?)",
                (name, self._step, state_json, time.time()),
            )

    def rollback_to_snapshot(self, step_n: int) -> bool:
        """Restore world state to the snapshot taken at step_n."""
        name = f"snapshot_step_{step_n}"
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM snapshots WHERE snapshot_name = ?", (name,)
            ).fetchone()

        if row is None:
            return False

        state = json.loads(row["state_json"])
        self._step = state["step"]
        self._schema_version = state["schema_version"]
        self._resource_pool = ResourcePool(**state["resource_pool"])
        self._tickets = {k: TicketItem(**v) for k, v in state["tickets"].items()}
        self._deals = {k: DealItem(**v) for k, v in state["deals"].items()}
        self._tasks = {k: ProjectTask(**v) for k, v in state["tasks"].items()}
        self._budget_history = state["budget_history"]
        self._pending_approvals = state["pending_approvals"]
        return True

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT snapshot_name, step_number, created_at FROM snapshots ORDER BY step_number"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Causal log
    # ------------------------------------------------------------------

    def _write_causal_log(
        self,
        agent_id: str,
        action: ActionSchema,
        changed_fields: list[dict[str, Any]],
        reason: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO causal_log (step_number, agent_id, action_json, changed_fields, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (self._step, agent_id, action.model_dump_json(),
                 json.dumps(changed_fields), reason, time.time()),
            )

    def get_causal_log(
        self,
        agent_id: Optional[str] = None,
        since_step: int = 0,
    ) -> list[dict[str, Any]]:
        """Query the causal log, optionally filtered by agent and step."""
        query = "SELECT * FROM causal_log WHERE step_number >= ?"
        params: list[Any] = [since_step]
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY id"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            entry = dict(row)
            entry["action"] = json.loads(entry.pop("action_json"))
            entry["changed_fields"] = json.loads(entry["changed_fields"])
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Internal state mutators
    # ------------------------------------------------------------------

    def _apply_ticket_update(self, update: dict[str, Any]) -> list[dict[str, Any]]:
        ticket_id = update["id"]
        changes: list[dict[str, Any]] = []
        if ticket_id not in self._tickets:
            self._tickets[ticket_id] = TicketItem(**update)
            changes.append({"field": f"tickets.{ticket_id}", "before": None, "after": update})
            return changes
        ticket = self._tickets[ticket_id]
        old = ticket.model_dump()
        updated = old | {k: v for k, v in update.items() if k != "id"}
        self._tickets[ticket_id] = TicketItem(**updated)
        for field, old_val in old.items():
            new_val = updated.get(field)
            if new_val != old_val:
                changes.append({"field": f"tickets.{ticket_id}.{field}", "before": old_val, "after": new_val})
        return changes

    def _apply_deal_update(self, update: dict[str, Any]) -> list[dict[str, Any]]:
        deal_id = update["id"]
        changes: list[dict[str, Any]] = []
        if deal_id not in self._deals:
            self._deals[deal_id] = DealItem(**update)
            changes.append({"field": f"deals.{deal_id}", "before": None, "after": update})
            return changes
        deal = self._deals[deal_id]
        old = deal.model_dump()
        updated = old | {k: v for k, v in update.items() if k != "id"}
        self._deals[deal_id] = DealItem(**updated)
        for field, old_val in old.items():
            new_val = updated.get(field)
            if new_val != old_val:
                changes.append({"field": f"deals.{deal_id}.{field}", "before": old_val, "after": new_val})
        return changes

    def _apply_task_update(self, update: dict[str, Any]) -> list[dict[str, Any]]:
        task_id = update["id"]
        changes: list[dict[str, Any]] = []
        if task_id not in self._tasks:
            self._tasks[task_id] = ProjectTask(**update)
            changes.append({"field": f"tasks.{task_id}", "before": None, "after": update})
            return changes
        task = self._tasks[task_id]
        old = task.model_dump()
        updated = old | {k: v for k, v in update.items() if k != "id"}
        self._tasks[task_id] = ProjectTask(**updated)
        for field, old_val in old.items():
            new_val = updated.get(field)
            if new_val != old_val:
                changes.append({"field": f"tasks.{task_id}.{field}", "before": old_val, "after": new_val})
        return changes

    def _apply_resource_update(self, delta: dict[str, Any]) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        old = self._resource_pool.model_dump()
        new_engineers = max(0, old["engineers_available"] + delta.get("engineers", 0))
        new_budget    = max(0.0, old["budget_remaining"]  + delta.get("budget",    0.0))
        new_compute   = max(0, old["compute_units"]       + delta.get("compute",   0))
        self._resource_pool = ResourcePool(
            engineers_available=new_engineers,
            budget_remaining=new_budget,
            compute_units=new_compute,
        )
        for field, old_val in old.items():
            new_val = self._resource_pool.model_dump()[field]
            if new_val != old_val:
                changes.append({"field": f"resource_pool.{field}", "before": old_val, "after": new_val})
        return changes
