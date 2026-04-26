"""
tools.py — Mock enterprise tool APIs for EnterpriseOps Arena.

All tools live inside ToolRegistry which owns the noise RNG and the
SQLite call log. Every call is logged regardless of success/failure.

Noise model: 8% of calls fail with a transient error dict before the
tool body executes — agents must handle {"error": ..., "transient": True}.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from contracts import ActionSchema, ResourcePool, TicketItem

# ---------------------------------------------------------------------------
# JSON meta-schemas for input validation
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_tickets": {
        "type": "object",
        "properties": {
            "priority_filter": {"type": ["integer", "null"], "minimum": 1, "maximum": 3},
        },
        "additionalProperties": False,
    },
    "resolve_ticket": {
        "type": "object",
        "required": ["ticket_id"],
        "properties": {
            "ticket_id":       {"type": "string"},
            "resolution_note": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
    "allocate_resource": {
        "type": "object",
        "required": ["resource_type", "amount", "requester_agent"],
        "properties": {
            "resource_type":   {"type": "string", "enum": ["engineers", "budget", "compute"]},
            "amount":          {"type": "number", "exclusiveMinimum": 0},
            "requester_agent": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "approve_budget": {
        "type": "object",
        "required": ["amount", "justification", "requester_agent"],
        "properties": {
            "amount":              {"type": "number", "exclusiveMinimum": 0},
            "justification":       {"type": "string"},
            "requester_agent":     {"type": "string"},
            "manager_countersign": {"type": "boolean"},
        },
        "additionalProperties": False,
    },
    "get_project_status": {
        "type": "object",
        "properties": {
            "task_id": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
    "resolve_subtask": {
        "type": "object",
        "required": ["ticket_id", "subtask_id"],
        "properties": {
            "ticket_id":       {"type": "string"},
            "subtask_id":      {"type": "string"},
            "resolution_note": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
}

_DDL = """
CREATE TABLE IF NOT EXISTS tool_call_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    step_number     INTEGER NOT NULL,
    agent_id        TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    success         INTEGER NOT NULL,
    noise_triggered INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL,
    timestamp       REAL NOT NULL
);
"""


def _validate_params(tool_name: str, params: dict[str, Any]) -> Optional[str]:
    """Validate params against the tool's JSON meta-schema. Returns error string or None."""
    schema = _TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return f"Unknown tool: {tool_name}"

    for req in schema.get("required", []):
        if req not in params:
            return f"Missing required param: '{req}'"

    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extra = set(params.keys()) - allowed
        if extra:
            return f"Unexpected params: {extra}"

    for prop, val in params.items():
        prop_schema = schema.get("properties", {}).get(prop, {})
        if not prop_schema:
            continue

        expected_types = prop_schema.get("type")
        if expected_types:
            if isinstance(expected_types, str):
                expected_types = [expected_types]
            type_map = {
                "string": str, "integer": int, "number": (int, float),
                "boolean": bool, "null": type(None), "object": dict, "array": list,
            }
            allowed_py = tuple(t for name in expected_types if (t := type_map.get(name)) is not None)
            if allowed_py and val is not None and not isinstance(val, allowed_py):
                return f"Param '{prop}' expected types {expected_types}, got {type(val).__name__}"

        if "enum" in prop_schema and val not in prop_schema["enum"]:
            return f"Param '{prop}' must be one of {prop_schema['enum']}, got {val!r}"

        if val is not None and isinstance(val, (int, float)):
            if "minimum" in prop_schema and val < prop_schema["minimum"]:
                return f"Param '{prop}' must be >= {prop_schema['minimum']}"
            if "exclusiveMinimum" in prop_schema and val <= prop_schema["exclusiveMinimum"]:
                return f"Param '{prop}' must be > {prop_schema['exclusiveMinimum']}"
            if "maximum" in prop_schema and val > prop_schema["maximum"]:
                return f"Param '{prop}' must be <= {prop_schema['maximum']}"

    return None


class ToolRegistry:
    """
    Registry of all enterprise mock tools.

    Args:
        world_model:  Live WorldModel instance.
        db_path:      SQLite file path (shared with WorldModel).
        noise_rate:   Probability of transient failure per call (default 0.08).
        seed:         RNG seed.
        drift_engine: SchemaDriftEngine instance (injected by env.py).
    """

    LARGE_BUDGET_THRESHOLD: float = 10_000.0

    def __init__(
        self,
        world_model: Any,
        db_path: str = "episodes.db",
        noise_rate: float = 0.08,
        seed: int = 42,
        drift_engine: Any = None,
    ) -> None:
        self._wm = world_model
        self._db_path = db_path
        self._noise_rate = noise_rate
        self._rng = random.Random(seed + 1)
        self._drift: Any = drift_engine
        self._current_step_num: int = 0
        self._current_step_logs: list[dict[str, Any]] = []
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
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

    def _log(self, agent_id: str, tool_name: str, params: dict[str, Any],
             result: dict[str, Any], success: bool, noise_triggered: bool) -> None:
        step_number = self._wm.step
        if step_number != self._current_step_num:
            self._current_step_num = step_number
            self._current_step_logs = []

        self._current_step_logs.append({
            "step_number": step_number,
            "agent_id": agent_id,
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "success": bool(success),
            "noise_triggered": bool(noise_triggered),
            "schema_version": self._wm.schema_version,
        })

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tool_call_log "
                "(step_number, agent_id, tool_name, params_json, result_json, success, noise_triggered, schema_version, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (step_number, agent_id, tool_name, json.dumps(params),
                 json.dumps(result), int(success), int(noise_triggered),
                 self._wm.schema_version, time.time()),
            )

    def _gate(self, tool_name: str, params: dict[str, Any], agent_id: str) -> Optional[dict[str, Any]]:
        """Run validation then noise roll. Returns error dict or None."""
        err = _validate_params(tool_name, params)
        if err:
            result = {"error": err, "transient": False, "tool": tool_name}
            self._log(agent_id, tool_name, params, result, success=False, noise_triggered=False)
            return result

        if self._rng.random() < self._noise_rate:
            result = {"error": "Transient service error — retry may succeed", "transient": True, "tool": tool_name}
            self._log(agent_id, tool_name, params, result, success=False, noise_triggered=True)
            return result

        return None

    def call(self, tool_name: str, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        """
        Dispatch a tool call by name. All results are logged to SQLite.

        Args:
            tool_name: One of the 5 registered tool names.
            params:    Tool parameters (validated against JSON schema).
            agent_id:  Calling agent identifier (for logging).
        """
        dispatch = {
            "get_tickets":        self._get_tickets,
            "resolve_ticket":     self._resolve_ticket,
            "allocate_resource":  self._allocate_resource,
            "approve_budget":     self._approve_budget,
            "get_project_status": self._get_project_status,
            "resolve_subtask":    self._resolve_subtask,
        }
        if tool_name not in dispatch:
            result = {"error": f"Unknown tool '{tool_name}'", "transient": False}
            self._log(agent_id, tool_name, params, result, success=False, noise_triggered=False)
            return result

        gate_result = self._gate(tool_name, params, agent_id)
        if gate_result is not None:
            return gate_result

        result = dispatch[tool_name](params, agent_id)

        # Apply drift transforms BEFORE logging so __deprecated_fields__ is captured
        if self._drift is not None and "error" not in result:
            result = self._drift.transform_response(tool_name, result, agent_id, self._wm.step)
            self._log(agent_id, tool_name, params, result, success=True, noise_triggered=False)

        return result

    # ------------------------------------------------------------------
    # Tool 1 — get_tickets
    # ------------------------------------------------------------------

    def _get_tickets(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        priority_filter: Optional[int] = params.get("priority_filter")
        tickets = self._wm.get_tickets()
        if priority_filter is not None:
            tickets = [t for t in tickets if t.priority == priority_filter]
        result: dict[str, Any] = {
            "tickets": [t.model_dump() for t in tickets],
            "count": len(tickets),
            "schema_version": self._wm.schema_version,
        }
        if self._drift is None:
            self._log(agent_id, "get_tickets", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Tool 2 — resolve_ticket
    # ------------------------------------------------------------------

    def _resolve_ticket(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        ticket_id: str = params["ticket_id"]
        resolution_note: str = params.get("resolution_note") or "Resolved by agent"
        tickets = {t.id: t for t in self._wm.get_tickets()}

        if ticket_id not in tickets:
            result = {"error": f"Ticket '{ticket_id}' not found", "transient": False, "resolved": False}
            self._log(agent_id, "resolve_ticket", params, result, success=False, noise_triggered=False)
            return result

        if tickets[ticket_id].resolved:
            result = {"error": f"Ticket '{ticket_id}' is already resolved", "transient": False, "resolved": False}
            self._log(agent_id, "resolve_ticket", params, result, success=False, noise_triggered=False)
            return result

        action = ActionSchema(tool_call="resolve_ticket", tool_params=params)
        self._wm.apply_action(
            agent_id=agent_id, action=action,
            state_delta={"ticket_update": {"id": ticket_id, "resolved": True, "resolution_note": resolution_note}},
            reason=f"{agent_id} resolved ticket {ticket_id}: {resolution_note}",
        )
        result = {"resolved": True, "ticket_id": ticket_id, "schema_version": self._wm.schema_version}
        if self._drift is None:
            self._log(agent_id, "resolve_ticket", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Tool 3 — allocate_resource
    # ------------------------------------------------------------------

    def _allocate_resource(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        resource_type: str = params["resource_type"]
        amount: float = params["amount"]
        rp = self._wm.get_resource_pool()
        available_map = {
            "engineers": rp.engineers_available,
            "budget":    rp.budget_remaining,
            "compute":   rp.compute_units,
        }
        available = available_map[resource_type]

        if amount > available:
            result = {
                "success": False,
                "error": f"Insufficient {resource_type}: requested {amount}, available {available}",
                "remaining": available, "resource_type": resource_type,
                "schema_version": self._wm.schema_version,
            }
            self._log(agent_id, "allocate_resource", params, result, success=False, noise_triggered=False)
            return result

        delta_map = {
            "engineers": {"engineers": -int(amount)},
            "budget":    {"budget":    -float(amount)},
            "compute":   {"compute":   -int(amount)},
        }
        action = ActionSchema(tool_call="allocate_resource", tool_params=params)
        self._wm.apply_action(
            agent_id=agent_id, action=action,
            state_delta={"resource_update": delta_map[resource_type]},
            reason=f"{agent_id} allocated {amount} {resource_type}",
        )
        updated_rp = self._wm.get_resource_pool()
        remaining_map = {
            "engineers": updated_rp.engineers_available,
            "budget":    updated_rp.budget_remaining,
            "compute":   updated_rp.compute_units,
        }
        result = {
            "success": True, "allocated": amount, "resource_type": resource_type,
            "remaining": remaining_map[resource_type], "schema_version": self._wm.schema_version,
        }
        if self._drift is None:
            self._log(agent_id, "allocate_resource", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Tool 4 — approve_budget
    # ------------------------------------------------------------------

    def _approve_budget(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        amount: float = params["amount"]
        justification: str = params["justification"]
        manager_countersign: bool = params.get("manager_countersign", False)

        if amount > self.LARGE_BUDGET_THRESHOLD and not manager_countersign:
            result = {
                "approved": False,
                "error": (f"Budget request of ${amount:,.2f} exceeds "
                          f"${self.LARGE_BUDGET_THRESHOLD:,.0f} threshold — manager_countersign required"),
                "amount": amount, "schema_version": self._wm.schema_version,
            }
            self._log(agent_id, "approve_budget", params, result, success=False, noise_triggered=False)
            return result

        rp = self._wm.get_resource_pool()
        if amount > rp.budget_remaining:
            result = {
                "approved": False,
                "error": f"Insufficient budget: requested ${amount:,.2f}, available ${rp.budget_remaining:,.2f}",
                "amount": amount, "schema_version": self._wm.schema_version,
            }
            self._log(agent_id, "approve_budget", params, result, success=False, noise_triggered=False)
            return result

        budget_event = {
            "step": self._wm.step, "agent_id": agent_id, "amount": amount,
            "justification": justification, "manager_countersign": manager_countersign,
        }
        action = ActionSchema(tool_call="approve_budget", tool_params=params)
        self._wm.apply_action(
            agent_id=agent_id, action=action,
            state_delta={"resource_update": {"budget": -float(amount)}, "add_budget_event": budget_event},
            reason=f"{agent_id} approved budget ${amount}: {justification}",
        )
        result = {
            "approved": True, "amount": amount,
            "budget_remaining": self._wm.get_resource_pool().budget_remaining,
            "schema_version": self._wm.schema_version,
        }
        if self._drift is None:
            self._log(agent_id, "approve_budget", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Tool 5 — get_project_status
    # ------------------------------------------------------------------

    def _get_project_status(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        task_id: Optional[str] = params.get("task_id")
        tasks = {t.id: t for t in self._wm.get_tasks()}

        if task_id is not None:
            if task_id not in tasks:
                result = {"error": f"Task '{task_id}' not found", "transient": False,
                          "schema_version": self._wm.schema_version}
                self._log(agent_id, "get_project_status", params, result, success=False, noise_triggered=False)
                return result
            result = {"task": tasks[task_id].model_dump(), "schema_version": self._wm.schema_version}
        else:
            result = {"tasks": [t.model_dump() for t in tasks.values()],
                      "count": len(tasks), "schema_version": self._wm.schema_version}

        if self._drift is None:
            self._log(agent_id, "get_project_status", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Tool 6 — resolve_subtask
    # ------------------------------------------------------------------

    def _resolve_subtask(self, params: dict[str, Any], agent_id: str) -> dict[str, Any]:
        ticket_id: str = params["ticket_id"]
        subtask_id: str = params["subtask_id"]

        tickets = {t.id: t for t in self._wm.get_tickets()}
        if ticket_id not in tickets:
            result = {"success": False, "error": f"Ticket {ticket_id} not found", "transient": False}
            self._log(agent_id, "resolve_subtask", params, result, success=False, noise_triggered=False)
            return result

        ticket = tickets[ticket_id]
        target = next((s for s in ticket.subtasks if s.id == subtask_id), None)
        if target is None:
            result = {"success": False, "error": f"Subtask {subtask_id} not found", "transient": False}
            self._log(agent_id, "resolve_subtask", params, result, success=False, noise_triggered=False)
            return result

        # Enforce sequential ordering — cannot skip steps
        if target.sequence > 1:
            prev = next((s for s in ticket.subtasks if s.sequence == target.sequence - 1), None)
            if prev and prev.status != "completed":
                result = {
                    "success": False,
                    "error": f"Must complete subtask sequence {target.sequence - 1} first",
                    "transient": False,
                }
                self._log(agent_id, "resolve_subtask", params, result, success=False, noise_triggered=False)
                return result

        updated_subtasks = [
            {**s.model_dump(), "status": "completed"} if s.id == subtask_id else s.model_dump()
            for s in ticket.subtasks
        ]
        all_done = all(s["status"] == "completed" for s in updated_subtasks)

        ticket_update: dict[str, Any] = {"id": ticket_id, "subtasks": updated_subtasks}
        if all_done:
            ticket_update["resolved"] = True

        action = ActionSchema(tool_call="resolve_subtask", tool_params=params)
        self._wm.apply_action(
            agent_id=agent_id, action=action,
            state_delta={"ticket_update": ticket_update},
            reason=f"{agent_id} resolved subtask {subtask_id} of ticket {ticket_id}",
        )

        result = {
            "success": True,
            "subtask_id": subtask_id,
            "ticket_id": ticket_id,
            "ticket_fully_resolved": all_done,
            "message": f"Subtask {subtask_id} completed",
            "schema_version": self._wm.schema_version,
        }
        if self._drift is None:
            self._log(agent_id, "resolve_subtask", params, result, success=True, noise_triggered=False)
        return result

    # ------------------------------------------------------------------
    # Log query helpers
    # ------------------------------------------------------------------

    def get_call_log(
        self,
        step_number: Optional[int] = None,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query the tool call log with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if step_number is not None:
            clauses.append("step_number = ?")
            params.append(step_number)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT * FROM tool_call_log {where} ORDER BY id"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            entry = dict(row)
            entry["params"]  = json.loads(entry.pop("params_json"))
            entry["result"]  = json.loads(entry.pop("result_json"))
            entry["success"] = bool(entry["success"])
            entry["noise_triggered"] = bool(entry["noise_triggered"])
            result.append(entry)
        return result

    def get_current_step_logs(self) -> list[dict[str, Any]]:
        """All tool calls made in the current step."""
        if self._wm.step != self._current_step_num:
            return []
        return list(self._current_step_logs)

    def list_tools(self) -> list[str]:
        return list(_TOOL_SCHEMAS.keys())
