"""
schema_drift.py — Runtime schema mutation engine for EnterpriseOps Arena.

Every 20 steps the SchemaDriftEngine selects 1-2 fields from a tool's
return schema and applies one of three mutation types:

    RENAME     — old_name -> new_name  (old_name becomes invalid)
    ADD_FIELD  — injects a new required field into the return payload
    DEPRECATE  — field still returned, but marked deprecated:true in payload
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator, Optional


class MutationType(str, Enum):
    RENAME    = "RENAME"
    ADD_FIELD = "ADD_FIELD"
    DEPRECATE = "DEPRECATE"


_MUTATION_CATALOGUE: list[tuple[str, str, MutationType, Any]] = [
    ("get_tickets",       "count",          MutationType.RENAME,    "total_count"),
    ("get_tickets",       "tickets",        MutationType.RENAME,    "ticket_list"),
    ("get_tickets",       "schema_version", MutationType.DEPRECATE, None),
    ("get_tickets",       "fetched_at",     MutationType.ADD_FIELD, "__auto_timestamp__"),
    ("resolve_ticket",    "resolved",       MutationType.RENAME,    "is_resolved"),
    ("resolve_ticket",    "ticket_id",      MutationType.RENAME,    "id"),
    ("resolve_ticket",    "resolved_by",    MutationType.ADD_FIELD, "__agent_id__"),
    ("resolve_ticket",    "ticket_id",      MutationType.DEPRECATE, None),
    ("allocate_resource", "remaining",      MutationType.RENAME,    "units_remaining"),
    ("allocate_resource", "allocated",      MutationType.RENAME,    "units_allocated"),
    ("allocate_resource", "resource_type",  MutationType.DEPRECATE, None),
    ("allocate_resource", "allocation_id",  MutationType.ADD_FIELD, "__auto_id__"),
    ("approve_budget",    "budget_remaining", MutationType.RENAME,  "remaining_budget"),
    ("approve_budget",    "amount",           MutationType.RENAME,  "approved_amount"),
    ("approve_budget",    "approved",         MutationType.DEPRECATE, None),
    ("approve_budget",    "approval_ref",     MutationType.ADD_FIELD, "__auto_id__"),
    ("get_project_status", "count",          MutationType.RENAME,   "task_count"),
    ("get_project_status", "tasks",          MutationType.RENAME,   "task_list"),
    ("get_project_status", "task",           MutationType.RENAME,   "project_task"),
    ("get_project_status", "retrieved_at",   MutationType.ADD_FIELD, "__auto_timestamp__"),
]

_DDL = """
CREATE TABLE IF NOT EXISTS schema_drift_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version  INTEGER NOT NULL,
    step_number     INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    field_name      TEXT NOT NULL,
    mutation_type   TEXT NOT NULL,
    new_name        TEXT,
    added_value     TEXT,
    timestamp       REAL NOT NULL
);
"""


class SchemaDriftEngine:
    """
    Manages runtime schema mutations applied every 20 steps.

    Args:
        db_path:     SQLite file (shared with WorldModel + ToolRegistry).
        seed:        RNG seed for reproducible drift sequences.
        drift_every: Steps between drift events (default 20).
    """

    DRIFT_INTERVAL: int = 20

    def __init__(self, db_path: str = "episodes.db", seed: int = 42, drift_every: int = 20) -> None:
        self._db_path = db_path
        self._rng = random.Random(seed + 99)
        self._drift_every = drift_every
        self._schema_version: int = 1
        self._active: dict[str, list[DriftRecord]] = {}
        self._used_catalogue_indices: set[int] = set()
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

    @property
    def schema_version(self) -> int:
        return self._schema_version

    def maybe_drift(self, step: int) -> list["DriftRecord"]:
        """Trigger 1-2 mutations when step % drift_every == 0 and step > 0."""
        if step == 0 or step % self._drift_every != 0:
            return []
        return self._apply_drift_event(step)

    def transform_response(
        self, tool_name: str, response: dict[str, Any], agent_id: str, step: int
    ) -> dict[str, Any]:
        """Rewrite a tool response to reflect active mutations for that tool."""
        mutations = self._active.get(tool_name, [])
        if not mutations:
            return response

        result = dict(response)
        deprecated_fields: list[str] = []

        for m in mutations:
            if m.mutation_type == MutationType.RENAME:
                if m.field_name in result:
                    result[m.new_name] = result.pop(m.field_name)
            elif m.mutation_type == MutationType.ADD_FIELD:
                if m.field_name not in result:
                    result[m.field_name] = self._resolve_default(m.added_value, agent_id, step)
            elif m.mutation_type == MutationType.DEPRECATE:
                if m.field_name in result:
                    deprecated_fields.append(m.field_name)

        if deprecated_fields:
            existing = result.get("__deprecated_fields__", [])
            result["__deprecated_fields__"] = list(set(existing + deprecated_fields))

        return result

    def validate_field_access(self, tool_name: str, field_name: str) -> tuple[bool, Optional[str]]:
        """Check whether a field has been renamed away. Returns (is_valid, error_msg)."""
        for m in self._active.get(tool_name, []):
            if m.mutation_type == MutationType.RENAME and m.field_name == field_name:
                return False, (
                    f"Field '{field_name}' was renamed to '{m.new_name}' "
                    f"at schema_version={m.schema_version}. Update your field reference."
                )
        return True, None

    def is_deprecated_field(self, tool_name: str, field_name: str) -> bool:
        for m in self._active.get(tool_name, []):
            if m.mutation_type == MutationType.DEPRECATE and m.field_name == field_name:
                return True
        return False

    def get_drift_history(self, since_version: int = 1) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM schema_drift_history WHERE schema_version >= ? ORDER BY id",
                (since_version,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_mutations(self) -> dict[str, list[dict[str, Any]]]:
        return {tool: [m.to_dict() for m in mutations] for tool, mutations in self._active.items()}

    def reset(self) -> None:
        """Clear all active mutations and reset version to 1."""
        self._active.clear()
        self._used_catalogue_indices.clear()
        self._schema_version = 1
        with self._conn() as conn:
            conn.execute("DELETE FROM schema_drift_history")

    def _apply_drift_event(self, step: int) -> list["DriftRecord"]:
        available = [i for i in range(len(_MUTATION_CATALOGUE)) if i not in self._used_catalogue_indices]
        if not available:
            return []
        n_mutations = min(self._rng.randint(1, 2), len(available))
        chosen_indices = self._rng.sample(available, n_mutations)
        self._schema_version += 1
        records: list[DriftRecord] = []
        for idx in chosen_indices:
            self._used_catalogue_indices.add(idx)
            tool_name, field_name, mut_type, new_val = _MUTATION_CATALOGUE[idx]
            record = DriftRecord(
                schema_version=self._schema_version, step_number=step,
                tool_name=tool_name, field_name=field_name, mutation_type=mut_type,
                new_name=new_val if mut_type == MutationType.RENAME else None,
                added_value=new_val if mut_type == MutationType.ADD_FIELD else None,
            )
            self._active.setdefault(tool_name, []).append(record)
            records.append(record)
            self._persist_record(record)
        return records

    def _persist_record(self, record: "DriftRecord") -> None:
        added_json = json.dumps(record.added_value) if record.added_value is not None else None
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO schema_drift_history "
                "(schema_version, step_number, tool_name, field_name, mutation_type, new_name, added_value, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record.schema_version, record.step_number, record.tool_name, record.field_name,
                 record.mutation_type.value, record.new_name, added_json, time.time()),
            )

    @staticmethod
    def _resolve_default(template: Any, agent_id: str, step: int) -> Any:
        if template == "__auto_timestamp__":
            return step
        if template == "__auto_id__":
            return f"{agent_id}-{step}-{int(time.time() * 1000) % 100000}"
        if template == "__agent_id__":
            return agent_id
        return template


class DriftRecord:
    """Immutable record of a single field mutation."""

    __slots__ = ("schema_version", "step_number", "tool_name", "field_name",
                 "mutation_type", "new_name", "added_value")

    def __init__(self, schema_version: int, step_number: int, tool_name: str,
                 field_name: str, mutation_type: MutationType,
                 new_name: Optional[str], added_value: Any) -> None:
        self.schema_version = schema_version
        self.step_number    = step_number
        self.tool_name      = tool_name
        self.field_name     = field_name
        self.mutation_type  = mutation_type
        self.new_name       = new_name
        self.added_value    = added_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step_number":    self.step_number,
            "tool_name":      self.tool_name,
            "field_name":     self.field_name,
            "mutation_type":  self.mutation_type.value,
            "new_name":       self.new_name,
            "added_value":    self.added_value,
        }

    def __repr__(self) -> str:
        if self.mutation_type == MutationType.RENAME:
            detail = f"{self.field_name!r} -> {self.new_name!r}"
        elif self.mutation_type == MutationType.ADD_FIELD:
            detail = f"add {self.field_name!r}={self.added_value!r}"
        else:
            detail = f"deprecate {self.field_name!r}"
        return f"<DriftRecord v{self.schema_version} {self.tool_name} {detail}>"
