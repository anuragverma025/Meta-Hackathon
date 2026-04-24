"""
models.py — OpenEnv Action + Observation dataclasses for EnterpriseOps Arena.

These wrap the existing Pydantic schemas from contracts.py into the
dataclass-based interface that OpenEnv expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Try importing OpenEnv base classes; provide stubs if not installed.
# If the openenv bases are Pydantic models, fall back to plain dataclass stubs
# because @dataclass on a Pydantic subclass skips __pydantic_extra__ init.
# ---------------------------------------------------------------------------
def _load_openenv_bases():
    for mod in ("openenv.core.env_server", "openenv.core"):
        try:
            import importlib
            m = importlib.import_module(mod)
            _Action = getattr(m, "Action", None)
            _Obs = getattr(m, "Observation", None)
            if _Action is None or _Obs is None:
                continue
            # Reject Pydantic-based bases — incompatible with @dataclass
            try:
                from pydantic import BaseModel
                if issubclass(_Action, BaseModel) or issubclass(_Obs, BaseModel):
                    continue
            except Exception:
                pass
            return _Action, _Obs
        except ImportError:
            continue
    return None, None

_Action, _Obs = _load_openenv_bases()
if _Action is not None:
    Action = _Action
    Observation = _Obs
else:
    @dataclass
    class Action:
        """Fallback base when openenv-core is not installed or is Pydantic-based."""
        pass

    @dataclass
    class Observation:
        """Fallback base when openenv-core is not installed or is Pydantic-based."""
        pass


# ---------------------------------------------------------------------------
# EnterpriseAction — what the agent sends each step
# ---------------------------------------------------------------------------

@dataclass
class EnterpriseAction(Action):
    """
    Single-agent action for one step.

    Fields mirror contracts.ActionSchema so conversion is trivial.
    """

    agent_id: str = ""
    tool_call: Optional[str] = None
    tool_params: dict[str, Any] = field(default_factory=dict)
    message_to: Optional[str] = None
    message_content: Optional[str] = None
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tool_call": self.tool_call,
            "tool_params": self.tool_params,
            "message_to": self.message_to,
            "message_content": self.message_content,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnterpriseAction":
        return cls(
            agent_id=d.get("agent_id", ""),
            tool_call=d.get("tool_call"),
            tool_params=d.get("tool_params", {}),
            message_to=d.get("message_to"),
            message_content=d.get("message_content"),
            reasoning=d.get("reasoning"),
        )


# ---------------------------------------------------------------------------
# EnterpriseObservation — what the agent receives each step
# ---------------------------------------------------------------------------

@dataclass
class EnterpriseObservation(Observation):
    """
    Per-agent observation returned by the environment.

    Fields mirror contracts.ObservationSchema plus reward/done metadata
    so the training loop has everything in one object.
    """

    agent_id: str = ""
    inbox: list[dict[str, Any]] = field(default_factory=list)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    resource_pool: Optional[dict[str, Any]] = None
    active_deals: list[dict[str, Any]] = field(default_factory=list)
    project_tasks: list[dict[str, Any]] = field(default_factory=list)
    step_number: int = 0
    schema_version: int = 1
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "inbox": self.inbox,
            "tickets": self.tickets,
            "resource_pool": self.resource_pool,
            "active_deals": self.active_deals,
            "project_tasks": self.project_tasks,
            "step_number": self.step_number,
            "schema_version": self.schema_version,
            "reward_breakdown": self.reward_breakdown,
            "reward": self.reward,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnterpriseObservation":
        return cls(
            agent_id=d.get("agent_id", ""),
            inbox=d.get("inbox", []),
            tickets=d.get("tickets", []),
            resource_pool=d.get("resource_pool"),
            active_deals=d.get("active_deals", []),
            project_tasks=d.get("project_tasks", []),
            step_number=d.get("step_number", 0),
            schema_version=d.get("schema_version", 1),
            reward_breakdown=d.get("reward_breakdown", {}),
            reward=d.get("reward", 0.0),
            done=d.get("done", False),
        )
