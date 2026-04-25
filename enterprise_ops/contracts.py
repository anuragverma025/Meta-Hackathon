"""
contracts.py — Shared Pydantic v2 schemas for EnterpriseOps Arena.

ALL teammates import from here. Do not modify existing field names or types
without coordinating with the full team — these are the integration contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class TicketCategory(str, Enum):
    URGENT  = "urgent"
    ROUTINE = "routine"
    UNKNOWN = "unknown"


class TicketSubtask(BaseModel):
    id: str = Field(..., description="e.g. SUBTASK-001")
    ticket_id: str
    sequence: int = Field(..., description="Must complete in order: 1, 2, 3")
    description: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


class TicketItem(BaseModel):
    """An IT support ticket in the system."""

    id: str = Field(..., description="Unique ticket identifier, e.g. 'TKT-001'")
    priority: int = Field(..., ge=1, le=3, description="1=critical, 2=high, 3=normal")
    description: str = Field(..., description="Human-readable ticket description")
    sla_steps_remaining: int = Field(..., ge=0, description="Steps until SLA breach")
    resolved: bool = Field(default=False, description="Whether the ticket is resolved")
    resolution_note: Optional[str] = Field(default=None)
    category: TicketCategory = Field(
        default=TicketCategory.UNKNOWN,
        description="Auto-classified based on priority and description",
    )
    subtasks: list[TicketSubtask] = Field(
        default_factory=list,
        description="Optional subtasks for complex tickets",
    )


class DealItem(BaseModel):
    """A sales deal tracked by the enterprise CRM."""

    id: str = Field(..., description="Unique deal identifier, e.g. 'DEAL-001'")
    stage: Literal["prospecting", "proposal", "negotiation", "closed_won", "closed_lost"] = Field(
        ..., description="Current pipeline stage"
    )
    value: float = Field(..., ge=0.0, description="Deal value in USD")
    client_name: str
    steps_since_contact: int = Field(default=0, ge=0, description="Steps elapsed since last contact")


class ProjectTask(BaseModel):
    """A project task with dependency tracking."""

    id: str = Field(..., description="Unique task identifier, e.g. 'TASK-001'")
    name: str
    depends_on: list[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    assigned_agent: Optional[str] = Field(default=None, description="agent_id of assignee")
    status: Literal["pending", "in_progress", "blocked", "completed", "failed"] = Field(default="pending")
    deadline_steps: int = Field(..., ge=0, description="Steps remaining until deadline")


class ResourcePool(BaseModel):
    """Shared enterprise resource pool."""

    engineers_available: int = Field(..., ge=0)
    budget_remaining: float = Field(..., ge=0.0, description="Remaining budget in USD")
    compute_units: int = Field(..., ge=0, description="Available compute units")


# ---------------------------------------------------------------------------
# Agent communication schemas
# ---------------------------------------------------------------------------

class MessageSchema(BaseModel):
    """Inter-agent message payload."""

    from_agent: str = Field(..., description="Sender agent_id")
    to_agent: str = Field(..., description="Recipient agent_id or 'broadcast'")
    content: str = Field(..., description="Message body")
    step_sent: int = Field(..., ge=0)
    priority: int = Field(default=2, ge=1, le=3, description="1=urgent, 2=normal, 3=low")


class ActionHistoryItem(BaseModel):
    step: int
    agent_id: str
    tool_call: Optional[str] = None
    tool_params: dict = Field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    reward_delta: float = 0.0


# ---------------------------------------------------------------------------
# Observation & action schemas — INTERFACE STABLE (Anurag imports these)
# ---------------------------------------------------------------------------

class ObservationSchema(BaseModel):
    """
    Per-agent observation. Each agent receives a partial view filtered by dept.
    INTERFACE STABLE — Anurag's agents consume this as input.
    """

    agent_id: str
    inbox: list[MessageSchema] = Field(default_factory=list, description="Messages delivered this step")
    tickets: list[TicketItem] = Field(default_factory=list, description="Visible tickets (IT agent only)")
    resource_pool: Optional[ResourcePool] = Field(default=None)
    active_deals: list[DealItem] = Field(default_factory=list, description="Visible deals (Finance/Manager)")
    project_tasks: list[ProjectTask] = Field(default_factory=list)
    step_number: int = Field(..., ge=0)
    schema_version: int = Field(default=1, description="Current drift version — agents must track this")
    recent_history: list[ActionHistoryItem] = Field(
        default_factory=list,
        description="Last 5 actions by this agent",
    )


class ActionSchema(BaseModel):
    """
    Agent action for one step. Agent may call a tool, send a message, or both.
    INTERFACE STABLE — Anurag's agents return this; Ayush's trainer wraps it.
    """

    tool_call: Optional[str] = Field(default=None, description="Tool name to invoke, e.g. 'get_tickets'")
    tool_params: dict[str, Any] = Field(default_factory=dict, description="Kwargs passed to the tool")
    message_to: Optional[str] = Field(default=None, description="agent_id to send a message to")
    message_content: Optional[str] = Field(default=None, description="Message body to send")
    reasoning: Optional[str] = Field(default=None, description="Chain-of-thought (logged but not executed)")

    @field_validator("tool_params", mode="before")
    @classmethod
    def ensure_dict(cls, v: Any) -> dict:
        return v if isinstance(v, dict) else {}


# ---------------------------------------------------------------------------
# Reward schema — INTERFACE STABLE (Ayush's reward_fn returns this)
# ---------------------------------------------------------------------------

class RewardComponents(BaseModel):
    """
    Decomposed reward signal for one agent in one step.
    INTERFACE STABLE — Ayush's compute_reward() must return this per agent.
    """

    task_completion: float = Field(default=0.0, description="+reward when tasks/tickets are completed")
    sla_adherence: float = Field(default=0.0, description="+reward for resolving tickets before SLA breach")
    coordination_bonus: float = Field(default=0.0, description="+reward for effective inter-agent messaging")
    schema_adaptation: float = Field(default=0.0, description="+5 when agent correctly uses post-drift field")
    sla_breach_penalty: float = Field(default=0.0, description="Negative: ticket breached SLA")
    hallucination_penalty: float = Field(default=0.0, description="Negative: agent called non-existent field")
    oversight_detection: float = Field(default=0.0, description="OversightAgent reward delta from flags")

    def total(self) -> float:
        """Sum all reward components into a scalar."""
        return (
            self.task_completion
            + self.sla_adherence
            + self.coordination_bonus
            + self.schema_adaptation
            + self.sla_breach_penalty
            + self.hallucination_penalty
            + self.oversight_detection
        )


# ---------------------------------------------------------------------------
# Step result — INTERFACE STABLE (env.step() returns this)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Return value of EnterpriseOpsEnv.step().
    INTERFACE STABLE — both Ayush (training loop) and Anurag (agent eval) consume this.
    """

    observations: dict[str, ObservationSchema] = Field(
        ..., description="Per-agent observations after this step"
    )
    rewards: dict[str, RewardComponents] = Field(
        ..., description="Per-agent decomposed reward for this step"
    )
    done: bool = Field(..., description="True if episode has ended")
    info: dict[str, Any] = Field(default_factory=dict, description="Debug/diagnostic info")


# ---------------------------------------------------------------------------
# Agent ID constants — canonical identifiers used by env and all agents
# ---------------------------------------------------------------------------

AGENT_IT          = "it_agent"
AGENT_MANAGER     = "manager_agent"
AGENT_FINANCE     = "finance_agent"
AGENT_OVERSIGHT   = "oversight_agent"

AGENT_IT_TACTICAL  = "it_tactical_agent"
AGENT_IT_STRATEGIC = "it_strategic_agent"

ALL_AGENTS = [
    AGENT_IT_TACTICAL,
    AGENT_IT_STRATEGIC,
    AGENT_MANAGER,
    AGENT_FINANCE,
    AGENT_OVERSIGHT,
]
