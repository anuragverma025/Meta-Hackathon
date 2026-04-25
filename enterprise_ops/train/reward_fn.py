"""
reward_fn.py — Research-paper-grounded reward system for EnterpriseOps Arena GRPO training.

Improvements (from paper findings)
------------------------------------
1. Potential-based shaping   — dependency-graph progress bonus (policy-invariant)
2. Dynamic weights           — BiPaRS-style auto-rebalancing when score drops
3. Urgency-scaled SLA        — time-dependent potential for deadline-sensitive tasks
4. Exploration bonus         — EXPLORS count-based intrinsic reward
5. Schema adaptation         — explicit post-drift field-usage reward
6. PRM-based process reward  — step-level supervision for credit assignment
7. Trajectory-level reward   — trend + consistency bonus over episode
8. KL regularization         — limits policy drift from base model

Public API (stable — trainer.py and other modules import these):
    compute_reward()
    reward_sla_adherence()
    reward_schema_adaptation()
    reward_subtask_completion()
    reward_recovery()
    reset_episode_state()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

# ── Module-level state ──────────────────────────────────────────────────────

_episode_scores: list[float] = []
_prev_state: dict = {}
_prev_schema_version: int = 1
_failed_attempts: dict[str, int] = {}
_visitation_counts: dict[str, int] = defaultdict(int)
_dynamic_weights: dict[str, float] = {
    "task_completion":    0.35,
    "sla_adherence":      0.25,
    "coordination_bonus": 0.20,
    "schema_adaptation":  0.10,
    "oversight_detection": 0.10,
}


def reset_episode_state() -> None:
    """Call at the start of each episode to clear carry-over state."""
    global _prev_state, _prev_schema_version, _failed_attempts
    _prev_state = {}
    _prev_schema_version = 1
    _failed_attempts = {}


# ── IMPROVEMENT 1: Potential-based shaping ─────────────────────────────────
# From paper: "potential-based shaping significantly accelerates convergence
# for dependency-heavy tasks"

def potential_function(state: dict) -> float:
    completed = 0
    total_deps = 0
    for task in state.get("project_tasks", []):
        deps = (
            task.get("depends_on", [])
            if isinstance(task, dict)
            else getattr(task, "depends_on", [])
        )
        total_deps += len(deps)
        for dep_id in deps:
            all_tasks = state.get("project_tasks", [])
            dep = next(
                (
                    t for t in all_tasks
                    if (t.get("id") if isinstance(t, dict)
                        else getattr(t, "id", "")) == dep_id
                ),
                None,
            )
            if dep:
                status = (
                    dep.get("status") if isinstance(dep, dict)
                    else getattr(dep, "status", "")
                )
                if status == "completed":
                    completed += 1
    return (completed / total_deps * 2.0) if total_deps > 0 else 0.0


def apply_potential_shaping(
    base_reward: float,
    current_state: dict,
    next_state: dict,
    gamma: float = 0.99,
) -> float:
    """F(s,s') = gamma * Phi(s') - Phi(s) — policy-invariant shaping."""
    return (
        base_reward
        + gamma * potential_function(next_state)
        - potential_function(current_state)
    )


# ── IMPROVEMENT 2: Dynamic weights ─────────────────────────────────────────
# From paper: "BiPaRS learns optimal weight allocation rather than using
# fixed weights"

def _update_weights() -> None:
    global _dynamic_weights
    if len(_episode_scores) < 3:
        return
    recent = sum(_episode_scores[-3:]) / 3
    overall = sum(_episode_scores) / len(_episode_scores)
    if recent < overall * 0.9:
        _dynamic_weights["task_completion"] = min(
            0.5, _dynamic_weights["task_completion"] + 0.05
        )
        total = sum(_dynamic_weights.values())
        _dynamic_weights = {k: v / total for k, v in _dynamic_weights.items()}


def _weighted_sum(components: dict[str, float]) -> float:
    return sum(components.get(k, 0.0) * w for k, w in _dynamic_weights.items())


# ── IMPROVEMENT 3: Urgency-scaled SLA ──────────────────────────────────────
# From paper: "time-dependent potential functions for deadline-sensitive tasks"

def reward_sla_adherence(
    step_result: Any,
    current_step: int = 0,
    episode_length: int = 8,
) -> float:
    """Base SLA reward scaled by urgency; +2.0 early P1-resolution bonus."""
    base = sum(
        getattr(r, "sla_adherence", 0.0)
        for r in step_result.rewards.values()
    )
    time_used = current_step / max(episode_length, 1)
    urgency = 1.0 + time_used * 0.5
    early_bonus = 2.0 if current_step < episode_length * 0.3 else 0.0
    return base * urgency + early_bonus


# ── IMPROVEMENT 4: Exploration bonus ───────────────────────────────────────
# From paper: "EXPLORS — intrinsic rewards for sparse reward regions"

def _exploration_bonus(
    agent_id: str, tool_call: str, scale: float = 0.1
) -> float:
    if not tool_call:
        return 0.0
    key = f"{agent_id}:{tool_call}"
    _visitation_counts[key] += 1
    return scale / (1.0 + _visitation_counts[key])


# ── IMPROVEMENT 5: Schema adaptation ───────────────────────────────────────
# From paper: "explicit rewards for correct post-drift field usage"

def reward_schema_adaptation(
    tool_call_logs: list,
    schema_version: int,
    prev_schema_version: int,
) -> float:
    """
    +5.0 per successful call after schema drift.
    -2.0 per deprecated-field error after drift.
    -0.5 per generic field error after drift.
    -1.0 per field error when no drift.
    """
    total = 0.0
    drifted = schema_version > prev_schema_version
    for log in tool_call_logs:
        success = log.get("success", False)
        error = log.get("error", "") or ""
        if drifted:
            if success:
                total += 5.0
            elif "deprecated" in error.lower():
                total -= 2.0
            elif "field" in error.lower():
                total -= 0.5
        else:
            if not success and "field" in error.lower():
                total -= 1.0
    return total


# ── FROM PAPER: Process reward (PRM-based) ─────────────────────────────────
# From paper: "PRM-based RL provides intermediate step-level supervision —
# improves credit assignment"

def reward_process(
    tool_call_logs: list,
    current_step: int,
    episode_length: int,
) -> float:
    """Step-level progress bonus + subtask completion signal."""
    total = 0.0
    for log in tool_call_logs:
        if log.get("success"):
            step_progress = current_step / max(episode_length, 1)
            total += 0.2 * (1.0 - step_progress)
        if log.get("tool_name") == "resolve_subtask" and log.get("success"):
            total += 0.3
            if log.get("result", {}).get("ticket_fully_resolved"):
                total += 2.0
    return total


# ── FROM PAPER: Trajectory-level reward ────────────────────────────────────
# From paper: "trajectory-level rewards for structured multi-step sequences"

def reward_trajectory(episode_scores: list) -> float:
    if len(episode_scores) < 3:
        return 0.0
    trend = episode_scores[-1] - episode_scores[0]
    consistency = 1.0 - (
        max(episode_scores) - min(episode_scores)
    ) / (abs(max(episode_scores)) + 1e-6)
    return trend * 0.1 + consistency * 0.05


# ── FROM PAPER: KL regularization ──────────────────────────────────────────
# From paper: "KL regularizer limits policy drift from base model"

def kl_penalty(kl_divergence: float, beta: float = 0.01) -> float:
    return -beta * kl_divergence


# ── Subtask completion (stable named export) ────────────────────────────────

def reward_subtask_completion(tool_call_logs: list) -> float:
    """+0.3 per completed subtask; +2.0 bonus when all subtasks finish the ticket."""
    total = 0.0
    for log in tool_call_logs:
        if log.get("tool_name") == "resolve_subtask" and log.get("success"):
            total += 0.3
            if log.get("result", {}).get("ticket_fully_resolved"):
                total += 2.0
    return total


# ── Recovery reward ─────────────────────────────────────────────────────────

def reward_recovery(tool_call_logs: list, failed_attempts: dict) -> float:
    """Bonus when an agent succeeds on a ticket that previously failed (cap +2.0)."""
    total = 0.0
    for log in tool_call_logs:
        if log.get("success") and log.get("tool_name") == "resolve_ticket":
            tid = log.get("params", {}).get("ticket_id", "")
            fails = failed_attempts.get(tid, 0)
            if fails >= 1:
                total += min(0.5 * fails, 2.0)
    return total


# ── Per-agent type rewards ──────────────────────────────────────────────────

def reward_tactical_agent(tool_call_logs: list) -> float:
    """Higher reward for urgency-driven resolution; bonus for SLA-critical saves."""
    total = 0.0
    for log in tool_call_logs:
        if log.get("agent_id") != "it_tactical_agent":
            continue
        if log.get("tool_name") == "resolve_ticket":
            if log.get("success"):
                total += 1.5
                if log.get("sla_steps_remaining", 99) <= 2:
                    total += 2.0
            else:
                total -= 0.5
    return total


def reward_strategic_agent(tool_call_logs: list) -> float:
    """Throughput reward with batch efficiency bonus for 2+ resolutions per step."""
    total = 0.0
    resolved = 0
    for log in tool_call_logs:
        if log.get("agent_id") != "it_strategic_agent":
            continue
        if log.get("tool_name") == "resolve_ticket" and log.get("success"):
            total += 1.0
            resolved += 1
    if resolved >= 2:
        total += 0.5 * resolved
    return total


# ── MASTER COMPUTE FUNCTION ─────────────────────────────────────────────────
# INTERFACE STABLE — trainer.py calls this exact signature

def compute_reward(
    step_result: Any,
    actions: Optional[dict] = None,
    current_step: int = 0,
    episode_length: int = 8,
    current_state: Optional[dict] = None,
    next_state: Optional[dict] = None,
    schema_version: int = 1,
    tool_call_logs: Optional[list] = None,
    kl_divergence: float = 0.0,
) -> float:
    """
    Main reward scalar for GRPOTrainer.

    All parameters except step_result are optional; old callers that pass
    only step_result continue to work unchanged.

    Returns
    -------
    float — total shaped reward for this step.
    """
    global _prev_state, _prev_schema_version, _failed_attempts

    actions = actions or {}
    current_state = current_state or {}
    next_state = next_state or {}
    tool_call_logs = tool_call_logs or []

    # -- Base components from step result ----------------------------------
    task_total = sum(
        getattr(r, "task_completion", 0.0)
        for r in step_result.rewards.values()
    )
    coord_total = sum(
        getattr(r, "coordination_bonus", 0.0)
        for r in step_result.rewards.values()
    )
    oversight_total = sum(
        getattr(r, "oversight_detection", 0.0)
        for r in step_result.rewards.values()
    )
    hallucination_total = sum(
        getattr(r, "hallucination_penalty", 0.0)
        for r in step_result.rewards.values()
    )

    # -- Improvement 3: urgency-scaled SLA ---------------------------------
    sla = reward_sla_adherence(step_result, current_step, episode_length)

    # -- Improvement 5: schema adaptation ----------------------------------
    schema = reward_schema_adaptation(
        tool_call_logs, schema_version, _prev_schema_version
    )

    # -- PRM-based process reward ------------------------------------------
    process = reward_process(tool_call_logs, current_step, episode_length)

    # -- Per-agent type rewards --------------------------------------------
    tactical = reward_tactical_agent(tool_call_logs)
    strategic = reward_strategic_agent(tool_call_logs)

    # -- Recovery reward ---------------------------------------------------
    recovery = reward_recovery(tool_call_logs, _failed_attempts)

    # Update module-level failed_attempts from this step's logs
    for log in tool_call_logs:
        if not log.get("success") and log.get("tool_name") == "resolve_ticket":
            tid = log.get("params", {}).get("ticket_id", "")
            if tid:
                _failed_attempts[tid] = _failed_attempts.get(tid, 0) + 1

    # -- Improvement 4: exploration bonus ----------------------------------
    exploration = sum(
        _exploration_bonus(aid, getattr(a, "tool_call", "") or "")
        for aid, a in actions.items()
    )

    # -- Improvement 2: dynamic weighted components ------------------------
    _update_weights()
    components = {
        "task_completion":    task_total + tactical + strategic,
        "sla_adherence":      sla,
        "coordination_bonus": coord_total,
        "schema_adaptation":  schema,
        "oversight_detection": oversight_total,
    }
    weighted = _weighted_sum(components)

    # -- Combine all signals -----------------------------------------------
    total = (
        weighted
        + hallucination_total
        + process
        + recovery
        + exploration * 0.1
        + kl_penalty(kl_divergence)
    )

    # -- Improvement 1: potential-based shaping ----------------------------
    if _prev_state:
        total = apply_potential_shaping(total, _prev_state, next_state)

    # -- Trajectory-level reward -------------------------------------------
    total += reward_trajectory(_episode_scores)

    # -- Update module state -----------------------------------------------
    _prev_state = next_state.copy() if next_state else {}
    _prev_schema_version = schema_version
    _episode_scores.append(total)

    return float(total)
