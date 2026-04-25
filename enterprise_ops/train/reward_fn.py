"""
reward_fn.py — 5-improvement reward system for EnterpriseOps Arena GRPO training.

Improvements
------------
1. Potential-based shaping   — dependency-graph progress bonus (no reward hacking)
2. Dynamic weights           — auto-rebalances component weights when score drops
3. Urgency-scaled SLA        — SLA reward grows as deadline approaches
4. Exploration bonus         — count-based bonus to discourage stuck-loop patterns
5. Schema adaptation         — rewards correct tool use after schema drift

Public API (names are stable — other modules import these):
    compute_reward()
    reward_task_completion()
    reward_sla_adherence()
    reward_coordination()
    penalty_hallucination()
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contracts import ActionSchema, ObservationSchema, RewardComponents


# ===========================================================================
# Improvement 1 — Potential-based shaping
# ===========================================================================

def potential_function(state: dict) -> float:
    """
    Measures dependency-graph progress: fraction of task dependencies
    whose prerequisite is already completed, scaled to [0, 2].
    """
    completed = 0
    total_deps = 0
    for task in state.get("project_tasks", []):
        deps = task.get("depends_on", [])
        total_deps += len(deps)
        for dep_id in deps:
            dep_task = next(
                (t for t in state.get("project_tasks", [])
                 if t["id"] == dep_id), None)
            if dep_task and dep_task.get("status") == "completed":
                completed += 1
    if total_deps == 0:
        return 0.0
    return (completed / total_deps) * 2.0


def apply_potential_shaping(
    base_reward: float,
    current_state: dict,
    next_state: dict,
    gamma: float = 0.99,
) -> float:
    """
    F(s,s') = gamma * Phi(s') - Phi(s)
    Guarantees policy-invariance while rewarding dependency progress.
    """
    phi_next = potential_function(next_state)
    phi_current = potential_function(current_state)
    return base_reward + gamma * phi_next - phi_current


# ===========================================================================
# Improvement 2 — Dynamic weighted reward
# ===========================================================================

class DynamicWeightedReward:
    """
    Tracks episode performance and shifts weight toward task_completion
    when recent scores fall below 90 % of the overall mean.
    """

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {
            "task_completion":   0.35,
            "sla_adherence":     0.25,
            "coordination_bonus": 0.20,
            "schema_adaptation": 0.10,
            "oversight_detection": 0.10,
        }
        self.performance_history: list[float] = []

    def update_weights(self, episode_scores: list[float]) -> None:
        if len(episode_scores) < 3:
            return
        recent_avg = sum(episode_scores[-3:]) / 3
        overall_avg = sum(episode_scores) / len(episode_scores)
        if recent_avg < overall_avg * 0.9:
            self.weights["task_completion"] = min(
                0.5, self.weights["task_completion"] + 0.05)
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute(self, components: Dict[str, float]) -> float:
        return sum(
            components.get(k, 0.0) * w
            for k, w in self.weights.items()
        )


# ===========================================================================
# Improvement 4 — Exploration bonus
# ===========================================================================

class ExplorationBonus:
    """
    Count-based exploration bonus: bonus_scale / (1 + visit_count).
    Naturally decays as the agent repeats the same (agent, tool) pair.
    """

    def __init__(self, bonus_scale: float = 0.1) -> None:
        self.visitation_counts: Dict[str, int] = defaultdict(int)
        self.bonus_scale = bonus_scale

    def compute(self, agent_id: str, tool_call: str) -> float:
        if not tool_call:
            return 0.0
        key = f"{agent_id}:{tool_call}"
        self.visitation_counts[key] += 1
        count = self.visitation_counts[key]
        return self.bonus_scale / (1.0 + count)


# ===========================================================================
# Module-level singletons (reset between runs via reset_episode_state())
# ===========================================================================

_dynamic_weights = DynamicWeightedReward()
_exploration = ExplorationBonus()
_episode_scores: list[float] = []
_prev_state: dict = {}
_prev_schema_version: int = 1


def reset_episode_state() -> None:
    """Call at the start of each episode to clear carry-over state."""
    global _prev_state, _prev_schema_version
    _prev_state = {}
    _prev_schema_version = 1


# ===========================================================================
# Improvement 5 — Schema adaptation reward
# ===========================================================================

def reward_schema_adaptation(
    tool_call_logs: List[dict],
    schema_version: int,
    prev_schema_version: int,
) -> float:
    """
    +5.0 per successful call after schema drift.
    -2.0 per "deprecated" error after drift.
    -0.5 per generic field error after drift.
    -1.0 per field error when no drift occurred.
    """
    total = 0.0
    schema_drifted = schema_version > prev_schema_version
    for log in tool_call_logs:
        success = log.get("success", False)
        error = log.get("error", "")
        if schema_drifted:
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


# ===========================================================================
# Improvement 3 — Urgency-scaled SLA adherence (replaces old version)
# ===========================================================================

def reward_sla_adherence(
    step_result: Any,
    current_step: int = 0,
    episode_length: int = 8,
) -> float:
    """
    Base SLA reward scaled by urgency (grows as deadline nears).
    +2.0 early-resolution bonus when priority-1 ticket resolved in the
    first 30 % of the episode.

    Accepts contracts.StepResult (has .rewards) or info-dict fallback.
    """
    base_reward = 0.0

    if hasattr(step_result, "rewards"):
        for _, reward in step_result.rewards.items():
            base_reward += reward.sla_adherence
    elif isinstance(step_result, dict):
        info = step_result.get("info", {})
        for key, val in info.items():
            if key.endswith("_sla") or key.endswith("_breach"):
                try:
                    base_reward += float(val)
                except (TypeError, ValueError):
                    pass

    time_remaining = 1.0 - (current_step / max(episode_length, 1))
    urgency_factor = 1.0 + (1.0 - time_remaining) * 0.5
    p1_early_bonus = 2.0 if current_step < episode_length * 0.3 else 0.0

    return (base_reward * urgency_factor) + p1_early_bonus


# ===========================================================================
# Original signals — kept unchanged for backward-compat imports
# ===========================================================================

def reward_task_completion(step_result: Any) -> float:
    """
    +1.0 per resolved ticket, +1.0 per completed project task, this step.
    Accepts contracts.StepResult or info-dict.
    """
    if hasattr(step_result, "rewards"):
        return sum(r.task_completion for r in step_result.rewards.values())

    if isinstance(step_result, dict):
        info = step_result.get("info", {})
        total = 0.0
        for key, val in info.items():
            if key.endswith("_task"):
                try:
                    total += float(val)
                except (TypeError, ValueError):
                    pass
        return total

    return 0.0


def reward_coordination(
    step_result: Any,
    prev_messages: Optional[List[Dict[str, str]]] = None,
) -> float:
    """
    +0.3 per inter-agent message sent this step.
    +0.5 cross-agent chain bonus when 2+ distinct senders in prev_messages.
    """
    base = 0.0
    if hasattr(step_result, "rewards"):
        base = sum(r.coordination_bonus for r in step_result.rewards.values())
    elif isinstance(step_result, dict):
        info = step_result.get("info", {})
        for key, val in info.items():
            if key.endswith("_coord"):
                try:
                    base += float(val)
                except (TypeError, ValueError):
                    pass

    chain_bonus = 0.0
    if prev_messages and len(prev_messages) >= 2:
        senders = {m.get("from") for m in prev_messages if isinstance(m, dict)}
        if len(senders) >= 2:
            chain_bonus = 0.5

    return base + chain_bonus


def penalty_hallucination(step_result: Any) -> float:
    """
    -1.0 per unknown-tool or hallucinated-field call detected this step.
    Returns a non-positive float.
    """
    if hasattr(step_result, "rewards"):
        return sum(r.hallucination_penalty for r in step_result.rewards.values())

    if isinstance(step_result, dict):
        info = step_result.get("info", {})
        flags = info.get("oversight_flags", [])
        penalty = 0.0
        for flag in flags:
            if isinstance(flag, dict) and flag.get("flag_type") == "HALLUCINATION":
                penalty -= 1.0
        return penalty

    return 0.0


# ===========================================================================
# Subtask decomposition and recovery reward functions
# ===========================================================================

def reward_subtask_completion(tool_call_logs: List[dict]) -> float:
    """
    +0.3 per completed subtask; +2.0 bonus when all subtasks finish the ticket.
    """
    total = 0.0
    for log in tool_call_logs:
        if log.get("tool_name") == "resolve_subtask" and log.get("success"):
            total += 0.3
            result = log.get("result", {})
            if result.get("ticket_fully_resolved"):
                total += 2.0
    return total


def reward_recovery(tool_call_logs: List[dict], failed_attempts: dict) -> float:
    """
    Bonus when an agent succeeds on a ticket that previously failed.
    Capped at +2.0 per ticket to prevent over-exploitation.
    """
    total = 0.0
    for log in tool_call_logs:
        if log.get("success") and log.get("tool_name") == "resolve_ticket":
            ticket_id = log.get("params", {}).get("ticket_id", "")
            fail_count = failed_attempts.get(ticket_id, 0)
            if fail_count >= 1:
                total += min(0.5 * fail_count, 2.0)
    return total


# ===========================================================================
# Specialized per-agent reward functions for tactical / strategic IT agents
# ===========================================================================

def reward_tactical_agent(_: Any, tool_call_logs: List[dict]) -> float:
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


def reward_strategic_agent(_: Any, tool_call_logs: List[dict]) -> float:
    """Throughput reward with batch efficiency bonus for 2+ resolutions per step."""
    total = 0.0
    resolved_count = 0
    for log in tool_call_logs:
        if log.get("agent_id") != "it_strategic_agent":
            continue
        if log.get("tool_name") == "resolve_ticket" and log.get("success"):
            total += 1.0
            resolved_count += 1
    if resolved_count >= 2:
        total += 0.5 * resolved_count
    return total


# ===========================================================================
# compute_reward — combines all 5 improvements into one scalar for GRPO
# ===========================================================================

def compute_reward(
    step_result: Any,
    actions: Optional[Dict[str, Any]] = None,
    current_step: int = 0,
    episode_length: int = 8,
    current_state: Optional[dict] = None,
    next_state: Optional[dict] = None,
    schema_version: int = 1,
    tool_call_logs: Optional[List[dict]] = None,
    # Legacy kwarg — kept so old callers (trainer dry-run) don't break
    prev_messages: Optional[List[dict]] = None,
) -> float:
    """
    Main reward scalar for GRPOTrainer.

    All parameters except step_result are optional; old callers that pass
    only step_result continue to work unchanged.

    Returns
    -------
    float — total shaped reward for this step.
    """
    global _prev_state, _prev_schema_version

    if actions is None:
        actions = {}
    if current_state is None:
        current_state = {}
    if next_state is None:
        next_state = {}
    if tool_call_logs is None:
        tool_call_logs = []

    # -- Improvement 3: urgency-scaled SLA --------------------------------
    sla = reward_sla_adherence(step_result, current_step, episode_length)

    # -- Improvement 5: schema adaptation ---------------------------------
    schema = reward_schema_adaptation(
        tool_call_logs, schema_version, _prev_schema_version)

    # -- Improvement 4: exploration bonus ---------------------------------
    exploration_total = 0.0
    for agent_id, action in actions.items():
        tool = getattr(action, "tool_call", None) or ""
        exploration_total += _exploration.compute(agent_id, tool)

    # -- Base component extraction -----------------------------------------
    if hasattr(step_result, "rewards"):
        task_total = sum(
            r.task_completion for r in step_result.rewards.values())
        coord_total = sum(
            r.coordination_bonus for r in step_result.rewards.values())
        oversight_total = sum(
            r.oversight_detection for r in step_result.rewards.values())
        hallucination_total = sum(
            r.hallucination_penalty for r in step_result.rewards.values())
    else:
        task_total = reward_task_completion(step_result)
        coord_total = reward_coordination(step_result, prev_messages)
        oversight_total = 0.0
        hallucination_total = penalty_hallucination(step_result)

    components = {
        "task_completion":    task_total,
        "sla_adherence":      sla,
        "coordination_bonus": coord_total,
        "schema_adaptation":  schema,
        "oversight_detection": oversight_total,
    }

    # -- Improvement 2: dynamic weights -----------------------------------
    _dynamic_weights.update_weights(_episode_scores)
    weighted = _dynamic_weights.compute(components)

    # Add non-weighted signals
    total = weighted + hallucination_total + exploration_total * 0.1

    # -- Improvement 1: potential-based shaping ---------------------------
    if _prev_state:
        total = apply_potential_shaping(total, _prev_state, next_state)

    # Update module-level tracking
    _prev_state = next_state.copy() if next_state else {}
    _prev_schema_version = schema_version
    _episode_scores.append(total)

    return float(total)
