"""
environment.py - OpenEnv wrapper around EnterpriseOpsEnv.

Imports the existing env/env.py class and exposes it via the OpenEnv
Environment interface (reset, step, state).

Anti-reward-hacking protections
--------------------------------
1. Timeout:       step > 30 s  -> done=True, reward = -10
2. Loop detection: same action 3x consecutively -> reward -= 5
3. Max-steps cap:  episode ends at step 8 regardless
4. State lock:     WorldModel is read-only outside of step()
5. Audit log:      every step logged to episodes.db
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

# -- Ensure enterprise_ops/ root is importable ------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# -- OpenEnv base -----------------------------------------------------------
try:
    from openenv.core import Environment
except ImportError:
    class Environment:
        """Stub when openenv-core is not installed."""
        def _reset_rubric(self) -> None:
            pass

# -- Existing enterprise_ops imports (flat, relative to _ROOT) --------------
from contracts import (
    ActionSchema,
    ObservationSchema,
    RewardComponents,
    StepResult as InnerStepResult,
    AGENT_IT,
    AGENT_MANAGER,
    AGENT_FINANCE,
    AGENT_OVERSIGHT,
)
from env.env import EnterpriseOpsEnv
from agents import ITAgent, ManagerAgent, FinanceAgent
from models import EnterpriseAction, EnterpriseObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_AGENTS = [AGENT_IT, AGENT_MANAGER, AGENT_FINANCE, AGENT_OVERSIGHT]

STEP_TIMEOUT_S = 30.0
LOOP_WINDOW = 3
LOOP_PENALTY = -5.0
MAX_STEPS_HARD_CAP = 8
TIMEOUT_PENALTY = -10.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action_fingerprint(action: EnterpriseAction) -> str:
    """Deterministic hash of an action for loop detection."""
    raw = (
        f"{action.agent_id}|{action.tool_call}|"
        f"{json.dumps(action.tool_params, sort_keys=True, default=str)}|"
        f"{action.message_to}"
    )
    return hashlib.md5(raw.encode()).hexdigest()


def _obs_to_openenv(
    obs: ObservationSchema,
    reward_breakdown: dict[str, float],
    reward: float,
    done: bool,
) -> EnterpriseObservation:
    """Convert a Pydantic ObservationSchema -> dataclass EnterpriseObservation."""
    return EnterpriseObservation(
        agent_id=obs.agent_id,
        inbox=[m.model_dump() for m in obs.inbox],
        tickets=[t.model_dump() for t in obs.tickets],
        resource_pool=obs.resource_pool.model_dump() if obs.resource_pool else None,
        active_deals=[d.model_dump() for d in obs.active_deals],
        project_tasks=[t.model_dump() for t in obs.project_tasks],
        step_number=obs.step_number,
        schema_version=obs.schema_version,
        reward_breakdown=reward_breakdown,
        reward=reward,
        done=done,
    )


# ---------------------------------------------------------------------------
# Default reward function (Ayush overrides via env.reward_fn = custom_fn)
# ---------------------------------------------------------------------------


def default_reward_fn(
    step_result: InnerStepResult,
    actions: dict[str, ActionSchema],
) -> float:
    """Sum of all agents' RewardComponents.total()."""
    return sum(r.total() for r in step_result.rewards.values())


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


def _ensure_audit_table(db_path: str) -> None:
    """Create the audit table if it does not exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS step_audit (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                event       TEXT    NOT NULL,
                step        INTEGER,
                agent_id    TEXT,
                action_json TEXT,
                reward      REAL,
                done        INTEGER,
                info_json   TEXT
            )
        """)
        conn.commit()


def _log_audit(
    db_path: str,
    event: str,
    step: int = 0,
    agent_id: str = "",
    action_json: str = "{}",
    reward: float = 0.0,
    done: bool = False,
    info_json: str = "{}",
) -> None:
    """Write one row to the step_audit table."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO step_audit "
                "(timestamp, event, step, agent_id, action_json, reward, done, info_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), event, step, agent_id, action_json, reward, int(done), info_json),
            )
            conn.commit()
    except Exception:
        pass  # audit must never crash the env


# ===========================================================================
# EnterpriseEnvironment - OpenEnv wrapper
# ===========================================================================


class EnterpriseEnvironment(Environment):
    """
    OpenEnv-compatible wrapper around the existing EnterpriseOpsEnv.

    Usage::

        env = EnterpriseEnvironment()
        obs = env.reset()
        result = env.step(EnterpriseAction(agent_id="it_agent", tool_call="get_tickets"))

    For multi-agent training, call ``get_all_observations()`` after each
    ``step()`` to retrieve every agent's partial view.

    Anti-reward-hacking:
      - 30 s timeout per step
      - 3x identical-action loop -> -5 reward
      - Hard cap at 8 steps
      - All steps audited in episodes.db
    """

    def __init__(
        self,
        scenario_path: Optional[str] = None,
        seed: int = 42,
        max_steps: int = MAX_STEPS_HARD_CAP,
        noise_rate: float = 0.08,
        db_path: str = "episodes.db",
    ) -> None:
        super().__init__()

        if scenario_path is None:
            scenario_path = str(_ROOT / "env" / "scenarios" / "scenario_01.yaml")

        self.inner_env = EnterpriseOpsEnv(
            scenario_path=scenario_path,
            seed=seed,
            max_steps=max_steps,
            noise_rate=noise_rate,
            db_path=db_path,
        )

        # -- Pluggable reward - INTERFACE STABLE (Ayush overrides this) -----
        self.reward_fn: Callable[..., float] = default_reward_fn

        # -- Rule-based fallback agents for untrained roles -----------------
        self._fallback_agents: dict[str, Any] = {
            AGENT_IT: ITAgent(AGENT_IT),
            AGENT_MANAGER: ManagerAgent(AGENT_MANAGER),
            AGENT_FINANCE: FinanceAgent(AGENT_FINANCE),
        }

        # -- Anti-hacking state ---------------------------------------------
        self._action_history: deque[str] = deque(maxlen=20)
        self._step_count: int = 0
        self._max_steps: int = max_steps
        self._done: bool = False
        self._db_path: str = db_path

        # -- Observation cache ----------------------------------------------
        self._current_obs: dict[str, ObservationSchema] = {}
        self._last_step_result: Optional[InnerStepResult] = None

        _ensure_audit_table(db_path)
        print(
            f"[EnterpriseEnvironment] Wrapper ready | "
            f"scenario={Path(scenario_path).stem} | "
            f"max_steps={max_steps} | db={db_path}"
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnterpriseObservation:
        """Reset the environment. Returns manager-agent view as primary obs."""
        kwargs: dict[str, Any] = {}
        if scenario:
            if not scenario.endswith(".yaml"):
                scenario = str(_ROOT / "env" / "scenarios" / f"{scenario}.yaml")
            kwargs["scenario"] = scenario
        if seed is not None:
            kwargs["seed"] = seed

        self._current_obs = self.inner_env.reset(**kwargs)
        self._action_history.clear()
        self._step_count = 0
        self._done = False
        self._last_step_result = None

        _log_audit(self._db_path, "reset", info_json=json.dumps(kwargs, default=str))

        return _obs_to_openenv(
            self._current_obs[AGENT_MANAGER],
            reward_breakdown={},
            reward=0.0,
            done=False,
        )

    # ------------------------------------------------------------------
    # step (single-agent primary interface)
    # ------------------------------------------------------------------

    def step(self, action: EnterpriseAction) -> dict[str, Any]:
        """
        Execute one environment step.

        The provided ``action`` controls one agent; all other agents use
        their rule-based fallback policies.

        Returns
        -------
        dict with keys: observation, reward, done, info
        """

        # -- 1. Hard cap ----------------------------------------------------
        if self._done or self._step_count >= self._max_steps:
            self._done = True
            fallback_obs = next(iter(self._current_obs.values()))
            obs = _obs_to_openenv(fallback_obs, {}, 0.0, True)
            return {"observation": obs, "reward": 0.0, "done": True,
                    "info": {"reason": "max_steps_exceeded"}}

        start_time = time.time()
        reward_penalty = 0.0

        # -- 2. Loop detection ----------------------------------------------
        fp = _action_fingerprint(action)
        self._action_history.append(fp)
        if len(self._action_history) >= LOOP_WINDOW:
            tail = list(self._action_history)[-LOOP_WINDOW:]
            if len(set(tail)) == 1:
                reward_penalty += LOOP_PENALTY

        # -- 3. Build full action dict --------------------------------------
        primary = ActionSchema(
            tool_call=action.tool_call,
            tool_params=action.tool_params or {},
            message_to=action.message_to,
            message_content=action.message_content,
            reasoning=action.reasoning,
        )

        all_actions: dict[str, ActionSchema] = {}
        for aid in ALL_AGENTS:
            if aid == action.agent_id:
                all_actions[aid] = primary
            elif aid == AGENT_OVERSIGHT:
                all_actions[aid] = ActionSchema()
            elif aid in self._fallback_agents and aid in self._current_obs:
                all_actions[aid] = self._fallback_agents[aid].act(self._current_obs[aid])
            else:
                all_actions[aid] = ActionSchema()

        # -- 4. Execute inner step ------------------------------------------
        inner_result: InnerStepResult = self.inner_env.step(all_actions)
        elapsed = time.time() - start_time

        # -- 5. Timeout check ----------------------------------------------
        if elapsed > STEP_TIMEOUT_S:
            self._done = True
            obs = _obs_to_openenv(
                inner_result.observations.get(AGENT_MANAGER, next(iter(inner_result.observations.values()))),
                {}, TIMEOUT_PENALTY, True,
            )
            _log_audit(self._db_path, "timeout", self._step_count,
                       action.agent_id, json.dumps(action.to_dict(), default=str),
                       TIMEOUT_PENALTY, True, json.dumps({"elapsed": elapsed}))
            return {"observation": obs, "reward": TIMEOUT_PENALTY, "done": True,
                    "info": {"reason": "timeout", "elapsed_s": elapsed}}

        # -- 6. Compute reward ----------------------------------------------
        base_reward = self.reward_fn(inner_result, all_actions)
        total_reward = base_reward + reward_penalty

        # -- 7. Build reward breakdown --------------------------------------
        breakdown: dict[str, float] = {}
        for aid, rc in inner_result.rewards.items():
            breakdown[f"{aid}_total"] = rc.total()
            breakdown[f"{aid}_task"] = rc.task_completion
            breakdown[f"{aid}_sla"] = rc.sla_adherence
            breakdown[f"{aid}_coord"] = rc.coordination_bonus
            breakdown[f"{aid}_drift"] = rc.schema_adaptation
            breakdown[f"{aid}_breach"] = rc.sla_breach_penalty
            breakdown[f"{aid}_halluc"] = rc.hallucination_penalty
            breakdown[f"{aid}_oversight"] = rc.oversight_detection
        if reward_penalty != 0.0:
            breakdown["loop_penalty"] = reward_penalty

        # -- 8. Update state -----------------------------------------------
        self._current_obs = inner_result.observations
        self._last_step_result = inner_result
        self._step_count += 1
        self._done = inner_result.done or self._step_count >= self._max_steps

        # -- 9. Build primary observation ----------------------------------
        primary_agent = action.agent_id if action.agent_id in inner_result.observations else AGENT_MANAGER
        obs = _obs_to_openenv(
            inner_result.observations[primary_agent],
            breakdown, total_reward, self._done,
        )

        # -- 10. Audit log -------------------------------------------------
        _log_audit(
            self._db_path, "step", self._step_count, action.agent_id,
            json.dumps(action.to_dict(), default=str), total_reward, self._done,
            json.dumps({
                "elapsed_s": elapsed,
                "loop_penalty": reward_penalty,
                "oversight_flags": inner_result.info.get("oversight_flags", []),
                "schema_version": inner_result.info.get("schema_version", 1),
            }, default=str),
        )

        return {
            "observation": obs,
            "reward": total_reward,
            "done": self._done,
            "info": {
                "step": self._step_count,
                "elapsed_s": elapsed,
                "loop_penalty": reward_penalty,
                "oversight_flags": inner_result.info.get("oversight_flags", []),
                "schema_version": inner_result.info.get("schema_version", 1),
                "tool_results": inner_result.info.get("tool_results", {}),
            },
        }

    # ------------------------------------------------------------------
    # step_multi - all agents in one call (for multi-agent training)
    # ------------------------------------------------------------------

    def step_multi(
        self, actions: dict[str, EnterpriseAction],
    ) -> dict[str, Any]:
        """
        Execute one step with explicit actions for multiple agents.

        Agents not in ``actions`` use their fallback rule policy.

        Returns dict with: observations (per-agent), reward, done, info
        """
        if self._done or self._step_count >= self._max_steps:
            self._done = True
            return {"observations": {}, "reward": 0.0, "done": True,
                    "info": {"reason": "max_steps_exceeded"}}

        start_time = time.time()

        # Build full action dict
        all_actions: dict[str, ActionSchema] = {}
        for aid in ALL_AGENTS:
            if aid in actions:
                a = actions[aid]
                all_actions[aid] = ActionSchema(
                    tool_call=a.tool_call,
                    tool_params=a.tool_params or {},
                    message_to=a.message_to,
                    message_content=a.message_content,
                    reasoning=a.reasoning,
                )
            elif aid == AGENT_OVERSIGHT:
                all_actions[aid] = ActionSchema()
            elif aid in self._fallback_agents and aid in self._current_obs:
                all_actions[aid] = self._fallback_agents[aid].act(self._current_obs[aid])
            else:
                all_actions[aid] = ActionSchema()

        inner_result = self.inner_env.step(all_actions)
        elapsed = time.time() - start_time

        if elapsed > STEP_TIMEOUT_S:
            self._done = True
            return {"observations": {}, "reward": TIMEOUT_PENALTY, "done": True,
                    "info": {"reason": "timeout", "elapsed_s": elapsed}}

        base_reward = self.reward_fn(inner_result, all_actions)
        self._current_obs = inner_result.observations
        self._last_step_result = inner_result
        self._step_count += 1
        self._done = inner_result.done or self._step_count >= self._max_steps

        per_agent_obs = {
            aid: _obs_to_openenv(
                obs, {k: v for k, v in inner_result.rewards.get(aid, RewardComponents()).model_dump().items()},
                inner_result.rewards.get(aid, RewardComponents()).total(),
                self._done,
            )
            for aid, obs in inner_result.observations.items()
        }

        return {
            "observations": per_agent_obs,
            "reward": base_reward,
            "done": self._done,
            "info": {
                "step": self._step_count,
                "elapsed_s": elapsed,
                "schema_version": inner_result.info.get("schema_version", 1),
            },
        }

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Full world state snapshot (read-only view)."""
        raw = self.inner_env.state
        raw["wrapper_step_count"] = self._step_count
        raw["wrapper_done"] = self._done
        return raw

    # ------------------------------------------------------------------
    # get_all_observations - multi-agent partial views
    # ------------------------------------------------------------------

    def get_all_observations(self) -> dict[str, EnterpriseObservation]:
        """Return the latest per-agent EnterpriseObservation dict."""
        result: dict[str, EnterpriseObservation] = {}
        for aid, obs in self._current_obs.items():
            reward_total = 0.0
            breakdown: dict[str, float] = {}
            if self._last_step_result and aid in self._last_step_result.rewards:
                rc = self._last_step_result.rewards[aid]
                reward_total = rc.total()
                breakdown = rc.model_dump()
            result[aid] = _obs_to_openenv(obs, breakdown, reward_total, self._done)
        return result

    # ------------------------------------------------------------------
    # set_scenario - curriculum switching
    # ------------------------------------------------------------------

    def set_scenario(self, scenario_name: str) -> None:
        """
        Switch scenario for the next reset().

        Args:
            scenario_name: e.g. "scenario_03" or full path to YAML.
        """
        if not scenario_name.endswith(".yaml"):
            scenario_name = str(_ROOT / "env" / "scenarios" / f"{scenario_name}.yaml")
        self.inner_env._scenario_path = scenario_name
        print(f"[EnterpriseEnvironment] Scenario set -> {Path(scenario_name).stem}")

    # ------------------------------------------------------------------
    # per_agent_rewards - convenience for training loops
    # ------------------------------------------------------------------

    def per_agent_rewards(self) -> dict[str, float]:
        """Return per-agent scalar rewards from the last step."""
        if not self._last_step_result:
            return {aid: 0.0 for aid in ALL_AGENTS}
        return {
            aid: rc.total()
            for aid, rc in self._last_step_result.rewards.items()
        }
