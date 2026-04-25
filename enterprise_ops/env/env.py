"""
env.py — EnterpriseOpsEnv: main RL environment for EnterpriseOps Arena.

Inherits from openenv.core.Environment and wires together:
  WorldModel        → persistent state + causal log
  ToolRegistry      → 5 mock enterprise tools with noise + drift transforms
  SchemaDriftEngine → runtime schema mutations every 20 steps
  OversightAgent    → policy + hallucination checker
  ScenarioLoader    → YAML scenario files

Integration contract
--------------------
Teammates import ONLY contracts.py and call ONLY:
  env.reset()         → dict[agent_id, ObservationSchema]
  env.step(actions)   → StepResult
  env.state           → dict  (full world snapshot)
  env.reward_fn       → Ayush plugs in his custom reward function here
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

from openenv.core import Environment

from contracts import (
    ActionSchema,
    ActionHistoryItem,
    DealItem,
    MessageSchema,
    ObservationSchema,
    ProjectTask,
    RewardComponents,
    ResourcePool,
    StepResult,
    TicketItem,
    AGENT_IT_TACTICAL,
    AGENT_IT_STRATEGIC,
)
from env.schema_drift import SchemaDriftEngine
from env.scenarios.scenario_loader import ScenarioLoader
from env.tools import ToolRegistry
from env.world_model import WorldModel

AGENT_IDS = ["it_agent", "manager_agent", "finance_agent", "oversight_agent"]


class EnterpriseOpsEnv(Environment):
    """
    Multi-agent RL environment for EnterpriseOps Arena.

    Args:
        scenario_path: Default scenario YAML to load on reset().
        seed:          Master RNG seed propagated to all sub-components.
        max_steps:     Hard episode cutoff (overridden by scenario's episode_length).
        noise_rate:    Base tool failure probability (overridden by scenario).
        db_path:       SQLite file for WorldModel + ToolRegistry logs.
    """

    AGENT_IDS = AGENT_IDS

    def __init__(
        self,
        scenario_path: Optional[str] = None,
        seed: int = 42,
        max_steps: int = 50,
        noise_rate: float = 0.08,
        db_path: str = "episodes.db",
    ) -> None:
        super().__init__()
        self._seed = seed
        self._default_max_steps = max_steps
        self._default_noise_rate = noise_rate
        self._db_path = db_path
        self._scenario_path = scenario_path

        self._world_model    = WorldModel(db_path=db_path, seed=seed)
        self._drift_engine   = SchemaDriftEngine(db_path=db_path, seed=seed)
        self._tool_registry  = ToolRegistry(
            world_model=self._world_model, db_path=db_path,
            noise_rate=noise_rate, seed=seed, drift_engine=self._drift_engine,
        )
        self._scenario_loader = ScenarioLoader()

        self._step_count: int = 0
        self._max_steps:  int = max_steps
        self._done:       bool = False
        self._current_scenario: dict[str, Any] = {}

        self._message_bus:      dict[str, list[MessageSchema]] = {aid: [] for aid in AGENT_IDS}
        self._pending_messages: dict[str, list[MessageSchema]] = {aid: [] for aid in AGENT_IDS}

        # INTERFACE STABLE — Ayush plugs his reward function here:
        #   env.reward_fn = fn(agent_id, action, tool_result, world_state) -> RewardComponents
        self.reward_fn: Optional[Callable] = None

        self._oversight_agent: Any = None

        self.action_history: list = []
        self.failed_attempts: dict = {}

        print(f"[EnterpriseOpsEnv] Initialised | seed={seed} | db={db_path}")

    # ------------------------------------------------------------------
    # INTERFACE STABLE — reset
    # ------------------------------------------------------------------

    def reset(                                          # INTERFACE STABLE
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, ObservationSchema]:
        """
        Reset the environment and return initial observations for all agents.

        Args:
            seed:      Override master seed for this episode.
            episode_id: Unused; kept for OpenEnv API compliance.
            scenario:  Path or short name of a scenario YAML.
        """
        self._reset_rubric()

        scenario_path = scenario or self._scenario_path
        if scenario_path is None:
            scenario_path = str(Path(__file__).parent / "scenarios" / "scenario_01.yaml")

        self._current_scenario = self._scenario_loader.load(scenario_path)

        self._max_steps   = self._current_scenario.get("episode_length", self._default_max_steps)
        noise_rate        = self._current_scenario.get("noise_rate", self._default_noise_rate)
        drift_enabled     = self._current_scenario.get("schema_drift_enabled", True)

        if seed is not None:
            self._seed = seed

        self._world_model = WorldModel(db_path=self._db_path, seed=self._seed)
        self._drift_engine = SchemaDriftEngine(
            db_path=self._db_path, seed=self._seed,
            drift_every=20 if drift_enabled else 999999,
        )
        self._tool_registry = ToolRegistry(
            world_model=self._world_model, db_path=self._db_path,
            noise_rate=noise_rate, seed=self._seed, drift_engine=self._drift_engine,
        )

        self._world_model.load_scenario(self._current_scenario)

        self._step_count       = 0
        self._done             = False
        self._message_bus      = {aid: [] for aid in AGENT_IDS}
        self._pending_messages = {aid: [] for aid in AGENT_IDS}
        self.action_history    = []
        self.failed_attempts   = {}

        self._boot_oversight_agent()

        print(
            f"[EnterpriseOpsEnv] Episode reset | "
            f"scenario={Path(scenario_path).stem} | "
            f"difficulty={self._current_scenario.get('difficulty', '?')} | "
            f"max_steps={self._max_steps} | "
            f"noise={noise_rate:.0%} | "
            f"drift={'on' if drift_enabled else 'off'} | "
            f"schema_v={self._drift_engine.schema_version} | "
            f"agents={AGENT_IDS}"
        )

        return {aid: self._get_observation(aid) for aid in AGENT_IDS}

    # ------------------------------------------------------------------
    # INTERFACE STABLE — step
    # ------------------------------------------------------------------

    def step(                                           # INTERFACE STABLE
        self,
        action: dict[str, ActionSchema],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StepResult:
        """
        Advance the environment by one step.

        Pipeline:
          1.  Validate all incoming ActionSchema objects
          2.  Deliver pending inbox messages (flush message bus)
          3.  Execute tool calls via ToolRegistry
          4.  Queue outbound messages
          5.  Collect step logs BEFORE advancing step counter
          6.  Advance WorldModel step counter (ages SLA/deadlines)
          7.  Trigger SchemaDrift if step % 20 == 0
          8.  Compute rewards (via reward_fn if set, else default placeholder)
          9.  Run OversightAgent on this step's tool call log
          10. Build per-agent partial observations
          11. Check done condition
          12. Return StepResult

        Args:
            action: dict mapping agent_id -> ActionSchema for each agent.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() before step().")

        actions: dict[str, ActionSchema] = self._validate_actions(action)

        # ── 1. Deliver messages from previous step ─────────────────────
        self._flush_message_bus()

        # ── 2-4. Process each agent's action ───────────────────────────
        tool_results: dict[str, Any] = {}
        for agent_id in AGENT_IDS:
            if agent_id == "oversight_agent":
                continue
            act = actions.get(agent_id)
            if act is None:
                continue

            if act.tool_call:
                result = self._tool_registry.call(
                    tool_name=act.tool_call, params=act.tool_params, agent_id=agent_id,
                )
                tool_results[agent_id] = result

            if act.message_to and act.message_content:
                self._queue_message(agent_id, act.message_to, act.message_content, self._step_count)

        # ── 4b. Record action history and failure tracking ─────────────
        for _aid, act in actions.items():
            tool_result = tool_results.get(_aid, {})
            self.action_history.append({
                "step":        self._step_count,
                "agent_id":    _aid,
                "tool_call":   act.tool_call,
                "tool_params": act.tool_params,
                "success":     tool_result.get("success", False),
                "error":       tool_result.get("error"),
                "retry_count": self.failed_attempts.get(
                    act.tool_params.get("ticket_id", ""), 0),
                "reward_delta": 0.0,
            })
            if not tool_result.get("success") and act.tool_call == "resolve_ticket":
                tid = act.tool_params.get("ticket_id", "")
                self.failed_attempts[tid] = self.failed_attempts.get(tid, 0) + 1

        # ── 5. Collect logs BEFORE advance so step number matches ───────
        step_logs = self._tool_registry.get_current_step_logs()

        # ── 6. Advance WorldModel step ──────────────────────────────────
        self._world_model.advance_step()
        self._step_count += 1

        # ── 7. Trigger schema drift ─────────────────────────────────────
        drift_records = self._drift_engine.maybe_drift(self._step_count)
        if drift_records:
            self._world_model.set_schema_version(self._drift_engine.schema_version)
            print(
                f"[SchemaDrift] step={self._step_count} "
                f"new schema_v={self._drift_engine.schema_version} | "
                f"mutations: {[r.to_dict() for r in drift_records]}"
            )

        # ── 8. Compute rewards ──────────────────────────────────────────
        world_state = self._world_model.get_state()
        rewards: dict[str, RewardComponents] = {}
        for agent_id in AGENT_IDS:
            act = actions.get(agent_id)
            tool_result = tool_results.get(agent_id, {})
            rewards[agent_id] = self._compute_reward(agent_id, act, tool_result, world_state)

        # ── 9. Run OversightAgent ───────────────────────────────────────
        oversight_flags: list[dict[str, Any]] = []
        if self._oversight_agent is not None:
            oversight_flags = self._oversight_agent.observe(step_logs)
            for flag in oversight_flags:
                target = flag.get("agent_id")
                delta  = flag.get("reward_delta", 0.0)
                if target in rewards:
                    rewards[target].oversight_detection -= abs(delta)
            oversight_reward = sum(
                abs(f["reward_delta"]) for f in oversight_flags if f["reward_delta"] > 0
            )
            rewards["oversight_agent"].oversight_detection += oversight_reward

        # ── 10. Build observations ──────────────────────────────────────
        observations = {aid: self._get_observation(aid) for aid in AGENT_IDS}

        # ── 11. Done condition ──────────────────────────────────────────
        self._done = self._step_count >= self._max_steps or self._check_terminal()

        info: dict[str, Any] = {
            "step":             self._step_count,
            "schema_version":   self._drift_engine.schema_version,
            "drift_records":    [r.to_dict() for r in drift_records],
            "oversight_flags":  oversight_flags,
            "tool_results":     tool_results,
        }

        return StepResult(observations=observations, rewards=rewards, done=self._done, info=info)

    # ------------------------------------------------------------------
    # INTERFACE STABLE — state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:                 # INTERFACE STABLE
        """Full world state snapshot. Safe to mutate."""
        return {
            **self._world_model.get_state(),
            "schema_version":        self._drift_engine.schema_version,
            "active_drift_mutations": self._drift_engine.get_active_mutations(),
            "step_count":            self._step_count,
            "max_steps":             self._max_steps,
            "done":                  self._done,
            "scenario":              self._current_scenario.get("description", ""),
            "difficulty":            self._current_scenario.get("difficulty", 0),
            "episode_length":        self._current_scenario.get("episode_length", self._max_steps),
        }

    # ------------------------------------------------------------------
    # Partial observation builder
    # ------------------------------------------------------------------

    def _get_observation(self, agent_id: str) -> ObservationSchema:
        """
        Build a partial-view ObservationSchema filtered to the agent's dept.

        it_agent       : tickets + resource_pool + inbox
        manager_agent  : resource_pool + inbox + project_tasks + open tickets + deals
        finance_agent  : resource_pool + inbox + active_deals
        oversight_agent: inbox only (full visibility via observe(tool_logs))
        """
        inbox    = list(self._message_bus.get(agent_id, []))
        wm       = self._world_model
        step     = wm.step
        schema_v = self._drift_engine.schema_version
        recent   = [
            ActionHistoryItem(**h)
            for h in self.action_history
            if h["agent_id"] == agent_id
        ][-5:]

        if agent_id == "it_agent":
            return ObservationSchema(
                agent_id=agent_id, inbox=inbox,
                tickets=wm.get_tickets(), resource_pool=wm.get_resource_pool(),
                step_number=step, schema_version=schema_v,
                recent_history=recent,
            )

        if agent_id == AGENT_IT_TACTICAL:
            all_tickets = wm.get_tickets()
            filtered = [
                t for t in all_tickets
                if t.priority == 1 or t.sla_steps_remaining <= 3
            ]
            return ObservationSchema(
                agent_id=agent_id, inbox=inbox,
                tickets=filtered, resource_pool=wm.get_resource_pool(),
                step_number=step, schema_version=schema_v,
                recent_history=recent,
            )

        if agent_id == AGENT_IT_STRATEGIC:
            all_tickets = wm.get_tickets()
            filtered = [t for t in all_tickets if t.priority >= 2]
            return ObservationSchema(
                agent_id=agent_id, inbox=inbox,
                tickets=filtered, resource_pool=wm.get_resource_pool(),
                step_number=step, schema_version=schema_v,
                recent_history=recent,
            )

        if agent_id == "manager_agent":
            return ObservationSchema(
                agent_id=agent_id, inbox=inbox,
                resource_pool=wm.get_resource_pool(),
                project_tasks=wm.get_tasks(),
                active_deals=wm.get_deals(),
                tickets=[t for t in wm.get_tickets() if not t.resolved],
                step_number=step, schema_version=schema_v,
                recent_history=recent,
            )

        if agent_id == "finance_agent":
            return ObservationSchema(
                agent_id=agent_id, inbox=inbox,
                resource_pool=wm.get_resource_pool(),
                active_deals=wm.get_deals(),
                step_number=step, schema_version=schema_v,
                recent_history=recent,
            )

        # oversight_agent
        return ObservationSchema(
            agent_id=agent_id, inbox=inbox,
            step_number=step, schema_version=schema_v,
            recent_history=recent,
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        agent_id: str,
        action: Optional[ActionSchema],
        tool_result: dict[str, Any],
        world_state: dict[str, Any],
    ) -> RewardComponents:
        """
        Compute rewards for one agent this step.
        Delegates to env.reward_fn if set (Ayush's hook), else uses defaults.
        """
        if self.reward_fn is not None:                 # INTERFACE STABLE hook
            return self.reward_fn(agent_id, action, tool_result, world_state)

        rc = RewardComponents()
        if action is None or not tool_result:
            return rc

        if action.tool_call == "resolve_ticket" and tool_result.get("resolved"):
            rc.task_completion += 10.0
            tickets = {t.id: t for t in self._world_model.get_tickets()}
            tid = action.tool_params.get("ticket_id", "")
            if tid in tickets:
                rc.sla_adherence += max(0.0, tickets[tid].sla_steps_remaining * 0.5)

        if action.tool_call == "get_project_status" and not tool_result.get("error"):
            rc.task_completion += 1.0

        if "error" in tool_result and not tool_result.get("transient", False):
            rc.hallucination_penalty -= 5.0

        for ticket in self._world_model.get_tickets():
            if not ticket.resolved and ticket.sla_steps_remaining == 0:
                rc.sla_breach_penalty -= 8.0

        return rc

    # ------------------------------------------------------------------
    # Message bus
    # ------------------------------------------------------------------

    def _queue_message(self, from_agent: str, to_agent: str, content: str, step_sent: int) -> None:
        msg = MessageSchema(from_agent=from_agent, to_agent=to_agent,
                            content=content, step_sent=step_sent)
        if to_agent == "broadcast":
            for aid in AGENT_IDS:
                if aid != from_agent:
                    self._pending_messages[aid].append(msg)
        elif to_agent in self._pending_messages:
            self._pending_messages[to_agent].append(msg)

    def _flush_message_bus(self) -> None:
        self._message_bus      = {aid: list(msgs) for aid, msgs in self._pending_messages.items()}
        self._pending_messages = {aid: [] for aid in AGENT_IDS}

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def _validate_actions(self, raw_actions: dict[str, Any]) -> dict[str, ActionSchema]:
        validated: dict[str, ActionSchema] = {}
        for agent_id, act in raw_actions.items():
            if isinstance(act, ActionSchema):
                validated[agent_id] = act
            elif isinstance(act, dict):
                try:
                    validated[agent_id] = ActionSchema(**act)
                except Exception as exc:
                    print(f"[EnterpriseOpsEnv] Invalid action from {agent_id}: {exc}")
                    validated[agent_id] = ActionSchema()
            else:
                validated[agent_id] = ActionSchema()
        return validated

    # ------------------------------------------------------------------
    # Terminal condition
    # ------------------------------------------------------------------

    def _check_terminal(self) -> bool:
        tickets_clear = all(t.resolved for t in self._world_model.get_tickets())
        tasks_clear   = all(
            t.status in ("completed", "failed") for t in self._world_model.get_tasks()
        )
        return tickets_clear and tasks_clear

    # ------------------------------------------------------------------
    # OversightAgent boot
    # ------------------------------------------------------------------

    def _boot_oversight_agent(self) -> None:
        try:
            from agents.oversight_agent import OversightAgent
            self._oversight_agent = OversightAgent(
                drift_engine=self._drift_engine,
                tool_registry=self._tool_registry,
            )
        except ImportError:
            self._oversight_agent = None
