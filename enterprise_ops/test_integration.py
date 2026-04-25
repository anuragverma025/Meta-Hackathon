"""
test_integration.py — Smoke test for EnterpriseOps Arena env + agents.

Verifies env/ and agents/ work correctly together before train/ is built.

Run from enterprise_ops/:
    python test_integration.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ── Contracts ──────────────────────────────────────────────────────────────
from contracts import (
    ActionSchema,
    ObservationSchema,
    RewardComponents,
    StepResult,
    TicketItem,
    AGENT_IT_TACTICAL,
    AGENT_IT_STRATEGIC,
    AGENT_MANAGER,
    AGENT_FINANCE,
    AGENT_OVERSIGHT,
)

# ── Env ────────────────────────────────────────────────────────────────────
from env.env import EnterpriseOpsEnv

# ── Agents ─────────────────────────────────────────────────────────────────
from agents import ITTacticalAgent, ITStrategicAgent, ManagerAgent, FinanceAgent, OversightAgent

# ── Config ─────────────────────────────────────────────────────────────────
SCENARIO = str(Path(__file__).parent / "env" / "scenarios" / "scenario_01.yaml")
SEED     = 42
STEPS    = 5

ALL_AGENTS = [AGENT_IT_TACTICAL, AGENT_IT_STRATEGIC, AGENT_MANAGER, AGENT_FINANCE, AGENT_OVERSIGHT]

# ── Tiny assertion helper ───────────────────────────────────────────────────
_failures: list[str] = []


def check(condition: bool, label: str) -> bool:
    if condition:
        print(f"  [OK]   {label}")
        return True
    _failures.append(label)
    print(f"  [FAIL] {label}")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Main smoke test
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("EnterpriseOps Arena - Integration Smoke Test")
    print("=" * 64)

    # ── 1. Init env ────────────────────────────────────────────────────────
    print("\n[1] Initialise EnterpriseOpsEnv")
    env = EnterpriseOpsEnv(scenario_path=SCENARIO, seed=SEED)

    # ── 2. Reset ───────────────────────────────────────────────────────────
    print("\n[2] env.reset()")
    obs: dict[str, ObservationSchema] = env.reset()

    check(isinstance(obs, dict), "reset() returns dict")
    check(len(obs) == 5, f"reset() dict has 5 agents (got {len(obs)})")
    for aid in ALL_AGENTS:
        check(aid in obs, f"obs has key '{aid}'")
        check(isinstance(obs.get(aid), ObservationSchema),
              f"obs['{aid}'] is ObservationSchema")

    # ── 3. Schema version present in every initial observation ─────────────
    print("\n[3] schema_version field in initial observations")
    for aid in ALL_AGENTS:
        o = obs.get(aid)
        if o is not None:
            check(isinstance(o.schema_version, int),
                  f"obs['{aid}'].schema_version is int (={o.schema_version})")

    # ── 4. Initialise all 5 agents ──────────────────────────────────────────
    print("\n[4] Initialise agents")
    it_tactical_agent  = ITTacticalAgent()
    it_strategic_agent = ITStrategicAgent()
    manager_agent      = ManagerAgent(AGENT_MANAGER)
    finance_agent      = FinanceAgent(AGENT_FINANCE)
    oversight_agent    = OversightAgent(
        drift_engine=env._drift_engine,
        tool_registry=env._tool_registry,
    )
    check(True, "All 5 agents instantiated without error")

    # ── 5. 5-step loop ─────────────────────────────────────────────────────
    print("\n[5] Running 5 steps")

    cumulative: dict[str, float] = {aid: 0.0 for aid in ALL_AGENTS}
    step1_ticket_items: list[TicketItem] | None = None
    step2_manager_inbox_len: int = 0

    for step_idx in range(1, STEPS + 1):
        print(f"\n  -- Step {step_idx} --")

        # Each agent acts on its current observation
        actions: dict[str, ActionSchema] = {
            AGENT_IT_TACTICAL:  it_tactical_agent.act(obs[AGENT_IT_TACTICAL]),
            AGENT_IT_STRATEGIC: it_strategic_agent.act(obs[AGENT_IT_STRATEGIC]),
            AGENT_MANAGER:      manager_agent.act(obs[AGENT_MANAGER]),
            AGENT_FINANCE:      finance_agent.act(obs[AGENT_FINANCE]),
            AGENT_OVERSIGHT:    ActionSchema(),   # oversight monitors via env internals
        }

        for aid, act in actions.items():
            check(isinstance(act, ActionSchema),
                  f"agent '{aid}' returned ActionSchema")

        # ── Step 1 specific: IT_TACTICAL targets urgent/P1 tickets ─────────
        if step_idx == 1:
            tact_act = actions[AGENT_IT_TACTICAL]
            check(
                tact_act.tool_call in ("resolve_ticket", "get_tickets"),
                f"Step 1: IT_TACTICAL calls resolve_ticket or get_tickets "
                f"(got {tact_act.tool_call!r})",
            )

        # ── env.step ────────────────────────────────────────────────────────
        result: StepResult = env.step(actions)

        check(isinstance(result, StepResult),   "env.step() returns StepResult")
        check(isinstance(result.observations, dict), "StepResult.observations is dict")
        check(isinstance(result.rewards, dict), "StepResult.rewards is dict")
        check(isinstance(result.done, bool),    "StepResult.done is bool")
        check(isinstance(result.info, dict),    "StepResult.info is dict")

        for aid in ALL_AGENTS:
            check(aid in result.observations, f"observations has key '{aid}'")
            check(aid in result.rewards,      f"rewards has key '{aid}'")
            r = result.rewards[aid]
            check(isinstance(r, RewardComponents),
                  f"rewards['{aid}'] is RewardComponents")
            total = r.total()
            check(isinstance(total, float),
                  f"rewards['{aid}'].total() is float (={total})")
            cumulative[aid] += total

        # ── Step 1 post-step: verify IT_TACTICAL tool result ───────────────
        if step_idx == 1:
            raw = result.info.get("tool_results", {}).get(AGENT_IT_TACTICAL)
            tact_act = actions[AGENT_IT_TACTICAL]
            if raw and tact_act.tool_call == "get_tickets" and "tickets" in raw:
                try:
                    items = [TicketItem(**t) for t in raw["tickets"]]
                    check(len(items) > 0,
                          f"get_tickets returned {len(items)} TicketItem(s)")
                    step1_ticket_items = items
                except Exception as exc:
                    check(False, f"get_tickets result not parseable as TicketItem: {exc}")
            elif raw and tact_act.tool_call == "resolve_ticket":
                check(
                    raw.get("resolved") is True or raw.get("ticket_id") is not None,
                    f"Step 1: IT_TACTICAL resolve_ticket result valid (raw={raw!r})",
                )
            else:
                check(False,
                      f"Step 1: unexpected IT_TACTICAL tool result (raw={raw!r})")

        # ── Step 2: Manager inbox is accessible ────────────────────────────
        if step_idx == 2:
            mgr_obs = result.observations[AGENT_MANAGER]
            it_msgs = [
                m for m in mgr_obs.inbox
                if m.from_agent in (AGENT_IT_TACTICAL, AGENT_IT_STRATEGIC)
            ]
            step2_manager_inbox_len = len(mgr_obs.inbox)
            check(
                isinstance(mgr_obs.inbox, list),
                f"Step 2: Manager inbox accessible "
                f"(size={step2_manager_inbox_len}, IT msgs={len(it_msgs)})",
            )

        # Print per-step reward summary
        reward_line = " | ".join(
            f"{aid.split('_')[0]}={result.rewards[aid].total():+.1f}"
            for aid in ALL_AGENTS
        )
        print(f"  Rewards: {reward_line}  |  done={result.done}")

        obs = result.observations

    # ── 6. Non-zero reward verification ────────────────────────────────────
    print("\n[6] Non-zero reward verification")
    any_nonzero = any(v != 0.0 for v in cumulative.values())
    check(any_nonzero,
          f"At least one agent earned non-zero cumulative reward "
          f"(TACT={cumulative[AGENT_IT_TACTICAL]:+.1f}, "
          f"STRAT={cumulative[AGENT_IT_STRATEGIC]:+.1f}, "
          f"MGR={cumulative[AGENT_MANAGER]:+.1f}, "
          f"FIN={cumulative[AGENT_FINANCE]:+.1f}, "
          f"OVR={cumulative[AGENT_OVERSIGHT]:+.1f})")

    # ── 7. Schema version tracked in final observations ────────────────────
    print("\n[7] schema_version in final observations")
    for aid in ALL_AGENTS:
        o = obs[aid]
        check(o.schema_version >= 1,
              f"Final obs['{aid}'].schema_version >= 1 (={o.schema_version})")

    # ── Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    if not _failures:
        print("SMOKE TEST PASSED")
        print(f"  Steps run        : {STEPS}")
        print(f"  Tickets verified : {len(step1_ticket_items) if step1_ticket_items else 0}")
        print(f"  Manager inbox    : {step2_manager_inbox_len} message(s) at step 2")
        print(f"  Cumulative reward: "
              + ", ".join(f"{aid}={v:+.2f}" for aid, v in cumulative.items()))
    else:
        print(f"SMOKE TEST FAILED — {len(_failures)} failure(s):")
        for i, f in enumerate(_failures, 1):
            print(f"  {i}. {f}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL: Uncaught exception — {exc}")
        traceback.print_exc()
        sys.exit(1)
