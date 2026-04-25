import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enterprise_ops.env.env import EnterpriseOpsEnv
from enterprise_ops.agents.it_tactical_agent import ITTacticalAgent
from enterprise_ops.agents.it_strategic_agent import ITStrategicAgent
from enterprise_ops.agents.manager_agent import ManagerAgent
from enterprise_ops.agents.finance_agent import FinanceAgent
from enterprise_ops.agents.oversight_agent import OversightAgent
from enterprise_ops.contracts import (
    AGENT_IT_TACTICAL,
    AGENT_IT_STRATEGIC,
    AGENT_MANAGER,
    AGENT_FINANCE,
    AGENT_OVERSIGHT,
    ActionSchema,
)


env = EnterpriseOpsEnv(
    scenario_path="enterprise_ops/env/scenarios/scenario_08.yaml",
    seed=42,
    max_steps=30,
)

agents = {
    AGENT_IT_TACTICAL: ITTacticalAgent(),
    AGENT_IT_STRATEGIC: ITStrategicAgent(),
    AGENT_MANAGER: ManagerAgent(AGENT_MANAGER),
    AGENT_FINANCE: FinanceAgent(AGENT_FINANCE),
    AGENT_OVERSIGHT: OversightAgent(
        drift_engine=env._drift_engine,
        tool_registry=env._tool_registry,
    ),
}

obs_dict = env.reset()
total_reward = 0.0
step_totals = []

print("=" * 60)
print("SCENARIO_08 FULL 5-AGENT VALIDATION")
print("=" * 60)

for step in range(8):
    actions = {}
    for agent_id, obs in obs_dict.items():
        agent = agents.get(agent_id)
        if agent:
            try:
                actions[agent_id] = agent.act(obs)
            except Exception as e:
                actions[agent_id] = ActionSchema(reasoning=f"fallback: {e}")
        else:
            actions[agent_id] = ActionSchema(reasoning="no agent")

    result = env.step(actions)

    step_reward = sum(r.total() for r in result.rewards.values())
    step_totals.append(step_reward)
    total_reward += step_reward

    print(f"\nStep {step + 1} | total={step_reward:.2f}")
    for agent_id, reward in result.rewards.items():
        if reward.total() != 0:
            print(
                f"  {agent_id}: {reward.total():.2f} "
                f"(task={reward.task_completion:.1f}, "
                f"sla={reward.sla_adherence:.1f}, "
                f"coord={reward.coordination_bonus:.1f})"
            )

    obs_dict = result.observations
    if result.done:
        print("Episode done early")
        break

print(f"\n{'=' * 60}")
print(f"TOTAL REWARD: {total_reward:.2f}")
print("TARGET: > 0 each step")
if total_reward > 0:
    print("PASS: Positive reward achieved")
else:
    print("FAIL: Still getting zero/negative reward")

zero_steps = [i + 1 for i, r in enumerate(step_totals) if abs(r) < 1e-9]
if zero_steps:
    print(f"ZERO-REWARD STEPS: {zero_steps}")
else:
    print("NO ZERO-REWARD STEPS")
