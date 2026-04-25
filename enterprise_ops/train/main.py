"""
main.py — CLI entry point for EnterpriseOps Arena GRPO training.

Usage
-----
    python -m enterprise_ops.train.main --scenario scenario_01 --steps 200

Logging
-------
    logs/metrics.csv   — one row per log_every steps:
                         step, episode_score, task_completion, sla_rate,
                         coordination_score, curriculum_difficulty
Checkpoints
-----------
    checkpoints/step_XXXXX/   — LoRA adapters (not merged to 16-bit)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.config import TrainingConfig, EpisodeBufferConfig
from train.trainer import train_main, EnterpriseOpsTrainer


# ---------------------------------------------------------------------------
# run_single_episode — returns float reward (used by integration test)
# ---------------------------------------------------------------------------

def run_single_episode(scenario: str = "scenario_01") -> float:
    """
    Run one complete episode with rule-based agents and return total reward.

    Args:
        scenario: e.g. "scenario_01" or a path ending in ".yaml".

    Returns:
        Sum of all per-step, per-agent rewards over the episode.
    """
    from env.env import EnterpriseOpsEnv
    from agents import ITAgent, ManagerAgent, FinanceAgent, OversightAgent
    from contracts import AGENT_IT, AGENT_MANAGER, AGENT_FINANCE, AGENT_OVERSIGHT

    scenario_path: str
    if scenario.endswith(".yaml"):
        p = Path(scenario)
        scenario_path = str(p) if p.is_absolute() else str(_ROOT / "env" / "scenarios" / scenario)
    else:
        scenario_path = str(_ROOT / "env" / "scenarios" / f"{scenario}.yaml")

    env = EnterpriseOpsEnv(scenario_path=scenario_path, seed=42)
    agents = {
        AGENT_IT: ITAgent(),
        AGENT_MANAGER:  ManagerAgent(AGENT_MANAGER),
        AGENT_FINANCE:  FinanceAgent(AGENT_FINANCE),
        AGENT_OVERSIGHT: OversightAgent(
            drift_engine=env._drift_engine,
            tool_registry=env._tool_registry,
        ),
    }

    obs = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        actions = {aid: agent.act(obs[aid]) for aid, agent in agents.items()}
        result = env.step(actions)
        obs = result.observations
        done = result.done
        total_reward += sum(r.total() for r in result.rewards.values())

    return total_reward


# ---------------------------------------------------------------------------
# Reward curve summary
# ---------------------------------------------------------------------------

def _print_reward_curve(csv_path: Path) -> None:
    """Print a compact ASCII reward curve from logs/metrics.csv."""
    if not csv_path.exists():
        print("  (no metrics.csv found)")
        return

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("  (metrics.csv is empty)")
        return

    print("\n" + "=" * 70)
    print("Reward Curve Summary")
    print("=" * 70)
    print(f"{'Step':>6}  {'Score':>8}  {'Task':>6}  {'SLA':>6}  {'Coord':>6}  {'Diff':>4}")
    print("-" * 70)
    for r in rows:
        print(
            f"{int(r['step']):>6}  "
            f"{float(r['episode_score']):>8.3f}  "
            f"{float(r['task_completion']):>6.2f}  "
            f"{float(r['sla_rate']):>6.2f}  "
            f"{float(r['coordination_score']):>6.2f}  "
            f"{r['curriculum_difficulty']:>4}"
        )

    scores = [float(r["episode_score"]) for r in rows]
    if scores:
        print("-" * 70)
        print(f"  First={scores[0]:.3f}  Last={scores[-1]:.3f}  "
              f"Max={max(scores):.3f}  Min={min(scores):.3f}  "
              f"Improvement={scores[-1]-scores[0]:+.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EnterpriseOps Arena agents with GRPO + Unsloth"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_01",
        help="Starting scenario name (e.g. scenario_01) or .yaml path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total GRPO training steps (default: 200)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=8,
        help="Max env steps per episode (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for metrics.csv (default: ./logs)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for LoRA adapter checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log CSV row every N steps (default: 10)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--grpo-gens",
        type=int,
        default=4,
        help="GRPO completions per prompt (default: 4)",
    )
    parser.add_argument(
        "--no-reward-fn",
        action="store_true",
        help="Disable custom reward function",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        max_steps=args.steps,
        episode_length=args.episode_length,
        max_steps_per_episode=args.episode_length,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        log_every_n_episodes=args.log_every,
        save_every=args.save_every,
        save_checkpoint_every_n_episodes=args.save_every,
        grpo_num_generations=args.grpo_gens,
        use_custom_reward_fn=not args.no_reward_fn,
    )
    buffer_config = EpisodeBufferConfig()

    trainer = EnterpriseOpsTrainer(config, buffer_config)
    trainer.train(start_scenario=args.scenario)

    _print_reward_curve(Path(args.log_dir) / "metrics.csv")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
