"""
trainer.py — GRPO training loop for EnterpriseOps Arena.

Stack
-----
  unsloth==2025.3.19     4-bit Qwen2.5-3B-Instruct loading + PEFT
  trl>=0.15.0            GRPOTrainer
  EnterpriseOpsEnv       local env (no HTTP server needed during training)

Modes
-----
  GRPO mode  — GPU with unsloth + trl installed → trains the LLM
  Dry-run    — CPU / missing deps → collects episodes, logs metrics, no LLM

LoRA adapters are saved with model.save_pretrained() (NOT merged to 16-bit,
which degrades quality per Unsloth official docs).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── sys.path: ensure enterprise_ops/ root is importable ───────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Optional heavy deps (GPU-only) ────────────────────────────────────────
try:
    from unsloth import FastLanguageModel  # type: ignore[import]
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False

try:
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]
    from transformers import TrainerCallback  # type: ignore[import]
    HAS_TRL = True
except Exception:
    HAS_TRL = False
    class TrainerCallback:  # type: ignore[no-redef]
        """Stub when transformers is not installed."""

try:
    from datasets import Dataset  # type: ignore[import]
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# ── Enterprise imports ─────────────────────────────────────────────────────
from contracts import (
    ActionSchema,
    ObservationSchema,
    AGENT_IT_TACTICAL,
    AGENT_IT_STRATEGIC,
    AGENT_MANAGER,
    AGENT_FINANCE,
    AGENT_OVERSIGHT,
)
from env.env import EnterpriseOpsEnv
from agents.it_tactical_agent import ITTacticalAgent
from agents.it_strategic_agent import ITStrategicAgent
from agents.manager_agent import ManagerAgent
from agents.finance_agent import FinanceAgent
from agents.oversight_agent import OversightAgent

from .config import TrainingConfig, EpisodeBufferConfig
from .episode_buffer import EpisodeBuffer, Transition
from .reward_fn import (
    compute_reward,
    reward_task_completion,
    reward_sla_adherence,
    reward_coordination,
    penalty_hallucination,
)

# ── Agent system prompts ───────────────────────────────────────────────────
_SYSTEM = {
    AGENT_IT_TACTICAL: (
        "You are the IT Agent in an enterprise operations environment. "
        "Resolve support tickets, manage compute resources, coordinate with the manager. "
        "Available tools: get_tickets, resolve_ticket, allocate_resource, get_project_status. "
        'Respond ONLY with valid JSON: {"tool_call":"<name>","tool_params":{},'
        '"message_to":null,"message_content":null}'
    ),
    AGENT_MANAGER: (
        "You are the Manager Agent. Oversee project tasks, set priorities, coordinate teams. "
        "Available tools: get_project_status, allocate_resource. "
        'Respond ONLY with valid JSON: {"tool_call":"<name>","tool_params":{},'
        '"message_to":null,"message_content":null}'
    ),
    AGENT_FINANCE: (
        "You are the Finance Agent. Manage budget approvals and financial requests. "
        "Available tool: approve_budget. "
        'Respond ONLY with valid JSON: {"tool_call":"approve_budget",'
        '"tool_params":{"amount":0,"justification":""},'
        '"message_to":null,"message_content":null}'
    ),
}
_TRAINABLE_AGENTS = [AGENT_IT_TACTICAL, AGENT_IT_STRATEGIC, AGENT_MANAGER, AGENT_FINANCE]

CSV_FIELDS = [
    "step", "episode_score", "task_completion",
    "sla_rate", "coordination_score", "curriculum_difficulty",
]


# ---------------------------------------------------------------------------
# CSV logging callback (for GRPO mode)
# ---------------------------------------------------------------------------

class _CSVCallback(TrainerCallback):
    """Writes one metrics row per logging checkpoint and checks curriculum."""

    def __init__(self, trainer_ref: "EnterpriseOpsTrainer") -> None:
        self._t = trainer_ref

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
        step = state.global_step
        scenario_path = self._t._current_scenario_path()
        metrics = self._t._collect_episode_metrics(scenario_path, seed=step)
        self._t._write_csv_row(step, metrics)
        self._t.buffer.add_episode_reward(metrics["episode_score"])
        self._t._maybe_advance_curriculum()

    def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        print(f"  -> Checkpoint saved at step {state.global_step}")


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class EnterpriseOpsTrainer:
    """
    GRPO trainer for EnterpriseOps Arena.

    Falls back to a dry-run episode-collection loop when Unsloth / TRL are
    not available (CPU machines, CI, pipeline verification).
    """

    def __init__(
        self,
        config: TrainingConfig,
        buffer_config: EpisodeBufferConfig,
        db_path: str = "episodes.db",
    ) -> None:
        self.config = config
        self.buffer_config = buffer_config
        self.db_path = db_path

        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = EpisodeBuffer(
            max_buffer_size=buffer_config.max_buffer_size,
            prioritize_high_reward=buffer_config.prioritize_high_reward,
            discount_factor=buffer_config.discount_factor,
        )

        self.current_scenario_idx: int = 0
        self.episode_log: list[dict[str, Any]] = []

        # LLM (loaded only when Unsloth is available)
        self.model: Any = None
        self.tokenizer: Any = None
        if HAS_UNSLOTH:
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading (Unsloth 4-bit)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Qwen2.5-3B-Instruct with Unsloth 4-bit quantization + LoRA."""
        print(f"[Trainer] Loading {self.config.model_name} with Unsloth 4-bit ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
        )
        self.model = model
        self.tokenizer = tokenizer
        print("[Trainer] Model ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def scenario_progression(self) -> list[str]:
        return self.config.scenario_progression

    def _current_scenario_path(self) -> str:
        scenario = self.scenario_progression[self.current_scenario_idx]
        return str(_ROOT / "env" / "scenarios" / scenario)

    def _make_agents(self, env: EnterpriseOpsEnv) -> dict[str, Any]:
        return {
            AGENT_IT_TACTICAL: ITTacticalAgent(AGENT_IT_TACTICAL),
            AGENT_IT_STRATEGIC: ITStrategicAgent(AGENT_IT_STRATEGIC),
            AGENT_MANAGER: ManagerAgent(AGENT_MANAGER),
            AGENT_FINANCE: FinanceAgent(AGENT_FINANCE),
            AGENT_OVERSIGHT: OversightAgent(AGENT_OVERSIGHT),
        }

    def _format_prompt(self, obs: ObservationSchema, agent_id: str) -> str:
        """Serialize an observation as a Qwen chat prompt."""
        system = _SYSTEM.get(agent_id, _SYSTEM[AGENT_IT_TACTICAL])
        obs_data = {
            "step": obs.step_number,
            "tickets": [t.model_dump() for t in obs.tickets[:3]],
            "inbox": [m.model_dump() for m in obs.inbox[:3]],
            "tasks": [t.model_dump() for t in obs.project_tasks[:3]],
        }
        if obs.resource_pool:
            obs_data["resources"] = obs.resource_pool.model_dump()

        obs_json = json.dumps(obs_data)

        if self.tokenizer is not None:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": obs_json},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback plain text (dry-run / no tokenizer)
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{obs_json}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # ------------------------------------------------------------------
    # Episode metrics collection
    # ------------------------------------------------------------------

    def _collect_episode_metrics(self, scenario_path: str, seed: int = 0) -> dict[str, float]:
        """Run one full episode with rule-based agents; return decomposed metrics."""
        env = EnterpriseOpsEnv(
            scenario_path=scenario_path,
            seed=seed,
            max_steps=self.config.episode_length,
        )
        obs_dict = env.reset()
        agents = self._make_agents(env)

        total_task = total_sla = total_coord = total_halluc = 0.0
        prev_msgs: list[dict[str, str]] = []

        for _ in range(self.config.episode_length):
            actions = {aid: agents[aid].act(obs_dict[aid]) for aid in agents}
            result = env.step(actions)

            total_task += reward_task_completion(result)
            total_sla += reward_sla_adherence(result)
            total_coord += reward_coordination(result, prev_msgs)
            total_halluc += penalty_hallucination(result)

            for aid, act in actions.items():
                if act.message_to and act.message_content:
                    prev_msgs.append({"from": aid, "to": act.message_to})

            obs_dict = result.observations
            if result.done:
                break

        episode_score = total_task + total_sla + total_coord + total_halluc
        sla_rate = (total_sla / max(1.0, total_task)) if total_task > 0 else 0.0

        return {
            "episode_score": round(episode_score, 4),
            "task_completion": round(total_task, 4),
            "sla_rate": round(min(1.0, sla_rate), 4),
            "coordination_score": round(total_coord, 4),
        }

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        csv_path = self.log_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _write_csv_row(self, step: int, metrics: dict[str, float]) -> None:
        csv_path = self.log_dir / "metrics.csv"
        row = {
            "step": step,
            "episode_score": metrics["episode_score"],
            "task_completion": metrics["task_completion"],
            "sla_rate": metrics["sla_rate"],
            "coordination_score": metrics["coordination_score"],
            "curriculum_difficulty": self.current_scenario_idx + 1,
        }
        exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self) -> None:
        if self.buffer.should_advance_curriculum(
            threshold=self.config.curriculum_threshold,
            window=self.config.curriculum_window,
        ):
            if self.current_scenario_idx < len(self.scenario_progression) - 1:
                self.current_scenario_idx += 1
                self.buffer.recent_rewards.clear()
                print(
                    f"  [Curriculum] Advanced -> scenario "
                    f"{self.current_scenario_idx + 1}: "
                    f"{self.scenario_progression[self.current_scenario_idx]}"
                )

    # ------------------------------------------------------------------
    # Checkpoint saving (LoRA adapters only — NOT merged to 16-bit)
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        """Save LoRA adapter weights.  Never merge to 16-bit (degrades quality)."""
        step_dir = self.checkpoint_dir / f"step_{step:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(str(step_dir))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(step_dir))
            print(f"  -> LoRA adapter saved: {step_dir}")
        else:
            # Dry-run: save metrics snapshot
            meta = {
                "step": step,
                "scenario_idx": self.current_scenario_idx,
                "buffer_stats": self.buffer.get_buffer_stats(),
                "recent_avg_reward": self.buffer.get_recent_avg_reward(),
            }
            with open(step_dir / "checkpoint_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  -> Checkpoint metadata saved: {step_dir}")

    def save_logs(self) -> None:
        log_path = self.log_dir / "episode_log.json"
        with open(log_path, "w") as f:
            json.dump(self.episode_log, f, indent=2)
        print(f"  -> Episode log saved: {log_path}")

    # ------------------------------------------------------------------
    # GRPO dataset building
    # ------------------------------------------------------------------

    def _build_prompt_dataset(self, scenario_path: str) -> "Dataset":
        """Collect observations with rule-based agents to form the GRPO dataset."""
        env = EnterpriseOpsEnv(
            scenario_path=scenario_path,
            seed=self.config.seed,
            max_steps=self.config.episode_length,
        )
        obs_dict = env.reset()
        agents = self._make_agents(env)

        records: list[dict[str, str]] = []
        collected = 0
        target = 32  # prompts per dataset rebuild

        while collected < target:
            for aid in _TRAINABLE_AGENTS:
                obs = obs_dict.get(aid)
                if obs is None:
                    continue
                records.append({
                    "prompt": self._format_prompt(obs, aid),
                    "agent_id": aid,
                })
                collected += 1

            actions = {aid: agents[aid].act(obs_dict[aid]) for aid in agents}
            result = env.step(actions)
            obs_dict = result.observations
            if result.done:
                obs_dict = env.reset()

        return Dataset.from_list(records)

    # ------------------------------------------------------------------
    # GRPO reward function
    # ------------------------------------------------------------------

    def _make_grpo_reward_fn(self, scenario_path: str) -> Callable:
        """Return a TRL-compatible reward function that executes completions in the env."""
        config = self.config

        def grpo_reward_fn(
            prompts: list[str],
            completions: list[str],
            **kwargs: Any,
        ) -> list[float]:
            rewards: list[float] = []
            for completion in completions:
                try:
                    m = re.search(r"\{.*\}", completion, re.DOTALL)
                    if not m:
                        rewards.append(-0.5)
                        continue
                    d = json.loads(m.group())
                    action = ActionSchema(
                        tool_call=d.get("tool_call"),
                        tool_params=d.get("tool_params", {}),
                        message_to=d.get("message_to"),
                        message_content=d.get("message_content"),
                    )
                    env = EnterpriseOpsEnv(
                        scenario_path=scenario_path,
                        seed=config.seed,
                        max_steps=config.episode_length,
                    )
                    obs_dict = env.reset()
                    all_actions = {aid: ActionSchema() for aid in
                                   [AGENT_IT_TACTICAL, AGENT_IT_STRATEGIC, AGENT_MANAGER, AGENT_FINANCE, AGENT_OVERSIGHT]}
                    all_actions[AGENT_IT_TACTICAL] = action
                    result = env.step(all_actions)
                    rewards.append(float(compute_reward(result)))
                except Exception:
                    rewards.append(0.0)
            return rewards

        return grpo_reward_fn

    # ------------------------------------------------------------------
    # Dry-run training loop (CPU / no GPU)
    # ------------------------------------------------------------------

    def _dry_run_train(self, start_scenario: Optional[str] = None) -> list[dict]:
        """Episode collection loop — logs metrics without LLM updates."""
        if start_scenario:
            names = [s.replace(".yaml", "") for s in self.scenario_progression]
            key = start_scenario.replace(".yaml", "")
            if key in names:
                self.current_scenario_idx = names.index(key)

        self._init_csv()
        summary: list[dict] = []

        print(f"[Dry-run] max_steps={self.config.max_steps} | "
              f"episode_length={self.config.episode_length}")

        for step in range(self.config.max_steps):
            scenario_path = self._current_scenario_path()
            metrics = self._collect_episode_metrics(scenario_path, seed=self.config.seed + step)
            self.buffer.add_episode_reward(metrics["episode_score"])
            self.episode_log.append({"step": step, **metrics,
                                     "difficulty": self.current_scenario_idx + 1})

            if step % self.config.log_every == 0:
                self._write_csv_row(step, metrics)
                print(
                    f"  step={step:>4} | score={metrics['episode_score']:>7.3f} | "
                    f"task={metrics['task_completion']:>5.2f} | "
                    f"sla={metrics['sla_rate']:>4.2f} | "
                    f"coord={metrics['coordination_score']:>5.2f} | "
                    f"diff={self.current_scenario_idx + 1}"
                )
                summary.append({"step": step, **metrics,
                                 "difficulty": self.current_scenario_idx + 1})

            if step > 0 and step % self.config.save_every == 0:
                self.save_checkpoint(step)

            self._maybe_advance_curriculum()

        self.save_checkpoint(self.config.max_steps)
        self.save_logs()
        return summary

    # ------------------------------------------------------------------
    # GRPO training loop (GPU + Unsloth + TRL)
    # ------------------------------------------------------------------

    def _grpo_train(self, start_scenario: Optional[str] = None) -> None:
        """Full GRPO training with Unsloth model and TRL GRPOTrainer."""
        if start_scenario:
            names = [s.replace(".yaml", "") for s in self.scenario_progression]
            key = start_scenario.replace(".yaml", "")
            if key in names:
                self.current_scenario_idx = names.index(key)

        self._init_csv()
        scenario_path = self._current_scenario_path()

        print(f"[GRPO] Building prompt dataset from {scenario_path} ...")
        dataset = self._build_prompt_dataset(scenario_path)

        grpo_cfg = GRPOConfig(
            output_dir=str(self.checkpoint_dir),
            max_steps=self.config.max_steps,
            num_generations=self.config.grpo_num_generations,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_new_tokens,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            logging_steps=self.config.log_every,
            save_steps=self.config.save_every,
            report_to="none",
        )

        env_reward_fn = self._make_grpo_reward_fn(scenario_path)
        callback = _CSVCallback(self)

        trainer = GRPOTrainer(
            model=self.model,
            args=grpo_cfg,
            reward_funcs=[env_reward_fn],
            train_dataset=dataset,
            processing_class=self.tokenizer,
            callbacks=[callback],
        )
        print(f"[GRPO] Training {self.config.max_steps} steps ...")
        trainer.train()

        # Save LoRA adapters only — merging to 16-bit damages quality
        final_dir = self.checkpoint_dir / "final_lora"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        print(f"[GRPO] LoRA adapters saved: {final_dir}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self, start_scenario: Optional[str] = None) -> None:
        """
        Dispatch to GRPO training (GPU) or dry-run (CPU / no deps).

        Args:
            start_scenario: e.g. "scenario_01" or "scenario_01.yaml".
                            Overrides the curriculum starting point.
        """
        print("=" * 70)
        print("EnterpriseOps Arena - GRPO Training")
        print(f"  model       : {self.config.model_name}")
        print(f"  max_steps   : {self.config.max_steps}")
        print(f"  grpo_gens   : {self.config.grpo_num_generations}")
        print(f"  episode_len : {self.config.episode_length}")
        print(f"  unsloth     : {HAS_UNSLOTH}  |  trl : {HAS_TRL}")
        print("=" * 70)

        if HAS_UNSLOTH and HAS_TRL and self.model is not None:
            self._grpo_train(start_scenario)
        else:
            if not HAS_UNSLOTH:
                print("[Trainer] unsloth not found — running dry-run mode (CPU).")
            elif not HAS_TRL:
                print("[Trainer] trl not found — running dry-run mode (CPU).")
            self._dry_run_train(start_scenario)

    # backward-compat alias used by smoke test / old callers
    def collect_episode(self, episode_idx: int) -> dict[str, Any]:
        scenario_path = self._current_scenario_path()
        metrics = self._collect_episode_metrics(scenario_path, seed=self.config.seed + episode_idx)
        return {"episode": episode_idx, **metrics}


# Alias
Trainer = EnterpriseOpsTrainer


def train_main(
    config: Optional[TrainingConfig] = None,
    buffer_config: Optional[EpisodeBufferConfig] = None,
    start_scenario: Optional[str] = None,
) -> None:
    """Module-level entry point for programmatic use."""
    from .config import TrainingConfig as _TC, EpisodeBufferConfig as _BC
    config = config or _TC()
    buffer_config = buffer_config or _BC()
    trainer = EnterpriseOpsTrainer(config, buffer_config)
    trainer.train(start_scenario=start_scenario)
