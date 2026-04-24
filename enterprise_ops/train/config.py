"""
config.py — Training hyperparameters for GRPO + Unsloth fine-tuning.

Model:  unsloth/Qwen2.5-3B-Instruct  (4-bit, fits A100 in 48 h)
Stack:  Unsloth == 2025.3.19,  trl >= 0.15.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Central configuration for GRPO + Unsloth training."""

    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # ── GRPO training ─────────────────────────────────────────────────────
    max_steps: int = 200
    episode_length: int = 8            # max steps per env episode
    grpo_num_generations: int = 4      # G — completions per prompt
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_new_tokens: int = 256
    max_prompt_length: int = 512
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99

    # ── Logging & checkpoints ─────────────────────────────────────────────
    log_every: int = 10                # write CSV row every N steps
    save_every: int = 50               # save adapter every N steps
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # ── Curriculum ────────────────────────────────────────────────────────
    curriculum_threshold: float = 0.6  # advance when avg_reward > this …
    curriculum_window: int = 10        # … over this many episodes
    scenario_progression: List[str] = field(default_factory=lambda: [
        "scenario_01.yaml",
        "scenario_02.yaml",
        "scenario_03.yaml",
        "scenario_04.yaml",
        "scenario_05.yaml",
        "scenario_06.yaml",
        "scenario_07.yaml",
        "scenario_08.yaml",
    ])

    # ── Env ───────────────────────────────────────────────────────────────
    seed: int = 42
    noise_rate: float = 0.02
    enable_schema_drift: bool = False
    use_custom_reward_fn: bool = True

    # ── Backward-compat aliases (used by legacy trainer / smoke test) ──────
    num_episodes: int = 1000
    max_steps_per_episode: int = 8
    reward_scale: float = 1.0
    log_every_n_episodes: int = 10
    save_checkpoint_every_n_episodes: int = 50
    episodes_per_scenario: int = 125
    drift_detection_enabled: bool = True
    collect_trajectories: bool = True
    save_buffer_every_n_episodes: int = 50

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_scenario_for_episode(self, episode_idx: int) -> str:
        idx = min(
            episode_idx // max(1, self.episodes_per_scenario),
            len(self.scenario_progression) - 1,
        )
        return self.scenario_progression[idx]

    def get_current_difficulty(self, episode_idx: int) -> int:
        idx = min(
            episode_idx // max(1, self.episodes_per_scenario),
            len(self.scenario_progression) - 1,
        )
        return idx + 1


@dataclass
class EpisodeBufferConfig:
    """Replay-buffer configuration."""
    max_buffer_size: int = 10_000
    prioritize_high_reward: bool = True
    discount_factor: float = 0.99


@dataclass
class AgentConfig:
    """Per-agent training knobs."""
    agent_id: str
    learning_enabled: bool = True
    use_policy_gradient: bool = True
    entropy_coeff: float = 0.01


# Aliases for ergonomic imports
TrainConfig = TrainingConfig
