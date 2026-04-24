"""
train/ — Multi-agent RL training pipeline for EnterpriseOps Arena.

Provides:
  - reward_fn.py      → Custom reward computation (per-agent decomposed rewards)
  - trainer.py        → Main training loop + policy optimization (GRPO/DPO via HF TRL)
  - episode_buffer.py → Trajectory collection + replay buffer
  - config.py         → Hyperparameter schedules and scenario progression
"""
