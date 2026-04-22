"""
episode_buffer.py — Trajectory collection and curriculum tracking.

EpisodeBuffer stores per-agent transitions for replay and tracks the
last `maxlen=10` episode-level rewards for curriculum advancement.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contracts import ObservationSchema, ActionSchema, RewardComponents


@dataclass
class Transition:
    """Single-step transition for one agent."""

    agent_id: str
    observation: ObservationSchema
    action: ActionSchema
    reward: RewardComponents
    next_observation: ObservationSchema
    done: bool
    step_number: int


class EpisodeBuffer:
    """
    Replay buffer for multi-agent trajectories.

    Also tracks the last 10 episode-level rewards so the training loop can
    decide when to advance the curriculum scenario.
    """

    def __init__(
        self,
        max_buffer_size: int = 10_000,
        prioritize_high_reward: bool = True,
        discount_factor: float = 0.99,
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.prioritize_high_reward = prioritize_high_reward
        self.discount_factor = discount_factor

        self.buffers: dict[str, deque[Transition]] = {}
        self.episode_count: int = 0
        self.total_transitions: int = 0

        # Curriculum tracking — window of last 10 episode total rewards
        self.recent_rewards: deque[float] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Transition management
    # ------------------------------------------------------------------

    def add_transition(self, transition: Transition) -> None:
        """Add one step transition to the per-agent buffer."""
        aid = transition.agent_id
        if aid not in self.buffers:
            self.buffers[aid] = deque(maxlen=self.max_buffer_size)
        self.buffers[aid].append(transition)
        self.total_transitions += 1

    def get_episode_trajectory(self, agent_id: str) -> list[Transition]:
        """Return all stored transitions for a given agent."""
        return list(self.buffers.get(agent_id, []))

    def sample_batch(self, agent_id: str, batch_size: int = 32) -> list[Transition]:
        """Uniform random sample from an agent's buffer."""
        buf = self.buffers.get(agent_id, deque())
        if not buf:
            return []
        return random.sample(list(buf), min(batch_size, len(buf)))

    def clear_agent_buffer(self, agent_id: str) -> None:
        if agent_id in self.buffers:
            self.buffers[agent_id].clear()

    def clear_all(self) -> None:
        self.buffers.clear()
        self.total_transitions = 0

    def increment_episode(self) -> None:
        self.episode_count += 1

    def get_buffer_stats(self) -> dict[str, int]:
        return {aid: len(buf) for aid, buf in self.buffers.items()}

    # ------------------------------------------------------------------
    # Curriculum tracking
    # ------------------------------------------------------------------

    def add_episode_reward(self, reward: float) -> None:
        """Record an episode-level total reward for curriculum tracking."""
        self.recent_rewards.append(reward)

    def get_recent_avg_reward(self) -> float:
        """Average reward across the last (up to 10) episodes."""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def should_advance_curriculum(
        self,
        threshold: float = 0.6,
        window: int = 10,
    ) -> bool:
        """
        Return True when the rolling average of the last `window` episode
        rewards exceeds `threshold`.  Requires at least `window` episodes.
        """
        if len(self.recent_rewards) < window:
            return False
        recent = list(self.recent_rewards)[-window:]
        return (sum(recent) / len(recent)) > threshold
