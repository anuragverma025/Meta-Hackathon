"""
client.py — HTTP client for the EnterpriseOps Arena OpenEnv server.

INTERFACE STABLE — Ayush's training loop uses this.

Usage::

    from enterprise_ops.client import EnterpriseClient
    from enterprise_ops.models import EnterpriseAction

    client = EnterpriseClient(base_url="http://localhost:8000")
    obs = client.reset(scenario="scenario_01")
    result = client.step(EnterpriseAction(
        agent_id="it_agent",
        tool_call="get_tickets",
    ))
    print(result["reward"], result["done"])
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

# Ensure enterprise_ops/ is on sys.path for flat imports
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import EnterpriseAction, EnterpriseObservation  # noqa: E402

# Re-export so callers can do: from enterprise_ops.client import EnterpriseAction
__all__ = ["EnterpriseClient", "EnterpriseEnv", "EnterpriseAction",
           "EnterpriseObservation", "StepResult"]


@dataclass
class StepResult:
    """Return value of EnterpriseEnv.step() — provides attribute access."""
    observation: EnterpriseObservation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class EnterpriseClient:
    """
    Synchronous HTTP client for the EnterpriseOps Arena server.

    Wraps the FastAPI endpoints into a clean Python API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()

    def __enter__(self) -> "EnterpriseClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, str]:
        """Check server liveness."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnterpriseObservation:
        """
        Reset the environment and return the primary observation.

        Args:
            scenario: e.g. "scenario_01" or full path to YAML.
            seed:     RNG seed override.
        """
        payload: dict[str, Any] = {}
        if scenario is not None:
            payload["scenario"] = scenario
        if seed is not None:
            payload["seed"] = seed

        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return EnterpriseObservation.from_dict(data["observation"])

    # ------------------------------------------------------------------
    # step (single agent)
    # ------------------------------------------------------------------

    def step(self, action: EnterpriseAction) -> dict[str, Any]:
        """
        Execute one step for a single agent.

        Returns dict with keys: observation, reward, done, info
        """
        payload = {
            "agent_id": action.agent_id,
            "tool_call": action.tool_call,
            "tool_params": action.tool_params or {},
            "message_to": action.message_to,
            "message_content": action.message_content,
            "reasoning": action.reasoning,
        }
        resp = self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "observation": EnterpriseObservation.from_dict(data["observation"]),
            "reward": data["reward"],
            "done": data["done"],
            "info": data.get("info", {}),
        }

    # ------------------------------------------------------------------
    # step_multi (all agents)
    # ------------------------------------------------------------------

    def step_multi(
        self, actions: dict[str, EnterpriseAction],
    ) -> dict[str, Any]:
        """
        Execute one step with explicit actions for multiple agents.

        Returns dict with keys: observations (per-agent), reward, done, info
        """
        payload = {
            "actions": {
                aid: {
                    "agent_id": a.agent_id,
                    "tool_call": a.tool_call,
                    "tool_params": a.tool_params or {},
                    "message_to": a.message_to,
                    "message_content": a.message_content,
                    "reasoning": a.reasoning,
                }
                for aid, a in actions.items()
            }
        }
        resp = self._client.post("/step_multi", json=payload)
        resp.raise_for_status()
        data = resp.json()

        obs_dict: dict[str, EnterpriseObservation] = {}
        for aid, obs_data in data.get("observations", {}).items():
            obs_dict[aid] = EnterpriseObservation.from_dict(obs_data)

        return {
            "observations": obs_dict,
            "reward": data["reward"],
            "done": data["done"],
            "info": data.get("info", {}),
        }

    # ------------------------------------------------------------------
    # get_all_observations
    # ------------------------------------------------------------------

    def get_all_observations(self) -> dict[str, EnterpriseObservation]:
        """
        Retrieve per-agent partial observations from the server.

        Returns dict mapping agent_id → EnterpriseObservation:
          - it_agent:        tickets + resource_pool + inbox
          - manager_agent:   all summaries + project_tasks + inbox
          - finance_agent:   budget + pending_approvals + inbox
          - oversight_agent: full tool logs visibility
        """
        resp = self._client.get("/observations")
        resp.raise_for_status()
        data = resp.json()
        return {
            aid: EnterpriseObservation.from_dict(obs_data)
            for aid, obs_data in data.items()
        }

    # ------------------------------------------------------------------
    # get_state
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return the full world state snapshot from the server."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # set_scenario (curriculum)
    # ------------------------------------------------------------------

    def set_scenario(self, scenario_name: str) -> dict[str, str]:
        """
        Switch scenario for the next reset() call.

        Args:
            scenario_name: e.g. "scenario_03" or full YAML path.
        """
        resp = self._client.post("/scenario", json={"scenario_name": scenario_name})
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# EnterpriseEnv — thin sync wrapper used by the training loop
# ---------------------------------------------------------------------------


class EnterpriseEnv:
    """
    Synchronous high-level client for the EnterpriseOps Arena server.

    Designed for use as a context manager::

        with EnterpriseEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(EnterpriseAction(agent_id="it_agent", ...))
            print(result.reward)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ) -> None:
        self._client = EnterpriseClient(base_url=base_url, timeout=timeout)

    def sync(self) -> "EnterpriseEnv":
        """Return self — client is already synchronous. Enables .sync() idiom."""
        return self

    def __enter__(self) -> "EnterpriseEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def reset(
        self,
        scenario: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnterpriseObservation:
        return self._client.reset(scenario=scenario, seed=seed)

    def step(self, action: EnterpriseAction) -> StepResult:
        """Execute one step and return a StepResult with .reward attribute."""
        raw = self._client.step(action)
        return StepResult(
            observation=raw["observation"],
            reward=raw["reward"],
            done=raw["done"],
            info=raw.get("info", {}),
        )

    def step_multi(
        self, actions: dict[str, EnterpriseAction],
    ) -> dict[str, Any]:
        return self._client.step_multi(actions)

    def get_all_observations(self) -> dict[str, EnterpriseObservation]:
        return self._client.get_all_observations()

    def get_state(self) -> dict[str, Any]:
        return self._client.get_state()

    def set_scenario(self, scenario_name: str) -> dict[str, str]:
        return self._client.set_scenario(scenario_name)
