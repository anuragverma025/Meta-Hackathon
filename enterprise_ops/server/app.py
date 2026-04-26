"""
app.py — FastAPI server for EnterpriseOps Arena.

Exposes the OpenEnv-compatible endpoints:
  POST /reset          → reset environment, return primary observation
  POST /step           → execute one step, return observation + reward + done
  POST /step_multi     → multi-agent step (all agents at once)
  GET  /state          → full world state snapshot
  GET  /observations   → per-agent observations
  POST /scenario       → switch scenario for next reset (curriculum)
  GET  /health         → liveness check
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr

# ── Flat HuggingFace Space layout: all files are in /app ───────────────────
from environment import EnterpriseEnvironment
from models import EnterpriseAction, EnterpriseObservation
from gradio_app import demo

# ---------------------------------------------------------------------------
# Pydantic request/response models for FastAPI serialisation
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    scenario: Optional[str] = None
    seed: Optional[int] = None
    use_trained_model: bool = False


class ActionRequest(BaseModel):
    agent_id: str
    tool_call: Optional[str] = None
    tool_params: dict[str, Any] = {}
    message_to: Optional[str] = None
    message_content: Optional[str] = None
    reasoning: Optional[str] = None
    use_trained_model: bool = False


class MultiActionRequest(BaseModel):
    actions: dict[str, ActionRequest]


class ScenarioRequest(BaseModel):
    scenario_name: str


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EnterpriseOps Arena",
    description="Multi-agent RL environment — OpenEnv server",
    version="1.0.0",
)

env = EnterpriseEnvironment()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "environment": "EnterpriseOps Arena"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    """Reset the environment and return the primary observation."""
    obs = env.reset(
        scenario=req.scenario,
        seed=req.seed,
        use_trained_model=req.use_trained_model,
    )
    return {
        "observation": obs.to_dict(),
        "it_agent_status": env.it_agent_status(),
    }


@app.post("/step", response_model=StepResponse)
def step(req: ActionRequest) -> StepResponse:
    """Execute a single-agent step."""
    action = EnterpriseAction(
        agent_id=req.agent_id,
        tool_call=req.tool_call,
        tool_params=req.tool_params,
        message_to=req.message_to,
        message_content=req.message_content,
        reasoning=req.reasoning,
    )
    result = env.step(action, use_trained_model=req.use_trained_model)
    return StepResponse(
        observation=result["observation"].to_dict(),
        reward=result["reward"],
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/step_multi")
def step_multi(req: MultiActionRequest) -> dict[str, Any]:
    """Execute a multi-agent step (all agents in one call)."""
    actions: dict[str, EnterpriseAction] = {}
    for aid, a in req.actions.items():
        actions[aid] = EnterpriseAction(
            agent_id=aid,
            tool_call=a.tool_call,
            tool_params=a.tool_params,
            message_to=a.message_to,
            message_content=a.message_content,
            reasoning=a.reasoning,
        )
    result = env.step_multi(actions)

    obs_dicts: dict[str, Any] = {}
    for aid, obs in result.get("observations", {}).items():
        obs_dicts[aid] = obs.to_dict() if hasattr(obs, "to_dict") else obs

    return {
        "observations": obs_dicts,
        "reward": result["reward"],
        "done": result["done"],
        "info": result.get("info", {}),
    }


@app.get("/state")
def get_state() -> dict[str, Any]:
    """Return the full world state snapshot."""
    return env.state


@app.get("/observations")
def get_observations() -> dict[str, Any]:
    """Return per-agent observations from the last step."""
    all_obs = env.get_all_observations()
    return {aid: obs.to_dict() for aid, obs in all_obs.items()}


@app.post("/scenario")
def set_scenario(req: ScenarioRequest) -> dict[str, str]:
    """Switch scenario for the next reset() call (curriculum support)."""
    env.set_scenario(req.scenario_name)
    return {"status": "ok", "scenario": req.scenario_name}

# Mount Gradio AFTER all API routes are defined
app = gr.mount_gradio_app(app, demo, path="/")
