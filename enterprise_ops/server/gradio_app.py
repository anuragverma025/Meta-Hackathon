from __future__ import annotations

import json
from typing import Any

import gradio as gr
import requests

BASE_URL = "http://localhost:7860"
TIMEOUT = 45

SCENARIO_CHOICES = [
    ("Scenario 1", "scenario_01"),
    ("Scenario 2", "scenario_02"),
    ("Scenario 3", "scenario_03"),
    ("Scenario 4", "scenario_04"),
    ("Scenario 5", "scenario_05"),
    ("Scenario 6", "scenario_06"),
    ("Scenario 7", "scenario_07"),
    ("Scenario 8", "scenario_08"),
]

AGENT_CHOICES = [
    ("IT Agent", "it_agent"),
    ("Manager Agent", "manager_agent"),
    ("Finance Agent", "finance_agent"),
    ("Oversight Agent", "oversight_agent"),
]

TOOL_CHOICES = [
    ("Get Tickets", "get_tickets"),
    ("Resolve Ticket", "resolve_ticket"),
    ("Allocate Resource", "allocate_resource"),
    ("Approve Budget", "approve_budget"),
    ("Get Project Status", "get_project_status"),
]

TOOL_PARAM_PRESETS = {
    "get_tickets": "{}",
    "resolve_ticket": '{\n  "ticket_id": ""\n}',
    "allocate_resource": '{\n  "resource_type": "engineers",\n  "amount": 1,\n  "requester_agent": ""\n}',
    "approve_budget": '{\n  "amount": 1000,\n  "justification": "",\n  "requester_agent": "",\n  "manager_countersign": false\n}',
    "get_project_status": "{}",
}


def _pretty(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=f"{BASE_URL}{path}",
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _reset_episode(scenario_name: str) -> tuple[str, str, str]:
    data = _request("post", "/reset", {"scenario": scenario_name})
    observation = data.get("observation", {})
    formatted = _pretty(observation)
    return formatted, formatted, "Active"


def _step_episode(
    agent_id: str,
    tool_call: str,
    tool_params_json: str,
    message_to: str,
    message_content: str,
    reasoning: str,
) -> tuple[str, str, str, str]:
    try:
        tool_params = json.loads(tool_params_json) if tool_params_json.strip() else {}
        if not isinstance(tool_params, dict):
            raise ValueError("Tool params must decode to a JSON object.")
    except Exception as exc:
        error_text = f"Invalid tool params JSON: {exc}"
        return error_text, error_text, "0.0", "Active"

    payload = {
        "agent_id": agent_id,
        "tool_call": tool_call or None,
        "tool_params": tool_params,
        "message_to": message_to or None,
        "message_content": message_content or None,
        "reasoning": reasoning or None,
    }
    data = _request("post", "/step", payload)
    observation = data.get("observation", {})
    formatted = _pretty(observation)
    reward = f"{data.get('reward', 0.0):.3f}"
    status = "Done" if data.get("done") else "Active"
    return formatted, formatted, reward, status


def _load_world_state() -> str:
    data = _request("get", "/state")
    return _pretty(data)


def _preset_tool_params(tool_call: str) -> str:
    return TOOL_PARAM_PRESETS.get(tool_call, "{}")


with gr.Blocks(theme=gr.themes.Monochrome(), title="EnterpriseOps Arena - Meta PyTorch OpenEnv Hackathon") as demo:
    gr.Markdown(
        """
        # EnterpriseOps Arena - Meta PyTorch OpenEnv Hackathon

        **Themes:** OpenEnv Themes 1 and 3.1  
        **Bonus Prizes:** Fleet AI, Halluminate, Scale AI, Scaler AI Labs, Patronus AI  
        **Team Names:** Hackathon teams and contributors
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Reset Panel")
            scenario = gr.Dropdown(
                choices=SCENARIO_CHOICES,
                value="scenario_01",
                label="Scenario",
            )
            reset_button = gr.Button("Reset Episode", variant="primary")
            reset_observation = gr.Textbox(label="Observation", lines=12, interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("## Step Panel")
            agent_id = gr.Dropdown(
                choices=AGENT_CHOICES,
                value="it_agent",
                label="Agent",
            )
            tool_call = gr.Dropdown(
                choices=TOOL_CHOICES,
                value="get_tickets",
                label="Tool",
            )
            tool_params = gr.Textbox(
                label="Tool params JSON",
                lines=8,
                value=_preset_tool_params("get_tickets"),
            )
            message_to = gr.Textbox(label="Message To", placeholder="manager_agent")
            message_content = gr.Textbox(label="Message Content", lines=3)
            reasoning = gr.Textbox(label="Reasoning", lines=3)
            step_button = gr.Button("Step Episode", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Results Panel")
            result_observation = gr.Textbox(label="Observation", lines=12, interactive=False)
            reward_score = gr.Textbox(label="Reward Score", value="0.0", interactive=False)
            episode_status = gr.Textbox(label="Episode Status", value="Active", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("## World State")
            state_button = gr.Button("Load World State", variant="secondary")
            world_state = gr.Textbox(label="State", lines=20, interactive=False)

    tool_call.change(
        fn=_preset_tool_params,
        inputs=tool_call,
        outputs=tool_params,
    )
    reset_button.click(
        fn=_reset_episode,
        inputs=scenario,
        outputs=[reset_observation, result_observation, episode_status],
    )
    step_button.click(
        fn=_step_episode,
        inputs=[agent_id, tool_call, tool_params, message_to, message_content, reasoning],
        outputs=[reset_observation, result_observation, reward_score, episode_status],
    )
    state_button.click(fn=_load_world_state, inputs=None, outputs=world_state)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
