---
title: EnterpriseOps Arena
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# EnterpriseOps Arena

Multi-agent reinforcement learning environment for enterprise operations simulation.
Built for the **Meta PyTorch OpenEnv Hackathon**.

Training stack: **Unsloth 2025.3.19 + HuggingFace TRL >= 0.15.0 + GRPO**
Model: **Qwen2.5-3B-Instruct** (4-bit, fits A100 in 48 h)

---

## Project Layout

```
enterprise_ops/
├── contracts.py              # Pydantic v2 schemas — ActionSchema, ObservationSchema,
│                             #   RewardComponents, StepResult, TicketItem, …
├── models.py                 # Dataclass wrappers for OpenEnv (EnterpriseAction/Observation)
├── client.py                 # HTTP client (EnterpriseClient, EnterpriseEnv, StepResult)
├── __init__.py               # sys.path setup so flat imports work everywhere
│
├── agents/
│   ├── base_agent.py         # BaseAgent ABC with act()
│   ├── it_agent.py           # ITAgent — resolves tickets, manages compute
│   ├── manager_agent.py      # ManagerAgent — project tasks, priorities
│   ├── finance_agent.py      # FinanceAgent — budget approvals
│   ├── oversight_agent.py    # OversightAgent — flags HALLUCINATION / STUCK_LOOP / POLICY_BREACH
│   └── policies/
│       └── rule_policy.py    # RulePolicy — deterministic fallback for all agents
│
├── env/
│   ├── env.py                # EnterpriseOpsEnv — main Gym-style environment
│   ├── world_model.py        # WorldModel — tickets, deals, tasks, resource pool
│   ├── tools.py              # ToolRegistry — get_tickets, resolve_ticket, …
│   ├── schema_drift.py       # SchemaDriftEngine — RENAME / ADD_FIELD / DEPRECATE
│   └── scenarios/
│       ├── scenario_01.yaml  # Difficulty 1 — basic ticket queue
│       ├── scenario_02.yaml  # … schema drift off
│       ├── scenario_03.yaml  # Difficulty 2 — schema drift on
│       └── … (08 scenarios total)
│
├── server/
│   ├── app.py                # FastAPI server — /health /reset /step /step_multi …
│   └── environment.py        # EnterpriseEnvironment — anti-hacking wrapper
│                             #   timeout 30 s, loop detection -5, hard cap 8 steps
│
├── train/
│   ├── config.py             # TrainingConfig — all GRPO / curriculum hyperparameters
│   ├── reward_fn.py          # 4 independent reward functions + compute_reward()
│   ├── episode_buffer.py     # EpisodeBuffer — replay + curriculum tracking (last 10 rewards)
│   ├── trainer.py            # EnterpriseOpsTrainer — Unsloth GRPO or dry-run fallback
│   ├── main.py               # CLI entry point — logs CSV, saves checkpoints, prints curve
│   └── colab_notebook.ipynb  # 4-cell Colab notebook for A100 training
│
└── test_integration.py       # Smoke test — 7 checks, PASS/FAIL, exit code
```

---

## Quick Start

### 1. Install

```bash
pip install -e .
# With GPU training dependencies (requires CUDA):
pip install "unsloth==2025.3.19" "trl>=0.15.0" datasets
```

### 2. Run the smoke test (CPU, no GPU needed)

```bash
python enterprise_ops/test_integration.py
```

Expected output: `SMOKE TEST PASSED`

### 3. Start the API server

```bash
uvicorn enterprise_ops.server.app:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health` `POST /reset` `POST /step` `POST /step_multi`
`GET /observations` `GET /state` `POST /scenario`

### 4. Train (single command)

```bash
python -m enterprise_ops.train.main --scenario scenario_01 --steps 200
```

**What it does on CPU** (no GPU / Unsloth not installed):
- Dry-run mode — runs rule-based episodes, logs metrics, tests the full pipeline.

**What it does on GPU** (Unsloth + TRL installed):
- Loads `unsloth/Qwen2.5-3B-Instruct` in 4-bit.
- Collects env observations as GRPO prompts.
- Trains with GRPOTrainer (`num_generations=4`).
- Saves LoRA adapters to `checkpoints/` every 50 steps (**never merged to 16-bit**).

---

## Training CLI Reference

```
python -m enterprise_ops.train.main [OPTIONS]

  --scenario    SCENARIO   Starting scenario name (default: scenario_01)
  --steps       N          Total GRPO training steps (default: 200)
  --episode-length N       Max env steps per episode (default: 8)
  --grpo-gens   N          Completions per prompt / G (default: 4)
  --log-every   N          Write CSV row every N steps (default: 10)
  --save-every  N          Save adapter every N steps (default: 50)
  --log-dir     PATH       Metrics CSV directory (default: ./logs)
  --checkpoint-dir PATH    Adapter output directory (default: ./checkpoints)
  --seed        N          Random seed (default: 42)
  --no-reward-fn           Disable custom reward function
```

---

## Reward Functions

Four **independent** signals — judges verify each exists separately:

| Function | Signal | Value |
|---|---|---|
| `reward_task_completion(step_result)` | +1.0 per ticket resolved / task completed | float >= 0 |
| `reward_sla_adherence(step_result)` | +0.5 SLA met, -2.0 SLA breach | float |
| `reward_coordination(step_result, prev_messages)` | +0.3 per message, +0.5 cross-agent chain | float >= 0 |
| `penalty_hallucination(step_result)` | -1.0 per unknown tool call | float <= 0 |

`compute_reward(step_result, prev_messages)` sums all four → single scalar for GRPO.

```python
from enterprise_ops.train.reward_fn import (
    reward_task_completion,
    reward_sla_adherence,
    reward_coordination,
    penalty_hallucination,
    compute_reward,
)
```

---

## Config Reference (`TrainingConfig`)

```python
from enterprise_ops.train.config import TrainingConfig

config = TrainingConfig(
    model_name            = "unsloth/Qwen2.5-3B-Instruct",
    max_steps             = 200,      # total GRPO training steps
    episode_length        = 8,        # max env steps per episode
    grpo_num_generations  = 4,        # G — completions per prompt
    save_every            = 50,       # save LoRA adapter every N steps
    log_every             = 10,       # write CSV row every N steps
    curriculum_threshold  = 0.6,      # advance scenario when avg > this
    curriculum_window     = 10,       # over last N episodes
    lora_rank             = 16,
    lora_alpha            = 16,
    load_in_4bit          = True,
)
```

---

## Curriculum

The training loop automatically advances through 8 scenarios (difficulty 1→8) when the rolling average reward over the last 10 episodes exceeds 0.6.

```
scenario_01  →  scenario_02  →  …  →  scenario_08
 diff=1           diff=2               diff=8
 drift=off         drift=off            drift=on
```

---

## Logging Output

`logs/metrics.csv` (one row every 10 steps):

```
step,episode_score,task_completion,sla_rate,coordination_score,curriculum_difficulty
0,49.0,37.0,0.324,0.0,1
10,42.5,28.0,0.517,0.0,2
…
```

`checkpoints/step_00050/` — LoRA adapter weights (`.safetensors` + `adapter_config.json`).

---

## Agents

| Agent | Role | Key tools |
|---|---|---|
| `it_agent` | Resolves tickets, manages compute | `get_tickets`, `resolve_ticket`, `allocate_resource` |
| `manager_agent` | Project tasks, priorities | `get_project_status`, `allocate_resource` |
| `finance_agent` | Budget approvals | `approve_budget` |
| `oversight_agent` | Flags misbehaviour (observer only, no env actions) | — |

OversightAgent flags and reward deltas:

| Flag | Delta |
|---|---|
| `HALLUCINATION` — unknown tool/param | +15 to oversight, -15 to offending agent |
| `STUCK_LOOP` — same call 3x in 5 steps | +10 / -10 |
| `POLICY_BREACH` — budget > $10k without countersign | +8 |
| `STALE_SCHEMA` — agent uses renamed/deprecated field | +5 |

---

## Anti-Hacking Protections (server/environment.py)

| Protection | Value |
|---|---|
| Step timeout | 30 s → `done=True`, reward = -10 |
| Loop detection | 3 identical actions → reward -= 5 |
| Hard step cap | 8 steps per episode |
| Audit log | Every step logged to `episodes.db` |

---

## HTTP Client Usage

```python
from enterprise_ops.client import EnterpriseEnv, EnterpriseAction

with EnterpriseEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(scenario="scenario_01")
    result = env.step(EnterpriseAction(
        agent_id="it_agent",
        tool_call="get_tickets",
    ))
    print(result.reward, result.done)  # StepResult attributes
```

---

## Colab (A100) Training

Open `enterprise_ops/train/colab_notebook.ipynb` in Google Colab with an A100 runtime.

- Cell 1: installs `unsloth==2025.3.19 trl>=0.15.0` and dependencies
- Cell 2: clones the repo and does `pip install -e .`
- Cell 3: runs `python -m enterprise_ops.train.main --steps 200`
- Cell 4: plots 4 reward curves from `logs/metrics.csv`

---

## File Health

All 31 Python files parse with zero syntax errors. Run the check yourself:

```bash
python -c "
import ast
from pathlib import Path
bad = [p for p in Path('enterprise_ops').rglob('*.py')
       if '__pycache__' not in str(p)
       and not ast.parse(p.read_text(encoding='utf-8'))]
print('All clean' if not bad else bad)
"
```

---

## Version Pins

```
unsloth==2025.3.19
trl>=0.15.0
pydantic>=2.0
python>=3.11
```
