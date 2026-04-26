# EnterpriseOps Arena
### Multi-Agent RL Environment for Enterprise Coordination

> Teaching LLMs to coordinate like a real enterprise team.

## Quick Links
- 🚀 HF Space: https://huggingface.co/spaces/Anurag137/enterprise-ops-arena
- 🤖 Trained Model: https://huggingface.co/Anurag137/enterprise-ops-lora
- 📊 Wandb: https://wandb.ai/kanhaiyakumar76618-indian-institute-of-information-techn/enterprise-ops-arena
- 📝 Blog: https://github.com/anuragverma025/Meta-Hackathon/blob/main/BLOG.md
- 💻 GitHub: https://github.com/anuragverma025/Meta-Hackathon

## The Problem
Picture this: IT resolves a critical server ticket.
But Finance blocked the budget 2 steps earlier.
The ticket re-opens. SLA breaches. Customer escalates.

Each agent acted correctly in isolation.
Together they failed. This is the coordination gap
we are training agents to close.

## What We Built
4 specialized LLM agents coordinating in a simulated enterprise:

| Agent | Role |
|-------|------|
| IT Agent | Resolves tickets, manages resources |
| Manager Agent | Allocates resources, coordinates teams |
| Finance Agent | Approves budgets, blocks violations |
| Oversight Agent | Monitors all agents, catches hallucinations |

## What Makes It Hard
- **Partial observability** — IT cannot see Finance decisions
- **Schema drift** — API fields mutate every 20 steps
- **SLA pressure** — tickets expire in real time
- **12% noise** — random tool failures at max difficulty
- **8 difficulty levels** — automatic curriculum advancement

## Reward Design (7 components, arXiv:2601.19100)
1. Potential-based shaping — dependency graph progress
2. Dynamic weight optimization — BiPaRS rebalancing
3. Urgency-scaled SLA — time-dependent deadline rewards
4. Exploration bonus — EXPLORS intrinsic reward
5. Schema adaptation — explicit post-drift field usage reward
6. Process reward — PRM step-level supervision
7. Trajectory reward — trend and consistency bonus

## Training Results
![Reward curves](reward_curves.png)

| Metric | Value |
|--------|-------|
| Peak episode score | **114** (+77%) |
| Task completion | **35 → 75** (+114%) |
| GRPO reward_std | **0.5** (variance confirmed) |
| Scenarios completed | **All 8** automatically |
| Backtracking | Triggered 2x (MARL adaptive) |
| Total steps | 700 across 3 runs |
| GPU | Tesla T4 |
| Model | Qwen2.5-3B-Instruct 4-bit LoRA |

## Before vs After Training

**Prompt:** IT Agent. TKT-001, P1, SLA=2 steps. What do you do?

**Before training:**
```json
{"tool_call":"Assign Engineer to Ticket",
 "tool_params":{"engineer":"Engineer 1"}}
```
❌ Wrong tool name | ❌ Missing ticket_id | ❌ No reasoning

**After 700 steps GRPO:**
```json
{"tool_call":"resolve_ticket",
 "tool_params":{"ticket_id":"TKT-001","engineer":"Engineer 1"},
 "reasoning":"P1, SLA=2 steps remaining, resolve immediately"}
```
✅ Correct tool | ✅ ticket_id included | ✅ SLA-aware reasoning

## Tech Stack
| Component | Choice | Why |
|-----------|--------|-----|
| Model | Qwen2.5-3B-Instruct | Enterprise knowledge, JSON following |
| Training | GRPO via TRL | No critic needed, fits T4 GPU |
| Quantization | Unsloth 4-bit | 2x faster training |
| Reward | 7-component research | arXiv:2601.19100 |
| Curriculum | MARL adaptive backtracking | Prevents policy collapse |

## Project Structure
enterprise_ops/
├── contracts.py       — Pydantic schemas + agent constants
├── agents/            — IT, Manager, Finance, Oversight agents
├── env/               — Environment, tools, world model, schema drift
│   └── scenarios/     — 8 difficulty scenarios
├── server/            — FastAPI + Gradio deployment
└── train/             — GRPO training pipeline + reward functions

## Bonus Prize Coverage
- **Patronus AI** — Schema drift engine forces real API adaptation
- **Fleet AI** — OversightAgent monitors all agents every step
