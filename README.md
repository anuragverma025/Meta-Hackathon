---
title: EnterpriseOps Arena
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# EnterpriseOps Arena 🏢

> Multi-agent RL environment where IT, Manager, Finance and Oversight agents 
> collaborate to manage a simulated enterprise under partial observability, 
> schema drift, and SLA pressure.

## Quick Links
- 🚀 **HuggingFace Space**: https://huggingface.co/spaces/Anurag137/enterprise-ops-arena
- 📓 **Colab Notebook**: https://github.com/anuragverma025/Meta-Hackathon/blob/main/enterprise_ops/train/colab_notebook.ipynb
- 📝 **Blog Post**: https://github.com/anuragverma025/Meta-Hackathon/blob/main/BLOG.md
- 💻 **GitHub**: https://github.com/anuragverma025/Meta-Hackathon

---

## The Problem

Enterprise AI agents fail because they work in silos. The IT agent 
resolving a critical server ticket does not know the Finance agent 
just blocked the budget it needs. The Manager does not know which 
tickets are about to breach SLA. No coordination = cascading failures.

We built an RL environment that trains LLM agents to coordinate 
across departments — developing theory-of-mind reasoning through 
reinforcement learning.

---

## The Environment

4 specialized LLM agents operate inside a simulated enterprise:

| Agent | Role | Sees |
|---|---|---|
| IT Agent | Resolves support tickets before SLA breach | Tickets + resource pool + inbox |
| Manager Agent | Allocates shared resources, coordinates tasks | All dept summaries + project tasks |
| Finance Agent | Approves budgets, blocks policy violations | Budget history + pending approvals |
| Oversight Agent | Monitors all agents, catches hallucinations | ALL tool call logs (full visibility) |

### Key environment features
- **Partial observability** — each agent sees only its department
- **5 mock enterprise APIs** — get_tickets, resolve_ticket, allocate_resource, approve_budget, get_project_status
- **Schema drift** — API fields mutate every 20 steps, forcing real adaptation
- **8 scenarios** — difficulty 1 to 8, from simple IT tasks to full enterprise chaos
- **Message bus** — agents coordinate by sending structured messages
- **Anti-reward-hacking** — timeout, loop detection, state locks, oversight monitoring

---

## Reward Design

4 independent reward functions (composable, hard to game):

| Function | Signal |
|---|---|
| task_completion | +10 per resolved ticket/task, verified by state diff |
| sla_adherence | +7.5 before deadline, -5 on breach |
| coordination_bonus | +6 when message leads to correct action next step |
| hallucination_penalty | -8 for calling non-existent API fields |

The Oversight Agent earns +15 for catching hallucinations, +8 for 
policy breaches, +5 for stale schema usage.

---

## Training Results

Real training run — 200 steps on T4 GPU — 32 minutes

![Reward curves](reward_curves.png)

![Loss curve](loss_curve.png)

### Key findings
- **GRPO reward**: -1.0 → +1.5 (crossed zero — model is learning)
- **Curriculum**: Advanced automatically from scenario_01 → scenario_03
- **Train loss**: -0.023
- **Model**: Qwen2.5-3B-Instruct, 4-bit quantized via Unsloth
- **Method**: GRPO via HuggingFace TRL

### What the curves show
- Episode score dropped at step 110 when curriculum advanced to harder scenario — agents were challenged
- Score recovered by step 200 — agents adapted
- Curriculum difficulty staircase shows automatic advancement — no human intervention
- GRPO reward crossed from negative to positive — proof of learning

---

## How to Run

### Run the environment locally
```bash
git clone https://huggingface.co/spaces/Anurag137/enterprise-ops-arena
cd enterprise-ops-arena
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Test the API
```bash
# Health check
curl https://anurag137-enterprise-ops-arena.hf.space/health

# View all endpoints
open https://anurag137-enterprise-ops-arena.hf.space/docs
```

### Run training
```bash
git clone https://github.com/anuragverma025/Meta-Hackathon
cd Meta-Hackathon/enterprise_ops
pip install -e .
python -m enterprise_ops.train.main --scenario scenario_01 --steps 200
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Environment | OpenEnv + FastAPI + SQLite |
| Schemas | Pydantic v2 |
| Training | HuggingFace TRL + GRPO |
| Model | Qwen2.5-3B-Instruct |
| Efficiency | Unsloth 4-bit quantization |
| Deployment | HuggingFace Spaces + Docker |
| UI | Gradio mounted on FastAPI |

---

## Themes Covered

- **Theme 1** — Multi-Agent Interactions
- **Theme 3.1** — World Modeling: Professional Tasks

### Bonus prizes targeted
- Fleet AI — Scalable Oversight (OversightAgent)
- Halluminate — Multi-Actor Environments
- Scale AI — Sales/PM/IT enterprise workflows
- Scaler AI Labs — Multi-app enterprise RL
- Patronus AI — Schema drift + dynamic contracts

---

## Project Structure
