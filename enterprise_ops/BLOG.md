# EnterpriseOps Arena: Teaching LLMs to Coordinate Like a Real Enterprise Team

## The Problem Nobody Is Solving

Every enterprise AI deployment fails the same way.

IT resolves a critical server ticket.
Finance blocked the budget 2 steps earlier.
The ticket re-opens. SLA breaches. Customer escalates.

Each agent acted correctly in isolation.
Together they failed.

Current LLM benchmarks test individual agents on individual tasks.
Nobody tests whether agents can coordinate under real enterprise
pressure — partial information, tight deadlines, changing APIs,
and shared scarce resources. We built the environment to train that.

## What We Built

EnterpriseOps Arena is a multi-agent RL environment where 4
specialized LLM agents must coordinate to run a simulated enterprise.
Each agent sees only its own department. They share one resource pool.
They communicate through a message bus. They succeed or fail together.

**The 4 agents:**
- IT Agent — resolves support tickets before SLA breach
- Manager Agent — allocates shared resources, coordinates
- Finance Agent — approves budgets, blocks policy violations
- Oversight Agent — monitors all agents, catches hallucinations

**What makes it hard:**
- Partial observability — IT cannot see Finance budget decisions
- Schema drift — API fields change every 20 steps silently
- 8 difficulty levels — from simple tickets to full enterprise chaos
- 12% noise at max difficulty — tool calls fail randomly
- SLA timers — tickets expire if not resolved in time

## Schema Drift — Our Most Original Contribution

Every 20 training steps the API schemas mutate.
A field called ticket_id becomes tkt_ref.
An agent that memorized field names fails immediately.
An agent that learned to adapt succeeds.

This forces genuine world model adaptation rather than memorization.
This is our Patronus AI angle — testing whether agents can handle
real API versioning pressure that every enterprise faces.

## The Reward Design

Based on arXiv:2601.19100 — 7 independent reward components:

1. Potential-based shaping — accelerates convergence on dependency tasks
2. BiPaRS dynamic weights — rebalances components when performance drops
3. Urgency-scaled SLA — higher reward for early P1 resolution
4. EXPLORS exploration bonus — intrinsic reward for novel tool sequences
5. Schema adaptation — explicit reward for correct post-drift field usage
6. PRM process reward — step-level supervision for credit assignment
7. Trajectory reward — consistency and trend bonus over episode

Anti-reward-hacking: OversightAgent penalizes hallucinations,
stuck loops, and policy violations. An agent exploiting the reward
without solving the task gets caught immediately.

## MARL Adaptive Curriculum

Standard curriculum RL only moves forward.
Our backtracking monitors GRPO reward variance in real time.
When variance collapses — all completions get the same score
and GRPO cannot learn — the system steps back one difficulty level.

This is a self-healing training loop.
It triggered twice during training and recovered episode score
from 79 to 112 both times.

## Training

Model: Qwen2.5-3B-Instruct, 4-bit quantized via Unsloth
Method: GRPO via HuggingFace TRL
Total: 700 steps across 3 training runs
GPU: Tesla T4

GRPO was chosen because it trains without a critic model —
essential when you only have 16GB VRAM. Qwen2.5-3B-Instruct
because it already understands enterprise concepts and follows
structured JSON instructions. Unsloth because it makes 4-bit
QLoRA training 2x faster through custom CUDA kernels.

## Results

- Episode score: 64.5 → 114 (+77%)
- Task completion: 35 → 75 (+114%)
- All 8 scenarios completed automatically
- GRPO reward_std: 0.5 (variance confirmed)
- Backtracking triggered 2x, recovered both times
- LoRA adapters: https://huggingface.co/Anurag137/enterprise-ops-lora

## Before vs After

Same prompt — P1 ticket, SLA=2 steps remaining:

Before training: Wrong tool name, missing ticket_id, no reasoning
After 700 steps: Correct tool, correct params, SLA-aware reasoning

The model learned what the environment actually requires.

## Why It Matters

Enterprise AI coordination is the next frontier.
Every company deploying agents will face exactly this problem.
EnterpriseOps Arena is the first RL environment designed
specifically to train theory-of-mind coordination in LLMs
for enterprise settings.

A researcher could write a paper about training on this.
We just did.

## Links
- HF Space: https://huggingface.co/spaces/Anurag137/enterprise-ops-arena
- Trained Model: https://huggingface.co/Anurag137/enterprise-ops-lora
- Wandb: https://wandb.ai/kanhaiyakumar76618-indian-institute-of-information-techn/enterprise-ops-arena
- Research: arXiv:2601.19100
