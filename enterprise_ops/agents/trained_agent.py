from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contracts import ActionSchema, ObservationSchema, AGENT_IT


class TrainedITAgent:
    """
    IT Agent powered by trained LoRA model from HuggingFace.
    Falls back to rule-based if model not available.
    """

    MODEL_REPO = "Anurag137/enterprise-ops-lora"
    BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"

    def __init__(self):
        self.agent_id = AGENT_IT
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            print("[TrainedAgent] Loading base model without Unsloth...")

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct"
            )

            # Try 4-bit quantisation (needs bitsandbytes); fall back to fp16
            load_kwargs: dict = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                print("[TrainedAgent] Using 4-bit quantisation")
            except (ImportError, Exception):
                print("[TrainedAgent] bitsandbytes not available, using fp16")

            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                **load_kwargs,
            )

            print("[TrainedAgent] Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                "Anurag137/enterprise-ops-lora"
            )
            self.tokenizer = tokenizer
            self.model.eval()
            print("[TrainedAgent] Model loaded successfully")

        except Exception as e:
            print(f"[TrainedAgent] Could not load model: {e}")
            print("[TrainedAgent] Falling back to rule-based")
            self.model = None
            self.tokenizer = None

    def act(self, obs: ObservationSchema) -> ActionSchema:
        if self.model is None:
            return self._rule_based_act(obs)

        try:
            tickets = obs.tickets or []

            obs_data = {
                "step": obs.step_number,
                "tickets": [
                    {
                        "id": t.id,
                        "priority": t.priority,
                        "sla_steps_remaining": t.sla_steps_remaining,
                        "resolved": t.resolved,
                    }
                    for t in tickets[:5]
                ],
            }

            system = (
                "You are the IT Agent in an enterprise operations environment. "
                "Resolve support tickets, manage compute resources. "
                "Available tools: get_tickets, resolve_ticket, allocate_resource. "
                'Respond ONLY with valid JSON: {"tool_call":"<name>","tool_params":{},'
                '"reasoning":"<why>"}'
            )

            if self.tokenizer is None:
                return self._rule_based_act(obs)

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(obs_data)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            if self.model is not None:
                self.model = self.model.to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return ActionSchema(
                    tool_call=d.get("tool_call"),
                    tool_params=d.get("tool_params", {}),
                    message_to=d.get("message_to"),
                    message_content=d.get("message_content"),
                )
        except Exception as e:
            print(f"[TrainedAgent] Inference error: {e}")

        return self._rule_based_act(obs)

    def _rule_based_act(self, obs: ObservationSchema) -> ActionSchema:
        tickets = obs.tickets or []
        unresolved = [t for t in tickets if not t.resolved]

        sla_critical = [t for t in unresolved if t.sla_steps_remaining <= 2]
        if sla_critical:
            target = min(sla_critical, key=lambda t: t.sla_steps_remaining)
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": f"SLA rescue: {target.id}",
                },
            )

        p1 = [t for t in unresolved if t.priority == 1]
        if p1:
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": p1[0].id,
                    "resolution_note": f"P1: {p1[0].id}",
                },
            )

        if unresolved:
            target = min(unresolved, key=lambda t: (t.priority, t.sla_steps_remaining))
            return ActionSchema(
                tool_call="resolve_ticket",
                tool_params={
                    "ticket_id": target.id,
                    "resolution_note": f"Resolving: {target.id}",
                },
            )

        return ActionSchema(tool_call="get_tickets", tool_params={})
