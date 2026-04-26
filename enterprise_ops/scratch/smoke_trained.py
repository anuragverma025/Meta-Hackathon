import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

print("[Smoke] Importing TrainedITAgent...")
from agents.trained_agent import TrainedITAgent

print("[Smoke] Constructing TrainedITAgent()...")
a = TrainedITAgent()

print(f"Model loaded: {a.model is not None}")
print(f"Tokenizer loaded: {a.tokenizer is not None}")

if a.model is None:
    print("[Smoke] Model did NOT load - rule-based fallback active")
    print("[Smoke] This is OK on CPU if transformers/peft/bitsandbytes aren't installed")
else:
    print("[Smoke] Model loaded successfully!")

# Test rule-based fallback works regardless
from contracts import ObservationSchema, TicketItem

obs = ObservationSchema(
    agent_id="it_agent",
    step_number=0,
    tickets=[
        TicketItem(id="TKT-001", priority=1, description="Test ticket", sla_steps_remaining=3),
    ],
)

action = a.act(obs)
print(f"Action: tool_call={action.tool_call}, params={action.tool_params}")
print("PASS")
