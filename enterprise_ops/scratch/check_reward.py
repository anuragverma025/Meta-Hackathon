import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.environment import EnterpriseEnvironment
from models import EnterpriseAction
import json

env = EnterpriseEnvironment()
env.reset()
res = env.step(EnterpriseAction(agent_id='it_agent', tool_call='get_tickets'))

print(f"Total Reward: {res.get('reward')}")
print(f"Breakdown: {json.dumps(res['observation'].reward_breakdown, indent=2)}")
