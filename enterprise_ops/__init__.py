"""
enterprise_ops — EnterpriseOps Arena package.

Exports the OpenEnv client and model classes for external use.
Existing modules (contracts, env, agents) remain importable via
their original flat import paths when CWD is enterprise_ops/.
"""

import sys
from pathlib import Path

# Add enterprise_ops/ to sys.path so flat imports (contracts, env, agents,
# models) resolve correctly whether this package is imported from a parent
# directory or from enterprise_ops/ itself.
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# agents/ uses bare imports (e.g. "from base_agent import BaseAgent")
_AGENTS_ROOT = _PKG_ROOT / "agents"
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))

try:
    from .models import EnterpriseAction, EnterpriseObservation
    from .client import EnterpriseClient
except ImportError:
    # Flat-import fallback (when CWD is enterprise_ops/)
    try:
        from models import EnterpriseAction, EnterpriseObservation
        from client import EnterpriseClient
    except ImportError:
        pass

__all__ = [
    "EnterpriseAction",
    "EnterpriseObservation",
    "EnterpriseClient",
]
