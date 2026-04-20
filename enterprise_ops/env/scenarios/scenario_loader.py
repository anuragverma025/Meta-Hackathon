"""
scenario_loader.py — Loads scenario YAML files into plain dicts for WorldModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


class ScenarioLoader:
    """
    Loads and validates scenario YAML files.

    Args:
        scenarios_dir: Directory containing scenario_01.yaml through scenario_08.yaml.
                       Defaults to the same directory as this file.
    """

    _REQUIRED_KEYS = {
        "difficulty", "starting_tickets", "starting_deals",
        "starting_tasks", "resource_pool", "episode_length",
        "schema_drift_enabled", "noise_rate", "description",
    }

    def __init__(self, scenarios_dir: Optional[str] = None) -> None:
        self._dir = Path(scenarios_dir) if scenarios_dir else Path(__file__).parent

    def load(self, path_or_name: str) -> dict[str, Any]:
        """
        Load a scenario by file path or short name (e.g. 'scenario_01').
        Returns the scenario dict ready to pass to WorldModel.load_scenario().
        """
        path = Path(path_or_name)

        if not path.is_absolute() and not path.exists():
            stem = path.stem if path.suffix else path.name
            path = self._dir / f"{stem}.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)

        self._validate(data, path)
        return data

    def list_scenarios(self) -> list[Path]:
        """Return all scenario YAML paths in the scenarios directory, sorted."""
        return sorted(self._dir.glob("scenario_*.yaml"))

    def _validate(self, data: dict[str, Any], path: Path) -> None:
        missing = self._REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Scenario {path.name} missing keys: {missing}")
