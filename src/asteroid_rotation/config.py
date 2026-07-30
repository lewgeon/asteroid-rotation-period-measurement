"""Strict JSON configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration and its source path."""

    data: Dict[str, Any]
    path: Path
    schema_path: Path

    def section(self, name: str) -> Dict[str, Any]:
        return self.data[name]


def _resolve_schema_path(config_path: Path, data: Dict[str, Any]) -> Path:
    schema_value = data.get("$schema")
    if not schema_value:
        raise ValueError("Configuration must declare a local $schema path")
    schema_path = Path(schema_value)
    if not schema_path.is_absolute():
        schema_path = (config_path.parent / schema_path).resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"JSON schema not found: {schema_path}")
    return schema_path


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a JSON experiment configuration."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    schema_path = _resolve_schema_path(config_path, data)
    with schema_path.open("r", encoding="utf-8") as stream:
        schema = json.load(stream)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda item: list(item.path))
    if errors:
        messages = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            messages.append(f"{location}: {error.message}")
        raise ValueError("Invalid experiment configuration:\n" + "\n".join(messages))

    processing = data["processing"]
    if processing["period_min_s"] >= processing["period_max_s"]:
        raise ValueError("processing.period_min_s must be smaller than period_max_s")

    observation = data["observation"]
    if processing["period_min_s"] > observation["duration_s"]:
        raise ValueError("The observation is shorter than processing.period_min_s")

    return ExperimentConfig(data=data, path=config_path, schema_path=schema_path)

