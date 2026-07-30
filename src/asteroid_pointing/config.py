"""JSON configuration loading for the standalone pointing tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_pointing_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    schema_path = Path(data.get("$schema", ""))
    if not schema_path.is_absolute():
        schema_path = (config_path.parent / schema_path).resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(f"JSON schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as stream:
        schema = json.load(stream)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data),
        key=lambda error: list(error.path),
    )
    if errors:
        messages = [
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: "
            f"{error.message}"
            for error in errors
        ]
        raise ValueError("Invalid pointing configuration:\n" + "\n".join(messages))
    return data
