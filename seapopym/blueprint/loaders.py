"""File loaders for Blueprint and Config.

This module provides utility functions for loading YAML and JSON files
with automatic format detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary from the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary at root level, got {type(data).__name__}")

    return data


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary at root level, got {type(data).__name__}")

    return data


def load_file(path: str | Path) -> dict[str, Any]:
    """Load a file with automatic format detection based on extension.

    Supported extensions:
    - .yaml, .yml: YAML format
    - .json: JSON format

    Args:
        path: Path to the file.

    Returns:
        Parsed dictionary from the file.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return load_yaml(path)
    elif suffix == ".json":
        return load_json(path)
    else:
        # Try YAML as default (more permissive)
        try:
            return load_yaml(path)
        except yaml.YAMLError:
            raise ValueError(f"Unknown file format for '{path}'. Supported: .yaml, .yml, .json") from None


def detect_format(source: str | Path | dict[str, Any]) -> str:
    """Detect the format of a source.

    Args:
        source: File path or dictionary.

    Returns:
        Format string: "dict", "yaml", "json", or "unknown".
    """
    if isinstance(source, dict):
        return "dict"

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return "yaml"
    elif suffix == ".json":
        return "json"
    else:
        return "unknown"
