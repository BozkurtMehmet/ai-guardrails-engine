"""
Constitution Loader — Reads the AI Constitutional Policy YAML.

Provides functions to load and cache the constitution so that
the enforcement layer can validate decisions against governance principles.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Module-level cache
_constitution_cache: Optional[Dict[str, Any]] = None

DEFAULT_CONSTITUTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ai_constitution.yaml"
)


def load_constitution(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the AI constitution from a YAML file.

    Args:
        path: Path to the constitution file. Defaults to ai_constitution.yaml.

    Returns:
        Parsed constitution dictionary.
    """
    global _constitution_cache

    if path is None:
        path = DEFAULT_CONSTITUTION_PATH

    resolved = str(Path(path).resolve())

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Constitution file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        constitution = yaml.safe_load(f)

    if not isinstance(constitution, dict):
        raise ValueError("Constitution file must contain a YAML mapping")

    _constitution_cache = constitution
    return constitution


def get_constitution(path: Optional[str] = None) -> Dict[str, Any]:
    """Get the constitution, using cache if available."""
    global _constitution_cache

    if _constitution_cache is not None:
        return _constitution_cache

    return load_constitution(path)


def reload_constitution(path: Optional[str] = None) -> Dict[str, Any]:
    """Force-reload the constitution from disk."""
    global _constitution_cache
    _constitution_cache = None
    return load_constitution(path)
