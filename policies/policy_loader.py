"""
Policy Loader — Reads YAML policy-as-code files.

Provides functions to load and cache policy configurations so that
guardrails can reference rules without hardcoding thresholds.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Module-level cache for loaded policies
_policy_cache: Dict[str, Dict[str, Any]] = {}

# Default policy file path (relative to project root)
DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default_policy.yaml"
)


def load_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML policy file from disk.

    Args:
        policy_path: Absolute or relative path to the YAML file.
                     Defaults to policies/default_policy.yaml.

    Returns:
        Dictionary containing the parsed policy configuration.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if policy_path is None:
        policy_path = DEFAULT_POLICY_PATH

    resolved = str(Path(policy_path).resolve())

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Policy file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    if not isinstance(policy, dict):
        raise ValueError(f"Policy file must contain a YAML mapping, got {type(policy)}")

    # Cache for subsequent calls
    _policy_cache[resolved] = policy

    return policy


def get_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a policy, using cache if available.

    Args:
        policy_path: Path to the policy file. Defaults to default_policy.yaml.

    Returns:
        Cached or freshly-loaded policy dictionary.
    """
    if policy_path is None:
        policy_path = DEFAULT_POLICY_PATH

    resolved = str(Path(policy_path).resolve())

    if resolved in _policy_cache:
        return _policy_cache[resolved]

    return load_policy(resolved)


def reload_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Force-reload a policy from disk, bypassing cache.

    Args:
        policy_path: Path to the policy file.

    Returns:
        Freshly-loaded policy dictionary.
    """
    if policy_path is None:
        policy_path = DEFAULT_POLICY_PATH

    resolved = str(Path(policy_path).resolve())

    # Clear cache entry
    _policy_cache.pop(resolved, None)

    return load_policy(resolved)


def clear_cache() -> None:
    """Clear all cached policies."""
    _policy_cache.clear()
