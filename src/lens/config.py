"""LENS configuration management.

Config is stored as YAML at ~/.lens/config.yaml by default.
Missing keys fall back to defaults. Nested keys are accessed with dot notation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "default_model": "openrouter/anthropic/claude-sonnet-4-6",
        "extract_model": "openrouter/google/gemini-2.5-flash",
        "label_model": "openrouter/anthropic/claude-sonnet-4-6",
        "api_base": "",
        "api_key": "",
    },
    "acquire": {
        "arxiv_categories": ["cs.CL", "cs.LG", "cs.AI"],
        "openalex_mailto": "",
        "quality_min_citations": 0,
        "quality_venue_tiers": {
            "tier1": ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"],
            "tier2": ["AAAI", "NAACL", "EACL", "COLING"],
        },
    },
    "taxonomy": {
        "target_parameters": 25,
        "target_principles": 35,
        "target_arch_variants": 20,
        "target_agentic_patterns": 15,
        "min_cluster_size": 3,
        "embedding_provider": "local",
        "embedding_model": "specter2",
        "embedding_dim": 768,
    },
    "monitor": {
        "ideate": True,
        "ideate_llm": False,
        "ideate_top_n": 10,
        "ideate_min_gap_score": 0.5,
    },
    "storage": {
        "data_dir": "~/.lens/data",
    },
}

DEFAULT_CONFIG_PATH = Path("~/.lens/config.yaml").expanduser()


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = default_config()
    if Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    path = config_path or DEFAULT_CONFIG_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _coerce_value(value: str) -> Any:
    """Coerce a string value to its most likely Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def set_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    if isinstance(value, str):
        value = _coerce_value(value)
    keys = dotted_key.split(".")
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def resolve_data_dir(config: dict[str, Any]) -> str:
    return str(Path(config["storage"]["data_dir"]).expanduser())
