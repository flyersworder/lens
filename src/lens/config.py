"""LENS configuration management.

Config is stored as YAML at ~/.lens/config.yaml by default.
Missing keys fall back to defaults. Nested keys are accessed with dot notation.
Environment variables from .env are loaded automatically.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env from project root (if present) so API keys are available
load_dotenv()

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
    "embeddings": {
        "provider": "local",
        "model": "specter2",
        "dimensions": 1536,
        "api_base": "",
        "api_key": "",
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


logger = logging.getLogger(__name__)

_VALID_EMBEDDING_PROVIDERS = {"local", "cloud"}


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate config values and return list of warnings. Also logs each warning."""
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)
        logger.warning(msg)

    # LLM model names
    llm = config.get("llm", {})
    if not llm.get("default_model"):
        _warn("llm.default_model is empty — LLM calls will fail")
    if not llm.get("extract_model"):
        _warn("llm.extract_model is empty — extraction will fail")

    # Embedding provider
    emb = config.get("embeddings", {})
    provider = emb.get("provider", "")
    if provider not in _VALID_EMBEDDING_PROVIDERS:
        _warn(
            f"embeddings.provider '{provider}' is not recognized"
            f" (expected one of: {', '.join(sorted(_VALID_EMBEDDING_PROVIDERS))})"
        )

    # Embedding dimensions
    dims = emb.get("dimensions", 0)
    if not isinstance(dims, int) or dims <= 0:
        _warn("embeddings.dimensions must be a positive integer")

    # Storage data dir
    storage = config.get("storage", {})
    data_dir = storage.get("data_dir", "")
    if data_dir:
        try:
            Path(data_dir).expanduser()
        except RuntimeError:
            _warn(f"storage.data_dir '{data_dir}' is not a valid path")
    else:
        _warn("storage.data_dir is empty")

    # Arxiv categories
    categories = config.get("acquire", {}).get("arxiv_categories", [])
    if not isinstance(categories, list) or len(categories) == 0:
        _warn("acquire.arxiv_categories must be a non-empty list")

    # Monitor settings
    mon = config.get("monitor", {})
    top_n = mon.get("ideate_top_n", 0)
    if not isinstance(top_n, int) or top_n <= 0:
        _warn("monitor.ideate_top_n must be a positive integer")

    gap_score = mon.get("ideate_min_gap_score", 0.0)
    if not isinstance(gap_score, (int, float)) or gap_score < 0.0 or gap_score > 1.0:
        _warn("monitor.ideate_min_gap_score must be between 0.0 and 1.0")

    return warnings


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = default_config()
    if Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    validate_config(cfg)
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
