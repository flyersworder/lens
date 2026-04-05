"""Tests for LENS configuration management."""

import yaml

from lens.config import validate_config


def test_default_config():
    from lens.config import default_config

    cfg = default_config()
    assert cfg["llm"]["default_model"] == "openrouter/anthropic/claude-sonnet-4-6"
    assert cfg["llm"]["extract_model"] == "openrouter/google/gemini-2.5-flash"
    assert cfg["storage"]["data_dir"] == "~/.lens/data"
    assert "taxonomy" not in cfg
    assert cfg["monitor"]["ideate"] is True


def test_load_config_returns_defaults_when_no_file(tmp_path):
    from lens.config import load_config

    cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["default_model"] == "openrouter/anthropic/claude-sonnet-4-6"


def test_load_config_merges_with_file(tmp_path):
    from lens.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"llm": {"default_model": "custom/model"}}))
    cfg = load_config(config_path=config_file)
    assert cfg["llm"]["default_model"] == "custom/model"
    assert cfg["llm"]["extract_model"] == "openrouter/google/gemini-2.5-flash"


def test_save_config(tmp_path):
    from lens.config import load_config, save_config

    config_file = tmp_path / "config.yaml"
    save_config({"llm": {"default_model": "test/model"}}, config_path=config_file)
    assert config_file.exists()
    cfg = load_config(config_path=config_file)
    assert cfg["llm"]["default_model"] == "test/model"


def test_config_set_nested_key(tmp_path):
    from lens.config import load_config, set_config_value

    config_file = tmp_path / "config.yaml"
    cfg = load_config(config_path=config_file)
    set_config_value(cfg, "llm.default_model", "new/model")
    assert cfg["llm"]["default_model"] == "new/model"


def test_config_set_coerces_types():
    from lens.config import default_config, set_config_value

    cfg = default_config()
    set_config_value(cfg, "monitor.ideate", "false")
    assert cfg["monitor"]["ideate"] is False
    set_config_value(cfg, "monitor.ideate_min_gap_score", "0.75")
    assert cfg["monitor"]["ideate_min_gap_score"] == 0.75


def test_resolved_data_dir():
    from lens.config import resolve_data_dir

    cfg = {"storage": {"data_dir": "~/.lens/data"}}
    resolved = resolve_data_dir(cfg)
    assert "~" not in resolved
    assert resolved.endswith(".lens/data")


def test_validate_config_valid():
    """Default config should produce no warnings."""
    from lens.config import default_config

    warnings = validate_config(default_config())
    assert warnings == []


def test_validate_config_empty_model():
    """Empty model strings should produce warnings."""
    from lens.config import default_config

    cfg = default_config()
    cfg["llm"]["default_model"] = ""
    cfg["llm"]["extract_model"] = ""
    warnings = validate_config(cfg)
    assert any("llm.default_model" in w for w in warnings)
    assert any("llm.extract_model" in w for w in warnings)


def test_validate_config_invalid_provider():
    """Unknown embedding provider should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["embeddings"]["provider"] = "invalid"
    warnings = validate_config(cfg)
    assert any("embeddings.provider" in w for w in warnings)


def test_validate_config_bad_dimensions():
    """Non-positive dimensions should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["embeddings"]["dimensions"] = -1
    warnings = validate_config(cfg)
    assert any("embeddings.dimensions" in w for w in warnings)


def test_validate_config_invalid_categories():
    """Empty arxiv_categories should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["acquire"]["arxiv_categories"] = []
    warnings = validate_config(cfg)
    assert any("arxiv_categories" in w for w in warnings)


def test_validate_config_gap_score_range():
    """Out-of-range gap score should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["monitor"]["ideate_min_gap_score"] = 1.5
    warnings = validate_config(cfg)
    assert any("ideate_min_gap_score" in w for w in warnings)
