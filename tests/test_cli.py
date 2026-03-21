"""Tests for the LENS CLI skeleton."""

from typer.testing import CliRunner

from lens.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LENS" in result.output or "lens" in result.output


def test_cli_init(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Initialized" in result.output


def test_cli_init_force(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "data"))
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0


def test_cli_config_show(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "default_model" in result.output


def test_cli_config_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("LENS_CONFIG_PATH", str(config_path))
    result = runner.invoke(app, ["config", "set", "llm.default_model", "test/model"])
    assert result.exit_code == 0
    assert config_path.exists()


def test_cli_acquire_group_exists():
    result = runner.invoke(app, ["acquire", "--help"])
    assert result.exit_code == 0


def test_cli_extract_group_exists():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0


def test_cli_build_group_exists():
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0


def test_cli_analyze_exists():
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0


def test_cli_explain_exists():
    result = runner.invoke(app, ["explain", "--help"])
    assert result.exit_code == 0


def test_cli_explore_group_exists():
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0


def test_cli_monitor_exists():
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
