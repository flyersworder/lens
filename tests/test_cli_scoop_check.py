from unittest.mock import patch

from typer.testing import CliRunner

from lens.cli import app

runner = CliRunner()


def test_scoop_check_command_invokes_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    async def fake_run(store, client, limit=None, top_k=5, max_terms=5):
        return {"checked": 3, "by_verdict": {"novel": 1, "overlaps": 1, "scooped": 1}}

    # Avoid building a real LLM client / hitting the network.
    with (
        patch("lens.llm.client.LLMClient"),
        patch("lens.knowledge.scoop_check.run_scoop_check", side_effect=fake_run),
        patch("lens.cli._get_data_dir", return_value=tmp_path),
    ):
        result = runner.invoke(app, ["scoop-check", "--limit", "3"])

    assert result.exit_code == 0
    assert "3" in result.stdout  # checked count surfaced
    assert "scooped" in result.stdout.lower()


def test_scoop_check_command_passes_max_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured = {}

    async def fake_run(store, client, limit=None, top_k=5, max_terms=5):
        captured["max_terms"] = max_terms
        return {"checked": 0, "by_verdict": {}}

    with (
        patch("lens.llm.client.LLMClient"),
        patch("lens.knowledge.scoop_check.run_scoop_check", side_effect=fake_run),
        patch("lens.cli._get_data_dir", return_value=tmp_path),
    ):
        result = runner.invoke(app, ["scoop-check", "--max-terms", "3"])

    assert result.exit_code == 0
    assert captured["max_terms"] == 3
