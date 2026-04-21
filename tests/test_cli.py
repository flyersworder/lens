"""Tests for the LENS CLI skeleton."""

import pytest
from typer.testing import CliRunner

from lens.cli import app
from lens.store.store import LensStore

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


def _seed_taxonomy(store: LensStore) -> None:
    """Seed the minimum state that `analyze`/`explain` check before running."""
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-04-21T00:00:00",
                "paper_count": 0,
                "param_count": 0,
                "principle_count": 0,
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            }
        ],
    )


def test_cli_analyze_provenance(tmp_path, monkeypatch):
    """`lens analyze --provenance PATH` writes a YAML sidecar next to the answer."""
    import yaml

    from lens.store.store import LensStore

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("LENS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    _seed_taxonomy(store)

    async def _fake_analyze(query, store, llm_client):
        return {
            "query": query,
            "improving": "Inference Latency",
            "worsening": "Model Accuracy",
            "principles": [
                {
                    "principle_id": "quantization",
                    "name": "Quantization",
                    "count": 3,
                    "avg_confidence": 0.8,
                    "score": 2.4,
                    "paper_ids": ["p1", "p2"],
                }
            ],
        }

    monkeypatch.setattr("lens.serve.analyzer.analyze", _fake_analyze)

    out = tmp_path / "analyze.provenance.yaml"
    result = runner.invoke(app, ["analyze", "reduce latency", "--provenance", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = yaml.safe_load(out.read_text())
    assert parsed["command"] == "analyze"
    assert parsed["taxonomy_version"] == 1
    assert parsed["paper_ids"] == ["p1", "p2"]
    assert parsed["claims"][0]["principle_id"] == "quantization"


def test_cli_explain_provenance(tmp_path, monkeypatch):
    """`lens explain --provenance PATH` writes a YAML sidecar with vocab + paper refs."""
    import yaml

    from lens.store.models import ExplanationResult
    from lens.store.store import LensStore

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("LENS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    _seed_taxonomy(store)

    async def _fake_explain(query, store, llm_client, focus=None, embedding_kwargs=None):
        return ExplanationResult(
            resolved_type="parameter",
            resolved_id="inference-latency",
            resolved_name="Inference Latency",
            narrative="Latency is the time per token...",
            evolution=[],
            tradeoffs=[
                {
                    "improving_param_id": "inference-latency",
                    "worsening_param_id": "model-accuracy",
                    "principle_id": "quantization",
                    "count": 3,
                    "avg_confidence": 0.8,
                    "paper_ids": ["p1"],
                    "taxonomy_version": 1,
                }
            ],
            connections=["Model Accuracy"],
            paper_refs=[],
            alternatives=[],
        )

    monkeypatch.setattr("lens.serve.explainer.explain", _fake_explain)

    out = tmp_path / "explain.provenance.yaml"
    result = runner.invoke(app, ["explain", "latency", "--provenance", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = yaml.safe_load(out.read_text())
    assert parsed["command"] == "explain"
    assert parsed["resolved"]["id"] == "inference-latency"
    assert parsed["paper_ids"] == ["p1"]
    assert "quantization" in parsed["vocab_ids"]


def test_cli_explore_group_exists():
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0


def test_cli_monitor_exists():
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0


def test_monitor_no_interval_flag():
    """Monitor command should not have an --interval flag."""
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "--interval" not in result.output


def test_explore_architecture_help():
    result = runner.invoke(app, ["explore", "architecture", "--help"])
    assert result.exit_code == 0
    assert "architecture" in result.output.lower()


def test_explore_agents_help():
    result = runner.invoke(app, ["explore", "agents", "--help"])
    assert result.exit_code == 0


def test_explore_evolution_help():
    result = runner.invoke(app, ["explore", "evolution", "--help"])
    assert result.exit_code == 0


def test_export_creates_backup(tmp_path):
    """lens export should create a timestamped backup file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "lens.db"
    store = LensStore(str(db_path))
    store.init_tables()
    store.conn.close()

    output_path = tmp_path / "backup.db"

    from lens.cli import _export_db

    _export_db(source=db_path, destination=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_missing_db(tmp_path):
    """lens export should raise if source DB doesn't exist."""
    from lens.cli import _export_db

    with pytest.raises(FileNotFoundError):
        _export_db(source=tmp_path / "nonexistent.db", destination=tmp_path / "out.db")


def test_import_restores_db(tmp_path):
    """lens import should copy backup to data dir."""
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target_db = data_dir / "lens.db"

    from lens.cli import _import_db

    _import_db(source=backup_path, destination=target_db, force=False)

    assert target_db.exists()
    assert target_db.stat().st_size > 0


def test_import_refuses_overwrite(tmp_path):
    """lens import should refuse to overwrite without --force."""
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    target_db = tmp_path / "data" / "lens.db"
    target_db.parent.mkdir()
    target_db.write_text("existing")

    from lens.cli import _import_db

    with pytest.raises(FileExistsError):
        _import_db(source=backup_path, destination=target_db, force=False)


def test_import_with_force(tmp_path):
    """lens import --force should overwrite existing DB."""
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    target_db = tmp_path / "data" / "lens.db"
    target_db.parent.mkdir()
    target_db.write_text("existing")

    from lens.cli import _import_db

    _import_db(source=backup_path, destination=target_db, force=True)

    assert target_db.stat().st_size > len("existing")


def test_import_rejects_invalid_sqlite(tmp_path):
    """lens import should reject non-SQLite files."""
    bad_file = tmp_path / "not-a-db.db"
    bad_file.write_text("this is not a database")

    target_db = tmp_path / "data" / "lens.db"

    from lens.cli import _import_db

    with pytest.raises(ValueError, match="not a valid SQLite database"):
        _import_db(source=bad_file, destination=target_db, force=False)


def test_import_force_removes_stale_wal(tmp_path):
    """lens import --force should remove stale WAL/SHM files from old DB."""
    # Create a backup
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    # Create existing DB with fake WAL/SHM sidecar files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target_db = data_dir / "lens.db"
    target_db.write_text("old db")
    wal_file = data_dir / "lens.db-wal"
    shm_file = data_dir / "lens.db-shm"
    wal_file.write_text("stale wal data")
    shm_file.write_text("stale shm data")

    from lens.cli import _import_db

    _import_db(source=backup_path, destination=target_db, force=True)

    # DB should be restored
    assert target_db.exists()
    # Stale sidecar files should be gone
    assert not wal_file.exists()
    assert not shm_file.exists()


def test_search_by_query(tmp_path, sample_paper_data):
    """lens search with a text query returns matching papers."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(
        app,
        ["search", "Attention"],
        env={
            "LENS_DATA_DIR": str(tmp_path),
        },
    )
    assert result.exit_code == 0
    assert "Attention Is All You Need" in result.output


def test_search_by_author(tmp_path, sample_paper_data):
    """lens search --author filters by author name."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(
        app,
        ["search", "--author", "Vaswani"],
        env={
            "LENS_DATA_DIR": str(tmp_path),
        },
    )
    assert result.exit_code == 0
    assert "Attention Is All You Need" in result.output


def test_search_no_args(tmp_path):
    """lens search with no query and no filters shows an error."""
    from typer.testing import CliRunner

    from lens.cli import app

    runner = CliRunner()

    result = runner.invoke(
        app,
        ["search"],
        env={
            "LENS_DATA_DIR": str(tmp_path),
        },
    )
    assert result.exit_code == 1
    assert "Provide a search query" in result.output


def test_explore_paper_shows_date(tmp_path, sample_paper_data):
    """lens explore paper should display the paper date, not a missing 'year' field."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(
        app,
        ["explore", "paper", "2401.12345"],
        env={"LENS_DATA_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "2017-06-12" in result.output


def test_verbose_flag_accepted():
    """The -v flag should be accepted without error."""
    result = runner.invoke(app, ["-v", "--help"])
    assert result.exit_code == 0


def test_verbose_flag_in_help():
    """The --verbose flag should appear in top-level help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output or "-v" in result.output


def test_extract_without_api_key_shows_error(tmp_path, monkeypatch):
    """lens extract should fail early with a clear message when no API key is set."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["extract"])
    assert result.exit_code == 1
    assert "not configured" in result.output


def test_analyze_without_api_key_shows_error(tmp_path, monkeypatch):
    """lens analyze should fail early with a clear message when no API key is set."""
    from typer.testing import CliRunner

    from lens.cli import app

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = runner.invoke(app, ["analyze", "test query"])
    assert result.exit_code == 1
    assert "not configured" in result.output


def test_acquire_semantic_help():
    """acquire semantic subcommand should exist."""
    from typer.testing import CliRunner

    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["acquire", "semantic", "--help"])
    assert result.exit_code == 0
    assert "SPECTER2" in result.output or "Semantic Scholar" in result.output


def test_acquire_semantic_no_papers(tmp_path, monkeypatch):
    """acquire semantic with no papers should print a message."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["acquire", "semantic"])
    assert result.exit_code == 0
    assert "No papers" in result.output


def test_acquire_seed_computes_quality_score(tmp_path, monkeypatch):
    """acquire seed should compute quality_score for papers with citations/venue."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["acquire", "seed"])
    assert result.exit_code == 0

    store = LensStore(db_path)
    store.init_tables()
    papers = store.query("papers")
    # At least some seed papers should have a non-None quality_score
    scored = [p for p in papers if p.get("quality_score") is not None]
    assert len(scored) > 0


def test_monitor_has_skip_flags():
    """Monitor should accept --skip-enrich and --skip-build flags."""
    import re

    from typer.testing import CliRunner

    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    # Strip ANSI escape codes — Rich formatting can split flag names in CI
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--skip-enrich" in plain
    assert "--skip-build" in plain


def test_status_empty_db(tmp_path, monkeypatch):
    """lens status on an empty DB should show zeros gracefully."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Papers" in result.output
    assert "0" in result.output


def test_status_with_data(tmp_path, sample_paper_data, monkeypatch):
    """lens status with papers should show counts."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    from lens.taxonomy.vocabulary import load_seed_vocabulary

    load_seed_vocabulary(store)
    store.conn.close()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Papers" in result.output
    assert "pending: 1" in result.output
    assert "Vocabulary" in result.output
