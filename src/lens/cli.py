"""LENS CLI — LLM Engineering Navigation System command-line interface."""

import asyncio
import logging
import os
import shutil
import struct
from datetime import UTC
from pathlib import Path
from uuid import uuid4

import typer
import yaml
from rich import print as rprint

from lens.config import load_config, resolve_data_dir, save_config, set_config_value
from lens.knowledge.events import log_event
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore

# ---------------------------------------------------------------------------
# App and subcommand groups
# ---------------------------------------------------------------------------

app = typer.Typer(name="lens")

acquire_app = typer.Typer(help="Acquire papers from various sources.")
build_app = typer.Typer(help="Build taxonomy, matrix, and other derived artefacts.")
explore_app = typer.Typer(help="Explore the LENS knowledge base interactively.")
config_app = typer.Typer(help="View and modify LENS configuration.")
vocab_app = typer.Typer(help="Manage the canonical vocabulary.")

app.add_typer(acquire_app, name="acquire")
app.add_typer(build_app, name="build")
app.add_typer(explore_app, name="explore")
app.add_typer(config_app, name="config")
app.add_typer(vocab_app, name="vocab")


@app.callback()
def main(
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase log verbosity (-v=INFO, -vv=DEBUG)."
    ),
) -> None:
    """LENS — LLM Engineering Navigation System."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_data_dir(config: dict) -> Path:
    """Return the data directory, with LENS_DATA_DIR env-var override."""
    env_override = os.environ.get("LENS_DATA_DIR")
    if env_override:
        return Path(env_override)
    return Path(resolve_data_dir(config))


def _get_config_path() -> Path | None:
    """Return config path override from LENS_CONFIG_PATH env-var, or None."""
    env_override = os.environ.get("LENS_CONFIG_PATH")
    if env_override:
        return Path(env_override)
    return None


def _export_db(source: Path, destination: Path) -> None:
    """Back up the SQLite database using the sqlite3 backup API.

    Uses Connection.backup() for WAL-safe, consistent snapshots rather
    than raw file copy (which would miss WAL/SHM sidecar files).
    """
    import sqlite3

    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(source))
    try:
        dst_conn = sqlite3.connect(str(destination))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _import_db(source: Path, destination: Path, force: bool = False) -> None:
    """Restore a backup database to the destination path."""
    import sqlite3

    if not source.exists():
        raise FileNotFoundError(f"Backup file not found: {source}")
    # Verify source is a valid SQLite database before overwriting anything
    try:
        conn = sqlite3.connect(str(source))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                raise ValueError(f"Source database is corrupt: {source}")
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        raise ValueError(f"Source file is not a valid SQLite database: {source}") from e
    if destination.exists() and not force:
        raise FileExistsError(
            f"Database already exists at {destination}. Use --force to overwrite."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale WAL/SHM sidecar files before restoring — if left behind,
    # SQLite would apply the old WAL to the newly-restored database.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(destination) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    shutil.copy2(source, destination)


def _get_store() -> LensStore:
    """Return an initialised LensStore using the current config."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    return store


def _llm_kwargs(config: dict, key: str = "default_model") -> dict:
    """Extract api_base and api_key from config for LLMClient."""
    llm_cfg = config.get("llm", {})
    kwargs: dict = {}
    if llm_cfg.get("api_base"):
        kwargs["api_base"] = llm_cfg["api_base"]
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _embedding_kwargs(config: dict) -> dict:
    """Extract embedding config for embed_strings calls in the serve layer."""
    emb_cfg = config.get("embeddings", {})
    kwargs: dict = {}
    provider = emb_cfg.get("provider", "local")
    if provider != "local":
        kwargs["provider"] = provider
    if emb_cfg.get("model"):
        kwargs["model_name"] = emb_cfg["model"]
    if emb_cfg.get("api_base"):
        kwargs["api_base"] = emb_cfg["api_base"]
    api_key = emb_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _require_llm_config(config: dict) -> None:
    """Exit early with a clear message if no LLM backend is configured.

    Passes if any of: api_key in config, OPENROUTER_API_KEY env, or api_base (gateway mode).
    """
    llm_cfg = config.get("llm", {})
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    api_base = llm_cfg.get("api_base", "")
    if not api_key and not api_base:
        rprint(
            "[red]LLM backend not configured.[/red]\n"
            "Set an API key: [bold]lens config set llm.api_key YOUR_KEY[/bold]\n"
            "Or export: [bold]export OPENROUTER_API_KEY=YOUR_KEY[/bold]\n"
            "Or use gateway mode: [bold]lens config set llm.api_base http://your-gateway:4000/v1[/bold]"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Re-initialise, removing existing data."),
) -> None:
    """Initialise the LENS data directory and SQLite database."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)

    db_path = data_dir / "lens.db"

    if force and db_path.exists():
        db_path.unlink()

    data_dir.mkdir(parents=True, exist_ok=True)
    store = LensStore(str(db_path))
    store.init_tables()
    rprint(f"[green]Initialized LENS database at {db_path}[/green]")


@app.command()
def status() -> None:
    """Show a summary of the LENS knowledge base."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    db_path = data_dir / "lens.db"
    if not db_path.exists():
        rprint("[yellow]No database found. Run 'lens init' first.[/yellow]")
        raise typer.Exit(code=1)

    store = LensStore(str(db_path))
    store.init_tables()

    # Paper counts by extraction status
    status_rows = store.query_sql(
        "SELECT extraction_status, COUNT(*) AS cnt FROM papers GROUP BY extraction_status"
    )
    total = sum(r["cnt"] for r in status_rows)
    sorted_statuses = sorted(status_rows, key=lambda r: r["extraction_status"])
    status_parts = ", ".join(f"{r['extraction_status']}: {r['cnt']}" for r in sorted_statuses)

    rprint("\n[bold]LENS Knowledge Base Status[/bold]")
    rprint("=" * 40)
    rprint(f"Papers: {total} ({status_parts})" if status_parts else f"Papers: {total}")

    # Vocabulary counts by kind
    vocab_rows = store.query_sql("SELECT kind, COUNT(*) AS cnt FROM vocabulary GROUP BY kind")
    if vocab_rows:
        vocab_parts = ", ".join(
            f"{r['cnt']} {r['kind']}s" for r in sorted(vocab_rows, key=lambda r: r["kind"])
        )
        rprint(f"Vocabulary: {vocab_parts}")
    else:
        rprint("Vocabulary: empty")

    # Matrix
    matrix_rows = store.query_sql(
        "SELECT COUNT(*) AS cell_count, COALESCE(SUM(count), 0) AS total_evidence "
        "FROM matrix_cells"
    )
    cell_count = matrix_rows[0]["cell_count"] if matrix_rows else 0
    if cell_count:
        rprint(f"Matrix: {cell_count} cells, {matrix_rows[0]['total_evidence']} total evidence")
    else:
        rprint("Matrix: empty")

    # Top parameters by paper_count
    top_params = store.query_sql(
        "SELECT name, paper_count FROM vocabulary "
        "WHERE kind = ? AND paper_count > 0 "
        "ORDER BY paper_count DESC LIMIT 5",
        ("parameter",),
    )
    if top_params:
        top_str = ", ".join(f"{p['name']} ({p['paper_count']})" for p in top_params)
        rprint(f"Top parameters: {top_str}")

    # Taxonomy version
    versions = store.query_sql(
        "SELECT version_id, created_at FROM taxonomy_versions ORDER BY version_id DESC LIMIT 1"
    )
    if versions:
        v = versions[0]
        rprint(f"Taxonomy: v{v['version_id']} (built: {v.get('created_at', 'unknown')})")
    else:
        rprint("Taxonomy: not built yet")

    # Last event
    events = store.query_sql(
        "SELECT timestamp, kind FROM event_log ORDER BY timestamp DESC LIMIT 1"
    )
    if events:
        e = events[0]
        rprint(f"Last event: {e.get('timestamp', '?')} ({e.get('kind', '?')})")
    else:
        rprint("Last event: none")

    # Cheap lint checks
    from lens.knowledge.linter import (
        check_missing_embeddings,
        check_orphan_vocabulary,
        check_weak_evidence,
    )

    orphans = check_orphan_vocabulary(store)
    weak = check_weak_evidence(store)
    missing_emb = check_missing_embeddings(store)
    issues = []
    if orphans:
        issues.append(f"{len(orphans)} orphans")
    if weak:
        issues.append(f"{len(weak)} weak evidence")
    if missing_emb:
        issues.append(f"{len(missing_emb)} missing embeddings")
    if issues:
        rprint(f"Issues: {', '.join(issues)}")
    else:
        rprint("Issues: none")

    rprint()


@app.command()
def analyze(
    query: str = typer.Argument(..., help="Problem description."),
    type_: str | None = typer.Option(None, "--type", help="Query type."),
) -> None:
    """Analyze a tradeoff and suggest resolution techniques."""
    config = load_config(_get_config_path())
    _require_llm_config(config)
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.llm.client import LLMClient
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    client = LLMClient(model=config["llm"]["default_model"], **_llm_kwargs(config))

    if type_ == "architecture":
        from lens.serve.analyzer import analyze_architecture

        result = asyncio.run(analyze_architecture(query, store, client))
        rprint(f"\n[bold]Query:[/bold] {result['query']}")
        rprint(f"[bold]Slot:[/bold] {result.get('slot')}")
        if result["variants"]:
            rprint("\n[bold]Matching architecture variants:[/bold]")
            for v in result["variants"]:
                props = v.get("properties") or ""
                name = v.get("variant_name") or v.get("name", "")
                rprint(f"  • {name} — {props}" if props else f"  • {name}")
        else:
            rprint("[yellow]No matching architecture variants found.[/yellow]")
    elif type_ == "agentic":
        from lens.serve.analyzer import analyze_agentic

        result = asyncio.run(analyze_agentic(query, store, client))
        rprint(f"\n[bold]Query:[/bold] {result['query']}")
        if result["patterns"]:
            rprint("\n[bold]Matching agentic patterns:[/bold]")
            for p in result["patterns"]:
                name = p.get("pattern_name") or p.get("name", "")
                rprint(f"  • [{p.get('category')}] {name} — {p.get('use_case', '')}")
        else:
            rprint("[yellow]No matching agentic patterns found.[/yellow]")
    else:
        from lens.serve.analyzer import analyze as do_analyze

        result = asyncio.run(do_analyze(query, store, client))
        rprint(f"\n[bold]Query:[/bold] {result['query']}")
        rprint(f"[bold]Improving:[/bold] {result['improving']}")
        rprint(f"[bold]Worsening:[/bold] {result['worsening']}")
        if result["principles"]:
            rprint("\n[bold]Suggested techniques:[/bold]")
            for p in result["principles"]:
                conf = p["avg_confidence"]
                rprint(f"  • {p['name']} (evidence: {p['count']}, confidence: {conf:.2f})")
        else:
            rprint("[yellow]No matching techniques found.[/yellow]")


@app.command()
def explain(
    concept: str = typer.Argument(..., help="Concept to explain."),
    related: bool = typer.Option(False, "--related", help="Focus on related concepts."),
    evolution: bool = typer.Option(False, "--evolution", help="Focus on evolution."),
    tradeoffs: bool = typer.Option(False, "--tradeoffs", help="Focus on tradeoffs."),
) -> None:
    """Explain an LLM concept with adaptive depth."""
    config = load_config(_get_config_path())
    _require_llm_config(config)
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.llm.client import LLMClient
    from lens.serve.explainer import explain as do_explain
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    focus = None
    if tradeoffs:
        focus = "tradeoffs"
    elif related:
        focus = "related"
    elif evolution:
        focus = "evolution"

    client = LLMClient(model=config["llm"]["default_model"], **_llm_kwargs(config))
    emb_kw = _embedding_kwargs(config)
    result = asyncio.run(do_explain(concept, store, client, focus=focus, embedding_kwargs=emb_kw))

    if result is None:
        rprint(f"[yellow]Concept '{concept}' not found.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"\n[bold]{result.resolved_name}[/bold] ({result.resolved_type})\n")
    rprint(result.narrative)
    if result.connections:
        rprint(f"\n[bold]Related:[/bold] {', '.join(result.connections)}")
    if result.paper_refs:
        rprint(f"[bold]Papers:[/bold] {', '.join(result.paper_refs[:5])}")


@app.command()
def search(
    query: str | None = typer.Argument(None, help="Text to search for."),
    author: str | None = typer.Option(None, "--author", help="Filter by author name."),
    venue: str | None = typer.Option(None, "--venue", help="Filter by venue."),
    after: str | None = typer.Option(
        None, "--after", help="Papers on or after date (YYYY-MM-DD)."
    ),
    before: str | None = typer.Option(
        None, "--before", help="Papers on or before date (YYYY-MM-DD)."
    ),
    limit: int = typer.Option(10, "--limit", help="Max results."),
) -> None:
    """Search papers by keyword, semantic similarity, or metadata filters."""
    if not query and not author and not venue and not after and not before:
        rprint(
            "[red]Provide a search query or at least one filter "
            "(--author, --venue, --after, --before).[/red]"
        )
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import search_papers

    emb_kw = _embedding_kwargs(config)
    results = search_papers(
        store,
        query=query,
        author=author,
        venue=venue,
        after=after,
        before=before,
        limit=limit,
        embedding_kwargs=emb_kw,
    )

    if not results:
        rprint("[yellow]No papers found.[/yellow]")
        raise typer.Exit(code=0)

    rprint(f"\n[bold]Found {len(results)} paper{'s' if len(results) != 1 else ''}:[/bold]\n")
    for i, r in enumerate(results, 1):
        score_str = f"[{r['score']:.2f}] " if r.get("score") is not None else ""
        venue_str = f" · {r['venue']}" if r.get("venue") else ""
        rprint(f"  {i}. {score_str}{r['title']} ({r['date']})")
        rprint(f"     arxiv:{r['arxiv_id']}{venue_str} · {r['authors_display']}")
        rprint(f"     {r['abstract_snippet']}")
        rprint()


@app.command()
def extract(
    paper_id: str | None = typer.Option(None, "--paper-id", help="Extract specific paper."),
    model: str | None = typer.Option(None, "--model", help="LLM model override."),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent LLM calls."),
) -> None:
    """Extract tradeoffs, architecture, and agentic patterns from papers."""
    config = load_config(_get_config_path())
    _require_llm_config(config)
    data_dir = _get_data_dir(config)
    llm_model = model or config["llm"]["extract_model"]

    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.extract.extractor import extract_papers
    from lens.llm.client import LLMClient

    client = LLMClient(model=llm_model, **_llm_kwargs(config))
    session_id = str(uuid4())[:8]
    count = asyncio.run(
        extract_papers(
            store, client, concurrency=concurrency, paper_id=paper_id, session_id=session_id
        )
    )
    rprint(f"[green]Extracted {count} papers[/green]")


@app.command()
def monitor(
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
    skip_enrich: bool = typer.Option(
        False, "--skip-enrich", help="Skip OpenAlex enrichment stage."
    ),
    skip_build: bool = typer.Option(
        False, "--skip-build", help="Skip taxonomy and matrix rebuild."
    ),
) -> None:
    """Run one monitoring cycle: acquire -> enrich -> extract -> build -> ideate."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    if trending:
        gaps = store.query("ideation_gaps")
        if not gaps:
            rprint("[yellow]No ideation gaps found.[/yellow]")
        else:
            rprint("\n[bold]Ideation Gaps:[/bold]")
            gaps.sort(key=lambda x: x.get("score", 0), reverse=True)
            for row in gaps:
                hyp = ""
                if row.get("llm_hypothesis"):
                    h = row["llm_hypothesis"][:80]
                    hyp = f" — {h}..."
                rprint(f"  [{row['gap_type']}] {row['description']}{hyp}")
        raise typer.Exit(code=0)

    _require_llm_config(config)

    from lens.llm.client import LLMClient
    from lens.monitor.watcher import run_monitor_cycle

    client = LLMClient(model=config["llm"]["extract_model"], **_llm_kwargs(config))
    cats = config["acquire"]["arxiv_categories"]
    monitor_cfg = config["monitor"]
    openalex_mailto = config["acquire"].get("openalex_mailto", "")
    session_id = str(uuid4())[:8]

    result = asyncio.run(
        run_monitor_cycle(
            store,
            client,
            categories=cats,
            run_enrich=not skip_enrich,
            run_build=not skip_build,
            run_ideation_flag=monitor_cfg["ideate"],
            ideate_with_llm=monitor_cfg.get("ideate_llm", False),
            openalex_mailto=openalex_mailto,
            embedding_kwargs=_embedding_kwargs(config),
            venue_tiers=config["acquire"].get("quality_venue_tiers"),
            session_id=session_id,
        )
    )
    rprint("[green]Monitor cycle complete:[/green]")
    rprint(f"  Papers acquired: {result['papers_acquired']}")
    if result["papers_enriched"]:
        rprint(f"  Papers enriched: {result['papers_enriched']}")
    rprint(f"  Papers extracted: {result['papers_extracted']}")
    if result["taxonomy_built"]:
        rprint("  Taxonomy + matrix: rebuilt")
    if result.get("ideation_report"):
        rprint(f"  Gaps found: {result['ideation_report']['gap_count']}")


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Apply safe auto-fixes after reporting."),
    check: str | None = typer.Option(
        None,
        "--check",
        help=(
            "Comma-separated checks to run: "
            "orphans,contradictions,weak_evidence,missing_embeddings,stale,near_duplicates"
        ),
    ),
    threshold_confidence: float = typer.Option(
        0.5, "--threshold-confidence", help="Weak evidence confidence cutoff."
    ),
    threshold_similarity: float = typer.Option(
        0.92, "--threshold-similarity", help="Near-duplicate cosine similarity threshold."
    ),
) -> None:
    """Health-check the knowledge base for issues."""
    from lens.knowledge.linter import lint as run_lint

    store = _get_store()
    config = load_config(_get_config_path())
    session_id = str(uuid4())[:8]

    checks = [c.strip() for c in check.split(",")] if check else None

    emb_cfg = config.get("embeddings", {})
    report = run_lint(
        store,
        fix=fix,
        session_id=session_id,
        checks=checks,
        confidence_threshold=threshold_confidence,
        similarity_threshold=threshold_similarity,
        embedding_provider=emb_cfg.get("provider", "local"),
        embedding_model=emb_cfg.get("model"),
        embedding_api_base=emb_cfg.get("api_base"),
        embedding_api_key=emb_cfg.get("api_key"),
    )

    typer.echo(f"\nLint Report (session {session_id})")
    typer.echo("─" * 36)
    typer.echo(f"  Orphan vocabulary:     {len(report.orphans)} found")
    typer.echo(f"  Contradictions:        {len(report.contradictions)} found")
    typer.echo(f"  Weak evidence:         {len(report.weak_evidence)} found")
    typer.echo(f"  Missing embeddings:    {len(report.missing_embeddings)} found")
    typer.echo(f"  Stale extractions:     {len(report.stale_extractions)} found")
    typer.echo(f"  Near-duplicates:       {len(report.near_duplicates)} pairs found")
    typer.echo("─" * 36)

    total = (
        len(report.orphans)
        + len(report.contradictions)
        + len(report.weak_evidence)
        + len(report.missing_embeddings)
        + len(report.stale_extractions)
        + len(report.near_duplicates)
    )
    typer.echo(f"  Total issues:          {total}\n")

    if fix and report.fixes_applied:
        orphan_fixes = sum(1 for f in report.fixes_applied if f["action"] == "orphan.deleted")
        emb_fixes = sum(1 for f in report.fixes_applied if f["action"] == "embedding.repaired")
        requeue_fixes = sum(
            1 for f in report.fixes_applied if f["action"] == "extraction.requeued"
        )
        merge_fixes = sum(1 for f in report.fixes_applied if f["action"] == "duplicate.merged")
        if orphan_fixes:
            typer.echo(f"  Fixed: deleted {orphan_fixes} orphan vocabulary entries")
        if emb_fixes:
            typer.echo(f"  Fixed: embedded {emb_fixes} missing vocabulary entries")
        if requeue_fixes:
            typer.echo(f"  Fixed: requeued {requeue_fixes} stale extractions")
        if merge_fixes:
            typer.echo(f"  Fixed: merged {merge_fixes} near-duplicate entries")
        typer.echo()
    elif not fix and total > 0:
        typer.echo("Use --fix to apply safe auto-fixes.\n")


@app.command(name="log")
def show_log(
    kind: str | None = typer.Option(
        None, "--kind", help="Filter by event kind (ingest, extract, build, lint, fix)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Show events after this date (YYYY-MM-DD)."
    ),
    limit: int = typer.Option(20, "--limit", help="Max events to show."),
    session: str | None = typer.Option(
        None, "--session", help="Show events from a specific session."
    ),
) -> None:
    """Show the event log."""
    from lens.knowledge.events import query_events

    store = _get_store()
    events = query_events(store, kind=kind, since=since, limit=limit, session_id=session)

    if not events:
        typer.echo("No events found.")
        return

    for event in events:
        ts = event["timestamp"][:16].replace("T", " ")
        k = event["kind"]
        action = event["action"]
        target = ""
        if event.get("target_type") and event.get("target_id"):
            target = f"{event['target_type']}:{event['target_id']}"

        detail_str = ""
        detail = event.get("detail")
        if isinstance(detail, dict) and detail:
            parts = [f"{v}" for v in detail.values()]
            detail_str = f"  ({', '.join(parts[:3])})"

        typer.echo(f"{ts}  {k:<8} {action:<28} {target}{detail_str}")


@app.command()
def export(
    output: str | None = typer.Option(None, "--output", help="Destination path for backup file."),
) -> None:
    """Back up the LENS database to a file."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    db_path = data_dir / "lens.db"

    if not db_path.exists():
        rprint("[red]No database found. Run 'lens init' first.[/red]")
        raise typer.Exit(code=1)

    if output:
        dest = Path(output)
    else:
        from datetime import datetime

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
        dest = Path(f"lens-backup-{ts}.db")

    _export_db(source=db_path, destination=dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    rprint(f"[green]Exported database to {dest} ({size_mb:.1f} MB)[/green]")


@app.command(name="import")
def import_db(
    path: Path = typer.Argument(..., help="Path to backup database file."),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite existing database."),
) -> None:
    """Restore the LENS database from a backup file."""
    if not path.exists():
        rprint(f"[red]Backup file not found: {path}[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    target_db = data_dir / "lens.db"

    if target_db.exists() and not force:
        rprint(f"[red]Database already exists at {target_db}. Use --force to overwrite.[/red]")
        raise typer.Exit(code=1)

    data_dir.mkdir(parents=True, exist_ok=True)
    _import_db(source=path, destination=target_db, force=force)

    # Run migrations on imported database
    store = LensStore(str(target_db))
    store.init_tables()

    rprint(f"[green]Restored database from {path} to {target_db}[/green]")


# ---------------------------------------------------------------------------
# Acquire subcommands
# ---------------------------------------------------------------------------


@acquire_app.command()
def seed() -> None:
    """Ingest curated seed papers from the manifest."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    count = asyncio.run(_acquire_seed_async(store))
    session_id = str(uuid4())[:8]
    if count > 0:
        log_event(
            store,
            "ingest",
            "paper.added",
            detail={"source": "seed", "count": count},
            session_id=session_id,
        )
    rprint(f"[green]Acquired {count} seed papers[/green]")

    # Compute quality scores for newly-added seed papers
    if count > 0:
        from lens.acquire.quality import quality_score as compute_quality

        all_papers = store.query("papers")
        venue_tiers = config["acquire"].get("quality_venue_tiers")
        for p in all_papers:
            if p.get("quality_score", 0.0) > 0.0:
                continue
            score = compute_quality(
                citations=p.get("citations", 0) or 0,
                venue=p.get("venue"),
                paper_date=p.get("date", "2020-01-01"),
                venue_tiers=venue_tiers,
            )
            store.update(
                "papers",
                "quality_score = ?",
                "paper_id = ?",
                (score, p["paper_id"]),
            )


async def _acquire_seed_async(store: LensStore) -> int:
    from lens.acquire.seed import acquire_seed

    return await acquire_seed(store)


@acquire_app.command()
def arxiv(
    query: str | None = typer.Option("LLM", "--query", help="ArXiv search query."),
    since: str | None = typer.Option(
        None, "--since", help="Fetch papers since date (YYYY-MM-DD)."
    ),
    max_results: int = typer.Option(100, "--max-results", help="Maximum papers to fetch."),
) -> None:
    """Fetch papers from arxiv."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    categories = config["acquire"]["arxiv_categories"]
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    papers = asyncio.run(_fetch_arxiv_async(query, categories, since, max_results))
    if not papers:
        rprint("[yellow]No papers found[/yellow]")
        return

    # Add placeholder embeddings for papers without them
    for p in papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * EMBEDDING_DIM

    existing_ids = {
        r["paper_id"]
        for r in store.query_sql(
            "SELECT paper_id FROM papers WHERE paper_id IN ({})".format(
                ",".join("?" for _ in papers)
            ),
            tuple(p["paper_id"] for p in papers),
        )
    }
    new_count = store.add_papers(papers)
    session_id = str(uuid4())[:8]
    for p in papers:
        if p["paper_id"] not in existing_ids:
            log_event(
                store,
                "ingest",
                "paper.added",
                target_type="paper",
                target_id=p["paper_id"],
                detail={"title": p["title"], "source": "arxiv"},
                session_id=session_id,
            )
    skipped = len(papers) - new_count
    msg = f"[green]Acquired {new_count} papers from arxiv[/green]"
    if skipped:
        msg += f" [yellow]({skipped} duplicates skipped)[/yellow]"
    rprint(msg)


async def _fetch_arxiv_async(query, categories, since, max_results):
    from lens.acquire.arxiv import fetch_arxiv_papers

    return await fetch_arxiv_papers(
        query=query, categories=categories, since=since, max_results=max_results
    )


@acquire_app.command()
def file(
    path: Path = typer.Argument(..., help="Path to PDF file."),  # noqa: B008
) -> None:
    """Ingest a single paper from a local PDF.

    Stores metadata now; full text is read by the LLM during extraction.
    """
    from lens.acquire.pdf import ingest_pdf

    if not path.exists():
        rprint(f"[red]File not found: {path}[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    paper = ingest_pdf(path)

    # Check for duplicate paper_id
    existing = store.query("papers", "paper_id = ?", (paper["paper_id"],))
    if existing:
        rprint(f"[yellow]Paper '{paper['paper_id']}' already exists. Skipping.[/yellow]")
        return

    store.add_papers([paper])
    session_id = str(uuid4())[:8]
    log_event(
        store,
        "ingest",
        "paper.added",
        target_type="paper",
        target_id=paper["paper_id"],
        detail={"title": paper["title"], "source": "file"},
        session_id=session_id,
    )
    rprint(f"[green]Ingested {path.name} as paper '{paper['paper_id']}'[/green]")


@acquire_app.command()
def openalex(
    enrich: bool = typer.Option(
        False, "--enrich", help="Enrich existing papers with OpenAlex metadata."
    ),
) -> None:
    """Enrich papers with OpenAlex metadata (citations, venue)."""
    if not enrich:
        rprint("[yellow]Use --enrich to enrich existing papers with OpenAlex data[/yellow]")
        return

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    papers = store.query("papers")
    if not papers:
        rprint("[yellow]No papers to enrich[/yellow]")
        return

    # Remove embedding from dicts before sending to enrichment
    papers_for_enrich = [{k: v for k, v in p.items() if k != "embedding"} for p in papers]
    mailto = config["acquire"].get("openalex_mailto", "")
    enriched = asyncio.run(_enrich_openalex_async(papers_for_enrich, mailto=mailto))

    # Persist enrichment back to DB
    session_id = str(uuid4())[:8]
    updated_count = 0
    for paper in enriched:
        pid = paper.get("paper_id", "")
        store.update(
            "papers",
            "citations = ?, venue = ?",
            "paper_id = ?",
            (paper.get("citations", 0), paper.get("venue"), pid),
        )
        log_event(
            store,
            "ingest",
            "paper.enriched",
            target_type="paper",
            target_id=pid,
            session_id=session_id,
        )
        updated_count += 1
    rprint(f"[green]Enriched {updated_count} papers with OpenAlex data[/green]")

    # Recompute quality scores now that we have citations and venue
    from lens.acquire.quality import quality_score as compute_quality

    venue_tiers = config["acquire"].get("quality_venue_tiers")
    for paper in enriched:
        pid = paper.get("paper_id", "")
        score = compute_quality(
            citations=paper.get("citations", 0) or 0,
            venue=paper.get("venue"),
            paper_date=paper.get("date", "2020-01-01"),
            venue_tiers=venue_tiers,
        )
        store.update("papers", "quality_score = ?", "paper_id = ?", (score, pid))


async def _enrich_openalex_async(papers, mailto: str = ""):
    from lens.acquire.openalex import enrich_with_openalex

    return await enrich_with_openalex(papers, mailto=mailto)


@acquire_app.command()
def deepxiv(
    query: str | None = typer.Argument(None, help="Search query for DeepXiv."),
    paper: str | None = typer.Option(None, "--paper", help="Fetch single paper by arXiv ID."),
    since: str | None = typer.Option(
        None, "--since", help="Only papers after this date (YYYY-MM-DD)."
    ),
    max_results: int = typer.Option(20, "--max-results", help="Maximum papers to fetch."),
    categories: str | None = typer.Option(
        None, "--categories", help="Comma-separated arXiv categories (e.g. cs.AI,cs.CL)."
    ),
) -> None:
    """Search and fetch papers via DeepXiv (requires deepxiv-sdk)."""
    from lens.acquire.deepxiv import HAS_DEEPXIV

    if not HAS_DEEPXIV:
        rprint("[red]deepxiv-sdk not installed. Run: uv sync --extra deepxiv[/red]")
        raise typer.Exit(code=1)

    if not query and not paper:
        rprint("[red]Provide a search query or --paper ARXIV_ID[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    session_id = str(uuid4())[:8]

    try:
        if paper:
            from lens.acquire.deepxiv import fetch_deepxiv_paper

            paper_data = fetch_deepxiv_paper(paper)
            paper_data["embedding"] = [0.0] * EMBEDDING_DIM
            new_count = store.add_papers([paper_data])
            if new_count:
                log_event(
                    store,
                    "ingest",
                    "paper.added",
                    target_type="paper",
                    target_id=paper_data["paper_id"],
                    detail={"title": paper_data["title"], "source": "deepxiv"},
                    session_id=session_id,
                )
                rprint(f"[green]Acquired paper {paper} via DeepXiv[/green]")
            else:
                rprint(f"[yellow]Paper '{paper}' already exists. Skipping.[/yellow]")
        else:
            from lens.acquire.deepxiv import search_deepxiv

            assert query is not None  # validated above
            cat_list = [c.strip() for c in categories.split(",")] if categories else None
            papers = search_deepxiv(
                query=query,
                categories=cat_list,
                since=since,
                max_results=max_results,
            )

            if not papers:
                rprint("[yellow]No papers found[/yellow]")
                return

            existing_ids = {
                r["paper_id"]
                for r in store.query_sql(
                    "SELECT paper_id FROM papers WHERE paper_id IN ({})".format(
                        ",".join("?" for _ in papers)
                    ),
                    tuple(p["paper_id"] for p in papers),
                )
            }
            for p in papers:
                if "embedding" not in p:
                    p["embedding"] = [0.0] * EMBEDDING_DIM

            new_count = store.add_papers(papers)
            for p in papers:
                if p["paper_id"] not in existing_ids:
                    log_event(
                        store,
                        "ingest",
                        "paper.added",
                        target_type="paper",
                        target_id=p["paper_id"],
                        detail={"title": p["title"], "source": "deepxiv"},
                        session_id=session_id,
                    )
            skipped = len(papers) - new_count
            msg = f"[green]Acquired {new_count} papers via DeepXiv[/green]"
            if skipped:
                msg += f" [yellow]({skipped} duplicates skipped)[/yellow]"
            rprint(msg)
    except Exception as e:
        rprint(f"[red]DeepXiv API error: {e}[/red]")
        raise typer.Exit(code=1) from None


@acquire_app.command()
def semantic(
    paper_id: str | None = typer.Option(
        None,
        "--paper-id",
        help="Force-fetch SPECTER2 embedding for a specific paper (overwrites existing).",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Semantic Scholar API key."),
) -> None:
    """Fetch SPECTER2 embeddings from Semantic Scholar."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    if paper_id:
        papers = store.query("papers", "paper_id = ?", (paper_id,))
    else:
        papers = store.query("papers")

    if not papers:
        rprint("[yellow]No papers to fetch embeddings for.[/yellow]")
        return

    # Filter to papers with zero-vector or missing embeddings
    if not paper_id:
        papers_needing_embeddings = []
        for p in papers:
            pid = p["paper_id"]
            vec_row = store.query_sql(
                "SELECT embedding FROM papers_vec WHERE paper_id = ?", (pid,)
            )
            if not vec_row:
                papers_needing_embeddings.append(p)
                continue
            emb_bytes = vec_row[0]["embedding"]
            emb = struct.unpack(f"{EMBEDDING_DIM}f", emb_bytes)
            if all(v == 0.0 for v in emb):
                papers_needing_embeddings.append(p)
        papers = papers_needing_embeddings

    if not papers:
        rprint("[yellow]All papers already have embeddings.[/yellow]")
        return

    arxiv_ids = [p.get("arxiv_id", p["paper_id"]) for p in papers]
    pid_by_arxiv = {p.get("arxiv_id", p["paper_id"]): p["paper_id"] for p in papers}

    rprint(f"Fetching SPECTER2 embeddings for {len(arxiv_ids)} paper(s)...")

    from lens.acquire.semantic_scholar import fetch_embeddings_batch

    results = asyncio.run(fetch_embeddings_batch(arxiv_ids, api_key=api_key))

    session_id = str(uuid4())[:8]
    updated = 0
    for arxiv_id, embedding in results.items():
        if embedding is not None:
            pid = pid_by_arxiv.get(arxiv_id, arxiv_id)
            store.upsert_embedding("papers", pid, embedding)
            log_event(
                store,
                "ingest",
                "paper.embedding_updated",
                target_type="paper",
                target_id=pid,
                detail={"source": "semantic_scholar"},
                session_id=session_id,
            )
            updated += 1

    rprint(f"[green]Updated {updated} paper embeddings[/green]")
    if updated < len(arxiv_ids):
        missing = len(arxiv_ids) - updated
        rprint(f"[yellow]{missing} papers had no SPECTER2 embedding available[/yellow]")


# ---------------------------------------------------------------------------
# Build subcommands
# ---------------------------------------------------------------------------


@build_app.command()
def taxonomy() -> None:
    """Build taxonomy from current extractions."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.taxonomy import get_next_version, record_version
    from lens.taxonomy.vocabulary import build_vocabulary

    version_id = get_next_version(store)
    session_id = str(uuid4())[:8]

    emb_cfg = config.get("embeddings", {})
    stats = build_vocabulary(
        store,
        embedding_provider=emb_cfg.get("provider", "local"),
        embedding_model=emb_cfg.get("model"),
        embedding_api_base=emb_cfg.get("api_base"),
        embedding_api_key=emb_cfg.get("api_key"),
        session_id=session_id,
    )

    # Record version
    paper_count = len(store.query("papers"))
    vocab = store.query("vocabulary")
    record_version(
        store,
        version_id,
        paper_count=paper_count,
        param_count=len([v for v in vocab if v["kind"] == "parameter"]),
        principle_count=len([v for v in vocab if v["kind"] == "principle"]),
        slot_count=len([v for v in vocab if v["kind"] == "arch_slot"]),
        variant_count=0,
        pattern_count=len([v for v in vocab if v["kind"] == "agentic_category"]),
        session_id=session_id,
    )
    rprint(
        f"[green]Taxonomy v{version_id} built.[/green] "
        f"new={stats['new_entries']} updated={stats['updated_entries']}"
    )


@build_app.command(name="matrix")
def build_matrix_cmd() -> None:
    """Build contradiction matrix from taxonomy."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy yet. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)
    session_id = str(uuid4())[:8]
    build_matrix(store, session_id=session_id)
    rprint(f"[green]Built matrix for taxonomy v{version}[/green]")


@build_app.command(name="all")
def build_all() -> None:
    """Full rebuild: taxonomy + matrix."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.taxonomy import get_next_version, record_version
    from lens.taxonomy.vocabulary import build_vocabulary

    version_id = get_next_version(store)
    session_id = str(uuid4())[:8]

    emb_cfg = config.get("embeddings", {})
    stats = build_vocabulary(
        store,
        embedding_provider=emb_cfg.get("provider", "local"),
        embedding_model=emb_cfg.get("model"),
        embedding_api_base=emb_cfg.get("api_base"),
        embedding_api_key=emb_cfg.get("api_key"),
        session_id=session_id,
    )

    # Record version
    paper_count = len(store.query("papers"))
    vocab = store.query("vocabulary")
    record_version(
        store,
        version_id,
        paper_count=paper_count,
        param_count=len([v for v in vocab if v["kind"] == "parameter"]),
        principle_count=len([v for v in vocab if v["kind"] == "principle"]),
        slot_count=len([v for v in vocab if v["kind"] == "arch_slot"]),
        variant_count=0,
        pattern_count=len([v for v in vocab if v["kind"] == "agentic_category"]),
        session_id=session_id,
    )
    build_matrix(store, session_id=session_id)
    rprint(
        f"[green]Built taxonomy v{version_id} + matrix.[/green] "
        f"new={stats['new_entries']} updated={stats['updated_entries']}"
    )


# ---------------------------------------------------------------------------
# Explore subcommands
# ---------------------------------------------------------------------------


@explore_app.command()
def parameters() -> None:
    """List all taxonomy parameters."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import list_parameters

    params = list_parameters(store)
    if not params:
        rprint("[yellow]No parameters found. Run 'lens vocab init' first.[/yellow]")
        return
    for p in params:
        rprint(f"[bold]{p['id']}[/bold] {p['name']} — {p['description']}")


@explore_app.command()
def principles() -> None:
    """List all taxonomy principles."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import list_principles

    princs = list_principles(store)
    if not princs:
        rprint("[yellow]No principles found. Run 'lens vocab init' first.[/yellow]")
        return
    for p in princs:
        rprint(f"[bold]{p['id']}[/bold] {p['name']} — {p['description']}")


@explore_app.command()
def matrix(
    param_a: str | None = typer.Argument(None, help="First parameter ID (slug)."),
    param_b: str | None = typer.Argument(None, help="Second parameter ID (slug)."),
) -> None:
    """Explore the parameter-principle matrix."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import get_matrix_cell, list_matrix_overview

    if param_a is not None and param_b is not None:
        cells = get_matrix_cell(store, param_a, param_b)
        if not cells:
            rprint("[yellow]No matrix cells found for that parameter pair.[/yellow]")
            return
        for cell in cells:
            rprint(
                f"principle_id={cell['principle_id']} "
                f"count={cell['count']} "
                f"avg_confidence={cell['avg_confidence']:.2f}"
            )
    else:
        overview = list_matrix_overview(store)
        if not overview:
            rprint("[yellow]Matrix is empty. Run 'lens build matrix' first.[/yellow]")
            return
        for row in overview:
            rprint(
                f"improving={row['improving_param_id']} "
                f"worsening={row['worsening_param_id']} "
                f"principles={row['num_principles']} "
                f"evidence={row['total_evidence']}"
            )


@explore_app.command()
def architecture(
    slot: str | None = typer.Argument(None, help="Architecture slot to explore."),
) -> None:
    """Explore architecture slots."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import list_architecture_slots, list_architecture_variants
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    if slot is None:
        slots = list_architecture_slots(store)
        if not slots:
            rprint("[yellow]No architecture slots found.[/yellow]")
            return
        for s in slots:
            rprint(f"[bold]{s['name']}[/bold] — {s.get('variant_count', 0)} variant(s)")
    else:
        variants = list_architecture_variants(store, slot_name=slot)
        if not variants:
            rprint(f"[yellow]No variants found for slot '{slot}'.[/yellow]")
            return
        for v in variants:
            name = v.get("variant_name") or v.get("name", "")
            props = v.get("key_properties") or v.get("properties") or ""
            rprint(f"  [bold]{name}[/bold] — {props}" if props else f"  [bold]{name}[/bold]")


@explore_app.command()
def agents(
    category: str | None = typer.Argument(None, help="Agentic pattern category."),
) -> None:
    """Explore agentic patterns."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import list_agentic_patterns
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    patterns = list_agentic_patterns(store, category=category)
    if not patterns:
        rprint("[yellow]No agentic patterns found.[/yellow]")
        return

    # Group by category when no category filter provided
    if category is None:
        by_cat: dict[str, list] = {}
        for p in patterns:
            cat = p.get("category") or "uncategorized"
            by_cat.setdefault(cat, []).append(p)
        for cat, pats in sorted(by_cat.items()):
            rprint(f"\n[bold]{cat}[/bold]")
            for p in pats:
                name = p.get("pattern_name") or p.get("name", "")
                rprint(f"  • {name} — {p.get('use_case', '')}")
    else:
        for p in patterns:
            name = p.get("pattern_name") or p.get("name", "")
            rprint(f"  • {name} — {p.get('use_case', '')}")


@explore_app.command()
def evolution(
    slot: str = typer.Argument(..., help="Architecture slot to trace evolution for."),
) -> None:
    """Explore evolution of an architecture slot over time."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import get_architecture_timeline
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    timeline = get_architecture_timeline(store, slot_name=slot)
    if not timeline:
        rprint(f"[yellow]No variants found for slot '{slot}'.[/yellow]")
        return

    rprint(f"\n[bold]Evolution of '{slot}':[/bold]")
    for v in timeline:
        date_str = v.get("earliest_date") or "unknown date"
        replaces = v.get("replaces")
        replaces_str = f" (replaces: {replaces})" if replaces else ""
        name = v.get("variant_name") or v.get("name", "")
        rprint(f"  {date_str}  [bold]{name}[/bold]{replaces_str}")


@explore_app.command()
def paper(
    paper_id: str = typer.Argument(..., help="Paper ID to inspect."),
) -> None:
    """Inspect a specific paper."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import get_paper

    result = get_paper(store, paper_id)
    if result is None:
        rprint(f"[yellow]Paper '{paper_id}' not found.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"\n[bold]{result.get('title', paper_id)}[/bold]")
    if result.get("authors"):
        rprint(f"[dim]Authors:[/dim] {result['authors']}")
    if result.get("date"):
        rprint(f"[dim]Date:[/dim] {result['date']}")
    if result.get("abstract"):
        rprint(f"\n{result['abstract']}")


@explore_app.command()
def ideas(
    type_: str | None = typer.Option(None, "--type", help="Gap type filter."),
) -> None:
    """Browse ideation gaps and research opportunities."""
    config = load_config(_get_config_path())
    store = LensStore(str(_get_data_dir(config) / "lens.db"))
    store.init_tables()

    if type_:
        gaps = store.query("ideation_gaps", "gap_type = ?", (type_,))
    else:
        gaps = store.query("ideation_gaps")

    if not gaps:
        rprint("[yellow]No ideation gaps found.[/yellow]")
        raise typer.Exit(code=0)

    rprint(f"\n[bold]Research Opportunities ({len(gaps)} gaps):[/bold]\n")
    gaps.sort(key=lambda x: x.get("score", 0), reverse=True)
    for row in gaps:
        rprint(f"  [bold][{row['gap_type']}][/bold] {row['description']}")
        if row.get("llm_hypothesis"):
            rprint(f"    → {row['llm_hypothesis']}")
        rprint()


# ---------------------------------------------------------------------------
# Vocab subcommands
# ---------------------------------------------------------------------------


@vocab_app.command(name="init")
def vocab_init() -> None:
    """Load seed vocabulary into the database."""
    from lens.taxonomy.vocabulary import load_seed_vocabulary

    store = _get_store()
    count = load_seed_vocabulary(store)
    if count:
        typer.echo(f"Loaded {count} seed vocabulary entries.")
    else:
        typer.echo("Vocabulary already initialized — no new entries.")


@vocab_app.command(name="list")
def vocab_list(
    kind: str | None = typer.Option(None, help="Filter by kind: parameter or principle"),
) -> None:
    """List vocabulary entries with evidence stats."""
    store = _get_store()
    rows = store.query("vocabulary", "kind = ?", (kind,)) if kind else store.query("vocabulary")

    if not rows:
        typer.echo("No vocabulary entries found.")
        return

    for r in rows:
        marker = "S" if r["source"] == "seed" else "E"
        typer.echo(
            f"  [{marker}] {r['name']} ({r['kind']}) — "
            f"papers={r['paper_count']}, conf={r['avg_confidence']:.2f}"
        )


@vocab_app.command(name="show")
def vocab_show(
    entry_id: str = typer.Argument(..., help="Vocabulary entry ID (slug)"),
) -> None:
    """Show details for a vocabulary entry."""
    store = _get_store()
    rows = store.query("vocabulary", "id = ?", (entry_id,))
    if not rows:
        typer.echo(f"No vocabulary entry with ID '{entry_id}'")
        raise typer.Exit(1)

    r = rows[0]
    typer.echo(f"Name:        {r['name']}")
    typer.echo(f"Kind:        {r['kind']}")
    typer.echo(f"Description: {r['description']}")
    typer.echo(f"Source:      {r['source']}")
    typer.echo(f"First seen:  {r['first_seen']}")
    typer.echo(f"Papers:      {r['paper_count']}")
    typer.echo(f"Avg conf:    {r['avg_confidence']:.4f}")


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


@config_app.command()
def show() -> None:
    """Show the current LENS configuration (API keys masked)."""
    import copy

    config_path = _get_config_path()
    config = load_config(config_path)
    # Mask sensitive keys
    display = copy.deepcopy(config)
    for section in display.values():
        if isinstance(section, dict):
            for key in section:
                if "key" in key.lower() and section[key]:
                    val = str(section[key])
                    section[key] = val[:8] + "..." if len(val) > 8 else "***"
    print(yaml.dump(display, default_flow_style=False, sort_keys=False), end="")


@config_app.command()
def set(
    key: str = typer.Argument(..., help="Dotted config key (e.g. llm.default_model)."),
    value: str = typer.Argument(..., help="Value to set."),
) -> None:
    """Set a configuration value."""
    config_path = _get_config_path()
    config = load_config(config_path)
    set_config_value(config, key, value)
    save_config(config, config_path)
    rprint(f"[green]Set {key} = {value}[/green]")
