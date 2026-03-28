"""LENS CLI — LLM Engineering Navigation System command-line interface."""

import asyncio
import os
import shutil
from pathlib import Path

import polars as pl
import typer
import yaml
from rich import print as rprint

from lens.config import load_config, resolve_data_dir, save_config, set_config_value
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore, escape_sql_string

# ---------------------------------------------------------------------------
# App and subcommand groups
# ---------------------------------------------------------------------------

app = typer.Typer(name="lens", help="LENS — LLM Engineering Navigation System")

acquire_app = typer.Typer(help="Acquire papers from various sources.")
build_app = typer.Typer(help="Build taxonomy, matrix, and other derived artefacts.")
explore_app = typer.Typer(help="Explore the LENS knowledge base interactively.")
config_app = typer.Typer(help="View and modify LENS configuration.")

app.add_typer(acquire_app, name="acquire")
app.add_typer(build_app, name="build")
app.add_typer(explore_app, name="explore")
app.add_typer(config_app, name="config")


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


def _llm_kwargs(config: dict) -> dict:
    """Extract api_base and api_key from config for LLMClient."""
    llm_cfg = config.get("llm", {})
    kwargs: dict = {}
    if llm_cfg.get("api_base"):
        kwargs["api_base"] = llm_cfg["api_base"]
    if llm_cfg.get("api_key"):
        kwargs["api_key"] = llm_cfg["api_key"]
    return kwargs


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Re-initialise, removing existing data."),
) -> None:
    """Initialise the LENS data directory and LanceDB store."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)

    lance_dir = data_dir / "lance"

    if force and lance_dir.exists():
        shutil.rmtree(lance_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    store = LensStore(str(data_dir))
    store.init_tables()
    rprint(f"[green]Initialized LENS data directory at {data_dir}[/green]")


@app.command()
def analyze(
    query: str = typer.Argument(..., help="Problem description."),
    type_: str | None = typer.Option(None, "--type", help="Query type."),
) -> None:
    """Analyze a tradeoff and suggest resolution techniques."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
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

        result = asyncio.run(analyze_architecture(query, store, client, taxonomy_version=version))
        rprint(f"\n[bold]Query:[/bold] {result['query']}")
        rprint(f"[bold]Slot:[/bold] {result.get('slot')}")
        if result["variants"]:
            rprint("\n[bold]Matching architecture variants:[/bold]")
            for v in result["variants"]:
                props = v.get("properties") or ""
                rprint(f"  • {v['name']} — {props}" if props else f"  • {v['name']}")
        else:
            rprint("[yellow]No matching architecture variants found.[/yellow]")
    elif type_ == "agentic":
        from lens.serve.analyzer import analyze_agentic

        result = asyncio.run(analyze_agentic(query, store, client, taxonomy_version=version))
        rprint(f"\n[bold]Query:[/bold] {result['query']}")
        if result["patterns"]:
            rprint("\n[bold]Matching agentic patterns:[/bold]")
            for p in result["patterns"]:
                rprint(f"  • [{p.get('category')}] {p['name']} — {p.get('description', '')}")
        else:
            rprint("[yellow]No matching agentic patterns found.[/yellow]")
    else:
        from lens.serve.analyzer import analyze as do_analyze

        result = asyncio.run(do_analyze(query, store, client, taxonomy_version=version))
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
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
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
    result = asyncio.run(do_explain(concept, store, client, taxonomy_version=version, focus=focus))

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
def extract(
    paper_id: str | None = typer.Option(None, "--paper-id", help="Extract specific paper."),
    model: str | None = typer.Option(None, "--model", help="LLM model override."),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent LLM calls."),
) -> None:
    """Extract tradeoffs, architecture, and agentic patterns from papers."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    llm_model = model or config["llm"]["extract_model"]

    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.extract.extractor import extract_papers
    from lens.llm.client import LLMClient

    client = LLMClient(model=llm_model, **_llm_kwargs(config))
    count = asyncio.run(extract_papers(store, client, concurrency=concurrency, paper_id=paper_id))
    rprint(f"[green]Extracted {count} papers[/green]")


@app.command()
def monitor(
    interval: str = typer.Option("weekly", "--interval", help="Check interval (not yet used)."),
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
) -> None:
    """Run one monitoring cycle: acquire → extract → ideate."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    if trending:
        from lens.taxonomy.versioning import get_latest_version

        version = get_latest_version(store)
        if version is None:
            rprint("[yellow]No taxonomy yet.[/yellow]")
            raise typer.Exit(code=0)

        gaps = store.get_table("ideation_gaps").to_polars()
        gaps = gaps.filter(pl.col("taxonomy_version") == version)
        if len(gaps) == 0:
            rprint("[yellow]No ideation gaps found.[/yellow]")
        else:
            rprint(f"\n[bold]Ideation Gaps (v{version}):[/bold]")
            for row in gaps.sort("score", descending=True).to_dicts():
                hyp = ""
                if row.get("llm_hypothesis"):
                    h = row["llm_hypothesis"][:80]
                    hyp = f" — {h}..."
                rprint(f"  [{row['gap_type']}] {row['description']}{hyp}")
        raise typer.Exit(code=0)

    from lens.llm.client import LLMClient
    from lens.monitor.watcher import run_monitor_cycle

    client = LLMClient(model=config["llm"]["extract_model"], **_llm_kwargs(config))
    cats = config["acquire"]["arxiv_categories"]
    monitor_cfg = config["monitor"]
    result = asyncio.run(
        run_monitor_cycle(
            store,
            client,
            categories=cats,
            run_ideation_flag=monitor_cfg["ideate"],
        )
    )
    rprint("[green]Monitor cycle complete:[/green]")
    rprint(f"  Papers acquired: {result['papers_acquired']}")
    rprint(f"  Papers extracted: {result['papers_extracted']}")
    if result.get("ideation_report"):
        rprint(f"  Gaps found: {result['ideation_report']['gap_count']}")


# ---------------------------------------------------------------------------
# Acquire subcommands
# ---------------------------------------------------------------------------


@acquire_app.command()
def seed() -> None:
    """Ingest curated seed papers from the manifest."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()
    count = asyncio.run(_acquire_seed_async(store))
    rprint(f"[green]Acquired {count} seed papers[/green]")


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
    store = LensStore(str(data_dir))
    store.init_tables()

    papers = asyncio.run(_fetch_arxiv_async(query, categories, since, max_results))
    if not papers:
        rprint("[yellow]No papers found[/yellow]")
        return

    # Add placeholder embeddings for papers without them
    for p in papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * EMBEDDING_DIM

    store.add_papers(papers)
    rprint(f"[green]Acquired {len(papers)} papers from arxiv[/green]")


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
    store = LensStore(str(data_dir))
    store.init_tables()

    paper = ingest_pdf(path)

    # Check for duplicate paper_id
    existing_df = store.get_table("papers").to_polars()
    if len(existing_df) > 0 and paper["paper_id"] in existing_df["paper_id"].to_list():
        rprint(f"[yellow]Paper '{paper['paper_id']}' already exists. Skipping.[/yellow]")
        return

    store.add_papers([paper])
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
    store = LensStore(str(data_dir))
    store.init_tables()

    df = store.get_table("papers").to_polars()
    if len(df) == 0:
        rprint("[yellow]No papers to enrich[/yellow]")
        return

    papers = df.drop("embedding").to_dicts()
    mailto = config["acquire"].get("openalex_mailto", "")
    enriched = asyncio.run(_enrich_openalex_async(papers, mailto=mailto))

    # Persist enrichment back to LanceDB
    papers_table = store.get_table("papers")
    updated_count = 0
    for paper in enriched:
        pid = escape_sql_string(paper.get("paper_id", ""))
        papers_table.update(
            where=f"paper_id = '{pid}'",
            values={"citations": paper.get("citations", 0), "venue": paper.get("venue")},
        )
        updated_count += 1
    rprint(f"[green]Enriched {updated_count} papers with OpenAlex data[/green]")


async def _enrich_openalex_async(papers, mailto: str = ""):
    from lens.acquire.openalex import enrich_with_openalex

    return await enrich_with_openalex(papers, mailto=mailto)


# ---------------------------------------------------------------------------
# Build subcommands
# ---------------------------------------------------------------------------


@build_app.command()
def taxonomy() -> None:
    """Build taxonomy by clustering extraction strings."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.llm.client import LLMClient
    from lens.taxonomy import build_taxonomy

    llm_model = config["llm"]["label_model"]
    client = LLMClient(model=llm_model, **_llm_kwargs(config))
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
            target_arch_variants=tax_config["target_arch_variants"],
            target_agentic_patterns=tax_config["target_agentic_patterns"],
            embedding_provider=tax_config.get("embedding_provider", "local"),
            embedding_model=tax_config.get("embedding_model"),
        )
    )
    rprint(f"[green]Built taxonomy version {version}[/green]")


@build_app.command(name="matrix")
def build_matrix_cmd() -> None:
    """Build contradiction matrix from taxonomy."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy yet. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)
    build_matrix(store, taxonomy_version=version)
    rprint(f"[green]Built matrix for taxonomy v{version}[/green]")


@build_app.command(name="all")
def build_all() -> None:
    """Full rebuild: taxonomy + matrix."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.llm.client import LLMClient
    from lens.taxonomy import build_taxonomy

    llm_model = config["llm"]["label_model"]
    client = LLMClient(model=llm_model, **_llm_kwargs(config))
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
            target_arch_variants=tax_config["target_arch_variants"],
            target_agentic_patterns=tax_config["target_agentic_patterns"],
            embedding_provider=tax_config.get("embedding_provider", "local"),
            embedding_model=tax_config.get("embedding_model"),
        )
    )
    build_matrix(store, taxonomy_version=version)
    rprint(f"[green]Built taxonomy v{version} + matrix[/green]")


# ---------------------------------------------------------------------------
# Explore subcommands
# ---------------------------------------------------------------------------


@explore_app.command()
def parameters() -> None:
    """List all taxonomy parameters."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import list_parameters
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    params = list_parameters(store, taxonomy_version=version)
    if not params:
        rprint("[yellow]No parameters found.[/yellow]")
        return
    for p in params:
        rprint(f"[bold]{p['id']}[/bold] {p['name']} — {p['description']}")


@explore_app.command()
def principles() -> None:
    """List all taxonomy principles."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import list_principles
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    princs = list_principles(store, taxonomy_version=version)
    if not princs:
        rprint("[yellow]No principles found.[/yellow]")
        return
    for p in princs:
        rprint(f"[bold]{p['id']}[/bold] {p['name']} — {p['description']}")


@explore_app.command()
def matrix(
    param_a: int | None = typer.Argument(None, help="First parameter ID."),
    param_b: int | None = typer.Argument(None, help="Second parameter ID."),
) -> None:
    """Explore the parameter-principle matrix."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import get_matrix_cell, list_matrix_overview
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    if param_a is not None and param_b is not None:
        cells = get_matrix_cell(store, param_a, param_b, taxonomy_version=version)
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
        overview = list_matrix_overview(store, taxonomy_version=version)
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
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import list_architecture_slots, list_architecture_variants
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    if slot is None:
        slots = list_architecture_slots(store, taxonomy_version=version)
        if not slots:
            rprint("[yellow]No architecture slots found.[/yellow]")
            return
        for s in slots:
            rprint(f"[bold]{s['name']}[/bold] — {s.get('variant_count', 0)} variant(s)")
    else:
        variants = list_architecture_variants(store, slot_name=slot, taxonomy_version=version)
        if not variants:
            rprint(f"[yellow]No variants found for slot '{slot}'.[/yellow]")
            return
        for v in variants:
            props = v.get("properties") or ""
            rprint(
                f"  [bold]{v['name']}[/bold] — {props}" if props else f"  [bold]{v['name']}[/bold]"
            )


@explore_app.command()
def agents(
    category: str | None = typer.Argument(None, help="Agentic pattern category."),
) -> None:
    """Explore agentic patterns."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import list_agentic_patterns
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    patterns = list_agentic_patterns(store, taxonomy_version=version, category=category)
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
                rprint(f"  • {p['name']} — {p.get('description', '')}")
    else:
        for p in patterns:
            rprint(f"  • {p['name']} — {p.get('description', '')}")


@explore_app.command()
def evolution(
    slot: str = typer.Argument(..., help="Architecture slot to trace evolution for."),
) -> None:
    """Explore evolution of an architecture slot over time."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import get_architecture_timeline
    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[red]No taxonomy. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)

    timeline = get_architecture_timeline(store, slot_name=slot, taxonomy_version=version)
    if not timeline:
        rprint(f"[yellow]No variants found for slot '{slot}'.[/yellow]")
        return

    rprint(f"\n[bold]Evolution of '{slot}':[/bold]")
    for v in timeline:
        date_str = v.get("earliest_date") or "unknown date"
        replaces = v.get("replaces")
        replaces_str = f" (replaces: {replaces})" if replaces else ""
        rprint(f"  {date_str}  [bold]{v['name']}[/bold]{replaces_str}")


@explore_app.command()
def paper(
    paper_id: str = typer.Argument(..., help="Paper ID to inspect."),
) -> None:
    """Inspect a specific paper."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.serve.explorer import get_paper

    result = get_paper(store, paper_id)
    if result is None:
        rprint(f"[yellow]Paper '{paper_id}' not found.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"\n[bold]{result.get('title', paper_id)}[/bold]")
    if result.get("authors"):
        rprint(f"[dim]Authors:[/dim] {result['authors']}")
    if result.get("year"):
        rprint(f"[dim]Year:[/dim] {result['year']}")
    if result.get("abstract"):
        rprint(f"\n{result['abstract']}")


@explore_app.command()
def ideas(
    type_: str | None = typer.Option(None, "--type", help="Gap type filter."),
) -> None:
    """Browse ideation gaps and research opportunities."""
    config = load_config(_get_config_path())
    store = LensStore(str(_get_data_dir(config)))
    store.init_tables()

    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[yellow]No taxonomy yet.[/yellow]")
        raise typer.Exit(code=0)

    gaps = store.get_table("ideation_gaps").to_polars()
    gaps = gaps.filter(pl.col("taxonomy_version") == version)

    if type_:
        gaps = gaps.filter(pl.col("gap_type") == type_)

    if len(gaps) == 0:
        rprint("[yellow]No ideation gaps found.[/yellow]")
        raise typer.Exit(code=0)

    rprint(f"\n[bold]Research Opportunities ({len(gaps)} gaps):[/bold]\n")
    for row in gaps.sort("score", descending=True).to_dicts():
        rprint(f"  [bold][{row['gap_type']}][/bold] {row['description']}")
        if row.get("llm_hypothesis"):
            rprint(f"    → {row['llm_hypothesis']}")
        rprint()


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


@config_app.command()
def show() -> None:
    """Show the current LENS configuration."""
    config_path = _get_config_path()
    config = load_config(config_path)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False), end="")


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
