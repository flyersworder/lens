"""LENS CLI — LLM Engineering Navigation System command-line interface."""

import asyncio
import os
import shutil
from pathlib import Path

import typer
import yaml
from rich import print as rprint

from lens.config import load_config, resolve_data_dir, save_config, set_config_value
from lens.store.store import LensStore

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
    query: str = typer.Argument(..., help="Analysis query."),
    type_: str | None = typer.Option(None, "--type", help="Analysis type."),
) -> None:
    """Analyze the LENS knowledge base. [stub]"""
    rprint("[yellow]analyze not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@app.command()
def explain(
    concept: str = typer.Argument(..., help="Concept to explain."),
    related: bool = typer.Option(False, "--related", help="Show related concepts."),
    evolution: bool = typer.Option(False, "--evolution", help="Show evolution over time."),
    tradeoffs: bool = typer.Option(False, "--tradeoffs", help="Show tradeoffs."),
) -> None:
    """Explain a concept from the LENS knowledge base. [stub]"""
    rprint("[yellow]explain not yet implemented[/yellow]")
    raise typer.Exit(code=0)


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

    client = LLMClient(model=llm_model)
    count = asyncio.run(extract_papers(store, client, concurrency=concurrency, paper_id=paper_id))
    rprint(f"[green]Extracted {count} papers[/green]")


@app.command()
def monitor(
    interval: int = typer.Option(60, "--interval", help="Polling interval in seconds."),
    trending: bool = typer.Option(False, "--trending", help="Show trending papers."),
) -> None:
    """Monitor for new papers. [stub]"""
    rprint("[yellow]monitor not yet implemented[/yellow]")
    raise typer.Exit(code=0)


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
            p["embedding"] = [0.0] * 768

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
    enriched = asyncio.run(_enrich_openalex_async(papers))
    rprint(f"[green]Enriched {len(enriched)} papers with OpenAlex data[/green]")


async def _enrich_openalex_async(papers):
    from lens.acquire.openalex import enrich_with_openalex

    return await enrich_with_openalex(papers)


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
    client = LLMClient(model=llm_model)
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
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
    client = LLMClient(model=llm_model)
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
        )
    )
    build_matrix(store, taxonomy_version=version)
    rprint(f"[green]Built taxonomy v{version} + matrix[/green]")


# ---------------------------------------------------------------------------
# Explore subcommands
# ---------------------------------------------------------------------------


@explore_app.command()
def parameters() -> None:
    """Explore taxonomy parameters. [stub]"""
    rprint("[yellow]explore parameters not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def principles() -> None:
    """Explore taxonomy principles. [stub]"""
    rprint("[yellow]explore principles not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def matrix(
    param_a: str | None = typer.Argument(None, help="First parameter."),
    param_b: str | None = typer.Argument(None, help="Second parameter."),
) -> None:
    """Explore the parameter-principle matrix. [stub]"""
    rprint("[yellow]explore matrix not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def architecture(
    slot: str | None = typer.Argument(None, help="Architecture slot to explore."),
) -> None:
    """Explore architecture slots. [stub]"""
    rprint("[yellow]explore architecture not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def agents(
    category: str | None = typer.Argument(None, help="Agentic pattern category."),
) -> None:
    """Explore agentic patterns. [stub]"""
    rprint("[yellow]explore agents not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def evolution(
    slot: str = typer.Argument(..., help="Architecture slot to trace evolution for."),
) -> None:
    """Explore evolution of an architecture slot over time. [stub]"""
    rprint("[yellow]explore evolution not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def paper(
    paper_id: str = typer.Argument(..., help="Paper ID to inspect."),
) -> None:
    """Inspect a specific paper. [stub]"""
    rprint("[yellow]explore paper not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def ideas(
    type_: str | None = typer.Option(None, "--type", help="Idea type filter."),
) -> None:
    """Explore generated research ideas. [stub]"""
    rprint("[yellow]explore ideas not yet implemented[/yellow]")
    raise typer.Exit(code=0)


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
