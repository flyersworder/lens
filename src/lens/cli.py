"""LENS CLI — LLM Engineering Navigation System command-line interface."""
import os
import shutil
from pathlib import Path
from typing import Optional

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


def _get_config_path() -> Optional[Path]:
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
    type_: Optional[str] = typer.Option(None, "--type", help="Analysis type."),
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
    paper_id: Optional[str] = typer.Option(None, "--paper-id", help="Paper ID to extract from."),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model to use."),
    concurrency: int = typer.Option(1, "--concurrency", help="Number of concurrent extractions."),
) -> None:
    """Extract structured data from papers. [stub]"""
    rprint("[yellow]extract not yet implemented[/yellow]")
    raise typer.Exit(code=0)


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
    """Seed the paper database with curated papers. [stub]"""
    rprint("[yellow]acquire seed not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def arxiv(
    query: Optional[str] = typer.Option(None, "--query", help="ArXiv search query."),
    since: Optional[str] = typer.Option(None, "--since", help="Fetch papers since this date (YYYY-MM-DD)."),
) -> None:
    """Acquire papers from ArXiv. [stub]"""
    rprint("[yellow]acquire arxiv not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def file(
    path: Path = typer.Argument(..., help="Path to the paper file."),
) -> None:
    """Acquire a paper from a local file. [stub]"""
    rprint("[yellow]acquire file not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def openalex(
    enrich: bool = typer.Option(False, "--enrich", help="Enrich existing papers with OpenAlex metadata."),
) -> None:
    """Acquire papers from OpenAlex. [stub]"""
    rprint("[yellow]acquire openalex not yet implemented[/yellow]")
    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Build subcommands
# ---------------------------------------------------------------------------

@build_app.command()
def taxonomy() -> None:
    """Build the taxonomy from extracted data. [stub]"""
    rprint("[yellow]build taxonomy not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@build_app.command()
def matrix() -> None:
    """Build the parameter-principle matrix. [stub]"""
    rprint("[yellow]build matrix not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@build_app.command(name="all")
def build_all() -> None:
    """Run all build steps in sequence. [stub]"""
    rprint("[yellow]build all not yet implemented[/yellow]")
    raise typer.Exit(code=0)


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
    param_a: Optional[str] = typer.Argument(None, help="First parameter."),
    param_b: Optional[str] = typer.Argument(None, help="Second parameter."),
) -> None:
    """Explore the parameter-principle matrix. [stub]"""
    rprint("[yellow]explore matrix not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def architecture(
    slot: Optional[str] = typer.Argument(None, help="Architecture slot to explore."),
) -> None:
    """Explore architecture slots. [stub]"""
    rprint("[yellow]explore architecture not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def agents(
    category: Optional[str] = typer.Argument(None, help="Agentic pattern category."),
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
    type_: Optional[str] = typer.Option(None, "--type", help="Idea type filter."),
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
