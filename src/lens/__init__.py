"""LENS — LLM Engineering Navigation System.

Public API for programmatic use. CLI is a thin wrapper over these functions.
Pipeline functions (acquire, extract, build) will be implemented in Plans 2-4.
Query functions (analyze, explain, explore) will be implemented in Plan 5.
"""

from lens.config import load_config, resolve_data_dir
from lens.store.models import ExplanationResult
from lens.store.store import LensStore

__all__ = [
    "ExplanationResult",
    "LensStore",
    "load_config",
    "resolve_data_dir",
]
