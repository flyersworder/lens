"""LENS taxonomy pipeline — vocabulary-based guided extraction."""

from __future__ import annotations

from lens.taxonomy.versioning import get_next_version, record_version  # noqa: F401

# Backward compat alias
from lens.taxonomy.vocabulary import (
    build_tradeoff_taxonomy,  # noqa: F401
    build_vocabulary,  # noqa: F401
)

__all__ = [
    "build_vocabulary",
    "build_tradeoff_taxonomy",
    "get_next_version",
    "record_version",
]
