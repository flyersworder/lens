"""LensStore — LanceDB connection and table management."""

from __future__ import annotations

from typing import Any

import lancedb
import polars as pl

from lens.store.models import (
    AgenticExtraction,
    AgenticPattern,
    ArchitectureExtraction,
    ArchitectureSlot,
    ArchitectureVariant,
    IdeationGap,
    IdeationReport,
    MatrixCell,
    Paper,
    Parameter,
    Principle,
    TaxonomyVersion,
    TradeoffExtraction,
)

# Maps table names to their LanceModel schema classes.
TABLE_SCHEMAS: dict[str, type] = {
    "papers": Paper,
    "tradeoff_extractions": TradeoffExtraction,
    "architecture_extractions": ArchitectureExtraction,
    "agentic_extractions": AgenticExtraction,
    "parameters": Parameter,
    "principles": Principle,
    "architecture_slots": ArchitectureSlot,
    "architecture_variants": ArchitectureVariant,
    "agentic_patterns": AgenticPattern,
    "matrix_cells": MatrixCell,
    "taxonomy_versions": TaxonomyVersion,
    "ideation_reports": IdeationReport,
    "ideation_gaps": IdeationGap,
}


def escape_sql_string(value: str) -> str:
    """Escape single quotes in a string for LanceDB SQL filter expressions."""
    return value.replace("'", "''")


class _TableWrapper:
    """Thin wrapper around a LanceDB table that normalises the to_polars() API.

    LanceDB >= 0.17 returns a ``polars.LazyFrame`` from ``table.to_polars()``.
    Test code expects a concrete ``polars.DataFrame``, so we collect here.
    All other attribute accesses are forwarded transparently to the underlying
    LanceDB table object.
    """

    def __init__(self, table: object) -> None:
        self._table = table

    def to_polars(self) -> pl.DataFrame:
        result = self._table.to_polars()  # type: ignore[attr-defined]
        if hasattr(result, "collect"):
            return result.collect()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


class _DatabaseWrapper:
    """Thin wrapper around a LanceDB database that normalises the table_names() API.

    The default ``table_names()`` call uses a limit of 10, which truncates
    results when more than 10 tables exist.  This wrapper overrides
    ``table_names()`` to fetch all table names without pagination.
    All other attribute accesses are forwarded transparently.
    """

    def __init__(self, db: object) -> None:
        self._db = db

    def table_names(self, **kwargs: object) -> list[str]:
        """Return all table names, bypassing the default page-size limit."""
        result = self._db.list_tables(limit=10_000)  # type: ignore[attr-defined]
        return result.tables if hasattr(result, "tables") else list(result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


class LensStore:
    """High-level interface to the LENS LanceDB database.

    Parameters
    ----------
    data_dir:
        Path to the LanceDB database directory.  If the path does not end with
        ``.lance`` the suffix ``/lens.lance`` is appended automatically.
    """

    def __init__(self, data_dir: str) -> None:
        if not data_dir.endswith(".lance"):
            data_dir = data_dir.rstrip("/") + "/lens.lance"
        self.db: _DatabaseWrapper = _DatabaseWrapper(lancedb.connect(data_dir))

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """Create all LENS tables that do not yet exist in the database."""
        existing = set(self.db.table_names())
        for name, schema in TABLE_SCHEMAS.items():
            if name not in existing:
                self.db.create_table(name, schema=schema, exist_ok=True)

    def get_table(self, name: str) -> _TableWrapper:
        """Return a wrapped LanceDB table by name."""
        return _TableWrapper(self.db.open_table(name))

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def add_papers(self, data: list[dict]) -> None:
        """Append rows to the ``papers`` table.

        Parameters
        ----------
        data:
            List of dicts whose keys match the :class:`~lens.store.models.Paper`
            schema.  The table must already exist (call :meth:`init_tables`
            first).
        """
        self.add_rows("papers", data)

    def add_rows(self, table_name: str, data: list[dict]) -> None:
        """Append rows to an arbitrary table.

        Parameters
        ----------
        table_name:
            Name of the target table.
        data:
            List of dicts matching the table schema.
        """
        self.get_table(table_name).add(data)
