"""Structural-typing protocols for LENS storage backends.

The :class:`ReadableStore` protocol lists the read-only API that
``serve/analyzer.py``, ``serve/explainer.py`` and ``serve/explorer.py``
depend on. Both :class:`lens.store.store.LensStore` (sqlite-vec, used
for CLI / build / tests) and :class:`lens.store.turso_store.TursoStore`
(libSQL native, used by the Vercel runtime) satisfy it structurally —
no inheritance, no runtime cost, just a type-level guarantee that the
serve layer can take either backend.

Why a protocol and not a base class:

* ``LensStore`` predates ``TursoStore`` and shouldn't be retrofitted with
  inheritance just to satisfy the typer.
* The protocol surface is *deliberately* the read subset only —
  mutation methods (``add_rows``, ``upsert_embedding``, etc.) live on
  ``LensStore`` and are used during the build pipeline, never by
  ``serve/*``. Forcing those onto ``TursoStore`` would conflict with
  its read-only design.

This module is a thin leaf: importing it should not pull in either
concrete backend. Drift between the protocol and the backends is
caught by ``tests/test_store_protocols.py`` rather than at import
time, so a constrained context (e.g. a serverless bundle that
deliberately excludes one backend) can import the protocol freely.

Notes on what is NOT in the protocol:

* ``close()`` and context-manager methods (``__enter__`` / ``__exit__``):
  both backends expose ``close()`` but ``serve/*`` never calls it —
  the connection lifecycle is managed by whichever process owns the
  store (CLI for ``LensStore``, Vercel function module-globals for
  ``TursoStore``). Adding it to the protocol would obligate every
  future implementation without delivering any guarantee that
  ``serve/*`` actually relies on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReadableStore(Protocol):
    """Minimum read API the serve layer requires.

    Decorated ``runtime_checkable`` so unit tests can ``isinstance``-check
    a candidate object against the protocol; production paths rely on
    static typing and never invoke ``isinstance``.
    """

    def query(
        self,
        table: str,
        where: str = "",
        params: tuple | None = None,
    ) -> list[dict]:
        """SELECT * from *table* with optional WHERE clause.

        JSON columns listed in :data:`lens.store.store.JSON_FIELDS` are
        deserialized back to Python ``list``/``dict``.
        """
        ...

    def query_sql(
        self,
        sql: str,
        params: tuple | None = None,
    ) -> list[dict]:
        """Execute raw SQL and return ``list[dict]``. No JSON deserialization."""
        ...

    def vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int = 5,
        where: str = "",
        params: tuple | None = None,
    ) -> list[dict]:
        """k-NN search against the embedding index for ``table``.

        ``where`` filter syntax (mandatory, both backends):
            Filter columns MUST be qualified with the alias ``t``
            because both backends JOIN the main row as ``t``. Using
            an unaliased column name will fail on TursoStore (the
            libSQL backend) with a SQL error.

            Correct:    ``where="t.kind = ?"``
            Incorrect:  ``where="kind = ?"``

        ``limit`` applies after the filter; both backends over-fetch
        and trim, so ``limit=k`` may return fewer than ``k`` rows
        when ``where`` excludes most candidates.
        """
        ...

    def search_papers(
        self,
        query: str | None = None,
        embedding: list[float] | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid FTS5 + vector search on ``papers`` with RRF fusion.

        Modes:

        * Hybrid (``query`` + ``embedding``): keyword + semantic.
        * FTS-only (``query``, no ``embedding``): keyword only.
        * Filter-only (no ``query``): SQL filter ordered by date DESC.
        """
        ...

    def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid FTS5 + vector search on ``vocabulary`` with RRF fusion."""
        ...


# Sentinel used by serve/* type hints. Implementations:
# - LensStore (sqlite-vec backend, CLI / build / tests)
# - TursoStore (libSQL native backend, Vercel runtime)
#
# Drift detection lives in tests/test_store_protocols.py — see
# `test_lens_store_satisfies_readable_protocol` and
# `test_turso_store_satisfies_readable_protocol`.
__all__: list[str] = ["ReadableStore"]
