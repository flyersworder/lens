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
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

        Both backends accept an optional ``where`` clause; the filter
        column references must be qualified with the alias ``t`` on the
        joined main row.
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
__all__: list[str] = ["ReadableStore"]


# Defensive runtime registration: if either concrete backend ever drifts
# from the protocol surface, this fails at import time rather than at
# the first request. Cheap insurance.
def _verify_backends_satisfy_protocol() -> None:
    from lens.store.store import LensStore  # noqa: PLC0415 — lazy, avoids cycle

    _check_protocol_methods(LensStore)
    try:
        from lens.store.turso_store import TursoStore  # noqa: PLC0415
    except ImportError:
        # TursoStore requires the optional `turso` extra. If it's not
        # installed, we don't enforce the protocol against it.
        return
    _check_protocol_methods(TursoStore)


def _check_protocol_methods(cls: Any) -> None:
    required = {"query", "query_sql", "vector_search", "search_papers", "hybrid_search"}
    missing = required - {name for name in dir(cls) if not name.startswith("_")}
    if missing:
        raise TypeError(
            f"{cls.__name__} is missing methods required by ReadableStore: {sorted(missing)}"
        )


_verify_backends_satisfy_protocol()
