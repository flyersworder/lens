"""Read-only LENS store backed by Turso (libSQL native vectors).

Mirrors the read API of :class:`lens.store.store.LensStore` so that
``serve/analyzer.py``, ``serve/explainer.py`` and ``serve/explorer.py``
can be wired to either backend without code changes.

Schema convention (produced by ``scripts/publish_to_turso.py``):

* Each indexable table (``papers``, ``vocabulary``) has an ``embedding
  F32_BLOB(EMBEDDING_DIM)`` column on the main table — *no* companion
  ``_vec`` virtual table.
* A libSQL vector index lives at ``<table>_emb_idx`` and is created via
  ``CREATE INDEX <table>_emb_idx ON <table> (libsql_vector_idx(embedding))``.
* FTS5 virtual tables (``papers_fts``, ``vocabulary_fts``) are unchanged
  from the local schema.

Only read methods are implemented. Writes happen on the build side via
the local :class:`LensStore` (sqlite-vec); the publish script then
syncs to Turso.
"""

from __future__ import annotations

import json
import struct
from typing import Any

try:
    import libsql_client
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "TursoStore requires the 'turso' optional extra. Install with: uv sync --extra turso"
    ) from e

from lens.store.store import JSON_FIELDS

# libSQL vector index names. Tables not listed here have no vector index
# and only support keyword/SQL queries.
VECTOR_INDEXES: dict[str, str] = {
    "papers": "papers_emb_idx",
    "vocabulary": "vocabulary_emb_idx",
}


def _pack_embedding(emb: list[float]) -> bytes:
    """Pack a float list into little-endian F32 bytes (libSQL ``vector32`` input)."""
    return struct.pack(f"{len(emb)}f", *emb)


def _normalize_url(url: str) -> str:
    """Convert ``libsql://`` to ``https://`` for the Python libsql-client.

    The ``turso db show`` command emits ``libsql://`` by convention, but
    the Python client interprets that as a WebSocket transport that
    fails with HTTP 505 against current Turso edge servers.
    """
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://") :]
    return url


class TursoStore:
    """Read-only Turso-backed view of a published LENS database.

    Construct with a Turso URL (libsql:// or https://) and an auth token,
    typically loaded from ``TURSO_DATABASE_URL`` / ``TURSO_AUTH_TOKEN``
    environment variables in production.
    """

    def __init__(self, url: str, auth_token: str) -> None:
        self.url = _normalize_url(url)
        self.client = libsql_client.create_client_sync(url=self.url, auth_token=auth_token)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: tuple | list | None = None) -> Any:
        """Execute a single statement and return the libsql ResultSet."""
        return self.client.execute(sql, list(params) if params else [])

    @staticmethod
    def _rows_to_dicts(rs: Any, table: str) -> list[dict]:
        """Convert a libsql ResultSet to a list of dicts with JSON deserialization."""
        json_cols = JSON_FIELDS.get(table, set())
        results = []
        for row in rs.rows:
            d: dict[str, Any] = {}
            for col, val in zip(rs.columns, row, strict=True):
                if col in json_cols and isinstance(val, str):
                    d[col] = json.loads(val)
                else:
                    d[col] = val
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Read API (mirrors LensStore)
    # ------------------------------------------------------------------

    def query(
        self,
        table: str,
        where: str = "",
        params: tuple | None = None,
    ) -> list[dict]:
        """SELECT * from *table* with optional WHERE clause.

        JSON list fields are auto-deserialized back to Python lists.
        """
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        rs = self._execute(sql, params)
        return self._rows_to_dicts(rs, table)

    def query_sql(self, sql: str, params: tuple | None = None) -> list[dict]:
        """Execute raw SQL and return list[dict]. No JSON deserialization.

        Note: scalar types in the returned dicts come from the libSQL HTTP
        protocol, which may differ subtly from sqlite3's adapters
        (e.g., numeric promotion, BLOB representation). Callers that
        compare values across backends should normalize explicitly.
        """
        rs = self._execute(sql, params)
        return [dict(zip(rs.columns, row, strict=True)) for row in rs.rows]

    def vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int = 5,
        where: str = "",
        params: tuple | None = None,
    ) -> list[dict]:
        """k-NN search via libSQL native vector index.

        Translates the sqlite-vec ``vec0`` MATCH pattern into the libSQL
        ``vector_top_k(idx, vector32(?), k) JOIN tbl ON rowid`` pattern.

        For filtered search, over-fetches by 3x then applies ``where``
        via the JOIN clause.
        """
        idx = VECTOR_INDEXES.get(table)
        if not idx:
            raise ValueError(
                f"Table '{table}' has no libSQL vector index registered "
                f"(known: {sorted(VECTOR_INDEXES)})"
            )

        emb_bytes = _pack_embedding(embedding)

        # libSQL's `vector_top_k` returns only the matched `id` (rowid);
        # cosine distance must be computed via `vector_distance_cos` on the
        # joined row's embedding column. The query vector is bound twice.
        if where:
            fetch_limit = limit * 3
            sql = (
                f"SELECT t.*, vector_distance_cos(t.embedding, vector32(?)) AS _distance "
                f"FROM vector_top_k('{idx}', vector32(?), ?) AS k "
                f"JOIN {table} t ON t.rowid = k.id "
                f"WHERE {where} "
                f"LIMIT ?"
            )
            all_params = (emb_bytes, emb_bytes, fetch_limit, *(params or ()), limit)
        else:
            sql = (
                f"SELECT t.*, vector_distance_cos(t.embedding, vector32(?)) AS _distance "
                f"FROM vector_top_k('{idx}', vector32(?), ?) AS k "
                f"JOIN {table} t ON t.rowid = k.id "
                f"LIMIT ?"
            )
            # Tail LIMIT mirrors the filtered branch: defends against any
            # JOIN-side row duplication and keeps both branches symmetric.
            all_params = (emb_bytes, emb_bytes, limit, limit)

        rs = self._execute(sql, all_params)
        return self._rows_to_dicts(rs, table)

    def search_papers(
        self,
        query: str | None = None,
        embedding: list[float] | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid FTS5 + libSQL-vector search on papers, fused with RRF.

        Mirrors :meth:`LensStore.search_papers`. Modes:

        * Hybrid (query + embedding): FTS5 keyword + vector with RRF.
        * FTS-only (query, no embedding): keyword-only.
        * Filter-only (no query): plain SQL ordered by date DESC.
        """
        # --- Build filter clauses (used in all modes) ---
        filter_clauses: list[str] = []
        filter_params: list[Any] = []
        if filters:
            if "author" in filters:
                filter_clauses.append("t.authors LIKE ?")
                filter_params.append(f"%{filters['author']}%")
            if "venue" in filters:
                filter_clauses.append("t.venue LIKE ?")
                filter_params.append(f"%{filters['venue']}%")
            if "after" in filters:
                filter_clauses.append("t.date >= ?")
                filter_params.append(filters["after"])
            if "before" in filters:
                filter_clauses.append("t.date <= ?")
                filter_params.append(filters["before"])
        filter_where = " AND ".join(filter_clauses) if filter_clauses else ""

        # --- Filter-only mode ---
        if not query:
            sql = "SELECT * FROM papers t"
            if filter_where:
                sql += f" WHERE {filter_where}"
            sql += " ORDER BY t.date DESC LIMIT ?"
            rs = self._execute(sql, (*filter_params, limit))
            return self._rows_to_dicts(rs, "papers")

        # FTS5: wrap each token in double quotes, OR-combine.
        fts_query = " OR ".join(f'"{token}"' for token in query.strip().split())

        # --- FTS-only mode ---
        if embedding is None:
            sql = """
            WITH fts_matches AS (
                SELECT
                    rowid,
                    row_number() OVER (ORDER BY rank) AS rank_number
                FROM papers_fts
                WHERE papers_fts MATCH ?
                LIMIT ?
            ),
            combined AS (
                SELECT
                    f.rowid,
                    1.0 / (? + f.rank_number) AS fts_score
                FROM fts_matches f
            )
            SELECT
                t.*,
                c.fts_score AS _rrf_score
            FROM combined c
            JOIN papers t ON t.rowid = c.rowid
            """
            if filter_where:
                sql += f" WHERE {filter_where}"
            sql += " ORDER BY _rrf_score DESC LIMIT ?"
            params_tuple = (
                fts_query,
                limit * 3,
                rrf_k,
                *filter_params,
                limit,
            )
            rs = self._execute(sql, params_tuple)
            return self._rows_to_dicts(rs, "papers")

        # --- Hybrid mode ---
        emb_bytes = _pack_embedding(embedding)
        idx = VECTOR_INDEXES["papers"]
        sql = f"""
        WITH fts_matches AS (
            SELECT
                rowid,
                row_number() OVER (ORDER BY rank) AS rank_number
            FROM papers_fts
            WHERE papers_fts MATCH ?
            LIMIT ?
        ),
        vec_matches AS (
            -- Compute distance once via a subquery, then rank.
            SELECT
                rowid,
                distance,
                row_number() OVER (ORDER BY distance) AS rank_number
            FROM (
                SELECT
                    k.id AS rowid,
                    vector_distance_cos(p.embedding, vector32(?)) AS distance
                FROM vector_top_k('{idx}', vector32(?), ?) AS k
                JOIN papers p ON p.rowid = k.id
            )
        ),
        combined AS (
            SELECT
                p.rowid AS rowid,
                coalesce(1.0 / (? + f.rank_number), 0.0) AS fts_score,
                coalesce(1.0 / (? + vm.rank_number), 0.0) AS vec_score,
                vm.distance AS vec_distance
            FROM papers p
            LEFT JOIN fts_matches f ON p.rowid = f.rowid
            LEFT JOIN vec_matches vm ON p.rowid = vm.rowid
            WHERE f.rowid IS NOT NULL OR vm.rowid IS NOT NULL
        )
        SELECT
            t.*,
            c.fts_score + c.vec_score AS _rrf_score,
            c.vec_distance AS _distance
        FROM combined c
        JOIN papers t ON t.rowid = c.rowid
        """
        if filter_where:
            sql += f" WHERE {filter_where}"
        sql += " ORDER BY _rrf_score DESC LIMIT ?"
        params_tuple = (
            fts_query,
            limit * 3,
            emb_bytes,  # vector_distance_cos in subquery
            emb_bytes,  # vector_top_k argument
            limit * 3,  # k for vector_top_k
            rrf_k,
            rrf_k,
            *filter_params,
            limit,
        )
        rs = self._execute(sql, params_tuple)
        return self._rows_to_dicts(rs, "papers")

    def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid FTS5 + libSQL-vector search on vocabulary, fused with RRF.

        Mirrors :meth:`LensStore.hybrid_search`.
        """
        emb_bytes = _pack_embedding(embedding)
        idx = VECTOR_INDEXES["vocabulary"]
        fts_query = " OR ".join(f'"{token}"' for token in query.strip().split())

        sql = f"""
        WITH fts_matches AS (
            SELECT
                rowid,
                row_number() OVER (ORDER BY rank) AS rank_number
            FROM vocabulary_fts
            WHERE vocabulary_fts MATCH ?
            LIMIT ?
        ),
        vec_matches AS (
            -- Compute distance once via a subquery, then rank.
            SELECT
                rowid,
                distance,
                row_number() OVER (ORDER BY distance) AS rank_number
            FROM (
                SELECT
                    k.id AS rowid,
                    vector_distance_cos(v.embedding, vector32(?)) AS distance
                FROM vector_top_k('{idx}', vector32(?), ?) AS k
                JOIN vocabulary v ON v.rowid = k.id
            )
        ),
        combined AS (
            SELECT
                v.rowid AS rowid,
                coalesce(1.0 / (? + f.rank_number), 0.0) AS fts_score,
                coalesce(1.0 / (? + vm.rank_number), 0.0) AS vec_score,
                vm.distance AS vec_distance
            FROM vocabulary v
            LEFT JOIN fts_matches f ON v.rowid = f.rowid
            LEFT JOIN vec_matches vm ON v.rowid = vm.rowid
            WHERE f.rowid IS NOT NULL OR vm.rowid IS NOT NULL
        )
        SELECT
            t.*,
            c.fts_score + c.vec_score AS _rrf_score,
            c.vec_distance AS _distance
        FROM combined c
        JOIN vocabulary t ON t.rowid = c.rowid
        ORDER BY _rrf_score DESC
        LIMIT ?
        """
        params = (
            fts_query,
            limit * 3,
            emb_bytes,  # vector_distance_cos in subquery
            emb_bytes,  # vector_top_k argument
            limit * 3,  # k for vector_top_k
            rrf_k,
            rrf_k,
            limit,
        )
        rs = self._execute(sql, params)
        return self._rows_to_dicts(rs, "vocabulary")
