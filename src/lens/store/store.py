"""LensStore — SQLite + sqlite-vec backend for LENS."""

from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime
from typing import Any

import sqlite_vec

from lens.store.models import EMBEDDING_DIM

# Tables that have a companion vec0 virtual table.
# Maps table_name -> primary key column name and type.
VEC_TABLES: dict[str, tuple[str, str]] = {
    "papers": ("paper_id", "TEXT"),
    "parameters": ("id", "INTEGER"),
    "principles": ("id", "INTEGER"),
    "vocabulary": ("id", "TEXT"),
    "architecture_variants": ("id", "INTEGER"),
    "agentic_patterns": ("id", "INTEGER"),
}

# Maps table_name -> set of columns that are JSON-serialized lists.
JSON_FIELDS: dict[str, set[str]] = {
    "papers": {"authors"},
    "agentic_extractions": {"components"},
    "parameters": {"raw_strings", "paper_ids"},
    "principles": {"sub_techniques", "raw_strings", "paper_ids"},
    "architecture_variants": {"replaces", "paper_ids"},
    "agentic_patterns": {"components", "use_cases", "paper_ids"},
    "matrix_cells": {"paper_ids"},
    "ideation_gaps": {"related_params", "related_principles", "related_slots"},
}

# SQL CREATE TABLE statements for all 13 regular tables.
_TABLE_DDL = [
    """CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        abstract TEXT NOT NULL,
        authors TEXT NOT NULL,
        venue TEXT,
        date TEXT NOT NULL,
        arxiv_id TEXT NOT NULL,
        citations INTEGER NOT NULL DEFAULT 0,
        quality_score REAL NOT NULL DEFAULT 0.0,
        extraction_status TEXT NOT NULL DEFAULT 'pending'
    )""",
    """CREATE TABLE IF NOT EXISTS tradeoff_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        improves TEXT NOT NULL,
        worsens TEXT NOT NULL,
        technique TEXT NOT NULL,
        context TEXT NOT NULL,
        confidence REAL NOT NULL,
        evidence_quote TEXT NOT NULL,
        new_concept_description TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS architecture_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        component_slot TEXT NOT NULL,
        variant_name TEXT NOT NULL,
        replaces TEXT,
        key_properties TEXT NOT NULL,
        confidence REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS agentic_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        pattern_name TEXT NOT NULL,
        structure TEXT NOT NULL,
        use_case TEXT NOT NULL,
        components TEXT NOT NULL,
        confidence REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS parameters (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        raw_strings TEXT NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS principles (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        sub_techniques TEXT NOT NULL,
        raw_strings TEXT NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS architecture_slots (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS architecture_variants (
        id INTEGER PRIMARY KEY,
        slot_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        replaces TEXT NOT NULL,
        properties TEXT NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS agentic_patterns (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT NOT NULL,
        components TEXT NOT NULL,
        use_cases TEXT NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS vocabulary (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        description TEXT NOT NULL,
        source TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        paper_count INTEGER NOT NULL DEFAULT 0,
        avg_confidence REAL NOT NULL DEFAULT 0.0
    )""",
    """CREATE TABLE IF NOT EXISTS matrix_cells (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        improving_param_id TEXT NOT NULL,
        worsening_param_id TEXT NOT NULL,
        principle_id TEXT NOT NULL,
        count INTEGER NOT NULL,
        avg_confidence REAL NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS taxonomy_versions (
        version_id INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL,
        paper_count INTEGER NOT NULL,
        param_count INTEGER NOT NULL,
        principle_count INTEGER NOT NULL,
        slot_count INTEGER NOT NULL DEFAULT 0,
        variant_count INTEGER NOT NULL DEFAULT 0,
        pattern_count INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS ideation_reports (
        id INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL,
        paper_batch_size INTEGER NOT NULL,
        gap_count INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS ideation_gaps (
        id INTEGER PRIMARY KEY,
        report_id INTEGER NOT NULL,
        gap_type TEXT NOT NULL,
        description TEXT NOT NULL,
        related_params TEXT NOT NULL,
        related_principles TEXT NOT NULL,
        related_slots TEXT NOT NULL,
        score REAL NOT NULL,
        llm_hypothesis TEXT,
        created_at TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
]


def _pack_embedding(emb: list[float]) -> bytes:
    """Pack a float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(emb)}f", *emb)


class LensStore:
    """High-level interface to the LENS SQLite database with sqlite-vec.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. If the path does not end with
        ``.db`` the suffix is appended automatically.
    """

    def __init__(self, db_path: str) -> None:
        if not db_path.endswith(".db"):
            db_path = db_path + ".db"
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA journal_mode=WAL")

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """Create all LENS tables (regular + vec) if they do not exist."""
        for ddl in _TABLE_DDL:
            self.conn.execute(ddl)

        for table_name, (id_col, id_type) in VEC_TABLES.items():
            vec_ddl = (
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vec "
                f"USING vec0("
                f"{id_col} {id_type} PRIMARY KEY, "
                f"embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine"
                f")"
            )
            self.conn.execute(vec_ddl)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def add_rows(self, table: str, rows: list[dict]) -> None:
        """INSERT rows into *table*, auto-serializing JSON fields and extracting embeddings."""
        if not rows:
            return

        json_cols = JSON_FIELDS.get(table, set())
        vec_info = VEC_TABLES.get(table)  # (id_col, id_type) or None

        for row in rows:
            processed = {}
            embedding = None

            for key, value in row.items():
                if key == "embedding":
                    embedding = value
                    continue
                if key in json_cols and isinstance(value, list):
                    processed[key] = json.dumps(value)
                elif isinstance(value, datetime):
                    processed[key] = value.isoformat()
                else:
                    processed[key] = value

            cols = list(processed.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            vals = [processed[c] for c in cols]

            self.conn.execute(
                f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
                vals,
            )

            # Insert into companion vec table if applicable.
            if vec_info and embedding is not None:
                id_col, _ = vec_info
                id_val = processed[id_col]
                self.conn.execute(
                    f"INSERT INTO {table}_vec ({id_col}, embedding) VALUES (?, ?)",
                    (id_val, _pack_embedding(embedding)),
                )

        self.conn.commit()

    def add_papers(self, data: list[dict]) -> None:
        """Convenience alias for ``add_rows("papers", data)``."""
        self.add_rows("papers", data)

    def query(self, table: str, where: str = "", params: tuple | None = None) -> list[dict]:
        """SELECT * from *table* with optional WHERE clause.

        JSON list fields are automatically deserialized back to Python lists.
        """
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cursor = self.conn.execute(sql, params or ())
        columns = [desc[0] for desc in cursor.description]
        json_cols = JSON_FIELDS.get(table, set())

        results = []
        for row in cursor.fetchall():
            d: dict[str, Any] = {}
            for i, col in enumerate(columns):
                val = row[i]
                if col in json_cols and isinstance(val, str):
                    d[col] = json.loads(val)
                else:
                    d[col] = val
            results.append(d)
        return results

    def query_sql(self, sql: str, params: tuple | None = None) -> list[dict]:
        """Execute raw SQL and return list[dict]. No JSON deserialization."""
        cursor = self.conn.execute(sql, params or ())
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def update(self, table: str, values: str, where: str, params: tuple | None = None) -> None:
        """UPDATE *table* SET *values* WHERE *where*."""
        self.conn.execute(
            f"UPDATE {table} SET {values} WHERE {where}",
            params or (),
        )
        self.conn.commit()

    def delete(self, table: str, where: str, params: tuple | None = None) -> None:
        """DELETE from *table* WHERE *where*. Also deletes from companion vec table."""
        vec_info = VEC_TABLES.get(table)
        if vec_info:
            id_col, _ = vec_info
            # Fetch IDs to delete from vec table before deleting from main table.
            id_rows = self.conn.execute(
                f"SELECT {id_col} FROM {table} WHERE {where}",
                params or (),
            ).fetchall()
            if id_rows:
                id_placeholders = ", ".join(["?"] * len(id_rows))
                id_vals = [r[0] for r in id_rows]
                self.conn.execute(
                    f"DELETE FROM {table}_vec WHERE {id_col} IN ({id_placeholders})",
                    id_vals,
                )

        self.conn.execute(f"DELETE FROM {table} WHERE {where}", params or ())
        self.conn.commit()

    def vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int = 5,
        where: str = "",
        params: tuple | None = None,
    ) -> list[dict]:
        """k-NN search via sqlite-vec with optional WHERE filter on the main table.

        For filtered search, over-fetches by 3x then applies the filter via JOIN.
        """
        vec_info = VEC_TABLES.get(table)
        if not vec_info:
            raise ValueError(f"Table '{table}' does not have a companion vec table")

        id_col, _ = vec_info
        emb_bytes = _pack_embedding(embedding)
        json_cols = JSON_FIELDS.get(table, set())

        if where:
            # Over-fetch 3x, JOIN with main table, apply filter, truncate.
            fetch_limit = limit * 3
            sql = (
                f"SELECT t.*, v.distance AS _distance FROM {table} t "
                f"INNER JOIN {table}_vec v ON t.{id_col} = v.{id_col} "
                f"WHERE v.embedding MATCH ? AND v.k = ? AND {where} "
                f"LIMIT ?"
            )
            all_params = (emb_bytes, fetch_limit) + (params or ()) + (limit,)
        else:
            sql = (
                f"SELECT t.*, v.distance AS _distance FROM {table} t "
                f"INNER JOIN {table}_vec v ON t.{id_col} = v.{id_col} "
                f"WHERE v.embedding MATCH ? AND v.k = ?"
            )
            all_params = (emb_bytes, limit)

        cursor = self.conn.execute(sql, all_params)
        columns = [desc[0] for desc in cursor.description]

        results = []
        for row in cursor.fetchall():
            d: dict[str, Any] = {}
            for i, col in enumerate(columns):
                val = row[i]
                if col in json_cols and isinstance(val, str):
                    d[col] = json.loads(val)
                else:
                    d[col] = val
            results.append(d)
        return results
