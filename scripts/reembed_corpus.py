"""Re-embed the local LENS corpus at the current ``EMBEDDING_DIM``.

Used after switching embedding model or dimension (e.g. local
sentence-transformers @ 768 → cloud ``text-embedding-3-small`` @ 1536).
sqlite-vec virtual tables don't support ``ALTER`` for the embedding
dimension, so this script:

1. Drops ``papers_vec`` and ``vocabulary_vec``.
2. Re-runs :meth:`LensStore.init_tables` — recreates them at
   ``EMBEDDING_DIM`` from :mod:`lens.store.models`.
3. Re-embeds every paper (``title + abstract``) and every vocabulary
   row (``name + description``) via :func:`embed_strings` with the
   ``embeddings.*`` config block.
4. Verifies the new vectors land at the expected dimension.

Idempotent: safe to run multiple times. Read-only against the rest of
the schema (``papers``, ``vocabulary``, etc.) — only the ``_vec``
companions are mutated.

Usage::

    uv run python scripts/reembed_corpus.py
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from lens.config import load_config
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings

logger = logging.getLogger(__name__)


def _embedding_kwargs(config: dict) -> dict:
    """Mirror of cli.py ``_embedding_kwargs`` so this script is standalone."""
    import os

    emb = config.get("embeddings", {})
    kwargs: dict = {}
    if (provider := emb.get("provider", "local")) != "local":
        kwargs["provider"] = provider
    if model := emb.get("model"):
        kwargs["model_name"] = model
    if api_base := emb.get("api_base"):
        kwargs["api_base"] = api_base
    api_key = emb.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _drop_vec_tables(store: LensStore) -> None:
    """Drop the two sqlite-vec virtual tables so init_tables can recreate
    them at the current ``EMBEDDING_DIM``."""
    for table in ("papers_vec", "vocabulary_vec"):
        store.conn.execute(f"DROP TABLE IF EXISTS {table}")
    store.conn.commit()


def _reembed_table(
    store: LensStore,
    table: str,
    id_col: str,
    rows: list[dict],
    text_fn,
    emb_kwargs: dict,
) -> tuple[int, int]:
    """Embed ``rows`` and upsert each into ``table``'s companion _vec.

    Returns ``(embedded, skipped)`` — skipped rows are those with empty
    text (no signal to embed)."""
    keep: list[dict] = []
    texts: list[str] = []
    for r in rows:
        text = text_fn(r)
        if not text or not text.strip():
            continue
        keep.append(r)
        texts.append(text)
    if not texts:
        return 0, len(rows)
    logger.info("embedding %d %s rows...", len(texts), table)
    vectors = embed_strings(texts, **emb_kwargs)
    for row, vec in zip(keep, vectors, strict=True):
        store.upsert_embedding(table, row[id_col], vec.tolist())
    return len(texts), len(rows) - len(texts)


def _verify_dim(store: LensStore, table: str, expected: int) -> None:
    """Probe one row of ``{table}_vec`` and check the blob length matches."""
    cur = store.conn.execute(f"SELECT embedding FROM {table}_vec LIMIT 1")
    row = cur.fetchone()
    if row is None:
        logger.warning("%s_vec is empty — nothing to verify", table)
        return
    blob = row[0]
    actual = len(blob) // 4
    if actual != expected:
        sys.exit(
            f"error: {table}_vec embedding has dim {actual}, expected {expected}. "
            "Did you bump EMBEDDING_DIM but forget to drop the _vec tables?"
        )
    # Also verify struct unpack succeeds — cheap sanity check.
    struct.unpack(f"{expected}f", blob)


def reembed(db_path: str) -> None:
    config = load_config()
    emb_kwargs = _embedding_kwargs(config)
    if emb_kwargs.get("provider") != "cloud":
        logger.warning(
            "embeddings.provider is %r — local re-embed will run. "
            "Set provider: cloud + model: openai/text-embedding-3-small "
            "in ~/.lens/config.yaml to use the dim-1536 OpenAI model.",
            emb_kwargs.get("provider", "local"),
        )

    store = LensStore(db_path)
    started = time.monotonic()

    logger.info("step 1/4: dropping existing _vec tables")
    _drop_vec_tables(store)

    logger.info("step 2/4: recreating _vec tables at EMBEDDING_DIM=%d", EMBEDDING_DIM)
    store.init_tables()

    logger.info("step 3/4: re-embedding papers and vocabulary")
    papers = store.query("papers")
    vocab = store.query("vocabulary")

    p_done, p_skip = _reembed_table(
        store,
        "papers",
        "paper_id",
        papers,
        lambda r: f"{r.get('title', '')}\n\n{r.get('abstract', '')}",
        emb_kwargs,
    )
    v_done, v_skip = _reembed_table(
        store,
        "vocabulary",
        "id",
        vocab,
        lambda r: f"{r.get('name', '')}: {r.get('description', '')}",
        emb_kwargs,
    )

    logger.info("step 4/4: verifying dimensions")
    _verify_dim(store, "papers", EMBEDDING_DIM)
    _verify_dim(store, "vocabulary", EMBEDDING_DIM)

    elapsed = time.monotonic() - started
    logger.info(
        "done in %.1fs: papers=%d (skipped %d), vocabulary=%d (skipped %d), dim=%d",
        elapsed,
        p_done,
        p_skip,
        v_done,
        v_skip,
        EMBEDDING_DIM,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--db",
        default=str(Path.home() / ".lens" / "data" / "lens.db"),
        help="Path to the local LENS sqlite-vec DB (default: ~/.lens/data/lens.db).",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose >= 1 else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    reembed(args.db)


if __name__ == "__main__":
    main()
