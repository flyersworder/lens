"""Recovery step 2: backfill embeddings + regenerate idea cards on a restored DB.

Companion to ``restore_from_turso.py``. The reverse-copy drops embeddings (the
remote stores them as F32_BLOB; locally they live in sqlite-vec companion
tables), so both ``papers_vec`` and ``vocabulary_vec`` come up empty. This
script:

1. cloud-embeds every paper (title + abstract) into ``papers_vec``
2. backfills every vocabulary embedding (``fix_missing_embeddings``)
3. regenerates idea cards against the recovered matrix (``run_ideation_with_llm``)

The recovered 197-term vocabulary and 299-cell matrix are preserved verbatim —
we do NOT rebuild taxonomy/matrix, only fill in the embeddings they need.

Embedding source note: the pre-loss corpus embeddings are unrecoverable, so all
papers are re-embedded uniformly with the configured cloud model
(text-embedding-3-small, 1536-d) — self-consistent and matches how
``build_vocabulary`` embeds vocab.

Uses the configured OpenRouter key for both embeddings and the LLM (they share
the same provider). Run against the real ``~/.lens/data/lens.db`` after the
reverse-copy has been swapped into place, or against a temp path to validate.

Usage:
    uv run python scripts/enrich_restored_db.py --target /path/to/lens.db
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lens.config import load_config
from lens.knowledge.linter import fix_missing_embeddings
from lens.llm.client import LLMClient
from lens.monitor.ideation import run_ideation_with_llm
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings


def _emb_cfg(config: dict) -> dict:
    """Cloud embedding kwargs, falling back to the LLM key if needed."""
    emb = config.get("embeddings", {})
    key = emb.get("api_key") or config.get("llm", {}).get("api_key", "")
    return {
        "provider": emb.get("provider", "cloud"),
        "model": emb.get("model", "openai/text-embedding-3-small"),
        "api_base": emb.get("api_base"),
        "api_key": key,
        "dimensions": emb.get("dimensions", 1536),
    }


def embed_papers(store: LensStore, cfg: dict) -> int:
    """Cloud-embed every paper's title+abstract into papers_vec. Returns count."""
    papers = store.query("papers")
    if not papers:
        return 0
    texts = [f"{p.get('title', '')}. {p.get('abstract', '')}".strip() for p in papers]
    done = 0
    batch = 100
    for i in range(0, len(papers), batch):
        chunk_p = papers[i : i + batch]
        chunk_t = texts[i : i + batch]
        vecs = embed_strings(
            chunk_t,
            provider=cfg["provider"],
            model_name=cfg["model"],
            dimensions=cfg["dimensions"],
            api_base=cfg["api_base"],
            api_key=cfg["api_key"],
        )
        for p, v in zip(chunk_p, vecs, strict=True):
            store.upsert_embedding("papers", p["paper_id"], v.tolist())
            done += 1
        print(f"  embedded papers {done}/{len(papers)}")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Path to the restored lens.db")
    ap.add_argument("--max-cards", type=int, default=40)
    ap.add_argument("--min-gap-score", type=float, default=0.5)
    ap.add_argument("--dedup-threshold", type=float, default=0.32)
    args = ap.parse_args()

    config = load_config(Path.home() / ".lens" / "config.yaml")
    cfg = _emb_cfg(config)
    if not cfg["api_key"]:
        raise SystemExit("No embeddings/LLM api_key in config; cannot embed or ideate.")

    store = LensStore(args.target)
    store.init_tables()

    print("Step 1: embedding papers (cloud, title+abstract)")
    n_papers = embed_papers(store, cfg)
    print(f"  {n_papers} papers embedded")

    print("Step 2: backfilling vocabulary embeddings")
    fixed = fix_missing_embeddings(
        store,
        embedding_provider=cfg["provider"],
        embedding_model=cfg["model"],
        embedding_api_base=cfg["api_base"],
        embedding_api_key=cfg["api_key"],
    )
    print(f"  {len(fixed)} vocabulary embeddings backfilled")

    print("Step 3: regenerating idea cards against the recovered matrix")
    llm = LLMClient(
        model=config["llm"].get("default_model", "google/gemini-3.1-flash-lite-preview"),
        api_base=config["llm"].get("api_base"),
        api_key=config["llm"].get("api_key") or cfg["api_key"],
    )
    report = asyncio.run(
        run_ideation_with_llm(
            store,
            llm,
            max_cards=args.max_cards,
            min_gap_score=args.min_gap_score,
            dedup_threshold=args.dedup_threshold,
        )
    )
    cards = report.get("idea_cards", [])
    grounded = sum(1 for c in cards if c.get("paper_ids"))
    print(f"  generated {len(cards)} idea cards ({grounded} grounded in >=1 paper)")


if __name__ == "__main__":
    main()
