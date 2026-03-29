"""Tests for seed paper loader."""

import pytest
import yaml


def test_load_seed_manifest():
    from lens.acquire.seed import load_seed_manifest

    papers = load_seed_manifest()
    assert len(papers) >= 10
    assert all("arxiv_id" in p for p in papers)
    assert all("title" in p for p in papers)


def test_load_seed_manifest_custom_path(tmp_path):
    from lens.acquire.seed import load_seed_manifest

    manifest = tmp_path / "custom_seeds.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "papers": [
                    {"arxiv_id": "9999.99999", "title": "Test Paper", "category": "test"},
                ]
            }
        )
    )
    papers = load_seed_manifest(manifest)
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "9999.99999"


def test_seed_manifest_has_categories():
    from lens.acquire.seed import load_seed_manifest

    papers = load_seed_manifest()
    categories = {p.get("category") for p in papers}
    assert "foundational" in categories
    assert "agentic" in categories


def test_seed_manifest_has_abstracts():
    """Enriched manifest should include pre-fetched abstracts."""
    from lens.acquire.seed import load_seed_manifest

    papers = load_seed_manifest()
    with_abstracts = [p for p in papers if p.get("abstract")]
    assert len(with_abstracts) >= 40  # most should have abstracts


@pytest.mark.asyncio
async def test_acquire_seed_papers(tmp_path):
    """Test seed acquisition from enriched manifest — no API calls needed."""
    from lens.acquire.seed import acquire_seed
    from lens.store.store import LensStore

    manifest = tmp_path / "seeds.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "papers": [
                    {
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "abstract": "We propose a new architecture based on attention.",
                        "authors": ["Vaswani", "Shazeer"],
                        "date": "2017-06-12",
                        "category": "foundational",
                    },
                    {
                        "arxiv_id": "2305.18290",
                        "title": "Direct Preference Optimization",
                        "abstract": "DPO is a simple approach to RLHF.",
                        "authors": ["Rafailov"],
                        "date": "2023-05-29",
                        "category": "training",
                    },
                ]
            }
        )
    )

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    count = await acquire_seed(store, manifest_path=manifest)
    assert count == 2

    papers = store.query("papers")
    assert len(papers) == 2
    assert papers[0]["abstract"] == "We propose a new architecture based on attention."
    assert papers[0]["authors"] == ["Vaswani", "Shazeer"]


@pytest.mark.asyncio
async def test_acquire_seed_skips_existing(tmp_path):
    """Papers already in the database should be skipped."""
    from lens.acquire.seed import acquire_seed
    from lens.store.store import LensStore

    manifest = tmp_path / "seeds.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "papers": [
                    {
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "abstract": "Test",
                        "authors": [],
                        "date": "2017-06-12",
                        "category": "foundational",
                    },
                ]
            }
        )
    )

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # First run — acquires 1
    count1 = await acquire_seed(store, manifest_path=manifest)
    assert count1 == 1

    # Second run — skips (already stored)
    count2 = await acquire_seed(store, manifest_path=manifest)
    assert count2 == 0
