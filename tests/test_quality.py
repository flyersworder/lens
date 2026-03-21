"""Tests for paper quality scoring."""


def test_quality_score_high_citation_recent():
    from lens.acquire.quality import quality_score
    score = quality_score(citations=5000, venue="NeurIPS", paper_date="2024-06-01")
    assert 0.8 <= score <= 1.0


def test_quality_score_zero_citations_no_venue():
    from lens.acquire.quality import quality_score
    score = quality_score(citations=0, venue=None, paper_date="2024-01-01")
    assert 0.0 <= score <= 0.3


def test_quality_score_old_paper_high_citations():
    from lens.acquire.quality import quality_score
    score = quality_score(citations=50000, venue="NeurIPS", paper_date="2017-06-01")
    assert 0.5 <= score <= 1.0


def test_quality_score_recent_no_citations():
    from lens.acquire.quality import quality_score
    score = quality_score(citations=0, venue=None, paper_date="2026-03-01")
    assert 0.0 <= score <= 0.4


def test_quality_score_tier2_venue():
    from lens.acquire.quality import quality_score
    score_t1 = quality_score(citations=100, venue="NeurIPS", paper_date="2024-01-01")
    score_t2 = quality_score(citations=100, venue="AAAI", paper_date="2024-01-01")
    score_none = quality_score(citations=100, venue=None, paper_date="2024-01-01")
    assert score_t1 > score_t2 > score_none


def test_quality_score_bounds():
    from lens.acquire.quality import quality_score
    score = quality_score(citations=999999, venue="NeurIPS", paper_date="2026-03-01")
    assert 0.0 <= score <= 1.0
    score = quality_score(citations=0, venue=None, paper_date="2015-01-01")
    assert 0.0 <= score <= 1.0


def test_venue_tiers_configurable():
    from lens.acquire.quality import quality_score
    custom_tiers = {"tier1": ["CustomConf"], "tier2": ["OtherConf"]}
    score = quality_score(citations=100, venue="CustomConf", paper_date="2024-01-01", venue_tiers=custom_tiers)
    assert score > quality_score(citations=100, venue=None, paper_date="2024-01-01", venue_tiers=custom_tiers)
