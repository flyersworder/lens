"""Tests for paper quality scoring.

``quality_score`` decays with a paper's age, so every assertion here pins an
explicit ``today`` and expresses paper dates as offsets from it. Hardcoding
absolute dates against a real clock makes these tests fail with the passage of
time rather than with a change in behaviour.
"""

from datetime import date, timedelta

import pytest

TODAY = date(2026, 1, 1)


def _days_before(days: int) -> str:
    return (TODAY - timedelta(days=days)).isoformat()


def _years_before(years: float) -> str:
    return _days_before(round(years * 365))


def test_quality_score_high_citation_recent():
    from lens.acquire.quality import quality_score

    score = quality_score(
        citations=5000, venue="NeurIPS", paper_date=_years_before(0.5), today=TODAY
    )
    assert 0.8 <= score <= 1.0


def test_quality_score_zero_citations_no_venue():
    from lens.acquire.quality import quality_score

    score = quality_score(citations=0, venue=None, paper_date=_years_before(2), today=TODAY)
    assert 0.0 <= score <= 0.3


def test_quality_score_old_paper_high_citations():
    from lens.acquire.quality import quality_score

    score = quality_score(
        citations=50000, venue="NeurIPS", paper_date=_years_before(8), today=TODAY
    )
    assert 0.5 <= score <= 1.0


def test_quality_score_recent_no_citations():
    from lens.acquire.quality import quality_score

    score = quality_score(citations=0, venue=None, paper_date=_days_before(0), today=TODAY)
    assert 0.0 <= score <= 0.4


def test_quality_score_tier2_venue():
    from lens.acquire.quality import quality_score

    published = _years_before(1)
    score_t1 = quality_score(citations=100, venue="NeurIPS", paper_date=published, today=TODAY)
    score_t2 = quality_score(citations=100, venue="AAAI", paper_date=published, today=TODAY)
    score_none = quality_score(citations=100, venue=None, paper_date=published, today=TODAY)
    assert score_t1 > score_t2 > score_none


def test_quality_score_bounds():
    from lens.acquire.quality import quality_score

    score = quality_score(
        citations=999999, venue="NeurIPS", paper_date=_days_before(0), today=TODAY
    )
    assert 0.0 <= score <= 1.0
    score = quality_score(citations=0, venue=None, paper_date=_years_before(11), today=TODAY)
    assert 0.0 <= score <= 1.0


def test_venue_tiers_configurable():
    from lens.acquire.quality import quality_score

    custom_tiers = {"tier1": ["CustomConf"], "tier2": ["OtherConf"]}
    published = _years_before(2)
    score = quality_score(
        citations=100,
        venue="CustomConf",
        paper_date=published,
        venue_tiers=custom_tiers,
        today=TODAY,
    )
    score_none = quality_score(
        citations=100,
        venue=None,
        paper_date=published,
        venue_tiers=custom_tiers,
        today=TODAY,
    )
    assert score > score_none


def test_quality_score_is_stable_across_reference_dates():
    """A paper of a given age scores the same whenever it is evaluated.

    Regression guard for time-dependent tests: the score must be a function of
    the paper's *age*, not of the absolute wall-clock date.
    """
    from lens.acquire.quality import quality_score

    def score_at(reference: date) -> float:
        return quality_score(
            citations=5000,
            venue="NeurIPS",
            paper_date=(reference - timedelta(days=180)).isoformat(),
            today=reference,
        )

    assert score_at(date(2026, 1, 1)) == score_at(date(2040, 6, 15))


def test_quality_score_defaults_to_today():
    """Omitting ``today`` uses the module's own clock."""
    from lens.acquire import quality

    now = quality.date.today()
    published = (now - timedelta(days=365)).isoformat()
    assert quality.quality_score(
        citations=100, venue="ICML", paper_date=published
    ) == quality.quality_score(citations=100, venue="ICML", paper_date=published, today=now)


def test_quality_score_future_paper_does_not_exceed_full_recency():
    """A publication date ahead of the reference clamps to zero age."""
    from lens.acquire.quality import quality_score

    future = quality_score(citations=0, venue=None, paper_date=_days_before(-90), today=TODAY)
    now = quality_score(citations=0, venue=None, paper_date=_days_before(0), today=TODAY)
    assert future == now


def test_quality_score_unparseable_date_takes_fixed_penalty():
    """An unparseable date scores as a fixed age, not a fixed calendar date."""
    from lens.acquire.quality import (
        UNKNOWN_DATE_HALF_LIVES,
        quality_score,
    )

    score = quality_score(citations=0, venue=None, paper_date="not-a-date", today=TODAY)
    # Recency is the only non-zero term: 0.3 * 2**-UNKNOWN_DATE_HALF_LIVES.
    assert score == pytest.approx(0.3 * 2.0**-UNKNOWN_DATE_HALF_LIVES, rel=1e-3)


def test_quality_score_unparseable_date_does_not_decay_over_time():
    """Regression guard: the unknown-date penalty must not deepen with the years.

    An absolute fallback publication date would make an unchanged input score
    lower every year.
    """
    from lens.acquire.quality import quality_score

    def score_at(reference: date) -> float:
        return quality_score(citations=10, venue="ICML", paper_date="", today=reference)

    assert score_at(date(2026, 1, 1)) == score_at(date(2099, 12, 31))


def test_quality_score_unparseable_date_ranks_below_a_known_recent_paper():
    """The penalty still has to bite: unknown beats nothing, loses to recent."""
    from lens.acquire.quality import quality_score

    unknown = quality_score(citations=100, venue="ICML", paper_date="???", today=TODAY)
    recent = quality_score(citations=100, venue="ICML", paper_date=_years_before(0.5), today=TODAY)
    ancient = quality_score(citations=100, venue="ICML", paper_date=_years_before(30), today=TODAY)
    assert ancient < unknown < recent
