"""Quality scoring for paper extraction prioritization.

Combines citation count, venue tier, and recency into a 0-1 score.
"""

from __future__ import annotations

import math
from datetime import date, datetime

RECENCY_HALF_LIFE_DAYS = 2 * 365
# Papers whose date will not parse take a fixed recency penalty of this many
# half-lives (3 -> a recency factor of 1/8). Expressed as an age rather than an
# absolute fallback date, whose implied age — and so whose penalty — would
# deepen every year for an input that has not changed.
UNKNOWN_DATE_HALF_LIVES = 3

DEFAULT_VENUE_TIERS: dict[str, list[str]] = {
    "tier1": ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"],
    "tier2": ["AAAI", "NAACL", "EACL", "COLING"],
}


def quality_score(
    citations: int,
    venue: str | None,
    paper_date: str,
    venue_tiers: dict[str, list[str]] | None = None,
    *,
    today: date | None = None,
) -> float:
    """Compute a 0-1 quality score for a paper.

    Components (weighted):
    - Citation score (0.5): log-scaled, saturates ~10k
    - Venue score (0.2): tier1=1.0, tier2=0.5, unknown/None=0.0
    - Recency score (0.3): exponential decay, half-life ~2 years

    An unparseable ``paper_date`` is scored as
    :data:`UNKNOWN_DATE_HALF_LIVES` half-lives old rather than raising.

    ``today`` overrides the reference date used for the recency term. It
    defaults to :func:`datetime.date.today`; pass it explicitly to make the
    score reproducible (tests must, or their assertions decay with wall time).
    """
    tiers = venue_tiers or DEFAULT_VENUE_TIERS

    citation_score = min(1.0, math.log1p(citations) / math.log1p(10000))

    venue_score = 0.0
    if venue:
        venue_upper = venue.upper()
        if any(venue_upper == v.upper() for v in tiers.get("tier1", [])):
            venue_score = 1.0
        elif any(venue_upper == v.upper() for v in tiers.get("tier2", [])):
            venue_score = 0.5

    try:
        pub_date = datetime.strptime(paper_date[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        days_old = UNKNOWN_DATE_HALF_LIVES * RECENCY_HALF_LIFE_DAYS
    else:
        reference_date = today if today is not None else date.today()
        days_old = max(0, (reference_date - pub_date).days)
    recency_score = math.exp(-0.693 * days_old / RECENCY_HALF_LIFE_DAYS)

    score = 0.5 * citation_score + 0.2 * venue_score + 0.3 * recency_score
    return max(0.0, min(1.0, score))
