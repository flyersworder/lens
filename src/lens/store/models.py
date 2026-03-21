"""Pydantic LanceModel schemas for all LENS data types."""

import re
from datetime import datetime

from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, field_validator

VALID_EXTRACTION_STATUSES = {"pending", "complete", "incomplete", "failed"}

# ---------------------------------------------------------------------------
# Layer 0 — Raw ingested papers
# ---------------------------------------------------------------------------


class Paper(LanceModel):
    """A research paper ingested into LENS."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    venue: str | None = None
    date: str
    arxiv_id: str
    citations: int = 0
    quality_score: float = 0.0
    extraction_status: str = "pending"
    embedding: Vector(768)  # type: ignore[valid-type]

    @field_validator("date")
    @classmethod
    def _check_date_format(cls, v: str) -> str:
        if v and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"date must be YYYY-MM-DD format, got '{v}'")
        return v

    @field_validator("extraction_status")
    @classmethod
    def _check_extraction_status(cls, v: str) -> str:
        if v not in VALID_EXTRACTION_STATUSES:
            raise ValueError(
                f"extraction_status must be one of {VALID_EXTRACTION_STATUSES}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Layer 1 — Raw extractions from papers
# ---------------------------------------------------------------------------


class TradeoffExtraction(LanceModel):
    """A single tradeoff extracted from a paper."""

    paper_id: str
    improves: str
    worsens: str
    technique: str
    context: str
    confidence: float
    evidence_quote: str


class ArchitectureExtraction(LanceModel):
    """An architecture component extracted from a paper."""

    paper_id: str
    component_slot: str
    variant_name: str
    replaces: str | None = None
    key_properties: str
    confidence: float


class AgenticExtraction(LanceModel):
    """An agentic pattern extracted from a paper."""

    paper_id: str
    pattern_name: str
    structure: str
    use_case: str
    components: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Layer 2 — Taxonomy entities
# ---------------------------------------------------------------------------


class Parameter(LanceModel):
    """A canonicalised performance/cost parameter in the taxonomy."""

    id: int
    name: str
    description: str
    raw_strings: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class Principle(LanceModel):
    """A design principle / technique in the taxonomy."""

    id: int
    name: str
    description: str
    sub_techniques: list[str]
    raw_strings: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class ArchitectureSlot(LanceModel):
    """A named slot in the transformer architecture taxonomy."""

    id: int
    name: str
    description: str
    taxonomy_version: int


class ArchitectureVariant(LanceModel):
    """A concrete variant that fills an ArchitectureSlot."""

    id: int
    slot_id: int
    name: str
    replaces: list[int]
    properties: str
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class AgenticPattern(LanceModel):
    """A named agentic design pattern in the taxonomy."""

    id: int
    name: str
    category: str
    description: str
    components: list[str]
    use_cases: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Layer 3 — Aggregated matrix
# ---------------------------------------------------------------------------


class MatrixCell(LanceModel):
    """One cell in the tradeoff matrix (improving × worsening × principle)."""

    improving_param_id: int
    worsening_param_id: int
    principle_id: int
    count: int
    avg_confidence: float
    paper_ids: list[str]
    taxonomy_version: int


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


class TaxonomyVersion(LanceModel):
    """Metadata for a taxonomy snapshot."""

    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int


# ---------------------------------------------------------------------------
# Ideation
# ---------------------------------------------------------------------------


class IdeationGap(LanceModel):
    """A gap or opportunity identified during ideation analysis."""

    id: int
    report_id: int
    gap_type: str
    description: str
    related_params: list[int]
    related_principles: list[int]
    related_slots: list[int]
    score: float
    llm_hypothesis: str | None = None
    created_at: datetime
    taxonomy_version: int


class IdeationReport(LanceModel):
    """Summary of one ideation run."""

    id: int
    created_at: datetime
    taxonomy_version: int
    paper_batch_size: int
    gap_count: int


# ---------------------------------------------------------------------------
# Query response (not stored in LanceDB)
# ---------------------------------------------------------------------------


class ExplanationResult(BaseModel):
    """Result returned by the /explain query endpoint."""

    resolved_type: str
    resolved_id: int
    resolved_name: str
    narrative: str
    evolution: list[str]
    tradeoffs: list[dict]
    connections: list[str]
    paper_refs: list[str]
    alternatives: list[dict]
