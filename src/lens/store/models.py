"""Pydantic model schemas for all LENS data types."""

import re
from datetime import datetime

from pydantic import BaseModel, field_validator

EMBEDDING_DIM = 768
"""Default embedding vector dimension. All Vector fields use this value.

To change the dimension, set this before importing any model classes
or calling init_tables(). Changing it after tables are created requires
reinitializing the database (``lens init --force``).
"""

VALID_EXTRACTION_STATUSES = {"pending", "complete", "incomplete", "failed"}

# ---------------------------------------------------------------------------
# Layer 0 — Raw ingested papers
# ---------------------------------------------------------------------------


class Paper(BaseModel):
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
    keywords: list[str] = []
    github_url: str | None = None
    embedding: list[float] = []

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


class TradeoffExtraction(BaseModel):
    """A single tradeoff extracted from a paper."""

    paper_id: str
    improves: str
    worsens: str
    technique: str
    context: str
    confidence: float
    evidence_quote: str
    new_concepts: dict[str, str] = {}


class ArchitectureExtraction(BaseModel):
    """An architecture component extracted from a paper."""

    paper_id: str
    component_slot: str
    variant_name: str
    replaces: str | None = None
    key_properties: str
    confidence: float
    new_concepts: dict[str, str] = {}


class AgenticExtraction(BaseModel):
    """An agentic pattern extracted from a paper."""

    paper_id: str
    pattern_name: str
    category: str = ""
    structure: str
    use_case: str
    components: list[str]
    confidence: float
    new_concepts: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Layer 2 — Taxonomy entities
# ---------------------------------------------------------------------------


class VocabularyEntry(BaseModel):
    """A canonical parameter or principle in the vocabulary."""

    id: str
    name: str
    kind: str  # "parameter", "principle", "arch_slot", or "agentic_category"
    description: str
    source: str  # "seed" or "extracted"
    first_seen: str
    paper_count: int = 0
    avg_confidence: float = 0.0
    embedding: list[float] = []

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        valid = ("parameter", "principle", "arch_slot", "agentic_category")
        if v not in valid:
            raise ValueError(f"kind must be one of {valid}, got '{v}'")
        return v

    @field_validator("source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        if v not in ("seed", "extracted"):
            raise ValueError(f"source must be 'seed' or 'extracted', got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Layer 3 — Aggregated matrix
# ---------------------------------------------------------------------------


class MatrixCell(BaseModel):
    """One cell in the tradeoff matrix (improving × worsening × principle)."""

    improving_param_id: str
    worsening_param_id: str
    principle_id: str
    count: int
    avg_confidence: float
    paper_ids: list[str]
    taxonomy_version: int


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


class TaxonomyVersion(BaseModel):
    """Metadata for a taxonomy snapshot."""

    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int
    slot_count: int = 0
    variant_count: int = 0
    pattern_count: int = 0


# ---------------------------------------------------------------------------
# Ideation
# ---------------------------------------------------------------------------


class IdeationGap(BaseModel):
    """A gap or opportunity identified during ideation analysis."""

    id: int
    report_id: int
    gap_type: str
    description: str
    related_params: list[str]
    related_principles: list[str]
    related_slots: list[int]
    score: float
    llm_hypothesis: str | None = None
    created_at: datetime
    taxonomy_version: int


class IdeationReport(BaseModel):
    """Summary of one ideation run."""

    id: int
    created_at: datetime
    taxonomy_version: int
    paper_batch_size: int
    gap_count: int


# ---------------------------------------------------------------------------
# Query response (not stored in DB)
# ---------------------------------------------------------------------------


class ExplanationResult(BaseModel):
    """Result returned by the /explain query endpoint."""

    resolved_type: str
    resolved_id: str
    resolved_name: str
    narrative: str
    evolution: list[str]
    tradeoffs: list[dict]
    connections: list[str]
    paper_refs: list[str]
    alternatives: list[dict]


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class EventLog(BaseModel):
    """A single event in the LENS audit log."""

    id: int | None = None
    timestamp: str
    kind: str  # ingest | extract | build | lint | fix
    action: str  # e.g. paper.added, orphan.found
    target_type: str | None = None  # paper | vocabulary | extraction | matrix
    target_id: str | None = None
    detail: dict | None = None
    session_id: str | None = None


class LintReport(BaseModel):
    """Summary of a lint run."""

    orphans: list[dict] = []
    contradictions: list[dict] = []
    weak_evidence: list[dict] = []
    missing_embeddings: list[dict] = []
    stale_extractions: list[dict] = []
    near_duplicates: list[dict] = []
    fixes_applied: list[dict] = []
