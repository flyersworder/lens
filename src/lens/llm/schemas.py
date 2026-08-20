"""LLM-facing response models for schema-constrained structured output.

These mirror what the model is asked to *produce* — deliberately NOT the
persistence models in ``lens.store.models``. The DB models additionally carry
``paper_id`` (assigned by the caller) and ``verification_status`` (derived by
``compute_verification_status``). Under strict json_schema every declared
property is required, so reusing the DB models here would oblige the model to
invent both fields.

Keeping the two apart means the wire contract, the prompt's schema section and
the validation gate all derive from one definition without dragging caller-owned
state into the request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NewConcept(BaseModel):
    """A novel concept introduced by a paper.

    Modelled as a list of typed pairs rather than an open-ended map: strict
    structured outputs forbid free-form objects (constrained decoding has no
    grammar for arbitrary keys), so the mapping is rebuilt caller-side.
    """

    name: str = Field(
        description=(
            "The concept name exactly as written after the 'NEW: ' prefix in "
            "improves/worsens/technique. Matched verbatim when the vocabulary is "
            "built, so do not slugify or reword it."
        )
    )
    description: str = Field(description="Short description of the concept")


class TradeoffItem(BaseModel):
    """A single engineering tradeoff claimed by a paper."""

    improves: str = Field(description="What the technique improves")
    worsens: str = Field(description="What gets worse as a result")
    technique: str = Field(description="The technique or method used")
    context: str = Field(description="Conditions or constraints mentioned")
    confidence: float = Field(description="Extraction confidence, 0.0-1.0")
    evidence_quote: str = Field(description="Relevant sentence from the paper")
    new_concepts: list[NewConcept] = Field(
        default_factory=list, description="Novel concepts introduced by this paper"
    )


class ArchitectureItem(BaseModel):
    """An architecture component variant introduced by a paper."""

    component_slot: str = Field(description="Architecture component category")
    variant_name: str = Field(description="Specific variant introduced")
    replaces: str | None = Field(
        default=None, description="What it replaces or generalizes; null if novel"
    )
    key_properties: str = Field(description="Key properties or advantages")
    confidence: float = Field(description="Extraction confidence, 0.0-1.0")
    new_concepts: list[NewConcept] = Field(default_factory=list)


class AgenticItem(BaseModel):
    """An agentic pattern described by a paper."""

    pattern_name: str = Field(description="Name of the agent pattern")
    category: str = Field(default="", description="Agentic category")
    structure: str = Field(description="High-level structure description")
    use_case: str = Field(description="Primary use case")
    components: list[str] = Field(description="Components making up the pattern")
    confidence: float = Field(description="Extraction confidence, 0.0-1.0")
    new_concepts: list[NewConcept] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """Top-level envelope returned by the extraction prompt.

    Every array may be empty — an empty array is the correct answer when the
    paper says nothing about that category, and is preferable to fabrication.
    """

    tradeoffs: list[TradeoffItem] = Field(default_factory=list)
    architecture: list[ArchitectureItem] = Field(default_factory=list)
    agentic: list[AgenticItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Novelty verification (knowledge/scoop_check.py)
# ---------------------------------------------------------------------------


class NoveltyVerdict(BaseModel):
    """Prior-art judgement for one idea card.

    ``verdict`` is a closed enum, which strict decoding enforces directly. The
    prompt path previously had to detect an out-of-range verdict after the fact
    and discard the whole response.
    """

    verdict: Literal["novel", "overlaps", "scooped"] = Field(
        description=(
            "scooped = core idea already published; overlaps = substantial "
            "related work but a distinct angle; novel = no close prior art"
        )
    )
    colliding_papers: list[str] = Field(
        default_factory=list, description="Titles of papers that collide with the idea"
    )
    rationale: str = Field(default="", description="One sentence justifying the verdict")


# ---------------------------------------------------------------------------
# Idea cards (monitor/ideation.py)
# ---------------------------------------------------------------------------


class IdeaCardResponse(BaseModel):
    """One pattern-guided research idea generated from a matrix gap."""

    title: str = Field(description="At most 15 words")
    patterns: list[str] = Field(
        default_factory=list,
        description="1-2 ideation pattern names, copied verbatim from the catalog",
    )
    hook: str = Field(default="", description="1-2 sentences")
    mechanism: str = Field(
        default="", description="How the chosen pattern applies, producing a concrete artifact"
    )
    falsification: str = Field(
        default="",
        description="A minimal experiment with a metric and a distinguishing prediction",
    )
    differentiation: list[str] = Field(
        default_factory=list, description="How this differs from the referenced principles"
    )
    signature_terms: list[str] = Field(default_factory=list, description="3-6 key terms")
    confidence: float = Field(default=0.5, description="0-1 float")

    @field_validator("patterns", "differentiation", "signature_terms", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: object) -> list[str]:
        """Tolerate a bare string or a malformed shape for the list fields.

        Enforced decoding cannot emit these, but the prompt fallback can, and
        ideation prefers a card with one empty field over discarding the whole
        card. Mirrors the pre-schema ``_as_str_list`` behaviour.
        """
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v] if v.strip() else []
        return []
