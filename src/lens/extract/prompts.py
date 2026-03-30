"""Extraction prompt template for LLM-based paper analysis.

Adapted from Trapp & Warschat (2024) single-prompt contradiction extraction.
A single prompt per paper extracts all three tuple types.
"""

from __future__ import annotations

EXTRACTION_RESPONSE_SCHEMA = """{
  "tradeoffs": [
    {
      "improves": "what the technique improves",
      "worsens": "what gets worse as a result",
      "technique": "the technique or method used",
      "context": "conditions or constraints mentioned",
      "confidence": 0.85,
      "evidence_quote": "relevant sentence from the paper",
      "new_concepts": {}
    }
  ],
  "architecture": [
    {
      "component_slot": "architecture component category",
      "variant_name": "specific variant introduced",
      "replaces": "what it replaces or generalizes (null if novel)",
      "key_properties": "key properties or advantages",
      "confidence": 0.9,
      "new_concepts": {}
    }
  ],
  "agentic": [
    {
      "pattern_name": "name of the agent pattern",
      "category": "agentic category",
      "structure": "high-level structure description",
      "use_case": "primary use case",
      "components": ["list", "of", "components"],
      "confidence": 0.8,
      "new_concepts": {}
    }
  ]
}"""

_TASK_SECTION = (
    "## Task\n"
    "Extract three types of structured information from this paper. Return a JSON object"
    ' with three arrays: "tradeoffs", "architecture", and "agentic". Each array may be'
    " empty — return empty arrays rather than fabricating extractions when the paper does"
    " not contain relevant information for that category."
)


def _build_tradeoffs_section(
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the tradeoffs section, optionally with guided vocabulary."""
    base = (
        "### 1. Tradeoffs (TradeoffExtraction)\n"
        "Identify engineering tradeoffs: when improving one aspect worsens another.\n"
    )

    if vocabulary:
        params = [v["name"] for v in vocabulary if v["kind"] == "parameter"]
        principles = [v["name"] for v in vocabulary if v["kind"] == "principle"]
        base += (
            "\nUse EXACT names from the vocabulary below for improves, worsens, "
            "and technique fields.\n"
            "\nParameters:\n"
            + "\n".join(f"- {p}" for p in params)
            + "\n\nPrinciples:\n"
            + "\n".join(f"- {p}" for p in principles)
            + "\n\nIf a concept genuinely does not match any entry above, prefix it "
            'with NEW: (e.g., "NEW: Energy Efficiency") and add an entry to '
            "new_concepts mapping the concept name to a one-line definition.\n"
        )

    base += (
        '\n- "improves": what the technique/method improves (use a Parameter name)\n'
        '- "worsens": what gets worse as a consequence (use a Parameter name)\n'
        '- "technique": the specific technique or method (use a Principle name)\n'
        '- "context": conditions, benchmarks, or constraints mentioned\n'
        '- "confidence": your confidence score (see scale below)\n'
        '- "evidence_quote": a relevant sentence from the paper\n'
        '- "new_concepts": dict mapping each NEW: concept name (without the '
        '"NEW: " prefix) to a one-line definition, e.g. '
        '{"Energy Efficiency": "Power consumption relative to throughput"}. '
        "Empty {} if no NEW: concepts used"
    )
    return base


def _build_architecture_section(vocabulary=None):
    """Build the architecture section, optionally with guided vocabulary."""
    base = (
        "### 2. Architecture Contributions (ArchitectureExtraction)\n"
        "Identify novel or notable architecture components.\n"
    )
    if vocabulary:
        slots = [v["name"] for v in vocabulary if v["kind"] == "arch_slot"]
        if slots:
            base += (
                "\nUse EXACT names from the Architecture Slots below"
                " for component_slot.\n\nArchitecture Slots:\n"
            )
            base += "\n".join(f"- {s}" for s in slots)
            base += (
                "\n\nIf a slot genuinely does not match any entry above,"
                " prefix with NEW: and add an entry to new_concepts.\n"
            )
    base += (
        '\n- "component_slot": the category (use an Architecture Slot name)\n'
        '- "variant_name": the specific variant name (free text)\n'
        '- "replaces": what it replaces/generalizes (null if entirely novel)\n'
        '- "key_properties": key properties or advantages\n'
        '- "confidence": your confidence score\n'
        '- "new_concepts": dict mapping each NEW: concept name to a one-line '
        "definition. Empty {} if no NEW: concepts used"
    )
    return base


def _build_agentic_section(vocabulary=None):
    """Build the agentic section, optionally with guided vocabulary."""
    base = "### 3. Agentic Patterns (AgenticExtraction)\nIdentify LLM agent design patterns.\n"
    if vocabulary:
        categories = [v["name"] for v in vocabulary if v["kind"] == "agentic_category"]
        if categories:
            base += (
                "\nUse EXACT names from the Agentic Categories below"
                " for category.\n\nAgentic Categories:\n"
            )
            base += "\n".join(f"- {c}" for c in categories)
            base += (
                "\n\nIf a category genuinely does not match any entry above,"
                " prefix with NEW: and add an entry to new_concepts.\n"
            )
    base += (
        '\n- "pattern_name": name of the pattern (free text)\n'
        '- "category": the category (use an Agentic Category name)\n'
        '- "structure": high-level description of the agent structure\n'
        '- "use_case": primary use case or application\n'
        '- "components": list of key components\n'
        '- "confidence": your confidence score\n'
        '- "new_concepts": dict mapping each NEW: concept name to a one-line '
        "definition. Empty {} if no NEW: concepts used"
    )
    return base


_CONFIDENCE_SECTION = """\
## Confidence Scale
- 0.9-1.0: Explicitly stated in the paper text
- 0.7-0.9: Strongly implied by the results or methodology
- 0.5-0.7: Inferred from context but not directly stated
- Below 0.5: Speculative — include only if potentially valuable"""


def build_extraction_prompt(
    title: str,
    abstract: str,
    full_text: str | None = None,
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the extraction prompt for a single paper."""
    paper_content = f"Title: {title}\n\nAbstract: {abstract}"
    if full_text:
        paper_content += f"\n\nFull text:\n{full_text}"

    intro = (
        "You are an expert in LLM research. Analyze the following paper and extract"
        " structured information."
    )
    response_format = (
        "## Response Format\n"
        "Return ONLY valid JSON matching this schema:\n"
        f"{EXTRACTION_RESPONSE_SCHEMA}\n\n"
        "Do not include any text outside the JSON object."
    )

    sections = [
        intro,
        f"## Paper\n{paper_content}",
        _TASK_SECTION,
        _build_tradeoffs_section(vocabulary),
        _build_architecture_section(vocabulary),
        _build_agentic_section(vocabulary),
        _CONFIDENCE_SECTION,
        response_format,
    ]
    return "\n\n".join(sections)
