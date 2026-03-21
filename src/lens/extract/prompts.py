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
      "evidence_quote": "relevant sentence from the paper"
    }
  ],
  "architecture": [
    {
      "component_slot": "architecture component category",
      "variant_name": "specific variant introduced",
      "replaces": "what it replaces or generalizes (null if novel)",
      "key_properties": "key properties or advantages",
      "confidence": 0.9
    }
  ],
  "agentic": [
    {
      "pattern_name": "name of the agent pattern",
      "structure": "high-level structure description",
      "use_case": "primary use case",
      "components": ["list", "of", "components"],
      "confidence": 0.8
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

_TRADEOFFS_SECTION = """\
### 1. Tradeoffs (TradeoffExtraction)
Identify engineering tradeoffs: when improving one aspect worsens another.
- "improves": what the technique/method improves
- "worsens": what gets worse as a consequence
- "technique": the specific technique or method
- "context": conditions, benchmarks, or constraints mentioned
- "confidence": your confidence score (see scale below)
- "evidence_quote": a relevant sentence from the paper"""

_ARCHITECTURE_SECTION = (
    "### 2. Architecture Contributions (ArchitectureExtraction)\n"
    "Identify novel or notable architecture components.\n"
    '- "component_slot": the category (e.g., attention mechanism, positional encoding,'
    " normalization, FFN, activation function, MoE routing)\n"
    '- "variant_name": the specific variant name\n'
    '- "replaces": what it replaces/generalizes (null if entirely novel)\n'
    '- "key_properties": key properties or advantages\n'
    '- "confidence": your confidence score'
)

_AGENTIC_SECTION = """\
### 3. Agentic Patterns (AgenticExtraction)
Identify LLM agent design patterns.
- "pattern_name": name of the pattern
- "structure": high-level description of the agent structure
- "use_case": primary use case or application
- "components": list of key components
- "confidence": your confidence score"""

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
        _TRADEOFFS_SECTION,
        _ARCHITECTURE_SECTION,
        _AGENTIC_SECTION,
        _CONFIDENCE_SECTION,
        response_format,
    ]
    return "\n\n".join(sections)
