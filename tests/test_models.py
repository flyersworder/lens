"""Tests for Pydantic BaseModel schemas."""

from datetime import datetime

from lens.store.models import EMBEDDING_DIM


def test_paper_model_valid():
    from lens.store.models import Paper

    paper = Paper(
        paper_id="2401.12345",
        title="Attention Is All You Need",
        abstract="We propose a new architecture...",
        authors=["Vaswani", "Shazeer"],
        venue="NeurIPS",
        date="2017-06-12",
        arxiv_id="1706.03762",
        citations=100000,
        quality_score=0.95,
        extraction_status="pending",
        embedding=[0.1] * 768,
    )
    assert paper.paper_id == "2401.12345"
    assert paper.extraction_status == "pending"
    assert len(paper.embedding) == 768


def test_paper_model_nullable_venue():
    from lens.store.models import Paper

    paper = Paper(
        paper_id="2401.99999",
        title="Some Preprint",
        abstract="Abstract text",
        authors=["Author"],
        venue=None,
        date="2024-01-01",
        arxiv_id="2401.99999",
        citations=0,
        quality_score=0.1,
        extraction_status="pending",
        embedding=[0.0] * EMBEDDING_DIM,
    )
    assert paper.venue is None


def test_tradeoff_extraction_model():
    from lens.store.models import TradeoffExtraction

    ext = TradeoffExtraction(
        paper_id="2401.12345",
        improves="mathematical reasoning accuracy",
        worsens="inference time per token",
        technique="chain-of-thought with self-consistency",
        context="on GSM8K benchmark",
        confidence=0.85,
        evidence_quote="We observe a 15% improvement...",
    )
    assert ext.confidence == 0.85


def test_architecture_extraction_model():
    from lens.store.models import ArchitectureExtraction

    ext = ArchitectureExtraction(
        paper_id="2401.12345",
        component_slot="attention mechanism",
        variant_name="grouped-query attention",
        replaces="multi-head attention",
        key_properties="reduces KV cache by sharing keys/values",
        confidence=0.9,
    )
    assert ext.replaces == "multi-head attention"


def test_architecture_extraction_replaces_nullable():
    from lens.store.models import ArchitectureExtraction

    ext = ArchitectureExtraction(
        paper_id="2401.12345",
        component_slot="activation function",
        variant_name="SwiGLU",
        replaces=None,
        key_properties="smooth gated activation",
        confidence=0.7,
    )
    assert ext.replaces is None


def test_agentic_extraction_model():
    from lens.store.models import AgenticExtraction

    ext = AgenticExtraction(
        paper_id="2401.12345",
        pattern_name="reflexion",
        structure="single agent with self-critique loop and memory",
        use_case="code generation with iterative debugging",
        components=["actor", "evaluator", "memory"],
        confidence=0.8,
    )
    assert ext.components == ["actor", "evaluator", "memory"]


def test_architecture_slot_model():
    from lens.store.models import ArchitectureSlot

    slot = ArchitectureSlot(
        id=1,
        name="Attention Mechanism",
        description="Core attention computation in transformer blocks",
        taxonomy_version=1,
    )
    assert slot.name == "Attention Mechanism"


def test_architecture_variant_model():
    from lens.store.models import ArchitectureVariant

    variant = ArchitectureVariant(
        id=1,
        slot_id=1,
        name="Grouped-Query Attention",
        replaces=[2, 3],
        properties="reduces KV cache by sharing keys/values across query groups",
        paper_ids=["2305.13245"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert variant.replaces == [2, 3]


def test_agentic_pattern_model():
    from lens.store.models import AgenticPattern

    pattern = AgenticPattern(
        id=1,
        name="Reflexion",
        category="single-agent",
        description="Single agent with self-critique loop and memory",
        components=["actor", "evaluator", "memory"],
        use_cases=["code generation", "reasoning tasks"],
        paper_ids=["2303.11366"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert pattern.category == "single-agent"


def test_matrix_cell_model():
    from lens.store.models import MatrixCell

    cell = MatrixCell(
        improving_param_id="inference-latency",
        worsening_param_id="model-accuracy",
        principle_id="quantization",
        count=5,
        avg_confidence=0.82,
        paper_ids=["2401.12345", "2402.67890"],
        taxonomy_version=1,
    )
    assert cell.count == 5
    assert cell.avg_confidence == 0.82


def test_taxonomy_version_model():
    from lens.store.models import TaxonomyVersion

    tv = TaxonomyVersion(
        version_id=1,
        created_at=datetime(2026, 3, 20, 12, 0, 0),
        paper_count=200,
        param_count=25,
        principle_count=35,
    )
    assert tv.version_id == 1


def test_ideation_gap_model():
    from lens.store.models import IdeationGap

    gap = IdeationGap(
        id=1,
        report_id=1,
        gap_type="sparse_cell",
        description="Tradeoff (accuracy, latency) has only 1 known principle",
        related_params=[1, 2],
        related_principles=[3],
        related_slots=[],
        score=0.8,
        llm_hypothesis=None,
        created_at=datetime(2026, 3, 20),
        taxonomy_version=1,
    )
    assert gap.gap_type == "sparse_cell"
    assert gap.llm_hypothesis is None


def test_ideation_report_model():
    from lens.store.models import IdeationReport

    report = IdeationReport(
        id=1,
        created_at=datetime(2026, 3, 20),
        taxonomy_version=1,
        paper_batch_size=50,
        gap_count=10,
    )
    assert report.gap_count == 10


def test_taxonomy_version_with_catalog_counts():
    from datetime import datetime

    from lens.store.models import TaxonomyVersion

    tv = TaxonomyVersion(
        version_id=1,
        created_at=datetime.now(),
        paper_count=10,
        param_count=5,
        principle_count=10,
        slot_count=3,
        variant_count=12,
        pattern_count=8,
    )
    assert tv.slot_count == 3
    assert tv.variant_count == 12
    assert tv.pattern_count == 8


def test_taxonomy_version_defaults_backward_compat():
    from datetime import datetime

    from lens.store.models import TaxonomyVersion

    tv = TaxonomyVersion(
        version_id=1,
        created_at=datetime.now(),
        paper_count=10,
        param_count=5,
        principle_count=10,
    )
    assert tv.slot_count == 0
    assert tv.variant_count == 0
    assert tv.pattern_count == 0


def test_explanation_result_model():
    from lens.store.models import ExplanationResult

    result = ExplanationResult(
        resolved_type="parameter",
        resolved_id="inference-latency",
        resolved_name="Inference Latency",
        narrative="Inference latency is...",
        evolution=["v1", "v2"],
        tradeoffs=[{"improving": 1, "worsening": 2}],
        connections=["Model Size"],
        paper_refs=["2401.12345"],
        alternatives=[{"type": "parameter", "id": 2, "name": "Accuracy"}],
    )
    assert result.resolved_type == "parameter"
