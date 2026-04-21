"""Tests for the provenance sidecar builder."""

from __future__ import annotations

import yaml

from lens.serve.provenance import (
    build_analyze_provenance,
    build_explain_provenance,
    write_provenance,
)
from lens.store.models import ExplanationResult


def test_analyze_provenance_tradeoff():
    result = {
        "query": "reduce latency without accuracy loss",
        "improving": "inference-latency",
        "worsening": "model-accuracy",
        "principles": [
            {
                "principle_id": "quantization",
                "name": "Quantization",
                "count": 5,
                "avg_confidence": 0.82,
                "score": 4.1,
                "paper_ids": ["p1", "p2", "p3"],
            },
            {
                "principle_id": "distillation",
                "name": "Distillation",
                "count": 2,
                "avg_confidence": 0.7,
                "score": 1.4,
                "paper_ids": ["p3", "p4"],
            },
        ],
    }

    sidecar = build_analyze_provenance(
        query=result["query"],
        type_=None,
        result=result,
        session_id="abc12345",
        taxonomy_version=7,
    )

    assert sidecar["command"] == "analyze"
    assert sidecar["type"] == "tradeoff"
    assert sidecar["session_id"] == "abc12345"
    assert sidecar["taxonomy_version"] == 7
    assert sidecar["resolved"]["improving"] == "inference-latency"
    assert sidecar["paper_ids"] == ["p1", "p2", "p3", "p4"]
    assert len(sidecar["claims"]) == 2
    assert sidecar["claims"][0]["principle_id"] == "quantization"
    assert sidecar["claims"][0]["evidence_count"] == 5


def test_analyze_provenance_agentic():
    result = {
        "query": "reasoning patterns",
        "category": "Reasoning",
        "patterns": [
            {
                "pattern_name": "ReAct",
                "category": "Reasoning",
                "paper_ids": ["p1"],
            }
        ],
    }

    sidecar = build_analyze_provenance(
        query=result["query"],
        type_="agentic",
        result=result,
        session_id="agent001",
        taxonomy_version=2,
    )

    assert sidecar["type"] == "agentic"
    assert sidecar["resolved"]["category"] == "Reasoning"
    assert sidecar["paper_ids"] == ["p1"]


def test_analyze_provenance_architecture():
    result = {
        "query": "sub-quadratic attention",
        "slot": "attention-mechanism",
        "variants": [
            {
                "variant_name": "flash-attention",
                "slot": "attention-mechanism",
                "properties": "IO-aware",
                "paper_ids": ["p1", "p2"],
            },
            {
                "variant_name": "linear-attention",
                "slot": "attention-mechanism",
                "properties": "O(n)",
                "paper_ids": ["p2", "p3"],
            },
        ],
    }

    sidecar = build_analyze_provenance(
        query=result["query"],
        type_="architecture",
        result=result,
        session_id="sess0001",
        taxonomy_version=3,
    )

    assert sidecar["type"] == "architecture"
    assert sidecar["resolved"]["slot"] == "attention-mechanism"
    assert sidecar["paper_ids"] == ["p1", "p2", "p3"]
    assert len(sidecar["claims"]) == 2


def test_explain_provenance():
    result = ExplanationResult(
        resolved_type="parameter",
        resolved_id="inference-latency",
        resolved_name="Inference Latency",
        narrative="Latency is the time to produce a token...",
        evolution=[],
        tradeoffs=[
            {
                "improving_param_id": "inference-latency",
                "worsening_param_id": "model-accuracy",
                "principle_id": "quantization",
                "count": 3,
                "avg_confidence": 0.8,
                "paper_ids": ["p1", "p2"],
                "taxonomy_version": 1,
            }
        ],
        connections=["Model Accuracy", "Quantization"],
        paper_refs=[],
        alternatives=[
            {
                "resolved_type": "principle",
                "resolved_id": "latency-optimization",
                "resolved_name": "Latency Optimization",
            }
        ],
    )

    sidecar = build_explain_provenance(
        query="latency",
        focus="tradeoffs",
        result=result,
        session_id="expl0001",
        taxonomy_version=4,
    )

    assert sidecar["command"] == "explain"
    assert sidecar["focus"] == "tradeoffs"
    assert sidecar["resolved"]["id"] == "inference-latency"
    assert sidecar["paper_ids"] == ["p1", "p2"]
    # vocab_ids includes the resolved concept plus the three ids referenced in the cell.
    assert set(sidecar["vocab_ids"]) == {
        "inference-latency",
        "model-accuracy",
        "quantization",
    }
    assert sidecar["alternatives"][0]["id"] == "latency-optimization"


def test_write_provenance_round_trip(tmp_path):
    data = {"command": "analyze", "paper_ids": ["p1", "p2"]}
    out = write_provenance(data, tmp_path / "sub" / "out.yaml")

    assert out.exists()
    reloaded = yaml.safe_load(out.read_text())
    assert reloaded == data
