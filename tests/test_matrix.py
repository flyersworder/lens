"""Tests for contradiction matrix construction."""

import pytest

from lens.store.models import EMBEDDING_DIM


def test_build_matrix(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "Latency",
                "description": "Inference speed",
                "raw_strings": ["latency", "speed"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 2,
                "name": "Accuracy",
                "description": "Model accuracy",
                "raw_strings": ["accuracy", "performance"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )

    store.add_rows(
        "principles",
        [
            {
                "id": 1,
                "name": "Quantization",
                "description": "Reduce precision",
                "sub_techniques": ["int8", "int4"],
                "raw_strings": ["quantization"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "latency",
                "worsens": "accuracy",
                "technique": "quantization",
                "context": "",
                "confidence": 0.9,
                "evidence_quote": "quote",
            },
            {
                "paper_id": "p2",
                "improves": "speed",
                "worsens": "performance",
                "technique": "quantization",
                "context": "",
                "confidence": 0.8,
                "evidence_quote": "quote2",
            },
        ],
    )

    build_matrix(store, taxonomy_version=1)

    cells = store.query("matrix_cells")
    assert len(cells) >= 1
    cell = [
        c
        for c in cells
        if c["improving_param_id"] == 1 and c["worsening_param_id"] == 2 and c["principle_id"] == 1
    ]
    assert len(cell) == 1
    assert cell[0]["count"] == 2
    assert cell[0]["avg_confidence"] == pytest.approx(0.85)


def test_build_matrix_filters_low_confidence(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "A",
                "description": "A",
                "raw_strings": ["a"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 2,
                "name": "B",
                "description": "B",
                "raw_strings": ["b"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )
    store.add_rows(
        "principles",
        [
            {
                "id": 1,
                "name": "T",
                "description": "T",
                "sub_techniques": [],
                "raw_strings": ["t"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ],
    )
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "a",
                "worsens": "b",
                "technique": "t",
                "context": "",
                "confidence": 0.3,
                "evidence_quote": "q",
            }
        ],
    )

    build_matrix(store, taxonomy_version=1)
    cells = store.query("matrix_cells")
    assert len(cells) == 0  # confidence 0.3 < 0.5 threshold


def test_build_matrix_empty(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    build_matrix(store, taxonomy_version=1)
    cells = store.query("matrix_cells")
    assert len(cells) == 0


def test_get_ranked_matrix(tmp_path):
    from lens.knowledge.matrix import build_matrix, get_ranked_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "A",
                "description": "",
                "raw_strings": ["a"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 2,
                "name": "B",
                "description": "",
                "raw_strings": ["b"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )
    store.add_rows(
        "principles",
        [
            {
                "id": i,
                "name": f"P{i}",
                "description": "",
                "sub_techniques": [],
                "raw_strings": [f"p{i}"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            }
            for i in range(1, 7)  # 6 principles
        ],
    )

    # Create extractions: principle 1 has highest count
    extractions = []
    for i, (tech, count) in enumerate(
        [("p1", 10), ("p2", 8), ("p3", 6), ("p4", 4), ("p5", 2), ("p6", 1)]
    ):
        for j in range(count):
            extractions.append(
                {
                    "paper_id": f"paper_{i}_{j}",
                    "improves": "a",
                    "worsens": "b",
                    "technique": tech,
                    "context": "",
                    "confidence": 0.9,
                    "evidence_quote": "q",
                }
            )
    store.add_rows("tradeoff_extractions", extractions)

    build_matrix(store, taxonomy_version=1)

    ranked = get_ranked_matrix(store, taxonomy_version=1, top_k=4)
    # Should return top 4 principles only
    assert len(ranked) == 4
