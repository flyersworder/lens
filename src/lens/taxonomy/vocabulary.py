"""Canonical vocabulary for guided extraction — seed data and management."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Convert a display name to a URL-safe slug ID."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


SEED_VOCABULARY: list[dict[str, str]] = [
    # Parameters
    {
        "name": "Inference Latency",
        "kind": "parameter",
        "description": "Time required to generate output from input at deployment",
    },
    {
        "name": "Model Accuracy",
        "kind": "parameter",
        "description": "Quality of model predictions on target tasks",
    },
    {
        "name": "Training Cost",
        "kind": "parameter",
        "description": "Compute, time, and financial cost to train or fine-tune",
    },
    {
        "name": "Model Size",
        "kind": "parameter",
        "description": "Number of parameters in the model",
    },
    {
        "name": "Memory Usage",
        "kind": "parameter",
        "description": "RAM and VRAM required during inference or training",
    },
    {
        "name": "Context Length",
        "kind": "parameter",
        "description": "Maximum input sequence length the model can process",
    },
    {
        "name": "Safety/Alignment",
        "kind": "parameter",
        "description": "Degree to which model outputs align with human values and intent",
    },
    {
        "name": "Reasoning Capability",
        "kind": "parameter",
        "description": "Ability to perform multi-step logical or mathematical reasoning",
    },
    {
        "name": "Data Efficiency",
        "kind": "parameter",
        "description": "Amount of training data needed to reach target performance",
    },
    {
        "name": "Generalization",
        "kind": "parameter",
        "description": "Ability to perform well on unseen tasks or domains",
    },
    {
        "name": "Interpretability",
        "kind": "parameter",
        "description": "Degree to which model decisions can be understood by humans",
    },
    {
        "name": "Robustness",
        "kind": "parameter",
        "description": "Resilience to adversarial inputs, noise, and distribution shift",
    },
    # Principles
    {
        "name": "Knowledge Distillation",
        "kind": "principle",
        "description": "Training a smaller model to mimic a larger teacher model",
    },
    {
        "name": "Quantization",
        "kind": "principle",
        "description": "Reducing numerical precision of model weights and activations",
    },
    {
        "name": "Sparse Attention/MoE",
        "kind": "principle",
        "description": "Activating only a subset of parameters or attention heads per input",
    },
    {
        "name": "RAG",
        "kind": "principle",
        "description": "Augmenting generation with retrieved external knowledge",
    },
    {
        "name": "Chain-of-Thought",
        "kind": "principle",
        "description": "Prompting or training models to produce intermediate reasoning steps",
    },
    {
        "name": "Preference Optimization (RLHF/DPO)",
        "kind": "principle",
        "description": "Aligning model outputs to human preferences via reward signals",
    },
    {
        "name": "Parameter-Efficient Fine-Tuning (LoRA/QLoRA)",
        "kind": "principle",
        "description": "Adapting models by training a small number of added parameters",
    },
    {
        "name": "Speculative Decoding",
        "kind": "principle",
        "description": "Using a fast draft model to propose tokens verified by a larger model",
    },
    {
        "name": "Flash Attention",
        "kind": "principle",
        "description": "Memory-efficient attention computation via tiling and recomputation",
    },
    {
        "name": "Positional Encoding Innovation",
        "kind": "principle",
        "description": "Novel methods for representing token position in sequences",
    },
    {
        "name": "Scaling",
        "kind": "principle",
        "description": "Increasing model size, data, or compute to improve performance",
    },
    {
        "name": "Multi-Agent Collaboration",
        "kind": "principle",
        "description": "Multiple LLM agents coordinating to solve complex tasks",
    },
]


def load_seed_vocabulary(store: LensStore) -> int:
    """Load seed vocabulary into the database. Idempotent — skips existing entries.

    Ensures tables are initialized before inserting.
    Returns the number of newly inserted entries.
    """
    store.init_tables()
    existing = store.query("vocabulary")
    existing_ids = {r["id"] for r in existing}
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    new_rows: list[dict[str, Any]] = []
    for entry in SEED_VOCABULARY:
        entry_id = _slugify(entry["name"])
        if entry_id in existing_ids:
            continue
        new_rows.append(
            {
                "id": entry_id,
                "name": entry["name"],
                "kind": entry["kind"],
                "description": entry["description"],
                "source": "seed",
                "first_seen": today,
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        )

    if new_rows:
        store.add_rows("vocabulary", new_rows)
        logger.info("Loaded %d seed vocabulary entries", len(new_rows))

    return len(new_rows)
