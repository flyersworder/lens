"""Canonical vocabulary for guided extraction — seed data and management."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from lens.knowledge.events import log_event
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings

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
    # Architecture Slots
    {
        "name": "Attention Mechanism",
        "kind": "arch_slot",
        "description": "How the model attends to different parts of the input",
    },
    {
        "name": "Positional Encoding",
        "kind": "arch_slot",
        "description": "Methods for representing token position in sequences",
    },
    {
        "name": "FFN",
        "kind": "arch_slot",
        "description": "Feed-forward network layers within transformer blocks",
    },
    {
        "name": "Normalization",
        "kind": "arch_slot",
        "description": "Layer or batch normalization techniques",
    },
    {
        "name": "Activation Function",
        "kind": "arch_slot",
        "description": "Non-linear activation functions in the network",
    },
    {
        "name": "MoE Routing",
        "kind": "arch_slot",
        "description": "Routing strategies for mixture-of-experts architectures",
    },
    {
        "name": "Optimizer",
        "kind": "arch_slot",
        "description": "Training optimization algorithms and strategies",
    },
    {
        "name": "Loss Function",
        "kind": "arch_slot",
        "description": "Objective functions used during training",
    },
    {
        "name": "Quantization Method",
        "kind": "arch_slot",
        "description": "Techniques for reducing numerical precision",
    },
    {
        "name": "Retrieval Mechanism",
        "kind": "arch_slot",
        "description": "Methods for retrieving external knowledge",
    },
    # Agentic Categories
    {
        "name": "Reasoning",
        "kind": "agentic_category",
        "description": "Patterns for multi-step logical inference and problem solving",
    },
    {
        "name": "Planning",
        "kind": "agentic_category",
        "description": "Patterns for decomposing goals into executable steps",
    },
    {
        "name": "Tool Use",
        "kind": "agentic_category",
        "description": "Patterns for LLM interaction with external tools and APIs",
    },
    {
        "name": "Multi-Agent Systems",
        "kind": "agentic_category",
        "description": "Patterns involving multiple coordinating agents",
    },
    {
        "name": "Self-Reflection",
        "kind": "agentic_category",
        "description": "Patterns for self-evaluation and iterative improvement",
    },
    {
        "name": "Code Generation",
        "kind": "agentic_category",
        "description": "Patterns for generating, testing, and debugging code",
    },
    # Ideation Patterns (reusable moves for generating research ideas;
    # induced by ResearchStudio idea-spark from ~1,900 ICLR/ICML/NeurIPS papers)
    {
        "name": "Audit and Pivot an Assumption",
        "kind": "ideation_pattern",
        "description": (
            "Identify a load-bearing assumption in prior work, show it fails in some "
            "regime, and redesign around its negation."
        ),
    },
    {
        "name": "Substitute the Operator or Representation",
        "kind": "ideation_pattern",
        "description": (
            "Replace a core operator (attention, convolution, tokenizer) or data "
            "representation with a better-suited alternative while keeping the "
            "surrounding system fixed."
        ),
    },
    {
        "name": "Liberate a Fixed Generative Component",
        "kind": "ideation_pattern",
        "description": (
            "Take a component hard-coded or held fixed in prior work and make it "
            "learned, generative, or adaptive."
        ),
    },
    {
        "name": "Design a Confound-Isolating Diagnostic",
        "kind": "ideation_pattern",
        "description": (
            "Construct a controlled experiment or benchmark that isolates one factor to "
            "explain an otherwise-confounded phenomenon."
        ),
    },
    {
        "name": "Unify Heterogeneous Inputs into One Space",
        "kind": "ideation_pattern",
        "description": (
            "Map disparate modalities, tasks, or inputs into a single shared "
            "representation so one method handles all of them."
        ),
    },
    {
        "name": "Reframe as a Solvable Object",
        "kind": "ideation_pattern",
        "description": (
            "Recast an ill-posed or intractable problem as an instance of a "
            "well-studied, solvable formulation."
        ),
    },
    {
        "name": "Manufacture the Supervisory Signal",
        "kind": "ideation_pattern",
        "description": (
            "Invent a self-supervised or synthetic training signal where labels are "
            "unavailable or costly."
        ),
    },
    {
        "name": "Encode Structure by Construction",
        "kind": "ideation_pattern",
        "description": (
            "Bake a known invariance or prior (symmetry, sparsity, hierarchy) directly "
            "into the architecture instead of learning it."
        ),
    },
    {
        "name": "Prove Equivalence to Unify",
        "kind": "ideation_pattern",
        "description": (
            "Show two apparently distinct methods or objectives are mathematically "
            "equivalent, then exploit the unification."
        ),
    },
    {
        "name": "Decompose for Differentiated Treatment",
        "kind": "ideation_pattern",
        "description": (
            "Split a heterogeneous problem into sub-parts and apply a specialized method to each."
        ),
    },
    {
        "name": "Decompose and Delegate to Solvers",
        "kind": "ideation_pattern",
        "description": (
            "Break a task into sub-tasks routed to specialized existing solvers, tools, or agents."
        ),
    },
    {
        "name": "Relax Discrete Search to Continuous",
        "kind": "ideation_pattern",
        "description": (
            "Replace a discrete or combinatorial search with a differentiable continuous "
            "relaxation to enable gradient optimization."
        ),
    },
    {
        "name": "Adapt by Conditioning, Not Retraining",
        "kind": "ideation_pattern",
        "description": (
            "Achieve new behavior by conditioning inputs, prompts, or adapters instead "
            "of retraining weights."
        ),
    },
    {
        "name": "Characterize a Limit, Then Surpass It",
        "kind": "ideation_pattern",
        "description": (
            "Formally characterize a fundamental limit or bound of current methods, "
            "then design a method that provably exceeds it."
        ),
    },
    {
        "name": "Design a Property-Targeting Pretext Objective",
        "kind": "ideation_pattern",
        "description": (
            "Craft a pretraining objective specifically engineered to instill a "
            "targeted downstream property."
        ),
    },
]


def _collect_concept_ref(
    ext: dict[str, Any],
    field: str,
    kind: str,
    references: dict[str, list[tuple[str, float, str]]],
    new_concepts: dict[str, dict[str, str]],
) -> None:
    """Extract a concept reference from an extraction row."""
    raw_value = ext.get(field, "")
    if not raw_value:
        return
    if raw_value.startswith("NEW: "):
        name = raw_value[5:].strip()
        if name not in new_concepts:
            # Look up description from the new_concepts dict on the extraction
            concepts_dict = ext.get("new_concepts") or {}
            desc = concepts_dict.get(name, f"Extracted concept: {name}")
            new_concepts[name] = {"kind": kind, "description": desc}
    else:
        name = raw_value
    references.setdefault(name, []).append((ext["paper_id"], ext["confidence"], kind))


def process_new_concepts(store: LensStore, session_id: str | None = None) -> dict[str, int]:
    """Scan all extraction tables for NEW: concepts, accept them, update stats."""
    existing = store.query("vocabulary")
    existing_by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in existing}
    existing_ids: set[str] = {r["id"] for r in existing}
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    references: dict[str, list[tuple[str, float, str]]] = {}
    new_concepts: dict[str, dict[str, str]] = {}

    # Scan tradeoff extractions
    for ext in store.query("tradeoff_extractions"):
        for field, kind in [
            ("improves", "parameter"),
            ("worsens", "parameter"),
            ("technique", "principle"),
        ]:
            _collect_concept_ref(ext, field, kind, references, new_concepts)

    # Scan architecture extractions
    for ext in store.query("architecture_extractions"):
        _collect_concept_ref(ext, "component_slot", "arch_slot", references, new_concepts)

    # Scan agentic extractions
    for ext in store.query("agentic_extractions"):
        _collect_concept_ref(ext, "category", "agentic_category", references, new_concepts)

    # Insert new concepts
    new_rows: list[dict[str, Any]] = []
    for name, info in new_concepts.items():
        entry_id = _slugify(name)
        if entry_id in existing_ids:
            continue
        new_rows.append(
            {
                "id": entry_id,
                "name": name,
                "kind": info["kind"],
                "description": info["description"],
                "source": "extracted",
                "first_seen": today,
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        )
        existing_ids.add(entry_id)
        existing_by_name[name] = new_rows[-1]

    if new_rows:
        store.add_rows("vocabulary", new_rows)
        logger.info("Accepted %d new vocabulary entries", len(new_rows))
        for row in new_rows:
            log_event(
                store,
                "extract",
                "vocabulary.created",
                target_type="vocabulary",
                target_id=row["id"],
                detail={"name": row["name"], "kind": row["kind"]},
                session_id=session_id,
            )

    # Update paper_count and avg_confidence
    updated = 0
    for name, refs in references.items():
        if name not in existing_by_name:
            continue
        entry = existing_by_name[name]
        entry_id = entry["id"]
        unique_papers = {r[0] for r in refs}
        avg_conf = sum(r[1] for r in refs) / len(refs)
        store.update(
            "vocabulary",
            "paper_count = ?, avg_confidence = ?",
            "id = ?",
            (len(unique_papers), round(avg_conf, 4), entry_id),
        )
        log_event(
            store,
            "extract",
            "vocabulary.updated",
            target_type="vocabulary",
            target_id=entry_id,
            detail={"paper_count": len(unique_papers), "avg_confidence": round(avg_conf, 4)},
            session_id=session_id,
        )
        updated += 1

    return {"new_entries": len(new_rows), "updated_entries": updated}


def build_vocabulary(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    session_id: str | None = None,
) -> dict[str, int]:
    """Process new concepts from all extraction types, update stats, embed vocabulary."""
    stats = process_new_concepts(store, session_id=session_id)

    # Embed all vocabulary entries that lack embeddings
    vocab_rows = store.query("vocabulary")
    to_embed = [r for r in vocab_rows if not r.get("embedding")]

    if to_embed:
        texts = [f"{r['name']}: {r['description']}" for r in to_embed]
        embeddings = embed_strings(
            texts,
            provider=embedding_provider,
            model_name=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )
        for row, emb in zip(to_embed, embeddings, strict=True):
            store.upsert_embedding("vocabulary", row["id"], emb.tolist())

    # Rebuild FTS index for hybrid search
    store.rebuild_vocabulary_fts()

    logger.info(
        "Vocabulary: %d new, %d updated, %d embedded",
        stats["new_entries"],
        stats["updated_entries"],
        len(to_embed),
    )
    log_event(
        store,
        "build",
        "taxonomy.built",
        detail={
            "new_entries": stats["new_entries"],
            "updated_entries": stats["updated_entries"],
            "embedded": len(to_embed),
        },
        session_id=session_id,
    )
    return stats


# Backward compatibility alias
build_tradeoff_taxonomy = build_vocabulary


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
        store.rebuild_vocabulary_fts()
        logger.info("Loaded %d seed vocabulary entries", len(new_rows))

    return len(new_rows)
