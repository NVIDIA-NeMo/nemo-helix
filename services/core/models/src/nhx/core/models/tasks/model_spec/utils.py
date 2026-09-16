# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml

_EMBEDDING_PIPELINE_TAGS = frozenset({"feature-extraction", "sentence-similarity", "text-embeddings-inference"})
_CROSS_ENCODER_PIPELINE_TAGS = frozenset({"text-ranking", "reranking"})
_GENERATIVE_PIPELINE_TAGS = frozenset({"text-generation", "text2text-generation", "image-to-text"})
_RERANKING_TAGS = frozenset({"reranker", "reranking", "text-ranking"})
_SUPPORTING_TAGS = frozenset(
    {
        "sentence-transformers",
        "text-embeddings-inference",
        "sentence-similarity",
        "embedding",
        "multimodal",
        "retrieval",
        *_RERANKING_TAGS,
    }
)
_SENTENCE_TRANSFORMER_ARTIFACTS = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling",
)
_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---(?:\s*\n|$)", re.DOTALL)
ModelHeadType = Literal["causal_lm", "embedding", "cross_encoder", "unknown"]


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def _load_architectures(config_path: Path) -> list[str]:
    config = _load_config(config_path)

    architectures = config.get("architectures", [])
    if isinstance(architectures, str):
        return [architectures]
    if isinstance(architectures, list):
        return [str(arch) for arch in architectures]
    return []


def _load_readme_frontmatter(readme_path: Path) -> dict[str, Any]:
    if not readme_path.is_file():
        return {}

    try:
        content = readme_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    match = _FRONTMATTER_PATTERN.search(content)
    if not match:
        return {}

    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize_tags(raw_tags: Any) -> set[str]:
    if not isinstance(raw_tags, list):
        return set()
    return {str(tag).strip().lower() for tag in raw_tags if str(tag).strip()}


def infer_model_head_type(model_dir: str) -> tuple[ModelHeadType, str]:
    """
    Infer the task-specific head already persisted in a Hugging Face checkpoint.

    This intentionally describes the checkpoint as saved. A causal-LM checkpoint
    can still be used as the backbone for a new bi-encoder or cross-encoder job;
    that training intent must be selected by the job, not inferred here.

    Args:
        model_dir (str): The path to the downloaded model directory.

    Returns:
        The inferred head type and a string explaining the strongest signals.
    """

    model_path = Path(model_dir)
    if not model_path.is_dir():
        return "unknown", f"Directory not found: {model_dir}"

    embedding_signals: list[str] = []
    cross_encoder_signals: list[str] = []
    causal_lm_signals: list[str] = []
    supporting_signals: list[str] = []

    for artifact in _SENTENCE_TRANSFORMER_ARTIFACTS:
        artifact_path = model_path / artifact
        if artifact == "1_Pooling":
            if artifact_path.is_dir():
                embedding_signals.append("artifact:1_Pooling/")
            continue
        if artifact_path.is_file():
            embedding_signals.append(f"artifact:{artifact}")

    config = _load_config(model_path / "config.json")
    architectures = _load_architectures(model_path / "config.json")
    sequence_classification_architectures: list[str] = []
    for architecture in architectures:
        architecture_lower = architecture.lower()
        if "forsequenceclassification" in architecture_lower:
            sequence_classification_architectures.append(architecture)
        elif "bidirectional" in architecture_lower or "embedding" in architecture_lower:
            embedding_signals.append(f"architecture:{architecture}")
        elif "forcausallm" in architecture_lower or "forconditionalgeneration" in architecture_lower:
            causal_lm_signals.append(f"architecture:{architecture}")

    model_type = str(config.get("model_type", "")).lower()
    if "bidirec" in model_type:
        embedding_signals.append(f"model_type:{model_type}")

    auto_map = config.get("auto_map", {})
    if isinstance(auto_map, dict):
        if "AutoModel" in auto_map and ("bidirec" in model_type or "pooling" in config):
            embedding_signals.append("auto_map:AutoModel")

    if "pooling" in config and not cross_encoder_signals:
        embedding_signals.append("config:pooling")

    readme_frontmatter = _load_readme_frontmatter(model_path / "README.md")
    pipeline_tag_raw = readme_frontmatter.get("pipeline_tag")
    pipeline_tag = str(pipeline_tag_raw).strip().lower() if isinstance(pipeline_tag_raw, str) else None
    if pipeline_tag in _EMBEDDING_PIPELINE_TAGS:
        embedding_signals.append(f"pipeline_tag:{pipeline_tag}")
    if pipeline_tag in _CROSS_ENCODER_PIPELINE_TAGS:
        cross_encoder_signals.append(f"pipeline_tag:{pipeline_tag}")
    if pipeline_tag in _GENERATIVE_PIPELINE_TAGS:
        causal_lm_signals.append(f"pipeline_tag:{pipeline_tag}")

    tags = _normalize_tags(readme_frontmatter.get("tags"))
    id2label = config.get("id2label")
    num_labels = config.get("num_labels")
    is_single_score_head = num_labels == 1 or (isinstance(id2label, dict) and len(id2label) == 1)
    has_reranking_metadata = pipeline_tag in _CROSS_ENCODER_PIPELINE_TAGS or bool(tags.intersection(_RERANKING_TAGS))
    if sequence_classification_architectures and (is_single_score_head or has_reranking_metadata):
        cross_encoder_signals.extend(
            f"architecture:{architecture}" for architecture in sequence_classification_architectures
        )
    if (
        isinstance(auto_map, dict)
        and "AutoModelForSequenceClassification" in auto_map
        and (is_single_score_head or has_reranking_metadata)
    ):
        cross_encoder_signals.append("auto_map:AutoModelForSequenceClassification")

    supporting_tags = sorted(tags.intersection(_SUPPORTING_TAGS))
    if supporting_tags:
        supporting_signals.append(f"tags:{'|'.join(supporting_tags)}")

    if (model_path / "generation_config.json").is_file():
        causal_lm_signals.append("file:generation_config.json")

    # A persisted scoring head is more specific than generic embedding/card
    # metadata, and sentence-transformer artifacts are more specific than a
    # stale generation_config.json copied from a causal-LM backbone.
    for head_type, signals in (
        ("cross_encoder", cross_encoder_signals),
        ("embedding", embedding_signals),
        ("causal_lm", causal_lm_signals),
    ):
        if not signals:
            continue
        details = ",".join(signals)
        if supporting_signals:
            details = f"{details};supporting={','.join(supporting_signals)}"
        return head_type, f"{head_type}:{details}"

    if supporting_signals:
        return "unknown", f"inconclusive:supporting={','.join(supporting_signals)}"
    return "unknown", "inconclusive:no_strong_head_signals"


def is_embedding_model_v2(model_dir: str) -> tuple[bool, str]:
    """Compatibility wrapper for callers that still consume a boolean."""
    head_type, reason = infer_model_head_type(model_dir)
    return head_type == "embedding", reason
