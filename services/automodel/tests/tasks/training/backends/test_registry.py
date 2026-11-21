# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-reranker detection asks Automodel's dispatch; ``conftest`` stands in for it without Automodel."""

import json
from pathlib import Path

import pytest
from nhx.automodel.tasks.training.backends.registry import causal_reranker_class, causal_reranker_collator


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, "Qwen3RerankerForCausalReranking"),
        ({"model_type": "qwen3", "architectures": ["Qwen3ForSequenceClassification"]}, None),
        ({"model_type": "llama_bidirec", "architectures": ["LlamaBidirectionalForSequenceClassification"]}, None),
        (
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 131072,
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 32.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                },
            },
            None,
        ),
        (None, None),
    ],
    ids=["qwen3-causal-lm", "qwen3-converted-classifier", "score-head-reranker", "llama3-rope-scaling", "no-config"],
)
def test_causal_reranker_class_follows_automodel_dispatch(
    tmp_path: Path, config: dict[str, object] | None, expected: str | None
) -> None:
    """Only a class Automodel dispatches the checkpoint to, and that generates, is a causal reranker."""
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))

    reranker = causal_reranker_class(tmp_path, trust_remote_code=False)

    assert (reranker.__name__ if reranker is not None else None) == expected


@pytest.mark.parametrize(
    ("class_name", "collator"),
    [
        (
            "Qwen3RerankerForCausalReranking",
            "nemo_automodel.components.models.qwen3_reranker.collator.Qwen3ContextAwareRerankerCollator",
        ),
        ("NewCausalReranker", None),
    ],
    ids=["qwen3", "unknown-class"],
)
def test_causal_reranker_collator_needs_a_known_prompt(class_name: str, collator: str | None) -> None:
    """A causal reranker class with no known prompt must not train on another reranker's prompt."""
    reranker = type(class_name, (), {})

    if collator is None:
        with pytest.raises(ValueError, match="no training prompt"):
            causal_reranker_collator(reranker)
    else:
        assert causal_reranker_collator(reranker) == collator
