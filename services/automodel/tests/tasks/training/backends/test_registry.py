# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-reranker detection loads the checkpoint config with ``transformers`` and asks Automodel's dispatch.

``conftest`` stands in for the dispatch without Automodel. The config loading is real, so the dispatch
test needs the opt-in ``automodel-tests`` group: ``uv run --group automodel-tests pytest ...``.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from nhx.automodel.tasks.training.backends import registry
from nhx.automodel.tasks.training.backends.registry import causal_reranker_collator

_LLAMA3_ROPE = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}
# Llama Nemotron Rerank 1B v2's published config, cut to what its config class reads, and the
# remote-code config class it ships.
_RERANKER_CONFIG = {
    "architectures": ["LlamaBidirectionalForSequenceClassification"],
    "auto_map": {
        "AutoConfig": "llama_bidirectional_model.LlamaBidirectionalConfig",
        "AutoModelForSequenceClassification": "llama_bidirectional_model.LlamaBidirectionalForSequenceClassification",
    },
    "model_type": "llama_bidirec",
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 16,
    "max_position_embeddings": 131072,
    "rope_scaling": _LLAMA3_ROPE,
    "rope_theta": 500000.0,
    "pooling": "avg",
    "temperature": 1.0,
}
_RERANKER_REMOTE_CONFIG = """from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaBidirectionalConfig(LlamaConfig):
    model_type = "llama_bidirec"

    def __init__(self, pooling="avg", temperature=1.0, **kwargs):
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs)
"""
_LLAMA3_BASE_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "max_position_embeddings": 131072,
    "rope_scaling": _LLAMA3_ROPE,
}


@pytest.mark.parametrize(
    ("config", "trust_remote_code", "loaded_as", "expected"),
    [
        (
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
            False,
            "Qwen3Config",
            "Qwen3RerankerForCausalReranking",
        ),
        ({"model_type": "qwen3", "architectures": ["Qwen3ForSequenceClassification"]}, False, "Qwen3Config", None),
        (_RERANKER_CONFIG, True, "LlamaBidirectionalConfig", None),
        (_RERANKER_CONFIG, False, None, None),
        (_LLAMA3_BASE_CONFIG, False, "LlamaConfig", None),
        (None, False, None, None),
    ],
    ids=[
        "qwen3-causal-lm",
        "qwen3-converted-classifier",
        "rerank-1b-v2-remote-config",
        "rerank-1b-v2-untrusted",
        "llama3-rope-scaling-base",
        "no-config",
    ],
)
def test_causal_reranker_class_follows_automodel_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object] | None,
    trust_remote_code: bool,
    loaded_as: str | None,
    expected: str | None,
) -> None:
    """The checkpoint's own config class reaches Automodel's dispatch, RoPE fields intact.

    Only a class the dispatch picks, and that generates, is a causal reranker. A config that does not
    load, such as remote code the job does not trust, is not one and never reaches the dispatch.
    """
    pytest.importorskip("transformers", reason="needs the automodel-tests dependency group")
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "llama_bidirectional_model.py").write_text(_RERANKER_REMOTE_CONFIG)
    dispatched: list[Any] = []
    dispatch = registry._automodel_cross_encoder_class
    monkeypatch.setattr(registry, "_automodel_cross_encoder_class", lambda c: dispatched.append(c) or dispatch(c))

    reranker = registry.causal_reranker_class(tmp_path, trust_remote_code=trust_remote_code)

    assert (reranker.__name__ if reranker is not None else None) == expected
    assert [type(c).__name__ for c in dispatched] == ([loaded_as] if loaded_as else [])
    if config is not None and "rope_scaling" in config and dispatched:
        assert dispatched[0].max_position_embeddings == 131072
        assert dispatched[0].rope_parameters["rope_type"] == "llama3"


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
