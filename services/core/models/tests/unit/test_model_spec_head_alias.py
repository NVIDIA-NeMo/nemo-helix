# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from nemo_helix_plugin.models.types import ModelSpec as PluginModelSpec
from nhx.core.models.schemas import ModelSpec

_MINIMAL = {
    "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
    "family": "llama",
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_kv_heads": 32,
    "ffn_hidden_size": 16384,
    "vocab_size": 32000,
    "tied_embeddings": True,
    "gated_mlp": True,
    "base_num_parameters": 1000,
    "precision": "fp16",
}


def test_head_type_embedding_derives_alias() -> None:
    spec = ModelSpec(**_MINIMAL, head_type="embedding", is_embedding_model=False)
    assert spec.head_type == "embedding"
    assert spec.is_embedding_model is True


def test_legacy_alias_only_sets_embedding_head() -> None:
    spec = ModelSpec(**_MINIMAL, is_embedding_model=True)
    assert spec.head_type == "embedding"
    assert spec.is_embedding_model is True


def test_cross_encoder_wins_over_stale_embedding_alias() -> None:
    spec = ModelSpec(**_MINIMAL, head_type="cross_encoder", is_embedding_model=True)
    assert spec.head_type == "cross_encoder"
    assert spec.is_embedding_model is False


def test_unknown_plus_alias_true_normalizes_to_embedding() -> None:
    spec = ModelSpec(**_MINIMAL, head_type="unknown", is_embedding_model=True)
    assert spec.head_type == "embedding"
    assert spec.is_embedding_model is True


def test_plugin_model_spec_cross_encoder_wins_over_stale_embedding_alias() -> None:
    spec = PluginModelSpec(**_MINIMAL, head_type="cross_encoder", is_embedding_model=True)
    assert spec.head_type == "cross_encoder"
    assert spec.is_embedding_model is False


def test_false_like_alias_strings_stay_unknown() -> None:
    for spec_cls in (ModelSpec, PluginModelSpec):
        for alias in ("false", "0", "off"):
            spec = spec_cls.model_validate({**_MINIMAL, "is_embedding_model": alias})
            assert spec.head_type == "unknown"
            assert spec.is_embedding_model is False


def test_true_like_alias_strings_normalize_to_embedding() -> None:
    for spec_cls in (ModelSpec, PluginModelSpec):
        for alias in ("true", "1", "on"):
            spec = spec_cls.model_validate({**_MINIMAL, "is_embedding_model": alias})
            assert spec.head_type == "embedding"
            assert spec.is_embedding_model is True


def test_constructing_model_spec_does_not_warn_on_deprecated_alias() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ModelSpec(**_MINIMAL, head_type="embedding")
        PluginModelSpec(**_MINIMAL, head_type="embedding")
        _ = ModelSpec(**_MINIMAL, head_type="embedding").is_embedding_model
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
