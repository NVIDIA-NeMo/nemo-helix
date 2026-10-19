# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.model_utils import is_embedding_model


def test_is_embedding_model_returns_false_for_none() -> None:
    assert is_embedding_model(None) is False


def test_is_embedding_model_detects_embed_substring_case_insensitive() -> None:
    assert is_embedding_model("NVIDIA/Llama-Nemotron-Embed-1B-v2") is True


def test_is_embedding_model_returns_false_for_non_embedding_name() -> None:
    assert is_embedding_model("meta/llama-3.1-8b-instruct") is False
