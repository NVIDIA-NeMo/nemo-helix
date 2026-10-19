# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_agents_plugin.jobs.job_usage import fabric_output_token_usage


def test_fabric_output_accepts_openai_usage_names() -> None:
    usage = fabric_output_token_usage(
        {"response": "done", "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}
    )

    assert usage is not None
    assert usage.input_tokens == 12
    assert usage.output_tokens == 5


def test_fabric_output_includes_cache_input_tokens() -> None:
    usage = fabric_output_token_usage(
        {
            "response": "done",
            "usage": {
                "input_tokens": 12,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 5,
            },
        }
    )

    assert usage is not None
    assert usage.input_tokens == 17
    assert usage.output_tokens == 5


def test_fabric_output_rejects_total_only_or_invalid_counts() -> None:
    assert fabric_output_token_usage({"usage": {"total_tokens": 17}}) is None
    assert fabric_output_token_usage({"usage": {"input_tokens": True, "output_tokens": -1}}) is None
    assert fabric_output_token_usage({"usage": {"input_tokens": 1, "cache_read_input_tokens": -1}}) is None
