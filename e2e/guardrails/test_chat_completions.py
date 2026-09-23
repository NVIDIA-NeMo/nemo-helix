# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the Guardrails plugin on chat completions.

These tests verify that the ``nemo-guardrails`` inference middleware works
through the real platform subprocess, exercising content-safety input and
output rails on non-streaming and streaming Inference Gateway chat-completion
routes.

Mock provider mode is enabled by the NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX
env var set in e2e/conftest.py. Tests use ``add_mock_provider()`` from
nhx.testing to create one provider serving both the backend chat model and the
content-safety task model, so the full Guardrails path runs without a real
inference backend.
"""

from collections.abc import Callable
from typing import Any

import nemo_helix
import pytest

from e2e.guardrails.utils import (
    BACKEND_RESPONSE,
    CONTENT_SAFETY_INPUT_FLOW,
    CONTENT_SAFETY_OUTPUT_FLOW,
    REFUSAL_TEXT,
    post_chat_completion,
    post_streaming_chat_completion,
)


def _assert_blocked(response: dict[str, Any]) -> None:
    assert response["choices"][0]["finish_reason"] == "content_filter"
    assert response["choices"][0]["message"]["content"] == REFUSAL_TEXT


def _assert_streaming_blocked(response: dict[str, Any]) -> None:
    if response.get("error"):
        assert response["error"]["code"] == "content_blocked"
        return

    _assert_blocked(response)


def _assert_allowed(response: dict[str, Any]) -> None:
    assert response["choices"][0]["message"]["content"] == BACKEND_RESPONSE


def _activated_rails_by_name(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    guardrails_data = response.get("guardrails_data") or {}
    log = guardrails_data.get("log") or {}
    activated_rails = log.get("activated_rails") or []
    return {rail["name"]: rail for rail in activated_rails}


# ---------------------------------------------------------------------------
# Non-streaming chat completions
# ---------------------------------------------------------------------------


def test_chat_completions_blocks_unsafe_input(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="referenced", outcome="unsafe_input", rail_types=("input",))

    response = post_chat_completion(test_case)

    _assert_blocked(response)


def test_chat_completions_blocks_unsafe_llm_output(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="referenced", outcome="unsafe_output", rail_types=("output",))

    response = post_chat_completion(test_case)

    _assert_blocked(response)


def test_chat_completions_allows_safe_request(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="referenced", outcome="safe", rail_types=("input", "output"))

    response = post_chat_completion(test_case)

    _assert_allowed(response)


def test_chat_completions_blocks_unsafe_input_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="inline", outcome="unsafe_input", rail_types=("input",))

    response = post_chat_completion(test_case)

    _assert_blocked(response)


def test_chat_completions_blocks_unsafe_llm_output_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="inline", outcome="unsafe_output", rail_types=("output",))

    response = post_chat_completion(test_case)

    _assert_blocked(response)


def test_chat_completions_allows_safe_request_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="inline", outcome="safe", rail_types=("input", "output"))

    response = post_chat_completion(test_case)

    _assert_allowed(response)


# ---------------------------------------------------------------------------
# Streaming chat completions
# ---------------------------------------------------------------------------


def test_streaming_chat_completions_blocks_unsafe_input(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="referenced", outcome="unsafe_input", rail_types=("input",), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_streaming_blocked(response)


def test_streaming_chat_completions_blocks_unsafe_llm_output(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="referenced", outcome="unsafe_output", rail_types=("output",), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_streaming_blocked(response)


def test_streaming_chat_completions_allows_safe_request(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="referenced", outcome="safe", rail_types=("input", "output"), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_allowed(response)


def test_streaming_chat_completions_blocks_unsafe_input_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="inline", outcome="unsafe_input", rail_types=("input",), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_streaming_blocked(response)


def test_streaming_chat_completions_blocks_unsafe_llm_output_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="inline", outcome="unsafe_output", rail_types=("output",), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_streaming_blocked(response)


def test_streaming_chat_completions_allows_safe_request_with_inline_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="inline", outcome="safe", rail_types=("input", "output"), streaming=True
    )

    response = post_streaming_chat_completion(test_case)

    _assert_allowed(response)


# ---------------------------------------------------------------------------
# Chat completion request/response contract
# ---------------------------------------------------------------------------


def test_chat_completions_reports_guardrails_metadata_when_blocked(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(
        config_mode="referenced", outcome="unsafe_output", rail_types=("input", "output")
    )

    response = post_chat_completion(
        test_case,
        extra_body={"guardrails": {"options": {"log": {"activated_rails": True}}}},
    )

    guardrails_data = response.get("guardrails_data") or {}
    assert guardrails_data.get("config_ids") == [test_case.config_ref]
    activated_rails = _activated_rails_by_name(response)
    assert activated_rails[CONTENT_SAFETY_INPUT_FLOW]["stop"] is False
    assert activated_rails[CONTENT_SAFETY_OUTPUT_FLOW]["stop"] is True


def test_chat_completions_rejects_unsupported_body_guardrails_config(
    guardrails_chat_test_case: Callable[..., Any],
) -> None:
    test_case = guardrails_chat_test_case(config_mode="referenced", outcome="safe", rail_types=("input",))

    with pytest.raises(nemo_helix.APIStatusError) as exc_info:
        post_chat_completion(
            test_case,
            extra_body={"guardrails": {"config_id": test_case.config_ref}},
        )

    assert exc_info.value.status_code == 422
