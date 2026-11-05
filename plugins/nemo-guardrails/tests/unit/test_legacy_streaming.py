# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs saved before nemoguardrails 0.24 must keep loading in the plugin.

Stored rows carry ``streaming.enabled: true`` with the old defaults whether or not the user configured
streaming (the guardrails service persists entities with their defaults). nemoguardrails 0.24 rejects
that block next to a rewriting output rail, which would stop every such config from loading.
"""

from typing import Any

import pytest
from nemo_guardrails_plugin.legacy_streaming import relax_legacy_streaming_defaults
from nemo_guardrails_plugin.llmrails_cache import stabilize
from nemo_helix_plugin.guardrail.types import RailsConfig as HelixRailsConfig
from pydantic import ValidationError

REWRITING_FLOW = "mask sensitive data on output"
LEGACY_DEFAULTS = {"enabled": True, "chunk_size": 200, "context_size": 50, "stream_first": True}
SELF_CHECK_PROMPT = {"task": "self_check_output", "content": "Check: {{ bot_response }} Answer (Yes or No):"}


def _streaming(flow: str, **streaming: Any):
    """The streaming block the library receives for a stored row with ``flow`` as its output rail."""
    rails = HelixRailsConfig.model_validate(
        {
            "rails": {"output": {"flows": [flow], "streaming": {**LEGACY_DEFAULTS, **streaming}}},
            "prompts": [SELF_CHECK_PROMPT],
        }
    )

    def no_target(_: str):
        raise AssertionError("no model needs resolving")

    return stabilize(rails, no_target).rails.rails.output.streaming


@pytest.mark.parametrize(
    ("flow", "streaming", "expected"),
    [
        pytest.param(REWRITING_FLOW, {}, (True, 200, False, 0), id="rewriting-rail-with-default-streaming"),
        pytest.param(REWRITING_FLOW, {"chunk_size": 100}, (True, 100, False, 0), id="chunk-size-is-kept"),
        pytest.param(REWRITING_FLOW, {"enabled": False}, (False, 200, True, 50), id="streaming-disabled"),
        pytest.param("self check output", {}, (True, 200, True, 50), id="rail-that-does-not-rewrite"),
    ],
)
def test_stored_row_is_loaded_with_settings_the_library_accepts(
    flow: str, streaming: dict[str, Any], expected: tuple[bool, int, bool, int]
) -> None:
    result = _streaming(flow, **streaming)

    assert (result.enabled, result.chunk_size, result.stream_first, result.context_size) == expected


def test_deliberately_tuned_context_size_still_reaches_the_library_error() -> None:
    with pytest.raises(ValidationError, match="rewrite the response"):
        _streaming(REWRITING_FLOW, context_size=20)


def test_malformed_flow_name_is_ignored_and_left_for_the_library_to_report() -> None:
    config = {"rails": {"output": {"flows": ["mask sensitive data on output $model="], "streaming": LEGACY_DEFAULTS}}}

    assert relax_legacy_streaming_defaults(config) is config
