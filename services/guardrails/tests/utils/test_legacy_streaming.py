# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs saved before nemoguardrails 0.24 must keep loading.

The rows below are what the service stored before the upgrade: entities are persisted with their
defaults, so ``streaming.enabled`` is true even when the user never mentioned streaming. nemoguardrails
0.24 rejects that block next to a rewriting output rail. The service's model repairs it wherever a
config is validated, so these tests go through the model and the entity rather than a helper.
"""

from typing import Any

import pytest
from nhx.guardrails.app.handlers.utils import to_internal_rails_config
from nhx.guardrails.entities.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import OutputRails, RailsConfig
from nhx.guardrails.entities.values.common import GuardrailsDataInput
from pydantic import ValidationError

REWRITING_FLOW = "mask sensitive data on output"
LEGACY_DEFAULTS = {"enabled": True, "chunk_size": 200, "context_size": 50, "stream_first": True}
SELF_CHECK_PROMPT = {"task": "self_check_output", "content": "Check: {{ bot_response }} Answer (Yes or No):"}


def _row(flow: str, **streaming: Any) -> dict:
    """A stored config row: the output block carries the streaming defaults the old model persisted."""
    return {
        "rails": {"output": {"flows": [flow], "streaming": {**LEGACY_DEFAULTS, **streaming}}},
        "prompts": [SELF_CHECK_PROMPT],
    }


def _library_streaming(row: dict):
    return to_internal_rails_config(RailsConfig.model_validate(row)).rails.output.streaming


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(_row(REWRITING_FLOW), (True, 200, False, 0), id="rewriting-rail-with-default-streaming"),
        pytest.param(_row(REWRITING_FLOW, chunk_size=100), (True, 100, False, 0), id="chunk-size-is-kept"),
        pytest.param(
            {"rails": {"output": {"flows": [REWRITING_FLOW]}}}, (True, 200, False, 0), id="no-streaming-block"
        ),
        pytest.param(_row(REWRITING_FLOW, enabled=False), (False, 200, True, 50), id="streaming-disabled"),
        pytest.param(_row("self check output"), (True, 200, True, 50), id="rail-that-does-not-rewrite"),
    ],
)
def test_config_is_loaded_with_settings_the_library_accepts(row: dict, expected: tuple[bool, int, bool, int]) -> None:
    """The ``no-streaming-block`` case also pins the default every stored row relies on: enabled."""
    streaming = _library_streaming(row)

    assert (streaming.enabled, streaming.chunk_size, streaming.stream_first, streaming.context_size) == expected


def test_deliberately_tuned_context_size_still_reaches_the_library_error() -> None:
    """A value someone chose is not ours to overwrite; the library's message tells them what to do."""
    with pytest.raises(ValidationError, match="rewrite the response"):
        _library_streaming(_row(REWRITING_FLOW, context_size=20))


def test_stored_entity_is_repaired_when_it_is_read() -> None:
    """The entity store builds ``GuardrailConfig`` from a saved row; the plugin then fetches it over HTTP."""
    entity = GuardrailConfig(name="legacy", workspace="default", data=_row(REWRITING_FLOW))

    streaming = entity.model_dump(mode="json")["data"]["rails"]["output"]["streaming"]

    assert (streaming["enabled"], streaming["stream_first"], streaming["context_size"]) == (True, False, 0)


def test_inline_config_in_a_request_is_repaired() -> None:
    """A client may re-post a config it copied from a GET; it must not fail on the first request."""
    request = GuardrailsDataInput.model_validate({"config": _row(REWRITING_FLOW), "config_id": None})

    assert isinstance(request.config, RailsConfig)
    assert request.config.rails.output.streaming.stream_first is False


def test_malformed_flow_name_is_ignored_and_left_for_the_library_to_report() -> None:
    rails = OutputRails.model_validate(
        {"flows": ["mask sensitive data on output $model="], "streaming": LEGACY_DEFAULTS}
    )

    assert (rails.streaming.stream_first, rails.streaming.context_size) == (True, 50)
