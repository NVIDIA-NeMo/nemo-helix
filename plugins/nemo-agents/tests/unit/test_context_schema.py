# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference resolution in caller-supplied context schemas."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nemo_agents_plugin.context_schema import check_context_schema, validate_context

_REMOTE_REF = "http://169.254.169.254/latest/meta-data/schema.json"


def _urlopen() -> MagicMock:
    # Records calls rather than raising: ``referencing`` wraps any error a
    # retrieval raises as ``Unresolvable``, which would hide an assertion here.
    return MagicMock(side_effect=OSError("network disabled in tests"))


@pytest.mark.parametrize("ref", [_REMOTE_REF, "https://example.com/schema.json", "file:///etc/passwd"])
def test_a_remote_ref_is_rejected_without_being_fetched(ref: str) -> None:
    """A caller-supplied schema must not make the server request a host the caller names."""
    schema = {"type": "object", "properties": {"since": {"$ref": ref}}}
    check_context_schema(schema)

    with (
        patch("urllib.request.urlopen", _urlopen()) as urlopen,
        pytest.raises(ValueError, match=r"context_schema has a \$ref that cannot be resolved"),
    ):
        validate_context({"since": "2026-10-01T00:00:00Z"}, schema)
    urlopen.assert_not_called()


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(
            {"$defs": {"timestamp": {"type": "string"}}, "properties": {"since": {"$ref": "#/$defs/timestamp"}}},
            id="draft-2020-12-defs",
        ),
        pytest.param(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "definitions": {"timestamp": {"type": "string"}},
                "properties": {"since": {"$ref": "#/definitions/timestamp"}},
            },
            id="draft-07-definitions",
        ),
    ],
)
def test_a_ref_within_the_schema_still_resolves(schema: dict[str, Any]) -> None:
    with patch("urllib.request.urlopen", _urlopen()) as urlopen:
        validate_context({"since": "2026-10-01T00:00:00Z"}, schema)
        with pytest.raises(ValueError, match=r"context\['since'\]: 5 is not of type 'string'"):
            validate_context({"since": 5}, schema)
    urlopen.assert_not_called()
