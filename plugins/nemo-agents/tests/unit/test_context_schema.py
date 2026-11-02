# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""References in caller-supplied context schemas."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nemo_agents_plugin.context_schema import check_context_schema, validate_context
from referencing.exceptions import Unresolvable


@pytest.mark.parametrize(
    ("schema", "location"),
    [
        pytest.param({"$ref": "#"}, "context_schema['$ref']", id="root-refers-to-itself"),
        pytest.param(
            {"properties": {"since": {"$ref": "http://169.254.169.254/latest/meta-data/schema.json"}}},
            "context_schema['properties']['since']['$ref']",
            id="remote",
        ),
        pytest.param(
            {"$defs": {"t": {"type": "string"}}, "properties": {"since": {"$ref": "#/$defs/t"}}},
            "context_schema['properties']['since']['$ref']",
            id="local-defs",
        ),
        pytest.param(
            {"properties": {"ids": {"type": "array", "prefixItems": [{"$dynamicRef": "#node"}]}}},
            "context_schema['properties']['ids']['prefixItems'][0]['$dynamicRef']",
            id="dynamic-ref-in-a-list",
        ),
        pytest.param(
            {"$schema": "https://json-schema.org/draft/2019-09/schema", "$recursiveRef": "#"},
            "context_schema['$recursiveRef']",
            id="recursive-ref",
        ),
    ],
)
def test_a_reference_is_rejected_with_its_location(schema: dict[str, Any], location: str) -> None:
    with pytest.raises(ValueError, match=r"context_schema does not support \$ref") as excinfo:
        check_context_schema(schema)

    assert location in str(excinfo.value)


def test_a_schema_without_references_is_accepted() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"since": {"type": "string", "format": "date-time"}},
        "additionalProperties": False,
    }

    check_context_schema(schema)
    validate_context({"since": "2026-10-01T00:00:00Z"}, schema)


def test_validation_never_fetches_a_remote_reference() -> None:
    """The backstop for a schema that skipped ``check_context_schema``: unresolvable, never requested."""
    # Records calls rather than raising: ``referencing`` wraps any error a
    # retrieval raises as ``Unresolvable``, which would hide an assertion here.
    urlopen = MagicMock(side_effect=OSError("network disabled in tests"))
    schema = {"properties": {"since": {"$ref": "http://169.254.169.254/latest/meta-data/schema.json"}}}

    with patch("urllib.request.urlopen", urlopen), pytest.raises(Unresolvable):
        validate_context({"since": "x"}, schema)
    urlopen.assert_not_called()
