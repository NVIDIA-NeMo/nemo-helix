# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the ``nemo inference`` sub-groups."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from nemo_platform_ext.cli.core.stdin_utils import read_data_input_with_flags

_BOOL = TypeAdapter(bool)


def filter_query_value(value: str | dict[str, Any] | None) -> str | None:
    """Render a merged ``--filter`` value as the ``filter`` query string.

    Text-grammar expressions pass through; per-field / JSON-object filters are
    serialized as a JSON object, which the platform parses server-side.
    """
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def offset_query_params(
    *,
    filter_value: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    **extra: Any,
) -> dict[str, Any] | None:
    """Build an offset-paginated list query, omitting unset keys."""
    query_params: dict[str, Any] = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    for key, value in extra.items():
        if value is not None:
            query_params[key] = value
    return query_params or None


def read_input_payload(input_file: str | None, input_data: str | None) -> dict[str, Any]:
    """Read the optional ``--input-file`` / ``--input-data`` base payload."""
    if input_file or input_data:
        return read_data_input_with_flags(input_file=input_file, input_data=input_data)
    return {}


def pop_workspace(payload: dict[str, Any]) -> str | None:
    """Remove and return the ``workspace`` key, which is a path parameter rather than a body field."""
    workspace = payload.pop("workspace", None)
    return str(workspace) if workspace is not None else None


def pop_exist_ok(payload: dict[str, Any]) -> bool | None:
    """Remove and return the ``exist_ok`` client option, which is never sent on the wire."""
    exist_ok = payload.pop("exist_ok", None)
    # --input-data may carry it as a JSON/YAML string; parse it like the server would.
    return _BOOL.validate_python(exist_ok) if exist_ok is not None else None
