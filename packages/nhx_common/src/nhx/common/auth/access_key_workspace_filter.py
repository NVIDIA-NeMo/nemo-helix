# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop a workspace-limited Scoped Access Key from widening a list call with ``filter[workspace]``.

The PDP authorizes the workspace in the path, but several list endpoints let ``filter[workspace]``
replace it. For a limited key every workspace the filter names must equal the path workspace, and no
workspace comparison may sit under ``$not``.
"""

from __future__ import annotations

from collections.abc import Iterator

from nemo_helix_plugin.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    LogicalOperation,
    parse_bracket_filter,
    parse_json_filter,
)
from nemo_helix_plugin.api.text_filter import parse_text_filter
from nemo_helix_plugin.jobs.openapi_utils import parse_deep_object
from starlette.datastructures import QueryParams

FILTER_PARAM = "filter"
WORKSPACE_FIELD = "workspace"
INVALID_FILTER_DETAIL = "Invalid filter"
MIXED_FILTER_DETAIL = "Use either filter= or filter[...] query parameters, not both"
WORKSPACE_FILTER_MISMATCH_DETAIL = (
    "This access key is limited to specific workspaces; filter[workspace] must match the workspace in the path"
)


def _path_workspace(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part == "workspaces":
            return parts[index + 1]
    return None


def _parse_filter(query_params: QueryParams) -> FilterOperation | None:
    raw = query_params.get(FILTER_PARAM)
    if raw is not None:
        return parse_json_filter(raw) if raw.startswith("{") else parse_text_filter(raw)
    bracket = parse_deep_object(name=FILTER_PARAM, params=query_params)
    return parse_bracket_filter(bracket) if bracket else None


def _workspace_comparisons(
    operation: FilterOperation, *, negated: bool = False
) -> Iterator[tuple[ComparisonOperation, bool]]:
    """Each comparison on the workspace field, and whether it sits under a $not."""
    if isinstance(operation, ComparisonOperation):
        if operation.field == WORKSPACE_FIELD:
            yield operation, negated
    elif isinstance(operation, LogicalOperation):
        child_negated = negated or operation.operator == FilterOperator.NOT
        for child in operation.operations:
            yield from _workspace_comparisons(child, negated=child_negated)


def _is_bracket_filter_key(key: str) -> bool:
    # Same reading as parse_deep_object, which also accepts malformed keys such as "filter][workspace]".
    return "[" in key and key.replace("]", "").split("[")[0] == FILTER_PARAM


def workspace_filter_error(path: str, query_params: QueryParams) -> str | None:
    """Why a limited key's filter is refused, or None when it stays within the path workspace.

    A filter that can't be parsed is refused, so the guard fails closed.
    """
    has_raw = FILTER_PARAM in query_params
    has_bracket = any(_is_bracket_filter_key(key) for key in query_params)
    if not has_raw and not has_bracket:
        return None
    # Some endpoints read only the bracket form, so a raw filter must not hide a bracket one from this guard.
    if has_raw and has_bracket:
        return MIXED_FILTER_DETAIL
    try:
        operation = _parse_filter(query_params)
    # The text parser raises lark errors, not ValueError; the endpoint's own filter dependency also treats any
    # exception as a bad filter.
    except Exception:
        return INVALID_FILTER_DETAIL
    if operation is None:
        return None
    path_workspace = _path_workspace(path)
    for comparison, negated in _workspace_comparisons(operation):
        if (
            negated
            or path_workspace is None
            or comparison.operator != FilterOperator.EQ
            or comparison.value != path_workspace
        ):
            return WORKSPACE_FILTER_MISMATCH_DETAIL
    return None
