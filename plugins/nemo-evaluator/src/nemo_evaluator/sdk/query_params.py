# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared query-parameter helpers for evaluator SDK resources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeAlias

from nemo_helix_plugin.filter_ops import ElemMatchScalar

QueryParams: TypeAlias = dict[str, str | int | bool | None]


def list_params(page: int, page_size: int, sort: str | None) -> QueryParams:
    """Return pagination and optional sort query params."""
    params: QueryParams = {"page": page, "page_size": page_size}
    if sort is not None:
        params["sort"] = sort
    return params


def list_filter_params(
    *, kind: str | None = None, metadata: Mapping[str, ElemMatchScalar] | None = None
) -> QueryParams:
    """Return the ``filter`` query param for a task/taskset listing, or nothing when unfiltered."""
    conditions: dict[str, ElemMatchScalar] = {f"metadata.{key}": value for key, value in (metadata or {}).items()}
    if kind is not None:
        conditions["kind"] = kind
    return {"filter": json.dumps(conditions)} if conditions else {}


def project_params(project: str | None) -> QueryParams | None:
    """Return project query params only when a project was supplied."""
    return {"project": project} if project is not None else None


def revision_selector(revision: str | None, tag: str | None) -> str | None:
    """Return the selected revision path segment, rejecting ambiguous input."""
    if revision is not None and tag is not None:
        raise ValueError("pass either 'revision' (a content digest) or 'tag', not both")
    return revision if revision is not None else tag
