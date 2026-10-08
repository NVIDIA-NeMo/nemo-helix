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


def list_filter_params(*, metadata: Mapping[str, ElemMatchScalar] | None = None, **conditions: object) -> QueryParams:
    """Return the ``filter`` query param for a task/taskset listing, or nothing when unfiltered.

    ``conditions`` are filter fields; ``None`` values are left out.
    """
    filters: dict[str, object] = {f"metadata.{key}": value for key, value in (metadata or {}).items()}
    filters.update({field: value for field, value in conditions.items() if value is not None})
    return {"filter": json.dumps(filters)} if filters else {}


def like_filter(text: str | None) -> dict[str, str] | None:
    return None if text is None else {"$like": text}


def project_params(project: str | None) -> QueryParams | None:
    """Return project query params only when a project was supplied."""
    return {"project": project} if project is not None else None


def revision_selector(revision: str | None, tag: str | None) -> str | None:
    """Return the selected revision path segment, rejecting ambiguous input."""
    if revision is not None and tag is not None:
        raise ValueError("pass either 'revision' (a content digest) or 'tag', not both")
    return revision if revision is not None else tag
