# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared query-parameter helpers for evals SDK resources."""

from __future__ import annotations

from typing import TypeAlias

QueryParams: TypeAlias = dict[str, str | int | bool | None]


def list_params(page: int, page_size: int, sort: str | None) -> QueryParams:
    """Return pagination and optional sort query params."""
    params: QueryParams = {"page": page, "page_size": page_size}
    if sort is not None:
        params["sort"] = sort
    return params


def project_params(project: str | None) -> QueryParams | None:
    """Return project query params only when a project was supplied."""
    return {"project": project} if project is not None else None


def revision_selector(revision: str | None, tag: str | None) -> str | None:
    """Return the selected revision path segment, rejecting ambiguous input."""
    if revision is not None and tag is not None:
        raise ValueError("pass either 'revision' (a content digest) or 'tag', not both")
    return revision if revision is not None else tag
