# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for Projects (Entity Store).

Single source of truth for the HTTP contract. Replaces the Stainless-generated
``nemo_helix.resources.projects`` resource.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post, put
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.entities.types import DeleteResponse
from nemo_helix_plugin.projects.types import (
    CreateProjectRequest,
    ListProjectsQueryParams,
    Project,
    UpdateProjectRequest,
)

_PROJECTS = "/apis/entities/v2/workspaces/{workspace}/projects"


@get(f"{_PROJECTS}/{{name}}")
@abstractmethod
def get_project(*, workspace: str | None = None, name: str) -> Project: ...


@get(_PROJECTS)
@abstractmethod
def list_projects(
    *, workspace: str | None = None, query_params: ListProjectsQueryParams | None = None
) -> Paginated[Project]: ...


def _get_project_on_conflict(body: CreateProjectRequest, workspace: str | None) -> PreparedRequest[Project]:
    """Build the retrieve request replayed when ``create_project(exist_ok=True)`` 409s."""
    return get_project(name=body.name, workspace=workspace)


@post(_PROJECTS, get_on_conflict=_get_project_on_conflict)
@abstractmethod
def create_project(*, workspace: str | None = None, body: CreateProjectRequest, exist_ok: bool = False) -> Project: ...


@put(f"{_PROJECTS}/{{name}}")
@abstractmethod
def update_project(*, workspace: str | None = None, name: str, body: UpdateProjectRequest) -> Project: ...


@delete(f"{_PROJECTS}/{{name}}")
@abstractmethod
def delete_project(*, workspace: str | None = None, name: str) -> DeleteResponse: ...
