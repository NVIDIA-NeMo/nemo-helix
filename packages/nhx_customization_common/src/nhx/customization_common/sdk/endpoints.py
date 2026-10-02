# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Customizer SDK routes."""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.client.types import Paginated
from nhx.customization_common.sdk.types import (
    CustomizationHealthResponse,
    CustomizationJob,
    CustomizationJobCreateRequest,
    ListCustomizationJobsQueryParams,
)


@get("/apis/customization/v2/healthz")
@abstractmethod
def get_customization_health() -> CustomizationHealthResponse: ...


@post("/apis/customization/v2/workspaces/{workspace}/{backend}/jobs")
@abstractmethod
def create_customization_job(
    *,
    workspace: str | None = None,
    backend: str,
    body: CustomizationJobCreateRequest,
) -> CustomizationJob: ...


@get("/apis/customization/v2/workspaces/{workspace}/{backend}/jobs")
@abstractmethod
def list_customization_jobs(
    *,
    workspace: str | None = None,
    backend: str,
    query_params: ListCustomizationJobsQueryParams | None = None,
) -> Paginated[CustomizationJob]: ...


@get("/apis/customization/v2/workspaces/{workspace}/{backend}/jobs/{name}")
@abstractmethod
def get_customization_job(
    *,
    workspace: str | None = None,
    backend: str,
    name: str,
) -> CustomizationJob: ...
