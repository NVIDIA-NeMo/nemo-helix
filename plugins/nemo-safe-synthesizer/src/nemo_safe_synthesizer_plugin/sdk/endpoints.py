# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Safe Synthesizer plugin API."""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.client.types import Paginated
from nemo_safe_synthesizer_plugin.sdk.types import (
    CreateSafeSynthesizerJobRequest,
    ListSafeSynthesizerJobsQueryParams,
    SafeSynthesizerJobResponse,
)

_JOBS = "/apis/safe-synthesizer/v2/workspaces/{workspace}/jobs"


@post(_JOBS)
@abstractmethod
def create_job(
    *, workspace: str | None = None, body: CreateSafeSynthesizerJobRequest
) -> SafeSynthesizerJobResponse: ...


@get(_JOBS)
@abstractmethod
def list_jobs(
    *, workspace: str | None = None, query_params: ListSafeSynthesizerJobsQueryParams | None = None
) -> Paginated[SafeSynthesizerJobResponse]: ...


@get(_JOBS + "/{name}")
@abstractmethod
def get_job(*, workspace: str | None = None, name: str) -> SafeSynthesizerJobResponse: ...
