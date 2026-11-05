# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for Kubernetes taskset runs (``…/agent-evaluate/runs``).

They live in the plugin, typed by ``taskset_run_schemas``, so the contract has one source of truth.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_evals.api.taskset_run_schemas import (
    TasksetRunInput,
    TasksetRunStatus,
    TasksetRunSubmitResponse,
    TasksetRunTask,
)
from nemo_evals.taskset_runs.entities import MemberPhase
from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.schema import Page
from typing_extensions import NotRequired, TypedDict

_RUNS = "/apis/evals/v2/workspaces/{workspace}/agent-evaluate/runs"


class ListRunsQueryParams(TypedDict, total=False):
    """Query of ``GET …/agent-evaluate/runs``."""

    active: NotRequired[bool]
    mine: NotRequired[bool]
    page: NotRequired[int]
    page_size: NotRequired[int]


class ListRunTasksQueryParams(TypedDict, total=False):
    """Query of ``GET …/agent-evaluate/runs/{run_id}/tasks``; the client drops a ``None`` phase."""

    phase: NotRequired[MemberPhase | None]
    page: NotRequired[int]
    page_size: NotRequired[int]


@post(_RUNS)
@abstractmethod
def submit_run(*, workspace: str | None = None, body: TasksetRunInput) -> TasksetRunSubmitResponse:
    """Submit a taskset run; a repeat with the same idempotency key and body returns the accepted run."""


@get(_RUNS)
@abstractmethod
def list_runs(
    *, workspace: str | None = None, query_params: ListRunsQueryParams | None = None
) -> Page[TasksetRunStatus]:
    """One page of the workspace's runs, newest first."""


@get(_RUNS + "/{run_id}")
@abstractmethod
def get_run(*, workspace: str | None = None, run_id: str) -> TasksetRunStatus:
    """One run's status."""


@post(_RUNS + "/{run_id}/cancel")
@abstractmethod
def cancel_run(*, workspace: str | None = None, run_id: str) -> TasksetRunStatus:
    """Request a run's cancel; the run controller acts on it."""


@get(_RUNS + "/{run_id}/tasks")
@abstractmethod
def list_run_tasks(
    *, workspace: str | None = None, run_id: str, query_params: ListRunTasksQueryParams | None = None
) -> Page[TasksetRunTask]:
    """One page of a run's task rows in task-index order."""
