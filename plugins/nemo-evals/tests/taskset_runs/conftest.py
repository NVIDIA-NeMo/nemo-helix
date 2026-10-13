# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved, network-free inputs for the run-record contract."""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.jobs.agent_spec import AgentEvalSpec, FabricConfigSource, FabricRunnerTarget, ResolvedTask
from nemo_evals.jobs.metric_resolution import to_inline
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.hybrid import HybridMetricBundlePackager
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import RunAcceptance, TasksetRunStore
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


class FakeDownload:
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def read(self) -> bytes:
        return self.content


class FakeFiles(AsyncFilesClient):
    """Request-owned filesets with injectable failures and a two-upload barrier."""

    def __init__(self) -> None:
        self.filesets: dict[tuple[str, str], dict[str, bytes]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.errors: dict[str, Exception] = {}
        self.delete_errors: set[str] = set()
        self.upload_barrier: asyncio.Event | None = None
        self.upload_count = 0

    def _record(self, operation: str, workspace: str, name: str) -> None:
        self.calls.append((operation, workspace, name))
        if operation in self.errors:
            raise self.errors[operation]

    async def create_fileset(
        self, *, workspace: str | None = None, body: CreateFilesetRequest, exist_ok: bool = False
    ) -> object:
        workspace = workspace or "default"
        self._record("create", workspace, body.name)
        key = (workspace, body.name)
        if key in self.filesets:
            assert not exist_ok
            raise ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://files.invalid")))
        self.filesets[key] = {}
        return object()

    async def upload_file(self, *, workspace: str | None = None, name: str, path: str, content: bytes) -> object:
        workspace = workspace or "default"
        self._record("upload", workspace, name)
        self.filesets[(workspace, name)][path] = content
        self.upload_count += 1
        if self.upload_barrier is not None:
            if self.upload_count == 2:
                self.upload_barrier.set()
            await self.upload_barrier.wait()
        return object()

    async def download_file(self, *, workspace: str | None = None, name: str, path: str) -> FakeDownload:
        workspace = workspace or "default"
        self._record("download", workspace, name)
        return FakeDownload(self.filesets[(workspace, name)][path])

    async def delete_fileset(self, *, workspace: str | None = None, name: str) -> object:
        workspace = workspace or "default"
        self._record("delete", workspace, name)
        if name in self.delete_errors:
            raise RuntimeError(f"cannot delete {workspace}/{name}")
        self.filesets.pop((workspace, name), None)
        return object()


@pytest.fixture
def files_client() -> FakeFiles:
    return FakeFiles()


@pytest.fixture
def plan(resolved_spec: AgentEvalSpec) -> TasksetRunPlan:
    return TasksetRunPlan(
        run_id=derive_run_id("default", "user:alice", "key"),
        workspace="default",
        taskset_ref="default/suite#3",
        spec=resolved_spec,
        missing_prebuilt_image="fail",
        max_concurrent_task_jobs=None,
        max_failed_tasks=None,
    )


@pytest.fixture
def acceptance(plan: TasksetRunPlan) -> RunAcceptance:
    accepted_at = datetime(2026, 10, 5, tzinfo=timezone.utc)
    return RunAcceptance(
        plan=plan,
        idempotency_key="key",
        request_digest="request-digest",
        accepted_at=accepted_at,
        execution_deadline_at=accepted_at + timedelta(hours=1),
        finalization_deadline_at=accepted_at + timedelta(hours=2),
    )


@pytest.fixture
def store(entity_store, files_client: FakeFiles) -> TasksetRunStore:
    return TasksetRunStore(entity_store, files_client)


@pytest.fixture
def resolved_spec() -> AgentEvalSpec:
    return AgentEvalSpec(
        tasks=[
            ResolvedTask(
                id="task-1",
                spec=ResolvedEvaluatorTaskDefinition(
                    kind="evaluator",
                    intent="Answer exactly.",
                    metrics=[
                        to_inline(
                            bundle_metric(
                                ExactMatchMetric(reference="yes", candidate="yes"), HybridMetricBundlePackager()
                            )
                        )
                    ],
                ),
            )
        ],
        target=FabricRunnerTarget(source=FabricConfigSource(config={}, model="provider/model")),
    )
