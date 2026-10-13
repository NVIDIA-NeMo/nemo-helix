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
    """Wrap supplied file bytes in an async read response without network access.

    ``content`` is the stored payload; ``read`` returns those exact bytes so
    integrity tests exercise the same response boundary as the Files client.
    """

    def __init__(self, content: bytes) -> None:
        self.content = content

    async def read(self) -> bytes:
        return self.content


class FakeFiles(AsyncFilesClient):
    """Model Files ownership and failures for run-acceptance tests without HTTP.

    Tests populate ``filesets`` by (workspace, name) and configure ``errors`` or
    ``delete_errors`` to inject failures. Operations record attempted calls and
    return opaque write responses or a readable download. An optional upload
    barrier makes two submissions reach entity creation with overlapping plans.
    """

    def __init__(self) -> None:
        """Create isolated empty storage, call history, and disabled fault controls.

        No arguments or underlying HTTP client are needed. Tests receive mutable
        maps and an optional barrier to arrange storage races and cleanup failures.
        """
        self.filesets: dict[tuple[str, str], dict[str, bytes]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.errors: dict[str, Exception] = {}
        self.delete_errors: set[str] = set()
        self.upload_barrier: asyncio.Event | None = None
        self.upload_count = 0

    def _record(self, operation: str, workspace: str, name: str) -> None:
        """Record an attempted operation on a named fileset, then raise its injected error."""
        self.calls.append((operation, workspace, name))
        if operation in self.errors:
            raise self.errors[operation]

    async def create_fileset(
        self, *, workspace: str | None = None, body: CreateFilesetRequest, exist_ok: bool = False
    ) -> object:
        """Create an empty fileset while reproducing exclusive-create conflicts.

        Args:
            workspace: Storage namespace, defaulting to ``default`` when absent.
            body: Request supplying the new fileset name.
            exist_ok: Must be false on a collision, as required by plan ownership.

        Returns:
            An opaque success placeholder after adding the empty fileset.

        Raises:
            ConflictError: The workspace/name already exists.
            Exception: A configured create failure, before modifying storage.
        """
        workspace = workspace or "default"
        self._record("create", workspace, body.name)
        key = (workspace, body.name)
        if key in self.filesets:
            assert not exist_ok
            raise ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://files.invalid")))
        self.filesets[key] = {}
        return object()

    async def upload_file(self, *, workspace: str | None = None, name: str, path: str, content: bytes) -> object:
        """Store bytes and optionally gate two uploads to exercise acceptance races.

        Args:
            workspace: Namespace, defaulting to ``default`` when absent.
            name: Existing fileset to receive the upload.
            path: File key within the fileset.
            content: Exact bytes retained for later download and tampering tests.

        Returns:
            An opaque success placeholder after storage and any configured barrier.
            Configured errors propagate before storage; a missing fileset raises
            KeyError. The barrier opens once two uploads have stored their bytes.
        """
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
        """Return a readable response for the named workspace/file, recording the read.

        An absent workspace uses ``default``. Missing files raise KeyError and
        configured download failures propagate, letting tests detect cached reads.
        """
        workspace = workspace or "default"
        self._record("download", workspace, name)
        return FakeDownload(self.filesets[(workspace, name)][path])

    async def delete_fileset(self, *, workspace: str | None = None, name: str) -> object:
        """Remove a whole named fileset, or inject a failure to test best-effort cleanup.

        Args:
            workspace: Namespace, defaulting to ``default`` when absent.
            name: Fileset to remove from in-memory storage.

        Returns:
            An opaque success placeholder, including when the fileset is absent.

        Raises:
            RuntimeError: The name is configured in ``delete_errors``.
            Exception: An operation-wide delete failure configured in ``errors``.
        """
        workspace = workspace or "default"
        self._record("delete", workspace, name)
        if name in self.delete_errors:
            raise RuntimeError(f"cannot delete {workspace}/{name}")
        self.filesets.pop((workspace, name), None)
        return object()


@pytest.fixture
def files_client() -> FakeFiles:
    """Return fresh isolated Files storage for each test's resource assertions."""
    return FakeFiles()


@pytest.fixture
def plan(resolved_spec: AgentEvalSpec) -> TasksetRunPlan:
    """Bind the resolved-spec fixture to a fixed keyed run and pinned taskset revision.

    The returned plan has explicit unlimited caps and a fail-on-missing-image
    policy, providing deterministic input for serialization and acceptance tests.
    """
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
    """Return resolved acceptance inputs for the plan fixture with fixed UTC deadlines.

    Execution and finalization end one and two hours after acceptance, allowing
    retries and races to prove that the winning request's times are preserved.
    """
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
    """Return a store over the entity and Files fixtures without production retries."""
    return TasksetRunStore._with_entity_adapter(entity_store, files_client)


@pytest.fixture
def resolved_spec() -> AgentEvalSpec:
    """Build a fresh fully resolved evaluation spec without external services.

    No fixture inputs are needed. The returned spec contains one evaluator task,
    an inline exact-match metric bundle, and a Fabric target so plan validation
    exercises real nested models without metric or target resolution requests.
    """
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
