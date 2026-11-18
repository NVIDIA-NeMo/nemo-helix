# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a Gym dataset file in a FileSet into agent-eval tasks at submission, and optionally publish them.

:func:`load_task_inputs` reads the file as the submitter, within a byte cap, and converts its rows
with ``discover_gym_tasks`` into ordinary inline task inputs, which then pass every check any inline
task passes. When the source names a snapshot taskset, :func:`publish_snapshot` runs after those
checks and, on the job route, after compilation too: it writes each changed task, then a taskset revision pinning every member to the digest its
write returned, and pins the evaluation's tasks to those same revisions. Nothing is looked up
afterwards by name or "newest", so a concurrent submission cannot swap another file's rows in.

Writes go through the submitter's request-scoped client, so the platform authorizes each as the
submitter. ``authorize``, which the job route supplies, additionally checks the file read and the
writes with the submitter's token scopes.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from collections.abc import Sequence
from functools import cache
from urllib.parse import quote

from filesets import parse_fileset_ref
from nemo_evals.api.fields import MetadataItem, MetricInline, TaskInputs
from nemo_evals.api.schemas import (
    EvaluatorTaskDefinition,
    TaskInput,
    TaskRef,
    TasksetInput,
    parse_subentity_ref,
)
from nemo_evals.api.service.task_service import stored_evaluator_task_digest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
from nemo_evals.harbor.resolution import map_with_limited_concurrency
from nemo_evals.jobs.agent_spec import AgentEvalTaskInput, FilesetTaskSource, ResolvedTask
from nemo_evals.sdk.task_resources import AsyncEvaluatorTasksResource
from nemo_evals.sdk.taskset_resources import AsyncEvaluatorTasksetsResource
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evals.source_authorization import SourceReadAuthorizer
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import (
    ConflictError,
    NemoTransportError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx_evals_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks

logger = logging.getLogger(__name__)

#: Largest dataset file a submission may name. It is read and converted inside the submit request.
MAX_BYTES = 64 * 1024 * 1024
#: Most rows a dataset file may hold: each becomes a task the submission validates and freezes.
MAX_ROWS = 50_000
#: Bytes read per chunk while streaming the dataset file.
_STREAM_CHUNK_BYTES = 1 << 20
#: Task writes in flight at once. Bounded so one large file can't flood the evals service with itself.
_WRITE_CONCURRENCY = 16


@cache
def gym_reward_metric() -> MetricInline:
    """The Gym reward as one inline bundle, built once and shared by every task a file yields."""
    return MetricInline.model_validate(
        bundle_metric(GymRewardMetric(), InlineMetricBundlePackager()).model_dump(mode="json")
    )


async def _read_file(
    source: FilesetTaskSource, *, workspace: str, async_client: AsyncHelixClient, authorize: SourceReadAuthorizer | None
) -> tuple[str, bytes]:
    """The file's qualified reference and bytes, read as the submitter within :data:`MAX_BYTES`."""
    fs_workspace, fs_name, path = parse_fileset_ref(source.fileset.root, workspace_fallback=workspace)
    qualified = f"{fs_workspace}/{fs_name}#{path}"
    if authorize is not None:
        await authorize(
            "GET",
            f"/apis/files/v2/workspaces/{quote(fs_workspace, safe='')}/filesets/{quote(fs_name, safe='')}"
            f"/-/{quote(path)}",
        )
    files = client_from_platform(async_client, AsyncFilesClient)
    body = bytearray()
    try:
        response = await files.download_file(workspace=fs_workspace, name=fs_name, path=path)
        async with response.stream(chunk_size=_STREAM_CHUNK_BYTES) as chunks:
            async for chunk in chunks:
                if len(body) + len(chunk) > MAX_BYTES:
                    raise ValueError(f"dataset file {qualified} is larger than the {MAX_BYTES}-byte limit")
                body.extend(chunk)
    except PermissionDeniedError as exc:
        raise PermissionError(f"access denied to dataset file {qualified}") from exc
    except NotFoundError as exc:
        raise ValueError(f"dataset file {qualified} does not exist") from exc
    return qualified, bytes(body)


def _task_inputs(source: FilesetTaskSource, qualified: str, body: bytes) -> list[AgentEvalTaskInput]:
    """Convert the file's rows to inline task inputs, each scored by the shared Gym reward bundle."""
    tasks = discover_gym_tasks(
        io.BytesIO(body),
        name=qualified,
        metrics=[],
        max_rows=MAX_ROWS,
        id_field=source.id_field,
        id_prefix=source.id_prefix,
    )
    reward = gym_reward_metric()
    return [
        AgentEvalTaskInput(
            id=task.id,
            intent=task.intent,
            inputs=TaskInputs(**task.inputs),
            metadata=[MetadataItem(key=key, value=value) for key, value in task.metadata.items()],
            metrics=[reward],
        )
        for task in tasks
    ]


async def load_task_inputs(
    source: FilesetTaskSource,
    *,
    workspace: str,
    async_client: AsyncHelixClient,
    authorize: SourceReadAuthorizer | None,
) -> tuple[str, list[AgentEvalTaskInput]]:
    """Read the source file and convert it off the event loop: its SHA-256 and its task inputs.

    Raises:
        PermissionError: The submitter may not read the file.
        ValueError: The file is missing, too large, has too many rows, or has a bad row.
    """
    qualified, body = await _read_file(source, workspace=workspace, async_client=async_client, authorize=authorize)
    inputs = await asyncio.to_thread(_task_inputs, source, qualified, body)
    return hashlib.sha256(body).hexdigest(), inputs


async def publish_snapshot(
    source: FilesetTaskSource,
    inputs: Sequence[AgentEvalTaskInput],
    resolved: Sequence[ResolvedTask],
    *,
    source_sha256: str,
    workspace: str,
    async_client: AsyncHelixClient,
    authorize: SourceReadAuthorizer | None,
) -> list[ResolvedTask]:
    """Publish ``inputs`` as tasks and a taskset revision, and return ``resolved`` pinned to them.

    Call only after every submission check has passed on ``resolved``, which must be ``inputs``
    resolved inline, in the same order: this is the submission's first write.

    Raises:
        PermissionError: The submitter may not read the taskset, or write it or its tasks.
        ValueError: The writes kept conflicting, or the evals service did not respond.
    """
    assert source.snapshot is not None
    ts_workspace, ts_name, _ = parse_subentity_ref(source.snapshot.root, workspace)
    taskset_path = f"/apis/evals/v2/workspaces/{quote(ts_workspace, safe='')}/tasksets/{quote(ts_name, safe='')}"
    if authorize is not None:
        # One representative task path: the platform authorizes task writes per workspace collection.
        await authorize(
            "PUT", f"/apis/evals/v2/workspaces/{quote(ts_workspace, safe='')}/tasks/{quote(inputs[0].id, safe='')}"
        )
        # Reading the current revision is how unchanged rows are skipped, so a snapshot needs both.
        await authorize("GET", taskset_path)
        await authorize("PUT", taskset_path)
    client = client_from_platform(async_client, AsyncEvaluatorClient)
    tasks_resource = AsyncEvaluatorTasksResource(client)
    tasksets_resource = AsyncEvaluatorTasksetsResource(client)
    try:
        try:
            previous = await tasksets_resource.retrieve(ts_name, workspace=ts_workspace)
        except NotFoundError:
            previous = None
        pinned_before: dict[tuple[str, str], str | None] = {}
        for ref in previous.tasks if previous is not None else []:
            member_workspace, member_name, digest = parse_subentity_ref(ref.root, ts_workspace)
            pinned_before[(member_workspace, member_name)] = digest

        bodies = [_task_body(task) for task in inputs]
        # The digest each task would publish under, computed exactly as the task service would. A row the
        # previous revision already pins at that digest is not written again: an unchanged file costs no
        # task writes, and an edit costs one write per edited row.
        expected = await asyncio.to_thread(
            lambda: [stored_evaluator_task_digest(body, workspace=ts_workspace) for body in bodies]
        )

        reused: list[int] = []

        async def put_task(index: int, *, write: bool = False) -> str:
            task_id = inputs[index].id
            if not write and pinned_before.get((ts_workspace, task_id)) == expected[index]:
                reused.append(index)
                return expected[index]
            stored = await tasks_resource.replace(task_id, task=bodies[index], workspace=ts_workspace)
            if stored.content_hash is None:
                raise RuntimeError("the evals service returned no content_hash for a task write")
            return stored.content_hash

        digests = await map_with_limited_concurrency(put_task, list(range(len(inputs))), concurrency=_WRITE_CONCURRENCY)

        async def publish() -> None:
            # The service dedupes: a taskset identical to its current revision publishes nothing new.
            await tasksets_resource.replace(
                ts_name,
                taskset=TasksetInput(
                    description=previous.description if previous is not None else None,
                    tasks=[TaskRef(f"{ts_workspace}/{t.id}#{d}") for t, d in zip(inputs, digests, strict=True)],
                    metadata=[
                        MetadataItem(key="source_fileset", value=source.fileset.root),
                        MetadataItem(key="source_sha256", value=source_sha256),
                    ],
                    tags=[f"src-{source_sha256}"],
                ),
                workspace=ts_workspace,
            )

        try:
            await publish()
        except UnprocessableEntityError:
            if not reused:
                raise
            # A reused pin can name a task deleted since the previous revision, which the service refuses.
            # Write every reused row (the service dedupes those still stored, so they gain no revision)
            # and publish once more. Only this rare case pays for the writes the skip saves.
            logger.info(
                "taskset %s/%s pins a deleted task; rewriting %d reused rows", ts_workspace, ts_name, len(reused)
            )
            rewritten = await map_with_limited_concurrency(
                lambda index: put_task(index, write=True), reused, concurrency=_WRITE_CONCURRENCY
            )
            for index, digest in zip(reused, rewritten, strict=True):
                digests[index] = digest
            await publish()
    except PermissionDeniedError as exc:
        raise PermissionError(
            f"publishing taskset {ts_workspace}/{ts_name} needs read and write access to its tasks and tasksets"
        ) from exc
    except ConflictError as exc:
        raise ValueError(
            f"publishing taskset {ts_workspace}/{ts_name} kept conflicting with concurrent writes; resubmit"
        ) from exc
    except NemoTransportError as exc:
        # A timeout's own message is often empty, which would otherwise surface as a blank 422.
        raise ValueError(
            f"publishing taskset {ts_workspace}/{ts_name} failed: the evals service did not respond "
            f"({type(exc.__cause__ or exc).__name__})"
        ) from exc
    return [
        _with_provenance(task, entity_name=f"{ts_workspace}/{task.id}", digest=digest)
        for task, digest in zip(resolved, digests, strict=True)
    ]


def _task_body(task: AgentEvalTaskInput) -> TaskInput:
    """The stored-task write for one converted row."""
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=task.intent,
            inputs=task.inputs,
            reference=task.reference,
            views=task.views,
            metrics=list(task.metrics),
        ),
        metadata=task.metadata,
    )


def _with_provenance(task: ResolvedTask, *, entity_name: str, digest: str) -> ResolvedTask:
    """``task`` as a snapshot of the stored revision ``digest`` of ``entity_name``.

    The stored task holds this content, with its inline reward moved to a content-addressed derived
    metric that resolves to the same bundle. The frozen task therefore matches what loading the pinned
    revision would produce, without reloading it or fetching the bundle once per task.
    """
    spec = task.spec
    if not isinstance(spec, ResolvedEvaluatorTaskDefinition):
        raise ValueError("a FileSet task source only yields evaluator tasks")
    pinned = spec.model_copy(update={"provenance": TaskProvenance(entity_name=entity_name, revision_digest=digest)})
    return ResolvedTask.model_validate({**task.model_dump(mode="json"), "spec": pinned.model_dump(mode="json")})
