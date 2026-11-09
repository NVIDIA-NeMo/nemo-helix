# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a dataset file in a FileSet into agent-eval tasks at submission, and optionally publish them.

Submission does three things for a :class:`~nemo_evals.jobs.agent_spec.FilesetTaskSource`:

1. :func:`read_source_file` reads the file as the submitter, within a byte cap, hashing what it reads.
2. :func:`build_task_inputs` converts its rows with ``discover_gym_tasks`` (off the event loop) into
   ordinary inline task inputs, which then pass every check any inline task passes.
3. When snapshotting, :func:`publish_snapshot` runs after those checks. It writes each task, then a
   taskset revision that pins each member to the digest its write returned, and pins the evaluation
   to the taskset digest the write returned. Nothing is looked up afterwards by name, tag, or
   "newest", so a concurrent submission cannot swap another file's rows into this one.

Writes go through the submitter's request-scoped client, so the platform authorizes each as the
submitter; ``authorize``, when the caller supplies one, additionally checks them with the submitter's
token scopes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from filesets import parse_fileset_ref
from nemo_evals.api.fields import MetadataItem, MetricInline, TaskInputs
from nemo_evals.api.schemas import (
    EvaluatorTaskDefinition,
    TaskInput,
    TaskRef,
    Taskset,
    TasksetInput,
    TasksetRef,
    parse_subentity_ref,
)
from nemo_evals.api.service.task_service import stored_evaluator_task_digest
from nemo_evals.api.task_definitions.evaluator import ResolvedEvaluatorTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
from nemo_evals.filesets import FilesetRef
from nemo_evals.harbor.resolution import map_with_limited_concurrency
from nemo_evals.jobs.agent_spec import (
    AgentEvalTaskInput,
    FilesetTaskSource,
    ResolvedTask,
    TaskSnapshotRecord,
    TaskSnapshotSpec,
    TaskSourceDelta,
    TaskSourceRecord,
)
from nemo_evals.sdk.task_resources import AsyncEvaluatorTasksResource
from nemo_evals.sdk.taskset_resources import AsyncEvaluatorTasksetsResource
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evals.source_authorization import SourceReadAuthorizer
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx_evals_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks

logger = logging.getLogger(__name__)

#: Bytes read per chunk while streaming the dataset file.
_STREAM_CHUNK_BYTES = 1 << 20
#: Task writes in flight at once. Bounded so one large file can't flood the evals service with itself.
_WRITE_CONCURRENCY = 16
#: Attempts per write that loses a concurrent-write race (409) before the submission fails.
_WRITE_ATTEMPTS = 3
#: Task metadata a local discovery stamps but a stored or frozen task must never carry: a temporary path.
_LOCAL_ONLY_METADATA = frozenset({"gym_dataset_path"})
_TASK_NAME = re.compile(NAME_PATTERN)


@dataclass(frozen=True)
class SourceFile:
    """The dataset file as read: its fully qualified reference, its bytes, and their SHA-256."""

    fileset: FilesetRef
    body: bytes
    sha256: str


@cache
def gym_reward_metric() -> MetricInline:
    """The Gym reward as one inline bundle, built once and shared by every task a file yields."""
    return MetricInline.model_validate(
        bundle_metric(GymRewardMetric(), InlineMetricBundlePackager()).model_dump(mode="json")
    )


async def read_source_file(
    source: FilesetTaskSource,
    *,
    workspace: str,
    async_client: AsyncHelixClient,
    authorize: SourceReadAuthorizer | None,
    max_bytes: int,
) -> SourceFile:
    """Read the source's dataset file as the submitter, refusing it once it passes ``max_bytes``.

    Raises:
        PermissionError: The submitter may not read the file.
        ValueError: The file does not exist or is larger than ``max_bytes``.
    """
    fs_workspace, fs_name, path = parse_fileset_ref(source.source.fileset.root, workspace_fallback=workspace)
    qualified = f"{fs_workspace}/{fs_name}#{path}"
    if authorize is not None:
        await authorize(
            "GET",
            f"/apis/files/v2/workspaces/{quote(fs_workspace, safe='')}/filesets/{quote(fs_name, safe='')}"
            f"/-/{quote(path)}",
        )
    files = client_from_platform(async_client, AsyncFilesClient)
    body = bytearray()
    digest = hashlib.sha256()
    try:
        response = await files.download_file(workspace=fs_workspace, name=fs_name, path=path)
        async with response.stream(chunk_size=_STREAM_CHUNK_BYTES) as chunks:
            async for chunk in chunks:
                if len(body) + len(chunk) > max_bytes:
                    raise ValueError(
                        f"dataset file {qualified} is larger than the {max_bytes}-byte limit "
                        "(evals config `fileset_task_source_max_bytes`)"
                    )
                body.extend(chunk)
                digest.update(chunk)
    except PermissionDeniedError as exc:
        raise PermissionError(f"access denied to dataset file {qualified}") from exc
    except NotFoundError as exc:
        raise ValueError(f"dataset file {qualified} does not exist") from exc
    return SourceFile(fileset=FilesetRef(qualified), body=bytes(body), sha256=digest.hexdigest())


def build_task_inputs(
    source_file: SourceFile,
    *,
    id_field: str | None,
    task_id_prefix: str | None,
    max_rows: int,
) -> list[AgentEvalTaskInput]:
    """Convert the file's rows to inline task inputs, each scored by the shared Gym reward bundle.

    Synchronous and CPU-bound: run it off the event loop. Errors name the line in the source file,
    never the temporary copy discovery reads.
    """
    rows = sum(1 for line in source_file.body.splitlines() if line.strip())
    if rows > max_rows:
        raise ValueError(
            f"dataset file {source_file.fileset.root} has {rows} rows, more than the {max_rows}-row limit "
            "(evals config `fileset_task_source_max_rows`)"
        )
    _, _, path = parse_fileset_ref(source_file.fileset.root, workspace_fallback="_unused")
    with tempfile.TemporaryDirectory(prefix="fileset-task-source-") as scratch:
        local = Path(scratch) / (PurePosixPath(path).name or "dataset.jsonl")
        local.write_bytes(source_file.body)
        try:
            tasks = discover_gym_tasks(local, metrics=[], id_field=id_field, task_id_prefix=task_id_prefix)
        except ValueError as exc:
            raise ValueError(str(exc).replace(str(local), source_file.fileset.root)) from exc
    reward = gym_reward_metric()
    return [
        AgentEvalTaskInput(
            id=task.id,
            intent=task.intent,
            inputs=TaskInputs(**task.inputs),
            metadata=[
                MetadataItem(key=key, value=value)
                for key, value in task.metadata.items()
                if key not in _LOCAL_ONLY_METADATA
            ],
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
    max_bytes: int,
    max_rows: int,
) -> tuple[SourceFile, list[AgentEvalTaskInput]]:
    """Read the source file and convert it to inline task inputs, converting off the event loop."""
    source_file = await read_source_file(
        source, workspace=workspace, async_client=async_client, authorize=authorize, max_bytes=max_bytes
    )
    task_id_prefix = source.snapshot.task_id_prefix if source.snapshot is not None else None
    inputs = await asyncio.to_thread(
        build_task_inputs,
        source_file,
        id_field=source.source.id_field,
        task_id_prefix=task_id_prefix,
        max_rows=max_rows,
    )
    return source_file, inputs


def task_source_record(source: FilesetTaskSource, source_file: SourceFile, task_count: int) -> TaskSourceRecord:
    """The provenance a run records for tasks read from a file, without a snapshot."""
    return TaskSourceRecord(
        fileset=source_file.fileset,
        format=source.source.format,
        file_sha256=source_file.sha256,
        task_count=task_count,
        id_field=source.source.id_field,
    )


def _require_task_names(inputs: Sequence[AgentEvalTaskInput]) -> None:
    """Reject task ids the entity store cannot hold as names, before anything is written."""
    for task in inputs:
        if len(task.id) > NAME_MAX_LENGTH or not _TASK_NAME.fullmatch(task.id):
            raise ValueError(
                f"task id {task.id!r} cannot be a stored task name. {NAME_PATTERN_DESCRIPTION} "
                "Change the row's id or the task_id_prefix."
            )


async def _previous_members(
    tasksets: AsyncEvaluatorTasksetsResource, name: str, workspace: str
) -> tuple[Taskset | None, dict[tuple[str, str], str]]:
    """The snapshot taskset's current head and its members as ``{(workspace, name): digest}``."""
    try:
        head = await tasksets.retrieve(name, workspace=workspace)
    except NotFoundError:
        return None, {}
    members: dict[tuple[str, str], str] = {}
    for ref in head.tasks:
        member_workspace, member_name, digest = parse_subentity_ref(ref.root, workspace)
        members[(member_workspace, member_name)] = digest or ""
    return head, members


async def _with_conflict_retries[T](write: Callable[[], Awaitable[T]], what: str) -> T:
    """Run a write, retrying one that lost a concurrent-write race; the write itself is idempotent."""
    for attempt in range(1, _WRITE_ATTEMPTS + 1):
        try:
            return await write()
        except ConflictError:
            if attempt == _WRITE_ATTEMPTS:
                raise ValueError(f"{what} kept conflicting with concurrent writes; resubmit") from None
        except NemoTransportError as exc:
            # A timeout's own message is often empty, which used to surface as a blank 422.
            cause = exc.__cause__ or exc
            raise ValueError(
                f"writing {what} failed: the evals service did not respond ({type(cause).__name__})"
            ) from exc
    raise AssertionError("unreachable")


async def publish_snapshot(
    snapshot: TaskSnapshotSpec,
    inputs: Sequence[AgentEvalTaskInput],
    resolved: Sequence[ResolvedTask],
    source_file: SourceFile,
    source: FilesetTaskSource,
    *,
    workspace: str,
    async_client: AsyncHelixClient,
    authorize: SourceReadAuthorizer | None,
) -> tuple[list[ResolvedTask], TaskSourceRecord]:
    """Publish ``inputs`` as tasks and a taskset revision, and pin ``resolved`` to what was published.

    Call only after every submission check has passed on ``resolved``: this is the first write.
    ``resolved`` must be ``inputs`` resolved inline, in the same order; each comes back carrying the
    provenance of the revision its write returned, so the frozen job runs exactly the published tasks.
    """
    if len(inputs) != len(resolved) or any(a.id != b.id for a, b in zip(inputs, resolved, strict=True)):
        raise ValueError("resolved tasks must be the snapshot's inputs, in order")
    _require_task_names(inputs)
    ts_workspace, ts_name, _ = parse_subentity_ref(snapshot.taskset.root, workspace)
    if authorize is not None:
        # One representative task path: the platform authorizes task writes per workspace collection.
        await authorize(
            "PUT", f"/apis/evals/v2/workspaces/{quote(ts_workspace, safe='')}/tasks/{quote(inputs[0].id, safe='')}"
        )
        await authorize(
            "PUT", f"/apis/evals/v2/workspaces/{quote(ts_workspace, safe='')}/tasksets/{quote(ts_name, safe='')}"
        )

    client = client_from_platform(async_client, AsyncEvaluatorClient)
    tasks_resource = AsyncEvaluatorTasksResource(client)
    tasksets_resource = AsyncEvaluatorTasksetsResource(client)
    previous_head, previous = await _previous_members(tasksets_resource, ts_name, ts_workspace)

    bodies = [_task_body(task) for task in inputs]
    # The digest each task would publish under, computed here exactly as the task service would.
    # A row whose digest the previous revision already pins is published, so it is not written again:
    # an unchanged file costs no task writes at all, and an edit costs one write per edited row.
    expected = await asyncio.to_thread(
        lambda: [stored_evaluator_task_digest(body, workspace=ts_workspace) for body in bodies]
    )

    async def put_task(index: int) -> str:
        task, body = inputs[index], bodies[index]
        if previous.get((ts_workspace, task.id)) == expected[index]:
            return expected[index]
        stored = await _with_conflict_retries(
            lambda: tasks_resource.replace(task.id, task=body, workspace=ts_workspace), f"task {task.id!r}"
        )
        if stored.content_hash is None:
            raise RuntimeError("the evals service returned no content_hash for a task write")
        if stored.content_hash != expected[index]:
            logger.warning(
                "Task %s published under digest %s, not the %s computed for it; unchanged rows will be "
                "rewritten until the two agree.",
                task.id,
                stored.content_hash,
                expected[index],
            )
        return stored.content_hash

    digests = await map_with_limited_concurrency(put_task, list(range(len(inputs))), concurrency=_WRITE_CONCURRENCY)

    metadata = [
        MetadataItem(key="source_fileset", value=source_file.fileset.root),
        MetadataItem(key="source_sha256", value=source_file.sha256),
        MetadataItem(key="format", value=source.source.format),
    ]
    if source.source.id_field is not None:
        metadata.append(MetadataItem(key="id_field", value=source.source.id_field))
        metadata.append(MetadataItem(key="task_id_prefix", value=snapshot.task_id_prefix))
    taskset_body = TasksetInput(
        # Keep a description someone gave the taskset; everything else is the file's.
        description=previous_head.description if previous_head is not None else None,
        tasks=[TaskRef(f"{ts_workspace}/{task.id}#{digest}") for task, digest in zip(inputs, digests, strict=True)],
        metadata=metadata,
        tags=[f"src-{source_file.sha256}"],
    )
    if _unchanged(previous_head, taskset_body):
        # Same members, metadata and description: the current revision already is this snapshot.
        assert previous_head is not None and previous_head.content_hash is not None
        published_digest, published_revision = previous_head.content_hash, previous_head.revision
    else:
        published = await _with_conflict_retries(
            lambda: tasksets_resource.replace(ts_name, taskset=taskset_body, workspace=ts_workspace),
            f"taskset {ts_workspace}/{ts_name}",
        )
        if published.content_hash is None:
            raise RuntimeError("the evals service returned no content_hash for a taskset write")
        published_digest, published_revision = published.content_hash, published.revision

    current = {(ts_workspace, task.id): digest for task, digest in zip(inputs, digests, strict=True)}
    delta = TaskSourceDelta(
        added=sum(1 for key in current if key not in previous),
        changed=sum(1 for key, digest in current.items() if key in previous and previous[key] != digest),
        removed=sum(1 for key in previous if key not in current),
        unchanged=sum(1 for key, digest in current.items() if previous.get(key) == digest),
    )
    pinned_tasks = [
        _with_provenance(task, entity_name=f"{ts_workspace}/{task.id}", digest=digest)
        for task, digest in zip(resolved, digests, strict=True)
    ]
    record = task_source_record(source, source_file, len(inputs)).model_copy(
        update={
            "snapshot": TaskSnapshotRecord(
                taskset=TasksetRef(f"{ts_workspace}/{ts_name}#{published_digest}"),
                taskset_revision=published_revision,
                task_id_prefix=snapshot.task_id_prefix,
                delta=delta,
            )
        }
    )
    return pinned_tasks, record


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


def _unchanged(previous: Taskset | None, body: TasksetInput) -> bool:
    """Whether writing ``body`` would publish nothing: its content equals the current revision's.

    Membership is compared as a set, as the service stores it. A server that predates
    ``content_hash`` gives no digest to pin, so the write goes ahead and returns one.
    """
    if previous is None or previous.content_hash is None:
        return False
    return (
        {ref.root for ref in previous.tasks} == {ref.root for ref in body.tasks}
        and [item.model_dump(mode="json") for item in previous.metadata]
        == [item.model_dump(mode="json") for item in body.metadata]
        and previous.description == body.description
        and previous.files_ref == body.files_ref
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
