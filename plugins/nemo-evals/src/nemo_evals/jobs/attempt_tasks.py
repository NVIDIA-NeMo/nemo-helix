# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep an agent evaluation's frozen tasks out of its step config.

A step config reaches a Kubernetes pod as a ConfigMap, which Kubernetes caps at 1 MiB, and an agent
evaluation's frozen tasks grow with the task count. So the compiler leaves ``tasks`` and ``trials`` out
of the agent-evaluate step config, marks it with :data:`TASKS_FROM_KEY`, and pins a digest of the two
under :data:`TASKS_DIGEST_KEY`. The worker reads them back from its own job attempt, which keeps an
immutable copy of the spec the step config was compiled from, and refuses to run when the digest
differs.

The digest covers the wire JSON exactly as the job stores it, not re-validated models: free-form fields
(a task's ``reference``, a trial's ``metadata``) would otherwise hash differently on each side, and the
digest would depend on the worker's model defaults.

An attempt's spec holds the tasks in one of two shapes (:class:`StoredAttemptSpec`). Jobs stored with their
tasks inline carry ``tasks`` (and ``trials``) in the spec. Jobs stored with their tasks frozen in a file carry
``frozen_tasks`` instead (:class:`~nemo_evals.jobs.agent_spec.FrozenTasksRef`), naming that file and the digest of
its exact bytes; the worker downloads it and checks those bytes against the step config's digest before parsing
them.

The job route freezes a new job's tasks in a file (:func:`freeze_tasks_in_file`): ``tasks.json`` in a new
fileset of the job's workspace with a random name, which the job's replies show. Helix never deletes that
fileset; deleting it is left to the user.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from functools import partial
from typing import Any, Final

from filesets import parse_fileset_ref
from nemo_evals.jobs.agent_spec import AgentEvalOutputSpec, AgentEvalSpec, FrozenTasksRef
from nemo_evals.taskset_runs.plan_storage import (
    PlanDigestMismatchError,
    delete_fileset_ref,
    read_verified_bytes_sync,
    store_in_new_fileset,
)
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import StoredJobSpec
from pydantic import BaseModel, ConfigDict, ValidationError

TASKS_FROM_KEY: Final = "tasks_from"
TASKS_DIGEST_KEY: Final = "tasks_digest"
TASKS_FROM_ATTEMPT: Final = "attempt"
_DIGEST_PREFIX: Final = "sha256-v1:"
_FROZEN_KEYS: Final = ("tasks", "trials")
#: The path of the file that holds a job's frozen tasks and trials, inside its fileset.
FROZEN_TASKS_PATH: Final = "tasks.json"
#: What names that fileset, before its random suffix.
FROZEN_TASKS_FILESET_PREFIX: Final = "job-tasks-"

#: The service the agent-evaluate task runs as (see :mod:`nemo_evals.tasks.agent_evaluate`).
TASK_SERVICE_NAME: Final = "evals"


class AttemptTasksError(RuntimeError):
    """The worker could not load its frozen tasks from its job attempt, or they don't match the step config."""


class FrozenTasks(BaseModel):
    """A job's frozen tasks and trials: what its tasks file holds, and what the worker restores to its step config.

    Both stay raw JSON, because the digest covers them exactly as stored.
    """

    tasks: list[Any]
    trials: list[Any] | None = None


class StoredAttemptSpec(BaseModel):
    """The parts of an attempt's stored spec the worker reads: its tasks, or the file that holds them."""

    model_config = ConfigDict(extra="ignore")

    frozen_tasks: FrozenTasksRef | None = None
    # Raw, as in :class:`FrozenTasks`: the digest covers the stored JSON.
    tasks: list[Any] | None = None
    trials: list[Any] | None = None


def frozen_tasks_bytes(spec: dict) -> bytes:
    """``spec``'s ``tasks`` and ``trials`` as canonical JSON bytes: what the digest covers, and what a frozen-tasks file holds."""
    frozen = {key: spec.get(key) for key in _FROZEN_KEYS}
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return canonical.encode("utf-8")


def frozen_tasks_digest(spec: dict) -> str:
    """The versioned digest of ``spec``'s ``tasks`` and ``trials``, as JSON (``model_dump(mode="json")``).

    It is ``sha256-v1:`` and the SHA-256 of :func:`frozen_tasks_bytes`, so it binds a frozen-tasks file whose
    ``digest`` is that SHA-256.
    """
    return _DIGEST_PREFIX + hashlib.sha256(frozen_tasks_bytes(spec)).hexdigest()


async def freeze_tasks_in_file(
    spec: AgentEvalSpec, *, workspace: str, job_name: str | None, async_client: AsyncNemoClient
) -> StoredJobSpec:
    """Write ``spec``'s tasks and trials to a file, and return the spec a new job stores in their place.

    The file is :data:`FROZEN_TASKS_PATH` in a new fileset named :data:`FROZEN_TASKS_FILESET_PREFIX` and a random
    suffix, and holds :func:`frozen_tasks_bytes`, the bytes the step config's digest covers. The stored spec is the
    job's settings, its task count and that file (:meth:`AgentEvalOutputSpec.keeping_tasks_in`). Its ``discard``
    deletes the fileset, for when the Jobs service then refuses the job.

    Args:
        spec: The canonical spec, as the step config was compiled from.
        workspace: The job's workspace, where the fileset goes.
        job_name: The job's name, for the fileset's description; ``None`` when the Jobs service will name it.
        async_client: The request-scoped client; the fileset is created as the caller.

    Raises:
        PlanStoragePermissionDeniedError: Files refused the fileset or the upload.
        PlanStorageError: Creating the fileset or uploading the file failed.
    """
    files_client = AsyncFilesClient.from_client(async_client)
    job = (
        f"agent-evaluate job {workspace}/{job_name}"
        if job_name is not None
        else f"an agent-evaluate job in {workspace}"
    )
    stored = await store_in_new_fileset(
        files_client,
        workspace=workspace,
        # Nothing else in the name tells two jobs' filesets apart, so the suffix carries 64 random bits.
        name=f"{FROZEN_TASKS_FILESET_PREFIX}{secrets.token_hex(8)}",
        path=FROZEN_TASKS_PATH,
        # The same JSON the step config's digest was computed from, so the file's bytes match it.
        body=frozen_tasks_bytes(spec.model_dump(mode="json", include=set(_FROZEN_KEYS))),
        description=f"Tasks frozen for {job}.",
        what="frozen tasks",
    )
    return StoredJobSpec(
        spec=AgentEvalOutputSpec.keeping_tasks_in(spec, FrozenTasksRef(ref=stored.ref, digest=stored.digest)),
        discard=partial(delete_fileset_ref, files_client, stored.ref, fileset_id=stored.fileset_id),
    )


def step_config_without_tasks(spec: dict) -> dict:
    """The agent-evaluate step config for ``spec``: everything but ``tasks``/``trials``, plus the marker and digest."""
    config = {key: value for key, value in spec.items() if key not in _FROZEN_KEYS}
    config[TASKS_FROM_KEY] = TASKS_FROM_ATTEMPT
    config[TASKS_DIGEST_KEY] = frozen_tasks_digest(spec)
    return config


def load_attempt_tasks(config: dict, *, ctx: JobContext, client: NemoClient | None = None) -> dict:
    """``config`` with its frozen tasks and trials restored from the job attempt it runs in.

    A config without :data:`TASKS_FROM_KEY` (a local run, a taskset-run task Job, a job compiled before
    the tasks moved out) is returned unchanged, and nothing is read.

    ``client`` reads the attempt. Without one (the async worker), the platform's client provider builds a sync
    task client for the read, the same way the task runner builds the worker's clients, so the read uses the
    deployment's CA bundle, transport and routing.

    Raises:
        AttemptTasksError: The config names an unknown source or carries no digest; the run has no job
            attempt; the read fails; or the attempt's tasks don't match the digest.
    """
    if TASKS_FROM_KEY not in config:
        return config
    rest = dict(config)
    source = rest.pop(TASKS_FROM_KEY)
    expected = rest.pop(TASKS_DIGEST_KEY, None)
    if source != TASKS_FROM_ATTEMPT:
        raise AttemptTasksError(f"step config names an unknown task source {source!r}; expected {TASKS_FROM_ATTEMPT!r}")
    if not isinstance(expected, str) or not expected.startswith(_DIGEST_PREFIX):
        raise AttemptTasksError(f"step config's {TASKS_DIGEST_KEY} is missing or not a {_DIGEST_PREFIX} digest")
    if ctx.job_id is None or ctx.attempt_id is None:
        raise AttemptTasksError("step config's tasks are on its job attempt, but this run has no job attempt")
    job, attempt_id = ctx.job_id, ctx.attempt_id
    try:
        if client is not None:
            frozen = _frozen_tasks(client, workspace=ctx.workspace, job=job, attempt=attempt_id, expected=expected)
        else:
            with get_task_nemo_client(TASK_SERVICE_NAME, workspace=ctx.workspace) as task_client:
                frozen = _frozen_tasks(
                    task_client, workspace=ctx.workspace, job=job, attempt=attempt_id, expected=expected
                )
    except AttemptTasksError:
        raise
    except Exception as exc:
        raise AttemptTasksError(f"could not read the tasks of job {job!r} from its attempt {attempt_id!r}") from exc
    rest.update(frozen.model_dump())
    return rest


def _frozen_tasks(client: NemoClient, *, workspace: str, job: str, attempt: str, expected: str) -> FrozenTasks:
    """The attempt's tasks and trials, from its spec or from the file its spec names, checked against ``expected``."""
    spec = _attempt_spec(client, workspace=workspace, job=job, attempt=attempt)
    if spec.frozen_tasks is not None:
        return _read_frozen_tasks_file(client, spec.frozen_tasks, workspace=workspace, expected=expected)
    if spec.tasks is None:
        raise AttemptTasksError(f"attempt {attempt!r} of job {job!r} has no task list in its spec")
    frozen = FrozenTasks(tasks=spec.tasks, trials=spec.trials)
    actual = frozen_tasks_digest(frozen.model_dump())
    if actual != expected:
        raise AttemptTasksError(
            f"the tasks on attempt {attempt!r} of job {job!r} don't match its step config "
            f"(expected {expected}, got {actual}); refusing to run them"
        )
    return frozen


def _read_frozen_tasks_file(
    client: NemoClient, frozen_tasks: FrozenTasksRef, *, workspace: str, expected: str
) -> FrozenTasks:
    """Download the file ``frozen_tasks`` names, check its exact bytes against ``expected``, then parse it.

    The file is stored and read the way a taskset run's plan is (:mod:`nemo_evals.taskset_runs.plan_storage`).
    """
    ref = frozen_tasks.ref
    if _DIGEST_PREFIX + frozen_tasks.digest != expected:
        raise AttemptTasksError(
            f"the attempt's frozen tasks digest {frozen_tasks.digest!r} doesn't match its step config "
            f"({expected}); refusing to run it"
        )
    try:
        file_workspace, _, _ = parse_fileset_ref(ref, workspace_fallback=None)
    except ValueError as exc:
        raise AttemptTasksError(f"the attempt's frozen tasks file {ref!r} is not a fileset file ref") from exc
    if file_workspace != workspace:
        raise AttemptTasksError(f"the attempt's frozen tasks file {ref!r} is outside workspace {workspace!r}")
    try:
        body = read_verified_bytes_sync(FilesClient.from_client(client), ref, expected_digest=frozen_tasks.digest)
    except PlanDigestMismatchError as exc:
        raise AttemptTasksError(
            f"the frozen tasks file {ref!r} doesn't match its step config; refusing to run it"
        ) from exc
    try:
        return FrozenTasks.model_validate_json(body)
    except ValidationError as exc:
        raise AttemptTasksError(f"the frozen tasks file {ref!r} doesn't hold a task list") from exc


def _attempt_spec(client: NemoClient, *, workspace: str, job: str, attempt: str) -> StoredAttemptSpec:
    """The parts of the attempt's stored spec the worker reads: its tasks, or the pointer to their file."""
    spec = JobsClient.from_client(client).get_job_attempt(workspace=workspace, job=job, attempt=attempt).data().spec
    try:
        return StoredAttemptSpec.model_validate(spec)
    except ValidationError as exc:
        raise AttemptTasksError(
            f"attempt {attempt!r} of job {job!r} has neither a task list nor a tasks file's ref and digest in its spec"
        ) from exc


__all__ = [
    "FROZEN_TASKS_FILESET_PREFIX",
    "FROZEN_TASKS_PATH",
    "TASKS_DIGEST_KEY",
    "TASKS_FROM_ATTEMPT",
    "TASKS_FROM_KEY",
    "TASK_SERVICE_NAME",
    "AttemptTasksError",
    "FrozenTasks",
    "StoredAttemptSpec",
    "frozen_tasks_bytes",
    "freeze_tasks_in_file",
    "frozen_tasks_digest",
    "load_attempt_tasks",
    "step_config_without_tasks",
]
