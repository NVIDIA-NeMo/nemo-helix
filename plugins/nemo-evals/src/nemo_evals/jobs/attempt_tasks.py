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

An attempt's spec holds the tasks in one of two shapes. Jobs stored with their tasks inline carry ``tasks``
(and ``trials``) in the spec. Jobs stored with their tasks frozen in a file carry :data:`FROZEN_TASKS_KEY`
instead, naming that file and the digest of its exact bytes; the worker downloads it and checks those bytes
against the step config's digest before parsing them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from filesets import parse_fileset_ref
from nemo_evals.taskset_runs.plan_storage import PlanDigestMismatchError, read_verified_bytes_sync
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import JobsClient

TASKS_FROM_KEY: Final = "tasks_from"
TASKS_DIGEST_KEY: Final = "tasks_digest"
TASKS_FROM_ATTEMPT: Final = "attempt"
_DIGEST_PREFIX: Final = "sha256-v1:"
_FROZEN_KEYS: Final = ("tasks", "trials")
#: The stored-spec key naming the file that holds a job's frozen tasks and trials.
FROZEN_TASKS_KEY: Final = "frozen_tasks"
#: The path of that file inside its fileset.
FROZEN_TASKS_PATH: Final = "tasks.json"

#: The service the agent-evaluate task runs as (see :mod:`nemo_evals.tasks.agent_evaluate`).
TASK_SERVICE_NAME: Final = "evals"


class AttemptTasksError(RuntimeError):
    """The worker could not load its frozen tasks from its job attempt, or they don't match the step config."""


def frozen_tasks_bytes(spec: dict) -> bytes:
    """``spec``'s ``tasks`` and ``trials`` as canonical JSON bytes: what the digest covers, and what a frozen-tasks file holds."""
    frozen = {key: spec.get(key) for key in _FROZEN_KEYS}
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return canonical.encode("utf-8")


def frozen_tasks_digest(spec: dict) -> str:
    """The versioned digest of ``spec``'s ``tasks`` and ``trials``, as JSON (``model_dump(mode="json")``).

    It is ``sha256-v1:`` and the SHA-256 of :func:`frozen_tasks_bytes`, so it binds a frozen-tasks file whose
    ``plan_digest`` is that SHA-256.
    """
    return _DIGEST_PREFIX + hashlib.sha256(frozen_tasks_bytes(spec)).hexdigest()


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
    rest.update(frozen)
    return rest


def _frozen_tasks(client: NemoClient, *, workspace: str, job: str, attempt: str, expected: str) -> dict:
    """The attempt's ``tasks`` and ``trials``, from its spec or from the file its spec names, checked against ``expected``."""
    spec = _attempt_spec(client, workspace=workspace, job=job, attempt=attempt)
    # A null pointer is no pointer, as the reply model reads it: the tasks are inline.
    if spec.get(FROZEN_TASKS_KEY) is not None:
        return _read_frozen_tasks_file(client, spec[FROZEN_TASKS_KEY], workspace=workspace, expected=expected)
    if not isinstance(spec.get("tasks"), list):
        raise AttemptTasksError(f"attempt {attempt!r} of job {job!r} has no task list in its spec")
    actual = frozen_tasks_digest(spec)
    if actual != expected:
        raise AttemptTasksError(
            f"the tasks on attempt {attempt!r} of job {job!r} don't match its step config "
            f"(expected {expected}, got {actual}); refusing to run them"
        )
    return {"tasks": spec["tasks"], "trials": spec.get("trials")}


def _read_frozen_tasks_file(client: NemoClient, pointer: object, *, workspace: str, expected: str) -> dict:
    """Download the frozen-tasks file ``pointer`` names, check its exact bytes against ``expected``, then parse it.

    The file is stored and read the way a taskset run's plan is (:mod:`nemo_evals.taskset_runs.plan_storage`).
    """
    if not isinstance(pointer, dict) or not isinstance(pointer.get("plan_ref"), str):
        raise AttemptTasksError(f"the attempt's {FROZEN_TASKS_KEY} names no file")
    plan_ref = pointer["plan_ref"]
    plan_digest = pointer.get("plan_digest")
    if not isinstance(plan_digest, str) or _DIGEST_PREFIX + plan_digest != expected:
        raise AttemptTasksError(
            f"the attempt's {FROZEN_TASKS_KEY} digest {plan_digest!r} doesn't match its step config "
            f"({expected}); refusing to run it"
        )
    try:
        file_workspace, _, path = parse_fileset_ref(plan_ref, workspace_fallback=None)
    except ValueError as exc:
        raise AttemptTasksError(f"the attempt's frozen tasks file {plan_ref!r} is not a fileset file ref") from exc
    if file_workspace != workspace or path != FROZEN_TASKS_PATH:
        raise AttemptTasksError(
            f"the attempt's frozen tasks file {plan_ref!r} must be {FROZEN_TASKS_PATH!r} in workspace {workspace!r}"
        )
    try:
        body = read_verified_bytes_sync(FilesClient.from_client(client), plan_ref, expected_digest=plan_digest)
    except PlanDigestMismatchError as exc:
        raise AttemptTasksError(
            f"the frozen tasks file {plan_ref!r} doesn't match its step config; refusing to run it"
        ) from exc
    try:
        frozen = json.loads(body)
    except ValueError as exc:
        raise AttemptTasksError(f"the frozen tasks file {plan_ref!r} is not JSON") from exc
    if not isinstance(frozen, dict) or not isinstance(frozen.get("tasks"), list):
        raise AttemptTasksError(f"the frozen tasks file {plan_ref!r} has no task list")
    return {"tasks": frozen["tasks"], "trials": frozen.get("trials")}


def _attempt_spec(client: NemoClient, *, workspace: str, job: str, attempt: str) -> dict:
    return JobsClient.from_client(client).get_job_attempt(workspace=workspace, job=job, attempt=attempt).data().spec


__all__ = [
    "FROZEN_TASKS_KEY",
    "FROZEN_TASKS_PATH",
    "TASKS_DIGEST_KEY",
    "TASKS_FROM_ATTEMPT",
    "TASKS_FROM_KEY",
    "TASK_SERVICE_NAME",
    "AttemptTasksError",
    "frozen_tasks_bytes",
    "frozen_tasks_digest",
    "load_attempt_tasks",
    "step_config_without_tasks",
]
