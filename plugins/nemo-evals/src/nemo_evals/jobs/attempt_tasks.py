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
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from nemo_evals.jobs.utils import run_with_isolated_async_client
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.types import HelixJobAttemptResponse

TASKS_FROM_KEY: Final = "tasks_from"
TASKS_DIGEST_KEY: Final = "tasks_digest"
TASKS_FROM_ATTEMPT: Final = "attempt"
_DIGEST_PREFIX: Final = "sha256-v1:"
_FROZEN_KEYS: Final = ("tasks", "trials")


class AttemptTasksError(RuntimeError):
    """The worker could not load its frozen tasks from its job attempt, or they don't match the step config."""


def frozen_tasks_digest(spec: dict) -> str:
    """The versioned digest of ``spec``'s ``tasks`` and ``trials``, as JSON (``model_dump(mode="json")``)."""
    frozen = {key: spec.get(key) for key in _FROZEN_KEYS}
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def step_config_without_tasks(spec: dict) -> dict:
    """The agent-evaluate step config for ``spec``: everything but ``tasks``/``trials``, plus the marker and digest."""
    config = {key: value for key, value in spec.items() if key not in _FROZEN_KEYS}
    config[TASKS_FROM_KEY] = TASKS_FROM_ATTEMPT
    config[TASKS_DIGEST_KEY] = frozen_tasks_digest(spec)
    return config


def load_attempt_tasks(config: dict, *, ctx: JobContext, client: NemoClient | AsyncNemoClient | None) -> dict:
    """``config`` with its frozen tasks and trials restored from the job attempt it runs in.

    A config without :data:`TASKS_FROM_KEY` (a local run, a taskset-run task Job, a job compiled before
    the tasks moved out) is returned unchanged, and nothing is read.

    Raises:
        AttemptTasksError: The config names an unknown source or carries no digest; the run has no job
            attempt or client to read from; the read fails; or the attempt's tasks don't match the digest.
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
    if ctx.job_id is None or ctx.attempt_id is None or client is None:
        raise AttemptTasksError(
            "step config's tasks are on its job attempt, but this run has no job attempt or platform client to read them"
        )
    job, attempt_id = ctx.job_id, ctx.attempt_id
    try:
        attempt = _get_attempt(client, workspace=ctx.workspace, job=job, attempt=attempt_id)
    except Exception as exc:
        raise AttemptTasksError(f"could not read the tasks of job {job!r} from its attempt {attempt_id!r}") from exc
    spec = attempt.spec
    if not isinstance(spec.get("tasks"), list):
        raise AttemptTasksError(f"attempt {attempt_id!r} of job {job!r} has no task list in its spec")
    actual = frozen_tasks_digest(spec)
    if actual != expected:
        raise AttemptTasksError(
            f"the tasks on attempt {attempt_id!r} of job {job!r} don't match its step config "
            f"(expected {expected}, got {actual}); refusing to run them"
        )
    rest["tasks"] = spec["tasks"]
    rest["trials"] = spec.get("trials")
    return rest


def _get_attempt(
    client: NemoClient | AsyncNemoClient, *, workspace: str, job: str, attempt: str
) -> HelixJobAttemptResponse:
    if isinstance(client, AsyncNemoClient):

        async def _read(async_client: AsyncNemoClient) -> HelixJobAttemptResponse:
            response = await AsyncJobsClient.from_client(async_client).get_job_attempt(
                workspace=workspace, job=job, attempt=attempt
            )
            return response.data()

        return run_with_isolated_async_client(client, _read)
    return JobsClient.from_client(client).get_job_attempt(workspace=workspace, job=job, attempt=attempt).data()


__all__ = [
    "TASKS_DIGEST_KEY",
    "TASKS_FROM_ATTEMPT",
    "TASKS_FROM_KEY",
    "AttemptTasksError",
    "frozen_tasks_digest",
    "load_attempt_tasks",
    "step_config_without_tasks",
]
