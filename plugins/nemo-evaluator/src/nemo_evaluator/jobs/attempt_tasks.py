# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep an agent evaluation's frozen tasks and trials out of its step ConfigMap.

Backported from PR #3869 (21aadb7ac): Kubernetes caps ConfigMaps at 1 MiB, so the
compiler omits tasks and trials from the agent-evaluate step config, marks
tasks_from='attempt', and pins tasks_digest='sha256-v1:...'. The worker reads
the immutable spec back from its own job record and verifies the digest before running.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client_provider import get_task_nemo_client
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.client import JobsClient

TASKS_FROM_KEY: Final = "tasks_from"
TASKS_DIGEST_KEY: Final = "tasks_digest"
TASKS_FROM_ATTEMPT: Final = "attempt"
_DIGEST_PREFIX: Final = "sha256-v1:"
_FROZEN_KEYS: Final = ("tasks", "trials")

TASK_SERVICE_NAME: Final = "evaluator"


class AttemptTasksError(RuntimeError):
    """The worker could not load its frozen tasks from its job record, or the digest mismatched."""


def frozen_tasks_digest(spec: dict) -> str:
    """Return the versioned SHA-256 digest of spec's tasks and trials."""
    frozen = {key: spec.get(key) for key in _FROZEN_KEYS}
    canonical = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def step_config_without_tasks(spec: dict) -> dict:
    """Return spec without tasks/trials, stamped with source marker and digest."""
    config = {key: value for key, value in spec.items() if key not in _FROZEN_KEYS}
    config[TASKS_FROM_KEY] = TASKS_FROM_ATTEMPT
    config[TASKS_DIGEST_KEY] = frozen_tasks_digest(spec)
    return config


def _job_spec(
    *,
    workspace: str,
    job_name: str,
    sdk: NeMoPlatform | None = None,
    client: NemoClient | None = None,
) -> dict:
    if client is not None:
        return JobsClient.from_client(client).get_job(workspace=workspace, name=job_name).data().spec
    if sdk is not None:
        return client_from_platform(sdk, JobsClient).get_job(workspace=workspace, name=job_name).data().spec
    with get_task_nemo_client(TASK_SERVICE_NAME, workspace=workspace) as task_client:
        return JobsClient.from_client(task_client).get_job(workspace=workspace, name=job_name).data().spec


def load_attempt_tasks(
    config: dict,
    *,
    ctx: JobContext,
    sdk: NeMoPlatform | None = None,
    client: NemoClient | None = None,
) -> dict:
    """Restore tasks and trials onto config from the persisted job spec."""
    if TASKS_FROM_KEY not in config:
        return config
    rest = dict(config)
    source = rest.pop(TASKS_FROM_KEY)
    expected = rest.pop(TASKS_DIGEST_KEY, None)
    if source != TASKS_FROM_ATTEMPT:
        raise AttemptTasksError(
            f"step config names an unknown task source {source!r}; expected {TASKS_FROM_ATTEMPT!r}"
        )
    if not isinstance(expected, str) or not expected.startswith(_DIGEST_PREFIX):
        raise AttemptTasksError(
            f"step config's {TASKS_DIGEST_KEY} is missing or not a {_DIGEST_PREFIX} digest"
        )
    if not ctx.job_id:
        raise AttemptTasksError(
            "step config's tasks are on its job record, but this run has no job_id"
        )
    job_name = ctx.job_id
    try:
        spec = _job_spec(workspace=ctx.workspace, job_name=job_name, sdk=sdk, client=client)
    except Exception as exc:
        raise AttemptTasksError(
            f"could not read the tasks of job {job_name!r} in workspace {ctx.workspace!r}"
        ) from exc
    if not isinstance(spec.get("tasks"), list):
        raise AttemptTasksError(f"job {job_name!r} has no task list in its spec")
    actual = frozen_tasks_digest(spec)
    if actual != expected:
        raise AttemptTasksError(
            f"the tasks on job {job_name!r} do not match its step config "
            f"(expected {expected}, got {actual}); refusing to run them"
        )
    rest["tasks"] = spec["tasks"]
    rest["trials"] = spec.get("trials")
    return rest


__all__ = [
    "TASKS_DIGEST_KEY",
    "TASKS_FROM_ATTEMPT",
    "TASKS_FROM_KEY",
    "TASK_SERVICE_NAME",
    "AttemptTasksError",
    "frozen_tasks_digest",
    "load_attempt_tasks",
    "step_config_without_tasks",
]
