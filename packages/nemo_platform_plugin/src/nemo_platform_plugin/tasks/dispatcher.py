# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint dispatcher for :class:`~nemo_platform_plugin.job.NemoJob` subclasses.

Mirrors :meth:`~nemo_platform_plugin.scheduler.NemoJobScheduler.run_local` for any
process the platform spawns with the ``NEMO_JOB_*`` environment populated —
both Docker-backed task containers and host subprocess executors land here.
Reads the step config from the platform-injected file path, builds a
:class:`~nemo_platform_plugin.job_context.JobContext` from the environment, and
invokes ``job.run(...)`` with the same signature-based DI used locally
(see :func:`~nemo_platform_plugin.run_dependencies.resolve_run_kwargs`).

Because ``run_task`` only runs inside a platform-spawned process, the
default ``ctx.results`` is :class:`~nemo_platform_plugin.job_results.PlatformJobResults`
— results upload through the Files service whether the deployment is
docker-backed or subprocess-on-local-host. Storage backend (local FS, S3,
…) is the Files service's concern, not the plugin's. Plugins that need a
different sink build a :class:`JobContext` and pass it via ``ctx=``.

Usage from a plugin's ``__main__.py``::

    import signal
    import sys
    from types import FrameType

    from nemo_platform_plugin.sdk_provider import get_task_sdk
    from nemo_platform_plugin.tasks.dispatcher import run_task
    from my_plugin.jobs.train import TrainJob


    def _shutdown(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)


    if __name__ == "__main__":
        signal.signal(signal.SIGTERM, _shutdown)
        sys.exit(run_task(TrainJob, sdk=get_task_sdk("my-service")))
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import PlatformJobResults
from nemo_platform_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_platform_plugin.run_dependencies import LocalRunError, resolve_run_kwargs
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging

logger = logging.getLogger(__name__)


def run_task(
    job_cls: type[NemoJob],
    *,
    sdk: Any | None = None,
    async_sdk: Any | None = None,
    ctx: JobContext | None = None,
) -> int:
    """Run *job_cls* in a platform-spawned task process; return a process exit code.

    Reads the step config from :data:`NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR`,
    builds a :class:`JobContext` from the platform-injected env when *ctx* is
    omitted, and invokes ``job.run(config, **kwargs)`` with signature DI of
    ``ctx`` / ``sdk`` / ``async_sdk``. Exit codes follow :func:`_exit_code_for`;
    :class:`~nemo_platform_plugin.run_dependencies.LocalRunError` propagates verbatim.

    The auto-built ``ctx`` wires
    :class:`~nemo_platform_plugin.job_results.PlatformJobResults` as ``ctx.results``,
    so *sdk* is required when *ctx* is not supplied.

    Root logging is bootstrapped via
    :func:`~nemo_platform_plugin.tasks.logging_setup.configure_task_logging`
    before anything else, since a task container configures none of its own.
    A caller that already configured logging keeps it.

    Args:
        job_cls: The :class:`~nemo_platform_plugin.job.NemoJob` subclass to run.
        sdk: :class:`~nemo_platform.NeMoPlatform` handle, typically built via
            ``get_task_sdk("<plugin>")`` so ``NMP_PRINCIPAL`` threads through
            as on-behalf-of auth. Required unless *ctx* is supplied.
        async_sdk: Async SDK counterpart for jobs that delegate to async helpers.
        ctx: Override for the auto-built :class:`JobContext` — used to inject
            a non-default ``results`` sink (e.g. :class:`LocalJobResults` for
            offline runs).

    Returns:
        Process exit code suitable for :func:`sys.exit`.
    """
    # First thing in the process: a task container configures nothing itself,
    # so without this every message below INFO+lastResort is lost - including
    # the failure reporting further down. Must precede the first log call.
    configure_task_logging()

    try:
        config = _read_step_config()
    except Exception:
        logger.exception("Failed to read step config")
        return 2

    if ctx is None:
        if sdk is None:
            logger.error(
                "%s.run_task requires sdk= when no explicit ctx is provided (default ctx wires PlatformJobResults).",
                job_cls.__name__,
            )
            return 2
        try:
            ctx = build_ctx_from_env(sdk)
        except Exception:
            logger.exception("Failed to build JobContext from environment")
            return 2

    try:
        job = job_cls()
    except Exception:
        # Constructor failures are setup errors, not run errors — surface as 2.
        logger.exception("Failed to instantiate %s", job_cls.__name__)
        return 2

    try:
        kwargs = resolve_run_kwargs(job_cls, job.run, sdk=sdk, async_sdk=async_sdk, ctx=ctx, is_local=False)
    except LocalRunError:
        # Plugin-author bug (e.g. required sdk param without a handle); propagate
        # rather than collapse into the same exit-2 bucket as a missing env var.
        raise
    except Exception:
        logger.exception("Failed to resolve run kwargs for %s", job_cls.__name__)
        return 2

    try:
        result = job.run(config, **kwargs)
    except LocalRunError:
        # Propagate verbatim per run_task's contract.
        raise
    except Exception:
        logger.exception("%s.run raised", job_cls.__name__)
        return 1

    logger.info("%s result: %s", job_cls.__name__, result)
    return _exit_code_for(result)


def _exit_code_for(result: Any) -> int:
    """Map a ``NemoJob.run`` return value to a process exit code.

    Recognises two in-tree failure shapes — ``{"status": "failed", ...}``
    and ``{"exit_code": <non-zero>, ...}`` — and treats ``None`` as failure
    (likely a missing ``return``).  Anything else is success.
    """
    if result is None:
        logger.warning("Job returned None; treating as failure")
        return 1
    if not isinstance(result, dict):
        return 0
    if result.get("status") == "failed":
        # A job that reports failure through its return value never raises, so
        # nothing else in this process says why the exit code is non-zero.
        logger.error("Job reported status=failed; result: %s", result)
        return 1
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        logger.error("Job reported non-zero exit_code=%s; result: %s", exit_code, result)
        return 1
    return 0


def _read_step_config() -> dict:
    path_str = os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR)
    if not path_str:
        raise RuntimeError(f"{NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} not set; running outside the platform?")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Step config not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # JSONDecodeError already carries line/column. The byte count is for
        # diagnosing empty or truncated platform-written config files.
        try:
            size = path.stat().st_size
        except OSError:
            size_text = "unknown size"
        else:
            size_text = f"{size} bytes"
        raise RuntimeError(f"Invalid JSON in step config at {path} ({size_text})") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Step config at {path} must be a JSON object, got {type(data).__name__}")
    return data


def build_ctx_from_env(sdk: NeMoPlatform) -> JobContext:
    """Build a :class:`JobContext` from the platform-injected ``NEMO_JOB_*`` env.

    Wires :attr:`JobContext.results` to :class:`PlatformJobResults` so results
    upload through the Files service — works the same in docker-backed and
    subprocess-on-local-host deployments. Each ``NEMO_JOB_*`` envvar is required;
    a missing one raises rather than silently falling back to a path that only
    exists inside task containers.

    ``PlatformJobResults.job_name`` is the *submitted* platform job name, not
    the job class name — :class:`~nemo_platform_plugin.jobs.result_manager.ResultManager`
    uses it to look up the job via the typed Jobs client so each
    ``ctx.results.save()`` registers against the correct job record. The
    backends inject this as ``NEMO_JOB_ID = step.job``; the NemoJob class
    identifier (e.g. ``"evaluate"``) would point at a non-existent job and
    break result registration.
    """
    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR, "").strip()
    if not workspace:
        raise RuntimeError(f"{NEMO_JOB_WORKSPACE_ENVVAR} not set; running outside the platform?")
    persistent_str = os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)
    ephemeral_str = os.environ.get(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR)
    if not ephemeral_str:
        raise RuntimeError(f"{EPHEMERAL_TASK_STORAGE_PATH_ENVVAR} not set; running outside the platform?")
    job_id = os.environ.get(NEMO_JOB_ID_ENVVAR, "").strip()
    if not job_id:
        raise RuntimeError(f"{NEMO_JOB_ID_ENVVAR} not set; running outside the platform?")
    return JobContext(
        workspace=workspace,
        storage=StoragePaths(
            ephemeral=Path(ephemeral_str),
            persistent=Path(persistent_str) if persistent_str else None,
        ),
        results=PlatformJobResults(
            job_name=job_id,
            workspace=workspace,
            client=client_from_platform(sdk, NemoClient),
        ),
        job_id=job_id,
    )


__all__ = ["build_ctx_from_env", "run_task"]
