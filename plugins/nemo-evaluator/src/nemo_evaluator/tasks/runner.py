# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared container/subprocess task entrypoint for evaluator plugin jobs.

Each job's ``tasks/<job>.py`` is a thin ``python -m`` target that calls the sync
or async task runner matching its job class. The lifecycle — SIGTERM handling,
building the task SDK, and dispatching to the job — is identical across jobs and
lives here.
"""

from __future__ import annotations

import logging
import signal
from types import FrameType

from nemo_platform_plugin.client_provider import get_async_task_nemo_client, get_task_nemo_client
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.sdk_provider import get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import build_ctx_from_env, run_task_with_async_client, run_task_with_client

logger = logging.getLogger(__name__)

#: Process exit code when the task SDK can't be built (setup failure, before the job runs).
SDK_INITIALIZATION_EXIT_CODE = 2


def _shutdown_handler(signum: int, frame: FrameType | None) -> None:
    logger.warning("Received shutdown signal (%d). Exiting.", signum)
    raise SystemExit(128 + signum)


def run_sync_task_main(job_cls: type[NemoJob], *, service_name: str) -> int:
    """Build the sync task client and dispatch to ``job_cls``.

    Returns :data:`SDK_INITIALIZATION_EXIT_CODE` if the task client can't be built.

    The generated platform SDK is used only to construct the platform-backed
    result sink. The job itself receives the typed sync client it declares.
    """
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk(service_name)
        ctx = build_ctx_from_env(sdk)
        client = get_task_nemo_client(service_name)
    except Exception:
        logger.exception("Failed to build task client for %s", service_name)
        return SDK_INITIALIZATION_EXIT_CODE
    return run_task_with_client(job_cls, client=client, ctx=ctx)


def run_async_task_main(job_cls: type[NemoJob], *, service_name: str) -> int:
    """Build the async task client and dispatch to ``job_cls``.

    Returns :data:`SDK_INITIALIZATION_EXIT_CODE` if the task client can't be built.

    The generated platform SDK is used only to construct the platform-backed
    result sink. The job itself receives the typed async client it declares.
    """
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        sdk = get_task_sdk(service_name)
        ctx = build_ctx_from_env(sdk)
        async_client = get_async_task_nemo_client(service_name)
    except Exception:
        logger.exception("Failed to build task client for %s", service_name)
        return SDK_INITIALIZATION_EXIT_CODE
    return run_task_with_async_client(job_cls, async_client=async_client, ctx=ctx)
