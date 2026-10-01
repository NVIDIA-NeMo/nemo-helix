# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Destroy every sandbox for a job when the owning process is cancelled or exits.

The actor that created a sandbox cannot be relied on to destroy it. Ray kills that
worker on cancel without running its teardown, and a failure in the actor leaves the
BatchSandbox until ``ttl_s``. The process that calls :func:`install_job_sandbox_reaper`
is the one the container runtime signals, so the sweep still runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from sandboxed_gym.host.provider import get_host_provider
from sandboxed_gym.termination import install_termination_cleanup

LOGGER = logging.getLogger(__name__)

#: Job ids this process has already armed. A second call must not replace the
#: signal handler the first one installed.
_INSTALLED: set[str] = set()


def reap_job_sandboxes(
    job_id: str,
    *,
    host_provider: str,
    host_provider_options: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Destroy every sandbox labeled with ``job_id``. Failures are logged, not raised.

    Raising from an ``atexit`` hook or a signal handler would skip the rest of
    process teardown. A sandbox that is already gone is not a failure of the exit.
    """
    options = dict(host_provider_options or {})
    try:
        provider = get_host_provider(host_provider, options)
        removed = asyncio.run(provider.destroy_job_sandboxes(job_id))
    except Exception:
        LOGGER.exception("failed to reap sandboxes for job %s", job_id)
        return ()
    if removed:
        LOGGER.warning(
            "reaped %d sandbox(es) for job %s: %s",
            len(removed),
            job_id,
            ", ".join(removed),
        )
    return removed


def install_job_sandbox_reaper(
    job_id: str,
    *,
    host_provider: str = "opensandbox",
    host_provider_options: Mapping[str, Any] | None = None,
) -> None:
    """Arm an ``atexit`` and termination-signal hook that reaps this job's sandboxes.

    Call it from the process the container runtime will signal, before the first
    sandbox is created. A later call for the same job id does nothing: replacing
    the signal handler would drop the hook the first call installed.
    """
    if not job_id:
        raise ValueError("sandbox reaper requires a job id")
    if job_id in _INSTALLED:
        return
    _INSTALLED.add(job_id)
    options = dict(host_provider_options or {})
    install_termination_cleanup(
        lambda: reap_job_sandboxes(
            job_id,
            host_provider=host_provider,
            host_provider_options=options,
        )
    )
