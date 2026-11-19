# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs service testing helpers."""

import logging
import threading
from collections.abc import Generator, Iterable
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.auth import Principal
from nhx.common.entities import SyncEntityClient
from nhx.core.jobs.app.event_publisher import JobEventPublisher
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobExecutionProfile


@contextmanager
def subprocess_job_executor_patch(
    executors: Iterable[ExecutionProfileT] = (),
    *,
    profile: str = "default",
) -> Generator[list[ExecutionProfileT], None, None]:
    """Patch Jobs execution profiles for CPU-step translation tests.

    The Jobs API translates ``cpu/<profile>`` container steps to
    ``subprocess/<profile>`` only when that subprocess profile is explicitly
    configured in ``jobs.config.executors``. The endpoint also validates against
    a module-level profile list, so both sources are patched together. Use this
    around tests that submit plugin-compiled CPU jobs against an in-process Jobs
    API.
    """
    from nhx.core.jobs import config as jobs_config_module
    from nhx.core.jobs.api.v2.jobs import endpoints as jobs_endpoints

    patched_executors = list(executors)
    if not any(executor.provider == "subprocess" and executor.profile == profile for executor in patched_executors):
        patched_executors.insert(0, SubprocessJobExecutionProfile(profile=profile))

    with ExitStack() as stack:
        stack.enter_context(patch.object(jobs_config_module.config, "executors", patched_executors))
        stack.enter_context(patch.object(jobs_config_module, "profiles", patched_executors))
        stack.enter_context(patch.object(jobs_endpoints, "profiles", patched_executors))
        yield patched_executors


@contextmanager
def job_creation_controller(client: NemoClient) -> Generator[None]:
    """Complete real creation handoffs without scheduling workloads in service tests."""
    service_client = client.with_headers(Principal(id="service:jobs", authz_aliases=["service:jobs"]).get_headers())
    publisher = JobEventPublisher(
        SyncEntityClient(EntitiesClient.from_client(service_client)), WorkspacesClient.from_client(service_client)
    )
    stop = threading.Event()

    def publish() -> None:
        while not stop.is_set():
            try:
                publisher._publish_pending_jobs(lambda: None, stop)
            except Exception:
                logging.getLogger(__name__).debug("Creation handoff will retry", exc_info=True)
            stop.wait(0.02)

    controller = threading.Thread(target=publish, daemon=True)
    controller.start()
    try:
        yield
    finally:
        stop.set()
        controller.join(timeout=5)
        assert not controller.is_alive(), "Test creation controller did not stop"
