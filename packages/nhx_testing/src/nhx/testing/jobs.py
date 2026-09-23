# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs service testing helpers."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobExecutionProfile


@contextmanager
def subprocess_job_executor_patch(
    executors: Iterable[ExecutionProfileT] = (),
    *,
    profile: str = "default",
) -> Iterator[list[ExecutionProfileT]]:
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
