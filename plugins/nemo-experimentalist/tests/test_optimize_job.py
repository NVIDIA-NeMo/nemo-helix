# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_experimentalist_plugin.jobs.optimize import ExperimentalistOptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext


def test_the_job_declares_its_strategy() -> None:
    assert ExperimentalistOptimizeJob.strategy == "experimentalist"
    assert issubclass(ExperimentalistOptimizeJob, AgentOptimizeJob)


def test_optimize_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="nemo agents experimentalist"):
        ExperimentalistOptimizeJob().optimize(
            source_agent_config={},
            config={},
            ctx=cast(JobContext, object()),
            workspace="my-ws",
            sdk=cast(NeMoPlatform, object()),
        )


def test_task_module_is_set() -> None:
    assert ExperimentalistOptimizeJob.task_module == "nemo_experimentalist_plugin.tasks.agent_optimize"
