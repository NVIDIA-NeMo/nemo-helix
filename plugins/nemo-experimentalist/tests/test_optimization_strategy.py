# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from nemo_agent_optimization_plugin.strategies import OptimizationStrategy
from nemo_experimentalist_plugin.optimization_strategy import ExperimentalistOptimizationStrategy
from nemo_platform_plugin.job_context import JobContext


def test_name_is_experimentalist() -> None:
    assert ExperimentalistOptimizationStrategy().name == "experimentalist"


def test_satisfies_optimization_strategy_protocol() -> None:
    assert isinstance(ExperimentalistOptimizationStrategy(), OptimizationStrategy)


def test_validate_config_accepts_any_mapping() -> None:
    ExperimentalistOptimizationStrategy().validate_config({"anything": "goes"}, agent="my-agent")


def test_run_raises_not_implemented() -> None:
    strategy = ExperimentalistOptimizationStrategy()
    with pytest.raises(NotImplementedError, match="experimentalist"):
        strategy.run(
            agent_config={"schema_version": "fabric.agent/v1alpha1"},
            source_agent_config=None,
            config={},
            # A real JobContext is never touched: run() deletes it and raises immediately. Cast
            # rather than build one so ty sees a JobContext without dragging in job-context fixtures.
            ctx=cast(JobContext, object()),
            workspace="default",
            sdk=None,
        )
