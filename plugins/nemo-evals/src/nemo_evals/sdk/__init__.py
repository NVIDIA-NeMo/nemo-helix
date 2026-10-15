# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Evals plugin SDK surface."""

from typing import TYPE_CHECKING

from nemo_evals.filesets import FilesetRef
from nemo_evals.sdk.types import (
    ExecutionMode,
    PluginDatasetInput,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)

if TYPE_CHECKING:
    from nemo_evals.sdk.job_resources import AsyncEvaluatorJobResource, EvaluatorJobResource
    from nemo_evals.sdk.resources import AsyncEvaluator, Evaluator

__all__ = [
    "AsyncEvaluator",
    "AsyncEvaluatorJobResource",
    "Evaluator",
    "EvaluatorJobResource",
    "RunConfig",
    "RunConfigOnline",
    "RunConfigOnlineModel",
    "ExecutionMode",
    "FilesetRef",
    "PluginDatasetInput",
]


def __getattr__(name: str) -> object:
    """Load resource classes only when callers access them.

    Keep ``import nemo_evals.sdk`` lightweight for examples and SDK users
    that only need value types such as ``FilesetRef`` or ``RunConfig``. Eagerly
    importing ``resources`` pulls in ``sdk._executor``, which imports
    ``jobs.evaluate`` and the plugin job/metric-bundle runtime. The lightweight
    path is used by imports such as ``from nemo_evals.sdk import FilesetRef``;
    preserve this lazy boundary so those imports do not initialize job code.
    """
    if name == "AsyncEvaluator":
        from nemo_evals.sdk.resources import AsyncEvaluator

        return AsyncEvaluator
    if name == "Evaluator":
        from nemo_evals.sdk.resources import Evaluator

        return Evaluator
    if name == "AsyncEvaluatorJobResource":
        from nemo_evals.sdk.job_resources import AsyncEvaluatorJobResource

        return AsyncEvaluatorJobResource
    if name == "EvaluatorJobResource":
        from nemo_evals.sdk.job_resources import EvaluatorJobResource

        return EvaluatorJobResource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
