# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public configuration types for the v4 evaluator API."""

from __future__ import annotations

from typing import TypeAlias

from nhx_evals_sdk.values import (
    AgentBase,
    Model,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nhx_evals_sdk.values.retrieval import Retrieval
from nhx_evals_sdk.values.targets import EvalTarget

_RunConfigT: TypeAlias = RunConfig | RunConfigOnline | RunConfigOnlineModel


def resolve_params(
    params: _RunConfigT | None = None,
    target: EvalTarget = None,
) -> _RunConfigT:
    """Return params after validating that they match the selected target mode."""
    if isinstance(target, Retrieval):
        if params is None:
            return RunConfigOnlineModel()
        if type(params) is RunConfig:
            return RunConfigOnlineModel.model_validate(params.model_dump())
        if type(params) is RunConfigOnline:
            return RunConfigOnlineModel.model_validate(params.model_dump())
        if not isinstance(params, RunConfigOnlineModel):
            raise TypeError("retrieval target requires RunConfigOnlineModel")
        return params
    if isinstance(target, Model):
        if params is None or type(params) is RunConfig:
            raise TypeError("model target requires RunConfigOnlineModel")
        if type(params) is RunConfigOnline:
            return RunConfigOnlineModel.model_validate(params.model_dump())
        if not isinstance(params, RunConfigOnlineModel):
            raise TypeError("model target requires RunConfigOnlineModel")
        return params
    if isinstance(target, AgentBase):
        if type(params) is not RunConfigOnline:
            raise TypeError("agent target requires RunConfigOnline")
        return params
    if params is None:
        return RunConfig()
    if type(params) is not RunConfig:
        raise TypeError("offline evaluation requires RunConfig")
    return params


def fail_fast_from_params(params: _RunConfigT) -> bool:
    """
    Return whether row failures should abort execution for the given params.
    When params is not an online params, return fail_fast is True - always fail fast.
    """
    return not (isinstance(params, RunConfigOnline) and params.ignore_request_failure)
