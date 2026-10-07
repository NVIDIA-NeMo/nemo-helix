# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public types and aliases for the evals plugin SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from nemo_evals.filesets import FilesetRef
from nhx_evals_sdk.values import (
    DatasetInput,
    FieldMapping,
    RunConfig,
    RunConfigOnline,
    RunConfigOnlineModel,
)

# TODO: remove this type if we decide nhx_evals_sdk will not support remote execution.
ExecutionMode: TypeAlias = Literal["local", "remote"]
PluginDatasetInput: TypeAlias = DatasetInput | str | Path | FilesetRef

__all__ = [
    "RunConfig",
    "RunConfigOnline",
    "RunConfigOnlineModel",
    "FilesetRef",
    "FieldMapping",
    "ExecutionMode",
    "PluginDatasetInput",
]
