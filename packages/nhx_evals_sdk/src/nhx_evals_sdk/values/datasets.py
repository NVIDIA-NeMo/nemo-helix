# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-related value types for evals SDK runtime."""

from typing import Any, TypeAlias

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class DatasetRows(BaseModel):
    """Inline dataset definition with embedded rows.

    Use this for quick evaluations without persisting the dataset first.
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        min_length=1,
        description="Array of data rows. Each row can be any valid JSON value (object, string, array, etc.).",
    )


DatasetInput: TypeAlias = list[dict[str, Any]] | DatasetRows | pa.Table
