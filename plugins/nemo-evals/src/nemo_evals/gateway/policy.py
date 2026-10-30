# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The models an evals credential may call through the Inference Gateway.

A trial token carries one (see :mod:`nemo_evals.gateway.tokens`), and the gateway's access check holds
every call made with it to the policy (see :mod:`nemo_evals.gateway.middleware.evals_access`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelAccess(StrEnum):
    CLOSED = "closed"
    OPEN = "open"


class ModelPolicy(BaseModel):
    """The models a credential may call through the Inference Gateway."""

    model_config = ConfigDict(frozen=True)

    model_access: ModelAccess
    models: tuple[str, ...] | None = Field(
        default=None,
        description="The VirtualModels the credential may call, as `workspace/name`; any when unset.",
    )

    @model_validator(mode="after")
    def _closed_lists_its_models(self) -> Self:
        if self.model_access is ModelAccess.CLOSED and not self.models:
            raise ValueError("a closed model policy must list the models it allows")
        return self
