# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inference parameter types used by models and evaluation."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceParams(BaseModel):
    """
    Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="Model identifier")
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Float value between 0 and 1. temp of 0 indicates greedy decoding, "
        "where the token with highest prob is chosen. Temperature can't be set to 0.0 currently",
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    max_completion_tokens: int | None = Field(default=None, ge=1, description="Max tokens to generate")
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Float value between 0 and 1; limits to the top tokens within a certain "
        "probability. top_p=0 means the model will only consider the single most likely "
        "token for the next prediction",
    )
    stop: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def check_max_tokens(self) -> Self:
        if self.max_tokens and self.max_completion_tokens:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be configured. "
                "Choose the appropriate tokens parameter for the model."
            )
        return self
