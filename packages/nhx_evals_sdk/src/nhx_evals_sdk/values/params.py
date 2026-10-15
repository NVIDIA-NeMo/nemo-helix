# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution parameter types for evals SDK and service runtimes."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nhx_evals_sdk.values.models import ReasoningParams


class InferenceParams(BaseModel):
    """
    Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation.
    """

    model_config = ConfigDict(extra="allow")

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


class RunConfig(BaseModel):
    """Job parameters."""

    model_config = ConfigDict(extra="forbid")

    parallelism: int = Field(
        default=8,
        ge=1,
        description="Parallelism to be used for the evaluation job. "
        "Typically, this represents the maximum number of concurrent requests made to the model.",
    )
    limit_samples: int | None = Field(
        default=None,
        ge=1,
        description="Limit number of evaluation samples, taking the first `limit` samples from the dataset.",
    )


class RunConfigOnline(RunConfig):
    """Job parameters for online evaluation."""

    ignore_request_failure: bool = Field(
        default=False,
        description="If True, request failures will be ignored and the result will be marked as NaN. "
        "If False (default), request failures will raise an exception.",
    )
    request_timeout: int | None = Field(
        default=None, description="The timeout to be used for requests made to the model."
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries for failed requests.")


class RunConfigOnlineModel(RunConfigOnline):
    """Job parameters for model online evaluation."""

    inference: InferenceParams | None = Field(
        default=None,
        description="Custom settings that control the model's text generation behavior.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Initial instructions that define the model's role and behavior for the conversation.",
    )
    reasoning: ReasoningParams | None = Field(
        default=None, description="Custom settings that control the model's reasoning behavior."
    )
    structured_output: dict | None = Field(
        default=None,
        description="JSON schema to apply structured output for the model.",
    )

    @field_validator("inference", mode="before")
    @classmethod
    def coerce_inference_params(cls, value: object) -> object:
        """Normalize equivalent Pydantic inference models before SDK validation runs."""
        # Compatibility shim for service-side inference models. This converts other
        # Pydantic models into plain data and lets SDK InferenceParams validation
        # decide whether the payload is compatible.
        if isinstance(value, BaseModel) and not isinstance(value, InferenceParams):
            return value.model_dump(mode="python", exclude_none=True)
        return value
