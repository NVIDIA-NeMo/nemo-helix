# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx_evals_sdk.values.params import InferenceParams, RunConfigOnlineModel
from pydantic import BaseModel, Field, ValidationError


class TestInferenceParamsContract:
    def test_inference_params_exposes_expected_sdk_contract(self):
        assert list(InferenceParams.model_fields) == [
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stop",
        ]
        assert InferenceParams.model_config.get("extra") == "allow"
        assert callable(getattr(InferenceParams, "check_max_tokens", None))

    def test_inference_params_enforces_token_limit_invariant(self):
        with pytest.raises(ValidationError, match="max_tokens and max_completion_tokens cannot both be configured"):
            InferenceParams(max_tokens=1, max_completion_tokens=1)


class TestRunConfigOnlineModelInferenceCoercion:
    def test_evaluation_job_params_coerces_compatible_foreign_inference_model(self):
        class CompatibleInferenceParams(BaseModel):
            temperature: float = 0.25
            max_tokens: int = 12
            extra_body: dict[str, dict[str, int]] = Field(
                default_factory=lambda: {"nvext": {"max_thinking_tokens": 64}}
            )

        params = RunConfigOnlineModel.model_validate({"inference": CompatibleInferenceParams()})

        assert isinstance(params.inference, InferenceParams)
        assert params.inference.model_dump(mode="python", exclude_none=True) == {
            "temperature": 0.25,
            "max_tokens": 12,
            "extra_body": {"nvext": {"max_thinking_tokens": 64}},
        }

    def test_evaluation_job_params_rejects_incompatible_foreign_model(self):
        class ForeignInferenceParams(BaseModel):
            temperature: str = "hot"

        with pytest.raises(ValidationError, match="Input should be a valid number"):
            RunConfigOnlineModel.model_validate({"inference": ForeignInferenceParams()})
