# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Synthesizer job config shared by API handlers and task containers."""

import logging
from typing import Any

from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_safe_synthesizer.config.job import SafeSynthesizerJobConfig as SafeSynthesizerJobConfigInternal
from nemo_safe_synthesizer.config.job import SafeSynthesizerParameters as SafeSynthesizerParametersInternal
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from pydantic import Field, model_validator
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

DEFAULT_PRETRAINED_MODEL = "HuggingFaceTB/SmolLM3-3B"


class SafeSynthesizerParameters(SafeSynthesizerParametersInternal):
    """NHX-facing Safe Synthesizer parameters with SDK convenience flags."""

    enable_synthesis: bool = Field(
        default=True,
        exclude=True,
        description="Whether to run LLM training and generation phases. "
        "When false the task only performs PII replacement and returns the processed data.",
    )
    enable_replace_pii: bool = Field(
        default=True,
        exclude=True,
        description="Whether to run the default PII replacement pipeline before synthesis.",
    )


class SafeSynthesizerJobConfig(SafeSynthesizerJobConfigInternal):
    """NHX-facing Safe Synthesizer job config with SDK convenience flags."""

    __doc__ = SafeSynthesizerJobConfigInternal.__doc__

    config: SafeSynthesizerParameters = Field(
        description="The Safe Synthesizer parameters configuration.",
    )
    pretrained_model_job: str | None = Field(
        default=None,
        description="Optional previous NSS job whose stored adapter artifact is reused for generation-only "
        "synthesis. Accepts either '<job>' in the current workspace or '<workspace>/<job>'. "
        "The plugin resolves the prior job's 'adapter' result from Files.",
    )

    enable_synthesis: SkipJsonSchema[bool] = Field(
        default=True,
        description="Whether to run LLM training and generation phases. "
        "When False the task only performs PII replacement and returns the processed data.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_enable_flags(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cfg = data.get("config")
        if not isinstance(cfg, dict):
            return data
        enable_synthesis = cfg.pop("enable_synthesis", True)
        enable_replace_pii = cfg.pop("enable_replace_pii", True)
        data.setdefault("enable_synthesis", enable_synthesis)
        if not enable_replace_pii:
            cfg["replace_pii"] = None
        return data

    @model_validator(mode="before")
    @classmethod
    def _apply_pii_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        config_data = data.get("config")
        if not isinstance(config_data, dict):
            return data
        replace_pii = config_data.get("replace_pii")
        if not isinstance(replace_pii, dict) or "steps" in replace_pii:
            return data

        def deep_update(base: dict, override: dict) -> dict:
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_update(base[k], v)
                else:
                    base[k] = v
            return base

        default = PiiReplacerConfig.get_default_config().model_dump()
        deep_update(default, replace_pii)
        config_data["replace_pii"] = default
        return data

    @model_validator(mode="before")
    @classmethod
    def _prefer_pretrained_model_job(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        pretrained_model_job = data.get("pretrained_model_job")
        if not pretrained_model_job:
            return data
        config_data = data.get("config")
        training_data = config_data.get("training") if isinstance(config_data, dict) else None
        if not isinstance(training_data, dict):
            return data
        pretrained_model = training_data.get("pretrained_model")
        if pretrained_model is not None and pretrained_model != DEFAULT_PRETRAINED_MODEL:
            logger.warning(
                "Both 'pretrained_model_job' (%s) and 'config.training.pretrained_model' (%s) were supplied; "
                "reusing the adapter from 'pretrained_model_job' and ignoring 'config.training.pretrained_model'.",
                pretrained_model_job,
                pretrained_model,
            )
        training_data.pop("pretrained_model", None)
        return data


def parse_pretrained_model_job_ref(job_ref: str, workspace_fallback: str) -> tuple[str, str]:
    """Parse a previous NSS job reference.

    Accepts either "<job>" in the current workspace or "<workspace>/<job>".
    """
    parts = job_ref.split("/", 1)
    if len(parts) == 1:
        workspace = workspace_fallback
        job_name = parts[0]
    else:
        workspace, job_name = parts

    if not workspace or not job_name:
        raise HelixJobCompilationError(
            f"Invalid pretrained_model_job format: {job_ref!r}. Expected '<job>' or '<workspace>/<job>'."
        )
    return workspace, job_name
