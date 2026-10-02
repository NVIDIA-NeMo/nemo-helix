# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Prompt Master Fabric executions."""

from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class PromptMasterConfigError(ValueError):
    """Raised when a Prompt Master configuration cannot be loaded."""


class PromptMasterModelConfig(BaseModel):
    """Model used by the Fabric optimizer agent."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_deepagents_model(self) -> Self:
        provider = self.provider.strip().lower()
        if provider in {"nvidia", "openai-compatible"} and not self.base_url:
            raise ValueError(f"model.base_url is required for provider {self.provider!r}")
        if self.base_url and urlparse(self.base_url).hostname == "build.nvidia.com":
            raise ValueError(
                "model.base_url must be an OpenAI-compatible API root, such as "
                "'https://integrate.api.nvidia.com/v1'; 'https://build.nvidia.com/' is the web UI"
            )
        if provider != "openai" and not self.api_key_env:
            raise ValueError(f"model.api_key_env is required for provider {self.provider!r}")
        return self


class PromptMasterConfig(BaseModel):
    """Inputs for one Prompt Master optimization run."""

    model_config = ConfigDict(extra="forbid")

    model: PromptMasterModelConfig
    prompt_override: str | None = None
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_non_blank_strings(self) -> Self:
        if self.prompt_override is not None and not self.prompt_override.strip():
            raise ValueError("prompt_override must contain non-whitespace text")
        if not self.model.provider.strip():
            raise ValueError("model.provider must contain non-whitespace text")
        if not self.model.model.strip():
            raise ValueError("model.model must contain non-whitespace text")
        return self


def load_prompt_master_config(path: str | Path) -> PromptMasterConfig:
    """Load and validate a YAML or JSON Prompt Master config."""
    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PromptMasterConfigError(f"Unable to read Prompt Master config {config_path}: {exc}") from exc

    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PromptMasterConfigError(f"YAML parse error in Prompt Master config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromptMasterConfigError(f"Prompt Master config {config_path} root must be a mapping.")

    try:
        return PromptMasterConfig.model_validate(payload)
    except ValidationError as exc:
        raise PromptMasterConfigError(f"Invalid Prompt Master config {config_path}: {exc}") from exc
