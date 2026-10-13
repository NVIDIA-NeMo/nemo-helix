# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment-tracking integrations schema shared across platform plugins and services."""

from __future__ import annotations

import warnings
from typing import Self

from nemo_helix_plugin.schema import SecretRef
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WandbIntegration(BaseModel):
    """Weights & Biases integration configuration.

    To enable W&B, provide a non-null ``wandb`` object on :class:`IntegrationsSpec`.
    Provide ``api_key_secret`` referencing a secret that contains ``WANDB_API_KEY``.
    Optionally set ``base_url`` for self-hosted W&B servers.
    """

    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(
        default=None,
        description="W&B project name (groups related runs). Defaults to output.name if not set.",
    )
    name: str | None = Field(
        default=None,
        description="W&B run name. Defaults to job_id if not provided.",
    )
    entity: str | None = Field(
        default=None,
        description="W&B entity (team or username).",
    )
    tags: list[str] | None = Field(
        default=None,
        description="W&B tags for filtering runs.",
    )
    notes: str | None = Field(
        default=None,
        description="W&B notes/description for the run.",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for self-hosted W&B server (e.g., 'https://wandb.mycompany.com'). "
        "If not provided, uses the default W&B cloud service.",
    )
    api_key_secret: SecretRef | None = Field(
        default=None,
        description="Reference to a secret containing the WANDB_API_KEY. "
        "Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "run_name" in normalized:
            warnings.warn(
                "integrations.wandb.run_name is deprecated; use 'name' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Always drop the legacy key so extra="forbid" can't trip when both
            # are present; only adopt it as 'name' when 'name' isn't already set.
            run_name = normalized.pop("run_name")
            if normalized.get("name") is None:
                normalized["name"] = run_name
        return normalized


class MlflowIntegration(BaseModel):
    """MLflow integration configuration.

    To enable MLflow, provide a non-null ``mlflow`` object on :class:`IntegrationsSpec`.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "run_name" in normalized:
            warnings.warn(
                "integrations.mlflow.run_name is deprecated; use 'name' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Always drop the legacy key so extra="forbid" can't trip when both
            # are present; only adopt it as 'name' when 'name' isn't already set.
            run_name = normalized.pop("run_name")
            if normalized.get("name") is None:
                normalized["name"] = run_name
        return normalized

    experiment_name: str | None = Field(
        default=None,
        description="MLflow experiment name (groups related runs). Defaults to output.name if not set.",
    )
    name: str | None = Field(
        default=None,
        description="MLflow run name. Defaults to job_id if not provided.",
    )
    tags: dict[str, str] | None = Field(
        default=None,
        description="MLflow tags as key-value pairs for filtering runs.",
    )
    description: str | None = Field(
        default=None,
        description="MLflow run description.",
    )
    tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking server URI (e.g., 'http://mlflow.mycompany.com:5000'). "
        "Can also be set via MLFLOW_TRACKING_URI environment variable.",
    )


class IntegrationsSpec(BaseModel):
    """Third-party experiment-tracking integrations for a job spec.

    Each integration is requested by presence: omit or set a field to ``null`` to
    disable it. Activation at training time still requires credentials/URIs
    (see compile-time warnings and runtime builders).
    """

    model_config = ConfigDict(extra="forbid")

    wandb: WandbIntegration | None = Field(
        default=None,
        description="Weights & Biases integration configuration.",
    )
    mlflow: MlflowIntegration | None = Field(
        default=None,
        description="MLflow integration configuration.",
    )

    @model_validator(mode="after")
    def _reject_empty_integration_blocks(self) -> Self:
        """Reject empty objects that look like accidental toggles."""
        if self.wandb is not None and not self.wandb.model_dump(exclude_none=True):
            raise ValueError(
                "integrations.wandb must include at least one configuration field or be omitted/null to disable W&B."
            )
        if self.mlflow is not None and not self.mlflow.model_dump(exclude_none=True):
            raise ValueError(
                "integrations.mlflow must include at least one configuration field "
                "or be omitted/null to disable MLflow."
            )
        return self
