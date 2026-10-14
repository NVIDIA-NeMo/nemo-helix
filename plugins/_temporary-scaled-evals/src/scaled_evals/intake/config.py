# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Intake upload targets from an intake config profile + settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from scaled_evals.intake.atif_payload import DEFAULT_SOURCE


class IntakeProfileConfig(BaseModel):
    """Typed, non-secret Intake routing configuration.

    The prefixed names remain accepted for profiles created before the
    canonical unprefixed fields were documented. Unknown metadata is preserved
    because it may be consumed by independently versioned Intake tooling.
    """

    model_config = ConfigDict(extra="allow")

    workspace: StrictStr | None = Field(default=None, min_length=1)
    intake_workspace: StrictStr | None = Field(default=None, min_length=1)
    base_url: StrictStr | None = Field(default=None, min_length=1)
    intake_base_url: StrictStr | None = Field(default=None, min_length=1)
    app: StrictStr | None = Field(default=None, min_length=1)
    intake_app: StrictStr | None = Field(default=None, min_length=1)
    task: StrictStr | None = Field(default=None, min_length=1)
    intake_task: StrictStr | None = Field(default=None, min_length=1)
    capture_content: bool | None = None
    experiment_context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator(
        "workspace",
        "intake_workspace",
        "base_url",
        "intake_base_url",
        "app",
        "intake_app",
        "task",
        "intake_task",
    )
    @classmethod
    def _reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _require_workspace(self) -> IntakeProfileConfig:
        if self.workspace is None and self.intake_workspace is None:
            raise ValueError("intake profile requires workspace")
        if self.workspace is not None and self.intake_workspace is not None and self.workspace != self.intake_workspace:
            raise ValueError("workspace and intake_workspace must match when both are supplied")
        return self


def validate_intake_profile_config(raw_config: dict[str, Any]) -> IntakeProfileConfig:
    """Validate a complete Intake profile at API write boundaries."""

    return IntakeProfileConfig.model_validate(raw_config)


@dataclass(frozen=True)
class IntakeTarget:
    """Where to POST Harbor ATIF trajectories for one evaluation.

    Experiment identity is derived from the run context at upload time (see
    :mod:`scaled_evals.intake.experiments`), not from the intake profile.
    """

    base_url: str
    workspace: str
    app: str
    source: str = DEFAULT_SOURCE


def resolve_intake_target(
    profile_config: dict[str, Any],
    *,
    task_slug: str | None,
    base_url: str,
    source: str = DEFAULT_SOURCE,
) -> IntakeTarget:
    """Build an :class:`IntakeTarget` from a live ``intake`` profile's ``config`` JSON.

    ``workspace`` defaults to ``default`` when omitted (still overridable per profile).
    ``app`` defaults to the task slug, then ``harbor-eval``.

    The profile accepts unprefixed keys (``workspace``, ``app``) and prefixed
    aliases (``intake_base_url``, ``intake_workspace``, ``intake_app``).
    """
    workspace = _string(profile_config.get("workspace"), profile_config.get("intake_workspace"))
    app = _string(profile_config.get("app"), profile_config.get("intake_app"))
    profile_base_url = _string(
        profile_config.get("intake_base_url"),
        profile_config.get("base_url"),
    )
    return IntakeTarget(
        base_url=(profile_base_url or base_url).rstrip("/"),
        workspace=workspace or "default",
        app=app or task_slug or "harbor-eval",
        source=source,
    )


def _string(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
