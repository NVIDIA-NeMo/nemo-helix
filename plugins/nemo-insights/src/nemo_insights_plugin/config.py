# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the NeMo Insights plugin."""

from enum import IntEnum, StrEnum
from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, field_validator


class Frequency(StrEnum):
    """How often each opted-in agent is analyzed."""

    DAILY = "daily"
    WEEKLY = "weekly"


class Weekday(IntEnum):
    """Day of week for weekly analysis, with Python's 0=Monday convention."""

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @classmethod
    def _missing_(cls, value: object) -> "Weekday | None":
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                return None
        return None


class AnalystSchedulerConfig(BaseModel):
    """Framework-managed periodic analysis settings."""

    enabled: bool = Field(
        default=True,
        description="Whether the insights periodic analysis controller should run.",
    )
    frequency: Frequency = Field(
        default=Frequency.DAILY,
        description="Global cadence for analyzing each opted-in agent.",
    )
    run_at_hour: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Local hour-of-day (0-23, in `timezone`) scheduled runs fire.",
    )
    run_on_weekday: Weekday = Field(
        default=Weekday.MONDAY,
        description="Day of week scheduled runs fire. Used only when frequency is weekly.",
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name (e.g. 'America/Denver') the schedule is "
            "interpreted in. Converted to the server clock at evaluation time "
            "so runs fire at the intended local hour across DST changes."
        ),
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value!r}") from exc
        return value

    job_profile: str = Field(
        default="default",
        description="Jobs execution profile used for scheduled analyst jobs.",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Optional platform base URL passed to analyst jobs. When unset, jobs use their active platform context."
        ),
    )


class InsightsConfig(NemoConfig):
    """Configuration for the NeMo Insights plugin."""

    plugin_name: ClassVar[str] = "insights"
    plugin_description: ClassVar[str] = "Configuration for NeMo Insights."

    analyst: AnalystSchedulerConfig = Field(
        default_factory=AnalystSchedulerConfig,
        description="Periodic insights analyst settings.",
    )
