# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for platform runner responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthLiveResponse(BaseModel):
    status: Literal["live"] = "live"


class HealthReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"


class HealthNotReadyDetail(BaseModel):
    status: Literal["not_ready"] = "not_ready"


class ClusterInfo(BaseModel):
    platform_version: str = Field(description="Platform version from package metadata or environment.")
    revision: str = Field(description='Code revision, or "dev" when unset.')


class NotReadyServiceInfo(BaseModel):
    name: str
    message: str = ""


class ServiceStatusBreakdown(BaseModel):
    ready: list[str] = Field(default_factory=list)
    not_ready: list[NotReadyServiceInfo] = Field(default_factory=list)


class ControllerStatusBreakdown(BaseModel):
    healthy: bool = Field(True)
    status: dict[str, bool] = Field(default_factory=dict)


class PlatformStatusResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: ServiceStatusBreakdown = Field(default_factory=ServiceStatusBreakdown)
    controllers: ControllerStatusBreakdown = Field(
        default_factory=lambda: ControllerStatusBreakdown(healthy=True, status={})
    )
