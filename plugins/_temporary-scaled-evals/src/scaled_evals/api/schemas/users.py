# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class PrincipalInfo(BaseModel):
    source: str
    owner_type: str
    owner_id: str
    username: str | None = None
    display_name: str | None = None
    groups: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


class UserQuotaSummary(BaseModel):
    evaluations_active_max: int
    evaluations_active: int
    tasks_owned: int
    sandbox_slots_max: int
    sandbox_slots_active: int


class CurrentUserResponse(BaseModel):
    id: str
    name: str
    email: str | None = None
    teams: list[str] = Field(default_factory=list)
    quotas: UserQuotaSummary
    principal: PrincipalInfo
    stub: bool = False
