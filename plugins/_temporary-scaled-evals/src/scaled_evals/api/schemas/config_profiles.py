# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConfigProfileType = Literal["harbor", "gym", "intake"]

# NOTE: a harbor profile may not be needed long-term — intake is the real reuse case.

_CONFIG_DESCRIPTION = (
    "Profile-specific non-secret configuration. OpenAPI intentionally represents this "
    "as an extensible JSON object because the sibling type field selects the schema and "
    "Harbor supports independently versioned fields. "
    "The API validates all known fields by profile type: Harbor runner envelopes, strict "
    "Gym v1 configs, and Intake routing configs "
    "with a required workspace. See docs/API.md#config-profiles for each contract."
)


# Request body: POST /v1/config-profiles
class ConfigProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    type: ConfigProfileType = Field(
        description=(
            "Profile kind. Framework profiles use 'harbor' for framework='harbor' "
            "and 'gym' for framework='nemo_gym'; 'intake' "
            "profiles wire optional observability config."
        )
    )
    config: dict[str, Any] = Field(default_factory=dict, description=_CONFIG_DESCRIPTION)


# Request body: PATCH /v1/config-profiles/{id}. Type is immutable; only name
# and config are mutable.
class ConfigProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    config: dict[str, Any] | None = Field(default=None, description=_CONFIG_DESCRIPTION)


# Response: GET /v1/config-profiles/{id} and list items
class ConfigProfile(BaseModel):
    id: str
    name: str
    type: ConfigProfileType = Field(description=("Profile kind: 'harbor', 'gym', or 'intake'."))
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
