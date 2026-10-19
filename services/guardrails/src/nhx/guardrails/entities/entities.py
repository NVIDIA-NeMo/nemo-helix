# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity definitions for the Guardrails service."""

from typing import Optional

from nhx.common.entities.client import EntityBase
from pydantic import Field

from .values._private import RailsConfig


class GuardrailConfig(EntityBase):
    """A guardrail configuration entity."""

    __entity_type__ = "guardrail_config"

    description: Optional[str] = Field(default=None, description="Description of the guardrail config")
    data: Optional[RailsConfig] = Field(
        default=None,
        description="Guardrail configuration data",
        json_schema_extra={"type": "object"},
    )
