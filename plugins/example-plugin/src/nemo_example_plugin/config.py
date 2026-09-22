# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Example plugin.

Demonstrates the :class:`~nemo_helix_plugin.config.NemoConfig` pattern: declare
:attr:`plugin_name` and :attr:`plugin_description` as ``ClassVar`` strings, then
add plugin-specific fields as regular Pydantic fields.

Operators set values via environment variables or the Helm ``platformConfig`` key:

    # Environment variables (highest priority)
    NHX_EXAMPLE_GREETING_STYLE=casual
    NHX_EXAMPLE_LOG_REQUESTS=true

    # Helm values.yaml (platformConfig key)
    platformConfig:
      example:
        greeting_style: casual
        log_requests: true
"""

from __future__ import annotations

from typing import ClassVar, Literal

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field


class ExampleConfig(NemoConfig):
    """Configuration for the NeMo Helix example plugin.

    All fields have defaults so the plugin runs out-of-the-box without any
    operator configuration.  Override via environment variables or the Helm
    ``platformConfig.example`` section.
    """

    plugin_name: ClassVar[str] = "example"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Helix example plugin."

    greeting_style: Literal["formal", "casual"] = Field(
        default="formal",
        description=(
            "Controls the tone of /hello/{name} responses. "
            '"formal" → "Hello, {name}!"  "casual" → "Hey, {name}!"'
            "  Set NHX_EXAMPLE_GREETING_STYLE to override."
        ),
    )
    log_requests: bool = Field(
        default=False,
        description=(
            "When True, each request to the items list endpoint emits a "
            "structured INFO log line including the workspace and page number. "
            "Set NHX_EXAMPLE_LOG_REQUESTS=true to enable."
        ),
    )
