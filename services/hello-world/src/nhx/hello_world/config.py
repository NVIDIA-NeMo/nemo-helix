# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello World service configuration."""

from nhx.common.config import create_service_config_class
from pydantic import Field


class HelloWorldConfig(create_service_config_class("hello_world")):  # ty: ignore[unsupported-base]
    """Configuration for the Hello World service.

    This configuration is loaded from the 'hello_world' section of the
    global config file or from environment variables with the prefix
    NHX_HELLO_WORLD_.
    """

    greeting_prefix: str = Field(
        default="Hello",
        description="Prefix to use for greeting messages.",
    )
    max_message_length: int = Field(
        default=100,
        description="Maximum length for messages.",
    )
