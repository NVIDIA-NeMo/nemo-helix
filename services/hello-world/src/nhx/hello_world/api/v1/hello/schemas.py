# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello API schemas."""

from pydantic import BaseModel


class HelloResponse(BaseModel):
    """Response schema for hello endpoint."""

    message: str


class ConfigInfoResponse(BaseModel):
    """Response schema for config-info endpoint demonstrating config DI."""

    platform_base_url: str
    greeting_prefix: str
    max_message_length: int
