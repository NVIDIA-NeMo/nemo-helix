# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.entities.client import EntityBase
from pydantic import Field


class HelloWorldMessage(EntityBase):
    """A message entity for the HelloWorld service."""

    description: str | None = Field(default=None, description="Message description")
    message: str = Field(default="Hello World", description="The greeting message")
