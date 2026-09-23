# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema definitions for HelloWorld message entities."""

from pydantic import BaseModel, Field


class CreateHelloWorldMessageRequest(BaseModel):
    """Input schema for creating a HelloWorld message."""

    name: str = Field(..., description="Message name")
    description: str | None = Field(default=None, description="Message description")
    message: str = Field(..., description="The greeting message")


class UpdateHelloWorldMessageRequest(BaseModel):
    """Input schema for updating a HelloWorld message."""

    description: str | None = Field(default=None, description="Message description")
    message: str | None = Field(default=None, description="The greeting message")
