# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job schemas for hello world service."""

from pydantic import BaseModel, Field


class HelloWorldJobConfig(BaseModel):
    """Configuration for a hello world job."""

    message: str = Field(default="Hello World", description="The greeting message to write")
