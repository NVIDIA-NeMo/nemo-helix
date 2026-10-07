# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common value types used throughout evals SDK runtime."""

from enum import Enum

from pydantic import Field, RootModel


class SupportedJobTypes(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class SecretRef(RootModel):
    root: str = Field(
        description="Reference to a platform secret or local environment variable. Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
        pattern=r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)?$",
        examples=[
            "my-secret",
            "my-workspace/my-secret",
            "NVIDIA_API_KEY",
        ],
    )
