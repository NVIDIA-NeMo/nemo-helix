# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enums for the Guardrails service."""

from enum import Enum


class StatusEnum(str, Enum):
    BLOCKED = "blocked"
    SUCCESS = "success"
    UNKNOWN = "unknown"


class RoleEnum(str, Enum):
    EXCEPTION = "exception"
    ASSISTANT = "assistant"
