# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging helpers shared across plugin API routes and services."""

from __future__ import annotations


def sanitize_for_log(value: object) -> str:
    """Strip line-break/control characters from a value before logging (prevents log injection)."""
    return str(value).replace("\r", "").replace("\n", "")
