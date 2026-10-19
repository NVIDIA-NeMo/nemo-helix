# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Intake and Experiments CLI groups."""

from __future__ import annotations

from typing import Any


def without_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in keys}


def list_query_params(**params: Any) -> dict[str, Any] | None:
    """Build a list-endpoint query dict, omitting unset (``None``) keys."""
    query_params = {key: value for key, value in params.items() if value is not None}
    return query_params or None
