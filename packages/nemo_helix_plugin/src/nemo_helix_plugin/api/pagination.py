# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP validation for SQL-backed offset pagination."""

from fastapi import HTTPException
from nemo_helix_plugin.entities.pagination import pagination_offset


def validate_pagination(page: int, page_size: int) -> None:
    """Reject an unrepresentable offset before calling downstream services."""
    try:
        pagination_offset(page, page_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
