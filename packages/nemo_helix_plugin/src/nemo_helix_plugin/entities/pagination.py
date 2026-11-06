# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounds shared by SQL-backed offset pagination."""


def pagination_offset(page: int, page_size: int) -> int:
    """Return the offset when both OFFSET and LIMIT fit signed 64-bit SQL integers."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    if page_size > 2**63 - 1:
        raise ValueError("page_size exceeds the signed 64-bit SQL LIMIT range")
    offset = (page - 1) * page_size
    if offset > 2**63 - 1:
        raise ValueError("page and page_size produce an offset exceeding the signed 64-bit integer range")
    return offset
