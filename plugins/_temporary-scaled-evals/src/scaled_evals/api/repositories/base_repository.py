# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


class RepositoryError(Exception):
    """Base class for repository-domain failures."""


@dataclass(slots=True)
class NotFound(RepositoryError):
    resource: str
    message: str


@dataclass(slots=True)
class Conflict(RepositoryError):
    code: str
    message: str


@dataclass(slots=True)
class InvalidReference(RepositoryError):
    message: str


def normalize_order(order: str) -> str:
    if order not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")
    return order
