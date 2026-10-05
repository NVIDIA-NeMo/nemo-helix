# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared value normalization for audit measurement and reporting."""

from __future__ import annotations

from collections.abc import Iterable


def optional_string(value: object) -> str | None:
    """Return stripped non-empty strings and drop every other value."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def dedupe_names(names: Iterable[str]) -> list[str]:
    """Dedupe a name iterable while preserving first-seen order."""
    return list(dict.fromkeys(names))
