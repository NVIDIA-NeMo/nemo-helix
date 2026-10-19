# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapping-style access helpers for job spec models.

Job spec request shapes are often built and adjusted as nested data objects.
Supporting both attribute access (``step.executor.profile``) and mapping-style
access (``step["executor"]["profile"]``) keeps plugin compilers, API handlers,
and tests ergonomic without requiring callers to care whether a nested shape is
a plain dict or a Pydantic model.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobSpecModel(BaseModel):
    """Base model for job spec request shapes, with a small mapping-style access surface."""

    model_config = ConfigDict(validate_assignment=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.model_fields_set

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            full = self.model_dump(mode="json", exclude_none=True)
            compact = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
            return other == full or other == compact
        return super().__eq__(other)
