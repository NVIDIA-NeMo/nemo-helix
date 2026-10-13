# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for building entity store search queries.

Provides helpers that translate service-layer filter values into the correct
entity store query operators, so callers don't have to hand-code operator dicts.
"""

from typing import Any


def coerce_existence_operator(value: Any) -> Any:
    """Coerce a filter value for a nullable (non-boolean) field into the correct
    entity-store query operator.

    When a field like ``finetuning_type`` or ``prompt`` is logically boolean
    (present/non-null = True, absent/null = False), boolean inputs are coerced
    to existence-check operators:

    - ``False`` → ``{"$eq": None}``   (field is absent or null)
    - ``True``  → ``{"$not": {"$eq": None}}`` (field is present and non-null)

    Non-boolean values pass through unchanged for direct matching.

    Examples::

        coerce_existence_operator(False)    # {"$eq": None}
        coerce_existence_operator(True)     # {"$not": {"$eq": None}}
        coerce_existence_operator("sft")    # "sft"
        coerce_existence_operator(None)     # None
    """
    if isinstance(value, bool):
        if value:
            return {"$not": {"$eq": None}}
        return {"$eq": None}
    return value
