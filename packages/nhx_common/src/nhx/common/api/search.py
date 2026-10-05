# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim — imports from filter.py.

This module existed before the search-to-filter migration. All types and
functions have been renamed (Search* -> Filter*, search -> filter) and moved
to ``nhx.common.api.filter``.  This shim keeps old import paths working
until all callers are migrated.

NOTE: ``parse_bracket_search`` retains the old ``$like`` default for bare
string values, while the new ``parse_bracket_filter`` uses ``$eq``.  This
preserves backward-compatible behavior for services that haven't migrated.
"""

import json
from typing import Any, Dict

from nhx.common.api.filter import (
    ComparisonOperation,
    LogicalOperation,
)
from nhx.common.api.filter import (
    FilterOperation as SearchOperation,
)
from nhx.common.api.filter import (
    FilterOperator as SearchOperator,
)
from nhx.common.api.filter import (
    FilterRepository as SearchRepository,
)
from nhx.common.api.filter import (
    parse_json_filter as parse_json_search,
)


def _apply_implicit_like(d: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap bare string values in ``{"$like": value}`` — the legacy search default."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = {"$like": value}
        else:
            result[key] = value
    return result


def parse_bracket_search(bracket_dict: Dict[str, Any]) -> SearchOperation:
    """Convert bracket-notation dict into a SearchOperation with ``$like`` default.

    Bare string values are implicitly treated as ``$like`` (substring match),
    preserving the legacy search behavior.
    """
    return parse_json_search(json.dumps(_apply_implicit_like(bracket_dict)))


__all__ = [
    "SearchOperator",
    "SearchRepository",
    "SearchOperation",
    "ComparisonOperation",
    "LogicalOperation",
    "parse_json_search",
    "parse_bracket_search",
]
