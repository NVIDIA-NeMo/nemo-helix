# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter dependency helper for garak list endpoints.

``make_filter_obj_dep`` parses ``filter[field]=value`` deepObject query params
into a validated dict.  On an unknown filter key it raises a raw
:class:`pydantic.ValidationError` (``NemoFilter`` is ``extra="forbid"``), which
FastAPI surfaces as a 500.  These endpoints wrap that dependency so a misspelled
or unsupported filter key fails loudly with a 422 instead — typos must not be
silently swallowed.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from pydantic import BaseModel, ValidationError
from starlette.requests import Request


def make_filter_dep(filter_model: type[BaseModel]) -> Callable[[Request], object]:
    """Build a FastAPI dependency that validates filter params and 422s on typos."""
    inner = make_filter_obj_dep(filter_model)

    async def _dep(request: Request) -> object:
        try:
            return await inner(request)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_url=False),
            ) from exc

    return _dep
