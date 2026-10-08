# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter dependency helper for agent-hardener list endpoints.

Wraps :func:`nemo_platform_plugin.api.filters.make_filter_obj_dep` so an unknown
``filter[field]=value`` key (``NemoFilter`` is ``extra="forbid"``) fails with a
422 instead of the raw ``ValidationError`` FastAPI would otherwise surface as a
500 — typos must fail loudly, not be silently swallowed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import HTTPException
from nemo_platform_plugin.api.filters import make_filter_obj_dep
from pydantic import BaseModel, ValidationError
from starlette.requests import Request


def make_filter_dep(filter_model: type[BaseModel]) -> Callable[[Request], object]:
    """Build a FastAPI dependency that validates filter params and 422s on typos."""
    inner = make_filter_obj_dep(filter_model)

    async def _dep(request: Request) -> object:
        try:
            return await cast(Awaitable[object], inner(request))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc

    return _dep
