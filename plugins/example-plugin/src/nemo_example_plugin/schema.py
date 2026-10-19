# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side schema definitions — filters and other FastAPI-specific models.

Request/response payloads live in :mod:`nemo_example_plugin.types.payloads`.
This module keeps only server-side concerns (filters, etc.) that are tied
to the FastAPI routing layer.
"""

from __future__ import annotations

from nemo_helix_plugin.schema import NemoFilter
from pydantic import Field

# ---------------------------------------------------------------------------
# Filter — extends NemoFilter so extra fields are rejected (extra="forbid")
# ---------------------------------------------------------------------------


class ExampleItemFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/items``.

    ``extra="forbid"`` is inherited from :class:`~nemo_helix_plugin.schema.NemoFilter`.
    A typo like ``?filter[ttle]=foo`` returns a 422 instead of silently
    returning unfiltered results.

    Usage (deepObject query params)::

        GET /items?filter[title]=hello
        GET /items?filter[tag]=ml
    """

    title: str | None = Field(
        default=None,
        description="Filter to items whose title matches this string.",
    )
    tag: str | None = Field(
        default=None,
        description="Filter to items that have this tag.",
    )
