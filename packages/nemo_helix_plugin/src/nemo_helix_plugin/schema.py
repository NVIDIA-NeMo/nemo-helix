# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin API schema interfaces — base classes for request/response models.

Plugin authors import from here rather than ``nhx.common`` directly:

    from nemo_helix_plugin.schema import NemoFilter, NemoListResponse, PaginationData

Typical usage
-------------

**Single-resource response** (GET /{name}, POST, PATCH) — return entity objects directly::

    from nemo_helix_plugin.entity import NemoEntity

    class Widget(NemoEntity, entity_type="my_plugin_widget"):
        colour: str
        weight_kg: float = 0.0

    # Entity objects are returned directly — no separate response class needed:
    @router.post("/widgets", response_model=Widget, status_code=201)
    async def create_widget(
        workspace: str, body: CreateWidgetRequest,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> Widget:
        widget = Widget(name=body.name, workspace=workspace, colour=body.colour)
        saved = await entity_client.create(widget)
        return saved

**List response** (GET /) — ``NemoListResponse[Widget]`` wraps the entity list::

    from nemo_helix_plugin.schema import NemoFilter, NemoListResponse, PaginationData

    class WidgetFilter(NemoFilter):
        colour: str | None = None  # extra="forbid" inherited — typos → 422

    WidgetPage = NemoListResponse[Widget]

    @router.get("/widgets", response_model=WidgetPage)
    async def list_widgets(...) -> WidgetPage:
        result = await entity_client.list(Widget, workspace=workspace, ...)
        pagination = (
            PaginationData.model_validate(result.pagination.model_dump())
            if result.pagination else None
        )
        return WidgetPage(data=result.data, pagination=pagination, sort=sort, filter=filter)

**Request bodies** — plain Pydantic ``BaseModel``, no shared base::

    from pydantic import BaseModel

    class CreateWidgetRequest(BaseModel):
        name: str
        colour: str
        weight_kg: float = 0.0

    class UpdateWidgetRequest(BaseModel):
        colour: str | None = None
        weight_kg: float | None = None

**Custom response models** — for non-CRUD endpoints or computed responses, use a
plain ``BaseModel``::

    from pydantic import BaseModel

    class WidgetStatsResponse(BaseModel):
        total: int
        avg_weight_kg: float
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

__all__ = [
    "DatetimeFilter",
    "Filter",
    "NemoFilter",
    "NemoListResponse",
    "Page",
    "PaginationData",
    "SecretRef",
    "StringFilter",
    "Value",
]

T = TypeVar("T", bound=BaseModel)


class Value(BaseModel):
    """The base class for all value types.

    This also helps avoid confusion since Model and BaseModel in Pydantic
    mean something different.
    """

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}


class SecretRef(RootModel):
    root: str = Field(
        description="Reference to a secret. Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
        pattern=r"^[a-z0-9_-]+(/[a-z0-9_-]+)?$",
        examples=[
            "my-secret",
            "my-workspace/my-secret",
        ],
    )


class PaginationData(Value):
    page: int = Field(description="The current page number.")
    page_size: int = Field(description="The page size used for the query.")
    current_page_size: int = Field(description="The size for the current page.")
    total_pages: int = Field(description="The total number of pages.")
    total_results: int = Field(description="The total number of results.")


class Filter(BaseModel):
    """Base class for filter schemas.

    We don't allow extra fields because filtering on the wrong property
    (e.g., spelling error) would fail silently.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def skip_validation_on_raw_objects(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("*"):
            data = {}
        return data


NemoFilter = Filter


class NemoListResponse(BaseModel, Generic[T]):
    """Paginated list response envelope.

    Wire format is identical to the ``nhx_common`` ``Page`` envelope used
    by all core NeMo Helix services::

        {
            "data": [...],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "current_page_size": 5,
                "total_pages": 1,
                "total_results": 5
            },
            "sort": "-created_at",
            "filter": {"colour": "red"}
        }

    Usage::

        WidgetPage = NemoListResponse[Widget]

        @router.get("/widgets", response_model=WidgetPage)
        async def list_widgets(
            workspace: str,
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=100),
            sort: str = Query(default="-created_at"),
            filter: WidgetFilter = Depends(make_filter_dep(WidgetFilter)),
            entity_client: NemoEntitiesClient = Depends(get_entity_client),
        ) -> WidgetPage:
            result = await entity_client.list(
                Widget, workspace=workspace,
                page=page, page_size=page_size, sort=sort, filter=filter,
            )
            pagination = (
                PaginationData.model_validate(result.pagination.model_dump())
                if result.pagination else None
            )
            return WidgetPage(data=result.data, pagination=pagination, sort=sort, filter=filter)
    """

    data: list[T]
    pagination: PaginationData | None = Field(
        default=None,
        description="Pagination metadata — page, page_size, total_results, etc.",
    )
    sort: str | None = Field(
        default=None,
        description="Sort field applied to this result set (e.g. '-created_at').",
    )
    filter: Any = Field(
        default=None,
        description="Filter criteria echoed back from the request.",
    )


EntityT = TypeVar("EntityT")


class Page(Value, Generic[EntityT]):
    data: list[EntityT]
    pagination: Optional[PaginationData] = Field(default=None, description="Pagination information.")
    sort: Optional[str] = Field(default=None, description="The field on which the results are sorted.")
    filter: Optional[dict] = Field(default=None, description="Filtering information.")


class DatetimeFilter(Filter):
    gte: Optional[datetime] = Field(
        None,
        alias="$gte",
        serialization_alias="$gte",
        description="Filter for results greater than or equal to this datetime.",
    )
    lte: Optional[datetime] = Field(
        None,
        alias="$lte",
        serialization_alias="$lte",
        description="Filter for results less than or equal to this datetime.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )


class StringFilter(Filter):
    eq: Optional[str] = Field(
        None,
        alias="$eq",
        serialization_alias="$eq",
        description="Filter for results equal to this value.",
    )
    like: Optional[str] = Field(
        None,
        alias="$like",
        serialization_alias="$like",
        description="Filter for results matching this pattern.",
    )
    in_: Optional[list[str]] = Field(
        None,
        alias="$in",
        serialization_alias="$in",
        description="Filter for results in this list of values.",
    )
    nin: Optional[list[str]] = Field(
        None,
        alias="$nin",
        serialization_alias="$nin",
        description="Filter for results not in this list of values.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )
