# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example plugin service — registered under ``nemo.services``.

Demonstrates two patterns side-by-side:

1. **Minimal route** — the original ``/hello/{name}`` endpoint, showing the
   simplest possible plugin service with a single typed response model.  The
   greeting style is driven by :class:`~nemo_example_plugin.config.ExampleConfig`
   (``NHX_EXAMPLE_GREETING_STYLE``), showing how config shapes route behaviour.

2. **Full entity-backed CRUD** — a complete ``/items`` resource demonstrating:
   - :class:`~nemo_helix_plugin.entity.NemoEntity` for entity definitions
   - :class:`~nemo_helix_plugin.schema.NemoListResponse` for paginated list responses
   - :class:`~nemo_helix_plugin.schema.NemoFilter` for filter query params
   - :class:`~nemo_helix_plugin.entity_client.NemoEntitiesClient` for CRUD operations
   - :class:`~nemo_helix_plugin.entity_client.NemoEntityNotFoundError` → 404
   - :class:`~nemo_helix_plugin.entity_client.NemoEntityConflictError` → 409
   - Pagination query params (``page``, ``page_size``, ``sort``)
   - deepObject filter syntax (``?filter[tag]=ml``)
   - Entity objects returned directly as API responses (no separate response class)
   - Named request body convention (``CreateXRequest`` / ``UpdateXRequest``)
   - ``log_requests`` config flag driving optional structured request logging
"""

from __future__ import annotations

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from nemo_example_plugin._perms import ExampleHelloPerms, ExampleItemPerms
from nemo_example_plugin.authz import scope
from nemo_example_plugin.config import ExampleConfig
from nemo_example_plugin.core import say_hello
from nemo_example_plugin.entities import ExampleItem
from nemo_example_plugin.functions.greet import CountFunction, GreetFunction
from nemo_example_plugin.middleware_service import build_middleware_config_router
from nemo_example_plugin.schema import ExampleItemFilter
from nemo_example_plugin.types.payloads import (
    BlobUploadResponse,
    CreateExampleItemRequest,
    ExampleItemPage,
    HelloResponse,
    UpdateExampleItemRequest,
)
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_helix_plugin.functions.routes import add_function_routes
from nemo_helix_plugin.schema import PaginationData
from nemo_helix_plugin.service import NemoService, RouterSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ExampleService(NemoService):
    """Reference plugin service demonstrating the full NemoPlugin surface.

    Registered under the ``nemo.services`` entry-point group.  The platform
    mounts all routes under ``/apis/example``.
    """

    name: ClassVar[str] = "example"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        # Function routers come from the per-class auto-derivation in
        # ``nemo_helix_plugin.functions.routes``. Mounting them here keeps
        # them alongside the rest of ``/apis/example`` in the OpenAPI
        # tree and lets the existing ExampleService own the URL
        # namespace.
        return [
            RouterSpec(
                _build_hello_router(),
                tag="Example",
                description="Minimal example endpoint.",
            ),
            RouterSpec(
                _build_items_router(),
                tag="Example Items",
                description="Full entity-backed CRUD demonstrating all plugin interfaces.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                build_middleware_config_router(),
                tag="Example Middleware Configs",
                description=(
                    "CRUD for ExampleMiddlewareConfig entities.  "
                    "Reference these via MiddlewareCall.config_id in a VirtualModel."
                ),
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                add_function_routes(
                    GreetFunction,
                    authz=scope,
                    permission_description="Invoke the greet function",
                ),
                tag="Example Functions",
                description="Non-streaming NemoFunction example.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                add_function_routes(
                    CountFunction,
                    authz=scope,
                    permission_description="Invoke the count function",
                ),
                tag="Example Functions",
                description="Streaming NDJSON NemoFunction example.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                _build_binary_router(),
                tag="Example Binary",
                description="Binary upload/download endpoints for testing.",
            ),
        ]


# ---------------------------------------------------------------------------
# Minimal hello router (original example, unchanged)
# ---------------------------------------------------------------------------


def _build_hello_router() -> APIRouter:
    router = APIRouter()

    @router.get("/hello/{name}", response_model=HelloResponse)
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleHelloPerms.READ],
    )
    async def hello(name: str) -> HelloResponse:
        """Greet a name.

        The greeting style is controlled by ``ExampleConfig.greeting_style``:

        - ``"formal"`` (default) → ``"Hello, {name}!"``
        - ``"casual"`` → ``"Hey, {name}!"``

        Override at runtime: ``NHX_EXAMPLE_GREETING_STYLE=casual``.
        """

        config = ExampleConfig.get()
        if config.greeting_style == "casual":
            message = f"Hey, {name}!"
        else:
            message = say_hello(name)
        return HelloResponse(message=message)

    return router


# ---------------------------------------------------------------------------
# Binary upload/download router
# ---------------------------------------------------------------------------


def _build_binary_router() -> APIRouter:
    """Simple binary endpoints for testing the typed client's binary support."""
    router = APIRouter()

    # In-memory store for uploaded bytes (keyed by name)
    _store: dict[str, bytes] = {}

    @router.put("/blob/{name}", status_code=200, response_model=BlobUploadResponse)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def upload_blob(name: str, request: Request) -> BlobUploadResponse:
        """Accept raw binary and store it. Returns byte count."""
        data = await request.body()
        _store[name] = data
        return BlobUploadResponse(name=name, size=len(data))

    @router.get("/blob/{name}", response_class=Response)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def download_blob(name: str) -> Response:
        """Return stored binary content."""
        if name not in _store:
            raise HTTPException(status_code=404, detail=f"Blob '{name}' not found")
        return Response(content=_store[name], media_type="application/octet-stream")

    return router


# ---------------------------------------------------------------------------
# Helper: receive the platform-managed EntityClient dependency
# ---------------------------------------------------------------------------


def _get_entity_client() -> NemoEntitiesClient:
    """FastAPI dependency — returns a request-scoped entity client.

    Imported from nemo_helix_plugin in real plugin code. Defined here inline
    so the example plugin is self-contained and easy to read.
    """
    # In a real plugin, import this from the plugin package:
    #   from nemo_helix_plugin.entity_client import get_entity_client
    #   entity_client: NemoEntitiesClient = Depends(get_entity_client)
    #
    # The platform overrides get_entity_client with a request-scoped client that
    # wires auth context, workspace scoping, and the shared HTTP client. This
    # stub is for illustration only.
    raise NotImplementedError("inject via nemo_helix_plugin.entity_client.get_entity_client")


# ---------------------------------------------------------------------------
# Full CRUD router for ExampleItem entities
# ---------------------------------------------------------------------------


def _build_items_router() -> APIRouter:
    """Build the items router.

    Entity objects are returned directly as API responses — no separate
    response class needed.  Each route shows the complete error-handling pattern:
    - NemoEntityNotFoundError  → 404
    - NemoEntityConflictError  → 409
    - Exception (catch-all)    → 500, with logger.exception for traceability
    """
    router = APIRouter()
    # Build filter dependency once and reuse across routes.
    _filter_dep = make_filter_obj_dep(ExampleItemFilter)

    # ------------------------------------------------------------------
    # POST /items — create
    # ------------------------------------------------------------------

    @router.post(
        "/items",
        response_model=ExampleItem,
        status_code=201,
        tags=["Example Items"],
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleItemPerms.CREATE],
    )
    async def create_item(
        workspace: str,
        body: CreateExampleItemRequest,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleItem:
        """Create a new ExampleItem.

        Returns 409 if an item with the same name already exists in the workspace.
        """
        item = ExampleItem(
            name=body.name,
            workspace=workspace,
            title=body.title,
            body=body.body,
            tags=body.tags,
        )
        try:
            saved = await entity_client.create(item)
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Item '{body.name}' already exists in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to create item '%s'", body.name)
            raise HTTPException(status_code=500, detail="Failed to create item.") from exc

        # Entity object returned directly — id, created_at, etc. are computed
        # fields on EntityBase and are serialised automatically.
        return saved

    # ------------------------------------------------------------------
    # GET /items — paginated list with filter
    # ------------------------------------------------------------------

    @router.get(
        "/items",
        response_model=ExampleItemPage,
        tags=["Example Items"],
    )
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleItemPerms.LIST],
    )
    async def list_items(
        workspace: str,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
        page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
        sort: str = Query(
            default="-created_at",
            description="Sort field.  Prefix with '-' for descending (e.g. '-created_at').",
        ),
        filter: ExampleItemFilter = Depends(_filter_dep),
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleItemPage:
        """List ExampleItems with pagination, sort, and filter support.

        **Pagination query params:**

            ?page=1&page_size=20&sort=-created_at

        **Filter via deepObject syntax** (extra fields rejected with 422):

            ?filter[tag]=ml

        Response shape (identical to the nhx_common ``Page`` envelope)::

            {
                "data": [...],
                "pagination": {
                    "page": 1, "page_size": 20, "current_page_size": 3,
                    "total_pages": 1, "total_results": 3
                },
                "sort": "-created_at",
                "filter": {"tag": "ml"},
                "search": null
            }

        When ``ExampleConfig.log_requests`` is ``True``, emits a structured
        INFO log line for each call (useful for request tracing in dev).
        Enable with ``NHX_EXAMPLE_LOG_REQUESTS=true``.
        """

        config = ExampleConfig.get()
        if config.log_requests:
            logger.info(
                "list_items called  workspace=%r  page=%d  page_size=%d  sort=%r",
                workspace,
                page,
                page_size,
                sort,
            )

        # make_filter_obj_dep returns a NemoFilter instance in normal operation
        # or a raw dict for advanced wildcard filters (?filter[*]=...).
        filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
        try:
            result = await entity_client.list(
                ExampleItem,
                workspace=workspace,
                page=page,
                page_size=page_size,
                sort=sort,
                filter_obj=filter_dict or None,
            )
        except Exception as exc:
            logger.exception("Failed to list items in workspace '%s'", workspace)
            raise HTTPException(status_code=500, detail="Failed to list items.") from exc

        # EntityClient.list() returns PaginationInfo (entity-client-internal);
        # NemoListResponse expects PaginationData (public API layer).
        # They are structurally identical — convert via model_dump.
        pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
        return ExampleItemPage(
            data=result.data,
            pagination=pagination,
            sort=sort,
            filter=filter,
        )

    # ------------------------------------------------------------------
    # GET /items/{name} — single resource
    # ------------------------------------------------------------------

    @router.get(
        "/items/{name}",
        response_model=ExampleItem,
        tags=["Example Items"],
    )
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleItemPerms.READ],
    )
    async def get_item(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleItem:
        """Get a single ExampleItem by name."""
        try:
            item = await entity_client.get(ExampleItem, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to get item '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to get item.") from exc

        return item

    # ------------------------------------------------------------------
    # PATCH /items/{name} — partial update
    # ------------------------------------------------------------------

    @router.patch(
        "/items/{name}",
        response_model=ExampleItem,
        tags=["Example Items"],
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleItemPerms.UPDATE],
    )
    async def update_item(
        workspace: str,
        name: str,
        body: UpdateExampleItemRequest,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleItem:
        """Partially update an ExampleItem.  Omitted fields are unchanged."""
        try:
            item = await entity_client.get(ExampleItem, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to fetch item '%s' for update", name)
            raise HTTPException(status_code=500, detail="Failed to fetch item.") from exc

        # Apply only the fields that were explicitly provided.
        if body.title is not None:
            item.title = body.title
        if body.body is not None:
            item.body = body.body
        if body.tags is not None:
            item.tags = body.tags

        try:
            saved = await entity_client.update(item)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception as exc:
            logger.exception("Failed to update item '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to update item.") from exc

        return saved

    # ------------------------------------------------------------------
    # DELETE /items/{name} — delete (204 No Content)
    # ------------------------------------------------------------------

    @router.delete(
        "/items/{name}",
        status_code=204,
        tags=["Example Items"],
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleItemPerms.DELETE],
    )
    async def delete_item(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> None:
        """Delete an ExampleItem by name.  Returns 204 on success."""
        try:
            await entity_client.delete(ExampleItem, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Item '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Item '{name}' was modified by another request in workspace '{workspace}'. "
                    "Refresh the item and try again."
                ),
            ) from exc
        except Exception as exc:
            logger.exception("Failed to delete item '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to delete item.") from exc

    return router
