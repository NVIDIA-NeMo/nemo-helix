# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD API for :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`.

When a ``MiddlewareCall`` uses ``config_id`` instead of an inline ``config``
dict, the referenced entity must live somewhere in the entity store.  Plugins
that support ``config_id`` must expose their own CRUD endpoints so operators can
create and manage those entities.  This module shows the full pattern.

Key points:

- The config entity class (:class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`)
  is a :class:`~nemo_helix_plugin.entity.NemoEntity` subclass and is stored in the
  NeMo Helix entity store under ``entity_type="example_middleware_config"``.
- The CRUD endpoints follow the NeMo Helix workspace-scoped resource pattern and
  live under ``/apis/example/v2/workspaces/{workspace}/middleware-configs``.
- Creating or updating a config entity does **not** require editing any
  VirtualModel — IGW picks up the change automatically on the next polling
  cycle.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_example_plugin._perms import ExampleMiddlewareConfigPerms
from nemo_example_plugin.authz import scope
from nemo_example_plugin.middleware_config import ExampleMiddlewareConfig
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CreateExampleMiddlewareConfigRequest(BaseModel):
    """Request body for creating an :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig`."""

    name: str
    blocked_keywords: list[str] = []
    block_message: str = "Your request contains content that is not permitted."


class UpdateExampleMiddlewareConfigRequest(BaseModel):
    """Request body for partially updating an existing config (PATCH semantics).

    Omitted fields retain their current values.
    """

    blocked_keywords: list[str] | None = None
    block_message: str | None = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _get_entity_client() -> NemoEntitiesClient:
    """FastAPI dependency — inject via the platform SDK in real plugins.

    In production use::

        from nemo_helix_plugin.entity_client import get_entity_client
        entity_client: Annotated[NemoEntitiesClient, Depends(get_entity_client)]
    """
    raise NotImplementedError("inject via nemo_helix_plugin.entity_client.get_entity_client")


def build_middleware_config_router() -> APIRouter:
    """Return the router for :class:`~nemo_example_plugin.middleware_config.ExampleMiddlewareConfig` CRUD."""

    router = APIRouter()

    @router.post(
        "/middleware-configs",
        response_model=ExampleMiddlewareConfig,
        status_code=status.HTTP_201_CREATED,
        summary="Create ExampleMiddlewareConfig",
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleMiddlewareConfigPerms.CREATE],
    )
    async def create_config(
        workspace: str,
        body: CreateExampleMiddlewareConfigRequest,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleMiddlewareConfig:
        """Create a new middleware config entity.

        Once created, reference it in a VirtualModel ``MiddlewareCall`` via::

            {"name": "nemo-example-middleware",
             "config_type": "example_middleware_config",
             "config_id": "<workspace>/<name>"}

        IGW resolves the entity on the next polling cycle.  No VirtualModel
        edit is needed when the config is updated — IGW detects the version
        change automatically.
        """
        cfg = ExampleMiddlewareConfig(
            name=body.name,
            workspace=workspace,
            blocked_keywords=body.blocked_keywords,
            block_message=body.block_message,
        )
        try:
            return await entity_client.create(cfg)
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Middleware config '{body.name}' already exists in workspace '{workspace}'.",
            ) from exc
        except Exception:
            logger.exception("Failed to create middleware config '%s'", body.name)
            raise HTTPException(status_code=500, detail="Failed to create middleware config.")

    @router.get(
        "/middleware-configs",
        response_model=list[ExampleMiddlewareConfig],
        summary="List ExampleMiddlewareConfigs",
    )
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleMiddlewareConfigPerms.LIST],
    )
    async def list_configs(
        workspace: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        sort: str = Query(default="-created_at"),
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> list[ExampleMiddlewareConfig]:
        """List middleware configs in *workspace*."""
        try:
            result = await entity_client.list(
                ExampleMiddlewareConfig,
                workspace=workspace,
                page=page,
                page_size=page_size,
                sort=sort,
            )
        except Exception:
            logger.exception("Failed to list middleware configs in workspace '%s'", workspace)
            raise HTTPException(status_code=500, detail="Failed to list middleware configs.")
        return result.data

    @router.get(
        "/middleware-configs/{name}",
        response_model=ExampleMiddlewareConfig,
        summary="Get ExampleMiddlewareConfig",
    )
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleMiddlewareConfigPerms.READ],
    )
    async def get_config(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleMiddlewareConfig:
        """Get a single middleware config by name."""
        try:
            return await entity_client.get(ExampleMiddlewareConfig, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Middleware config '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception:
            logger.exception("Failed to get middleware config '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to get middleware config.")

    @router.patch(
        "/middleware-configs/{name}",
        response_model=ExampleMiddlewareConfig,
        summary="Update ExampleMiddlewareConfig",
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleMiddlewareConfigPerms.UPDATE],
    )
    async def update_config(
        workspace: str,
        name: str,
        body: UpdateExampleMiddlewareConfigRequest,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> ExampleMiddlewareConfig:
        """Partially update a middleware config.

        Only provided fields are changed.  IGW detects the version bump
        (``updated_at``) on the next polling cycle and re-resolves any
        VirtualModel that references this config — no VirtualModel edit needed.
        """
        try:
            cfg = await entity_client.get(ExampleMiddlewareConfig, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Middleware config '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except Exception:
            logger.exception("Failed to fetch middleware config '%s' for update", name)
            raise HTTPException(status_code=500, detail="Failed to fetch middleware config.")

        diff = {k: v for k, v in body.model_dump().items() if v is not None}
        cfg = cfg.model_copy(update=diff)

        try:
            return await entity_client.update(cfg)
        except Exception:
            logger.exception("Failed to update middleware config '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to update middleware config.")

    @router.delete(
        "/middleware-configs/{name}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Delete ExampleMiddlewareConfig",
    )
    @scope.write
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[ExampleMiddlewareConfigPerms.DELETE],
    )
    async def delete_config(
        workspace: str,
        name: str,
        entity_client: NemoEntitiesClient = Depends(_get_entity_client),
    ) -> None:
        """Delete a middleware config.

        VirtualModels that reference this config via ``config_id`` will fail
        to resolve on the next polling cycle — update them to use a different
        config or switch to inline config before deleting.
        """
        try:
            await entity_client.delete(ExampleMiddlewareConfig, name=name, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Middleware config '{name}' not found in workspace '{workspace}'.",
            ) from exc
        except NemoEntityConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Middleware config '{name}' was modified by another request in workspace '{workspace}'. "
                    "Refresh the config and try again."
                ),
            ) from exc
        except Exception:
            logger.exception("Failed to delete middleware config '%s'", name)
            raise HTTPException(status_code=500, detail="Failed to delete middleware config.")

    return router
