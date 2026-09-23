# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VirtualModel CRUD endpoints for the Inference Gateway.

VirtualModel is the entity that maps a user-facing model name to an optional
default model entity and ordered middleware pipelines for the request, response,
and post-response phases.  Plugin authors consume these endpoints to manage the
virtual model routes that IGW resolves at inference time.

Endpoints follow the standard NeMo Platform V2 workspace-scoped resource pattern:

    POST   /v2/workspaces/{workspace}/virtual-models
    GET    /v2/workspaces/{workspace}/virtual-models
    GET    /v2/workspaces/{workspace}/virtual-models/{name}
    PATCH  /v2/workspaces/{workspace}/virtual-models/{name}
    DELETE /v2/workspaces/{workspace}/virtual-models/{name}

Cross-workspace listing (``workspace=-``) is handled transparently by
:class:`~nmp.common.entities.client.EntityClient`.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_platform_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_platform_plugin.inference_middleware import (
    InferenceMiddlewareError,
    MiddlewareConfigNotFoundError,
)
from nemo_platform_plugin.inference_middleware_models import (
    GUARDRAIL_CONFIG_IDS_FIELD,
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
    guardrail_config_membership_filter,
)
from nemo_platform_plugin.virtual_models.types import CreateVirtualModelRequest, UpdateVirtualModelRequest
from nmp.common.api.common import Page, PaginationData
from nmp.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nmp.common.api.utils import generate_openapi_extra_params
from nmp.common.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nmp.common.entities.values import DatetimeFilter, Filter, StringFilter, map_entity_field
from nmp.common.service.dependencies import get_entity_client
from nmp.core.inference_gateway.api.authz import enforce_model_refs_access
from nmp.core.inference_gateway.api.dependencies import global_middleware_registry
from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from pydantic import Field

logger = logging.getLogger(__name__)

router = APIRouter()

EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]
MiddlewareRegistryDep = Annotated[MiddlewareRegistry, Depends(global_middleware_registry)]


class VirtualModelFilter(Filter):
    """Filter for VirtualModel list queries."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    project: str | None = Field(None, description="Filter by project URN.")
    name: StringFilter | str | None = Field(None, description="Filter by name.")
    default_model_entity: StringFilter | str | None = Field(None, description="Filter by default model entity.")
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")
    guardrail_config: Annotated[str | None, map_entity_field(GUARDRAIL_CONFIG_IDS_FIELD)] = Field(
        None,
        description=(
            'Filter by a guardrail config reference in "workspace/name" form. Matches VirtualModels '
            "whose request, response, or post-response middleware applies that stored guardrail config."
        ),
    )


def _rewrite_guardrail_filter(operation: FilterOperation | None) -> FilterOperation | None:
    """Rewrite the scalar ``guardrail_config`` equality into a membership match.

    The user-facing param is a single config reference, so it parses to an equality against the
    denormalized list field; "uses this config" actually means "that list contains it", spanning
    both storage representations.
    """
    if operation is None:
        return None
    if isinstance(operation, ComparisonOperation):
        if operation.field == GUARDRAIL_CONFIG_IDS_FIELD and operation.operator == FilterOperator.EQ:
            return guardrail_config_membership_filter(str(operation.value))
        return operation
    if isinstance(operation, LogicalOperation):
        return LogicalOperation(
            operator=operation.operator,
            operations=[
                rewritten for op in operation.operations if (rewritten := _rewrite_guardrail_filter(op)) is not None
            ],
        )
    return operation


def _model_entity_refs(default_model_entity: str | None, models: list[VirtualModelInferenceConfig] | None) -> list[str]:
    refs = [default_model_entity] if default_model_entity else []
    refs.extend(entry.model for entry in models or [])
    return refs


def _middleware_validation_error(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _middleware_config_error(
    detail: str, exc: ValueError | MiddlewareConfigNotFoundError | InferenceMiddlewareError
) -> HTTPException:
    # Order matters: MiddlewareConfigNotFoundError (404) must be checked first
    # because it inherits InferenceMiddlewareError but its status_code < 500,
    # so the 5xx gate below would not match and it would fall through to 422.
    if isinstance(exc, MiddlewareConfigNotFoundError):
        return HTTPException(status_code=exc.status_code, detail=detail)
    # Plugin 5xx errors are server-side failures, not invalid VM input.
    if isinstance(exc, InferenceMiddlewareError) and status.HTTP_500_INTERNAL_SERVER_ERROR <= exc.status_code <= 599:
        return HTTPException(status_code=exc.status_code, detail=detail)
    return _middleware_validation_error(detail)


def _middleware_exception_detail(exc: ValueError | NotImplementedError | InferenceMiddlewareError) -> str:
    if isinstance(exc, InferenceMiddlewareError):
        return exc.detail
    return str(exc)


async def _validate_middleware_configs(
    vm: VirtualModel,
    registry: MiddlewareRegistry,
) -> None:
    """Validate middleware configs before persisting a VirtualModel."""
    vm_id = f"{vm.workspace}/{vm.name}"
    for phase, calls in (
        ("request", vm.request_middleware or []),
        ("response", vm.response_middleware or []),
        ("post_response", vm.post_response_middleware or []),
    ):
        for index, call in enumerate(calls):
            await _validate_middleware_call(vm_id, phase, index, call, registry)


async def _validate_middleware_call(
    vm_id: str,
    phase: str,
    index: int,
    call: MiddlewareCall,
    registry: MiddlewareRegistry,
) -> None:
    # The endpoint can only validate against plugins that IGW successfully loaded.
    plugin = registry.plugins.get(call.name)
    if plugin is None:
        raise _middleware_validation_error(
            f"VirtualModel {vm_id} references unknown plugin {call.name!r} in {phase}_middleware[{index}]."
        )

    config_type = call.config_type or ""
    raw_config = await _load_raw_middleware_config(vm_id, phase, index, call, plugin)

    # The plugin owns schema validation/coercion for its config payload.
    try:
        await plugin.validate_middleware_config(config_type, raw_config)
    except (ValueError, InferenceMiddlewareError) as exc:
        detail = _middleware_exception_detail(exc)
        raise _middleware_config_error(
            f"Config validation failed for plugin {call.name!r} in VirtualModel {vm_id} "
            f"{phase}_middleware[{index}]: {detail}",
            exc,
        ) from exc
    except Exception as exc:
        # Unexpected plugin exceptions are server failures, not user validation errors.
        logger.exception(
            "Plugin %r raised while validating VirtualModel %s %s_middleware[%d]",
            call.name,
            vm_id,
            phase,
            index,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate middleware config.",
        ) from exc


async def _load_raw_middleware_config(
    vm_id: str,
    phase: str,
    index: int,
    call: MiddlewareCall,
    plugin: Any,
) -> Any:
    config_type = call.config_type or ""
    if call.config_id is not None:
        # Stored config path: ask the plugin to resolve the user's config_id.
        try:
            return await plugin.get_middleware_config(config_type, call.config_id)
        except NotImplementedError as exc:
            # The VM used config_id, but this plugin only supports inline configs.
            raise _middleware_validation_error(
                f"Plugin {call.name!r} does not support config_id for "
                f"{phase}_middleware[{index}] on VirtualModel {vm_id}: {exc}"
            ) from exc
        except (ValueError, InferenceMiddlewareError) as exc:
            # Lookup failures can be invalid VM input or plugin-side server failures.
            detail = _middleware_exception_detail(exc)
            raise _middleware_config_error(
                f"Plugin {call.name!r} could not fetch config_id {call.config_id!r} for "
                f"{phase}_middleware[{index}] on VirtualModel {vm_id}: {detail}",
                exc,
            ) from exc
        except Exception as exc:
            # Unexpected plugin exceptions are server failures, not user validation errors.
            logger.exception(
                "Plugin %r raised while fetching config_id %r for VirtualModel %s %s_middleware[%d]",
                call.name,
                call.config_id,
                vm_id,
                phase,
                index,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch middleware config.",
            ) from exc

    # Inline config path: omitted config means the plugin validates an empty dict.
    return call.config or {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v2/workspaces/{workspace}/virtual-models",
    summary="Create VirtualModel",
    response_description="Created virtual model",
    operation_id="create_virtual_model",
    status_code=status.HTTP_201_CREATED,
    response_model=VirtualModel,
    responses={
        201: {"description": "VirtualModel created successfully."},
        409: {"description": "A VirtualModel with that name already exists in the workspace."},
        422: {"description": "Validation error."},
    },
)
async def create_virtual_model(
    workspace: str,
    body: CreateVirtualModelRequest,
    entity_client: EntityClientDep,
    registry: MiddlewareRegistryDep,
) -> VirtualModel:
    """Create a new VirtualModel in the given workspace.

    A VirtualModel defines an ordered middleware pipeline that IGW executes
    when an inference request arrives with ``model: "workspace/name"`` matching
    this entity.
    """
    await enforce_model_refs_access(workspace, _model_entity_refs(body.default_model_entity, body.models))
    entity = VirtualModel(
        name=body.name,
        workspace=workspace,
        default_model_entity=body.default_model_entity,
        autoprovisioned=body.autoprovisioned,
        models=body.models,
        request_middleware=body.request_middleware,
        response_middleware=body.response_middleware,
        post_response_middleware=body.post_response_middleware,
        override_proxy=body.override_proxy,
    )
    await _validate_middleware_configs(entity, registry)
    try:
        return await entity_client.create(entity)
    except EntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"VirtualModel '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create VirtualModel '%s' in workspace '%s'", body.name, workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create VirtualModel.",
        )


@router.get(
    "/v2/workspaces/{workspace}/virtual-models",
    summary="List VirtualModels",
    response_description="Paginated list of virtual models",
    operation_id="list_virtual_models",
    status_code=status.HTTP_200_OK,
    response_model=Page[VirtualModel],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=VirtualModelFilter,
        filter_description=(
            "Filter virtual models by workspace, project, name, default_model_entity, guardrail_config, "
            "created_at, and updated_at."
        ),
    ),
)
async def list_virtual_models(
    workspace: str,
    entity_client: EntityClientDep,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=200, description="Number of results per page."),
    sort: str = Query(
        default="-created_at",
        description="Sort field.  Prefix with ``-`` for descending order.",
    ),
    exclude_autoprovisioned: bool = Query(
        default=False,
        description=(
            "When true, controller-managed (autoprovisioned) passthrough VirtualModels are excluded from the results."
        ),
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(VirtualModelFilter)),
) -> Page[VirtualModel]:
    """List VirtualModels for the given workspace.

    Use ``workspace=-`` to list across all workspaces accessible to the caller.
    """
    filter_workspace = parsed_filter.remove("workspace") or workspace
    # "uses this guardrail config" is a membership test across two storage representations, not the
    # plain equality the scalar param parses to.
    parsed_filter.operation = _rewrite_guardrail_filter(parsed_filter.operation)
    if exclude_autoprovisioned:
        # ``autoprovisioned`` lives under the entity ``data`` blob; reference it
        # directly so this does not depend on VirtualModelFilter exposing the field.
        parsed_filter.and_with(
            ComparisonOperation(operator=FilterOperator.EQ, field="data.autoprovisioned", value=False)
        )
    try:
        result = await entity_client.list(
            VirtualModel,
            workspace=filter_workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list VirtualModels in workspace '%s'", workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list VirtualModels.",
        )
    pagination = PaginationData(
        page=result.pagination.page,
        page_size=result.pagination.page_size,
        current_page_size=len(result.data),
        total_pages=result.pagination.total_pages,
        total_results=result.pagination.total_results,
    )
    return Page(data=result.data, pagination=pagination, sort=sort, filter=parsed_filter.to_response())


@router.get(
    "/v2/workspaces/{workspace}/virtual-models/{name}",
    summary="Get VirtualModel",
    response_description="VirtualModel details",
    operation_id="get_virtual_model",
    status_code=status.HTTP_200_OK,
    response_model=VirtualModel,
    responses={
        404: {"description": "VirtualModel not found."},
    },
)
async def get_virtual_model(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
) -> VirtualModel:
    """Get a VirtualModel by workspace and name."""
    try:
        return await entity_client.get(VirtualModel, name=name, workspace=workspace)
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VirtualModel '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get VirtualModel '%s' in workspace '%s'", name, workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get VirtualModel.",
        )


@router.patch(
    "/v2/workspaces/{workspace}/virtual-models/{name}",
    summary="Update VirtualModel",
    response_description="Updated virtual model",
    operation_id="update_virtual_model",
    status_code=status.HTTP_200_OK,
    response_model=VirtualModel,
    responses={
        404: {"description": "VirtualModel not found."},
        409: {"description": "Concurrent modification conflict."},
        422: {"description": "Validation error."},
    },
)
async def update_virtual_model(
    workspace: str,
    name: str,
    body: UpdateVirtualModelRequest,
    entity_client: EntityClientDep,
    registry: MiddlewareRegistryDep,
) -> VirtualModel:
    """Partially update a VirtualModel.

    Only fields present in the request body are modified.  Fields absent from
    the request body retain their current values.
    """
    try:
        existing = await entity_client.get(VirtualModel, name=name, workspace=workspace)
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VirtualModel '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get VirtualModel '%s' in workspace '%s'", name, workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update VirtualModel.",
        )

    # Pull values directly from the body model so nested pydantic objects
    # (e.g. MiddlewareCall instances) are kept as typed instances rather than
    # being flattened to plain dicts by model_dump(), which would trigger a
    # pydantic serialization warning on model_copy.
    diff = {field: getattr(body, field) for field in body.model_fields_set}
    await enforce_model_refs_access(workspace, _model_entity_refs(diff.get("default_model_entity"), diff.get("models")))
    # model_copy skips validation, so the derived guardrail reference list would otherwise keep the
    # pre-patch pipelines' value. Everything else re-derives it through VirtualModel's validator.
    updated = existing.model_copy(update=diff).refresh_guardrail_config_ids()
    await _validate_middleware_configs(updated, registry)

    try:
        return await entity_client.update(updated)
    except EntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Concurrent modification — please retry.",
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update VirtualModel '%s' in workspace '%s'", name, workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update VirtualModel.",
        )


@router.delete(
    "/v2/workspaces/{workspace}/virtual-models/{name}",
    summary="Delete VirtualModel",
    response_description="VirtualModel deleted",
    operation_id="delete_virtual_model",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"description": "VirtualModel not found."},
        409: {"description": "VirtualModel was modified before it could be deleted."},
    },
)
async def delete_virtual_model(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
    expected_db_version: int | None = Query(
        default=None,
        description="Optional database version for optimistic locking. Delete only succeeds if the VirtualModel still has this version.",
    ),
) -> None:
    """Permanently delete a VirtualModel.

    This does not affect any in-flight requests already being routed through
    this VirtualModel.  IGW's model cache is refreshed on its next polling cycle.
    """
    try:
        await entity_client.delete(
            VirtualModel,
            name=name,
            workspace=workspace,
            expected_db_version=expected_db_version,
        )
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VirtualModel '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except EntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Concurrent modification - please retry.",
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete VirtualModel '%s' in workspace '%s'", name, workspace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete VirtualModel.",
        )
