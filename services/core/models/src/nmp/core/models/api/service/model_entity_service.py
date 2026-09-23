# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for Model Entity operations using EntityClient."""

import json
import logging
import re
from collections import defaultdict

from nemo_platform import AsyncNeMoPlatform, NotFoundError, PermissionDeniedError
from nemo_platform_plugin.files.storage_config import HuggingfaceStorageConfig, NGCStorageConfig
from nemo_platform_plugin.files.types import FilesetFileOutput, FilesetOutput
from nmp.common.api.common import Page, PaginationData
from nmp.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nmp.common.api.parsed_filter import ParsedFilter
from nmp.common.auth import AuthClient
from nmp.common.entities import ALL_WORKSPACES, ListResponse
from nmp.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nmp.common.sdk_factory import get_async_platform_sdk
from nmp.core.models.api.permissions import can_set_tool_call_plugin, check_fileset_access
from nmp.core.models.config import config
from nmp.core.models.entities import Adapter, Model, ModelDeploymentConfig
from nmp.core.models.schemas import Adapter as AdapterSchema
from nmp.core.models.schemas import (
    CreateModelEntityRequest,
    ModelEntity,
    UpdateModelEntityRequest,
)

logger = logging.getLogger(__name__)


class FilesetValidationError(ValueError):
    """Exception raised for invalid fileset references."""

    pass


class InvalidFilterError(ValueError):
    """Exception raised for filter shapes the service can't honor (e.g. nested cross-entity fields)."""

    pass


def _repo_id_matches_trusted(repo_id: str, patterns: list[str]) -> bool:
    """Return True if repo_id is trusted: direct string match or regex fullmatch for any pattern."""
    for item in patterns:
        if repo_id == item:
            return True
        try:
            if re.fullmatch(item, repo_id) is not None:
                return True
        except re.error:
            # Invalid regex: only direct match counts; already checked above
            pass
    return False


async def get_fileset_and_files_list(
    sdk: AsyncNeMoPlatform, workspace: str, fileset_ref: str | None
) -> tuple[FilesetOutput, list[FilesetFileOutput]]:
    """Validate that the fileset exists and the user has access."""
    if not fileset_ref:
        raise FilesetValidationError("Fileset reference is required")

    try:
        fileset = await check_fileset_access(sdk, fileset_ref, workspace)
        files = await sdk.files.list(workspace=fileset.workspace, fileset=fileset.name)
    except PermissionDeniedError:
        raise PermissionError(f"Access denied to fileset '{fileset_ref}'") from None
    except NotFoundError as err:
        raise FilesetValidationError(f"Fileset {fileset_ref}, does not exist") from err

    if len(files.data) == 0:
        raise FilesetValidationError(f"Fileset {fileset_ref}, exists but is empty")

    return fileset, files.data


async def is_trusted_repo_id(sdk: AsyncNeMoPlatform, workspace: str, fileset_ref: str) -> bool:
    if not config.trust_remote_code.enabled:
        return False

    try:
        fileset, _ = await get_fileset_and_files_list(sdk, workspace, fileset_ref)
        if fileset.storage.type == "huggingface":
            hf: HuggingfaceStorageConfig = fileset.storage
            if _repo_id_matches_trusted(hf.repo_id, config.trust_remote_code.hf_allow_list):
                return True
        if fileset.storage.type == "ngc":
            ns: NGCStorageConfig = fileset.storage
            if _repo_id_matches_trusted(f"{ns.org}/{ns.team}/{ns.target}", config.trust_remote_code.ngc_allow_list):
                return True
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")

    return False


def _adapter_to_adapter_schema(adapter: Adapter, model_name: str = "", model_workspace: str = "") -> AdapterSchema:
    adapter_schema = AdapterSchema(
        name=adapter.name,
        workspace=adapter.workspace if adapter.workspace else model_workspace,
        description=adapter.description,
        fileset=adapter.fileset,
        finetuning_type=adapter.finetuning_type,
        enabled=adapter.enabled,
        lora_config=adapter.lora_config,
        model=adapter.model or model_name,
        created_at=adapter.created_at,
        updated_at=adapter.updated_at,
    )
    return adapter_schema


def _model_to_model_entity(
    model: Model, adapters: dict[str, list[AdapterSchema]] | None = None, verbose: bool = True
) -> ModelEntity:
    """Convert an EntityBase Model to the API ModelEntity schema.

    Adapters are always included when provided. In non-verbose mode, only
    spec.linear_layers is removed to keep response payloads smaller.
    """
    if adapters is None:
        adapters = {}
    adapter_list = adapters.get(model.id, [])
    spec = model.spec
    if spec is not None:
        updates: dict[str, None] = {"minimum_gpus_all_weights": None, "minimum_gpus_lora": None}
        if not verbose:
            updates["linear_layers"] = None
        spec = spec.model_copy(update=updates)

    entity = ModelEntity(
        id=model.id,
        name=model.name,
        workspace=model.workspace,
        project=model.project,
        description=model.description,
        spec=spec,
        fileset=model.fileset,
        finetuning_type=model.finetuning_type,
        base_model=model.base_model,
        api_endpoint=model.api_endpoint,
        backend_format=model.backend_format,
        adapters=adapter_list,
        prompt=model.prompt,
        custom_fields=model.custom_fields or {},
        ownership=model.ownership,
        model_providers=model.model_providers or [],
        trust_remote_code=model.trust_remote_code or False,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
    return entity


def _build_lora_filter_operation(
    lora_model_ids: list[tuple[str, str]],
    lora_exclude: bool,
) -> FilterOperation:
    """Build a workspace-qualified FilterOperation for lora-filtered models.

    Always qualifies by both workspace and name to avoid cross-workspace
    name collisions when querying with workspace=ALL_WORKSPACES. Groups
    (workspace, name) pairs by workspace so each branch becomes a single
    ``workspace=ws AND name IN [...]`` clause; branches are OR-ed together
    and optionally negated with ``$not``.
    """
    ws_names: dict[str, list[str]] = {}
    for ws, name in lora_model_ids:
        ws_names.setdefault(ws, []).append(name)

    or_branches: list[FilterOperation] = [
        LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="workspace", value=ws),
                ComparisonOperation(operator=FilterOperator.IN, field="name", value=names),
            ],
        )
        for ws, names in ws_names.items()
    ]
    match: FilterOperation = LogicalOperation(operator=FilterOperator.OR, operations=or_branches)
    if lora_exclude:
        match = LogicalOperation(operator=FilterOperator.NOT, operations=[match])
    return match


def _has_tool_call_plugin(request) -> bool:
    """Return True if the request (or any nested config) sets tool_call_plugin."""
    spec = getattr(request, "spec", None)
    if spec and getattr(spec, "tool_call_config", None):
        if spec.tool_call_config.tool_call_plugin:
            return True
    model_spec = getattr(request, "model_spec", None)
    if model_spec and getattr(model_spec, "tool_call_config", None):
        if model_spec.tool_call_config.tool_call_plugin:
            return True
    return False


def fileset_has_tool_call_plugin(fileset: FilesetOutput) -> bool:
    """Return True if a fileset's metadata contains a tool_call_plugin value."""
    if not fileset.metadata:
        return False
    if not fileset.metadata.model:
        return False
    tc = fileset.metadata.model.tool_calling
    if not tc:
        return False
    return bool(tc.tool_call_plugin)


async def validate_tool_call_plugin_allowed(auth_client: AuthClient, workspace: str) -> None:
    """Raise if tool_call_plugin is requested but not allowed.

    Checks:
    1. Platform-level enabled flag (models.tool_call_plugin.enabled)
    2. Per-user permission (models.tool-call-plugin.set)
    """
    if not config.tool_call_plugin.enabled:
        raise PermissionError(
            "tool_call_plugin is disabled at the platform level. "
            "Set models.tool_call_plugin.enabled=true in the platform configuration to allow custom parser plugins."
        )

    if not await can_set_tool_call_plugin(auth_client, workspace):
        raise PermissionError(
            "Insufficient permissions to set tool_call_plugin. Requires the models.tool-call-plugin.set permission."
        )


class ModelEntityService:
    """Service layer for Model Entity operations."""

    def __init__(self, entity_client: EntityClient, sdk: AsyncNeMoPlatform | None = None):
        self.entity_client = entity_client
        self.sdk = sdk or get_async_platform_sdk()

    async def _fetch_all_entities(
        self,
        entity_type: type,
        workspace: str,
        filter_str: str | None = None,
    ) -> list:
        """Fetch all entities of a type, paginating past the 1000-entity page limit."""
        first_page: ListResponse = await self.entity_client.list(
            entity_type,
            workspace=workspace,
            filter_str=filter_str,
            page_size=1000,
        )

        all_entities = list(first_page.data)
        if first_page.pagination.total_results > 1000:
            logger.warning(f"Found more than 1000 {entity_type.__name__} entities in workspace {workspace}")
            next_page = first_page.pagination.page + 1
            while next_page <= first_page.pagination.total_pages:
                page_result: ListResponse = await self.entity_client.list(
                    entity_type,
                    workspace=workspace,
                    filter_str=filter_str,
                    page=next_page,
                    page_size=1000,
                )
                all_entities.extend(page_result.data)
                next_page = page_result.pagination.page + 1

        return all_entities

    async def get_adapters(
        self,
        workspace: str,
        ids: list[str],
        model_name_map: dict[str, str] | None = None,
        schema: bool = True,
    ) -> dict[str, list[Adapter]] | dict[str, list[AdapterSchema]]:
        """Resolve adapters parented to the given model entity ``ids``.

        Adapters are queried with ``ALL_WORKSPACES`` because cross-workspace
        adapters can be parented to a base model in a different workspace
        (AALGO-129). Constraining the entity-store query to a single
        ``workspace`` would silently drop those rows from the response, so the
        sidecar would never see them and the corresponding ``{adapter_ws}--{name}``
        directories would never be materialized. Parent ids are globally unique
        in the entity store, so the cross-workspace fan-out cannot return rows
        belonging to other models.

        ``workspace`` is retained as the **legacy** schema fallback passed to
        :func:`_adapter_to_adapter_schema` for adapter rows whose own
        ``workspace`` field is unset (rows written before AALGO-117 introduced
        first-class adapter workspaces). New rows carry their own ``workspace``
        which takes precedence over this fallback.
        """
        if len(ids) == 0:
            return {}

        adapter_filter = json.dumps({"parent": {"$in": ids}})
        all_adapters = await self._fetch_all_entities(Adapter, ALL_WORKSPACES, filter_str=adapter_filter)

        adapter_map = defaultdict(list)
        for a in all_adapters:
            if schema:
                model_name = model_name_map.get(a.parent, a.parent)
                adapter_map[a.parent].append(_adapter_to_adapter_schema(a, model_name, workspace))
            else:
                adapter_map[a.parent].append(a)

        return adapter_map

    async def _resolve_lora_filter(self, workspace: str) -> list[tuple[str, str]]:
        """Query deployment configs, return (workspace, name) pairs for models with lora_enabled on their latest version.

        Returns workspace-qualified pairs to avoid cross-workspace name collisions
        when querying across all workspaces.
        """
        all_configs = await self._fetch_all_entities(ModelDeploymentConfig, workspace)

        # Group by (workspace, base_name), keep only highest entity_version per group
        # to avoid cross-workspace collisions on base_name
        latest_by_key: dict[tuple[str, str], ModelDeploymentConfig] = {}
        for cfg in all_configs:
            key = (cfg.workspace, cfg.base_name)
            existing = latest_by_key.get(key)
            if existing is None or cfg.entity_version > existing.entity_version:
                latest_by_key[key] = cfg

        # From latest versions: filter where lora_enabled is True, extract (workspace, name) pairs
        model_ids: set[tuple[str, str]] = set()
        for cfg in latest_by_key.values():
            if cfg.model_spec is None:
                continue
            if not cfg.model_spec.lora_enabled:
                continue
            if cfg.model_entity_id is None:
                continue
            # model_entity_id format is "workspace/name"
            parts = cfg.model_entity_id.split("/")
            if len(parts) > 1:
                model_ids.add((parts[0], parts[-1]))
            else:
                model_ids.add((cfg.workspace, parts[0]))

        return list(model_ids)

    async def create_model_entity(self, request: CreateModelEntityRequest, workspace: str) -> ModelEntity:
        """Create a new model entity."""
        logger.debug(f"Creating model entity: {workspace}/{request.name}")
        # Check if entity already exists (entities API doesn't enforce uniqueness)
        try:
            existing: Model = await self.entity_client.get(
                Model, name=request.name, workspace=workspace, local_only=True
            )
            if existing:
                logger.warning(f"Model already exists: {workspace}/{request.name}")
                raise ValueError(f"Model with name '{request.name}' already exists in workspace '{workspace}'")
        except EntityNotFoundError:
            pass  # Expected - entity doesn't exist, proceed with creation

        # Create the Model entity
        model = Model(
            name=request.name,
            workspace=workspace,
            project=request.project,
            description=request.description,
            spec=request.spec,
            finetuning_type=request.finetuning_type,
            fileset=request.fileset,
            base_model=str(request.base_model) if request.base_model else None,
            api_endpoint=request.api_endpoint,
            backend_format=request.backend_format,
            prompt=request.prompt,
            custom_fields=request.custom_fields or {},
            ownership=request.ownership,
            model_providers=request.model_providers or [],
            trust_remote_code=request.trust_remote_code,
        )

        try:
            created: Model = await self.entity_client.create(model)
            logger.debug(f"Successfully created model entity: {created.workspace}/{created.name}")
            return _model_to_model_entity(created)
        except EntityConflictError as e:
            logger.warning(f"Model already exists: {workspace}/{request.name}")
            raise ValueError(f"Model with name '{request.name}' already exists in workspace '{workspace}'") from e

    async def get_model_entity(self, workspace: str, name: str, verbose: bool = True) -> ModelEntity | None:
        """Get a model entity by workspace and name."""
        logger.debug(f"Getting model entity: {workspace}/{name}")

        try:
            model: Model = await self.entity_client.get(Model, workspace=workspace, name=name)
            logger.debug(f"Found model entity: {workspace}/{name}")
            model_name_map = {model.id: f"{model.workspace}/{model.name}"}
            adapters_map = await self.get_adapters(
                workspace,
                [model.id],
                model_name_map=model_name_map,
                schema=True,
            )
            return _model_to_model_entity(model, adapters_map, verbose)
        except EntityNotFoundError:
            logger.debug(f"Model entity not found: {workspace}/{name}")
            return None

    async def list_model_entities(
        self,
        workspace: str,
        parsed_filter: ParsedFilter,
        page: int = 1,
        page_size: int = 100,
        sort: str = "created_at",
        verbose: bool = True,
    ) -> Page[ModelEntity]:
        """List model entities with filtering and pagination."""
        logger.debug(f"Listing model entities: page={page}, page_size={page_size}, sort={sort}")

        # Extract lora_enabled (cross-entity filter — must be resolved against
        # ModelDeploymentConfig and merged back into the operation tree).
        lora_enabled = parsed_filter.remove("lora_enabled")

        # Could be lifted by walking the tree and substituting each occurrence with the resolved condition.
        if parsed_filter.has("lora_enabled"):
            raise InvalidFilterError(
                "lora_enabled is only supported as a top-level equality filter; "
                "remove it from $or, $not, or nested expressions."
            )

        if lora_enabled is not None:
            lora_enabled = str(lora_enabled).lower() in ("true", "1", "yes")
            all_lora_ids = await self._resolve_lora_filter(workspace)
            if lora_enabled:
                if not all_lora_ids:
                    # No models have lora — short-circuit with empty result.
                    return Page(
                        data=[],
                        pagination=PaginationData(
                            page=page,
                            page_size=page_size,
                            current_page_size=0,
                            total_pages=0,
                            total_results=0,
                        ),
                        sort=sort,
                        filter=None,
                    )
                parsed_filter.and_with(_build_lora_filter_operation(all_lora_ids, lora_exclude=False))
            elif all_lora_ids:
                # lora_enabled=false: exclude models with lora.
                parsed_filter.and_with(_build_lora_filter_operation(all_lora_ids, lora_exclude=True))
            # If no models have lora and lora_enabled=false, all models qualify — no extra filter.

        result: ListResponse[Model] = await self.entity_client.list(
            Model,
            workspace=workspace,
            filter_operation=parsed_filter.operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )

        logger.debug(f"Found {len(result.data)} model entities")

        model_name_map = {model.id: f"{model.workspace}/{model.name}" for model in result.data}
        adapters_map = await self.get_adapters(
            workspace,
            [r.id for r in result.data],
            model_name_map=model_name_map,
            schema=True,
        )

        # Convert to ModelEntity
        model_entities = [_model_to_model_entity(model, adapters_map, verbose) for model in result.data]

        return Page(
            data=model_entities,
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(model_entities),
                total_pages=result.pagination.total_pages,
                total_results=result.pagination.total_results,
            ),
            sort=sort,
            filter=None,
        )

    async def update_model_entity(
        self, model: Model, workspace: str, name: str, request: UpdateModelEntityRequest, verbose: bool = True
    ) -> ModelEntity | None:
        """Update a model entity."""
        logger.debug(f"Updating model entity: {workspace}/{name}")

        model_name_map = {model.id: f"{model.workspace}/{model.name}"}
        adapters_map = await self.get_adapters(
            workspace,
            [model.id],
            model_name_map=model_name_map,
            schema=True,
        )

        # Apply updates (only non-None values). ``backend_format`` is nullable by
        # contract, so preserve an explicit null when the client wants to clear it.
        update_data = request.model_dump(exclude_none=True)
        if "backend_format" in request.model_fields_set:
            update_data["backend_format"] = request.backend_format

        if "description" in update_data:
            model.description = update_data["description"]
        if "spec" in update_data:
            model.spec = update_data["spec"]
        if "fileset" in update_data:
            await get_fileset_and_files_list(self.sdk, workspace, update_data["fileset"])
            model.fileset = update_data["fileset"]
        if "finetuning_type" in update_data:
            model.finetuning_type = update_data["finetuning_type"]
        if "base_model" in update_data:
            model.base_model = str(update_data["base_model"]) if update_data["base_model"] else None
        if "api_endpoint" in update_data:
            model.api_endpoint = update_data["api_endpoint"]
        if "backend_format" in update_data:
            model.backend_format = update_data["backend_format"]
        if "prompt" in update_data:
            model.prompt = update_data["prompt"]
        if "custom_fields" in update_data:
            model.custom_fields = update_data["custom_fields"]
        if "ownership" in update_data:
            model.ownership = update_data["ownership"]
        if "model_providers" in update_data:
            model.model_providers = update_data["model_providers"]
        # Note: trust_remote_code is assumed to have been permission checked
        if "trust_remote_code" in update_data:
            model.trust_remote_code = update_data["trust_remote_code"]

        updated: Model = await self.entity_client.update(model)
        logger.info(f"Successfully updated model entity: {workspace}/{name}")
        return _model_to_model_entity(updated, adapters_map, verbose)

    async def delete_model_entity(self, workspace: str, name: str) -> bool:
        """Delete a model entity by workspace and name."""
        logger.debug(f"Deleting model entity: {workspace}/{name}")

        try:
            await self.entity_client.delete(Model, name, workspace=workspace)
            logger.info(f"Successfully deleted model entity: {workspace}/{name}")
            return True
        except EntityNotFoundError:
            logger.warning(f"Model entity not found for deletion: {workspace}/{name}")
            return False

    async def model_entity_exists(self, workspace: str, name: str) -> bool:
        """Check if a model entity exists."""
        logger.debug(f"Checking if model entity exists: {workspace}/{name}")

        try:
            await self.entity_client.get(Model, workspace=workspace, name=name)
            return True
        except EntityNotFoundError:
            return False
