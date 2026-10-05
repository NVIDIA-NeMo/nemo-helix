# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for Adapter (sub) entities using EntityClient."""

import json
import logging

from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.entities import ALL_WORKSPACES, ListResponse
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.models.api.service.model_entity_service import _adapter_to_adapter_schema, get_fileset_and_files_list
from nhx.core.models.constants import parse_model_ref
from nhx.core.models.entities import Adapter, Model
from nhx.core.models.schemas import Adapter as AdapterSchema
from nhx.core.models.schemas import CreateModelAdapterRequest, UpdateAdapterRequest

logger = logging.getLogger(__name__)


class AdapterEntityService:
    """Service for adapter CRUD, scoped to a workspace, with model reference from path or body."""

    def __init__(self, entity_client: EntityClient, files: AsyncFilesClient) -> None:
        self.entity_client = entity_client
        self.files = files

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

    async def _adapters_by_name(self, adapter_workspace: str, adapter_name: str) -> list[Adapter]:
        """List all adapter entity rows in the workspace with this name (app-layer filter, not parent-scoped)."""
        filter_str = json.dumps({"name": adapter_name})
        all_matching = await self._fetch_all_entities(Adapter, adapter_workspace, filter_str=filter_str)
        return [a for a in all_matching if a.name == adapter_name]

    async def _find_adapter(self, adapter_workspace: str, adapter_name: str, parent_id: str) -> Adapter | None:
        """Find an adapter by name in the workspace."""
        criteria: dict[str, str] = {"name": adapter_name}
        if parent_id:
            criteria["parent"] = parent_id
        filter_str = json.dumps(criteria)
        all_matching = await self._fetch_all_entities(Adapter, adapter_workspace, filter_str=filter_str)
        return all_matching[0] if all_matching else None

    async def _assert_adapter_name_free(self, adapter_workspace: str, request_name: str) -> None:
        """For create: fail if any adapter in this workspace already uses request_name."""
        existing = await self._adapters_by_name(adapter_workspace, request_name)
        if existing:
            raise ValueError(f"Adapter with name {request_name!r} already exists in workspace {adapter_workspace!r}")

    async def create_adapter(
        self,
        adapter_workspace: str,
        request: CreateModelAdapterRequest,
        base_model: str,
    ):
        try:
            model_ws, model_name = parse_model_ref(base_model, adapter_workspace)
            model: Model = await self.entity_client.get(Model, workspace=model_ws, name=model_name)
        except ValueError as err:
            raise ValueError(str(err)) from err
        except EntityNotFoundError as err:
            logger.warning(f"Model entity not found for adapter create: {base_model!r} - {err}")
            return None

        await self._assert_adapter_name_free(adapter_workspace, request.name)

        await get_fileset_and_files_list(self.files, adapter_workspace, request.fileset)

        adapter = Adapter(
            workspace=adapter_workspace,
            name=request.name,
            description=request.description,
            fileset=request.fileset,
            finetuning_type=request.finetuning_type,
            enabled=request.enabled,
            lora_config=request.lora_config,
            model=f"{model_ws}/{model_name}",
        )
        adapter._parent = model.id

        try:
            adapter_entity: Adapter = await self.entity_client.create(adapter)
        except EntityConflictError as err:
            raise ValueError(
                f"Adapter with name '{request.name}' already exists in workspace '{adapter_workspace}'"
            ) from err

        logger.info(f"Successfully created adapter entity: {adapter_workspace}/{model_name}/{request.name}")

        return _adapter_to_adapter_schema(adapter_entity)

    async def _update_adapter_from_request(
        self, adapter: Adapter, adapter_workspace: str, request: UpdateAdapterRequest
    ) -> AdapterSchema:
        """Apply ``request`` to ``adapter`` and persist. Callers must resolve the row first."""
        if request.description is not None:
            adapter.description = request.description
        if request.enabled is not None:
            adapter.enabled = request.enabled
        if request.fileset is not None:
            await get_fileset_and_files_list(self.files, adapter_workspace, request.fileset)
            adapter.fileset = request.fileset
        updated = await self.entity_client.update(adapter)

        model: Model = await self.entity_client.get_by_id(Model, adapter.parent)
        return _adapter_to_adapter_schema(
            updated,
            f"{model.workspace}/{model.name}",
            model.workspace,
        )

    async def update_adapter(
        self,
        adapter_workspace: str,
        parent_model_ref: str,
        adapter_name: str,
        request: UpdateAdapterRequest,
    ) -> int | AdapterSchema:
        try:
            model_ws, model_n = parse_model_ref(parent_model_ref, adapter_workspace)
            model: Model = await self.entity_client.get(Model, workspace=model_ws, name=model_n)
        except ValueError as err:
            raise ValueError(str(err)) from err
        except EntityNotFoundError:
            logger.warning(f"Model entity not found for update: {parent_model_ref}")
            return -1

        adapter = await self._find_adapter(adapter_workspace, adapter_name, model.id)

        if not adapter:
            logger.warning(f"Adapter not found for update: {adapter_workspace}/{parent_model_ref}/{adapter_name}")
            return -2

        result = await self._update_adapter_from_request(adapter, adapter_workspace, request)
        logger.info(f"Successfully updated model entity: {adapter_workspace}/{parent_model_ref}/{adapter_name}")
        return result

    async def delete_adapter(self, adapter_workspace: str, parent_model_ref: str, adapter_name: str) -> int:
        try:
            model_ws, model_n = parse_model_ref(parent_model_ref, adapter_workspace)
            model: Model = await self.entity_client.get(Model, workspace=model_ws, name=model_n)
        except ValueError as err:
            raise ValueError(str(err)) from err
        except EntityNotFoundError:
            logger.warning(f"Model entity not found for adapter deletion: {parent_model_ref}")
            return -1

        adapter = await self._find_adapter(adapter_workspace, adapter_name, model.id)

        if not adapter:
            logger.warning(
                f"Adapter with name '{adapter_name}' not found for model '{parent_model_ref}' in {adapter_workspace}"
            )
            return -2

        await self.entity_client.delete(
            Adapter,
            adapter.name,
            workspace=adapter_workspace,
            parent=adapter.parent,
        )
        logger.info(
            f"Successfully deleted adapter {adapter_name} from model entity: {adapter_workspace}/{parent_model_ref}"
        )
        return 0

    async def _load_unique_adapter_in_workspace(self, adapter_workspace: str, adapter_name: str) -> Adapter | None:
        """Return the adapter in this workspace, or None if not found. Raises ValueError if ambiguous."""
        exact = await self._adapters_by_name(adapter_workspace, adapter_name)
        if not exact:
            return None
        if len(exact) > 1:
            raise ValueError(
                f"Adapter name {adapter_name!r} is ambiguous in workspace {adapter_workspace!r}; use the nested "
                "routes /v2/workspaces/.../models/.../adapters/... that include the model in the path."
            )
        return exact[0]

    async def get_adapter(self, adapter_workspace: str, adapter_name: str) -> AdapterSchema | None:
        """Return a single adapter by (workspace, name), or None if not found. Raises on ambiguous name.

        Top-level Adapters API update/delete use the child entity and its parent id in the
        entity store, without a separate model query parameter.
        """
        adapter = await self._load_unique_adapter_in_workspace(adapter_workspace, adapter_name)
        if adapter is None:
            return None
        # Deal with old adapters without model field
        if adapter.model:
            return _adapter_to_adapter_schema(adapter)

        model: Model = await self.entity_client.get_by_id(Model, adapter.parent)
        return _adapter_to_adapter_schema(adapter, f"{model.workspace}/{model.name}", model.workspace)

    async def update_adapter_in_workspace(
        self, adapter_workspace: str, adapter_name: str, request: UpdateAdapterRequest
    ) -> int | AdapterSchema:
        try:
            adapter: Adapter | None = await self._load_unique_adapter_in_workspace(adapter_workspace, adapter_name)
        except ValueError as err:
            raise ValueError(str(err)) from err

        if adapter is None:
            logger.warning(f"Adapter not found for update: {adapter_workspace}/{adapter_name}")
            return -2

        result = await self._update_adapter_from_request(adapter, adapter_workspace, request)
        logger.info(f"Successfully updated adapter entity: {adapter_workspace}/{adapter_name}")
        return result

    async def delete_adapter_in_workspace(self, adapter_workspace: str, adapter_name: str) -> int:
        try:
            to_delete: Adapter | None = await self._load_unique_adapter_in_workspace(adapter_workspace, adapter_name)
        except ValueError as err:
            raise ValueError(str(err)) from err
        if to_delete is None or not to_delete.parent:
            return -2
        await self.entity_client.delete(
            Adapter,
            to_delete.name,
            workspace=adapter_workspace,
            parent=to_delete.parent,
        )
        logger.info(f"Successfully deleted adapter: {adapter_workspace}/{adapter_name}")
        return 0

    async def list_adapters(
        self,
        adapter_workspace: str,
        parsed_filter: ParsedFilter,
        page: int = 1,
        page_size: int = 100,
        sort: str = "created_at",
    ) -> Page[AdapterSchema]:
        """List adapter entities with structured filter (entity-store ``filter_operation``)."""
        result: ListResponse[Adapter] = await self.entity_client.list(
            Adapter,
            workspace=adapter_workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
        parent_ids = {a.parent for a in result.data if a.parent}
        models: list[Model] = await self._fetch_all_entities(
            Model, ALL_WORKSPACES, filter_str=json.dumps({"id": {"$in": list(parent_ids)}})
        )
        model_name_map = {m.id: f"{m.workspace}/{m.name}" for m in models}
        return Page(
            data=[
                _adapter_to_adapter_schema(a, model_name_map.get(a.parent, a.parent), adapter_workspace)
                for a in result.data
            ],
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(result.data),
                total_pages=result.pagination.total_pages,
                total_results=result.pagination.total_results,
            ),
            sort=sort,
            filter=parsed_filter.to_response(),
        )
