# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar, Dict, Generic, List, Optional, Protocol, Set, Type, TypeVar, get_type_hints

from nemo_platform_plugin.client.errors import (
    ConflictError,
    NotFoundError,
    UnprocessableEntityError,
    raise_for_status,
)
from nemo_platform_plugin.entities.client import AsyncEntitiesClient, EntitiesClient
from nemo_platform_plugin.entities.types import (
    DeleteResponse,
    Entity,
    EntityByNameQueryParams,
    EntityCreateInput,
    EntityDeleteQueryParams,
    EntityUpdate,
    ListEntitiesQueryParams,
)
from nemo_platform_plugin.filter_ops import FilterOperation
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter, computed_field

# Regex pattern for valid workspace names
ID_PATTERN = r"^[\w\-\+.@:]+$"
BASE_FIELDS = {
    "id",
    "name",
    "workspace",
    "created_at",
    "updated_at",
    "created_by",
    "entity_type",
    "project",
}

# Default workspace when none is specified
DEFAULT_WORKSPACE = "default"


def parse_qualified_name(name: str, default_workspace: str | None = None) -> tuple[str, str]:
    """Parse a potentially workspace-qualified name.

    If name contains a workspace qualifier (e.g., "prod/my-model"), that workspace
    is always used. The default_workspace is only used as a fallback when name
    is not qualified. Defaults to DEFAULT_WORKSPACE if neither is provided.

    Args:
        name: Entity name, optionally workspace-qualified (e.g., "my-model" or "prod/my-model")
        default_workspace: Fallback workspace if name is not qualified

    Returns:
        Tuple of (workspace, entity_name)

    Examples:
        >>> parse_qualified_name("my-model")
        ('default', 'my-model')
        >>> parse_qualified_name("my-model", default_workspace="prod")
        ('prod', 'my-model')
        >>> parse_qualified_name("prod/my-model")
        ('prod', 'my-model')
        >>> parse_qualified_name("prod/my-model", default_workspace="ignored")
        ('prod', 'my-model')  # qualified name always takes precedence
    """
    if "/" in name:
        parts = name.split("/", 1)
        return parts[0], parts[1]
    return default_workspace or DEFAULT_WORKSPACE, name


class EntityTypeDefault(str):
    """Descriptor returning snake_case class name as default for __entity_type__."""

    def __get__(self, obj: object | None, objtype: type | None = None) -> str:
        if objtype is None:
            objtype = type(obj)
        # Check class's own __dict__ for explicit override
        for cls in objtype.__mro__:
            if cls.__name__ == "EntityBase":
                break
            if "__entity_type__" in cls.__dict__ and not isinstance(cls.__dict__["__entity_type__"], EntityTypeDefault):
                return cls.__dict__["__entity_type__"]
        # Default: snake_case of class name
        return re.sub(r"(?<!^)(?=[A-Z])", "_", objtype.__name__).lower()


class EntityBase(BaseModel):
    """Base class for all entities.

    Provides common fields and behavior for entity models.
    Uses 'workspace' field name (no namespace alias) for cleaner API.

    Parent-scoped uniqueness:
    - Root entities (parent is None): unique within (workspace, entity_type, name)
    - Child entities (parent is set): unique within (workspace, entity_type, parent, name)
    """

    # Base fields are the fields that are common to all entities.
    # This does not need to be configured in the subclass.
    __base_fields__: ClassVar[Set[str]] = {
        "_id",
        "_created_at",
        "_created_by",
        "_updated_at",
        "_updated_by",
        "name",
        "workspace",
        "_parent",
        "project",
        "_db_version",
    }
    # Private attrs managed by the entity store, not to be included in stored data
    __base_private_attrs__: ClassVar[Set[str]] = {
        "_id",
        "_created_at",
        "_created_by",
        "_updated_at",
        "_updated_by",
        "_parent",
        "_db_version",
    }

    __entity_type__: ClassVar[str] = EntityTypeDefault()

    model_config = {"populate_by_name": True}

    name: str = Field(default="", description="Entity name within the workspace")
    workspace: str = Field(
        ...,
        description="Workspace identifier",
        pattern=ID_PATTERN,
    )
    project: str | None = Field(
        default=None,
        description="The name of the project associated with this entity.",
    )

    _id: str | None = PrivateAttr(default=None)
    _created_at: datetime | None = PrivateAttr(default=None)
    _created_by: str | None = PrivateAttr(default=None)
    _updated_at: datetime | None = PrivateAttr(default=None)
    _updated_by: str | None = PrivateAttr(default=None)
    _parent: str | None = PrivateAttr(default=None)
    _db_version: int = PrivateAttr(default=1)

    @computed_field
    @property
    def id(self) -> str:
        if self._id is None:
            return ""
        return self._id

    @computed_field
    @property
    def created_at(self) -> datetime | None:
        return self._created_at

    @computed_field(json_schema_extra={"nullable": True})
    @property
    def created_by(self) -> str | None:
        return self._created_by

    @computed_field
    @property
    def updated_at(self) -> datetime | None:
        return self._updated_at

    @computed_field(json_schema_extra={"nullable": True})
    @property
    def updated_by(self) -> str | None:
        return self._updated_by

    @computed_field
    @property
    def entity_id(self) -> str:
        """Alias for id for backwards compatibility."""
        return self.id

    @computed_field
    @property
    def parent(self) -> str | None:
        """Parent entity ID for nested entities."""
        return self._parent

    @computed_field
    @property
    def db_version(self) -> int:
        """Database version of the entity for optimistic locking."""
        return self._db_version

    def _get_data_fields(self) -> Dict[str, Any]:
        # Use mode="json" to ensure datetime and other types are JSON-serializable
        data = {
            k: v
            for k, v in self.model_dump(exclude=self.__base_fields__, exclude_computed_fields=True, mode="json").items()
        }
        # Include PrivateAttr fields, excluding base attrs managed by entity store
        for field_name in self.__private_attributes__:
            if field_name not in self.__base_private_attrs__:
                field_value = getattr(self, field_name)
                if isinstance(field_value, BaseModel):
                    field_value = field_value.model_dump(mode="json")
                data[field_name] = field_value
        return data


class PaginationInfo(BaseModel):
    """Pagination metadata for list responses."""

    page: int
    page_size: int
    current_page_size: int
    total_pages: int
    total_results: int


EntityT = TypeVar("EntityT", bound=EntityBase)


class ListResponse(BaseModel, Generic[EntityT]):
    """Response from list operations including pagination."""

    data: List[EntityT]
    pagination: PaginationInfo


class EntityToken(Protocol):
    """Entity identifier token used for typed unions and entity classes."""

    __entity_type__: str


EntityTypeLike = Type[EntityT] | EntityToken


class EntityGetterProtocol(Protocol[EntityT]):
    """Protocol for entity clients that can fetch entities by workspace/name.

    ``parent`` addresses a **child** entity, which is unique within
    ``(workspace, entity_type, parent, name)`` rather than by name alone. It is optional, so
    fetching a root entity is unchanged.
    """

    async def get(
        self,
        entity_type: Type[EntityT],
        *,
        name: str,
        workspace: str,
        parent: Optional[str] = None,
    ) -> EntityT: ...


class EntityDeleteClientProtocol(EntityGetterProtocol[EntityT], Protocol[EntityT]):
    """Protocol for entity clients that can list and delete entities."""

    # ``filter_str`` and ``filter_obj`` exist on the client but are deliberately absent here:
    # ``filter_operation`` is the sanctioned structured form, and the other two are a JSON-string
    # variant and an exact-match shorthand kept for older callers.
    async def list(
        self,
        entity_type: Type[EntityT],
        *,
        workspace: str,
        filter_operation: Optional[FilterOperation] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[EntityT]: ...

    async def delete(
        self,
        entity_type: Type[EntityT],
        name: str,
        *,
        workspace: str,
        parent: Optional[str] = None,
        expected_db_version: Optional[int] = None,
    ) -> object: ...


class EntityClientProtocol(EntityDeleteClientProtocol[EntityT], Protocol[EntityT]):
    """Protocol for the common entity CRUD operations used by plugins."""

    async def create(self, entity: EntityT) -> EntityT: ...


class EntityUpdateClientProtocol(Protocol[EntityT]):
    """Protocol for entity clients that can update an existing entity.

    Separate from :class:`EntityClientProtocol` rather than folded into it: ``update`` is a
    read-modify-write against the ``db_version`` optimistic lock, and most services never need it.
    Compose it with the CRUD protocol where a service does::

        class Store(EntityClientProtocol[MyEntity], EntityUpdateClientProtocol[MyEntity], Protocol):
            ...
    """

    async def update(self, entity: EntityT, *, original_name: Optional[str] = None) -> EntityT: ...


class AnyEntityGetterProtocol(Protocol):
    """Protocol for clients that can fetch any entity model type."""

    async def get(self, entity_type: Type[EntityT], *, name: str, workspace: str) -> EntityT: ...


class AnyEntityDeleteClientProtocol(AnyEntityGetterProtocol, Protocol):
    """Protocol for clients that can list and delete any entity model type."""

    async def list(
        self,
        entity_type: Type[EntityT],
        *,
        workspace: str,
        filter_operation: Optional[FilterOperation] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[EntityT]: ...

    async def delete(
        self,
        entity_type: Type[EntityT],
        name: str,
        *,
        workspace: str,
        expected_db_version: Optional[int] = None,
    ) -> object: ...


# Error types
class EntityStoreError(Exception):
    """Base exception for EntityClient errors."""

    pass


class EntityNotFoundError(EntityStoreError):
    """Entity not found."""

    pass


class EntityConflictError(EntityStoreError):
    """Entity conflict error.

    Raised when an entity operation conflicts with the current store state:
    - Entity already exists (conflict on create)
    - Entity version mismatch (optimistic locking conflict on update)
    - Query expected one result but matched multiple entities
    """

    pass


class EntityValidationError(EntityStoreError):
    """Entity validation failed (e.g., referenced project does not exist)."""

    pass


def _get_entity_type(entity_class: EntityTypeLike) -> str:
    """Get the __entity_type__ from an entity class.

    EntityBase subclasses provide this through EntityTypeDefault. Annotated
    discriminated unions can participate by satisfying EntityToken.
    """
    return str(entity_class.__entity_type__)


def _convert_filter_obj_to_filter_str(filter_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a filter dict to API filter format.

    For EntityBase entities, fields are stored in the data JSON column,
    so we prefix them with 'data.' unless they're base fields.
    """
    filter_dict: Dict[str, Any] = {}

    for field, value in filter_obj.items():
        # Base fields don't need the data. prefix
        if field in BASE_FIELDS:
            api_field = field
        # Already has data. prefix - don't double-prefix
        elif field.startswith("data."):
            api_field = field
        # All other fields are stored in the data JSON column
        else:
            api_field = f"data.{field}"
        filter_dict[api_field] = value
    return filter_dict


def _convert_sort_to_api_sort(sort: str) -> str:
    """Convert a sort string to an API sort string.

    For EntityBase entities, fields are stored in the data JSON column,
    so we prefix them with 'data.' unless they're base fields.
    """
    field = sort.lstrip("-")
    if field not in BASE_FIELDS:
        return f"{'-' if sort.startswith('-') else ''}data.{field}"

    return sort


def _merge_header_items(target: dict[str, str], headers: object) -> None:
    if not isinstance(headers, Mapping):
        return
    for key, value in headers.items():
        target[str(key).lower()] = str(value)


def _client_headers(client: AsyncEntitiesClient | EntitiesClient) -> dict[str, str]:
    headers: dict[str, str] = {}
    _merge_header_items(headers, client._client.headers)
    _merge_header_items(headers, client._default_headers)
    return headers


def _convert_api_entity_to_model(
    entity: Entity,
    entity_type: EntityTypeLike,
    sdk_headers: dict[str, str],
) -> EntityT:
    """Convert an API entity to an EntityBase model."""
    entity_dict = entity.model_dump()
    entity_dict.update(entity.data)
    type_adapter = TypeAdapter(entity_type)
    result = type_adapter.validate_python(entity_dict)
    result._id = entity.id
    if entity.parent:
        result._parent = entity.parent
    result._created_at = entity.created_at
    result._created_by = entity.created_by
    result._updated_at = entity.updated_at
    result._updated_by = entity.updated_by
    result._db_version = entity.db_version
    # Set PrivateAttr fields from stored data, excluding base attrs (already set above)
    # Use TypeAdapter to properly deserialize values (handles BaseModel, Optional, etc.)
    # Note: use type(result) not entity_type, since entity_type may be an Annotated union
    type_hints = get_type_hints(type(result))
    for field_name in type(result).__private_attributes__:
        if field_name not in type(result).__base_private_attrs__ and field_name in entity.data:
            raw_value = entity.data[field_name]
            attr_type = type_hints.get(field_name)
            if attr_type is not None:
                validated = TypeAdapter(attr_type).validate_python(raw_value)
                setattr(result, field_name, validated)
            else:
                setattr(result, field_name, raw_value)

    # Strip _auth_context unless the effective caller is a service principal.
    # With on-behalf-of delegation the SDK authenticates as service:platform but
    # the real caller is in X-NMP-Principal-On-Behalf-Of.
    if hasattr(result, "_auth_context"):
        effective = sdk_headers.get("x-nmp-principal-on-behalf-of") or sdk_headers.get("x-nmp-principal-id", "")
        if not effective.startswith("service:"):
            setattr(result, "_auth_context", None)

    return result


def _entity_create_fields(entity: EntityBase) -> dict[str, Any]:
    create_fields: dict[str, Any] = {"data": entity._get_data_fields()}
    if entity.name:
        create_fields["name"] = entity.name
    if entity._parent:
        create_fields["parent"] = entity._parent
    if entity.project:
        create_fields["project"] = entity.project
    return create_fields


def _entity_update_fields(entity: EntityBase, *, original_name: str | None = None) -> dict[str, Any]:
    update_fields: dict[str, Any] = {
        "data": entity._get_data_fields(),
        "expected_db_version": entity.db_version,
    }
    if original_name:
        update_fields["new_name"] = entity.name
    if entity.project:
        update_fields["project"] = entity.project
    return update_fields


class EntityClient:
    """
    Unified async client for Entity operations.

    A single client handles all entity types - pass the type to each method.
    Primary lookup is by name (Kubernetes-style), with ID lookup available for debugging.

    Example:
        client = EntityClient(client_from_platform(sdk, AsyncEntitiesClient))

        # Create
        msg = HelloWorldMessage(name="my-message", workspace="default", message="Hello")
        saved = await client.create(msg)

        # Get by name (primary lookup)
        msg = await client.get(HelloWorldMessage, "my-message")
        msg = await client.get(HelloWorldMessage, "my-message", workspace="prod")
        msg = await client.get(HelloWorldMessage, "prod/my-message")  # workspace-qualified

        # Get by ID (debug/internal)
        msg = await client.get_by_id(HelloWorldMessage, "hello-world-message-5Q2LoF8z...")

        # List
        result = await client.list(HelloWorldMessage, workspace="default")
        for msg in result.data:
            print(msg.name)

        # Delete
        await client.delete(HelloWorldMessage, "my-message")
    """

    def __init__(self, client: AsyncEntitiesClient):
        """
        Initialize the EntityClient.

        Args:
            client: The NemoClient-based async entities client.
        """
        self._client = client

    async def close(self) -> None:
        """Close the underlying HTTP client.

        This should be called during shutdown to properly close HTTP connections.
        """
        await self._client.aclose()

    def _convert_api_entity_to_model(self, entity: Entity, entity_type: EntityTypeLike) -> EntityT:
        """Convert an API entity to an EntityBase model."""
        return _convert_api_entity_to_model(entity, entity_type, _client_headers(self._client))

    async def list(
        self,
        entity_type: EntityTypeLike,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_operation: FilterOperation | None = None,
        filter_str: Optional[str] = None,
        sort: Optional[str] = None,
        filter_obj: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[EntityT]:
        """List entities with filtering and pagination.

        ``filter_operation`` and ``filter_str`` are mutually exclusive:
        supplying both raises ``ValueError`` because there is no well-defined
        merge at this layer (service code should compose conditions into a
        single ``FilterOperation`` via ``ParsedFilter.and_with`` before calling).

        ``filter_obj`` is a fallback shorthand for exact-match filters; it is
        consulted only when neither ``filter_operation`` nor ``filter_str`` is
        supplied, and is silently ignored otherwise.

        Args:
            entity_type: The entity class to query
            workspace: Filter by workspace (defaults to DEFAULT_WORKSPACE)
            filter_operation: Structured filter operation tree
            filter_str: Filter as a JSON string (mutually exclusive with filter_operation)
            sort: Sort field (e.g., "name", "-created_at" for descending)
            filter_obj: Exact-match shorthand; only used when no other filter is supplied
            page: Page number
            page_size: Items per page

        Returns:
            ListResponse containing data and pagination info
        """

        if filter_operation is not None and filter_str is not None:
            raise ValueError(
                "EntityClient.list: pass either filter_operation or filter_str, not both. "
                "Combining them previously silently dropped one — merge into a single filter_operation "
                "via ParsedFilter.and_with."
            )

        if filter_operation is not None:
            effective_filter_str = json.dumps(filter_operation.to_dict())
        else:
            effective_filter_str = filter_str

        # Build filter string from filter_obj if provided
        if filter_obj and not effective_filter_str:
            # Convert filter_obj to filter JSON format
            filter_dict = _convert_filter_obj_to_filter_str(filter_obj)
            if filter_dict:
                effective_filter_str = json.dumps(filter_dict)

        query_params: ListEntitiesQueryParams = {"page": page, "page_size": page_size}
        if effective_filter_str:
            query_params["filter"] = effective_filter_str
        if sort:
            query_params["sort"] = _convert_sort_to_api_sort(sort)

        response = await self._client.list_entities(
            entity_type=_get_entity_type(entity_type),
            workspace=workspace,
            query_params=query_params,
        )
        result_page = response.page()

        entities = [self._convert_api_entity_to_model(entity, entity_type) for entity in result_page.items]
        metadata = result_page.metadata
        pagination = PaginationInfo(
            page=metadata["page"],
            page_size=metadata["page_size"],
            current_page_size=metadata["current_page_size"],
            total_pages=metadata["total_pages"],
            total_results=metadata["total_results"],
        )

        return ListResponse(data=entities, pagination=pagination)

    async def count_by(
        self,
        entity_type: EntityTypeLike,
        field: str,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_obj: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Return the number of matching entities grouped by ``field``."""
        if not field.isidentifier():
            raise ValueError(f"Field '{field}' is not a direct entity data field")

        filter_dict = _convert_filter_obj_to_filter_str(filter_obj) if filter_obj else {}

        # Only the grouped tallies are wanted, so ask for the smallest possible page.
        query_params: ListEntitiesQueryParams = {
            "page": 1,
            "page_size": 1,
            "count_by": f"data.{field}",
        }
        if filter_dict:
            query_params["filter"] = json.dumps(filter_dict)

        response = await self._client.list_entities(
            entity_type=_get_entity_type(entity_type),
            workspace=workspace,
            query_params=query_params,
        )
        # ``group_counts`` rides on the response envelope rather than the item
        # page, so read it off the raw body instead of ``response.page()``.
        # Paginated responses defer their status check to ``page()``/``items()``,
        # neither of which runs here, so a non-2xx would otherwise surface as a
        # bogus "grouped counts not found" instead of the real HTTP error.
        raise_for_status(response.http_response)
        group_counts = response.http_response.json().get("group_counts")
        if group_counts is None:
            raise EntityStoreError("Grouped counts not found in response")
        return TypeAdapter(dict[str, int]).validate_python(group_counts)

    async def find_one(
        self,
        entity_type: EntityTypeLike,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_operation: FilterOperation | None = None,
        filter_str: Optional[str] = None,
        filter_obj: Optional[Dict[str, Any]] = None,
        sort: Optional[str] = None,
    ) -> EntityT:
        """Find exactly one entity matching a query.

        This is intentionally query-based, not identity-based. It is useful for
        cross-workspace lookups where the caller expects a field such as ``name``
        to be globally unique by convention even though the entity store only
        enforces workspace-scoped uniqueness.

        Prefer this over ``get_by_field`` when duplicate matches are an error.

        Raises:
            EntityNotFoundError: No entity matches the query
            EntityConflictError: More than one entity matches the query
        """
        result = await self.list(
            entity_type,
            workspace=workspace,
            filter_operation=filter_operation,
            filter_str=filter_str,
            filter_obj=filter_obj,
            sort=sort,
            page=1,
            page_size=2,
        )

        entity_type_name = _get_entity_type(entity_type)
        match_count = result.pagination.total_results
        if not result.data:
            raise EntityNotFoundError(f"No {entity_type_name} entity found matching query in workspace '{workspace}'")
        if len(result.data) > 1 or match_count > 1:
            raise EntityConflictError(
                f"Multiple {entity_type_name} entities found matching query in workspace '{workspace}'"
            )

        return result.data[0]

    async def create(self, entity: EntityT) -> EntityT:
        """Create a new entity.

        Args:
            entity: Entity input data

        Returns:
            Created entity with ID and timestamps

        Raises:
            EntityConflictError: Entity already exists
        """
        entity_type = type(entity)
        try:
            response = await self._client.create_entity(
                workspace=entity.workspace,
                entity_type=_get_entity_type(entity_type),
                body=EntityCreateInput(**_entity_create_fields(entity)),
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except ConflictError as e:
            raise EntityConflictError(
                f"Entity with name '{entity.name}' already exists in workspace '{entity.workspace}'"
            ) from e
        except UnprocessableEntityError as e:
            detail = e.body.get("detail", str(e)) if isinstance(e.body, dict) else str(e)
            raise EntityValidationError(detail) from e

    async def get(
        self,
        entity_type: EntityTypeLike,
        name: str,
        *,
        workspace: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> EntityT:
        """Get entity by name (primary lookup method).

        Supports workspace-qualified names like "prod/my-model".

        Args:
            entity_type: The entity class to return
            name: Entity name (can be workspace-qualified like "prod/my-model")
            parent: Optional parent entity ID for nested entities
            workspace: Optional workspace override (ignored if name is qualified)

        Returns:
            Entity matching the name

        Raises:
            EntityNotFoundError: Entity not found
        """
        ws, entity_name = parse_qualified_name(name, default_workspace=workspace)
        query_params: EntityByNameQueryParams | None = {"parent": parent} if parent else None
        try:
            response = await self._client.get_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=entity_name,
                workspace=ws,
                query_params=query_params,
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{entity_name}' not found in workspace '{ws}'") from e

    async def get_by_id(
        self,
        entity_type: EntityTypeLike,
        entity_id: str,
    ) -> EntityT:
        """Get entity by ID (for debugging/internal use).

        Args:
            entity_type: The entity class to return
            entity_id: Entity UUID

        Returns:
            Entity with the given ID

        Raises:
            EntityNotFoundError: Entity not found
        """
        try:
            response = await self._client.get_entity_by_id(entity_id=entity_id)
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity with id '{entity_id}' not found") from e

    async def update(self, entity: EntityT, *, original_name: str | None = None) -> EntityT:
        """Update an entity by name.

        Automatically includes db_version for optimistic locking if the entity was
        fetched via get(). This ensures updates only succeed if the entity hasn't
        been modified by another request.

        Args:
            entity: Entity with updated data (must have valid workspace and name)
            original_name: Original name if renaming entity (use entity.name for current name)

        Returns:
            Updated entity

        Raises:
            EntityNotFoundError: Entity not found
            EntityConflictError: Version mismatch (entity was modified by another request)
        """
        entity_type = type(entity)
        path_name = original_name or entity.name

        query_params: EntityByNameQueryParams | None = {"parent": entity._parent} if entity._parent else None

        try:
            response = await self._client.update_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=path_name,
                workspace=entity.workspace,
                body=EntityUpdate(**_entity_update_fields(entity, original_name=original_name)),
                query_params=query_params,
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{path_name}' not found in workspace '{entity.workspace}'") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e
        except UnprocessableEntityError as e:
            detail = e.body.get("detail", str(e)) if isinstance(e.body, dict) else str(e)
            raise EntityValidationError(detail) from e

    async def delete(
        self,
        entity_type: EntityTypeLike,
        name: str,
        *,
        workspace: Optional[str] = None,
        parent: Optional[str] = None,
        expected_db_version: Optional[int] = None,
    ) -> DeleteResponse:
        """Delete an entity by name.

        Supports workspace-qualified names like "prod/my-model".

        Args:
            entity_type: The entity class (for type safety)
            name: Entity name (can be workspace-qualified)
            workspace: Optional workspace override
            parent: Optional parent entity ID for nested entities
            expected_db_version: Optional expected database version for optimistic locking

        Returns:
            Deleted entity response

        Raises:
            EntityNotFoundError: Entity not found
            EntityConflictError: Version mismatch (entity was modified by another request)
        """
        ws, entity_name = parse_qualified_name(name, default_workspace=workspace)
        delete_params: EntityDeleteQueryParams = {}
        if parent:
            delete_params["parent"] = parent
        if expected_db_version is not None:
            delete_params["expected_db_version"] = expected_db_version
        try:
            response = await self._client.delete_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=entity_name,
                workspace=ws,
                query_params=delete_params or None,
            )
            return response.data()
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{entity_name}' not found in workspace '{ws}'") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e

    async def delete_by_id(
        self,
        entity_type: EntityTypeLike,
        entity_id: str,
    ) -> DeleteResponse:
        """Delete an entity by ID.

        First retrieves the entity to get its workspace and name, then deletes by name.

        Args:
            entity_type: The entity class (for type safety)
            entity_id: Entity UUID to delete

        Returns:
            Deleted entity response

        Raises:
            EntityNotFoundError: Entity not found
            EntityConflictError: Version mismatch (entity was modified by another request)
        """
        try:
            entity = (await self._client.get_entity_by_id(entity_id=entity_id)).data()
            delete_params: EntityDeleteQueryParams = {"expected_db_version": entity.db_version}
            if entity.parent:
                delete_params["parent"] = entity.parent
            response = await self._client.delete_entity_by_name(
                entity_type=entity.entity_type,
                name=entity.name,
                workspace=entity.workspace,
                query_params=delete_params,
            )
            return response.data()
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity with id '{entity_id}' not found") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e

    async def save(self, entity: EntityT) -> EntityT:
        """
        Create or update an entity.

        - If entity.id is empty: creates new entity
        - If entity already exists: updates existing entity

        The entity type is inferred from the entity's class.

        Args:
            entity: Entity to save

        Returns:
            Saved entity with id and timestamps populated

        Raises:
            EntityConflictError: Entity with same name already exists (on create)
            EntityNotFoundError: Entity not found (on update)
        """
        # If entity has an ID, try to update
        if entity.id:
            try:
                return await self.update(entity)
            except EntityNotFoundError:
                # ID doesn't exist, fall through to create
                pass

        # Try to create
        try:
            return await self.create(entity)
        except EntityConflictError:
            # Entity exists - try to update if we have an ID
            if entity.id:
                try:
                    return await self.update(entity)
                except EntityNotFoundError as e:
                    raise EntityNotFoundError(f"Entity with id '{entity.id}' not found") from e
            raise

    async def add(self, entity: EntityT) -> EntityT:
        """
        Create a new entity (always creates, never updates).

        Args:
            entity: Entity to create

        Returns:
            Created entity with id and timestamps populated

        Raises:
            EntityConflictError: Entity already exists
        """
        return await self.create(entity)

    async def get_by_field(
        self,
        entity_type: EntityTypeLike,
        workspace: str,
        **field_filters: Any,
    ) -> EntityT:
        """
        Get a single entity by field value(s).

        This is a convenience method for looking up entities by arbitrary fields.
        It uses list() under the hood and returns the first match. Use
        ``find_one`` instead when the caller requires exactly one match and
        duplicates should raise ``EntityConflictError``.

        Args:
            entity_type: The entity class to return
            workspace: Optional workspace to scope the search
            **field_filters: Field=value pairs to filter by

        Returns:
            First entity matching the filters

        Raises:
            EntityNotFoundError: No entity matches the filters
            ValueError: No field filters provided

        Example:
            # Get message by external_id
            msg = await client.get_by_field(HelloWorldMessage, external_id="ext-123")

            # Get message by external_id within a workspace
            msg = await client.get_by_field(HelloWorldMessage, workspace="prod", external_id="ext-123")
        """
        if not field_filters:
            raise ValueError("At least one field filter is required")

        result = await self.list(
            entity_type,
            workspace=workspace,
            filter_obj=field_filters,
            page_size=1,
        )

        if not result.data:
            filter_desc = ", ".join(f"{k}={v!r}" for k, v in field_filters.items())
            raise EntityNotFoundError(f"Entity not found matching: {filter_desc}")

        return result.data[0]


class SyncEntityClient:
    """
    Unified sync client for Entity operations.

    A single client handles all entity types - pass the type to each method.
    Primary lookup is by name (Kubernetes-style), with ID lookup available for debugging.

    Example:
        client = SyncEntityClient(client_from_platform(sdk, EntitiesClient))

        # Create
        msg = HelloWorldMessage(name="my-message", workspace="default", message="Hello")
        saved = client.create(msg)

        # Get by name (primary lookup)
        msg = client.get(HelloWorldMessage, "my-message")
        msg = client.get(HelloWorldMessage, "my-message", workspace="prod")
        msg = client.get(HelloWorldMessage, "prod/my-message")  # workspace-qualified

        # Get by ID (debug/internal)
        msg = client.get_by_id(HelloWorldMessage, "hello-world-message-5Q2LoF8z...")

        # List
        result = client.list(HelloWorldMessage, workspace="default")
        for msg in result.data:
            print(msg.name)

        # Delete
        client.delete(HelloWorldMessage, "my-message")
    """

    def __init__(self, client: EntitiesClient):
        """
        Initialize the SyncEntityClient.

        Args:
            client: The NemoClient-based sync entities client.
        """
        self._client = client

    def close(self) -> None:
        """Close the underlying HTTP client.

        This should be called during shutdown to properly close HTTP connections.
        """
        self._client.close()

    def _convert_api_entity_to_model(self, entity: Entity, entity_type: EntityTypeLike) -> EntityT:
        """Convert an API entity to an EntityBase model."""
        return _convert_api_entity_to_model(entity, entity_type, _client_headers(self._client))

    def list(
        self,
        entity_type: EntityTypeLike,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_operation: FilterOperation | None = None,
        filter_str: Optional[str] = None,
        sort: Optional[str] = None,
        filter_obj: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[EntityT]:
        """List entities with filtering and pagination."""
        if filter_operation is not None and filter_str is not None:
            raise ValueError(
                "SyncEntityClient.list: pass either filter_operation or filter_str, not both. "
                "Combining them previously silently dropped one - merge into a single filter_operation "
                "via ParsedFilter.and_with."
            )

        if filter_operation is not None:
            effective_filter_str = json.dumps(filter_operation.to_dict())
        else:
            effective_filter_str = filter_str

        if filter_obj and not effective_filter_str:
            filter_dict = _convert_filter_obj_to_filter_str(filter_obj)
            if filter_dict:
                effective_filter_str = json.dumps(filter_dict)

        query_params: ListEntitiesQueryParams = {"page": page, "page_size": page_size}
        if effective_filter_str:
            query_params["filter"] = effective_filter_str
        if sort:
            query_params["sort"] = _convert_sort_to_api_sort(sort)

        response = self._client.list_entities(
            entity_type=_get_entity_type(entity_type),
            workspace=workspace,
            query_params=query_params,
        )
        result_page = response.page()

        entities = [self._convert_api_entity_to_model(entity, entity_type) for entity in result_page.items]
        metadata = result_page.metadata
        pagination = PaginationInfo(
            page=metadata["page"],
            page_size=metadata["page_size"],
            current_page_size=metadata["current_page_size"],
            total_pages=metadata["total_pages"],
            total_results=metadata["total_results"],
        )

        return ListResponse(data=entities, pagination=pagination)

    def count_by(
        self,
        entity_type: EntityTypeLike,
        field: str,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_obj: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Return the number of matching entities grouped by ``field``."""
        if not field.isidentifier():
            raise ValueError(f"Field '{field}' is not a direct entity data field")

        filter_dict = _convert_filter_obj_to_filter_str(filter_obj) if filter_obj else {}

        query_params: ListEntitiesQueryParams = {
            "page": 1,
            "page_size": 1,
            "count_by": f"data.{field}",
        }
        if filter_dict:
            query_params["filter"] = json.dumps(filter_dict)

        response = self._client.list_entities(
            entity_type=_get_entity_type(entity_type),
            workspace=workspace,
            query_params=query_params,
        )
        raise_for_status(response.http_response)
        group_counts = response.http_response.json().get("group_counts")
        if group_counts is None:
            raise EntityStoreError("Grouped counts not found in response")
        return TypeAdapter(dict[str, int]).validate_python(group_counts)

    def find_one(
        self,
        entity_type: EntityTypeLike,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_operation: FilterOperation | None = None,
        filter_str: Optional[str] = None,
        filter_obj: Optional[Dict[str, Any]] = None,
        sort: Optional[str] = None,
    ) -> EntityT:
        """Find exactly one entity matching a query."""
        result = self.list(
            entity_type,
            workspace=workspace,
            filter_operation=filter_operation,
            filter_str=filter_str,
            filter_obj=filter_obj,
            sort=sort,
            page=1,
            page_size=2,
        )

        entity_type_name = _get_entity_type(entity_type)
        match_count = result.pagination.total_results
        if not result.data:
            raise EntityNotFoundError(f"No {entity_type_name} entity found matching query in workspace '{workspace}'")
        if len(result.data) > 1 or match_count > 1:
            raise EntityConflictError(
                f"Multiple {entity_type_name} entities found matching query in workspace '{workspace}'"
            )

        return result.data[0]

    def create(self, entity: EntityT) -> EntityT:
        """Create a new entity."""
        entity_type = type(entity)
        try:
            response = self._client.create_entity(
                workspace=entity.workspace,
                entity_type=_get_entity_type(entity_type),
                body=EntityCreateInput(**_entity_create_fields(entity)),
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except ConflictError as e:
            raise EntityConflictError(
                f"Entity with name '{entity.name}' already exists in workspace '{entity.workspace}'"
            ) from e
        except UnprocessableEntityError as e:
            detail = e.body.get("detail", str(e)) if isinstance(e.body, dict) else str(e)
            raise EntityValidationError(detail) from e

    def get(
        self,
        entity_type: EntityTypeLike,
        name: str,
        *,
        workspace: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> EntityT:
        """Get entity by name (primary lookup method)."""
        ws, entity_name = parse_qualified_name(name, default_workspace=workspace)
        query_params: EntityByNameQueryParams | None = {"parent": parent} if parent else None
        try:
            response = self._client.get_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=entity_name,
                workspace=ws,
                query_params=query_params,
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{entity_name}' not found in workspace '{ws}'") from e

    def get_by_id(
        self,
        entity_type: EntityTypeLike,
        entity_id: str,
    ) -> EntityT:
        """Get entity by ID (for debugging/internal use)."""
        try:
            response = self._client.get_entity_by_id(entity_id=entity_id)
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity with id '{entity_id}' not found") from e

    def update(self, entity: EntityT, *, original_name: str | None = None) -> EntityT:
        """Update an entity by name."""
        entity_type = type(entity)
        path_name = original_name or entity.name
        query_params: EntityByNameQueryParams | None = {"parent": entity._parent} if entity._parent else None

        try:
            response = self._client.update_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=path_name,
                workspace=entity.workspace,
                body=EntityUpdate(**_entity_update_fields(entity, original_name=original_name)),
                query_params=query_params,
            )
            return self._convert_api_entity_to_model(response.data(), entity_type)
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{path_name}' not found in workspace '{entity.workspace}'") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e
        except UnprocessableEntityError as e:
            detail = e.body.get("detail", str(e)) if isinstance(e.body, dict) else str(e)
            raise EntityValidationError(detail) from e

    def delete(
        self,
        entity_type: EntityTypeLike,
        name: str,
        *,
        workspace: Optional[str] = None,
        parent: Optional[str] = None,
        expected_db_version: Optional[int] = None,
    ) -> DeleteResponse:
        """Delete an entity by name."""
        ws, entity_name = parse_qualified_name(name, default_workspace=workspace)
        delete_params: EntityDeleteQueryParams = {}
        if parent:
            delete_params["parent"] = parent
        if expected_db_version is not None:
            delete_params["expected_db_version"] = expected_db_version
        try:
            response = self._client.delete_entity_by_name(
                entity_type=_get_entity_type(entity_type),
                name=entity_name,
                workspace=ws,
                query_params=delete_params or None,
            )
            return response.data()
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity '{entity_name}' not found in workspace '{ws}'") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e

    def delete_by_id(
        self,
        entity_type: EntityTypeLike,
        entity_id: str,
    ) -> DeleteResponse:
        """Delete an entity by ID."""
        try:
            entity = self._client.get_entity_by_id(entity_id=entity_id).data()
            delete_params: EntityDeleteQueryParams = {"expected_db_version": entity.db_version}
            if entity.parent:
                delete_params["parent"] = entity.parent
            response = self._client.delete_entity_by_name(
                entity_type=entity.entity_type,
                name=entity.name,
                workspace=entity.workspace,
                query_params=delete_params,
            )
            return response.data()
        except NotFoundError as e:
            raise EntityNotFoundError(f"Entity with id '{entity_id}' not found") from e
        except ConflictError as e:
            raise EntityConflictError(str(e)) from e

    def save(self, entity: EntityT) -> EntityT:
        """
        Create or update an entity.

        - If entity.id is empty: creates new entity
        - If entity already exists: updates existing entity
        """
        if entity.id:
            try:
                return self.update(entity)
            except EntityNotFoundError:
                pass

        try:
            return self.create(entity)
        except EntityConflictError:
            if entity.id:
                try:
                    return self.update(entity)
                except EntityNotFoundError as e:
                    raise EntityNotFoundError(f"Entity with id '{entity.id}' not found") from e
            raise

    def add(self, entity: EntityT) -> EntityT:
        """Create a new entity (always creates, never updates)."""
        return self.create(entity)

    def get_by_field(
        self,
        entity_type: EntityTypeLike,
        workspace: str,
        **field_filters: Any,
    ) -> EntityT:
        """Get a single entity by field value(s)."""
        if not field_filters:
            raise ValueError("At least one field filter is required")

        result = self.list(
            entity_type,
            workspace=workspace,
            filter_obj=field_filters,
            page_size=1,
        )

        if not result.data:
            filter_desc = ", ".join(f"{k}={v!r}" for k, v in field_filters.items())
            raise EntityNotFoundError(f"Entity not found matching: {filter_desc}")

        return result.data[0]
