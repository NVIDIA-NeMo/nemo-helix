# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File management endpoints for the Files Service."""

import logging
from dataclasses import dataclass
from typing import ClassVar

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.responses import FileResponse
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.errors import NemoClientError
from nmp.common.api.common import GenericSortField, PaginationData
from nmp.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nmp.common.api.utils import generate_openapi_extra_params
from nmp.common.auth import AuthClient, get_auth_client
from nmp.common.entities.client import (
    EntityBase,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
    EntityStoreError,
)
from nmp.common.files.storage_config import LocalStorageConfig, S3StorageConfig
from nmp.common.observability import BaseContext, scoped_app_ctx
from nmp.common.secrets.exceptions import SecretAccessDeniedError, SecretNotFoundError
from nmp.common.service.dependencies import (
    get_entity_client,
    get_sdk_client,
    get_service_config_factory,
)
from nmp.core.files.api.endpoint_helpers import (
    CacheContext,
    get_cache_status_for_files,
    get_download_file_info,
    get_fileset,
    list_storage_files,
    resolve_storage_secrets_for_user,
    stream_file_download,
)
from nmp.core.files.api.v2.filesets.schemas import (
    CreateFilesetRequest,
    FilesetFileOutput,
    FilesetFilter,
    FilesetOutput,
    FilesetPage,
    ListFilesetFilesResponse,
    UpdateFilesetRequest,
    fileset_file_output_from_info,
    fileset_output_from_entity,
    list_fileset_files_from_infos,
)
from nmp.core.files.app.backends import storage_impl_factory
from nmp.core.files.app.backends.factory import StorageConfig
from nmp.core.files.app.cache import CacheStatus, warm_fileset_cache
from nmp.core.files.app.external_hosts import (
    ExternalHostInvalidError,
    ExternalHostNotAllowedError,
)
from nmp.core.files.app.file_lock import FileLockManager
from nmp.core.files.app.streaming import (
    MultipartChunkProcessor,
    OctetStreamChunkProcessor,
    streaming_file_upload,
)
from nmp.core.files.config import FilesConfig
from nmp.core.files.entities import Fileset
from nmp.core.files.exceptions import (
    InactivityTimeoutError,
    InvalidPathError,
    NotFoundError,
    StorageAccessError,
    StorageBackendError,
    StorageConfigError,
    StorageUnavailableError,
)
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_408_REQUEST_TIMEOUT,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_507_INSUFFICIENT_STORAGE,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_REFERENCE_COUNT_PAGE_SIZE = 1


def _sanitize_for_log(value: object) -> str:
    """Strip line breaks from a value before logging."""
    return str(value).replace("\r", "").replace("\n", "")


_HTTP_EXCEPTION_DETAIL = {
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {"detail": {"type": "string"}},
            }
        }
    }
}


class _ModelFilesetReference(EntityBase):
    """Minimal model projection used for fileset reference checks."""

    __entity_type__: ClassVar[str] = "model"
    fileset: str | None = None


class _AdapterFilesetReference(EntityBase):
    """Minimal adapter projection used for fileset reference checks."""

    __entity_type__: ClassVar[str] = "adapter"
    fileset: str


async def _count_fileset_references(
    entity_store: EntityClient,
    entity_type: type[_ModelFilesetReference] | type[_AdapterFilesetReference],
    fileset_ref: str,
    *,
    workspace: str = "-",
) -> int:
    """Count references using pagination metadata without fetching every entity."""
    result = await entity_store.list(
        entity_type,
        workspace=workspace,
        filter_obj={"fileset": fileset_ref},
        page_size=_REFERENCE_COUNT_PAGE_SIZE,
    )
    return result.pagination.total_results


async def _count_entity_fileset_references(
    entity_store: EntityClient,
    entity_type: type[_ModelFilesetReference] | type[_AdapterFilesetReference],
    workspace: str,
    name: str,
) -> int:
    """Count qualified, legacy URI, and same-workspace bare references."""
    qualified_ref = f"{workspace}/{name}"
    qualified_count = await _count_fileset_references(entity_store, entity_type, qualified_ref)
    legacy_uri_count = await _count_fileset_references(entity_store, entity_type, f"fileset://{qualified_ref}")
    bare_count = await _count_fileset_references(
        entity_store,
        entity_type,
        name,
        workspace=workspace,
    )
    legacy_bare_count = await _count_fileset_references(
        entity_store,
        entity_type,
        f"fileset://{name}",
        workspace=workspace,
    )
    return qualified_count + legacy_uri_count + bare_count + legacy_bare_count


def _format_fileset_dependents(workspace: str, name: str, model_count: int, adapter_count: int) -> str:
    """Format a referential-integrity error without leaking entity names."""
    return (
        f"Cannot delete fileset '{workspace}/{name}' because {model_count} model entity reference(s) "
        f"and {adapter_count} adapter entity reference(s) still use it. "
        "Relink or delete the dependent entities first."
    )


def _get_cache_lock_manager(
    entity_client: EntityClient,
    config: FilesConfig,
) -> FileLockManager:
    """
    Create a FileLockManager for cache coordination.

    Uses the "system" workspace because we want caches to be shared across workspaces.
    For example, if workspace1/llama3 and workspace2/llama3 both point to the same
    model HF model, they should share a cache. If we used a workspace-specific lock instead,
    we could unintentionally allow two writers to the same cache path.
    """
    return FileLockManager(
        entity_client=entity_client,
        workspace="system",
        lock_ttl_seconds=config.file_lock_ttl_seconds,
    )


def _validate_user_storage_config(storage: StorageConfig, config: FilesConfig) -> None:
    """Validate user-provided storage configuration.

    Raises HTTPException if the storage config is not allowed for user-provided filesets.
    """
    if isinstance(storage, LocalStorageConfig) and not config.allow_user_local_storage:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=(
                "Creating filesets with local storage is not allowed. "
                "Use NGC, HuggingFace, or omit storage to use the default backend."
            ),
        )
    if isinstance(storage, S3StorageConfig) and storage.use_sdk_auth:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=(
                "use_sdk_auth=True is not allowed for user-provided S3 storage. "
                "Provide explicit credentials via access_key_id_secret and secret_access_key_secret."
            ),
        )


@dataclass
class FilesContext(BaseContext):
    """
    Files context that will be enriched into logs and traces
    """

    otel_prefix = "files"

    fileset_name: str | None = None
    path: str | None = None


@router.post(
    "/v2/workspaces/{workspace}/filesets",
    summary="Create Fileset",
    status_code=HTTP_200_OK,
)
async def create_fileset(
    workspace: str,
    create_request: CreateFilesetRequest,
    background_tasks: BackgroundTasks,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetOutput:
    """
    Create a new fileset.

    If no storage configuration is provided, the default storage backend will be used.
    """
    logger.info(f"POST /filesets - workspace={workspace}, name={create_request.name}")

    # Check if the fileset already exists
    try:
        await entity_store.get(Fileset, create_request.name, workspace=workspace)
        logger.warning(f"Fileset already exists: {workspace}/{create_request.name}")
        raise HTTPException(
            status_code=HTTP_409_CONFLICT,
            detail=f"Fileset '{workspace}/{create_request.name}' already exists",
        )
    except EntityNotFoundError:
        pass

    try:
        if create_request.storage is None:
            default_config = config.default_storage_config
            storage = default_config.copy_config(f"filesets/{workspace}/{create_request.name}")
            secrets = {}  # Local storage doesn't have any secrets
        else:
            _validate_user_storage_config(create_request.storage, config)
            storage = create_request.storage
            secrets = await resolve_storage_secrets_for_user(storage, workspace, sdk, auth_client)

        storage_impl = storage_impl_factory(storage, secrets)
        await storage_impl.validate_storage()

        # Resolve mutable references (e.g., 'main' -> commit SHA, None -> version ID)
        # This ensures the fileset is pinned to a specific, immutable version.
        storage = await storage_impl.resolve_config()

        # Re-create storage_impl with resolved config for validation
        storage_impl = storage_impl_factory(storage, secrets)
    except ExternalHostNotAllowedError as exc:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Storage host or endpoint not in allowed list: {exc}",
        ) from exc
    except ExternalHostInvalidError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Invalid URL for external host: {exc}") from exc
    except StorageAccessError as exc:
        logger.warning(f"Storage access denied: {exc}")
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Access denied to storage backend: {exc}",
        ) from exc
    except SecretNotFoundError as exc:
        logger.warning(f"Secret not found: {exc}")
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Secret not found: {exc}",
        ) from exc
    except SecretAccessDeniedError as exc:
        logger.warning(f"Secret access denied: {exc}")
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Access denied to secret: {exc}",
        ) from exc
    except StorageConfigError as exc:
        logger.warning(f"Storage config invalid: {exc}")
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Invalid storage configuration: {exc}",
        ) from exc
    except StorageUnavailableError as exc:
        logger.warning(f"Storage backend unavailable: {exc}")
        raise HTTPException(
            HTTP_502_BAD_GATEWAY,
            f"Storage backend unavailable: {exc}",
        ) from exc
    except StorageBackendError as exc:
        logger.warning(f"Storage validation failed: {exc}")
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Error creating fileset: {[str(exc)]}") from exc

    try:
        custom_fields = create_request.custom_fields or dict()

        # Temporary fix for making immutable datasets and preserving trust_remote_code
        service_source = custom_fields.get("service_source", None)
        if service_source and not auth_client.principal.id.startswith("service:"):
            custom_fields.pop("service_source", None)

        fileset = Fileset(
            name=create_request.name,
            workspace=workspace,
            storage=storage,
            purpose=create_request.purpose,
            metadata=create_request.metadata,
            description=create_request.description,
            custom_fields=custom_fields,
        )
        created = await entity_store.create(fileset)

        # Start cache warming if requested and storage is non-default
        is_external_storage = storage.type != config.default_storage_config.type
        if create_request.cache and is_external_storage:
            cache_storage = storage_impl_factory(config.default_storage_config, {})
            lock_manager = _get_cache_lock_manager(entity_store, config)
            background_tasks.add_task(
                warm_fileset_cache,
                source_storage=storage_impl,
                cache_storage=cache_storage,
                lock_manager=lock_manager,
            )
            logger.info(f"Started cache warming for fileset {workspace}/{create_request.name}")

        return fileset_output_from_entity(created)
    except EntityConflictError as exc:
        logger.warning(f"Fileset already exists: {workspace}/{create_request.name}")
        raise HTTPException(
            status_code=HTTP_409_CONFLICT,
            detail=f"Fileset with workspace '{workspace}' and name '{create_request.name}' already exists",
        ) from exc


@router.get(
    "/v2/workspaces/{workspace}/filesets",
    summary="List Filesets",
    status_code=HTTP_200_OK,
    response_model=FilesetPage,
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=FilesetFilter,
        filter_description="Filter filesets by name, description, purpose, storage_type, created_at, and updated_at.",
    ),
)
async def list_filesets(
    workspace: str,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=10, ge=1, le=100, description="Page size."),
    sort: GenericSortField = Query(
        default=GenericSortField.CREATED_AT_DESC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(FilesetFilter)),
    entity_store: EntityClient = Depends(get_entity_client),
) -> FilesetPage:
    """
    List Filesets endpoint with filtering and pagination.

    Supports filtering by name, description, purpose, storage_type, created_at, and updated_at via query parameters.
    Returns paginated results with sorting options.
    """
    logger.info(f"GET /filesets - workspace={workspace}")

    res = await entity_store.list(
        Fileset,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort.value,
        filter_operation=parsed.operation,
    )

    return FilesetPage(
        data=[fileset_output_from_entity(e) for e in res.data],
        pagination=PaginationData.model_validate(res.pagination.model_dump()),
        sort=sort,
    )


@router.get(
    "/v2/workspaces/{workspace}/filesets/{name}",
    summary="Get Fileset by Workspace and Name",
    status_code=HTTP_200_OK,
)
async def retrieve_fileset(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
) -> FilesetOutput:
    """
    Get Fileset by Workspace and Name.

    Returns the details of a specific fileset identified by its workspace and name.
    """
    logger.info(f"GET /filesets/{name} - workspace={workspace}")
    retrieved = await get_fileset(workspace, name, entity_store)
    return fileset_output_from_entity(retrieved)


@router.delete(
    "/v2/workspaces/{workspace}/filesets/{name}",
    summary="Delete Fileset",
    status_code=HTTP_200_OK,
    responses={
        HTTP_409_CONFLICT: {
            "description": "Fileset is referenced by a model or adapter entity",
            **_HTTP_EXCEPTION_DETAIL,
        },
        HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Fileset references could not be verified",
            **_HTTP_EXCEPTION_DETAIL,
        },
    },
)
async def delete_fileset(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetOutput:
    """
    Delete Fileset.

    Permanently deletes an unreferenced fileset from the platform.
    Referencing model or adapter entities must be relinked or deleted first.
    For local storage backends, this also deletes the underlying files.
    """
    logger.info(f"DELETE /filesets/{name} - workspace={workspace}")
    fileset = await get_fileset(workspace, name, entity_store)

    try:
        # Referential checks and deletion are separate Entity Store requests. Model
        # consumers revalidate the fileset before use so a concurrent write cannot
        # turn into a late job-runtime failure.
        reference_store = entity_store.as_service("files", internal=True)
        model_count = await _count_entity_fileset_references(
            reference_store,
            _ModelFilesetReference,
            workspace,
            name,
        )
        adapter_count = await _count_entity_fileset_references(
            reference_store,
            _AdapterFilesetReference,
            workspace,
            name,
        )
    except (EntityStoreError, NemoClientError) as exc:
        # EntityClient.list() raises NemoHTTPError / NemoTransportError (both
        # NemoClientError). EntityStoreError is reserved for grouped-count parsing.
        logger.error(
            "Cannot verify dependents for fileset '%s/%s': %s",
            _sanitize_for_log(workspace),
            _sanitize_for_log(name),
            exc,
        )
        raise HTTPException(
            HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot delete fileset '{workspace}/{name}' because its model and adapter references "
            "could not be verified. Retry when the Entities service is available.",
        ) from exc

    if model_count or adapter_count:
        raise HTTPException(
            HTTP_409_CONFLICT,
            _format_fileset_dependents(workspace, name, model_count, adapter_count),
        )

    # Delete underlying source storage data. This is a no-op for external backends
    # like NGC/HuggingFace, and removes files for backends we own (local/S3).
    try:
        secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
        storage = storage_impl_factory(fileset.storage, secrets)
        await storage.delete_all()
    except (SecretNotFoundError, SecretAccessDeniedError) as exc:
        # For backends we own (local, S3), the secret is required to delete the source
        # data; silently skipping that would orphan data we're responsible for. Surface
        # it. For external backends (NGC, HuggingFace) the source isn't ours to delete,
        # so a missing secret must not block removing the fileset - proceed.
        if fileset.storage.owns_storage_data:
            logger.error(
                f"Cannot delete owned source data for fileset '{workspace}/{name}' "
                f"because its storage secret is unavailable: {exc}"
            )
            raise HTTPException(
                HTTP_400_BAD_REQUEST,
                f"Cannot delete fileset '{workspace}/{name}': its storage secret is "
                f"unavailable, so the underlying data cannot be removed. Restore the "
                f"secret and retry. ({exc})",
            ) from exc
        logger.warning(
            f"Storage secret unavailable while deleting external fileset '{workspace}/{name}'; "
            f"nothing to delete on the source, proceeding with entity deletion: {exc}"
        )

    await entity_store.delete(Fileset, fileset.name, workspace=workspace)

    # Return the fileset data that was captured before deletion
    return fileset_output_from_entity(fileset)


@router.patch(
    "/v2/workspaces/{workspace}/filesets/{name}",
    summary="Update Fileset Metadata",
    response_model=FilesetOutput,
    status_code=HTTP_200_OK,
)
async def update_fileset_metadata(
    workspace: str,
    name: str,
    request: UpdateFilesetRequest,
    entity_store: EntityClient = Depends(get_entity_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetOutput:
    """
    Update Fileset Metadata.
    """
    logger.info(f"PATCH /filesets/{name} - workspace={workspace}")
    try:
        fileset = await get_fileset(workspace, name, entity_store)
    except EntityNotFoundError:
        raise HTTPException(
            HTTP_404_NOT_FOUND,
            f"Fileset '{workspace}/{name}' not found",
        )

    # Temporary fix for making immutable datasets and preserving trust_remote_code
    if request.custom_fields:
        original_service_source = fileset.custom_fields.get("service_source", None)
        new_service_source = request.custom_fields.pop("service_source", original_service_source)
        if not auth_client.principal.id.startswith("service:"):
            new_service_source = original_service_source

        if new_service_source:
            request.custom_fields["service_source"] = new_service_source

    diff = request.model_dump(include=request.model_fields_set)
    fileset = fileset.model_copy(update=diff)
    await entity_store.update(fileset)

    return fileset_output_from_entity(fileset)


@router.post(
    "/v2/workspaces/{workspace}/filesets/{name}/refresh",
    summary="Refresh Fileset Revision",
    response_model=FilesetOutput,
    status_code=HTTP_200_OK,
    responses={
        HTTP_400_BAD_REQUEST: {
            "description": "Storage configuration, secret, or source rejected the refresh",
            **_HTTP_EXCEPTION_DETAIL,
        },
        HTTP_404_NOT_FOUND: {
            "description": "Fileset not found",
            **_HTTP_EXCEPTION_DETAIL,
        },
        HTTP_409_CONFLICT: {
            "description": "Fileset does not track a mutable revision",
            **_HTTP_EXCEPTION_DETAIL,
        },
        HTTP_502_BAD_GATEWAY: {
            "description": "Storage backend unavailable",
            **_HTTP_EXCEPTION_DETAIL,
        },
    },
)
async def refresh_fileset(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetOutput:
    """
    Re-resolve a fileset's tracked revision against its source.

    A fileset created from a mutable ref is pinned to an immutable id so its
    contents cannot shift under a deployment. This re-resolves that same ref and
    repoints the fileset at whatever it names now. Everything else about the
    storage config, including the repository and directory, is left alone.

    A running deployment keeps serving the revision it staged. It moves to this one
    when the runner next stages the fileset, and records the revision it staged.
    """
    logger.info("POST /filesets/%s/refresh - workspace=%s", _sanitize_for_log(name), _sanitize_for_log(workspace))
    try:
        fileset = await get_fileset(workspace, name, entity_store)
    except EntityNotFoundError as exc:
        raise HTTPException(
            HTTP_404_NOT_FOUND,
            f"Fileset '{workspace}/{name}' not found",
        ) from exc

    # Decided before secrets are resolved: whether a fileset tracks anything is a
    # property of its stored config, and resolving first would report a missing or
    # forbidden secret as 400 for a fileset whose real answer is 409.
    if not fileset.storage.tracked_revision:
        raise HTTPException(
            HTTP_409_CONFLICT,
            f"Fileset '{workspace}/{name}' does not track a revision that can be refreshed",
        )

    try:
        secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
        storage_impl = storage_impl_factory(fileset.storage, secrets)
        tracked_impl = storage_impl_factory(storage_impl.config_at_tracked_revision(), secrets)
        # The host allowlist is enforced at create time; re-check it here so a host
        # removed from it since then stops being reachable through a refresh.
        await tracked_impl.validate_storage()
        storage = await tracked_impl.resolve_config()
    except ExternalHostNotAllowedError as exc:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            f"Storage host or endpoint not in allowed list: {exc}",
        ) from exc
    except ExternalHostInvalidError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Invalid URL for external host: {exc}") from exc
    except (SecretNotFoundError, SecretAccessDeniedError) as exc:
        logger.warning(
            "Secret unavailable while refreshing %s/%s: %s",
            _sanitize_for_log(workspace),
            _sanitize_for_log(name),
            exc,
        )
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Secret unavailable: {exc}") from exc
    except StorageAccessError as exc:
        logger.warning(f"Storage access denied: {exc}")
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Access denied to storage backend: {exc}") from exc
    except StorageConfigError as exc:
        logger.warning(f"Storage config invalid: {exc}")
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Invalid storage configuration: {exc}") from exc
    except StorageUnavailableError as exc:
        logger.warning(f"Storage backend unavailable: {exc}")
        raise HTTPException(HTTP_502_BAD_GATEWAY, f"Storage backend unavailable: {exc}") from exc
    except StorageBackendError as exc:
        logger.warning(f"Storage refresh failed: {exc}")
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Error refreshing fileset: {exc}") from exc

    fileset = fileset.model_copy(update={"storage": storage})
    await entity_store.update(fileset)

    return fileset_output_from_entity(fileset)


@router.get(
    "/v2/workspaces/{workspace}/filesets/{name}/files",
    summary="List Fileset Files",
    status_code=HTTP_200_OK,
)
async def list_fileset_files(
    workspace: str,
    name: str,
    path: str | None = Query(default=None, description="Filter files by path prefix"),
    include_cache_status: bool = Query(
        default=False,
        description="Check and return cache status for each file. "
        "When false, storage files return null for cache_status.",
    ),
    entity_store: EntityClient = Depends(get_entity_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ListFilesetFilesResponse:
    """
    List Files in Fileset.

    Returns a list of files stored in the specified fileset.
    Optionally filter by path prefix to list files under a specific directory.

    Each file includes a cache_status field:
    - "not_cacheable": File is on default storage, caching not applicable
    - "cached": File exists in cache storage
    - "caching": File is currently being downloaded and cached
    - "not_cached": File not in cache, will be cached on next download
    - null: External storage, but cache status not checked (use include_cache_status=true)
    """
    logger.info(f"GET /filesets/{name}/files - workspace={workspace}, path={path}")
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)
    files = await list_storage_files(storage, path)

    is_external_storage = fileset.storage.type != config.default_storage_config.type

    # Determine cache status for each file
    if not is_external_storage:
        # Default storage: not cacheable for all files
        cache_status_map = {f.path: CacheStatus.NOT_CACHEABLE for f in files}
    elif include_cache_status:
        # External storage with opt-in: actually check cache
        cache_storage = storage_impl_factory(config.default_storage_config, {})
        lock_manager = _get_cache_lock_manager(entity_store, config)
        cache_status_map = await get_cache_status_for_files(files, storage, cache_storage, lock_manager)
    else:
        # External storage without opt-in: null (didn't check)
        cache_status_map = {}

    return list_fileset_files_from_infos(fileset, files, cache_status_map)


@router.head(
    "/v2/workspaces/{workspace}/filesets/{name}/-/{path:path}",
    summary="Get File Metadata",
    status_code=HTTP_200_OK,
)
async def head_file(
    workspace: str,
    name: str,
    path: str,
    entity_store: EntityClient = Depends(get_entity_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> Response:
    """
    Get file metadata without downloading content.

    HEAD requests are often used before Range GETs to ensure the server
    supports partial downloads (e.g., DuckDB's httpfs).
    Returns Accept-Ranges, Content-Length, and Content-Type headers.
    """
    logger.info(f"HEAD /filesets/{name}/-/{path} - workspace={workspace}")
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    cache_ctx: CacheContext | None = None
    if fileset.storage.type != config.default_storage_config.type:
        cache_ctx = CacheContext(
            storage=storage_impl_factory(config.default_storage_config, {}),
            lock_manager=_get_cache_lock_manager(entity_store, config),
        )

    file_info = await get_download_file_info(
        storage,
        path,
        f"{workspace}/{name}",
        cache_ctx=cache_ctx,
    )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_info.size),
        "Content-Type": "application/octet-stream",
    }

    return Response(status_code=HTTP_200_OK, headers=headers)


@router.get(
    "/v2/workspaces/{workspace}/filesets/{name}/-/{path:path}",
    summary="Download File Content",
    responses={
        HTTP_200_OK: {
            "description": "Successful Response",
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
    response_class=FileResponse,  # coerces SDK into returning a File
)
async def download_file(
    workspace: str,
    name: str,
    path: str,
    request: Request,
    background_tasks: BackgroundTasks,
    entity_store: EntityClient = Depends(get_entity_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> Response:
    """
    Download file content from a fileset.

    Supports HTTP Range requests for partial content retrieval (status 206).
    Returns the full file content (status 200) if no Range header is provided.
    For external resources (HuggingFace, NGC), content is cached locally on first access.
    """
    logger.info(f"GET /filesets/{name}/-/{path} - workspace={workspace}")
    fileset = await get_fileset(workspace, name, entity_store)
    with scoped_app_ctx(FilesContext(fileset_name=fileset.name, path=path)):
        secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
        storage = storage_impl_factory(fileset.storage, secrets)

        cache_ctx: CacheContext | None = None
        if fileset.storage.type != config.default_storage_config.type:
            # If the file being downloaded isn't from the default
            # storage backend, attempt to cache it into the default storage backend.
            # TODO: In the future we can allow filesets to specify whether they want to be cached,
            # and where.
            cache_ctx = CacheContext(
                storage=storage_impl_factory(config.default_storage_config, {}),
                lock_manager=_get_cache_lock_manager(entity_store, config),
            )

        file_info = await get_download_file_info(
            storage,
            path,
            f"{workspace}/{name}",
            cache_ctx=cache_ctx,
        )
        return await stream_file_download(
            storage=storage,
            path=path,
            request=request,
            file_size=file_info.size,
            cache_ctx=cache_ctx,
            background_tasks=background_tasks,
        )


@router.put(
    "/v2/workspaces/{workspace}/filesets/{name}/-/{path:path}",
    summary="Upload Fileset Content",
    status_code=HTTP_200_OK,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/octet-stream": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": "Raw binary file content",
                    }
                }
            },
            "required": True,
            "description": "Upload the file either as a raw octet stream.",
        }
    },
)
async def upload_file(
    workspace: str,
    name: str,
    path: str,
    request: Request,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetFileOutput:
    """Upload file content to a fileset."""
    logger.info(f"PUT /filesets/{name}/-/{path} - workspace={workspace}")
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    # Determine chunk processor based on Content-Type
    content_type = request.headers.get("content-type", "application/octet-stream").lower()

    if "multipart/form-data" in content_type:
        chunk_processor = MultipartChunkProcessor(request.headers)
    else:
        # Default to octet-stream for application/octet-stream or any other type
        chunk_processor = OctetStreamChunkProcessor()

    # Temporary fix for making immutable datasets and preserving trust_remote_code
    service_source = fileset.custom_fields.get("service_source")
    if service_source and not auth_client.principal.id.startswith("service:"):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Access denied: This fileset's files are immutable",
        )

    # Get content length from request headers for backends that need it (e.g., S3)
    content_length_header = request.headers.get("content-length")
    content_length = int(content_length_header) if content_length_header else None

    with scoped_app_ctx(FilesContext(fileset_name=fileset.name, path=path)):
        try:
            async with streaming_file_upload(request, chunk_processor) as upload:
                file_info = await storage.upload(path, upload, content_length=content_length)
            return fileset_file_output_from_info(
                workspace=workspace,
                name=name,
                file_info=file_info,
            )
        except InvalidPathError as e:
            logger.warning(f"Invalid path for upload attempt: {workspace}/{name}/-/{path}")
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except InactivityTimeoutError:
            msg = f"Connection terminated due to inactivity: {path}"
            logger.error(msg)
            raise HTTPException(status_code=HTTP_408_REQUEST_TIMEOUT, detail=msg)
        except OSError as e:
            logger.exception(f"Storage error for upload at {path}")
            raise HTTPException(
                status_code=HTTP_507_INSUFFICIENT_STORAGE,
                detail=f"Storage error: {str(e)}",
            )
        except Exception as e:
            logger.exception(f"Failed to upload at {path}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}",
            )


@router.delete(
    "/v2/workspaces/{workspace}/filesets/{name}/-/{path:path}",
    summary="Delete a specific file from a fileset",
    status_code=HTTP_200_OK,
)
async def delete_file(
    workspace: str,
    name: str,
    path: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FilesetFileOutput:
    """
    Delete a specific file from a fileset.

    Permanently deletes the file from the storage backend.
    Returns metadata about the deleted file.
    """
    logger.info(f"DELETE /filesets/{name}/-/{path} - workspace={workspace}")
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_storage_secrets_for_user(fileset.storage, workspace, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    try:
        file_info = await storage.delete(path)
    except InvalidPathError as e:
        logger.warning(f"Invalid path for deletion attempt: {workspace}/{name}/-/{path}")
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            str(e),
        ) from e
    except NotFoundError as e:
        logger.warning(f"File not found for deletion: {workspace}/{name}/-/{path}")
        raise HTTPException(
            HTTP_404_NOT_FOUND,
            f"File '{path}' not found in fileset '{workspace}/{name}'",
        ) from e

    return fileset_file_output_from_info(workspace, name, file_info)
