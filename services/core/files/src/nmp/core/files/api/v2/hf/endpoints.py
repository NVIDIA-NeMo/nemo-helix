# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub compatible API endpoints.

These endpoints provide compatibility with the HuggingFace Hub client library,
allowing users to download files from NeMo Platform filesets using huggingface_hub.

All endpoints assume model repo type (the default in huggingface_hub).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from nemo_platform import AsyncNeMoPlatform
from nmp.common.auth import AuthClient, get_auth_client
from nmp.common.entities.client import EntityClient
from nmp.common.service.dependencies import (
    get_entity_client,
    get_sdk_client,
    get_service_config_factory,
)
from nmp.core.files.api.endpoint_helpers import (
    CacheContext,
    get_download_file_info,
    get_file_info,
    get_fileset,
    list_storage_files,
    resolve_fileset_secrets,
    stream_file_download,
)
from nmp.core.files.api.v2.hf.schemas import (
    HfRepoInfo,
    HfSibling,
    HfTreeEntry,
    PathInfo,
    PathsInfoRequest,
)
from nmp.core.files.api.v2.hf.utils import (
    generate_commit_hash,
    generate_etag,
)
from nmp.core.files.app.backends import storage_impl_factory
from nmp.core.files.app.file_lock import FileLockManager
from nmp.core.files.config import FilesConfig
from starlette.status import HTTP_200_OK

router = APIRouter(prefix="/v2/hf", include_in_schema=False)


def _format_datetime_for_hf(dt: datetime | None) -> str:
    """Format datetime in HuggingFace expected format."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _hf_headers(fileset_name: str, updated_at: datetime | None, path: str, size: int) -> dict[str, str]:
    """Generate HuggingFace-specific headers (X-Repo-Commit, ETag)."""
    return {
        "X-Repo-Commit": generate_commit_hash(fileset_name, updated_at),
        "ETag": generate_etag(fileset_name, path, size),
    }


@router.head(
    "/{workspace}/{name}/resolve/{revision}/{path:path}",
    summary="Get file metadata (HF compatible)",
    status_code=HTTP_200_OK,
)
async def head_file(
    workspace: str,
    name: str,
    revision: str,
    path: str,
    entity_store: EntityClient = Depends(get_entity_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> Response:
    """Get file metadata without downloading content."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    cache_ctx: CacheContext | None = None
    if fileset.storage.type != config.default_storage_config.type:
        cache_ctx = CacheContext(
            storage=storage_impl_factory(config.default_storage_config, {}),
            lock_manager=FileLockManager(
                entity_client=entity_store,
                workspace="system",
                lock_ttl_seconds=config.file_lock_ttl_seconds,
            ),
        )

    file_info = await get_download_file_info(storage, path, cache_ctx=cache_ctx)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_info.size),
        "Content-Type": "application/octet-stream",
        **_hf_headers(fileset.name, fileset.updated_at, path, file_info.size),
    }

    return Response(status_code=HTTP_200_OK, headers=headers)


@router.get(
    "/{workspace}/{name}/resolve/{revision}/{path:path}",
    summary="Download file content (HF compatible)",
)
async def download_file(
    workspace: str,
    name: str,
    revision: str,
    path: str,
    request: Request,
    background_tasks: BackgroundTasks,
    entity_store: EntityClient = Depends(get_entity_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> Response:
    """Download file content with Range support."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    # Set up caching for external storage backends (HuggingFace, NGC, etc.)
    cache_ctx: CacheContext | None = None
    if fileset.storage.type != config.default_storage_config.type:
        cache_ctx = CacheContext(
            storage=storage_impl_factory(config.default_storage_config, {}),
            lock_manager=FileLockManager(
                entity_client=entity_store,
                workspace="system",
                lock_ttl_seconds=config.file_lock_ttl_seconds,
            ),
        )

    file_info = await get_download_file_info(storage, path, cache_ctx=cache_ctx)
    extra_headers = _hf_headers(fileset.name, fileset.updated_at, path, file_info.size)
    return await stream_file_download(
        storage=storage,
        path=path,
        request=request,
        file_size=file_info.size,
        background_tasks=background_tasks,
        cache_ctx=cache_ctx,
        extra_headers=extra_headers,
    )


@router.get(
    "/api/models/{workspace}/{name}",
    summary="Get repository info (HF compatible)",
    status_code=HTTP_200_OK,
)
async def get_repo_info(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> HfRepoInfo:
    """Get repository metadata including file list."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    files = await list_storage_files(storage)
    siblings = [HfSibling(rfilename=f.path, size=f.size) for f in files]

    repo_id = f"{workspace}/{name}"
    return HfRepoInfo(
        id=repo_id,
        sha=generate_commit_hash(fileset.name, fileset.updated_at),
        lastModified=_format_datetime_for_hf(fileset.updated_at or fileset.created_at),
        siblings=siblings,
        modelId=repo_id,
    )


@router.get(
    "/api/models/{workspace}/{name}/revision/{revision}",
    summary="Get repository info at revision (HF compatible)",
    status_code=HTTP_200_OK,
)
async def get_repo_info_at_revision(
    workspace: str,
    name: str,
    revision: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> HfRepoInfo:
    """Get repository metadata including file list (revision is ignored)."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    files = await list_storage_files(storage)
    siblings = [HfSibling(rfilename=f.path, size=f.size) for f in files]

    repo_id = f"{workspace}/{name}"
    return HfRepoInfo(
        id=repo_id,
        sha=generate_commit_hash(fileset.name, fileset.updated_at),
        lastModified=_format_datetime_for_hf(fileset.updated_at or fileset.created_at),
        siblings=siblings,
        modelId=repo_id,
    )


@router.get(
    "/api/models/{workspace}/{name}/tree/{revision}",
    summary="List repository files (HF compatible)",
    status_code=HTTP_200_OK,
)
async def get_tree(
    workspace: str,
    name: str,
    revision: str,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> list[HfTreeEntry]:
    """List files in repository."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)
    files = await list_storage_files(storage)

    return [
        HfTreeEntry(
            type="file",
            oid=generate_etag(fileset.name, f.path, f.size).strip('"'),
            size=f.size,
            path=f.path,
        )
        for f in files
    ]


@router.post(
    "/api/models/{workspace}/{name}/paths-info/{revision}",
    summary="Get info for specific paths (HF compatible)",
    status_code=HTTP_200_OK,
)
async def get_paths_info(
    workspace: str,
    name: str,
    revision: str,
    request_body: PathsInfoRequest,
    entity_store: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> list[PathInfo]:
    """Get info for specific file paths."""
    fileset = await get_fileset(workspace, name, entity_store)
    secrets = await resolve_fileset_secrets(fileset, sdk, auth_client)
    storage = storage_impl_factory(fileset.storage, secrets)

    result = []
    for path in request_body.paths:
        try:
            file_info = await get_file_info(storage, path)
            result.append(
                PathInfo(
                    path=path,
                    type="file",
                    size=file_info.size,
                    oid=generate_etag(fileset.name, path, file_info.size).strip('"'),
                )
            )
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            continue

    return result
