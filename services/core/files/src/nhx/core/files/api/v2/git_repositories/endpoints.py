# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lookups that help a caller set up a git fileset before creating it."""

from fastapi import APIRouter, Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.types import FindGitRepositoryFilesRequest, FindGitRepositoryFilesResponse
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.service.dependencies import get_nemo_client, get_service_config_factory
from nhx.core.files.api.endpoint_helpers import (
    HTTP_EXCEPTION_DETAIL,
    resolve_storage_secrets_for_user,
    storage_errors_as_http,
)
from nhx.core.files.app.external_hosts import validate_external_host
from nhx.core.files.config import FilesConfig
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY

router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/git-repositories/find-files",
    summary="Find Files in a Git Repository",
    response_model=FindGitRepositoryFilesResponse,
    status_code=HTTP_200_OK,
    responses={
        HTTP_400_BAD_REQUEST: {
            "description": "The storage, its secret, or its host was rejected",
            **HTTP_EXCEPTION_DETAIL,
        },
        HTTP_502_BAD_GATEWAY: {
            "description": "The host or repository could not be reached",
            **HTTP_EXCEPTION_DETAIL,
        },
    },
)
async def find_git_repository_files(
    workspace: str,
    request: FindGitRepositoryFilesRequest,
    client: AsyncNemoClient = Depends(get_nemo_client),
    auth_client: AuthClient = Depends(get_auth_client),
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
) -> FindGitRepositoryFilesResponse:
    """
    Find every file with a given name in the commit a git storage config resolves to.

    Without `known_hosts`, returns the host's public keys instead and reads nothing, as
    ssh prompts on a first connection: confirm one key's fingerprint, then repeat the
    request with only that key's `known_hosts_line`.

    With `known_hosts`, runs the same checks as creating a fileset with this storage,
    without creating one. A fileset created afterwards from the same branch or tag
    resolves it again, so if the branch has moved since, its files may differ.
    """
    from nhx.core.files.app.backends.git import GitStorageImpl, scan_host_keys

    storage = request.storage
    if not storage.known_hosts.strip():
        with storage_errors_as_http("Error scanning the host's keys"):
            validate_external_host(storage.remote.host_url, config.get_allowed_external_hosts())
            host_keys = await scan_host_keys(storage.remote)
        return FindGitRepositoryFilesResponse(host_keys=host_keys)

    with storage_errors_as_http("Error reading the repository"):
        secrets = await resolve_storage_secrets_for_user(storage, workspace, client, auth_client)
        impl = GitStorageImpl(storage, secrets)
        await impl.validate_storage()
        resolved = await impl.resolve_config()
        files = await impl.list_files()

    suffix = f"/{request.file_name}"
    paths = sorted(file.path for file in files if file.path == request.file_name or file.path.endswith(suffix))
    return FindGitRepositoryFilesResponse(revision=resolved.revision, paths=paths)
