# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lookups that help a caller set up a git fileset before creating it."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.storage_config import parse_ssh_remote
from nemo_helix_plugin.files.types import (
    FindGitRepositoryFilesRequest,
    FindGitRepositoryFilesResponse,
    ScanSshHostKeysRequest,
    ScanSshHostKeysResponse,
)
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.service.dependencies import get_nemo_client, get_service_config_factory
from nhx.core.files.api.endpoint_helpers import (
    HTTP_EXCEPTION_DETAIL,
    resolve_storage_secrets_for_user,
    storage_errors_as_http,
)
from nhx.core.files.app.external_hosts import ExternalHostNotAllowedError, validate_external_host
from nhx.core.files.config import FilesConfig
from nhx.core.files.exceptions import StorageBackendError
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/ssh-host-keys/scan",
    summary="Scan SSH Host Keys",
    response_model=ScanSshHostKeysResponse,
    status_code=HTTP_200_OK,
    responses={
        HTTP_400_BAD_REQUEST: {
            "description": "The URL is not an SSH remote, or its host is not in the allowed list",
            **HTTP_EXCEPTION_DETAIL,
        },
        HTTP_502_BAD_GATEWAY: {"description": "The host could not be scanned", **HTTP_EXCEPTION_DETAIL},
    },
)
async def scan_ssh_host_keys(
    workspace: str,
    request: ScanSshHostKeysRequest,
    config: FilesConfig = Depends(get_service_config_factory(FilesConfig)),
) -> ScanSshHostKeysResponse:
    """
    Return the public host keys an SSH remote presents, with their fingerprints.

    Each key type is fetched over its own unauthenticated connection, so confirm one
    key's fingerprint and use only that key's `known_hosts_line` as a git fileset's
    `known_hosts`. Only hosts in the allowed external hosts list are scanned.
    """
    from nhx.core.files.app.backends.git import scan_host_keys

    try:
        remote = parse_ssh_remote(request.url.strip())
        validate_external_host(remote.host_url, config.get_allowed_external_hosts())
    except ExternalHostNotAllowedError as exc:
        logger.warning("SSH host key scan refused: %s", exc)
        raise HTTPException(HTTP_400_BAD_REQUEST, f"Storage host or endpoint not in allowed list: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        keys = await scan_host_keys(remote)
    except StorageBackendError as exc:
        logger.warning("SSH host key scan failed: %s", exc)
        raise HTTPException(HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return ScanSshHostKeysResponse(host=remote.authority, keys=keys)


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
        HTTP_502_BAD_GATEWAY: {"description": "The repository could not be reached", **HTTP_EXCEPTION_DETAIL},
    },
)
async def find_git_repository_files(
    workspace: str,
    request: FindGitRepositoryFilesRequest,
    client: AsyncNemoClient = Depends(get_nemo_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> FindGitRepositoryFilesResponse:
    """
    Find every file with a given name in the commit a git storage config resolves to.

    Runs the same checks as creating a fileset with this storage, without creating
    one. A fileset created afterwards from the same branch or tag resolves it again,
    so if the branch has moved since, its files may differ from the paths returned.
    """
    from nhx.core.files.app.backends.git import GitStorageImpl

    with storage_errors_as_http("Error reading the repository"):
        secrets = await resolve_storage_secrets_for_user(request.storage, workspace, client, auth_client)
        impl = GitStorageImpl(request.storage, secrets)
        await impl.validate_storage()
        resolved = await impl.resolve_config()
        files = await impl.list_files()

    suffix = f"/{request.file_name}"
    paths = sorted(file.path for file in files if file.path == request.file_name or file.path.endswith(suffix))
    return FindGitRepositoryFilesResponse(revision=resolved.revision, paths=paths)
