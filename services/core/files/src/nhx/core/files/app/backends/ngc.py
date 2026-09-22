# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NGC storage backend for NVIDIA NGC resources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    AsyncIterator,
    Union,
)

import aiohttp
from anyio import to_thread
from ngcbase.constants import SCOPED_KEY_PREFIX
from ngcbase.errors import NgcException, ResourceNotFoundException
from ngcsdk import Client
from nhx.common.files.storage_config import NGCStorageConfig as NGCStorageConfig
from nhx.core.files.app.backends.base import (
    ByteRange,
    FileInfo,
    StorageImpl,
)
from nhx.core.files.app.external_hosts import validate_external_host
from nhx.core.files.app.http_session import get_http_session
from nhx.core.files.app.streaming import download_url, download_url_streaming
from nhx.core.files.exceptions import StorageBackendError
from registry.api.models import GuestModelAPI, ModelAPI
from registry.api.resources import ResourceAPI

logger = logging.getLogger(__name__)


# TODO: Add a better hierarchy of user-input errors vs server errors
class NGCBackendError(StorageBackendError):
    """Raised when there's issues talking to NGC."""


# Type alias for the NGC registry APIs we use. Resources always use ResourceAPI;
# models use GuestModelAPI (public org) or ModelAPI (private org). All share the
# same duck-typed interface for info/list_files/list_versions/get_direct_download_URL.
_RegistryAPI = Union[ResourceAPI, ModelAPI, GuestModelAPI]


@dataclass
class NGCStorageImpl(StorageImpl):
    config: NGCStorageConfig
    secrets: dict[str, str]

    _client: Client | None = field(init=False, default=None)
    _registry_api: _RegistryAPI | None = field(init=False, default=None)
    _version: str | None = field(init=False, default=None)

    # The NGC registry has several asset types: "resource" and "model" (among others).
    # Resources always use ResourceAPI (org-scoped); models use GuestModelAPI for
    # public orgs (e.g. nvidia) since org-scoped ModelAPI returns 403 for keys from
    # other orgs. GuestResourceAPI is not used—it returns 403 for most public resources.
    _PUBLIC_ORGS = {"nvidia"}

    def __post_init__(self):
        api_key = self.secrets.get("api_key")
        if not api_key or not api_key.startswith(SCOPED_KEY_PREFIX):
            raise NGCBackendError("Invalid API key. Legacy NGC keys are not supported.")

    async def _get_client(self) -> Client:
        """Lazily create and configure the NGC client."""
        if self._client is not None:
            return self._client

        api_key = self.secrets.get("api_key")
        is_public = self.config.org in self._PUBLIC_ORGS

        def _configure() -> Client:
            client = Client()
            if is_public:
                # Users create API keys at org level (no team) or team level; the SDK rejects
                # configure(org=target) when the key's org differs. Use team_name="no-team" and
                # omit org so the SDK auto-populates org from the key. This works for both key types.
                # Target org/team are passed explicitly in every Guest API call.
                client.configure(api_key=api_key, team_name="no-team")
            else:
                # For private orgs, configure with the org and team.
                client.configure(
                    api_key=api_key,
                    org_name=self.config.org,
                    team_name=self.config.team,
                )
            return client

        try:
            self._client = await to_thread.run_sync(_configure)
        except (ValueError, NgcException) as exc:
            raise NGCBackendError(f"Error creating NGC storage backend: [{str(exc)}]") from exc

        return self._client

    async def _get_registry_api(self) -> _RegistryAPI:
        """Lazily create the NGC registry API."""
        if self._registry_api is not None:
            return self._registry_api

        await self._get_client()
        is_public = self.config.org in self._PUBLIC_ORGS
        self._registry_api = self._create_registry_api(is_public)
        return self._registry_api

    def _create_registry_api(self, is_public: bool) -> _RegistryAPI:
        """Select the correct NGC registry API based on target_type and org visibility.
        Resources always use ResourceAPI. Models use GuestModelAPI for public orgs."""
        if self.config.target_type == "model":
            return GuestModelAPI(self._client) if is_public else ModelAPI(self._client)
        elif self.config.target_type == "resource":
            return ResourceAPI(self._client)
        else:
            raise NGCBackendError(f"Invalid target type: {self.config.target_type}")

    async def _get_version(self) -> str:
        """Lazily fetch and cache the NGC version."""
        if self._version is not None:
            return self._version

        if self.config.version is not None:
            self._version = self.config.version
        else:
            self._version = await self._get_latest_version()
        return self._version

    async def _get_latest_version(self) -> str:
        """Fetch the latest version of the NGC asset."""
        registry_api = await self._get_registry_api()

        def _fetch_latest() -> str:
            # Use positional args for the name parameter since ResourceAPI expects
            # `resource_name` while ModelAPI expects `model_name`.
            version_list = registry_api.list_versions(
                self.config.org,
                self.config.team,
                self.config.target,
            )
            try:
                latest = next(version_list)
                return latest.versionId
            except StopIteration:
                raise NGCBackendError(
                    f"No versions found for NGC {self.config.target_type} "
                    f"{self.config.org}/{self.config.team}/{self.config.target}"
                )

        return await to_thread.run_sync(_fetch_latest)

    async def resolve_config(self) -> NGCStorageConfig:
        """Resolve the version to a specific version ID.

        Returns:
            A new NGCStorageConfig with the resolved version ID.
        """
        version = await self._get_version()
        return self.config.model_copy(
            update={
                "original_version": self.config.version,
                "version": version,
            }
        )

    async def _get_target(self) -> str:
        """Returns org/team/target:version format for NGC API calls."""
        version = await self._get_version()
        return f"{self.config.org}/{self.config.team}/{self.config.target}:{version}"

    async def _get_target_with_version(self) -> str:
        """Returns target:version format for error messages."""
        version = await self._get_version()
        return f"{self.config.target}:{version}"

    async def _get_signed_url(self, filepath: str) -> str:
        """Get a signed URL for downloading a file from NGC."""
        registry_api = await self._get_registry_api()
        version = await self._get_version()

        # Get signed URL from NGC API
        route = registry_api.get_direct_download_URL(
            name=self.config.target,
            version=version,
            org=self.config.org,
            team=self.config.team,
            filepath=filepath,
        )
        url = f"{self.config.host}/{route}"

        # Get the actual signed URL
        try:
            data = await download_url(
                url=url,
                headers={"Authorization": f"Bearer {self.secrets.get('api_key')}"},
            )
            return data["urls"][0]
        except aiohttp.ClientResponseError as exc:
            raise NGCBackendError(f"Failed to get signed URL for {filepath}: HTTP {exc.status} {exc.message}") from exc
        except aiohttp.ClientError as exc:
            raise NGCBackendError(f"Network error getting signed URL for {filepath}") from exc

    async def list_files(self, path: str | None = None) -> list[FileInfo]:
        """List files in the NGC asset."""
        registry_api = await self._get_registry_api()
        target = await self._get_target()

        try:
            files = await to_thread.run_sync(registry_api.list_files, target)
        except ResourceNotFoundException as exc:
            raise NGCBackendError("Failed to list files") from exc

        file_infos = []
        for f in files:
            file_path = f.path
            if path is None or file_path.startswith(path):
                file_infos.append(
                    FileInfo(
                        path=file_path,
                        size=f.sizeInBytes,
                    )
                )

        return file_infos

    async def download(self, path: str, byte_range: ByteRange | None) -> AsyncIterator[bytes]:
        """Download a file from NGC.

        This method:
        1. Gets a signed URL from NGC
        2. Downloads the file via HTTP
        3. Streams the content back as an async iterator
        """
        try:
            signed_url = await self._get_signed_url(path)
        except Exception as exc:
            raise NGCBackendError(f"Failed to get signed URL for file {path}") from exc

        target_with_version = await self._get_target_with_version()

        async def _download() -> AsyncIterator[bytes]:
            session = get_http_session()
            try:
                async for chunk in download_url_streaming(
                    url=signed_url,
                    session=session,
                    byte_range=byte_range,
                    chunk_size=self.config.read_chunk_size,
                ):
                    yield chunk
            except aiohttp.ClientResponseError as exc:
                if exc.status == 404:
                    raise NGCBackendError(
                        f"File {path} not found in NGC {self.config.target_type} {target_with_version}"
                    ) from exc
                raise NGCBackendError(f"HTTP error downloading file {path}: {exc.status}") from exc
            except aiohttp.ClientError as exc:
                raise NGCBackendError(f"Network error downloading file {path}") from exc

        return _download()

    async def validate_storage(self):
        """Validate that we can access the NGC asset."""
        validate_external_host(self.config.host)

        try:
            registry_api = await self._get_registry_api()
            target = await self._get_target()
            await to_thread.run_sync(registry_api.info, target)
        except NGCBackendError:
            raise
        except ResourceNotFoundException as exc:
            raise NGCBackendError(f"NGC {self.config.target_type} not found: {self.config.target}") from exc
        except Exception as exc:
            raise NGCBackendError(
                f"Failed to access NGC {self.config.target_type} {self.config.target} [{exc}]"
            ) from exc

    async def upload(
        self,
        path: str,
        fstream: AsyncIterator[bytes],
        content_length: int | None = None,
    ) -> FileInfo:
        raise NotImplementedError("NGC upload is not implemented")

    async def delete(self, path: str) -> FileInfo:
        raise NotImplementedError("NGC delete is not implemented")

    async def get_cache_path_key(self, path: str | None = None) -> str:
        """
        Return cache path that includes NGC org, team, target, and version for uniqueness.

        Format: cache/ngc/{org}/{team}/{target}/{version}/{path}
        This ensures different versions don't overwrite each other in cache.

        Args:
            path: File path within the asset. If None, returns the cache root prefix.
        """
        version = await self._get_version()
        prefix = f"cache/ngc/{self.config.org}/{self.config.team}/{self.config.target}/{version}"
        if path is None:
            return prefix
        return f"{prefix}/{path}"
