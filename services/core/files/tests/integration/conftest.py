# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test fixtures for Files service.

This conftest provides fixtures for integration tests that require
external services (like Huggingface Hub).
"""

from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import huggingface_hub
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from filesets.resources import FilesResource
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import FilesetOutput
from nmp.common.auth import AuthClient, get_auth_client
from nmp.common.auth.models import Principal
from nmp.common.config import AuthConfig
from nmp.common.config.base import get_service_config
from nmp.core.files.app.backends import storage_impl_factory
from nmp.core.files.app.backends.base import StorageImpl
from nmp.core.files.app.backends.local import LocalStorageConfig
from nmp.core.files.config import FilesConfig
from nmp.core.files.service import FilesService
from nmp.core.files.testing.utils import create_fileset
from nmp.core.secrets.service import SecretsService
from nmp.testing import SDKTestClientAdapter, create_test_client
from packaging import version

# Mock auth client for fileset endpoints that depend on get_auth_client
# (create_fileset, update_fileset_metadata, upload_file).
_mock_auth_principal = Principal(id="test@example.com")
_mock_auth_config = AuthConfig(enabled=False)
_mock_auth_client = AuthClient(principal=_mock_auth_principal, config=_mock_auth_config)


def _mock_get_auth_client():
    return _mock_auth_client


# Dependency overrides for tests that call endpoints requiring auth_client.
FILESET_AUTH_DEPENDENCY_OVERRIDES = {get_auth_client: _mock_get_auth_client}


def _get_auth_client_from_request(request: Request) -> AuthClient:
    """Resolve principal from X-NMP-Principal-Id header for tests that need multiple principals."""
    pid = request.headers.get("x-nmp-principal-id", "test@example.com")
    return AuthClient(
        principal=Principal(id=pid),
        config=_mock_auth_config,
    )


def _install_asgi_files_resource(sdk: NeMoPlatform) -> None:
    sdk.__dict__["files"] = FilesResource(
        sdk,
        files_client=client_from_platform(sdk, FilesClient),
    )


@pytest.fixture
def sdk_user_and_service() -> Iterator[tuple[NeMoPlatform, NeMoPlatform]]:
    """Two SDKs sharing the same app: default user principal and service:customizer.

    Yields (sdk_user, sdk_service). Use when testing service_source immutability
    with both principals against the same fileset.
    """
    with create_test_client(
        FilesService,
        SecretsService,
        dependency_overrides={get_auth_client: _get_auth_client_from_request},
    ) as sdk_base:
        sdk_http_client = sdk_base._client
        assert isinstance(sdk_http_client, SDKTestClientAdapter)
        app = sdk_http_client.asgi_app
        base_url = "http://testserver"
        client_user = TestClient(
            app,
            base_url=base_url,
            headers={"x-nmp-principal-id": "test@example.com"},
        )
        client_service = TestClient(
            app,
            base_url=base_url,
            headers={"x-nmp-principal-id": "service:customizer"},
        )
        try:
            sdk_user = NeMoPlatform(
                base_url=base_url,
                http_client=SDKTestClientAdapter(client_user),
                max_retries=0,
            )
            sdk_service = NeMoPlatform(
                base_url=base_url,
                http_client=SDKTestClientAdapter(client_service),
                max_retries=0,
            )
            _install_asgi_files_resource(sdk_user)
            _install_asgi_files_resource(sdk_service)
            yield (sdk_user, sdk_service)
        finally:
            client_user.close()
            client_service.close()


@pytest.fixture
def sdk() -> Iterator[NeMoPlatform]:
    """SDK client backed by the test client."""
    with create_test_client(
        FilesService,
        SecretsService,
        dependency_overrides=FILESET_AUTH_DEPENDENCY_OVERRIDES,
    ) as sdk:
        _install_asgi_files_resource(sdk)
        yield sdk


@pytest.fixture
def files_client(sdk: NeMoPlatform) -> FilesClient:
    """Provide a FilesClient derived from the SDK."""
    return client_from_platform(sdk, FilesClient)


@pytest.fixture
async def async_files_client(sdk: NeMoPlatform) -> AsyncIterator[AsyncFilesClient]:
    """Provide an AsyncFilesClient backed by the SDK fixture's in-memory app."""
    sdk_http_client = sdk._client
    assert isinstance(sdk_http_client, SDKTestClientAdapter)
    base_url = str(sdk.base_url).rstrip("/")
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sdk_http_client.asgi_app),
        base_url=base_url,
        headers=dict(sdk_http_client.headers),
    )
    try:
        yield AsyncFilesClient(
            base_url=base_url,
            workspace=sdk.workspace,
            http_client=http_client,
        )
    finally:
        await http_client.aclose()


@pytest.fixture
def files_resource(sdk: NeMoPlatform) -> FilesResource:
    """Provide a FilesResource backed by the test FilesClient."""
    return sdk.files


@pytest.fixture
def sdk_allow_user_local_storage(tmp_path) -> Iterator[NeMoPlatform]:
    """SDK client with allow_user_local_storage enabled."""
    files_config = FilesConfig(
        default_storage_config=LocalStorageConfig(path=str(tmp_path / "default")),
        allow_user_local_storage=True,
    )
    with create_test_client(
        FilesService,
        SecretsService,
        service_configs={FilesService: files_config},
        tmp_dir=tmp_path,
        dependency_overrides=FILESET_AUTH_DEPENDENCY_OVERRIDES,
    ) as sdk:
        _install_asgi_files_resource(sdk)
        yield sdk


@pytest.fixture
def client(sdk: NeMoPlatform) -> httpx.Client:
    """TestClient extracted from SDK, sharing the same app context."""
    return sdk._client


@pytest.fixture
def hf_auth_headers() -> dict[str, str]:
    """Authorization headers for HF-compatible endpoints (service principal)."""
    return {"Authorization": "Bearer service:test"}


@pytest.fixture
def files_config() -> FilesConfig:
    return get_service_config(FilesConfig)


@pytest.fixture
def fileset(sdk: NeMoPlatform) -> Iterator[FilesetOutput]:
    with create_fileset(sdk) as fileset:
        yield fileset


@pytest.fixture
def fileset_cleanup(sdk: NeMoPlatform, files_client: FilesClient) -> Iterator[Callable[[str], None]]:
    """Fixture that provides a function to register filesets for cleanup.

    Usage:
        def test_something(sdk, fileset_cleanup):
            fileset_name = "my-test-fileset"
            fileset_cleanup(fileset_name)  # Register for cleanup
            # ... test code that creates the fileset ...
    """
    to_cleanup: list[tuple[str, str]] = []
    workspace = sdk.workspace or "default"

    def register(name: str, ws: str | None = None) -> None:
        to_cleanup.append((name, ws or workspace))

    yield register

    for name, ws in to_cleanup:
        try:
            files_client.delete_fileset(name=name, workspace=ws)
        except Exception:
            pass


@pytest.fixture
def cache_storage_impl(files_config: FilesConfig) -> StorageImpl:
    """Storage implementation for the cache (uses default storage config)."""
    return storage_impl_factory(files_config.default_storage_config, {})


# huggingface_hub v1.0+ uses httpx, while v0.x uses requests
# We need different approaches for each version
IS_HF_HUB_V1 = version.parse(huggingface_hub.__version__) >= version.parse("1.0.0")

if IS_HF_HUB_V1:
    # v1.0+ uses httpx with set_client_factory/close_session
    from huggingface_hub.utils import close_session, set_client_factory
else:
    # v0.x uses requests with configure_http_backend/reset_sessions
    import io

    import requests
    from huggingface_hub.utils import configure_http_backend, reset_sessions  # ty: ignore[unresolved-import]
    from requests.adapters import BaseAdapter
    from urllib3 import HTTPResponse as Urllib3Response

    class ASGIAdapter(BaseAdapter):
        """Requests adapter that forwards HTTP requests to an ASGI app via httpx.

        This adapter allows the requests-based huggingface_hub library (v0.x)
        to work with our ASGI test app without requiring a real HTTP server.
        """

        def __init__(self, httpx_client: httpx.Client):
            super().__init__()
            self.httpx_client = httpx_client

        def send(
            self,
            request: requests.PreparedRequest,
            stream: bool = False,
            timeout=None,
            verify: bool = True,
            cert=None,
            proxies=None,
        ) -> requests.Response:
            """Send a requests.PreparedRequest via the httpx client."""
            # Build httpx request
            httpx_response = self.httpx_client.request(
                method=request.method or "GET",
                url=request.url or "",
                headers=dict(request.headers) if request.headers else {},
                content=request.body,
            )

            # Convert httpx response to requests response
            response = requests.Response()
            response.status_code = httpx_response.status_code
            response.headers.update(httpx_response.headers)
            response.url = str(httpx_response.url)
            response.request = request
            response.encoding = httpx_response.encoding

            # Set up raw response for streaming support
            # requests.Response.iter_content() reads from response.raw
            content = httpx_response.content
            response._content = content
            response._content_consumed = True

            # Create a urllib3 HTTPResponse wrapper for streaming compatibility
            response.raw = Urllib3Response(
                body=io.BytesIO(content),
                headers=httpx_response.headers,
                status=httpx_response.status_code,
                preload_content=False,
            )

            return response

        def close(self) -> None:
            pass


class SharedASGIHttpxClient(httpx.Client):
    """httpx client for huggingface_hub that forwards through the test SDK client."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        super().__init__(
            base_url=str(client.base_url),
            headers=dict(client.headers),
        )

    def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: object = httpx.USE_CLIENT_DEFAULT,  # noqa: ARG002
        follow_redirects: object = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        if isinstance(follow_redirects, bool):
            return self._client.send(request, stream=stream, follow_redirects=follow_redirects)
        return self._client.send(request, stream=stream)

    def close(self) -> None:
        # huggingface_hub owns and closes its global client between tests. The
        # wrapped SDK client is owned by create_test_client and must remain open
        # for fixture cleanup.
        return None


def _default_hf_httpx_client_factory() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=None)


@pytest.fixture
def hf_asgi_client(client: httpx.Client) -> Iterator[None]:
    """Configure huggingface_hub to use ASGI transport for in-memory testing.

    This fixture injects a custom HTTP client that routes HuggingFace Hub
    requests through the test app's ASGI transport, eliminating the need
    for a real HTTP server.

    For huggingface_hub v1.0+ (httpx-based): We inject a custom httpx client
    that forwards through the SDK test client.

    For huggingface_hub v0.x (requests-based): We inject a custom requests
    Session with an adapter that forwards to the httpx test client.
    """
    if IS_HF_HUB_V1:
        # v1.0+: Use httpx client factory

        def asgi_client_factory() -> httpx.Client:
            return SharedASGIHttpxClient(client)

        set_client_factory(asgi_client_factory)
        yield
        close_session()
        set_client_factory(_default_hf_httpx_client_factory)
    else:
        # v0.x: Use requests adapter
        adapter = ASGIAdapter(client)
        base_url = str(client.base_url).rstrip("/")

        def backend_factory() -> requests.Session:
            session = requests.Session()
            # Mount the adapter for our test server URL
            session.mount(base_url, adapter)
            return session

        configure_http_backend(backend_factory)
        yield
        reset_sessions()  # Reset to default session factory
