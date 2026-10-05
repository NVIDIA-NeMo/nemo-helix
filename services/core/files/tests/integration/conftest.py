# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test fixtures for Files service.

This conftest provides fixtures for integration tests that require
external services (like Huggingface Hub).
"""

from collections.abc import Callable, Iterator

import httpx
import huggingface_hub
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from filesets.resources import FilesResource
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.config.base import get_service_config
from nhx.core.files.app.backends import storage_impl_factory
from nhx.core.files.app.backends.base import StorageImpl
from nhx.core.files.app.backends.local import LocalStorageConfig
from nhx.core.files.config import FilesConfig
from nhx.core.files.service import FilesService
from nhx.core.files.testing.utils import create_fileset
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, SDKTestClientAdapter, create_test_client
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
    """Resolve principal from X-NHX-Principal-Id header for tests that need multiple principals."""
    pid = request.headers.get("x-nhx-principal-id", "test@example.com")
    return AuthClient(
        principal=Principal(id=pid),
        config=_mock_auth_config,
    )


@pytest.fixture
def client_user_and_service() -> Iterator[tuple[NemoClient, NemoClient]]:
    """Two typed clients sharing the same app: default user principal and service:customizer.

    Yields (client_user, client_service). Use when testing service_source immutability
    with both principals against the same fileset.
    """
    with create_test_client(
        FilesService,
        SecretsService,
        client_type=TestClient,
        dependency_overrides={get_auth_client: _get_auth_client_from_request},
    ) as base_test_client:
        app = base_test_client.app
        base_url = "http://testserver"
        test_client_user = TestClient(
            app,
            base_url=base_url,
            headers={"x-nhx-principal-id": "test@example.com"},
        )
        test_client_service = TestClient(
            app,
            base_url=base_url,
            headers={"x-nhx-principal-id": "service:customizer"},
        )
        try:
            client_user = NemoClient(base_url=base_url, http_client=SDKTestClientAdapter(test_client_user))
            client_service = NemoClient(base_url=base_url, http_client=SDKTestClientAdapter(test_client_service))
            yield (client_user, client_service)
        finally:
            test_client_user.close()
            test_client_service.close()


@pytest.fixture
def client_context() -> Iterator[ClientContext]:
    """All test client flavors backed by one in-process Files + Secrets app."""
    with create_test_client(
        FilesService,
        SecretsService,
        client_type=ClientContext,
        dependency_overrides=FILESET_AUTH_DEPENDENCY_OVERRIDES,
    ) as ctx:
        yield ctx


@pytest.fixture
def client(client_context: ClientContext) -> NemoClient:
    """Sync typed platform client backed by the test app."""
    return client_context.client


@pytest.fixture
def files_client(client: NemoClient) -> FilesClient:
    """Provide a FilesClient derived from the platform client."""
    return FilesClient.from_client(client)


@pytest.fixture
def async_files_client(client_context: ClientContext) -> AsyncFilesClient:
    """Provide an AsyncFilesClient backed by the same in-memory app."""
    return AsyncFilesClient.from_client(client_context.async_client)


@pytest.fixture
def files_resource(client: NemoClient, files_client: FilesClient) -> FilesResource:
    """Provide a FilesResource backed by the test FilesClient."""
    return FilesResource(client, files_client=files_client)


@pytest.fixture
def client_allow_user_local_storage(tmp_path) -> Iterator[NemoClient]:
    """Typed client against a Files service with allow_user_local_storage enabled."""
    files_config = FilesConfig(
        default_storage_config=LocalStorageConfig(path=str(tmp_path / "default")),
        allow_user_local_storage=True,
    )
    with create_test_client(
        FilesService,
        SecretsService,
        client_type=NemoClient,
        service_configs={FilesService: files_config},
        tmp_dir=tmp_path,
        dependency_overrides=FILESET_AUTH_DEPENDENCY_OVERRIDES,
    ) as client:
        yield client


@pytest.fixture
def test_client(client_context: ClientContext) -> TestClient:
    """Raw TestClient sharing the same app context as the typed clients."""
    return client_context.test_client


@pytest.fixture
def hf_auth_headers() -> dict[str, str]:
    """Authorization headers for HF-compatible endpoints (service principal)."""
    return {"Authorization": "Bearer service:test"}


@pytest.fixture
def files_config() -> FilesConfig:
    return get_service_config(FilesConfig)


@pytest.fixture
def fileset(files_client: FilesClient) -> Iterator[FilesetOutput]:
    with create_fileset(files_client) as fileset:
        yield fileset


@pytest.fixture
def fileset_cleanup(client: NemoClient, files_client: FilesClient) -> Iterator[Callable[[str], None]]:
    """Fixture that provides a function to register filesets for cleanup.

    Usage:
        def test_something(client, fileset_cleanup):
            fileset_name = "my-test-fileset"
            fileset_cleanup(fileset_name)  # Register for cleanup
            # ... test code that creates the fileset ...
    """
    to_cleanup: list[tuple[str, str]] = []
    workspace = client.workspace or "default"

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
    """httpx client for huggingface_hub that forwards through the test client."""

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
        # wrapped test client is owned by create_test_client and must remain open
        # for fixture cleanup.
        return None


def _default_hf_httpx_client_factory() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=None)


@pytest.fixture
def hf_asgi_client(test_client: TestClient) -> Iterator[None]:
    """Configure huggingface_hub to use ASGI transport for in-memory testing.

    This fixture injects a custom HTTP client that routes HuggingFace Hub
    requests through the test app's ASGI transport, eliminating the need
    for a real HTTP server.

    For huggingface_hub v1.0+ (httpx-based): We inject a custom httpx client
    that forwards through the SDK test client.

    For huggingface_hub v0.x (requests-based): We inject a custom requests
    Session with an adapter that forwards to the httpx test client.
    """
    forwarder = SDKTestClientAdapter(test_client)
    if IS_HF_HUB_V1:
        # v1.0+: Use httpx client factory

        def asgi_client_factory() -> httpx.Client:
            return SharedASGIHttpxClient(forwarder)

        set_client_factory(asgi_client_factory)
        yield
        close_session()
        set_client_factory(_default_hf_httpx_client_factory)
    else:
        # v0.x: Use requests adapter
        adapter = ASGIAdapter(forwarder)
        base_url = str(forwarder.base_url).rstrip("/")

        def backend_factory() -> requests.Session:
            session = requests.Session()
            # Mount the adapter for our test server URL
            session.mount(base_url, adapter)
            return session

        configure_http_backend(backend_factory)
        yield
        reset_sessions()  # Reset to default session factory
