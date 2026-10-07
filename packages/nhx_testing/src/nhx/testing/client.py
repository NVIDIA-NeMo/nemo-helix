# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test client utilities for NeMo Helix services."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Coroutine, Generator, Mapping, Protocol, TypeVar

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.projects.client import ProjectsClient
from nemo_helix_plugin.projects.types import CreateProjectRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common import platform_client_context
from nhx.common.auth import Principal
from nhx.common.config.base import AuthConfig, Configuration, DatabaseConfig, HelixConfig, ServiceConfig
from nhx.common.entities.client import EntityClient
from nhx.common.service import Service
from nhx.common.service.dependencies import (
    get_entity_client,
    get_nemo_client,
    get_sync_nemo_client,
)
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.service import EntitiesService
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.platform_runner.loader import order_services_by_dependencies
from nhx.platform_runner.server import create_app
from nhx.testing.access_log import AccessLog, AccessLogMiddleware
from nhx.testing.asyncio_debug import bounded_event_loop_teardown
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)
_IN_PROCESS_PLATFORM_BASE_URL = "http://127.0.0.1"


@dataclass
class ClientContext:
    """Container for all test client types created by create_test_client.

    This allows tests to access any client type without needing to specify
    client_type upfront, which is useful when a test needs multiple client types.
    """

    client: NemoClient
    """Synchronous typed platform client routed through the in-process TestClient.

    Derive service clients with ``FilesClient.from_client(ctx.client)``.
    """

    async_client: AsyncNemoClient
    """Asynchronous typed platform client routed through the in-process ASGI transport.

    Derive service clients with ``AsyncFilesClient.from_client(ctx.async_client)``.
    """

    entity_client: EntityClient
    """EntityClient for direct entity operations."""

    test_client: TestClient
    """FastAPI TestClient for raw HTTP requests."""

    access_log: AccessLog | None = None
    """Captured requests when access_log=True was passed to create_test_client."""


ClientT = TypeVar("ClientT", TestClient, NemoClient, AsyncNemoClient, EntityClient, ClientContext)


class TestClientHttpAdapter(httpx.Client):
    """Expose Starlette's TestClient through the ``httpx.Client`` interface.

    Pass it as ``http_client`` to a sync :class:`NemoClient` (or any typed
    service client) to route requests through the in-process app::

        client = FilesClient(base_url=str(test_client.base_url), http_client=TestClientHttpAdapter(test_client))

    or use :func:`nemo_client_for_test_client`.
    """

    __test__ = False

    def __init__(self, test_client: TestClient) -> None:
        self._test_client = test_client
        super().__init__(base_url=str(test_client.base_url), headers=dict(test_client.headers))

    @property
    def asgi_app(self) -> ASGIApp:
        """Return the wrapped ASGI app for tests that need to create sibling clients."""
        return self._test_client.app

    def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,  # noqa: ARG002 - the in-process test response is materialized eagerly.
        auth: object = httpx.USE_CLIENT_DEFAULT,  # noqa: ARG002
        follow_redirects: object = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        follow_redirects_enabled = follow_redirects if isinstance(follow_redirects, bool) else False
        response = self._test_client.request(
            request.method,
            str(request.url),
            content=request.read(),
            headers=dict(request.headers),
            follow_redirects=follow_redirects_enabled,
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
            request=request,
        )


SyncClientT = TypeVar("SyncClientT", bound=NemoClient)
AsyncClientT = TypeVar("AsyncClientT", bound=AsyncNemoClient)

MockHandler = Callable[[httpx.Request], httpx.Response]
AsyncMockHandler = MockHandler | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]

_MOCK_BASE_URL = "http://nemo.test"


def nemo_client_for_test_client(
    test_client: TestClient,
    client_cls: type[SyncClientT],
    *,
    workspace: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> SyncClientT:
    """Build a sync typed client that sends requests through *test_client*.

    Args:
        test_client: The in-process FastAPI ``TestClient``. Its default headers
            (e.g. principal headers) are sent with every request.
        client_cls: Typed client class to build, e.g. ``FilesClient`` or ``NemoClient``.
        workspace: Default workspace used to fill ``{workspace}`` path params.
        headers: Extra default headers for the client.
    """
    return client_cls(
        base_url=str(test_client.base_url),
        workspace=workspace,
        default_headers=headers,
        http_client=TestClientHttpAdapter(test_client),
        owns_http_client=True,
    )


def async_nemo_client_for_app(
    app: FastAPI,
    client_cls: type[AsyncClientT],
    *,
    base_url: str = _IN_PROCESS_PLATFORM_BASE_URL,
    workspace: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> AsyncClientT:
    """Build an async typed client that sends requests to *app* over an ASGI transport.

    The client owns its HTTP transport; close it with ``await client.close()``
    or ``async with``. The app's lifespan is not run.
    """
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)
    return client_cls(
        base_url=base_url,
        workspace=workspace,
        default_headers=headers,
        http_client=http_client,
        owns_http_client=True,
    )


def mock_nemo_client(
    handler: MockHandler,
    client_cls: type[SyncClientT],
    *,
    base_url: str = _MOCK_BASE_URL,
    workspace: str | None = "default",
    headers: Mapping[str, str] | None = None,
) -> SyncClientT:
    """Build a sync typed client whose requests are answered by *handler*.

    *handler* receives each ``httpx.Request`` and returns an ``httpx.Response``;
    record requests in a list to assert on method, URL, headers, and body::

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={...})

        files = mock_nemo_client(handler, FilesClient)
    """
    return client_cls(
        base_url=base_url,
        workspace=workspace,
        default_headers=headers,
        http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url),
        owns_http_client=True,
    )


def mock_async_nemo_client(
    handler: AsyncMockHandler,
    client_cls: type[AsyncClientT],
    *,
    base_url: str = _MOCK_BASE_URL,
    workspace: str | None = "default",
    headers: Mapping[str, str] | None = None,
) -> AsyncClientT:
    """Async counterpart of :func:`mock_nemo_client`. *handler* may be sync or async."""
    return client_cls(
        base_url=base_url,
        workspace=workspace,
        default_headers=headers,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url),
        owns_http_client=True,
    )


def _current_request_headers() -> dict[str, str]:
    headers = platform_client_context.forwardable_otel_headers(include_internal=True)
    headers.update(platform_client_context.current_principal_auth_headers())
    return headers


def request_scoped_nemo_client_overrides(
    async_client: AsyncNemoClient,
    sync_client: NemoClient | None = None,
) -> dict[Callable, Callable]:
    """FastAPI dependency overrides that hand out request-scoped typed clients.

    The returned overrides bind ``get_nemo_client`` (and ``get_sync_nemo_client``
    when *sync_client* is given) to copies of the supplied clients carrying the
    current request's principal and trace headers, the same way
    ``DependencyProvider`` scopes clients in production. Use it for a bare
    ``FastAPI`` app under test::

        app.dependency_overrides.update(request_scoped_nemo_client_overrides(async_client))
    """

    def _request_scoped_async_client() -> AsyncNemoClient:
        headers = _current_request_headers()
        return async_client.with_headers(headers) if headers else async_client

    overrides: dict[Callable, Callable] = {get_nemo_client: _request_scoped_async_client}
    if sync_client is not None:

        def _request_scoped_sync_client() -> NemoClient:
            headers = _current_request_headers()
            return sync_client.with_headers(headers) if headers else sync_client

        overrides[get_sync_nemo_client] = _request_scoped_sync_client
    return overrides


# Default test user for auth-enabled tests
TEST_USER_EMAIL = "user@example.com"
# Admin user email for auth-enabled tests (has elevated permissions)
TEST_ADMIN_EMAIL = "admin@example.com"


def _default_service_configs(tmp_dir: Path) -> dict[type[object], ServiceConfig]:
    """Create default service configs for testing.

    Args:
        tmp_dir: Temp directory for services that need file storage.

    Returns:
        Map of service class → config with testing defaults.
    """
    # Import here to avoid circular imports
    from nhx.common.secrets.encryption import get_base64_encoded_random_bytes
    from nhx.core.files.app.backends.local import LocalStorageConfig
    from nhx.core.files.config import FilesConfig
    from nhx.core.files.service import FilesService
    from nhx.core.secrets.config import SecretsServiceConfig
    from nhx.core.secrets.service import SecretsService

    test_db_file = tmp_dir / f"test_{uuid.uuid4().hex}.db"
    db_url = f"sqlite:///{test_db_file}"

    return {
        EntitiesService: EntitiesConfig(database_config=DatabaseConfig(url=db_url)),
        FilesService: FilesConfig(default_storage_config=LocalStorageConfig(path=str(tmp_dir))),
        SecretsService: SecretsServiceConfig(
            encryption={
                "current_provider": "test",
                "providers": {
                    "secret_key": {
                        "test": {"value": get_base64_encoded_random_bytes(32)},
                    },
                },
            },
        ),
        HelixConfig: HelixConfig(base_url=_IN_PROCESS_PLATFORM_BASE_URL),
    }


class ServiceFactory(Protocol):
    """Protocol for service classes that accept an optional dependency provider.

    Concrete service subclasses (like HelloWorldService) implement their
    own __init__ that calls super().__init__(name=..., module_name=...).
    """

    def __call__(self) -> Service: ...


def _create_svc(
    service_type: ServiceFactory,
    service_configs: dict[type[object], ServiceConfig],
) -> Service:
    """Instantiate a service with optional config injection.

    Args:
        service_type: The service class to instantiate
        service_configs: Map of service class → config.

    Returns:
        The instantiated service, with config applied if one exists.
    """
    svc = service_type()

    # Apply config if one exists for this service type
    if type(svc) in service_configs:
        svc = svc.with_config(service_configs[type(svc)])

    return svc


_DEFAULT_WORKSPACES = ["default"]
_DEFAULT_PROJECTS = ["default/test-project"]


@contextmanager
def create_test_client(
    *service_types: ServiceFactory,
    client_type: type[ClientT] | None = None,
    dependency_overrides: dict[Callable, Callable] | None = None,
    service_configs: dict[type[object], ServiceConfig] | None = None,
    tmp_dir: Path | None = None,
    workspaces: list[str] | None = None,
    workspace: str | None = None,
    projects: list[str] | None = None,
    auth_enabled: bool = False,
    auth_bundle_cache_seconds: float = 0.1,
    auth_policy_data_refresh_interval: float = 0.1,
    access_log: bool = False,
    igw_mock_provider_mode: bool = True,
) -> Generator[ClientT, None, None]:
    """Create a test client for NeMo Helix services with optional dependency overrides.

    Pass one or more service types as positional arguments.

    Args:
        *service_types: One or more Service classes to test
        client_type: The client type to yield. One of TestClient, NemoClient, AsyncNemoClient,
                     EntityClient, or ClientContext. Defaults to NemoClient.
        dependency_overrides: Custom dependency overrides dict. If get_entity_client
                      is not in dependency_overrides, an EntityClient will be created.
        service_configs: Optional map of service class → config. Overrides defaults
                      (e.g., EntitiesService uses memory backend, FilesService uses tmp_dir).
        tmp_dir: Optional temp directory path.
        workspaces: List of workspace names to create. Defaults to ["default"].
                   Pass an empty list to skip workspace creation.
        workspace: Optional default workspace set on every yielded platform client.
        projects: List of projects to create in format "workspace/project_name".
                 Defaults to ["default/test-project"]. If a workspace is referenced
                 but not in the workspaces list, it will be added automatically.
                 Pass an empty list to skip project creation.
        auth_enabled: Enable authorization middleware with embedded PDP. When True:
                     - Adds AuthService to the services list (if not already present)
                     - Configures AuthConfig and AuthServiceConfig with auth enabled
                     - Creates workspaces/projects using TEST_USER_EMAIL as principal
                     Default: False.
        auth_bundle_cache_seconds: When ``auth_enabled`` is True, sets
                     ``AuthServiceConfig.bundle_cache_seconds``. Values ``> 0`` avoid reloading
                     policy from entities on every PDP evaluation (much faster integration
                     tests). Use ``0`` only when a test needs instant propagation of role
                     bindings to the embedded PDP without waiting for the background refresh
                     loop (see ``auth_policy_data_refresh_interval``). Ignored if you pass
                     your own ``AuthServiceConfig`` in ``service_configs``.
                     Default: 2.
        auth_policy_data_refresh_interval: When ``auth_enabled`` is True, sets
                     ``AuthServiceConfig.policy_data_refresh_interval`` (seconds between
                     background policy data refreshes). Should be short when
                     ``auth_bundle_cache_seconds`` is non-zero so binding changes propagate
                     quickly. Ignored if you pass your own ``AuthServiceConfig`` in
                     ``service_configs``. Default: 2.
        access_log: Enable request capture for test verification. When True, all
                   HTTP requests processed by the app are captured and can be
                   inspected via ClientContext.access_log or app.state.access_log.
                   Default: False.
        igw_mock_provider_mode: Enable mock provider mode for InferenceGatewayService.
                   When True, sets mock_provider_prefix="igw-mock-" so providers whose
                   names start with that prefix return mock responses instead of proxying
                   to real backends. Use add_mock_provider() from utils to add providers.
                   Default: True.

    Example (typed client):
        with create_test_client(SecretsService, client_type=NemoClient) as client:
            secrets = SecretsClient.from_client(client)
            secrets.create_secret(workspace="default", body=CreateSecretRequest(name="test", value="value"))

    Example (TestClient):
        with create_test_client(FilesService, client_type=TestClient) as client:
            response = client.get("/v1/files")

    Example (EntityClient for service-level tests):
        with create_test_client(client_type=EntityClient) as entity_client:
            await entity_client.create(my_entity)

    Example (with extra workspaces):
        with create_test_client(client_type=EntityClient, workspaces=["default", "ns1", "ns2"]) as client:
            # "default", "ns1", and "ns2" workspaces are available
            await client.create(entity_in_ns1)

    Example (with custom projects):
        with create_test_client(projects=["default/my-project", "ns1/other-project"], client_type=NemoClient) as client:
            # "my-project" in "default" and "other-project" in "ns1" are available
            # "ns1" workspace is auto-created since it's referenced in projects
            ModelsClient.from_client(client).create_model(workspace="default", body=...)

    Example (with auth enabled):
        with create_test_client(FilesService, auth_enabled=True, client_type=NemoClient) as client:
            # Authorization middleware is active
            # Requests without auth headers will be rejected
            ...

    Example (ClientContext for multiple client types):
        with create_test_client(FilesService, client_type=ClientContext) as ctx:
            # Access any client type
            FilesClient.from_client(ctx.client).list_filesets(workspace="default")  # sync typed
            await AsyncFilesClient.from_client(ctx.async_client).list_filesets(workspace="default")  # async typed
            ctx.test_client.get("/health")  # raw HTTP

    Example (with access_log for request verification):
        with create_test_client(
            FilesService, auth_enabled=True, access_log=True, client_type=ClientContext
        ) as ctx:
            ctx.access_log.clear()  # Clear requests from setup
            ctx.test_client.get(
                "/apis/files/v2/workspaces/default/filesets",
                headers={"X-NHX-Principal-Id": "test@example.com"},
            )
            # Verify internal entity requests used the same principal
            entity_requests = ctx.access_log.filter(path_contains="/entities/")
            for req in entity_requests:
                assert req.principal_id == "test@example.com"
    """
    selected_client_type: type[object] = client_type or NemoClient
    with ExitStack() as stack:
        # Create temp directory if not provided
        # Use ignore_cleanup_errors=True because fire-and-forget background tasks
        # may still hold SQLite file locks when cleanup runs
        if tmp_dir is None:
            tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True)))

        # Merge defaults (generated from tmp_dir) with user overrides
        configs = _default_service_configs(tmp_dir)
        if service_configs:
            configs.update(service_configs)
        platform_config = configs.get(HelixConfig)
        if not isinstance(platform_config, HelixConfig):
            raise TypeError("create_test_client requires a HelixConfig platform config")
        platform_base_url = platform_config.base_url

        # If auth is enabled, set up auth configs and add AuthService
        if auth_enabled:
            from nhx.common.config.base import AuthConfig as SharedAuthConfig
            from nhx.core.auth.config import AuthServiceConfig
            from nhx.core.auth.service import AuthService

            # Only add auth configs if not already provided by user.
            # PDP base is the platform root; get_pdp_url() appends /apis/auth/v2/authz/{entrypoint}.
            pdp_base = platform_base_url
            if SharedAuthConfig not in configs:
                configs[SharedAuthConfig] = SharedAuthConfig(
                    enabled=True,
                    policy_decision_point_provider="embedded",
                    policy_decision_point_base_url=pdp_base,
                    propagation_poll_interval_seconds=0.05,
                )
            if AuthServiceConfig not in configs:
                configs[AuthServiceConfig] = AuthServiceConfig(
                    enabled=True,
                    policy_decision_point_provider="embedded",
                    policy_decision_point_base_url=pdp_base,
                    policy_data_refresh_interval=auth_policy_data_refresh_interval,
                    bundle_cache_seconds=auth_bundle_cache_seconds,
                    admin_email=TEST_ADMIN_EMAIL,
                    allowed_service_principals=["integration-test"],
                )

        # If IGW mock provider mode is enabled, configure the prefix and register cleanup
        if igw_mock_provider_mode:
            from nhx.core.inference_gateway.api.dependencies import reset_global_model_cache
            from nhx.core.inference_gateway.api.mock_provider import reset_call_counts

            # Get existing config or create new one
            existing_igw_config = configs.get(InferenceGatewayService)
            if existing_igw_config is not None:
                # Merge with existing config by creating new instance with mock_provider_prefix
                configs[InferenceGatewayService] = InferenceGatewayConfig(
                    **{
                        **existing_igw_config.model_dump(),
                        "mock_provider_prefix": "igw-mock-",
                    }
                )
            else:
                configs[InferenceGatewayService] = InferenceGatewayConfig(
                    mock_provider_prefix="igw-mock-",
                )

            # Register cleanup for the global model cache
            stack.callback(reset_global_model_cache)

        # Set Configuration overrides so Configuration.get_service_config() returns
        # our test configs. Convert from Service type -> config to config type -> config.
        config_overrides = {type(cfg): cfg for cfg in configs.values()}
        Configuration.set_overrides(config_overrides)
        stack.callback(Configuration.clear_overrides)

        def _add_service(
            ordered_services: list[ServiceFactory],
            service: ServiceFactory,
        ) -> None:
            if service not in ordered_services:
                ordered_services.append(service)

        # Almost all Services depend on entities service. Might be nice
        # to have services define their deps in the future so we don't have to
        # hard code this.
        services_to_create: list[ServiceFactory] = []
        _add_service(services_to_create, EntitiesService)
        for service_type in service_types:
            _add_service(services_to_create, service_type)

        # Add AuthService if auth is enabled
        if auth_enabled:
            from nhx.core.auth.service import AuthService

            _add_service(services_to_create, AuthService)

        # Instantiate services with their configs, then order by dependencies so
        # roots (e.g. entities) start before dependents (e.g. guardrails).
        services_to_start = [_create_svc(svc, configs) for svc in services_to_create]
        services_to_start = order_services_by_dependencies(services_to_start)

        # Create transport and http_client BEFORE the app, so we can inject the client
        # into create_app() for middleware (AuthorizationMiddleware). We set transport.app
        # after app creation - this works because no requests are made until setup completes.
        async def _pending_asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
            raise RuntimeError("ASGI app was not initialized before test client request")

        transport = httpx.ASGITransport(app=_pending_asgi_app)
        pdp_timeout = Configuration.get_service_config(AuthConfig).policy_decision_point_request_timeout_seconds
        async_http_client = httpx.AsyncClient(transport=transport, base_url=platform_base_url, timeout=pdp_timeout)

        # Both auth callouts target this in-process ASGI app in tests.
        app = create_app(
            services_to_start,
            http_client=async_http_client,
            access_key_lifecycle_http_client=async_http_client,
        )
        transport.app = app

        # Clean up FastAPI app state that tracks model call counts.
        # This ensures call counts are reset between tests.
        if igw_mock_provider_mode:
            stack.callback(lambda: reset_call_counts(app.state))

        # Set up access log if enabled (must be done before any requests)
        access_log_instance: AccessLog | None = None
        if access_log:
            access_log_instance = AccessLog()
            app.add_middleware(AccessLogMiddleware, access_log=access_log_instance)
            # Store on app.state so tests can access it via test_client.app.state.access_log
            app.state.access_log = access_log_instance

        async_client = AsyncNemoClient(base_url=platform_base_url, http_client=async_http_client, workspace=workspace)

        # Create the EntityClient (used for DI and optionally yielded)
        entity_client = EntityClient(AsyncEntitiesClient.from_client(async_client))

        # Inject ASGI-transport clients into each service's DependencyProvider so
        # every typed client it builds (request-scoped, service-principal, entity)
        # routes through the in-process app, including clients created during
        # on_startup for background tasks like auth policy refresh.
        # See architecture/docs/http-client-injection.md for details.
        for svc in services_to_start:
            svc.dependency_provider._http_client = async_http_client

        # Merge dependency overrides
        all_overrides = {}
        if dependency_overrides:
            all_overrides.update(dependency_overrides)

        # Override get_entity_client to create a fresh EntityClient using
        # build_downstream_service_headers, matching DependencyProvider.get_entity_client()
        # in production. The current service is taken from AppContext (set by
        # create_app_context_dependency on each sub-router) so merged multi-service
        # apps get service:<name> and on-behalf-of the same as real deployments;
        # routes without that context (e.g. /health) fall back to "platform".
        if get_entity_client not in all_overrides:
            from nhx.common.observability.context import get_app_ctx
            from nhx.common.service.headers import build_downstream_service_headers

            def _get_entity_client_on_behalf_of() -> EntityClient:
                app_ctx = get_app_ctx()
                service_name = app_ctx.service_name if app_ctx is not None and app_ctx.service_name else "platform"
                headers = build_downstream_service_headers(service_name)
                return EntityClient(AsyncEntitiesClient.from_client(async_client.with_headers(headers)))

            all_overrides[get_entity_client] = _get_entity_client_on_behalf_of

        if all_overrides:
            app.dependency_overrides.update(all_overrides)

        with bounded_event_loop_teardown(), TestClient(app, base_url=platform_base_url) as client:
            sync_http_client = TestClientHttpAdapter(client)
            sync_client = NemoClient(base_url=platform_base_url, http_client=sync_http_client, workspace=workspace)
            for svc in services_to_start:
                svc.dependency_provider._sync_http_client = sync_http_client

            # Trigger middleware stack build with a health check (which skips auth).
            client.get("/health")

            # Wait for all services to become ready. Readiness is determined by each
            # service's is_ready() (e.g. entities uses DB health, auth uses policy refresh).
            # Poll /health/ready until 200 or timeout.
            startup_timeout = 30.0
            startup_start = time.time()
            while time.time() - startup_start < startup_timeout:
                response = client.get("/health/ready")
                if response.status_code == 200:
                    break
                time.sleep(0.1)
            else:
                # Include /status in timeout error for debugging
                status_text = ""
                try:
                    sr = client.get("/status")
                    if sr.status_code == 200:
                        status_text = f" /status: {sr.text}"
                except Exception:
                    pass
                raise TimeoutError(
                    f"Services not ready after {startup_timeout}s. "
                    f"Health response: {response.status_code} - {response.text}.{status_text}"
                )

            # When auth is enabled, seed role bindings (platform admin, wildcard default/system)
            # so tests can assume they exist (normally created by the platform-seed job).
            if auth_enabled:

                def _seed_auth_role_bindings() -> None:
                    import asyncio

                    from nhx.core.auth.app.embedded_pdp import load_policy_data
                    from nhx.core.auth.app.seeding import run_seeding

                    # Use service principal so entity store accepts role binding creation
                    seeding_client = async_client.with_headers(
                        Principal(id="service:auth", authz_aliases=["service:auth"]).get_headers()
                    )
                    seeding_entity_client = EntityClient(AsyncEntitiesClient.from_client(seeding_client))

                    async def _run() -> None:
                        success = await run_seeding(seeding_entity_client)
                        if not success:
                            raise RuntimeError("Auth role binding seeding failed in test setup")
                        # With bundle_cache_seconds > 0, PDP eval does not reload policy each time.
                        # Push seeded role bindings into WASM immediately so tests see admin/etc.
                        await load_policy_data(seeding_entity_client)

                    asyncio.run(_run())

                _seed_auth_role_bindings()

            # Parse projects (format: "workspace/project_name")
            projects_to_create = projects if projects is not None else _DEFAULT_PROJECTS
            parsed_projects: list[tuple[str, str]] = []
            for proj in projects_to_create:
                ws_id, proj_name = proj.split("/", 1)
                parsed_projects.append((ws_id, proj_name))

            # Collect workspaces: explicit list + any referenced by projects
            workspaces_to_create = set(workspaces if workspaces is not None else _DEFAULT_WORKSPACES)
            workspaces_to_create.update(ws_id for ws_id, _ in parsed_projects)

            # Create workspaces (with auth headers if auth is enabled)
            if auth_enabled:
                auth_headers = {"X-NHX-Principal-Id": TEST_USER_EMAIL}
                for ws_id in workspaces_to_create:
                    # Skip role propagation - policy data may not be loaded yet
                    client.post(
                        "/apis/entities/v2/workspaces?wait_role_propagation=false",
                        json={"name": ws_id},
                        headers=auth_headers,
                    )
                for ws_id, proj_name in parsed_projects:
                    client.post(
                        f"/apis/entities/v2/workspaces/{ws_id}/projects",
                        json={"name": proj_name},
                        headers=auth_headers,
                    )
            else:
                workspaces_client = WorkspacesClient.from_client(sync_client)
                for ws_id in workspaces_to_create:
                    try:
                        workspaces_client.create_workspace(body=CreateWorkspaceRequest(name=ws_id))
                    except ConflictError:
                        logger.warning(f"Workspace '{ws_id}' already exists (created by service startup)")

                projects_client = ProjectsClient.from_client(sync_client)
                for ws_id, proj_name in parsed_projects:
                    try:
                        projects_client.create_project(workspace=ws_id, body=CreateProjectRequest(name=proj_name))
                    except ConflictError:
                        logger.warning(f"Project '{proj_name}' in workspace '{ws_id}' already exists")

            if selected_client_type is TestClient:
                yield client  # ty: ignore[invalid-yield]
            elif selected_client_type is NemoClient:
                yield sync_client  # ty: ignore[invalid-yield]
            elif selected_client_type is AsyncNemoClient:
                yield async_client  # ty: ignore[invalid-yield]
            elif selected_client_type is EntityClient:
                yield entity_client  # ty: ignore[invalid-yield]
            elif selected_client_type is ClientContext:
                yield ClientContext(
                    client=sync_client,
                    async_client=async_client,
                    entity_client=entity_client,
                    test_client=client,
                    access_log=access_log_instance,
                )
            else:
                raise TypeError(f"Unsupported client_type: {selected_client_type!r}")

        app.dependency_overrides.clear()
