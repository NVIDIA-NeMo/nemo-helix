# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test client utilities for NeMo Platform services."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Mapping, Protocol, TypeVar

import httpx
from fastapi.testclient import TestClient
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform, NotGiven, not_given
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.common.config.base import AuthConfig, Configuration, DatabaseConfig, PlatformConfig, ServiceConfig
from nmp.common.entities.client import EntityClient
from nmp.common.service import Service
from nmp.common.service.dependencies import get_entity_client, get_sdk_client, get_sync_sdk_client
from nmp.core.entities.config import EntitiesConfig
from nmp.core.entities.service import EntitiesService
from nmp.core.inference_gateway.config import InferenceGatewayConfig
from nmp.core.inference_gateway.service import InferenceGatewayService
from nmp.platform_runner.loader import order_services_by_dependencies
from nmp.platform_runner.server import create_app
from nmp.testing.access_log import AccessLog, AccessLogMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


@dataclass
class ClientContext:
    """Container for all test client types created by create_test_client.

    This allows tests to access any client type without needing to specify
    client_type upfront, which is useful when a test needs multiple client types.
    """

    sdk: NeMoPlatform
    """Synchronous NeMoPlatform SDK client."""

    async_sdk: AsyncNeMoPlatform
    """Asynchronous NeMoPlatform SDK client."""

    async_client: AsyncNemoClient
    """Asynchronous typed platform client."""

    entity_client: EntityClient
    """EntityClient for direct entity operations."""

    test_client: TestClient
    """FastAPI TestClient for raw HTTP requests."""

    access_log: AccessLog | None = None
    """Captured requests when access_log=True was passed to create_test_client."""


ClientT = TypeVar("ClientT", TestClient, AsyncNemoClient, AsyncNeMoPlatform, NeMoPlatform, EntityClient, ClientContext)


class SDKTestClientAdapter(httpx.Client):
    """Expose Starlette's TestClient through the httpx.Client interface expected by the SDK."""

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
    from nmp.common.secrets.encryption import get_base64_encoded_random_bytes
    from nmp.core.files.app.backends.local import LocalStorageConfig
    from nmp.core.files.config import FilesConfig
    from nmp.core.files.service import FilesService
    from nmp.core.secrets.config import SecretsServiceConfig
    from nmp.core.secrets.service import SecretsService

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
        # PlatformConfig with testserver URL so get_service_url() works in tests
        PlatformConfig: PlatformConfig(base_url="http://testserver"),
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


def _install_asgi_files_resource(sdk: NeMoPlatform) -> None:
    """Route sync SDK file uploads through the in-process test app."""
    from filesets.resources import FilesResource
    from nemo_platform_plugin.client.adapter import client_from_platform
    from nemo_platform_plugin.files.client import FilesClient

    files_client = client_from_platform(sdk, FilesClient)
    sdk.__dict__["files"] = FilesResource(
        sdk,
        files_client=files_client,
    )
    original_copy: Callable[..., NeMoPlatform] = sdk.copy

    def copy_with_asgi_files(
        *,
        workspace: str | None = None,
        base_url: str | httpx.URL | None = None,
        inference_base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
        http_client: httpx.Client | None = None,
        max_retries: int | NotGiven = not_given,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, object] | None = None,
    ) -> NeMoPlatform:
        clone = original_copy(
            workspace=workspace,
            base_url=base_url,
            inference_base_url=inference_base_url,
            timeout=timeout,
            http_client=http_client,
            max_retries=max_retries,
            default_headers=default_headers,
            set_default_headers=set_default_headers,
            default_query=default_query,
            set_default_query=set_default_query,
            _extra_kwargs={} if _extra_kwargs is None else _extra_kwargs,
        )
        _install_asgi_files_resource(clone)
        return clone

    sdk.__dict__["copy"] = copy_with_asgi_files
    sdk.__dict__["with_options"] = copy_with_asgi_files


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
    """Create a test client for NeMo Platform services with optional dependency overrides.

    Pass one or more service types as positional arguments.

    Args:
        *service_types: One or more Service classes to test
        client_type: The client type to yield. One of TestClient, AsyncNemoClient,
                     AsyncNeMoPlatform, NeMoPlatform, or EntityClient. Defaults to NeMoPlatform.
        dependency_overrides: Custom dependency overrides dict. If get_entity_client
                      is not in dependency_overrides, an EntityClient will be created.
        service_configs: Optional map of service class → config. Overrides defaults
                      (e.g., EntitiesService uses memory backend, FilesService uses tmp_dir).
        tmp_dir: Optional temp directory path.
        workspaces: List of workspace names to create. Defaults to ["default"].
                   Pass an empty list to skip workspace creation.
        workspace: Optional default workspace to set on the NeMoPlatform client.
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

    Example (single service):
        with create_test_client(FilesService) as sdk:
            secret = sdk.secrets.create(workspace="default", name="test", value="value")

    Example (multi-service):
        with create_test_client(FilesService, EntitiesService) as sdk:
            sdk.entities.create(...)

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
        with create_test_client(projects=["default/my-project", "ns1/other-project"]) as sdk:
            # "my-project" in "default" and "other-project" in "ns1" are available
            # "ns1" workspace is auto-created since it's referenced in projects
            sdk.models.create(workspace="default", project="my-project", ...)

    Example (with auth enabled):
        with create_test_client(FilesService, auth_enabled=True) as sdk:
            # Authorization middleware is active
            # Requests without auth headers will be rejected
            ...

    Example (ClientContext for multiple client types):
        with create_test_client(FilesService, client_type=ClientContext) as ctx:
            # Access any client type
            ctx.sdk.files.list(workspace="default")  # sync SDK
            await ctx.async_sdk.files.list(workspace="default")  # async SDK
            ctx.test_client.get("/health")  # raw HTTP

    Example (with access_log for request verification):
        with create_test_client(
            FilesService, auth_enabled=True, access_log=True, client_type=ClientContext
        ) as ctx:
            ctx.access_log.clear()  # Clear requests from setup
            ctx.test_client.get(
                "/apis/files/v2/workspaces/default/filesets",
                headers={"X-NMP-Principal-Id": "test@example.com"},
            )
            # Verify internal entity requests used the same principal
            entity_requests = ctx.access_log.filter(path_contains="/entities/")
            for req in entity_requests:
                assert req.principal_id == "test@example.com"
    """
    selected_client_type: type[object] = client_type or NeMoPlatform
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

        # If auth is enabled, set up auth configs and add AuthService
        if auth_enabled:
            from nmp.common.config.base import AuthConfig as SharedAuthConfig
            from nmp.core.auth.config import AuthServiceConfig
            from nmp.core.auth.service import AuthService

            # Only add auth configs if not already provided by user.
            # PDP base is the platform root; get_pdp_url() appends /apis/auth/v2/authz/{entrypoint}.
            pdp_base = "http://testserver"
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
                )

        # If IGW mock provider mode is enabled, configure the prefix and register cleanup
        if igw_mock_provider_mode:
            from nmp.core.inference_gateway.api.dependencies import reset_global_model_cache
            from nmp.core.inference_gateway.api.mock_provider import reset_call_counts

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
            from nmp.core.auth.service import AuthService

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
        async_http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=pdp_timeout)

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

        from nmp.common.sdk_factory import get_async_platform_sdk

        async_sdk = get_async_platform_sdk(
            base_url="http://testserver",
            http_client=async_http_client,
        ).copy(workspace=workspace)
        async_client = AsyncNemoClient(base_url="http://testserver", http_client=async_http_client, workspace=workspace)

        # Create the EntityClient (used for DI and optionally yielded)
        entity_client = EntityClient(client_from_platform(async_sdk, AsyncEntitiesClient))

        # Inject ASGI-transport clients into each service's DependencyProvider.
        # This is critical for services that call dependency_provider.get_sdk_client()
        # directly (e.g., in on_startup for background tasks like auth policy refresh).
        # See architecture/docs/http-client-injection.md for details.
        for svc in services_to_start:
            svc.dependency_provider._http_client = async_http_client
            svc.dependency_provider._sdk_client = async_sdk

        # Merge dependency overrides
        all_overrides = {}
        if dependency_overrides:
            all_overrides.update(dependency_overrides)

        # Override get_sdk_client to return a request-scoped SDK with current auth headers.
        # This mirrors what DependencyProvider.get_request_scoped_sdk() does in production.
        if get_sdk_client not in all_overrides:
            from nmp.common.sdk_factory import get_request_scoped_sdk

            def _get_request_scoped_test_sdk() -> AsyncNeMoPlatform:
                return get_request_scoped_sdk(async_sdk)

            all_overrides[get_sdk_client] = _get_request_scoped_test_sdk

        # Override get_entity_client to create a fresh EntityClient using
        # build_downstream_service_headers, matching DependencyProvider._get_entity_sdk_on_behalf_of()
        # in production. The current service is taken from AppContext (set by
        # create_app_context_dependency on each sub-router) so merged multi-service
        # apps get service:<name> and on-behalf-of the same as real deployments;
        # routes without that context (e.g. /health) fall back to "platform".
        if get_entity_client not in all_overrides:
            from nmp.common.observability.context import get_app_ctx
            from nmp.common.service.headers import build_downstream_service_headers

            def _get_entity_client_on_behalf_of() -> EntityClient:
                app_ctx = get_app_ctx()
                service_name = app_ctx.service_name if app_ctx is not None and app_ctx.service_name else "platform"
                headers = build_downstream_service_headers(service_name)
                sdk = async_sdk.with_options(set_default_headers=headers)
                return EntityClient(client_from_platform(sdk, AsyncEntitiesClient))

            all_overrides[get_entity_client] = _get_entity_client_on_behalf_of

        if all_overrides:
            app.dependency_overrides.update(all_overrides)

        with TestClient(app) as client:
            # Use max_retries=0 to avoid retry delays on 409 Conflict errors
            sdk_http_client = SDKTestClientAdapter(client)
            sdk = NeMoPlatform(
                workspace=workspace,
                base_url="http://testserver",
                http_client=sdk_http_client,
                max_retries=0,
            )
            _install_asgi_files_resource(sdk)

            for svc in services_to_start:
                svc.dependency_provider._sync_http_client = sdk_http_client
                svc.dependency_provider._sync_sdk_client = sdk

            if get_sync_sdk_client not in all_overrides:
                from nmp.common.sdk_factory import get_request_scoped_sync_sdk

                def _get_request_scoped_sync_test_sdk() -> NeMoPlatform:
                    return get_request_scoped_sync_sdk(sdk)

                app.dependency_overrides[get_sync_sdk_client] = _get_request_scoped_sync_test_sdk

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

                    from nmp.core.auth.app.embedded_pdp import load_policy_data
                    from nmp.core.auth.app.seeding import run_seeding

                    # Use service principal so entity store accepts role binding creation
                    headers: dict[str, str] = {
                        key: value
                        for key, value in dict(async_sdk.default_headers or {}).items()
                        if isinstance(value, str)
                    }
                    headers["X-NMP-Principal-Id"] = "service:auth"
                    seeding_sdk = async_sdk.with_options(set_default_headers=headers)
                    seeding_entity_client = EntityClient(client_from_platform(seeding_sdk, AsyncEntitiesClient))

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
                auth_headers = {"X-NMP-Principal-Id": TEST_USER_EMAIL}
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
                # No auth - use SDK directly
                from nemo_platform_plugin.client.errors import ConflictError

                for ws_id in workspaces_to_create:
                    try:
                        client_from_platform(sdk, WorkspacesClient).create_workspace(
                            body=CreateWorkspaceRequest(name=ws_id)
                        ).data()
                    except ConflictError:
                        logger.warning(f"Workspace '{ws_id}' already exists (created by service startup)")
                from nemo_platform_plugin.projects.client import ProjectsClient
                from nemo_platform_plugin.projects.types import CreateProjectRequest

                projects_client = client_from_platform(sdk, ProjectsClient)
                for ws_id, proj_name in parsed_projects:
                    try:
                        projects_client.create_project(workspace=ws_id, body=CreateProjectRequest(name=proj_name))
                    except ConflictError:
                        logger.warning(f"Project '{proj_name}' in workspace '{ws_id}' already exists")

            if selected_client_type is TestClient:
                yield client  # ty: ignore[invalid-yield]
            elif selected_client_type is AsyncNemoClient:
                yield async_client  # ty: ignore[invalid-yield]
            elif selected_client_type is AsyncNeMoPlatform:
                yield async_sdk  # ty: ignore[invalid-yield]
            elif selected_client_type is EntityClient:
                yield entity_client  # ty: ignore[invalid-yield]
            elif selected_client_type is ClientContext:
                yield ClientContext(
                    sdk=sdk,
                    async_sdk=async_sdk,
                    async_client=async_client,
                    entity_client=entity_client,
                    test_client=client,
                    access_log=access_log_instance,
                )
            else:
                yield sdk  # ty: ignore[invalid-yield]

        app.dependency_overrides.clear()
