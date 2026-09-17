# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import builtins
import inspect
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from nemo_platform_plugin.jobs.openapi_utils import clear_query_param_schemas, generate_openapi_extra_params
from nmp.common.config import AuthConfig, Configuration
from nmp.common.config.base import OIDCConfig
from nmp.common.controller import ControllerManager
from nmp.common.platform_endpoint import _SyncPlatformEndpointRoutingTransport
from nmp.common.service import RouterConfig, Service
from nmp.platform_runner import config as runner_config
from nmp.platform_runner import server
from nmp.platform_runner.config import DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS
from nmp.platform_runner.health import ReadinessCheck, create_platform_health_router
from pydantic import BaseModel

_RUN_ENV_KEYS = (
    "NMP_CONFIG_FILE_PATH",
    "NMP_SERVICE_HOST",
    "NMP_SERVICE_PORT",
    "NMP_BASE_URL",
    "NMP_AUTH_POLICY_DECISION_POINT_BASE_URL",
    "NMP_SERVICES",
    "NMP_CONTROLLERS",
    "NMP_SIDECARS",
)


@pytest.fixture(autouse=True)
def restore_platform_runner_env():
    original_env = {key: os.environ.get(key) for key in _RUN_ENV_KEYS}
    yield
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    Configuration.clear_cache()


def _make_auth_config(*, enabled: bool) -> AuthConfig:
    return AuthConfig(
        enabled=enabled,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(enabled=False),
    )


class PluginService(Service):
    def __init__(self):
        super().__init__(name="agents", module_name="test.plugin")

    def get_routers(self):
        return []


class _DateFilter(BaseModel):
    gte: str | None = None


class _ListFilter(BaseModel):
    created_at: _DateFilter | None = None


class QueryParamSchemaService(Service):
    def __init__(self):
        super().__init__(name="query-service", module_name="test.query_service")

    def get_routers(self):
        router = APIRouter()

        @router.get("/items", openapi_extra=generate_openapi_extra_params(filter_schema=_ListFilter))
        async def list_items():
            return {"data": []}

        return [RouterConfig(router, tag="Query", description="Query endpoints")]


async def _ready() -> bool:
    return True


async def _not_ready() -> bool:
    return False


def _client_for_health_checks(checks: list[ReadinessCheck]) -> TestClient:
    app = FastAPI()
    app.include_router(create_platform_health_router([PluginService()], readiness_checks=checks))
    return TestClient(app)


def _patch_platform_app_config(monkeypatch, *, seed_on_startup: bool):
    auth_cfg = _make_auth_config(enabled=False)
    platform_cfg = _make_platform_config_mock()
    platform_cfg.seed_on_startup = seed_on_startup
    monkeypatch.setattr(server, "get_platform_config", lambda: platform_cfg)
    monkeypatch.setattr(server, "get_auth_config", lambda: auth_cfg)

    import nmp.common.auth.middleware as auth_middleware

    monkeypatch.setattr(auth_middleware, "get_auth_config", lambda: auth_cfg)
    return platform_cfg


def _wait_for_response(client: TestClient, path: str, status_code: int, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    last_response = None
    while time.monotonic() < deadline:
        last_response = client.get(path)
        if last_response.status_code == status_code:
            return last_response
        time.sleep(0.05)
    return last_response


def test_platform_health_ready_includes_startup_readiness_checks():
    client = _client_for_health_checks(
        [ReadinessCheck(name="platform-seed", is_ready=_not_ready, message=lambda: "pending")]
    )

    response = client.get("/health/ready")
    assert response.status_code == 503

    status = client.get("/status").json()
    assert "agents" in status["services"]["ready"]
    assert {"name": "platform-seed", "message": "pending"} in status["services"]["not_ready"]


def test_platform_health_status_reports_ready_startup_checks():
    client = _client_for_health_checks([ReadinessCheck(name="platform-seed", is_ready=_ready)])

    response = client.get("/health/ready")
    assert response.status_code == 200

    status = client.get("/status").json()
    assert "platform-seed" in status["services"]["ready"]


def test_create_platform_openapi_app_includes_explicit_service_instances(monkeypatch):
    plugin_service = PluginService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(server, "get_available_services", lambda: {"agents": plugin_service, "auth": "skip"})
    monkeypatch.setattr(server, "get_openapi_service_names", lambda _available: ["agents"])
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda services: services)

    def fake_create_app(services, _controller_run_funcs=None, _http_client=None):
        captured["services"] = services
        return FastAPI()

    monkeypatch.setattr(server, "create_app", fake_create_app)

    server.create_platform_openapi_app()

    assert captured["services"] == [plugin_service]


def test_create_default_app_uses_plugin_services_and_controllers(monkeypatch):
    plugin_service = PluginService()
    captured: dict[str, object] = {}

    def plugin_controller(_stop_signal):
        return None

    monkeypatch.setattr(server, "_obs_initialized", True)
    monkeypatch.setattr(server, "get_available_services", lambda: {"agents": plugin_service})
    monkeypatch.setattr(server, "get_available_controllers", lambda: {"agents-deployment": plugin_controller})
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda services: services)

    def fake_create_app(services, controller_run_funcs=None, _http_client=None):
        captured["services"] = services
        captured["controller_run_funcs"] = controller_run_funcs
        return FastAPI()

    monkeypatch.setattr(server, "create_app", fake_create_app)

    server.create_default_app()

    assert captured["services"] == [plugin_service]
    assert captured["controller_run_funcs"] == {"agents-deployment": plugin_controller}


def test_create_app_marks_mounted_services_as_local(monkeypatch):
    platform_cfg = _patch_platform_app_config(monkeypatch, seed_on_startup=False)
    platform_cfg.services = ""

    server.create_app(services=[PluginService()])

    assert platform_cfg.services == "agents"


def test_create_app_openapi_registers_rebased_query_param_schemas(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=False)
    clear_query_param_schemas()
    try:
        app = server.create_app(services=[QueryParamSchemaService()])
        spec = app.openapi()
        schemas = spec["components"]["schemas"]
        filter_param = next(
            param
            for param in spec["paths"]["/apis/query-service/items"]["get"]["parameters"]
            if param["name"] == "filter"
        )

        assert filter_param["schema"] == {"$ref": "#/components/schemas/_ListFilter"}
        assert "_ListFilter" in schemas
        assert "_DateFilter" in schemas
        assert "$defs" not in schemas["_ListFilter"]
    finally:
        clear_query_param_schemas()


def test_create_app_uses_separate_http_clients_for_auth_callouts(monkeypatch):
    platform_cfg = _patch_platform_app_config(monkeypatch, seed_on_startup=False)
    platform_cfg.services = ""
    http_client = MagicMock()
    lifecycle_http_client = MagicMock()

    app = server.create_app(
        services=[PluginService()],
        http_client=http_client,
        access_key_lifecycle_http_client=lifecycle_http_client,
    )

    auth_middleware = next(
        middleware for middleware in app.user_middleware if middleware.cls is server.AuthorizationMiddleware
    )
    assert auth_middleware.kwargs["http_client"] is http_client
    assert auth_middleware.kwargs["access_key_lifecycle_http_client"] is lifecycle_http_client


def test_create_app_mounted_services_drive_sdk_local_routing_without_services_env(monkeypatch):
    monkeypatch.delenv("NMP_SERVICES", raising=False)
    monkeypatch.setenv("NMP_BASE_URL", "https://nemo-gateway:8080")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()
    platform_cfg = Configuration.get_platform_config()

    try:
        auth_cfg = _make_auth_config(enabled=False)
        monkeypatch.setattr(server, "get_platform_config", lambda: platform_cfg)
        monkeypatch.setattr(server, "get_auth_config", lambda: auth_cfg)

        import nmp.common.auth.middleware as auth_middleware
        from nmp.common.sdk_factory import get_platform_sdk

        monkeypatch.setattr(auth_middleware, "get_auth_config", lambda: auth_cfg)

        server.create_app(services=[PluginService()])

        with get_platform_sdk() as sdk:
            transport = sdk._client._transport

        assert platform_cfg.services == "agents"
        assert isinstance(transport, _SyncPlatformEndpointRoutingTransport)
    finally:
        Configuration.clear_cache()


def test_build_platform_app_returns_app_without_running_uvicorn(monkeypatch):
    plugin_service = PluginService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner_config, "get_available_services", lambda: {"agents": plugin_service})
    monkeypatch.setattr(runner_config, "get_available_controllers", lambda: {})
    monkeypatch.setattr(runner_config, "get_controller_groups", lambda _controllers: {"all": [], "core": []})
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda services: services)

    def fake_create_app(
        services,
        controller_run_funcs=None,
        http_client=None,
        access_key_lifecycle_http_client=None,
        sidecar_run_funcs=None,
    ):
        captured["services"] = services
        captured["controller_run_funcs"] = controller_run_funcs
        captured["sidecar_run_funcs"] = sidecar_run_funcs
        captured["http_client"] = http_client
        captured["access_key_lifecycle_http_client"] = access_key_lifecycle_http_client
        return FastAPI()

    monkeypatch.setattr(server, "create_app", fake_create_app)

    lifecycle_http_client = MagicMock()
    app = server.build_platform_app(
        runner_config.PlatformAppConfig(services=["agents"], controllers=[]),
        access_key_lifecycle_http_client=lifecycle_http_client,
        env={},
    )

    assert isinstance(app, FastAPI)
    assert captured["services"] == [plugin_service]
    assert captured["controller_run_funcs"] == {}
    assert captured["sidecar_run_funcs"] == {}
    assert captured["http_client"] is None
    assert captured["access_key_lifecycle_http_client"] is lifecycle_http_client


def test_build_platform_app_accepts_platform_app_config(monkeypatch):
    plugin_service = PluginService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner_config, "get_available_services", lambda: {"agents": plugin_service})
    monkeypatch.setattr(runner_config, "get_available_controllers", lambda: {})
    monkeypatch.setattr(runner_config, "get_controller_groups", lambda _controllers: {"all": [], "core": []})
    monkeypatch.setattr(server, "order_services_by_dependencies", lambda services: services)

    def fake_create_app(
        services,
        controller_run_funcs=None,
        http_client=None,
        access_key_lifecycle_http_client=None,
        sidecar_run_funcs=None,
    ):
        captured["services"] = services
        captured["controller_run_funcs"] = controller_run_funcs
        captured["sidecar_run_funcs"] = sidecar_run_funcs
        captured["http_client"] = http_client
        captured["access_key_lifecycle_http_client"] = access_key_lifecycle_http_client
        return FastAPI()

    monkeypatch.setattr(server, "create_app", fake_create_app)

    app = server.build_platform_app(
        config=runner_config.PlatformAppConfig(services=("agents",), controllers=()),
        env={},
    )

    assert isinstance(app, FastAPI)
    assert captured["services"] == [plugin_service]
    assert captured["controller_run_funcs"] == {}
    assert captured["sidecar_run_funcs"] == {}
    assert captured["http_client"] is None
    assert captured["access_key_lifecycle_http_client"] is None


def test_embedded_auth_preflight_invokes_policy_wasm_helper(monkeypatch):
    calls: list[bool] = []
    auth_cfg = AuthConfig(
        enabled=True,
        policy_decision_point_provider="embedded",
        embedded_pdp_auto_build_wasm=False,
    )

    from nmp.core.auth.app.embedded_pdp import policy_wasm

    monkeypatch.setattr(policy_wasm, "ensure_embedded_policy_wasm", lambda *, auto_build: calls.append(auto_build))

    server.preflight_embedded_auth_policy_wasm(auth_cfg)

    assert calls == [False]


@pytest.mark.parametrize(
    "auth_cfg",
    [
        AuthConfig(enabled=False, policy_decision_point_provider="embedded"),
        AuthConfig(enabled=True, policy_decision_point_provider="opa"),
    ],
)
def test_embedded_auth_preflight_skips_when_not_needed(auth_cfg, monkeypatch):
    calls: list[bool] = []

    from nmp.core.auth.app.embedded_pdp import policy_wasm

    monkeypatch.setattr(policy_wasm, "ensure_embedded_policy_wasm", lambda *, auto_build: calls.append(auto_build))

    server.preflight_embedded_auth_policy_wasm(auth_cfg)

    assert calls == []


def test_run_server_runs_embedded_auth_preflight():
    auth_cfg = _make_auth_config(enabled=True)
    calls: list[AuthConfig] = []
    with (
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch(
            "nmp.platform_runner.server.preflight_embedded_auth_policy_wasm", side_effect=lambda cfg: calls.append(cfg)
        ),
        patch("nmp.platform_runner.server.create_app", return_value=FastAPI()) as create_app,
        patch("nmp.platform_runner.server.setup_fastapi_instrumentations"),
        patch("nmp.platform_runner.server.uvicorn.run") as uvicorn_run,
    ):
        server.run_server(services=[], host="127.0.0.1", port=9999, keep_alive_timeout_seconds=12)

    assert calls == [auth_cfg]
    create_app.assert_called_once_with([])
    uvicorn_run.assert_called_once()
    assert uvicorn_run.call_args.kwargs["timeout_keep_alive"] == 12


def test_run_server_can_bind_tcp_and_unix_domain_socket():
    auth_cfg = _make_auth_config(enabled=True)
    with (
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch("nmp.platform_runner.server.preflight_embedded_auth_policy_wasm"),
        patch("nmp.platform_runner.server.create_app", return_value=FastAPI()),
        patch("nmp.platform_runner.server.setup_fastapi_instrumentations"),
        patch("nmp.platform_runner.server._run_server_on_bound_sockets") as run_bound_sockets,
    ):
        server.run_server(services=[], host="127.0.0.1", port=9999, socket_path="/tmp/nemo-platform.sock")

    run_bound_sockets.assert_called_once()
    assert run_bound_sockets.call_args.kwargs == {
        "host": "127.0.0.1",
        "keep_alive_timeout_seconds": DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
        "port": 9999,
        "socket_path": "/tmp/nemo-platform.sock",
    }


def test_run_server_on_bound_sockets_sets_keep_alive_timeout():
    app = FastAPI()
    tcp_socket = MagicMock()
    uds_socket = MagicMock()
    tcp_config = MagicMock()
    uds_config = MagicMock()
    tcp_config.bind_socket.return_value = tcp_socket
    uds_config.bind_socket.return_value = uds_socket

    with (
        patch("nmp.platform_runner.server.uvicorn.Config", side_effect=[tcp_config, uds_config]) as config_cls,
        patch("nmp.platform_runner.server.uvicorn.Server") as server_cls,
        patch("nmp.platform_runner.server.asyncio.run") as asyncio_run,
    ):
        server._run_server_on_bound_sockets(
            app,
            host="127.0.0.1",
            port=9999,
            socket_path="/tmp/nemo-platform.sock",
            keep_alive_timeout_seconds=12,
        )

    assert config_cls.call_args_list[0].kwargs["timeout_keep_alive"] == 12
    assert config_cls.call_args_list[1].kwargs["timeout_keep_alive"] == 12
    server_cls.assert_called_once_with(tcp_config)
    server_cls.return_value.serve.assert_called_once_with(sockets=[tcp_socket, uds_socket])
    asyncio_run.assert_called_once_with(server_cls.return_value.serve.return_value)
    tcp_socket.close.assert_called_once_with()
    uds_socket.close.assert_called_once_with()


def test_run_server_with_reload_sets_keep_alive_timeout():
    auth_cfg = _make_auth_config(enabled=True)
    with (
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch("nmp.platform_runner.server.preflight_embedded_auth_policy_wasm"),
        patch("nmp.platform_runner.server.uvicorn.run") as uvicorn_run,
    ):
        server.run_server_with_reload(
            "nmp.platform_runner.server:create_default_app",
            host="127.0.0.1",
            port=9999,
            keep_alive_timeout_seconds=12,
        )

    uvicorn_run.assert_called_once()
    assert uvicorn_run.call_args.kwargs["timeout_keep_alive"] == 12


def test_server_default_keep_alive_matches_runner_config_default():
    assert (
        inspect.signature(server.run_server).parameters["keep_alive_timeout_seconds"].default
        == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS
    )
    assert (
        inspect.signature(server._run_server_on_bound_sockets).parameters["keep_alive_timeout_seconds"].default
        == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS
    )
    assert (
        inspect.signature(server.run_server_with_reload).parameters["keep_alive_timeout_seconds"].default
        == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS
    )


def test_create_default_app_raises_for_unknown_service_from_env(monkeypatch):
    monkeypatch.setattr(server, "_obs_initialized", True)
    monkeypatch.setenv("NMP_SERVICES", "missing-service")
    monkeypatch.setattr(server, "get_available_services", lambda: {})
    monkeypatch.setattr(server, "get_available_controllers", lambda: {})

    with pytest.raises(
        ValueError, match="Unknown service 'missing-service' requested via NMP_SERVICES='missing-service'"
    ):
        server.create_default_app()


def test_create_default_app_raises_for_unknown_controller_from_env(monkeypatch):
    monkeypatch.setattr(server, "_obs_initialized", True)
    monkeypatch.delenv("NMP_SERVICES", raising=False)
    monkeypatch.setenv("NMP_CONTROLLERS", "missing-controller")
    monkeypatch.setattr(server, "get_available_services", lambda: {})
    monkeypatch.setattr(server, "get_available_controllers", lambda: {})

    with pytest.raises(
        ValueError,
        match="Unknown controller 'missing-controller' requested via NMP_CONTROLLERS='missing-controller'",
    ):
        server.create_default_app()


def _make_platform_config_mock(*, redirect_root_to_studio: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.seed_on_startup = False
    cfg.redirect_root_to_studio = redirect_root_to_studio
    return cfg


def test_create_app_without_seed_on_startup_keeps_health_ready_unchanged(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=False)

    with TestClient(server.create_app(services=[])) as client:
        response = client.get("/health/ready")
        status = client.get("/status").json()

    assert response.status_code == 200
    assert "platform-seed" not in status["services"]["ready"]
    assert all(item["name"] != "platform-seed" for item in status["services"]["not_ready"])


def test_create_app_rejects_controller_sidecar_name_collision(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=False)

    with pytest.raises(ValueError, match="Controller/sidecar name collision: shared"):
        server.create_app(
            services=[],
            controller_run_funcs={"shared": lambda _stop_signal: None},
            sidecar_run_funcs={"shared": lambda _stop_signal: None},
        )


def test_create_app_reports_controller_that_crashes_before_registration(monkeypatch, caplog):
    _patch_platform_app_config(monkeypatch, seed_on_startup=False)
    crashed = threading.Event()

    def crashing_controller(_stop_signal: threading.Event) -> None:
        try:
            raise RuntimeError("startup failed")
        finally:
            crashed.set()

    manager = ControllerManager.get_instance()
    manager.clear()
    try:
        with caplog.at_level(logging.ERROR, logger="nmp.platform_runner.controller_threads"):
            with TestClient(
                server.create_app(services=[], controller_run_funcs={"models": crashing_controller})
            ) as client:
                assert crashed.wait(timeout=2)
                # The wrapper records failure after crashed is set.
                response = _wait_for_response(client, "/health/ready", 503)
                status = client.get("/status").json()

        assert response.status_code == 503
        assert status["controllers"] == {"healthy": False, "status": {"models": False}}
        assert "Controller 'models' crashed" in caplog.text
    finally:
        manager.clear()


def test_create_app_with_seed_on_startup_blocks_readiness_until_seed_completes(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=True)
    started = threading.Event()
    release = threading.Event()

    async def fake_seed() -> bool:
        started.set()
        await asyncio.to_thread(release.wait)
        return True

    monkeypatch.setitem(sys.modules, "nmp.platform_seed", SimpleNamespace(run_platform_seed_from_startup=fake_seed))

    with TestClient(server.create_app(services=[])) as client:
        assert started.wait(timeout=2)
        response = client.get("/health/ready")
        assert response.status_code == 503

        status = client.get("/status").json()
        assert {"name": "platform-seed", "message": "pending"} in status["services"]["not_ready"]

        release.set()
        response = _wait_for_response(client, "/health/ready", 200)

    assert response is not None
    assert response.status_code == 200


def test_create_app_with_failed_seed_keeps_health_not_ready(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=True)

    async def fake_seed() -> bool:
        return False

    monkeypatch.setitem(sys.modules, "nmp.platform_seed", SimpleNamespace(run_platform_seed_from_startup=fake_seed))

    with TestClient(server.create_app(services=[])) as client:
        response = _wait_for_response(client, "/health/ready", 503)
        status = client.get("/status").json()

    assert response is not None
    assert response.status_code == 503
    assert {"name": "platform-seed", "message": "platform seed failed"} in status["services"]["not_ready"]


def test_create_app_with_missing_seed_package_keeps_health_not_ready(monkeypatch):
    _patch_platform_app_config(monkeypatch, seed_on_startup=True)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nmp.platform_seed":
            raise ImportError("missing platform seed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with TestClient(server.create_app(services=[])) as client:
        response = client.get("/health/ready")
        status = client.get("/status").json()

    assert response.status_code == 503
    assert {"name": "platform-seed", "message": "platform seed is not installed"} in status["services"]["not_ready"]


@pytest.mark.parametrize("auth_enabled", [True, False])
@pytest.mark.parametrize("method", ["get", "head"])
def test_root_redirects_to_studio(auth_enabled, method):
    auth_cfg = _make_auth_config(enabled=auth_enabled)
    with (
        patch("nmp.platform_runner.server.get_platform_config", return_value=_make_platform_config_mock()),
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch("nmp.common.auth.middleware.get_auth_config", return_value=auth_cfg),
    ):
        app = server.create_app(services=[])
        client = TestClient(app, follow_redirects=False)
        response = getattr(client, method)("/")

    assert response.status_code == 301
    assert response.headers["location"] == "/studio"


def test_root_returns_ok_when_redirect_disabled():
    auth_cfg = _make_auth_config(enabled=False)
    with (
        patch(
            "nmp.platform_runner.server.get_platform_config",
            return_value=_make_platform_config_mock(redirect_root_to_studio=False),
        ),
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch("nmp.common.auth.middleware.get_auth_config", return_value=auth_cfg),
    ):
        app = server.create_app(services=[])
        client = TestClient(app, follow_redirects=False)
        response = client.get("/")

    assert response.status_code == 200


def test_non_get_root_still_requires_auth():
    auth_cfg = _make_auth_config(enabled=True)
    with (
        patch("nmp.platform_runner.server.get_platform_config", return_value=_make_platform_config_mock()),
        patch("nmp.platform_runner.server.get_auth_config", return_value=auth_cfg),
        patch("nmp.common.auth.middleware.get_auth_config", return_value=auth_cfg),
        patch("nmp.common.auth.client.AuthClient.authorize_request", return_value=MagicMock(allowed=False)),
    ):
        app = server.create_app(services=[])
        client = TestClient(app, raise_server_exceptions=False, follow_redirects=False)
        response = client.post("/")

    assert response.status_code == 401
