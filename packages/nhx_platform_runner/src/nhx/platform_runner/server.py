# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic API server helpers for platform services."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import asynccontextmanager
from typing import cast

import httpx
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nhx.common.api.utils import install_query_param_schema_openapi_hook
from nhx.common.auth import AuthorizationMiddleware
from nhx.common.config import get_auth_config, get_platform_config
from nhx.common.http_clients import close_shared_http_clients
from nhx.common.observability import initialize_obs, setup_fastapi_instrumentations, setup_global_instrumentations
from nhx.common.observability.context import create_app_context_dependency
from nhx.common.pyleak import detect_blocking
from nhx.common.service import Service
from nhx.platform_runner.config import DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS, PlatformAppConfig
from nhx.platform_runner.controller_threads import (
    join_and_untrack_runner_threads,
    start_controller_threads,
    start_sidecar_threads,
)
from nhx.platform_runner.health import ReadinessCheck, create_platform_health_router, get_platform_resource_attributes
from nhx.platform_runner.loader import (
    ControllerRunFunc,
    load_controller_run_func,
    load_service,
    order_services_by_dependencies,
)
from nhx.platform_runner.registry import (
    AVAILABLE_SIDECARS,
    check_no_controller_sidecar_collision,
    get_available_controllers,
    get_available_services,
    get_openapi_service_names,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

logger = logging.getLogger(__name__)


class _StartupReadinessState:
    def __init__(self, name: str) -> None:
        self.name = name
        self._ready = False
        self._message = "pending"

    async def is_ready(self) -> bool:
        return self._ready

    def message(self) -> str:
        return self._message

    def mark_ready(self) -> None:
        self._ready = True
        self._message = "ready"

    def mark_failed(self, message: str) -> None:
        self._ready = False
        self._message = message


async def platform_global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback exception handler for uncaught platform errors."""
    extra = {
        "method": request.method,
        "path": request.url.path,
        "exc_type": type(exc).__name__,
    }
    if service := getattr(request.state, "service", None):
        extra["service"] = service
    if workspace := getattr(request.state, "workspace", None):
        extra["workspace"] = workspace

    logger.error(
        "Unhandled exception",
        exc_info=exc,
        extra=extra,
    )
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})


class ConflictRetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if response.status_code == 409 and "x-should-retry" not in response.headers:
            response.headers["x-should-retry"] = "false"
        return response


class PyleakMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, threshold: float):
        super().__init__(app)
        self.threshold = threshold

    async def dispatch(self, request: Request, call_next) -> Response:
        async with detect_blocking(threshold=self.threshold):
            return await call_next(request)


def preflight_embedded_auth_policy_wasm(auth_config) -> None:
    """Ensure local embedded auth PDP has a loadable policy.wasm before serving traffic."""
    if not auth_config.enabled or auth_config.policy_decision_point_provider != "embedded":
        return

    try:
        from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
    except ImportError as exc:
        raise RuntimeError(
            "Auth is enabled with the embedded PDP, but the nhx-auth package is not installed. "
            "Install nhx-auth or set auth.policy_decision_point_provider='opa'."
        ) from exc

    ensure_embedded_policy_wasm(auto_build=getattr(auth_config, "embedded_pdp_auto_build_wasm", True))


def create_platform_openapi_app() -> FastAPI:
    """Create the platform app used for aggregate OpenAPI generation."""
    services = []
    available_services = get_available_services()
    for service_name in get_openapi_service_names(available_services):
        service_value = available_services[service_name]
        if isinstance(service_value, Service):
            services.append(service_value)
        else:
            services.append(load_service(service_name, service_value))
    return create_app(order_services_by_dependencies(services))


def create_app(
    services: list[Service] | None = None,
    controller_run_funcs: dict[str, ControllerRunFunc] | None = None,
    http_client: httpx.AsyncClient | None = None,
    access_key_lifecycle_http_client: httpx.AsyncClient | None = None,
    sidecar_run_funcs: dict[str, ControllerRunFunc] | None = None,
) -> FastAPI:
    """Create the FastAPI app from service instances."""
    services = services or []
    controller_run_funcs = controller_run_funcs or {}
    sidecar_run_funcs = sidecar_run_funcs or {}
    check_no_controller_sidecar_collision(controller_run_funcs.keys(), sidecar_run_funcs.keys())
    controller_stop_signal = threading.Event()
    platform_config = get_platform_config()
    platform_config.services = ",".join(sorted(service.name for service in services))
    readiness_checks: list[ReadinessCheck] = []
    platform_seed_state: _StartupReadinessState | None = None
    if platform_config.seed_on_startup:
        platform_seed_state = _StartupReadinessState("platform-seed")
        readiness_checks.append(
            ReadinessCheck(
                name=platform_seed_state.name,
                is_ready=platform_seed_state.is_ready,
                message=platform_seed_state.message,
            )
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Nemo Helix server")
        controller_threads = []
        thread_by_name: dict[str, threading.Thread] = {}
        platform_seed_task: asyncio.Task[None] | None = None
        try:
            # start_*_threads can block synchronously: on a mid-batch failure
            # (e.g. thread.start() raising under thread exhaustion) it rolls
            # back already-started components by joining them, which — like
            # the shutdown join below — can take up to each component's
            # configured timeout. Run it off the event loop so a failure here
            # doesn't stall every other in-flight request during startup.
            if controller_run_funcs:
                logger.info("Starting controllers in lifespan: %s", list(controller_run_funcs))
                started = await asyncio.to_thread(
                    start_controller_threads, controller_run_funcs, controller_stop_signal
                )
                thread_by_name.update(zip(controller_run_funcs, started))
                controller_threads.extend(started)
            if sidecar_run_funcs:
                logger.info("Starting sidecars in lifespan: %s", list(sidecar_run_funcs))
                started = await asyncio.to_thread(start_sidecar_threads, sidecar_run_funcs, controller_stop_signal)
                thread_by_name.update(zip(sidecar_run_funcs, started))
                controller_threads.extend(started)
        except Exception:
            controller_stop_signal.set()
            await asyncio.to_thread(
                join_and_untrack_runner_threads,
                controller_threads,
                thread_by_name,
                controller_run_funcs.keys() | sidecar_run_funcs.keys(),
            )
            await close_shared_http_clients()
            raise

        if platform_seed_state is not None:
            try:
                from nhx.platform_seed import run_platform_seed_from_startup

                async def run_platform_seed_and_update_readiness() -> None:
                    try:
                        ok = await run_platform_seed_from_startup()
                    except Exception:
                        logger.exception("Platform seed task failed")
                        platform_seed_state.mark_failed("platform seed failed")
                        return

                    if ok:
                        platform_seed_state.mark_ready()
                    else:
                        platform_seed_state.mark_failed("platform seed failed")

                platform_seed_task = asyncio.create_task(run_platform_seed_and_update_readiness())
                logger.info("Platform seed task scheduled")
            except ImportError as error:
                logger.warning("platform.seed_on_startup is True but platform_seed is not installed: %s", error)
                platform_seed_state.mark_failed("platform seed is not installed")

        app.state.controller_threads = controller_threads
        app.state.controller_stop_signal = controller_stop_signal
        app.state.platform_seed_task = platform_seed_task

        yield

        if platform_seed_task is not None and not platform_seed_task.done():
            platform_seed_task.cancel()
            try:
                await platform_seed_task
            except asyncio.CancelledError:
                pass

        controller_stop_signal.set()
        # join_and_untrack_runner_threads blocks synchronously (it polls with
        # time.sleep) for up to each component's shutdown timeout — offload it
        # so a slow-to-stop controller/sidecar (e.g. auth-proxy's 16s budget)
        # doesn't stall the event loop and every other in-flight request.
        await asyncio.to_thread(
            join_and_untrack_runner_threads,
            controller_threads,
            thread_by_name,
            controller_run_funcs.keys() | sidecar_run_funcs.keys(),
        )

        await close_shared_http_clients()
        logger.info("Shutting down Nemo Helix API server")

    app = FastAPI(
        title="Nemo Helix API",
        description="API for Nemo Helix services",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.add_middleware(ConflictRetryMiddleware)

    pyleak_threshold = float(os.environ.get("PYLEAK_THRESHOLD", "0"))
    if pyleak_threshold > 0:
        app.add_middleware(PyleakMiddleware, threshold=pyleak_threshold)

    auth_config = get_auth_config()
    logger.info("Adding AuthorizationMiddleware", extra={"auth_enabled": auth_config.enabled})
    app.add_middleware(
        AuthorizationMiddleware,
        service_name="platform",
        http_client=http_client,
        access_key_lifecycle_http_client=access_key_lifecycle_http_client,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    app.state.service_configs = {}
    app.include_router(create_platform_health_router(services, readiness_checks=readiness_checks))

    redirect_root_to_studio = platform_config.redirect_root_to_studio

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False, response_model=None)
    async def root_handler() -> Response:
        if redirect_root_to_studio:
            return RedirectResponse(url="/studio", status_code=301)
        return Response(status_code=200, content="OK")

    for service_instance in services:
        service_app = service_instance.app
        app.dependency_overrides.update(service_app.dependency_overrides)
        exception_handlers = {
            exc_type: handler
            for exc_type, handler in service_app.exception_handlers.items()
            if exc_type is not Exception
        }
        app.exception_handlers.update(exception_handlers)
        app.include_router(
            router=service_app.router,
            prefix=f"/apis/{service_instance.name}",
            dependencies=[Depends(create_app_context_dependency(service_instance.name))],
        )
        configure_app = getattr(service_instance, "configure_app", None)
        if configure_app is not None and callable(configure_app):
            if "app" in inspect.signature(configure_app).parameters:
                configure_app(app)
        setattr(app.state, f"{service_instance.name.replace('-', '_')}_service", service_instance)
        if service_instance._service_config is not None:
            app.state.service_configs[type(service_instance._service_config)] = service_instance._service_config

    install_query_param_schema_openapi_hook(app)
    app.add_exception_handler(Exception, platform_global_exception_handler)
    return app


def _load_run_functions(
    names: list[str],
    registry: Mapping[str, str | Callable[[threading.Event], object]],
) -> dict[str, Callable[[threading.Event], object]]:
    run_funcs: dict[str, Callable[[threading.Event], object]] = {}
    for name in names:
        value = registry[name]
        if isinstance(value, str):
            run_funcs[name] = load_controller_run_func(name, value)
        else:
            run_funcs[name] = value
    return run_funcs


def build_platform_app(
    config: PlatformAppConfig | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    access_key_lifecycle_http_client: httpx.AsyncClient | None = None,
    env: MutableMapping[str, str] | None = None,
) -> FastAPI:
    """Build a platform FastAPI app without starting uvicorn.

    Args:
        config: App-build selection and bind configuration. Prefer this over
            individual service/controller/sidecar keyword arguments for new
            callers.
        env: Environment mapping passed to :func:`apply_run_environment`.
            Defaults to ``None`` which writes to ``os.environ``.  Tests can
            pass an empty dict to avoid polluting the process environment.
    """
    from nhx.platform_runner.config import apply_run_environment, resolve_run_configuration

    resolved = resolve_run_configuration(config)
    apply_run_environment(resolved, env=env)

    service_instances = []
    for service_name in sorted(resolved.services):
        service_value = resolved.available_services[service_name]
        service_instances.append(
            service_value if isinstance(service_value, Service) else load_service(service_name, service_value)
        )
    service_instances = order_services_by_dependencies(service_instances)

    # Fail before loading colliding functions; create_app() checks direct callers.
    check_no_controller_sidecar_collision(resolved.controllers, resolved.sidecars)
    controller_run_funcs = _load_run_functions(sorted(resolved.controllers), resolved.available_controllers)
    sidecar_run_funcs = _load_run_functions(sorted(resolved.sidecars), AVAILABLE_SIDECARS)

    return create_app(
        service_instances,
        controller_run_funcs=controller_run_funcs,
        sidecar_run_funcs=sidecar_run_funcs,
        http_client=http_client,
        access_key_lifecycle_http_client=access_key_lifecycle_http_client,
    )


def run_server(
    services: list[Service] | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    socket_path: str | None = None,
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
) -> None:
    """Run the platform API server."""
    preflight_embedded_auth_policy_wasm(get_auth_config())
    app = create_app(services or [])
    setup_fastapi_instrumentations(app)
    if socket_path:
        _run_server_on_bound_sockets(
            app,
            host=host,
            port=port,
            socket_path=socket_path,
            keep_alive_timeout_seconds=keep_alive_timeout_seconds,
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_config=None,
            timeout_keep_alive=keep_alive_timeout_seconds,
        )


def _run_server_on_bound_sockets(
    app: FastAPI,
    *,
    host: str,
    port: int,
    socket_path: str,
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
) -> None:
    tcp_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        timeout_keep_alive=keep_alive_timeout_seconds,
    )
    uds_config = uvicorn.Config(
        app,
        uds=socket_path,
        log_config=None,
        timeout_keep_alive=keep_alive_timeout_seconds,
    )
    sockets = [tcp_config.bind_socket(), uds_config.bind_socket()]
    try:
        asyncio.run(uvicorn.Server(tcp_config).serve(sockets=sockets))
    finally:
        for sock in sockets:
            sock.close()


def run_server_with_reload(
    app_factory: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    keep_alive_timeout_seconds: int = DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_SECONDS,
) -> None:
    """Run the platform API server with uvicorn reload enabled."""
    preflight_embedded_auth_policy_wasm(get_auth_config())
    reload_dirs = [
        "packages/nhx_platform/src",
        "services/core",
        "packages/nhx_common/src",
        "packages/nhx_platform_runner/src",
    ]
    logger.warning("Hot reload is enabled. Controllers will restart with each reload.")
    uvicorn.run(
        app_factory,
        host=host,
        port=port,
        reload=True,
        reload_dirs=reload_dirs,
        log_config=None,
        access_log=False,
        log_level="warning",
        factory=True,
        timeout_keep_alive=keep_alive_timeout_seconds,
    )


_obs_initialized = False


def create_default_app() -> FastAPI:
    """Factory used by uvicorn reload mode."""
    global _obs_initialized

    if not _obs_initialized:
        initialize_obs(resource_attributes=get_platform_resource_attributes())
        setup_global_instrumentations()
        _obs_initialized = True

    service_names_env = os.environ.get("NHX_SERVICES", "")
    controller_names_env = os.environ.get("NHX_CONTROLLERS", "")

    available_services = get_available_services()
    available_controllers = get_available_controllers()

    service_names = (
        [name for name in service_names_env.split(",") if name] if service_names_env else list(available_services)
    )
    controller_names = (
        [name for name in controller_names_env.split(",") if name]
        if controller_names_env
        else list(available_controllers)
    )

    services = []
    for service_name in service_names:
        service_value = available_services.get(service_name)
        if service_value is None:
            available = ", ".join(sorted(available_services))
            raise ValueError(
                "Unknown service %r requested via NHX_SERVICES=%r. Available services: %s"
                % (service_name, service_names_env, available)
            )
        if isinstance(service_value, Service):
            services.append(service_value)
        else:
            services.append(load_service(service_name, service_value))
    services = order_services_by_dependencies(services)

    controller_run_funcs: dict[str, ControllerRunFunc] = {}
    for controller_name in controller_names:
        controller_value = available_controllers.get(controller_name)
        if controller_value is None:
            available = ", ".join(sorted(available_controllers))
            raise ValueError(
                "Unknown controller %r requested via NHX_CONTROLLERS=%r. Available controllers: %s"
                % (controller_name, controller_names_env, available)
            )
        if callable(controller_value):
            controller_run_funcs[controller_name] = cast(ControllerRunFunc, controller_value)
        else:
            controller_run_funcs[controller_name] = load_controller_run_func(controller_name, controller_value)

    return create_app(services, controller_run_funcs=controller_run_funcs)
