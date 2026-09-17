# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for the IGW middleware test harnesses.

Import :func:`igw_plugin_harness` or :func:`igw_loopback_harness` into a
test module (or re-export from a project ``conftest.py``) to register
the fixture. Both use ``pytest_httpserver``'s function-scoped
``httpserver`` so each test gets a fresh mock-NIM socket.

The ASGI stack is split across two scopes so each test only pays for
what changes between tests:

* :func:`_igw_app_context` (**module**) — the heavy
  ``create_test_client`` call (SQLite DB, FastAPI app, IGW + Models
  services, ``/health/ready`` polling, workspace seeding). The periodic
  ``refresh_model_cache_task`` is disabled so it can't wake mid-test
  and re-pollute the cache.
* :func:`_igw_loopback_context` (**module**) — a uvicorn thread bound
  on top of the app context. Only entered when a test asks for the
  loopback variant.
* :func:`igw_plugin_harness` / :func:`igw_loopback_harness`
  (**function**) — a fresh :class:`IGWPluginHarness` per test: mock-NIM
  handler mount, post-response task list re-init, entity teardown.

**xdist**: only ``--dist=loadfile`` and ``--dist=loadscope`` preserve
module scope. The default ``--dist=load`` distributes individual tests
across workers and defeats the speed-up. See the README for the
recommended command line.
"""

from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from typing import cast

import pytest
from fastapi import FastAPI
from nmp.core.inference_gateway.testing.harness import IGWLoopbackHarness, IGWPluginHarness
from nmp.testing.client import ClientContext, ServiceFactory, create_test_client
from pytest_httpserver import HTTPServer


def _app_from(client_context: ClientContext) -> FastAPI:
    """``TestClient.app`` is typed as bare ``ASGIApp``; we need :class:`FastAPI`.

    Centralised so the cast is justified in one place.
    """
    return cast(FastAPI, client_context.test_client.app)


def _enable_post_response_task_tracking(client_context: ClientContext) -> None:
    """Reset the per-test list ``proxy.py`` appends fire-and-forget tasks to.

    Production leaves ``app.state.pending_post_response_tasks`` unset and
    ``proxy.py`` skips tracking; only the harness sets it (read by
    :meth:`IGWPluginHarness.aflush_post_response`). Reset per test so a
    stale list from the previous test can't pin completed tasks or get
    re-awaited.
    """
    _app_from(client_context).state.pending_post_response_tasks = []


@contextmanager
def _build_app_context(
    *extra_services: ServiceFactory,
) -> Generator[ClientContext, None, None]:
    """Yield an IGW + Models + extras :class:`ClientContext` (module-lived).

    One module-scope hazard is neutralised here:

    The 3-second background ``refresh_model_cache_task`` reads
    ``refresh_model_cache_interval_sec`` from the module-level config snapshot
    (captured at first import), so a ``service_configs`` override is too late.
    Patch the snapshot field to 0 *before* entering ``create_test_client`` and
    ``on_startup`` never schedules the loop.
    """
    from unittest.mock import patch

    from nmp.core.inference_gateway import config as igw_config_module
    from nmp.core.inference_gateway.service import InferenceGatewayService
    from nmp.core.models.service import ModelsService

    service_types: list[ServiceFactory] = [InferenceGatewayService, ModelsService, *extra_services]

    with patch.object(igw_config_module.config, "refresh_model_cache_interval_sec", 0):
        with create_test_client(
            *service_types,
            client_type=ClientContext,
            igw_mock_provider_mode=False,
        ) as client_context:
            yield client_context


@pytest.fixture(scope="module")
def _igw_extra_services() -> tuple[ServiceFactory, ...]:
    """Override in a plugin conftest to mount extra services module-wide.

    Example::

        @pytest.fixture(scope="module")
        def _igw_extra_services() -> tuple[ServiceFactory, ...]:
            from nmp.guardrails.service import GuardrailsService

            return (GuardrailsService,)

    Module-scoped because the app it feeds is module-scoped.
    """
    return ()


@pytest.fixture(scope="module")
def _igw_app_context(
    _igw_extra_services: tuple[ServiceFactory, ...],
) -> Generator[ClientContext, None, None]:
    """Module-scoped IGW + Models ASGI stack.

    The expensive ``create_test_client`` call runs once per test file;
    function-scoped fixtures layer per-test concerns on top. Extra
    services come from :func:`_igw_extra_services` rather than fixture
    parameters so plugin conftests can declare their needs without
    rebuilding the app per-test.
    """
    with _build_app_context(*_igw_extra_services) as client_context:
        yield client_context


@pytest.fixture(scope="module")
def _igw_loopback_context(
    _igw_app_context: ClientContext,
) -> Generator[str, None, None]:
    """Run the module app on a real ``127.0.0.1:<port>`` and yield its URL.

    Only entered when a test actually requests :func:`igw_loopback_harness`
    — plain modules pay nothing for uvicorn.

    The per-request HTTP client override and the ``get_platform_config``
    patch live in :func:`_build_loopback_harness` instead, so plain
    ``igw_plugin_harness`` tests in a mixed module don't pay loopback's
    per-request session cost. Uvicorn is started with ``lifespan="off"``;
    the TestClient still owns startup/shutdown.
    """
    from nmp.core.inference_gateway.testing._loopback import serve_app_in_thread

    with serve_app_in_thread(_app_from(_igw_app_context)) as loopback_base_url:
        yield loopback_base_url


@contextmanager
def _per_test_plugin_setup(
    client_context: ClientContext,
    httpserver: HTTPServer,
) -> Generator[IGWPluginHarness, None, None]:
    """Per-test setup/teardown shared by the plain + loopback fixtures.

    Resets the post-response task list, builds a fresh harness on the
    per-test ``pytest_httpserver`` socket, and runs
    :meth:`IGWPluginHarness._cleanup` on teardown to delete this test's
    entities and rebuild the in-memory caches.

    Note we deliberately don't call the original ``reset_global_*``
    helpers between tests — the module-scoped app's ``on_startup`` is
    the only thing that re-initialises those globals, so nulling them
    would crash the next request.
    """
    _enable_post_response_task_tracking(client_context)
    harness = IGWPluginHarness._build(client_context=client_context, mock_nim=httpserver)
    try:
        yield harness
    finally:
        harness._cleanup()


@pytest.fixture
def igw_plugin_harness(
    _igw_app_context: ClientContext,
    httpserver: HTTPServer,
) -> Generator[IGWPluginHarness, None, None]:
    """Per-test IGW + Models harness — no real port, mock NIM only.

    Cheap: the heavy ASGI stack comes from the module-scoped
    :func:`_igw_app_context`. Per test you only pay for the harness
    construction, the post-response list reset, and the function-scoped
    mock-NIM socket.
    """
    with _per_test_plugin_setup(_igw_app_context, httpserver) as harness:
        yield harness


@contextmanager
def _build_loopback_harness(
    client_context: ClientContext,
    httpserver: HTTPServer,
    igw_loopback_base_url: str,
    *extra_services: ServiceFactory,
) -> Generator[IGWLoopbackHarness, None, None]:
    """Per-test setup/teardown for the loopback harness.

    Same per-test resets as :func:`_per_test_plugin_setup`, plus two
    loopback-only patches scoped to this test:

    * ``per_request_http_client`` overrides :func:`global_http_client` —
      loopback runs the app from two loops (TestClient + uvicorn) and
      a singleton :class:`aiohttp.ClientSession` would be loop-bound to
      whichever one created it.
    * ``get_platform_config`` is patched at IGW's middleware-registry
      import site so :meth:`get_openai_compatible_inference_url_and_model`
      returns URLs reachable from the test process.

    Both patches roll back before the next test runs, so a plain
    ``igw_plugin_harness`` test sharing the module doesn't observe them.

    Raises:
        TypeError: If *extra_services* is non-empty. The module-scoped
            app is already built by the time this runs; the previous
            ``igw_loopback_harness(GuardrailsService)`` pattern is dead.
            Override :func:`_igw_extra_services` in your conftest
            instead. (Hard error, not a warning, because pytest hides
            ``DeprecationWarning`` by default and silently-missing
            routes would be much harder to diagnose.)
    """
    if extra_services:
        names = ", ".join(getattr(s, "__name__", repr(s)) for s in extra_services)
        raise TypeError(
            f"igw_loopback_harness({names}): extra service args are no longer "
            "accepted under module-scoped fixtures. Override "
            "`_igw_extra_services` in your conftest to mount additional "
            "services for the whole module."
        )
    from nmp.core.inference_gateway.api.dependencies import global_http_client
    from nmp.core.inference_gateway.testing._loopback import (
        override_platform_base_url,
        per_request_http_client,
    )

    _enable_post_response_task_tracking(client_context)

    app = _app_from(client_context)

    with ExitStack() as stack:
        previous_override = app.dependency_overrides.get(global_http_client)
        app.dependency_overrides[global_http_client] = per_request_http_client

        def _restore_http_client_override() -> None:
            if previous_override is None:
                app.dependency_overrides.pop(global_http_client, None)
            else:
                app.dependency_overrides[global_http_client] = previous_override

        stack.callback(_restore_http_client_override)
        stack.enter_context(override_platform_base_url(igw_loopback_base_url))

        harness = cast(
            IGWLoopbackHarness,
            IGWLoopbackHarness._build(
                client_context=client_context,
                mock_nim=httpserver,
                igw_loopback_base_url=igw_loopback_base_url,
            ),
        )
        try:
            yield harness
        finally:
            harness._cleanup()


@pytest.fixture
def igw_loopback_harness(
    _igw_app_context: ClientContext,
    _igw_loopback_context: str,
    httpserver: HTTPServer,
) -> Generator[Callable[..., IGWLoopbackHarness], None, None]:
    """Factory for an IGW loopback harness — call ``igw_loopback_harness()``.

    Passing extra service classes raises :class:`TypeError`. Mount extra
    services by overriding :func:`_igw_extra_services` in the plugin's
    conftest.
    """
    with ExitStack() as stack:

        def factory(*extra_services: ServiceFactory) -> IGWLoopbackHarness:
            return stack.enter_context(
                _build_loopback_harness(
                    _igw_app_context,
                    httpserver,
                    _igw_loopback_context,
                    *extra_services,
                )
            )

        yield factory


__all__ = [
    "_igw_app_context",
    "_igw_extra_services",
    "_igw_loopback_context",
    "igw_loopback_harness",
    "igw_plugin_harness",
]
