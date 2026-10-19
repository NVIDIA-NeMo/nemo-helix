# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pytest
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceQueryParams, CreateWorkspaceRequest

from e2e.services_pool import E2EHarnessConfig, E2EServicesPool, RunningServices
from tests.auth_idp.authentik_live import authentik_gateway_tls_ca_bundle, prepare_authentik_compose_inputs
from tests.auth_idp.deployment_context import load_credential_environment, load_deployment_context
from tests.auth_idp.providers import ProviderConfig, load_provider_config
from tests.auth_idp.runtime import get_authentik_docker_test_runtime
from tests.auth_idp.runtime_contract import AuthIdpCase, AuthIdpRuntime
from tests.auth_idp.runtime_factory import iter_auth_idp_cases, parametrize_cases, runtime_class_for_case
from tests.auth_idp.runtime_host import HostAuthIdpRuntime
from tests.auth_idp.token_acquisition import exchange_token_with_retries, token_request_auth, token_request_body

PYTEST_TIMEOUT_SECONDS = 2400
pytest_plugins = ("e2e.conftest", "tests.auth_idp.browser_fixtures")

_exchange_token_with_retries = exchange_token_with_retries
_token_request_auth = token_request_auth
_token_request_body = token_request_body


def _auth_idp_case_for_item(item: pytest.Item) -> AuthIdpCase | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    case = callspec.params.get("auth_idp_case")
    if isinstance(case, AuthIdpCase):
        return case
    return None


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("auth-idp")
    group.addoption(
        "--auth-idp-context",
        action="store",
        default=None,
        help="Run contracts on the host against an immutable provider deployment context.",
    )
    group.addoption(
        "--auth-idp-provider",
        action="store",
        default=None,
        help="Run parametrized auth-idp contract tests only for the named provider.",
    )
    group.addoption(
        "--auth-idp-backend",
        action="store",
        choices=("compose", "kubernetes", "external"),
        default=None,
        help="Run parametrized auth-idp contract tests only for the selected backend.",
    )
    group.addoption(
        "--auth-idp-runtime",
        action="store",
        default=None,
        help="Run parametrized auth-idp contract tests only for the selected runtime id.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "auth_idp_case" not in metafunc.fixturenames:
        return
    context_path = metafunc.config.getoption("--auth-idp-context")
    if context_path:
        context = load_deployment_context(Path(context_path))
        load_credential_environment(context.credential_env_files)
        provider = load_provider_config(context.provider_manifest)
        case = AuthIdpCase(
            id=context.runtime_id,
            provider=provider,
            backend=context.backend,
            capabilities=frozenset(context.capabilities),
        )
        metafunc.parametrize("auth_idp_case", [pytest.param(case, id=case.id)], scope="session")
        return
    backend = metafunc.config.getoption("--auth-idp-backend")
    if backend is None and metafunc.definition.get_closest_marker("auth_idp_k8s") is not None:
        backend = "kubernetes"
    if backend is None and metafunc.definition.get_closest_marker("auth_idp_docker") is not None:
        backend = "compose"
    cases = iter_auth_idp_cases(
        backend=backend,
        provider_name=metafunc.config.getoption("--auth-idp-provider"),
        runtime_id=metafunc.config.getoption("--auth-idp-runtime"),
    )
    metafunc.parametrize("auth_idp_case", parametrize_cases(cases), scope="session")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    runtime_id = config.getoption("--auth-idp-runtime")
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        runtime_markers = list(item.iter_markers("auth_idp_runtime"))
        if runtime_markers:
            item.add_marker(pytest.mark.xdist_group("idp-live"))
        case = _auth_idp_case_for_item(item)
        if case is not None and case.backend == "kubernetes" and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(PYTEST_TIMEOUT_SECONDS))
        if not runtime_markers:
            selected.append(item)
            continue
        if not runtime_id:
            selected.append(item)
            continue
        runtime_marker_args = {str(argument) for marker in runtime_markers for argument in marker.args}
        if not runtime_marker_args or runtime_id in runtime_marker_args:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _compose_e2e_config_for_case(
    auth_idp_case: AuthIdpCase,
) -> tuple[tuple[str | dict[str, Any], ...], E2EHarnessConfig]:
    provider_module_name = auth_idp_case.provider.name.replace("-", "_")
    marker_name = f"{provider_module_name.upper()}_DOCKER_E2E_CONFIG"
    module = importlib.import_module(f"tests.auth_idp.{provider_module_name}_live")
    marker_decorator = getattr(module, marker_name, None)
    if marker_decorator is None:
        raise ValueError(
            f"compose auth-idp runtime {auth_idp_case.id!r} needs tests.auth_idp."
            f"{provider_module_name}_live.{marker_name}"
        )
    marker = marker_decorator.mark
    config_layers = cast(tuple[str | dict[str, Any], ...], marker.args)
    harness_config = cast(E2EHarnessConfig, dict(marker.kwargs.get("harness") or {}))
    lifecycle = os.environ.get("NHX_E2E_COMPOSE_LIFECYCLE")
    if lifecycle == "fresh":
        harness_config["lifecycle"] = "fresh"
    elif lifecycle == "reuse":
        harness_config["lifecycle"] = "reuse"
    elif lifecycle:
        raise ValueError("NHX_E2E_COMPOSE_LIFECYCLE must be 'fresh' or 'reuse'")
    compose_project_name = os.environ.get("NHX_AUTHENTIK_COMPOSE_PROJECT_NAME")
    if compose_project_name:
        harness_config["compose_project_name"] = compose_project_name
    gateway_port = os.environ.get("NHX_AUTHENTIK_COMPOSE_GATEWAY_PORT")
    if gateway_port:
        dynamic_ports = dict(harness_config.get("dynamic_ports") or {})
        gateway_config = dict(dynamic_ports.get("gateway") or {})
        gateway_config["port"] = gateway_port
        dynamic_ports["gateway"] = gateway_config
        harness_config["dynamic_ports"] = dynamic_ports
    return config_layers, harness_config


def _url_port(url: str) -> str:
    port = urlparse(url).port
    return str(port) if port is not None else "default"


def _auth_idp_runtime_event_line(
    event: str,
    auth_idp_case: AuthIdpCase,
    runtime: Any | None = None,
    services: RunningServices | None = None,
    status: str | None = None,
) -> str:
    gateway_base_url = str(getattr(runtime, "gateway_base_url", "")) if runtime is not None else ""
    parts = [
        f"Auth-idp runtime {event}:",
        f"id={auth_idp_case.id}",
        f"backend={auth_idp_case.backend}",
    ]
    if status is not None:
        parts.append(f"status={status}")
    if gateway_base_url:
        parts.extend((f"url={gateway_base_url}", f"port={_url_port(gateway_base_url)}"))
    if services is not None:
        if services.compose_project_name is not None:
            parts.append(f"compose_project={services.compose_project_name}")
        if services.log_path is not None:
            parts.append(f"log={services.log_path}")
    cluster = getattr(runtime, "cluster", None) if runtime is not None else None
    if cluster is not None:
        parts.extend(
            (
                f"cluster={cluster.name}",
                f"context={cluster.context}",
                f"runtime={cluster.runtime}",
            )
        )
        kubeconfig = getattr(cluster, "kubeconfig", None)
        if kubeconfig is not None:
            parts.append(f"kubeconfig={kubeconfig}")
    namespace = getattr(runtime, "namespace", None) if runtime is not None else None
    if namespace is not None:
        parts.append(f"namespace={namespace}")
    helm_release = getattr(runtime, "helm_release", None) if runtime is not None else None
    if helm_release is not None:
        parts.append(f"helm_release={helm_release}")
    return " ".join(parts)


def _write_terminal_line(request: pytest.FixtureRequest, message: str) -> None:
    capture_manager = request.config.pluginmanager.get_plugin("capturemanager")
    if capture_manager is None:
        print(message, flush=True)
        return
    with capture_manager.global_and_fixture_disabled():
        print(message, flush=True)


@pytest.fixture(scope="session")
def idp_e2e_enabled(pytestconfig: pytest.Config) -> bool:
    return bool(
        pytestconfig.getoption("--run-e2e")
        or pytestconfig.getoption("--auth-idp-runtime")
        or pytestconfig.getoption("--auth-idp-context")
    )


@pytest.fixture(scope="session")
def require_idp_e2e(idp_e2e_enabled: bool) -> Iterator[None]:
    if not idp_e2e_enabled:
        pytest.skip("set --auth-idp-runtime or --run-e2e to execute provider stack validation")
    yield


@pytest.fixture(scope="session", autouse=True)
def _prepare_authentik_compose_inputs_for_e2e(idp_e2e_enabled: bool, pytestconfig: pytest.Config) -> None:
    if idp_e2e_enabled and not pytestconfig.getoption("--auth-idp-context"):
        prepare_authentik_compose_inputs()
        os.environ.setdefault(
            NHX_CLIENT_SSL_CERT_FILE_ENVVAR,
            str(authentik_gateway_tls_ca_bundle()),
        )


@pytest.fixture(scope="session")
def auth_idp_runtime(
    auth_idp_case: AuthIdpCase,
    require_idp_e2e: None,
    request: pytest.FixtureRequest,
    _services_pool_manager: E2EServicesPool,
) -> Iterator[AuthIdpRuntime]:
    context_path = request.config.getoption("--auth-idp-context")
    if context_path:
        runtime = HostAuthIdpRuntime(load_deployment_context(Path(context_path)))
        runtime.assert_available()
        yield runtime
        runtime.assert_available()
        return
    services: RunningServices | None = None
    owner_id: str | None = None
    runtime: AuthIdpRuntime | None = None
    failures_before = request.session.testsfailed
    _write_terminal_line(request, _auth_idp_runtime_event_line("starting", auth_idp_case))
    try:
        runtime_class = runtime_class_for_case(auth_idp_case)
    except ValueError as exc:
        pytest.skip(str(exc))
    try:
        if auth_idp_case.backend == "compose":
            try:
                config_layers, harness_config = _compose_e2e_config_for_case(auth_idp_case)
            except ValueError as exc:
                pytest.skip(str(exc))
            owner_id = f"auth-idp-runtime::{auth_idp_case.id}"
            services = _services_pool_manager.acquire_for_config(owner_id, config_layers, harness_config)
            if services.log_path is not None:
                from e2e.conftest import _services_log_key

                request.session.stash[_services_log_key] = services.log_path
            runtime = runtime_class(
                auth_idp_case,
                services.url,
                cleanup=lambda: _services_pool_manager.release_for_config(owner_id),
            )
        elif auth_idp_case.backend == "kubernetes":
            runtime_class = runtime_class_for_case(auth_idp_case)
            runtime = runtime_class(auth_idp_case)
        else:
            runtime_class = runtime_class_for_case(auth_idp_case)
            runtime = runtime_class(auth_idp_case)
    except Exception:
        if owner_id is not None and runtime is None:
            _services_pool_manager.release_for_config(owner_id)
        _write_terminal_line(request, _auth_idp_runtime_event_line("startup_failed", auth_idp_case, runtime, services))
        raise
    assert runtime is not None
    _write_terminal_line(request, _auth_idp_runtime_event_line("available", auth_idp_case, runtime, services))
    try:
        yield runtime
    finally:
        status = "passed" if request.session.testsfailed == failures_before else "failed"
        _write_terminal_line(request, _auth_idp_runtime_event_line("result", auth_idp_case, runtime, services, status))
        _write_terminal_line(
            request, _auth_idp_runtime_event_line("teardown_starting", auth_idp_case, runtime, services)
        )
        try:
            runtime.cleanup()
        except Exception:
            _write_terminal_line(
                request,
                _auth_idp_runtime_event_line("teardown_failed", auth_idp_case, runtime, services),
            )
            raise
        _write_terminal_line(
            request, _auth_idp_runtime_event_line("teardown_complete", auth_idp_case, runtime, services)
        )


@pytest.fixture(autouse=True)
def _assert_host_runtime_available(request: pytest.FixtureRequest) -> Iterator[None]:
    if not request.config.getoption("--auth-idp-context"):
        yield
        return
    runtime = request.getfixturevalue("auth_idp_runtime")
    assert isinstance(runtime, HostAuthIdpRuntime)
    runtime.assert_available()
    yield
    runtime.assert_available()


@pytest.fixture
def auth_idp_workspace(auth_idp_runtime: AuthIdpRuntime) -> Iterator[str]:
    workspace_name = f"auth-idp-ws-{uuid.uuid4().hex[:8]}"
    workspaces = WorkspacesClient.from_client(auth_idp_runtime.e2e_setup_client())
    workspaces.create_workspace(
        query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
        body=CreateWorkspaceRequest(name=workspace_name, description="Workspace for auth-idp provider contract tests"),
    ).data()
    try:
        yield workspace_name
    finally:
        workspaces.delete_workspace(name=workspace_name).data()


@pytest.fixture(scope="module")
def authentik_provider(authentik_docker_runtime: ProviderConfig) -> ProviderConfig:
    provider = authentik_docker_runtime
    assert provider.token_endpoint is not None
    assert provider.e2e_setup_password_grant is not None
    assert provider.workload_provider_password_grant is not None
    return provider


@pytest.fixture(scope="session")
def authentik_docker_runtime() -> ProviderConfig:
    provider = get_authentik_docker_test_runtime()
    assert provider.token_endpoint is not None
    assert provider.e2e_setup_password_grant is not None
    assert provider.workload_provider_password_grant is not None
    return provider


@pytest.fixture(scope="module")
def authentik_stack(
    require_idp_e2e: None,
    authentik_provider: ProviderConfig,
    _services: str,
) -> ProviderConfig:
    return replace(
        authentik_provider,
        gateway_base_url=_services,
        discovery_url=f"{_services}/application/o/nemo/.well-known/openid-configuration",
        token_endpoint=f"{_services}/application/o/token/",
    )


@pytest.fixture(scope="module")
def e2e_setup_token(authentik_stack: ProviderConfig) -> str:
    grant = authentik_stack.e2e_setup_password_grant
    assert grant is not None
    assert authentik_stack.token_endpoint is not None
    return _exchange_token_with_retries(authentik_stack.token_endpoint, grant)


@pytest.fixture(scope="module")
def workload_provider_token(authentik_stack: ProviderConfig) -> str:
    grant = authentik_stack.workload_provider_password_grant
    assert grant is not None
    assert authentik_stack.token_endpoint is not None
    return _exchange_token_with_retries(authentik_stack.token_endpoint, grant)


@pytest.fixture(scope="module")
def interactive_user_token(authentik_stack: ProviderConfig) -> str:
    grant = authentik_stack.interactive_user_password_grant
    assert grant is not None
    assert authentik_stack.token_endpoint is not None
    return _exchange_token_with_retries(authentik_stack.token_endpoint, grant)


@pytest.fixture(scope="module")
def authentik_e2e_setup_client(authentik_stack: ProviderConfig, e2e_setup_token: str) -> NemoClient:
    return NemoClient(base_url=authentik_stack.gateway_base_url, auth=e2e_setup_token)


@pytest.fixture(scope="module")
def authentik_interactive_user_client(authentik_stack: ProviderConfig, interactive_user_token: str) -> NemoClient:
    return NemoClient(base_url=authentik_stack.gateway_base_url, auth=interactive_user_token)


@pytest.fixture(scope="module")
def workload_provider_client(authentik_stack: ProviderConfig, workload_provider_token: str) -> NemoClient:
    return NemoClient(base_url=authentik_stack.gateway_base_url, auth=workload_provider_token)


@pytest.fixture
def authentik_workspace(authentik_e2e_setup_client: NemoClient) -> Iterator[str]:
    workspaces = WorkspacesClient.from_client(authentik_e2e_setup_client)
    workspace_name = f"authentik-ws-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(
        query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
        body=CreateWorkspaceRequest(name=workspace_name, description="Workspace for Authentik live auth tests"),
    ).data()
    try:
        yield workspace_name
    finally:
        workspaces.delete_workspace(name=workspace_name).data()
