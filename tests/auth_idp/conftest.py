# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast
from urllib.parse import urlparse

import httpx
import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_ext.client.tls import NMP_CLIENT_SSL_CERT_FILE_ENVVAR, HttpxTLSConfig, httpx_tls_config_from_env
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceQueryParams, CreateWorkspaceRequest

from e2e.services_pool import E2EHarnessConfig, E2EServicesPool, RunningServices
from tests.auth_idp.authentik_live import authentik_gateway_tls_ca_bundle, prepare_authentik_compose_inputs
from tests.auth_idp.providers import ProviderConfig
from tests.auth_idp.runtime import get_authentik_docker_test_runtime
from tests.auth_idp.runtime_contract import AuthIdpCase
from tests.auth_idp.runtime_factory import iter_auth_idp_cases, parametrize_cases, runtime_class_for_case

pytest_plugins = ("e2e.conftest",)


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
            from tests.auth_idp.runtime_kubernetes import PYTEST_TIMEOUT_SECONDS

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


def _token_request_body(grant: dict[str, str]) -> dict[str, str]:
    grant_type = grant["grant_type"]
    body = {
        "grant_type": grant_type,
        "client_id": grant["client_id"],
    }
    if "client_secret" in grant and grant.get("client_auth_method") != "client_secret_basic":
        body["client_secret"] = grant["client_secret"]
    if grant_type == "password":
        body["username"] = grant["username"]
        body["password"] = grant["password"]
        if "scope" in grant:
            body["scope"] = grant["scope"]
        return body
    if grant_type == "client_credentials":
        if "scope" in grant:
            body["scope"] = grant["scope"]
        return body
    raise ValueError(f"unsupported grant_type for auth_idp token exchange: {grant_type}")


def _token_request_auth(grant: dict[str, str]) -> tuple[str, str] | None:
    if grant.get("client_auth_method") != "client_secret_basic":
        return None
    client_secret = grant.get("client_secret")
    if not client_secret:
        raise AssertionError("client_secret_basic token acquisition requires client_secret")
    return grant["client_id"], client_secret


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
    lifecycle = os.environ.get("NMP_E2E_COMPOSE_LIFECYCLE")
    if lifecycle:
        harness_config["lifecycle"] = cast(Any, lifecycle)
    compose_project_name = os.environ.get("NMP_AUTHENTIK_COMPOSE_PROJECT_NAME")
    if compose_project_name:
        harness_config["compose_project_name"] = compose_project_name
    gateway_port = os.environ.get("NMP_AUTHENTIK_COMPOSE_GATEWAY_PORT")
    if gateway_port:
        dynamic_ports = dict(harness_config.get("dynamic_ports") or {})
        gateway_config = dict(dynamic_ports.get("gateway") or {})
        gateway_config["port"] = gateway_port
        dynamic_ports["gateway"] = gateway_config
        harness_config["dynamic_ports"] = dynamic_ports
    return config_layers, harness_config


def _exchange_token_with_retries(
    token_endpoint: str,
    grant: dict[str, str],
    timeout: float = 60.0,
    tls_config: HttpxTLSConfig | None = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    request_tls_config: HttpxTLSConfig = httpx_tls_config_from_env() if tls_config is None else tls_config
    while time.monotonic() < deadline:
        try:
            response = httpx.post(
                token_endpoint,
                data=_token_request_body(grant),
                auth=_token_request_auth(grant),
                timeout=30.0,
                **request_tls_config,
            )
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"token endpoint not ready: {response.status_code}",
                    request=response.request,
                    response=response,
                )
                time.sleep(2)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise httpx.HTTPStatusError(
                    f"{exc}; response body: {response.text}",
                    request=response.request,
                    response=response,
                ) from exc
            return response.json()["access_token"]
        except httpx.RequestError as exc:
            last_error = exc
            time.sleep(2)
    if last_error is not None:
        raise last_error
    raise TimeoutError(f"token endpoint did not become ready: {token_endpoint}")


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
    return bool(pytestconfig.getoption("--run-e2e") or pytestconfig.getoption("--auth-idp-runtime"))


@pytest.fixture(scope="session")
def require_idp_e2e(idp_e2e_enabled: bool) -> Iterator[None]:
    if not idp_e2e_enabled:
        pytest.skip("set --auth-idp-runtime or --run-e2e to execute provider stack validation")
    yield


@pytest.fixture(scope="session", autouse=True)
def _prepare_authentik_compose_inputs_for_e2e(idp_e2e_enabled: bool) -> None:
    if idp_e2e_enabled:
        prepare_authentik_compose_inputs()
        os.environ.setdefault(
            NMP_CLIENT_SSL_CERT_FILE_ENVVAR,
            str(authentik_gateway_tls_ca_bundle()),
        )


@pytest.fixture(scope="session")
def auth_idp_runtime(
    auth_idp_case: AuthIdpCase,
    require_idp_e2e: None,
    request: pytest.FixtureRequest,
    _services_pool_manager: E2EServicesPool,
):
    services: RunningServices | None = None
    owner_id: str | None = None
    runtime: Any | None = None
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
            runtime = runtime_class(auth_idp_case)
        else:
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


@pytest.fixture
def auth_idp_workspace(auth_idp_runtime) -> Iterator[str]:
    workspace_name = f"auth-idp-ws-{uuid.uuid4().hex[:8]}"
    sdk = auth_idp_runtime.e2e_setup_sdk()
    workspaces = client_from_platform(sdk, WorkspacesClient)
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
def authentik_e2e_setup_sdk(authentik_stack: ProviderConfig, e2e_setup_token: str) -> NeMoPlatform:
    return NeMoPlatform(
        base_url=authentik_stack.gateway_base_url,
        default_headers={"Authorization": f"Bearer {e2e_setup_token}"},
        max_retries=0,
    )


@pytest.fixture(scope="module")
def authentik_interactive_user_sdk(authentik_stack: ProviderConfig, interactive_user_token: str) -> NeMoPlatform:
    return NeMoPlatform(
        base_url=authentik_stack.gateway_base_url,
        default_headers={"Authorization": f"Bearer {interactive_user_token}"},
        max_retries=0,
    )


@pytest.fixture(scope="module")
def workload_provider_sdk(authentik_stack: ProviderConfig, workload_provider_token: str) -> NeMoPlatform:
    return NeMoPlatform(
        base_url=authentik_stack.gateway_base_url,
        default_headers={"Authorization": f"Bearer {workload_provider_token}"},
        max_retries=0,
    )


@pytest.fixture
def authentik_workspace(authentik_e2e_setup_sdk: NeMoPlatform) -> Iterator[str]:
    workspaces = client_from_platform(authentik_e2e_setup_sdk, WorkspacesClient)
    workspace_name = f"authentik-ws-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(
        query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
        body=CreateWorkspaceRequest(name=workspace_name, description="Workspace for Authentik live auth tests"),
    ).data()
    try:
        yield workspace_name
    finally:
        workspaces.delete_workspace(name=workspace_name).data()
