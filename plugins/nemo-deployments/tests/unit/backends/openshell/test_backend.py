# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked-stub unit tests for OpenShellDeploymentBackend."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import httpx
import pytest
from nemo_deployments_plugin.backends.labels import (
    CONFIG_NAME_LABEL,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
)
from nemo_deployments_plugin.backends.openshell.backend import (
    _CONFIG_DELIVERED_MARKER,
    _CURL_ABS,
    _DEFAULT_READINESS_TIMEOUT_SECONDS,
    _LAUNCH_MARKER,
    _LIVENESS_PROBE,
    _MAX_ROUTABLE_NAME_LEN,
    _READINESS_EXEC_TIMEOUT_MARGIN_SECONDS,
    _SERVE_DEAD_EXIT,
    _SERVE_PENDING_EXIT,
    _SERVE_PID_GRACE_SECONDS,
    _SERVE_PIDFILE,
    OpenShellDeploymentBackend,
    _delivery_script,
    _readiness_probe_command,
    _sandbox_name,
    _service_name,
)
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    ContainerPort,
    DeploymentConfig,
    EnvVar,
    ExecAction,
    HTTPGetAction,
    Probe,
    TCPSocketAction,
    WorkloadIdentitySpec,
)
from nemo_deployments_plugin.secrets import SecretResolutionError
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

pytest.importorskip("openshell")  # platform-restricted extra; skip where not installed (e.g. CI)

from openshell._proto import openshell_pb2 as pb  # ty: ignore[unresolved-import]


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "boom") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


def _not_found() -> FakeRpcError:
    return FakeRpcError(grpc.StatusCode.NOT_FOUND, "missing")


def _sandbox(
    phase: int, *, sandbox_id: str = "sid", labels: dict | None = None, config_name: str = "cfg1"
) -> MagicMock:
    lbls = dict(labels or {})
    if config_name:
        lbls[CONFIG_NAME_LABEL] = config_name
    return MagicMock(
        sandbox=MagicMock(
            metadata=MagicMock(id=sandbox_id, labels=lbls),
            status=MagicMock(phase=phase, conditions=[]),
        )
    )


def _config(
    *,
    command: tuple[str, ...] = ("python3", "-m", "http.server"),
    args: tuple[str, ...] = ("8000",),
    with_port: bool = True,
) -> DeploymentConfig:
    ports = [ContainerPort(containerPort=8000, name="http")] if with_port else []
    return DeploymentConfig(
        name="cfg1",
        workspace="default",
        containers=[Container(name="web", image="img:latest", command=list(command), args=list(args), ports=ports)],
    )


def _config_with_workload_identity() -> DeploymentConfig:
    return _config().model_copy(
        update={
            "workload_identity": WorkloadIdentitySpec(
                enabled=True,
                workloadKind="agent_deployment",
                workloadId="srv",
            )
        }
    )


def _exec_events(exit_code: int, *, stdout: str = "", stderr: str = "") -> list[MagicMock]:
    """Build a realistic ExecSandbox event stream: optional stdout/stderr then an exit event."""
    events: list[MagicMock] = []
    if stdout:
        out = MagicMock()
        out.HasField.side_effect = lambda field: field == "stdout"
        out.stdout.data = stdout.encode()
        events.append(out)
    if stderr:
        err = MagicMock()
        err.HasField.side_effect = lambda field: field == "stderr"
        err.stderr.data = stderr.encode()
        events.append(err)
    exit_event = MagicMock()
    exit_event.HasField.side_effect = lambda field: field == "exit"
    exit_event.exit.exit_code = exit_code
    events.append(exit_event)
    return events


def _delivery_ok() -> list[MagicMock]:
    """A successful config-delivery exec: prints the completion marker on stdout, exits 0."""
    return _exec_events(0, stdout=_CONFIG_DELIVERED_MARKER)


def _stream_without_exit(stdout: str = "") -> list[MagicMock]:
    """An ExecSandbox stream that never yields an exit event (drains to exit_code None)."""
    events: list[MagicMock] = []
    if stdout:
        out = MagicMock()
        out.HasField.side_effect = lambda field: field == "stdout"
        out.stdout.data = stdout.encode()
        events.append(out)
    return events


def _config_with_readiness(probe: Probe) -> DeploymentConfig:
    cfg = _config()
    cfg.containers[0].readiness_probe = probe
    return cfg


def _config_with_env(env: list[EnvVar]) -> DeploymentConfig:
    return DeploymentConfig(
        name="cfg1",
        workspace="default",
        containers=[
            Container(
                name="web",
                image="img:latest",
                command=["python3", "-m", "http.server"],
                args=["8000"],
                ports=[ContainerPort(containerPort=8000, name="http")],
                env=env,
            )
        ],
    )


def _config_with_config_files(config_files: list[ConfigFile]) -> DeploymentConfig:
    return DeploymentConfig(
        name="cfg1",
        workspace="default",
        containers=[
            Container(
                name="web",
                image="img:latest",
                command=["python3", "-m", "http.server"],
                args=["8000"],
                ports=[ContainerPort(containerPort=8000, name="http")],
            )
        ],
        configFiles=config_files,
    )


def _exec_requests(mock_stub: MagicMock) -> list[pb.ExecSandboxRequest]:
    """Every ExecSandboxRequest the backend sent, in call order."""
    return [call.args[0] for call in mock_stub.ExecSandbox.call_args_list]


def _delivery_requests(mock_stub: MagicMock) -> list[pb.ExecSandboxRequest]:
    """The subset of exec calls that are config-file writes (``cat >`` scripts)."""
    return [req for req in _exec_requests(mock_stub) if any("cat >" in part for part in req.command)]


def test_registry_contains_openshell() -> None:
    assert BACKEND_CLASSES["openshell"] is OpenShellDeploymentBackend


async def test_load_deployment_config_wraps_an_entities_client_that_accepts_query_params() -> None:
    """init() must adapt the SDK with client_from_platform(AsyncEntitiesClient), not wrap the
    raw generated AsyncEntitiesResource (AIRCORE-977).

    NemoEntitiesClient.get() forwards a ``query_params`` kwarg. The generated resource does not
    accept it, so wrapping the resource made every first reconcile die with
    ``TypeError: ... unexpected keyword argument 'query_params'`` before any request left the box.
    Drive the real contract with a live entities client over a mock transport: a 404 must surface
    as NemoEntityNotFoundError, which is only reachable once ``get_entity_by_name(query_params=...)``
    is accepted and the request actually goes out.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    sdk = AsyncNeMoHelix(
        base_url="http://entities.test",
        workspace="default",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with patch("grpc.insecure_channel", return_value=MagicMock()):
        backend = OpenShellDeploymentBackend(sdk, {"gateway_endpoint": "http://127.0.0.1:17670"})

    # Old (buggy) wrapping raised TypeError about query_params here; the fix reaches the 404.
    with pytest.raises(NemoEntityNotFoundError):
        await backend._load_deployment_config("default", "missing-config")


def test_sandbox_name_within_limit_and_deterministic() -> None:
    name = _sandbox_name("a-long-workspace-name", "a-long-deployment-name")
    assert name.startswith("nhx-")
    assert len(name) <= _MAX_ROUTABLE_NAME_LEN
    assert name == _sandbox_name("a-long-workspace-name", "a-long-deployment-name")
    assert name != _sandbox_name("other", "name")


def test_service_name_clamped_and_distinct() -> None:
    short = _service_name(ContainerPort(containerPort=8000, name="http"))
    assert short == "http"

    unnamed = _service_name(ContainerPort(containerPort=8000))
    assert unnamed == "port-8000"

    long_a = _service_name(ContainerPort(containerPort=8000, name="a-very-long-port-name-one"))
    long_b = _service_name(ContainerPort(containerPort=8001, name="a-very-long-port-name-two"))
    assert len(long_a) <= _MAX_ROUTABLE_NAME_LEN
    assert not long_a.endswith("-")
    assert long_a != long_b, "truncation must not collapse distinct port names onto one service"


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        # READY is not here: a READY sandbox triggers the read_status provisioning state
        # machine (launch/expose), covered by the read_status provisioning tests below.
        (pb.SANDBOX_PHASE_PROVISIONING, "STARTING"),
        (pb.SANDBOX_PHASE_ERROR, "FAILED"),
        (pb.SANDBOX_PHASE_DELETING, "DELETING"),
    ],
)
async def test_read_status_maps_phase(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, phase: int, expected: str
) -> None:
    mock_stub.GetSandbox.return_value = _sandbox(phase)
    update = await openshell_backend.read_status(workspace="default", name="srv")
    assert update.status == expected


async def test_read_status_includes_exposed_endpoints(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    svc = MagicMock(url="http://nhx-x--http.openshell.localhost:18080/")
    svc.endpoint.service_name = "http"
    mock_stub.ListServices.return_value = MagicMock(services=[svc])

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"
    assert [e.url for e in update.endpoints] == ["http://nhx-x--http.openshell.localhost:18080/"]
    assert update.endpoints[0].name == "http"


async def test_read_status_not_found_is_lost(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    mock_stub.GetSandbox.side_effect = _not_found()
    update = await openshell_backend.read_status(workspace="default", name="srv")
    assert update.status == "LOST"


async def test_create_uses_executor_policy_when_no_policy_path_override(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    executor_policy = openshell_backend._policy
    await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert spec.policy == executor_policy


async def test_create_uses_per_deployment_policy_when_policy_path_set(
    openshell_backend: OpenShellDeploymentBackend,
    mock_stub: MagicMock,
    mock_entities: AsyncMock,
    tmp_path: Path,
) -> None:
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("version: 1\nprocess:\n  run_as_user: sandbox\n  run_as_group: sandbox\n")
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    executor_policy = openshell_backend._policy
    await openshell_backend.create_deployment(
        workspace="default",
        name="srv",
        config_name="cfg1",
        labels={},
        backend_config={"openshell": {"policyPath": str(policy_file)}},
    )

    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert spec.policy != executor_policy


async def test_create_requires_serve_command(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_stub.GetSandbox.side_effect = _not_found()  # not already present
    mock_entities.get.return_value = _config(command=(), args=())
    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )
    assert update.status == "FAILED"
    assert "command" in update.status_message


async def test_create_returns_starting_after_create(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Create returns STARTING right after CreateSandbox; provisioning (launch/expose) is
    # deferred to read_status so a slow sandbox never blocks the reconcile loop.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()  # idempotency probe: absent
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={"managed-by": MANAGED_BY_LABEL}, backend_config={}
    )

    assert update.status == "STARTING"
    mock_stub.CreateSandbox.assert_called_once()
    mock_stub.ExecSandbox.assert_not_called()
    mock_stub.ExposeService.assert_not_called()


async def test_create_fails_when_workload_identity_requested(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_workload_identity()
    mock_stub.GetSandbox.side_effect = _not_found()

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await openshell_backend.create_deployment(
            workspace="default",
            name="srv",
            config_name="cfg1",
            labels={},
            backend_config={},
            auth_context=AuthContext(principal_id="user:alice"),
        )

    assert update.status == "FAILED"
    assert "openshell backend" in update.status_message
    mock_stub.CreateSandbox.assert_not_called()


async def test_delete_idempotent_when_missing(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    mock_stub.DeleteSandbox.side_effect = _not_found()
    update = await openshell_backend.delete_deployment("default", "srv")
    assert update.status == "SUCCEEDED"


async def test_list_managed_deployment_names(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    sandbox = MagicMock(
        metadata=MagicMock(
            labels={
                MANAGED_BY_KEY: MANAGED_BY_LABEL,
                DEPLOYMENT_WORKSPACE_LABEL: "default",
                DEPLOYMENT_NAME_LABEL: "srv",
            }
        )
    )
    mock_stub.ListSandboxes.return_value = MagicMock(sandboxes=[sandbox])
    names = await openshell_backend.list_managed_deployment_names()
    assert names == ["default/srv"]


async def test_get_logs_formats_lines(openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock) -> None:
    # No workload log yet (the tail exec reports no exit code) -> supervisor logs.
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY, sandbox_id="sid")
    log_line = MagicMock(source="sandbox")
    log_line.message = "serving on 8000"
    mock_stub.GetSandboxLogs.return_value = MagicMock(logs=[log_line], buffer_total=1)

    result = await openshell_backend.get_logs(workspace="default", name="srv", tail=10)
    assert result.lines == ["[sandbox] serving on 8000"]


async def test_get_logs_returns_workload_log(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY, sandbox_id="sid")
    mock_stub.ExecSandbox.return_value = _exec_events(0, stdout="starting nat\nlistening on 8000\n")

    result = await openshell_backend.get_logs(workspace="default", name="srv", tail=10)

    assert result.lines == ["starting nat", "listening on 8000"]
    mock_stub.GetSandboxLogs.assert_not_called()


async def test_create_volume_unsupported(openshell_backend: OpenShellDeploymentBackend) -> None:
    update = await openshell_backend.create_volume(
        workspace="default", name="data", size="1Gi", access_modes=["ReadWriteOnce"], backend_config={}
    )
    assert update.status == "FAILED"


# --- #2: secrets resolved; unresolved env fails FAILED, no silent drop ---


async def test_create_resolves_plain_env_into_sandbox(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_env([EnvVar(name="FOO", value="bar")])
    mock_stub.GetSandbox.side_effect = _not_found()  # idempotency probe: absent
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert spec.environment["FOO"] == "bar"


async def test_create_fails_on_unresolved_value_from(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_stub.GetSandbox.side_effect = _not_found()  # not already present
    mock_entities.get.return_value = _config_with_env(
        [EnvVar(name="POD_IP", valueFrom={"fieldRef": {"fieldPath": "status.podIP"}})]
    )

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "FAILED"
    assert "POD_IP" in update.status_message
    mock_stub.CreateSandbox.assert_not_called()


async def test_create_injects_serve_path_into_sandbox_environment(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    expected = openshell_backend._executor_config.serve_path
    assert spec.environment["PATH"] == expected
    assert spec.template.environment["PATH"] == expected


async def test_create_injects_serve_virtual_env_into_sandbox_environment(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    """PATH alone does not keep a re-launched interpreter inside the venv.

    Fabric starts its adapter host from ``sys.base_prefix``, which resolves to the
    uv-managed interpreter rather than the venv and so cannot import the adapter
    packages. VIRTUAL_ENV is what keeps that child in the venv.
    """
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    expected = openshell_backend._executor_config.serve_virtual_env
    assert spec.environment["VIRTUAL_ENV"] == expected
    assert spec.template.environment["VIRTUAL_ENV"] == expected


async def test_create_respects_a_container_declared_virtual_env(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_env([EnvVar(name="VIRTUAL_ENV", value="/opt/other-venv")])
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert spec.environment["VIRTUAL_ENV"] == "/opt/other-venv"


async def test_create_omits_virtual_env_when_disabled(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    openshell_backend._executor_config.serve_virtual_env = ""
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert "VIRTUAL_ENV" not in spec.environment


async def test_create_respects_a_container_declared_path(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_env([EnvVar(name="PATH", value="/custom/bin")])
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_stub.CreateSandbox.return_value = MagicMock()

    update = await openshell_backend.create_deployment(
        workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
    )

    assert update.status == "STARTING"
    spec = mock_stub.CreateSandbox.call_args.args[0].spec
    assert spec.environment["PATH"] == "/custom/bin"


async def test_create_fails_when_secret_resolution_errors(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_stub.GetSandbox.side_effect = _not_found()
    mock_entities.get.return_value = _config()
    with patch(
        "nemo_deployments_plugin.backends.openshell.backend.resolve_deployment_config_secrets",
        side_effect=SecretResolutionError("Platform secret access failed"),
    ):
        update = await openshell_backend.create_deployment(
            workspace="default", name="srv", config_name="cfg1", labels={}, backend_config={}
        )

    assert update.status == "FAILED"
    assert "secret" in update.status_message.lower()
    mock_stub.CreateSandbox.assert_not_called()


# --- #6/#7/#8: read_status drives provisioning; failures clean up the sandbox ---


async def test_read_status_launches_serve_when_ready(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # READY, no services yet, marker absent -> launch serve, report STARTING.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _exec_events(0)]  # marker absent, launch ok

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    mock_stub.ExposeService.assert_not_called()


async def test_read_status_transient_rpc_error_is_unknown(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # A transient RPC error on the serve-marker probe must not escape read_status; it maps to
    # UNKNOWN (retry next poll) rather than deleting the sandbox or relaunching.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "gateway blip")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "UNKNOWN"
    mock_stub.DeleteSandbox.assert_not_called()


async def test_read_status_exposes_after_launch(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # READY, no services, marker present -> expose ports -> READY with endpoint.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.return_value = _exec_events(0)  # marker present
    mock_stub.ExposeService.return_value = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"
    assert [e.url for e in update.endpoints] == ["http://nhx-x--http.openshell.localhost:17670/"]


async def test_read_status_starting_until_default_tcp_probe_passes(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # No declared probe: the port must accept a connection before we expose it. Alive but
    # not yet reachable -> STARTING, and no port is exposed (exposing reads as READY).
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(0),  # liveness: alive
        _exec_events(1),  # readiness: port not yet accepting connections
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert "readiness" in update.status_message.lower()
    mock_stub.ExposeService.assert_not_called()


async def test_read_status_starting_when_readiness_probe_yields_no_exit(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Readiness fails closed: a probe whose exec stream ends without an exit event cannot
    # prove reachability, so it must not expose the port and flip to sticky READY. It
    # stays STARTING and self-heals on the next poll -- driven here by polling read_status
    # twice, undecidable then reachable, and asserting the second poll exposes and reads
    # READY. (Liveness fails open on the same undecidable signal; readiness gates
    # admission the other way.)
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    stdout_only = MagicMock()
    stdout_only.HasField.side_effect = lambda field: field == "stdout"
    stdout_only.stdout.data = b"partial"
    mock_stub.ExecSandbox.side_effect = [
        # First poll: alive but readiness stream ends with no exit event -> undecidable.
        _exec_events(0),  # marker present
        _exec_events(0),  # liveness: alive
        [stdout_only],  # readiness: no exit event
        # Second poll: still unexposed (first poll withheld the port), now reachable.
        _exec_events(0),  # marker present
        _exec_events(0),  # liveness: alive
        _exec_events(0),  # readiness: reachable
    ]
    mock_stub.ExposeService.return_value = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")

    first = await openshell_backend.read_status(workspace="default", name="srv")

    assert first.status == "STARTING"
    assert "readiness" in first.status_message.lower()
    mock_stub.ExposeService.assert_not_called()

    second = await openshell_backend.read_status(workspace="default", name="srv")

    assert second.status == "READY"
    mock_stub.ExposeService.assert_called_once()


async def test_read_status_ready_when_default_tcp_probe_connects(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Alive and the port now accepts a connection -> expose -> READY.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(0),  # liveness: alive
        _exec_events(0),  # readiness: reachable
    ]
    mock_stub.ExposeService.return_value = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"
    assert [e.url for e in update.endpoints] == ["http://nhx-x--http.openshell.localhost:17670/"]


async def test_read_status_gates_on_declared_httpget_probe(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # A declared httpGet readinessProbe is honoured against loopback; a failing probe
    # keeps the deployment STARTING and unexposed, and the probe hits the declared path.
    mock_entities.get.return_value = _config_with_readiness(Probe(httpGet=HTTPGetAction(path="/health", port=8000)))
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(0),  # liveness: alive
        _exec_events(1),  # readiness: /health not answering yet
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    mock_stub.ExposeService.assert_not_called()
    probe_scripts = [" ".join(c.args[0].command) for c in mock_stub.ExecSandbox.call_args_list]
    assert any("http://127.0.0.1:8000/health" in script for script in probe_scripts)


async def test_read_status_fails_when_serve_process_died(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    # Ports exposed but the workload is gone.
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    svc = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")
    svc.endpoint.service_name = "http"
    mock_stub.ListServices.return_value = MagicMock(services=[svc])
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(9),  # liveness probe: pid is gone
        _exec_events(0, stdout="ValueError: unknown model\n"),  # log tail
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "unknown model" in update.status_message
    mock_stub.DeleteSandbox.assert_not_called()  # kept so get_logs can still read the log


async def test_read_status_does_not_expose_a_dead_workload(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Launched on an earlier poll, process already dead.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(9),  # liveness probe: pid is gone
        _exec_events(0, stdout="Traceback: boom\n"),  # log tail
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "boom" in update.status_message
    mock_stub.ExposeService.assert_not_called()


async def test_read_status_stays_ready_when_liveness_is_undecidable(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock
) -> None:
    # Already serving and the probe cannot decide: only proof of death demotes a
    # deployment, so an undecided probe must not flap it.
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    svc = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")
    svc.endpoint.service_name = "http"
    mock_stub.ListServices.return_value = MagicMock(services=[svc])
    mock_stub.ExecSandbox.return_value = _exec_events(10)

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"


async def test_read_status_withholds_expose_until_the_serve_pid_exists(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # The launcher marks the launch unconditionally, so a marker alone proves nothing.
    # Until a pid is recorded the deployment stays STARTING rather than advancing to
    # expose and reporting READY over a workload that may never have started.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(10),  # liveness probe: no pidfile yet, still inside the grace window
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    mock_stub.ExposeService.assert_not_called()


async def test_read_status_fails_when_the_serve_pid_never_appears(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Past the grace window a missing pidfile is proof the launcher's background shell
    # never ran, which is a failed deployment rather than a slow one.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(0),  # marker present
        _exec_events(9),  # liveness probe: grace elapsed with no pidfile
        _exec_events(0, stdout="sh: 1: nat: not found\n"),  # log tail
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "not found" in update.status_message
    mock_stub.ExposeService.assert_not_called()


async def test_read_status_deletes_sandbox_when_launch_rpc_fails(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), FakeRpcError(grpc.StatusCode.INTERNAL, "exec boom")]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    mock_stub.DeleteSandbox.assert_called_once()


async def test_read_status_fails_on_nonzero_launch_exit(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _exec_events(127, stderr="sh: setsid: not found")]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "127" in update.status_message
    assert "setsid" in update.status_message
    mock_stub.ExposeService.assert_not_called()
    mock_stub.DeleteSandbox.assert_called_once()


async def test_read_status_deletes_sandbox_when_expose_fails(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.return_value = _exec_events(0)  # marker present -> expose path
    mock_stub.ExposeService.side_effect = FakeRpcError(grpc.StatusCode.INTERNAL, "expose boom")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    mock_stub.DeleteSandbox.assert_called_once()


async def test_read_status_cleanup_swallows_delete_error(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Cleanup must not mask the original provisioning failure even if DeleteSandbox errors.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.return_value = _exec_events(0)
    mock_stub.ExposeService.side_effect = FakeRpcError(grpc.StatusCode.INTERNAL, "expose boom")
    mock_stub.DeleteSandbox.side_effect = FakeRpcError(grpc.StatusCode.INTERNAL, "delete boom")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "expose" in update.status_message.lower()
    mock_stub.DeleteSandbox.assert_called_once()


def _require_probe_command(container: Container) -> tuple[list[str], str, int]:
    """Return the container's readiness probe command, asserting it exists (narrows None)."""
    probe_command = _readiness_probe_command(container)
    assert probe_command is not None
    return probe_command


def test_readiness_probe_command_defaults_to_tcp_on_the_first_port() -> None:
    command, description, timeout = _require_probe_command(_config().containers[0])
    assert command[:2] == ["/bin/sh", "-c"]
    assert "127.0.0.1" in command[2]
    assert description == "tcp 127.0.0.1:8000"
    assert "127.0.0.1:8000" in command[2]
    assert _CURL_ABS in command[2]
    assert "--noproxy '*'" in command[2]
    assert '[ "$_rc" -eq 7 ] && exit 1' in command[2]
    assert "python" not in command[2]
    assert timeout == _DEFAULT_READINESS_TIMEOUT_SECONDS + _READINESS_EXEC_TIMEOUT_MARGIN_SECONDS


def test_readiness_probe_command_uses_declared_httpget() -> None:
    container = _config_with_readiness(Probe(httpGet=HTTPGetAction(path="/ready", port=8000))).containers[0]
    command, description, _timeout = _require_probe_command(container)
    assert "http://127.0.0.1:8000/ready" in command[2]
    assert "--fail" in command[2]
    assert "--noproxy '*'" in command[2]
    assert _CURL_ABS in command[2]
    assert description == "httpGet http://127.0.0.1:8000/ready"


def test_readiness_probe_command_fails_closed_when_curl_is_missing() -> None:
    tcp = _require_probe_command(_config().containers[0])[0][2]
    http = _require_probe_command(
        _config_with_readiness(Probe(httpGet=HTTPGetAction(path="/health", port=8000))).containers[0]
    )[0][2]
    for script in (tcp, http):
        assert "else exit 1; fi" in script
        assert "else exit 0" not in script


def test_readiness_probe_command_runs_declared_exec_directly() -> None:
    probe = Probe(exec=ExecAction(command=["/bin/true"]), timeoutSeconds=4)
    container = _config_with_readiness(probe).containers[0]
    command, description, timeout = _require_probe_command(container)
    assert command == ["/bin/true"]
    assert description == "exec readiness probe"
    # An exec probe is bounded by its own timeoutSeconds, not the control-plane deadline,
    # so a hung probe cannot stall the serial reconcile loop.
    assert timeout == 4


def test_readiness_probe_command_resolves_named_tcp_socket_port() -> None:
    container = _config_with_readiness(Probe(tcpSocket=TCPSocketAction(port="http"))).containers[0]
    _command, description, _timeout = _require_probe_command(container)
    assert description == "tcp 127.0.0.1:8000"


def test_readiness_probe_command_is_none_without_probe_or_ports() -> None:
    assert _readiness_probe_command(_config(with_port=False).containers[0]) is None


def test_readiness_probe_command_https_uses_unverified_context() -> None:
    container = _config_with_readiness(
        Probe(httpGet=HTTPGetAction(path="/health", port=8000, scheme="HTTPS"))
    ).containers[0]
    command, _description, _timeout = _require_probe_command(container)
    assert "https://127.0.0.1:8000/health" in command[2]
    assert "--insecure" in command[2]


def test_readiness_probe_command_normalizes_httpget_path_without_leading_slash() -> None:
    container = _config_with_readiness(Probe(httpGet=HTTPGetAction(path="ready", port=8000))).containers[0]
    _command, description, _timeout = _require_probe_command(container)
    assert description == "httpGet http://127.0.0.1:8000/ready"


def test_readiness_probe_command_skips_udp_only_ports() -> None:
    container = Container(
        name="web",
        image="img:latest",
        command=["serve"],
        ports=[ContainerPort(containerPort=9000, name="udp", protocol="UDP")],
    )
    assert _readiness_probe_command(container) is None


def test_readiness_probe_command_falls_back_when_declared_port_unresolvable() -> None:
    # A declared probe naming a port absent from the container falls back to the first
    # TCP port rather than skipping the gate (which would re-open the bind race).
    container = _config_with_readiness(Probe(httpGet=HTTPGetAction(path="/health", port="does-not-exist"))).containers[
        0
    ]
    _command, description, _timeout = _require_probe_command(container)
    assert description == "httpGet http://127.0.0.1:8000/health"


def _run_probe(tmp_path: Path, *, pid: str | None, marker: str | None) -> int:
    """Run the real liveness probe against tmp files, returning its exit code.

    The probe is shell, not Python, so the three-state logic is only meaningful when
    executed by a shell. Paths are rewritten to tmp_path so the test never touches the
    constants' real locations.
    """
    pidfile = tmp_path / "serve.pid"
    marker_file = tmp_path / "serve.launched"
    if pid is not None:
        pidfile.write_text(pid)
    if marker is not None:
        marker_file.write_text(marker)
    script = _LIVENESS_PROBE.replace(_SERVE_PIDFILE, str(pidfile)).replace(_LAUNCH_MARKER, str(marker_file))
    return subprocess.run(["/bin/sh", "-c", script], capture_output=True, check=False).returncode


def test_liveness_probe_reports_a_live_pid_as_alive(tmp_path: Path) -> None:
    # The test process itself: alive, and owned by this user so kill -0 is permitted
    # (an unowned pid fails with EPERM and would read as dead).
    assert _run_probe(tmp_path, pid=str(os.getpid()), marker=str(int(time.time()))) == 0


def test_liveness_probe_reports_a_reaped_pid_as_dead(tmp_path: Path) -> None:
    dead = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()
    assert _run_probe(tmp_path, pid=str(dead.pid), marker=str(int(time.time()))) == _SERVE_DEAD_EXIT


def test_liveness_probe_is_pending_while_the_pidfile_is_still_expected(tmp_path: Path) -> None:
    # Launched a moment ago with no pid recorded yet: a slow start, not a failure.
    assert _run_probe(tmp_path, pid=None, marker=str(int(time.time()))) == _SERVE_PENDING_EXIT


def test_liveness_probe_ages_a_missing_pidfile_into_dead(tmp_path: Path) -> None:
    # This is the hole the three-state probe closes: the launcher marks the launch
    # unconditionally, so without the grace check a pidfile that never appears reads as
    # healthy forever and the deployment advances to READY over nothing.
    stale = int(time.time()) - _SERVE_PID_GRACE_SECONDS - 1
    assert _run_probe(tmp_path, pid=None, marker=str(stale)) == _SERVE_DEAD_EXIT


@pytest.mark.parametrize("marker", [None, "", "not-a-timestamp"])
def test_liveness_probe_is_pending_when_the_marker_is_unusable(tmp_path: Path, marker: str | None) -> None:
    # No usable launch time means no basis to call it dead, so stay undecided rather
    # than fail a deployment on a garbled marker.
    assert _run_probe(tmp_path, pid=None, marker=marker) == _SERVE_PENDING_EXIT


# --- AIRCORE-999: config_files are delivered into the sandbox, or fail loudly ---


async def test_delivers_config_file_before_launch_streaming_content_on_stdin(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # READY, marker absent -> deliver the config file, then launch. The file content must
    # ride the exec's stdin (opaque bytes), never the argv, and delivery must precede launch.
    content = "hello: world\nnested:\n  a: 1\n"
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/injected.yaml", content=content)]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    # marker absent, delivery ok (prints completion marker), launch ok
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _delivery_ok(), _exec_events(0)]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    deliveries = _delivery_requests(mock_stub)
    assert len(deliveries) == 1
    write = deliveries[0]
    script = write.command[-1]
    assert "cat > /home/sandbox/injected.yaml" in script
    assert "mkdir -p /home/sandbox" in script
    assert "chmod 644 /home/sandbox/injected.yaml" in script
    # Content is streamed, not embedded in the command.
    assert write.stdin == content.encode("utf-8")
    assert content not in script
    # Delivery happened before the serve launch (setsid) exec.
    kinds = [
        "deliver"
        if any("cat >" in p for p in r.command)
        else ("launch" if any("setsid" in p for p in r.command) else "probe")
        for r in _exec_requests(mock_stub)
    ]
    assert kinds.index("deliver") < kinds.index("launch")


async def test_delivers_multiple_config_files_each_on_its_own_stdin(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    files = [
        ConfigFile(path="/home/sandbox/agent.yaml", content="a: 1\n"),
        ConfigFile(path="/home/sandbox/skills/tool.yaml", content="tool: search\n"),
    ]
    mock_entities.get.return_value = _config_with_config_files(files)
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _delivery_ok(), _delivery_ok(), _exec_events(0)]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    deliveries = _delivery_requests(mock_stub)
    assert len(deliveries) == 2
    assert deliveries[0].stdin == b"a: 1\n"
    assert "cat > /home/sandbox/agent.yaml" in deliveries[0].command[-1]
    assert deliveries[1].stdin == b"tool: search\n"
    assert "mkdir -p /home/sandbox/skills" in deliveries[1].command[-1]


@pytest.mark.parametrize(("mode", "expected"), [(0o644, "chmod 644"), (0o600, "chmod 600"), (0o755, "chmod 755")])
async def test_config_file_mode_is_applied_as_octal(
    openshell_backend: OpenShellDeploymentBackend,
    mock_stub: MagicMock,
    mock_entities: AsyncMock,
    mode: int,
    expected: str,
) -> None:
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/x.yaml", content="k: v\n", mode=mode)]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _delivery_ok(), _exec_events(0)]

    await openshell_backend.read_status(workspace="default", name="srv")

    assert f"{expected} /home/sandbox/x.yaml" in _delivery_requests(mock_stub)[0].command[-1]


async def test_config_content_with_shell_metacharacters_is_not_interpreted(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # The adversarial case: content that would be catastrophic if it reached the shell.
    # Because it is streamed on stdin, it stays opaque bytes and never appears in the argv.
    content = 'x: "$(rm -rf /)"\ny: `reboot`\nz: ; cat /etc/shadow #\n'
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/evil.yaml", content=content)]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _delivery_ok(), _exec_events(0)]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    write = _delivery_requests(mock_stub)[0]
    assert write.stdin == content.encode("utf-8")
    script = write.command[-1]
    for needle in ("rm -rf /", "reboot", "cat /etc/shadow"):
        assert needle not in script


def test_delivery_script_writes_bytes_verbatim_and_never_executes_content(tmp_path: Path) -> None:
    # Run the exact script the backend generates through a real /bin/sh with adversarial
    # content on stdin, and prove the shell writes it verbatim without ever interpreting it:
    # shell-injection payloads in the content stay inert bytes and land in the file 1:1.
    target = tmp_path / "nested" / "dir" / "agent.yaml"
    canary = tmp_path / "canary"  # a side effect would delete or overwrite this
    canary.write_text("alive")
    # Every shell-injection vector: command substitution, backticks, and a statement break.
    content = f'x: "$(rm -f {canary})"\ny: `printf pwned > {canary}`\nz: ; printf pwned > {canary} #\n'

    proc = subprocess.run(
        ["/bin/sh", "-c", _delivery_script(str(target), 0o600)],
        input=content.encode("utf-8"),
        capture_output=True,
    )

    assert proc.returncode == 0
    assert _CONFIG_DELIVERED_MARKER.encode() in proc.stdout
    assert target.read_bytes() == content.encode("utf-8")
    assert canary.read_text() == "alive"  # untouched: the injection payload never executed
    assert (target.stat().st_mode & 0o777) == 0o600


@pytest.mark.skipif(hasattr(os, "getuid") and os.getuid() == 0, reason="root bypasses write permissions")
def test_delivery_script_fails_without_marker_when_target_is_unwritable(tmp_path: Path) -> None:
    # The real failure branch: a target under a read-only directory makes the shell exit
    # non-zero and never print the marker -- exactly what the backend treats as FAILED.
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)  # r-x: the sandbox user cannot create files here
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", _delivery_script(str(readonly / "denied.yaml"), 0o644)],
            input=b"k: v\n",
            capture_output=True,
        )
    finally:
        readonly.chmod(0o700)  # restore so tmp_path cleanup can remove it

    assert proc.returncode != 0
    assert _CONFIG_DELIVERED_MARKER.encode() not in proc.stdout


async def test_config_path_with_spaces_is_quoted(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/my config.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _delivery_ok(), _exec_events(0)]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    script = _delivery_requests(mock_stub)[0].command[-1]
    # shlex.quote wraps the path so the space cannot split it into two arguments.
    assert "'/home/sandbox/my config.yaml'" in script


async def test_unwritable_config_path_fails_loudly_and_deletes_sandbox(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # The /workspace boundary: delivery runs as the sandbox user, so a target it cannot
    # write (the image chowns /workspace to 'agent') makes cat exit non-zero. That must
    # surface as a terminal FAILED naming the path and the shell's error -- never a silent
    # drop that reaches READY -- and the sandbox is torn down. Launch never happens.
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/workspace/injected.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(1),  # marker absent
        # the real failure is a shell redirection error (cat never runs), not a cat error
        _exec_events(1, stderr="sh: 1: cannot create /workspace/injected.yaml: Permission denied"),
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "/workspace/injected.yaml" in update.status_message
    assert "Permission denied" in update.status_message
    mock_stub.DeleteSandbox.assert_called_once()
    # No launch was attempted after the failed write.
    assert not any("setsid" in p for r in _exec_requests(mock_stub) for p in r.command)


async def test_config_delivery_rpc_error_fails_loudly_and_deletes_sandbox(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/x.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), FakeRpcError(grpc.StatusCode.INTERNAL, "exec boom")]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "/home/sandbox/x.yaml" in update.status_message
    mock_stub.DeleteSandbox.assert_called_once()


async def test_no_config_files_delivers_nothing(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # A deployment with no config_files must not add any delivery exec: only the marker
    # probe and the launch. Guards the empty-list no-op that keeps every prior test valid.
    mock_entities.get.return_value = _config()
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), _exec_events(0)]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert _delivery_requests(mock_stub) == []
    assert mock_stub.ExecSandbox.call_count == 2


async def test_config_files_not_redelivered_once_serve_is_launched(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # Delivery lives inside the once-only launch guard. On a later poll where the marker is
    # present, the backend advances to expose without rewriting the files (write exactly once).
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/x.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.return_value = _exec_events(0)  # marker present, liveness alive
    mock_stub.ExposeService.return_value = MagicMock(url="http://nhx-x--http.openshell.localhost:17670/")

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "READY"
    assert _delivery_requests(mock_stub) == []


@pytest.mark.parametrize(
    "delivery_stream",
    [_exec_events(0), _stream_without_exit(stdout="partial")],
    ids=["zero-exit-no-marker", "stream-without-exit-event"],
)
async def test_config_delivery_without_completion_marker_fails_loudly(
    openshell_backend: OpenShellDeploymentBackend,
    mock_stub: MagicMock,
    mock_entities: AsyncMock,
    delivery_stream: list[MagicMock],
) -> None:
    # Self-attesting delivery: a write that cannot prove it happened -- a zero exit with no
    # marker, or a stream that ends without an exit event (exit_code None) -- must fail
    # loudly, never advance to launch. This is the residual silent-drop path the marker closes.
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/x.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [_exec_events(1), delivery_stream]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "could not be confirmed" in update.status_message
    assert "/home/sandbox/x.yaml" in update.status_message
    mock_stub.DeleteSandbox.assert_called_once()
    assert not any("setsid" in p for r in _exec_requests(mock_stub) for p in r.command)


async def test_multi_file_partial_failure_aborts_and_deletes_sandbox(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # File 1 delivers, file 2 fails: the loop aborts mid-stream, the deployment fails naming
    # the offending file, the sandbox is torn down, and serve is never launched.
    files = [
        ConfigFile(path="/home/sandbox/agent.yaml", content="a: 1\n"),
        ConfigFile(path="/home/sandbox/skills/tool.yaml", content="t: 1\n"),
    ]
    mock_entities.get.return_value = _config_with_config_files(files)
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(1),  # marker absent
        _delivery_ok(),  # file 1 ok
        _exec_events(1, stderr="sh: 1: cannot create /home/sandbox/skills/tool.yaml: Permission denied"),
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "FAILED"
    assert "/home/sandbox/skills/tool.yaml" in update.status_message
    assert len(_delivery_requests(mock_stub)) == 2  # both attempted; aborted after the failure
    mock_stub.DeleteSandbox.assert_called_once()
    assert not any("setsid" in p for r in _exec_requests(mock_stub) for p in r.command)


async def test_config_delivery_with_marker_but_no_exit_event_succeeds(
    openshell_backend: OpenShellDeploymentBackend, mock_stub: MagicMock, mock_entities: AsyncMock
) -> None:
    # A response stream can lose its exit event (exit_code None) yet still carry the
    # completion marker. The marker prints only after set -e clears mkdir+cat+chmod, and the
    # file content is sent atomically in the ExecSandbox *request* (stdin is one bytes field,
    # not the response stream), so a marker in the output is positive proof the file was
    # written. Delivery is therefore confirmed and launch proceeds -- requiring exit_code == 0
    # here would false-fail a proven-good delivery and tear down the sandbox.
    mock_entities.get.return_value = _config_with_config_files(
        [ConfigFile(path="/home/sandbox/x.yaml", content="k: v\n")]
    )
    mock_stub.GetSandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
    mock_stub.ExecSandbox.side_effect = [
        _exec_events(1),  # marker absent -> not yet launched
        _stream_without_exit(stdout=_CONFIG_DELIVERED_MARKER),  # write proven, exit event lost
        _exec_events(0),  # launch ok
    ]

    update = await openshell_backend.read_status(workspace="default", name="srv")

    assert update.status == "STARTING"
    assert len(_delivery_requests(mock_stub)) == 1
    # Launch proceeded after the confirmed delivery.
    assert any("setsid" in p for r in _exec_requests(mock_stub) for p in r.command)
