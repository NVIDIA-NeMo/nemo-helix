# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenSandboxDeploymentBackend."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fake_opensandbox import API_KEY, FakeOpenSandbox, running_fake_opensandbox
from nemo_deployments_plugin.backends.base import MissingBackendDependencyError
from nemo_deployments_plugin.backends.opensandbox import backend as backend_module
from nemo_deployments_plugin.backends.opensandbox import endpoint as endpoint_module
from nemo_deployments_plugin.backends.opensandbox.backend import (
    SERVE_COMMAND_KEY,
    SERVE_READY_KEY,
    OpenSandboxDeploymentBackend,
)
from nemo_deployments_plugin.backends.opensandbox.config import PlatformEgressConfig
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    ContainerPort,
    DeploymentConfig,
    EnvVar,
    HTTPGetAction,
    Probe,
    SecretRef,
    WorkloadIdentitySpec,
)
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

pytest.importorskip("opensandbox")  # optional extra

WORKSPACE = "default"
NAME = "calc-dep"
IDENTITY = {
    "managed-by": "nemo-deployments",
    "nemo.nvidia.com/deployment-workspace": WORKSPACE,
    "nemo.nvidia.com/deployment-name": NAME,
    "nemo.nvidia.com/deployment-config": NAME,
}


@pytest.fixture
def osb(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenSandbox]:
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", API_KEY)
    endpoint_module._endpoint_headers.clear()
    with running_fake_opensandbox() as state:
        yield state
    endpoint_module._endpoint_headers.clear()


def _config(**overrides: Any) -> DeploymentConfig:
    container = Container(
        name="agent",
        image="registry.example/agent:1",
        command=["python"],
        args=["-m", "server", "--port", "8000"],
        ports=[ContainerPort(name="http", containerPort=8000)],
        env=[
            EnvVar(name="NHX_WORKSPACE", value=WORKSPACE),
            EnvVar(name="API_TOKEN", secretRef=SecretRef(workspace=WORKSPACE, name="token")),
        ],
        readinessProbe=Probe(httpGet=HTTPGetAction(path="/health", port=8000)),
    )
    fields: dict[str, Any] = {
        "name": NAME,
        "workspace": WORKSPACE,
        "containers": [container.model_copy(update=overrides.pop("container", {}))],
        "configFiles": [ConfigFile(path="/tmp/nemo/agent.yaml", content="name: calc\n", mode=0o600)],
        "labels": {"nemo.agents/mode": "opensandbox"},
    }
    return DeploymentConfig(**{**fields, **overrides})


def _backend(
    osb: FakeOpenSandbox,
    monkeypatch: pytest.MonkeyPatch,
    config: DeploymentConfig | None = None,
    **executor: Any,
) -> OpenSandboxDeploymentBackend:
    backend = OpenSandboxDeploymentBackend(MagicMock(), {"domain": osb.base, "gateway_address": osb.base, **executor})
    backend._entities = MagicMock()
    backend._entities.get = AsyncMock(return_value=config or _config())

    async def resolve(client: Any, deployment_config: DeploymentConfig) -> DeploymentConfig:
        resolved = deployment_config.model_copy(deep=True)
        for var in resolved.containers[0].env:
            if var.secret_ref is not None:
                var.value, var.secret_ref = f"resolved-{var.secret_ref.name}", None
        return resolved

    monkeypatch.setattr(backend_module, "resolve_deployment_config_secrets", resolve)
    return backend


async def _create(backend: OpenSandboxDeploymentBackend) -> Any:
    return await backend.create_deployment(
        workspace=WORKSPACE, name=NAME, config_name=NAME, labels={"managed-by": "nemo-deployments"}, backend_config={}
    )


async def _read(backend: OpenSandboxDeploymentBackend) -> Any:
    return await backend.read_status(workspace=WORKSPACE, name=NAME)


async def test_create_requests_an_idle_sandbox_with_denied_egress(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch, platform_egress={"host": "10.96.0.12", "port": 8080})

    update = await _create(backend)

    assert update.status == "STARTING"
    (body,) = osb.create_bodies
    assert body["image"] == {"uri": "registry.example/agent:1"}
    assert body["entrypoint"] == ["tail", "-f", "/dev/null"]
    assert body["timeout"] is None
    assert body["resourceLimits"] == {"cpu": "1", "memory": "2Gi"}
    assert body["secureAccess"] is True
    assert body["networkPolicy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.96.0.12", "ports": [8080]}],
    }
    assert set(body["env"]) == {"EXECD_ACCESS_TOKEN"}
    assert body["metadata"].items() >= {**IDENTITY, "nemo.agents/mode": "opensandbox"}.items()


async def test_create_without_platform_egress_denies_all_egress(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _create(_backend(osb, monkeypatch))
    assert osb.create_bodies[0]["networkPolicy"] == {"defaultAction": "deny", "egress": []}


async def test_create_uses_the_containers_resources(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    resources = {"limits": {"cpu": "4", "memory": "8Gi"}, "requests": {"cpu": "2"}}
    config = _config(container={"resources": Container(name="x", image="x", resources=resources).resources})
    await _create(_backend(osb, monkeypatch, config))
    assert osb.create_bodies[0]["resourceLimits"] == {"cpu": "4", "memory": "8Gi"}
    assert osb.create_bodies[0]["resourceRequests"] == {"cpu": "2"}


async def test_create_is_idempotent_for_a_live_sandbox(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    await _create(backend)
    assert len(osb.create_bodies) == 1


async def test_create_replaces_a_failed_sandbox(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = osb.add(IDENTITY, state="Failed")
    update = await _create(_backend(osb, monkeypatch))
    assert update.status == "STARTING"
    assert osb.deleted == [failed]
    assert len(osb.sandboxes) == 1


async def test_deployment_reaches_ready_across_polls(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    osb.initial_state = "Pending"
    await _create(backend)
    (sandbox_id,) = osb.sandboxes

    assert (await _read(backend)).status == "STARTING"
    assert osb.commands == {}

    osb.sandboxes[sandbox_id]["status"]["state"] = "Running"
    osb.app_status = 503
    launched = await _read(backend)
    assert (launched.status, launched.status_message) == ("STARTING", "Serve command launched")
    assert osb.files[sandbox_id] == {"/tmp/nemo/agent.yaml": (b"name: calc\n", 600)}
    ((command_id, command),) = osb.commands.items()
    assert command.argv == ["python", "-m", "server", "--port", "8000"]
    assert command.envs == {"NHX_WORKSPACE": WORKSPACE, "API_TOKEN": "resolved-token"}
    assert osb.sandboxes[sandbox_id]["metadata"][SERVE_COMMAND_KEY] == command_id
    assert "API_TOKEN" not in osb.create_bodies[0]["env"]

    assert (await _read(backend)).status == "STARTING"
    assert osb.probes == ["/health"]
    assert len(osb.commands) == 1

    osb.app_status = 200
    ready = await _read(backend)
    assert ready.status == "READY"
    assert [endpoint.model_dump() for endpoint in ready.endpoints] == [
        {"name": "http", "url": f"http://{sandbox_id}-8000.osb.helix.internal", "protocol": "http"}
    ]
    assert osb.sandboxes[sandbox_id]["metadata"][SERVE_READY_KEY] == "true"

    osb.app_status = 503
    assert (await _read(backend)).status == "READY"
    assert len(osb.probes) == 2


async def test_a_container_without_an_http_probe_is_ready_once_serving(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch, _config(container={"readiness_probe": None}))
    await _create(backend)
    await _read(backend)
    assert (await _read(backend)).status == "READY"
    assert osb.probes == []


async def test_an_exited_serve_process_fails_the_deployment(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    await _read(backend)
    (command,) = osb.commands.values()
    command.running, command.exit_code, command.output = False, 3, "Traceback\nboom\n"

    update = await _read(backend)

    assert (update.status, update.exit_code) == ("FAILED", 3)
    assert update.status_message.endswith("Last output: Traceback\nboom")
    assert len(osb.sandboxes) == 1


async def test_a_serve_process_execd_no_longer_tracks_fails_the_deployment(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    await _read(backend)
    osb.commands.clear()
    assert (await _read(backend)).status == "FAILED"


@pytest.mark.parametrize("state", ["Failed", "Terminated", "Stopping"])
async def test_a_dead_sandbox_is_lost(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    osb.add(IDENTITY, state=state)
    assert (await _read(_backend(osb, monkeypatch))).status == "LOST"


async def test_a_missing_sandbox_is_lost(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    osb.add({**IDENTITY, "nemo.nvidia.com/deployment-name": "another"})
    assert (await _read(_backend(osb, monkeypatch))).status == "LOST"


async def test_a_paused_sandbox_is_unknown(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    osb.add(IDENTITY, state="Paused")
    assert (await _read(_backend(osb, monkeypatch))).status == "UNKNOWN"


@pytest.mark.parametrize("status", [500, 504])
async def test_server_errors_are_unknown_on_read_and_failed_on_create(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    backend = _backend(osb, monkeypatch)
    osb.server_status = status
    read = await _read(backend)
    assert (read.status, read.status_message) == (
        "UNKNOWN",
        f"read sandbox failed (HTTP {status}): BROKEN: server is broken",
    )
    created = await _create(backend)
    assert (created.status, created.status_message) == (
        "FAILED",
        f"create sandbox failed (HTTP {status}): BROKEN: server is broken",
    )


async def test_create_outlasts_the_request_timeout(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    osb.create_delay = 0.5
    backend = _backend(osb, monkeypatch, request_timeout_seconds=0.1, create_timeout_seconds=5)
    assert (await _create(backend)).status == "STARTING"


async def test_create_fails_after_the_create_timeout(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    osb.create_delay = 0.5
    backend = _backend(osb, monkeypatch, create_timeout_seconds=0.1)
    created = await _create(backend)
    assert created.status == "FAILED"
    assert created.status_message == "create sandbox failed: ReadTimeout"


async def test_launch_waits_for_execd(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    osb.execd_status = 502
    assert (await _read(backend)).status == "STARTING"
    osb.execd_status = None
    assert (await _read(backend)).status_message == "Serve command launched"


async def test_a_launch_that_cannot_be_recorded_is_interrupted(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    osb.patch_status = 409

    assert (await _read(backend)).status == "UNKNOWN"
    assert osb.interrupted == list(osb.commands)


async def test_a_rejected_endpoint_token_is_looked_up_again(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    assert (await _read(backend)).status_message == "Serve command launched"

    osb.secure_token_generation += 1
    rejected = await _read(backend)
    assert rejected.status == "UNKNOWN"
    assert rejected.status_message.startswith("read sandbox failed (HTTP 401)")
    assert (await _read(backend)).status == "READY"


async def test_an_unresolved_secret_fails_the_launch(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)

    async def resolve(client: Any, deployment_config: DeploymentConfig) -> DeploymentConfig:
        raise backend_module.SecretResolutionError("secret default/token is missing")

    monkeypatch.setattr(backend_module, "resolve_deployment_config_secrets", resolve)
    await _create(backend)

    update = await _read(backend)

    assert (update.status, update.status_message) == ("FAILED", "secret default/token is missing")
    assert osb.commands == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"containers": []}, "exactly one container"),
        ({"initContainers": [Container(name="init", image="busybox")]}, "initContainers"),
        ({"container": {"image": ""}}, "image is required"),
        ({"container": {"command": [], "args": []}}, "serve command"),
        ({"container": {"env": [EnvVar(name="X", valueFrom={"fieldRef": {}})]}}, "valueFrom"),
        ({"workloadIdentity": WorkloadIdentitySpec(enabled=True)}, "orkload"),
    ],
)
async def test_unsupported_configs_fail_before_a_sandbox_is_created(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], message: str
) -> None:
    update = await _create(_backend(osb, monkeypatch, _config(**overrides)))
    assert update.status == "FAILED"
    assert message in update.status_message
    assert osb.create_bodies == []


async def test_an_auth_proxy_request_fails_when_platform_auth_is_on(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(authProxySidecar=True, authProxySidecarIdentity="agents")
    monkeypatch.setattr(backend_module, "platform_auth_enabled", lambda: True)
    assert (await _create(_backend(osb, monkeypatch, config))).status == "FAILED"
    monkeypatch.setattr(backend_module, "platform_auth_enabled", lambda: False)
    assert (await _create(_backend(osb, monkeypatch, config))).status == "STARTING"


async def test_a_missing_config_fails_create(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    backend._entities.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
    assert (await _create(backend)).status == "FAILED"


async def test_delete_removes_every_sandbox_of_the_deployment(
    osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    osb.add(IDENTITY, state="Failed")
    other = osb.add({**IDENTITY, "nemo.nvidia.com/deployment-name": "another"})

    update = await backend.delete_deployment(WORKSPACE, NAME)

    assert update.status == "SUCCEEDED"
    assert list(osb.sandboxes) == [other]
    assert (await backend.delete_deployment(WORKSPACE, NAME)).status == "SUCCEEDED"


async def test_list_managed_deployment_names(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    await _create(backend)
    osb.add({**IDENTITY, "nemo.nvidia.com/deployment-name": "another"})
    osb.add({"managed-by": "someone-else", "nemo.nvidia.com/deployment-name": "foreign"})
    assert await backend.list_managed_deployment_names() == ["default/another", "default/calc-dep"]


async def test_get_logs_returns_the_serve_output_tail(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    assert (await backend.get_logs(workspace=WORKSPACE, name=NAME)).lines == ["Sandbox not found"]
    await _create(backend)
    assert (await backend.get_logs(workspace=WORKSPACE, name=NAME)).lines == ["The serve command has not started"]
    await _read(backend)
    next(iter(osb.commands.values())).output = "one\ntwo\nthree\n"

    logs = await backend.get_logs(workspace=WORKSPACE, name=NAME, tail=2)

    assert (logs.lines, logs.truncated) == (["two", "three"], True)


async def test_execd_tokens_differ_per_deployment(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    assert backend._execd_token(WORKSPACE, NAME) == backend._execd_token(WORKSPACE, NAME)
    assert backend._execd_token(WORKSPACE, NAME) != backend._execd_token(WORKSPACE, "another")


async def test_volumes_are_unsupported(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(osb, monkeypatch)
    update = await backend.create_volume(workspace=WORKSPACE, name="v", size="1Gi", access_modes=[], backend_config={})
    assert update.status == "FAILED"


def test_a_missing_api_key_makes_the_backend_unavailable(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY")
    with pytest.raises(MissingBackendDependencyError, match="OPEN_SANDBOX_API_KEY"):
        OpenSandboxDeploymentBackend(MagicMock(), {"domain": osb.base, "gateway_address": osb.base})


def test_a_missing_sdk_makes_the_backend_unavailable(osb: FakeOpenSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "_OPENSANDBOX_IMPORT_ERROR", ImportError("No module named 'opensandbox'"))
    with pytest.raises(MissingBackendDependencyError, match="opensandbox>=1.2.0"):
        OpenSandboxDeploymentBackend(MagicMock(), {"domain": osb.base, "gateway_address": osb.base})


def test_the_backend_is_registered() -> None:
    assert BACKEND_CLASSES["opensandbox"] is OpenSandboxDeploymentBackend


def test_platform_egress_rules_and_url() -> None:
    by_ip = PlatformEgressConfig(host="10.96.0.12", port=8080)
    by_name = PlatformEgressConfig(host="nhx-api.nemo.svc.cluster.local", port=80, scheme="https")
    assert backend_module._egress_rule(by_ip).model_dump() == {
        "action": "allow",
        "target": "10.96.0.12",
        "ports": [8080],
    }
    assert backend_module._egress_rule(by_name).model_dump(exclude_none=True) == {
        "action": "allow",
        "target": "nhx-api.nemo.svc.cluster.local",
    }
    assert (by_ip.url(), by_name.url()) == ("http://10.96.0.12:8080", "https://nhx-api.nemo.svc.cluster.local:80")
