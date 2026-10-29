# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenShell job backend."""

from __future__ import annotations

import datetime
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openshell")

import grpc
from nemo_helix_plugin.jobs.execution_profiles import OpenShellJobTLSConfig
from nemo_helix_plugin.jobs.types import AuthContext
from nemo_helix_plugin.openshell.types import GatewayCredentialConfig, GatewayProviderBinding, ProviderAccessRevoked
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.controllers.backends.openshell import (
    OpenShellJobBackend,
    OpenShellJobEgressConfig,
    OpenShellJobExecutionProfile,
    OpenShellJobExecutionProfileConfig,
)
from nhx.core.jobs.controllers.backends.openshell import backend as backend_module
from nhx.core.jobs.controllers.backends.openshell.backend import (
    _DEFAULT_ENTRYPOINT,
    _ensure_openshell,
    _sandbox_name,
)
from nhx.core.jobs.controllers.backends.registry import BackendKey, backend_registry
from openshell import SandboxRef, SandboxStatusRef, TlsConfig
from openshell._proto import openshell_pb2 as pb


@asynccontextmanager
async def _provisioner_context[T](provisioner: T) -> AsyncGenerator[T]:
    yield provisioner


def _backend(mock_nemo_client, mock_platform_config, **config_overrides) -> OpenShellJobBackend:
    config = OpenShellJobExecutionProfileConfig(**config_overrides)
    with patch(
        "nhx.core.jobs.controllers.backends.openshell.backend.get_platform_config",
        return_value=mock_platform_config,
    ):
        backend = OpenShellJobBackend(mock_nemo_client, config, profile_name="default")
    backend._client = MagicMock()
    return backend


def _no_container_executor() -> SimpleNamespace:
    return SimpleNamespace(container=SimpleNamespace(image=None, entrypoint=None, command=None))


def _create_request(backend: OpenShellJobBackend) -> SimpleNamespace:
    """The keyword arguments the backend passed to ``SandboxClient.create``."""
    return SimpleNamespace(**backend._client.create.call_args.kwargs)


def _ref(sandbox: Any) -> SandboxRef:
    """The SDK's list/get view of a ``Sandbox`` proto."""
    return SandboxRef(
        id="sandbox-id",
        name=sandbox.metadata.name,
        workspace="default",
        status=SandboxStatusRef(phase=sandbox.status.phase, current_policy_version=0),
        labels=dict(sandbox.metadata.labels),
    )


_OWNED_LABELS = {
    "nhx.nvidia.com/managed_by": "jobs-controller",
    "nhx.nvidia.com/job_execution_backend": "openshell",
    "nhx.nvidia.com/job_execution_profile": "default",
    "nhx.nvidia.com/job_workspace_id": "default",
    "nhx.nvidia.com/job_id": "test-job-id",
    "nhx.nvidia.com/job_attempt_id": "attempt-1",
    "nhx.nvidia.com/job_step_name": "test-step",
}


def _owned_sandbox(backend: OpenShellJobBackend, phase: int, exit_code: int | None = None) -> Any:
    labels = {**_OWNED_LABELS, "nhx.nvidia.com/jobs_controller_instance_id": backend._jobs_controller_instance_id}
    return _sandbox(phase, exit_code=exit_code, labels=labels)


class TestClient:
    """How the profile's endpoint and TLS material become a ``SandboxClient``."""

    @staticmethod
    def _client_kwargs(mock_nemo_client, mock_platform_config, **config_overrides) -> dict[str, Any]:
        _ensure_openshell()
        with patch.object(backend_module, "JobsSandboxClient") as client_cls:
            _backend(mock_nemo_client, mock_platform_config, **config_overrides)
        (target,) = client_cls.call_args.args
        return {"target": target, **client_cls.call_args.kwargs}

    def test_http_endpoint_is_plaintext(self, mock_nemo_client, mock_platform_config) -> None:
        kwargs = self._client_kwargs(mock_nemo_client, mock_platform_config, gateway_endpoint="http://gw:17670")

        assert kwargs["target"] == "gw:17670"
        assert kwargs["tls"] is None

    def test_https_without_material_uses_system_roots(self, mock_nemo_client, mock_platform_config) -> None:
        kwargs = self._client_kwargs(mock_nemo_client, mock_platform_config, gateway_endpoint="https://gw")

        assert kwargs["target"] == "gw:443"
        assert kwargs["tls"] == TlsConfig()

    def test_https_with_mtls_material(self, mock_nemo_client, mock_platform_config) -> None:
        kwargs = self._client_kwargs(
            mock_nemo_client,
            mock_platform_config,
            gateway_endpoint="https://gw:8443",
            tls=OpenShellJobTLSConfig(
                ca_cert_path="/c/ca.crt", client_cert_path="/c/tls.crt", client_key_path="/c/tls.key"
            ),
        )

        assert kwargs["tls"] == TlsConfig(
            ca_path=Path("/c/ca.crt"), cert_path=Path("/c/tls.crt"), key_path=Path("/c/tls.key")
        )

    def test_insecure_override_wins_over_https(self, mock_nemo_client, mock_platform_config) -> None:
        kwargs = self._client_kwargs(
            mock_nemo_client, mock_platform_config, gateway_endpoint="https://gw", insecure=True
        )

        assert kwargs["tls"] is None

    def test_client_cert_requires_key(self) -> None:
        with pytest.raises(ValueError, match="set together"):
            OpenShellJobTLSConfig(client_cert_path="/c/tls.crt")


class TestRegistry:
    def test_registry_contains_openshell(self) -> None:
        assert BackendKey("cpu", "openshell") in backend_registry
        assert backend_registry[BackendKey("cpu", "openshell")] is OpenShellJobBackend

    def test_profile_roundtrips(self) -> None:
        profile = OpenShellJobExecutionProfile(
            provider="cpu",
            profile="default",
            backend="openshell",
            config=OpenShellJobExecutionProfileConfig(gateway_endpoint="https://gw:443"),
        )
        assert profile.backend == "openshell"
        assert profile.config.grpc_target() == "gw:443"
        assert not profile.config.use_insecure()
        assert profile.supports_persistent_storage is False

    def test_profile_rejects_bad_endpoint(self) -> None:
        with pytest.raises(ValueError, match="gateway_endpoint"):
            OpenShellJobExecutionProfileConfig(gateway_endpoint="not-a-url")


class TestSandboxName:
    def test_within_limit_and_deterministic(self) -> None:
        a = _sandbox_name("default", "job-1", "attempt-1", "step-1")
        b = _sandbox_name("default", "job-1", "attempt-1", "step-1")
        assert a == b
        assert len(a) <= 19
        assert a.startswith("nhx-")

    def test_attempt_changes_name(self) -> None:
        a = _sandbox_name("default", "job-1", "attempt-1", "step-1")
        b = _sandbox_name("default", "job-1", "attempt-2", "step-1")
        assert a != b


class TestSchedule:
    @pytest.mark.parametrize("with_provider", [False, True])
    def test_rejects_env_secret_before_sandbox_creation(
        self, mock_nemo_client, mock_platform_config, test_step_pending, with_provider
    ):
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending
        payload = step.step_spec.model_dump()
        payload["environment"] = [{"name": "TOKEN", "from_secret": {"name": "default/token"}}]
        if with_provider:
            payload["gateway_providers"] = {
                "service": {"profile": "service-read", "credentials": {"API_TOKEN": "default/another-token"}}
            }
        step.step_spec = type(step.step_spec).model_validate(payload)
        result = backend.schedule(step.step_spec.executor, step)
        assert result.status == HelixJobStatus.ERROR
        assert "gateway provider bindings" in result.status_details["message"]
        backend._client.create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_enabled", [True, False])
    async def test_provider_preparation_uses_platform_auth_mode(
        self, mock_nemo_client, mock_platform_config, test_step_pending, auth_enabled
    ):
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending
        step.auth_context = AuthContext(principal_id="user-1") if auth_enabled else None
        provisioner = MagicMock(prepare=AsyncMock(return_value=["owned-provider"]))
        requester = MagicMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False))
        with (
            patch.object(backend_module, "platform_auth_enabled", return_value=auth_enabled),
            patch.object(backend, "_provider_provisioner", side_effect=lambda: _provisioner_context(provisioner)),
            patch.object(backend_module, "get_async_nemo_client", return_value=requester) as client,
            patch.object(backend_module.AsyncSecretsClient, "from_client"),
        ):
            assert await backend._prepare_providers(step, "sandbox", renew=True) == ["owned-provider"]
        assert provisioner.prepare.call_args.kwargs["auth_enabled"] is auth_enabled
        assert provisioner.prepare.call_args.kwargs["renew"] is True
        principal = client.call_args.kwargs["on_behalf_of"]
        assert (principal.id if principal else None) == ("user-1" if auth_enabled else None)

    @pytest.mark.parametrize("auth_enabled", [True, False])
    @pytest.mark.parametrize("profile_env_collision", [True, False])
    def test_provider_attachment_uses_restricted_policy(
        self, mock_nemo_client, mock_platform_config, test_step_pending, auth_enabled, profile_env_collision
    ):
        backend = _backend(
            mock_nemo_client,
            mock_platform_config,
            launcher_tool_path="/app/bin/custom-jobs-launcher",
            env={"API_TOKEN": "plaintext-collision"} if profile_env_collision else {},
            gateway_credentials={
                "gateway_id": "gateway",
                "registration_key_file": "/run/key",
                "credential_refresh_verified": True,
                "profiles": {"service-read": {"workspaces": ["default"], "credential_keys": ["API_TOKEN"]}},
            },
        )
        step = test_step_pending
        step.auth_context = AuthContext(principal_id="user-1") if auth_enabled else None
        step.step_spec.gateway_providers = {
            "service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "default/token"})
        }
        backend._prepare_providers = AsyncMock(return_value=["owned-provider"])
        with patch.object(backend_module, "platform_auth_enabled", return_value=auth_enabled):
            result = backend.schedule(step.step_spec.executor, step)
        if profile_env_collision:
            assert result.status == HelixJobStatus.ERROR
            backend._prepare_providers.assert_not_awaited()
            backend._client.create.assert_not_called()
            return
        assert result.status == HelixJobStatus.PENDING
        spec = _create_request(backend).spec
        assert list(spec.providers) == ["owned-provider"]
        assert set(spec.policy.network_policies) == {"nemo_workload"}
        rules = spec.policy.network_policies["nemo_workload"].endpoints[0].rules
        assert (
            "PATCH",
            f"/apis/jobs/v2/workspaces/{step.workspace}/jobs/{step.job}/attempts/{step.attempt_id}/status-details",
        ) in {(rule.allow.method, rule.allow.path) for rule in rules}
        assert "/app/bin/custom-jobs-launcher" in {
            binary.path for binary in spec.policy.network_policies["nemo_workload"].binaries
        }
        assert "API_TOKEN" not in spec.environment
        assert "token" not in spec.environment["NEMO_JOB_SECRETS"]

    def test_provider_failure_never_creates_sandbox(self, mock_nemo_client, mock_platform_config, test_step_pending):
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending
        step.step_spec.gateway_providers = {
            "service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "token"})
        }
        result = backend.schedule(step.step_spec.executor, step)
        assert result.status == HelixJobStatus.ERROR
        backend._client.create.assert_not_called()

    def test_creates_sandbox_with_launcher_command(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending
        executor = step.step_spec.executor

        update = backend.schedule(executor, step)

        assert update.status == HelixJobStatus.PENDING
        request = _create_request(backend)
        assert request.spec.command == ["/tools/jobs-launcher", "run", "--fetch-step-config", "--"] + list(
            _DEFAULT_ENTRYPOINT
        )
        # Step container image wins over the profile default.
        assert request.spec.template.image == "test-image"
        env = dict(request.spec.environment)
        assert env["NEMO_JOB_ID"] == step.job
        assert env["NEMO_JOB_STEP"] == step.name
        assert env["NEMO_JOB_WORKSPACE"] == step.workspace
        assert env["NEMO_JOB_STEP_CONFIG_FILE_PATH"].endswith("job_step_config.json")
        # Task storage paths: the task dispatcher refuses to run without the ephemeral one.
        # The fixture step overrides it; the config path keeps the default.
        assert env["NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"] == "/var/tmp"
        assert env["NEMO_JOB_STEP_CONFIG_STORAGE_PATH"] == "/tmp/nhx-job-config"
        # The supervisor applies the image ENV; the backend must not override it.
        assert not {"PATH", "VIRTUAL_ENV", "HOME"} & env.keys()
        assert "HTTP_PROXY" not in env
        assert env["ENV_VAR"] == "test_value"
        assert request.name == _sandbox_name(step.workspace, step.job, step.attempt_id, step.name)
        labels = dict(request.labels)
        assert labels["nhx.nvidia.com/job_id"] == step.job
        assert labels["nhx.nvidia.com/job_step_name"] == step.name
        assert labels["nhx.nvidia.com/managed_by"] == "jobs-controller"
        assert labels["nhx.nvidia.com/job_execution_backend"] == "openshell"

    def test_default_image_when_step_declares_none(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)

        update = backend.schedule(_no_container_executor(), test_step_pending)

        assert update.status == HelixJobStatus.PENDING
        request = _create_request(backend)
        assert request.spec.template.image.endswith("/nhx-tasks-openshell:local")

    def test_profile_image_overrides_default(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, image="registry.example/nhx-tasks-openshell:1")

        backend.schedule(_no_container_executor(), test_step_pending)

        request = _create_request(backend)
        assert request.spec.template.image == "registry.example/nhx-tasks-openshell:1"

    def test_step_entrypoint_wins_over_default(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        executor = SimpleNamespace(
            container=SimpleNamespace(image="img", entrypoint=["/custom/bin"], command=["--flag"])
        )

        backend.schedule(executor, test_step_pending)

        request = _create_request(backend)
        assert request.spec.command == [
            "/tools/jobs-launcher",
            "run",
            "--fetch-step-config",
            "--",
            "/custom/bin",
            "--flag",
        ]

    def test_profile_launcher_path_is_the_main_process_and_egress_binary(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, launcher_tool_path="/opt/nhx/jobs-launcher")

        backend.schedule(_no_container_executor(), test_step_pending)

        request = _create_request(backend)
        assert request.spec.command[0] == "/opt/nhx/jobs-launcher"
        binaries = [b.path for b in request.spec.policy.network_policies["nemo_helix"].binaries]
        assert binaries[0] == "/opt/nhx/jobs-launcher"

    def test_custom_egress_binaries_keep_the_launcher(self, mock_nemo_client, mock_platform_config) -> None:
        backend = _backend(
            mock_nemo_client,
            mock_platform_config,
            launcher_tool_path="/opt/nhx/jobs-launcher",
            platform_egress=OpenShellJobEgressConfig(host="h", port=1, binaries=["/usr/bin/python3"]),
        )

        policy = backend._build_executor_policy()

        binaries = [b.path for b in policy.network_policies["nemo_helix"].binaries]
        assert binaries == ["/opt/nhx/jobs-launcher", "/usr/bin/python3"]

    def test_egress_proxy_injected_when_configured(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, egress_proxy="http://127.0.0.1:3128")

        backend.schedule(_no_container_executor(), test_step_pending)

        request = _create_request(backend)
        env = dict(request.spec.environment)
        assert env["HTTP_PROXY"] == "http://127.0.0.1:3128"
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:3128"
        assert env["NO_PROXY"] == "127.0.0.1,localhost"

    def test_platform_urls_rewritten_to_egress_host(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(
            mock_nemo_client,
            mock_platform_config,
            platform_egress=OpenShellJobEgressConfig(host="nemo-helix-api.nemo-helix.svc.cluster.local", port=8080),
        )

        backend.schedule(_no_container_executor(), test_step_pending)

        request = _create_request(backend)
        env = dict(request.spec.environment)
        assert env["NHX_BASE_URL"] == "http://nemo-helix-api.nemo-helix.svc.cluster.local:8080"
        assert env["NHX_JOBS_URL"] == "http://nemo-helix-api.nemo-helix.svc.cluster.local:8080"
        assert env["NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT"].startswith(
            "http://nemo-helix-api.nemo-helix.svc.cluster.local:8080"
        )

    def test_platform_urls_unchanged_without_egress(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        # Keep the assertion independent of the host's Docker loopback override.
        mock_platform_config.loopback_address = "localhost"
        backend = _backend(mock_nemo_client, mock_platform_config, platform_egress=None)

        with patch.object(backend_module, "get_platform_config", return_value=mock_platform_config):
            backend.schedule(_no_container_executor(), test_step_pending)

        request = _create_request(backend)
        env = dict(request.spec.environment)
        assert env["NHX_BASE_URL"] == "http://localhost:8080"

    def test_task_storage_defaults_without_step_override(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending.model_copy(deep=True)
        step.step_spec.environment = []

        backend.schedule(step.step_spec.executor, step)

        env = dict(_create_request(backend).spec.environment)
        # /tmp exists in every image, so a launcher that does not create the task
        # directory still works; the launcher creates the config directory itself.
        assert env["NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"] == "/tmp"
        assert env["NEMO_JOB_STEP_CONFIG_STORAGE_PATH"] == "/tmp/nhx-job-config"
        assert env["NEMO_JOB_STEP_CONFIG_FILE_PATH"] == "/tmp/nhx-job-config/job_step_config.json"

    def test_existing_sandbox_for_attempt_is_adopted(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.create.side_effect = _rpc_error(grpc.StatusCode.ALREADY_EXISTS)

        update = backend.schedule(test_step_pending.step_spec.executor, test_step_pending)

        assert update.status == HelixJobStatus.PENDING

    def test_create_failure_returns_error(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.create.side_effect = _rpc_error(grpc.StatusCode.INTERNAL, "boom")

        update = backend.schedule(_no_container_executor(), test_step_pending)

        assert update.status == HelixJobStatus.ERROR


def _rpc_error(code, details: str = "") -> Exception:
    class FakeRpcError(grpc.RpcError):
        def code(self):
            return code

        def details(self):
            return details

    return FakeRpcError()


def _sandbox(
    phase: int,
    exit_code: int | None = None,
    condition_messages: tuple[str, ...] = (),
    labels: dict[str, str] | None = None,
) -> Any:
    """A real ``Sandbox`` proto, so field presence and shape match the gateway's."""
    sandbox = pb.Sandbox()
    sandbox.metadata.name = "nhx-digest"
    sandbox.metadata.labels.update(labels or {})
    sandbox.status.phase = phase
    if exit_code is not None:
        sandbox.status.exit_code = exit_code
    for message in condition_messages:
        sandbox.status.conditions.add(message=message)
    return sandbox


class TestSync:
    def test_cancel_retries_provider_cleanup_after_sandbox_disappears(
        self, mock_nemo_client, mock_platform_config, test_step_cancelling
    ):
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_cancelling
        step.step_spec.gateway_providers = {
            "service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "token"})
        }
        backend._client.get_sandbox.side_effect = _rpc_error(grpc.StatusCode.NOT_FOUND)
        provisioner = SimpleNamespace(cleanup=AsyncMock(side_effect=[RuntimeError("offline"), None]))
        with patch.object(backend, "_provider_provisioner", side_effect=lambda: _provisioner_context(provisioner)):
            assert backend.sync(step).status == HelixJobStatus.CANCELLING
            assert backend.sync(step).status == HelixJobStatus.CANCELLED
        assert provisioner.cleanup.await_count == 2

    def test_completed_zero_exit(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_COMPLETED, exit_code=0)

        update = backend.sync(test_step_pending)

        assert update.status == HelixJobStatus.COMPLETED

    def test_completed_nonzero_exit_is_error(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_COMPLETED, exit_code=7)

        update = backend.sync(test_step_pending)

        assert update.status == HelixJobStatus.ERROR
        assert update.error_details == {"exit_code": 7}

    def test_error_phase(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(
            pb.SANDBOX_PHASE_ERROR, condition_messages=("policy denied",)
        )

        update = backend.sync(test_step_pending)

        assert update.status == HelixJobStatus.ERROR
        assert update.status_details["message"] == "policy denied"
        # The main process never ran, so there is no exit code to report.
        assert update.error_details == {"sandbox_phase": "SANDBOX_PHASE_ERROR"}

    def test_error_phase_with_exit_code(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_ERROR, exit_code=3)

        update = backend.sync(test_step_pending)

        assert update.status_details["message"] == "Job exited with code 3"
        assert update.error_details == {"sandbox_phase": "SANDBOX_PHASE_ERROR", "exit_code": 3}

    def test_unrecognized_phase_is_pending(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(999)

        update = backend.sync(test_step_pending)

        assert update.status == HelixJobStatus.PENDING

    @pytest.mark.parametrize(
        ("phase", "expected"),
        [
            (pb.SANDBOX_PHASE_READY, HelixJobStatus.ACTIVE),
            (pb.SANDBOX_PHASE_PROVISIONING, HelixJobStatus.PENDING),
            (pb.SANDBOX_PHASE_STARTING, HelixJobStatus.PENDING),
        ],
    )
    def test_lifecycle_phases(self, mock_nemo_client, mock_platform_config, test_step_pending, phase, expected) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(phase)

        update = backend.sync(test_step_pending)

        assert update.status == expected

    def test_not_found_is_error(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.side_effect = _rpc_error(grpc.StatusCode.NOT_FOUND)

        update = backend.sync(test_step_pending)

        assert update.status == HelixJobStatus.ERROR

    def test_cancel_deletes_sandbox(self, mock_nemo_client, mock_platform_config, test_step_pending) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
        step = test_step_pending.model_copy(update={"status": HelixJobStatus.CANCELLING})

        update = backend.sync(step)

        assert update.status == HelixJobStatus.CANCELLED
        assert backend._client.delete.called

    def test_cancel_of_finished_sandbox_is_cancelled(
        self, mock_nemo_client, mock_platform_config, test_step_cancelling
    ) -> None:
        """CANCELLING cannot transition to COMPLETED; reporting it would 409 forever."""
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_COMPLETED, exit_code=0)

        update = backend.sync(test_step_cancelling)

        assert update.status == HelixJobStatus.CANCELLED
        assert not backend._client.delete.called

    def test_cancel_of_missing_sandbox_is_cancelled(
        self, mock_nemo_client, mock_platform_config, test_step_cancelling
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.side_effect = _rpc_error(grpc.StatusCode.NOT_FOUND)

        update = backend.sync(test_step_cancelling)

        assert update.status == HelixJobStatus.CANCELLED

    def test_transient_get_error_keeps_status(self, mock_nemo_client, mock_platform_config, test_step_active) -> None:
        """ACTIVE cannot transition back to PENDING."""
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.side_effect = _rpc_error(grpc.StatusCode.UNAVAILABLE)

        update = backend.sync(test_step_active)

        assert update.status == HelixJobStatus.ACTIVE

    def test_active_step_does_not_regress_to_pending(
        self, mock_nemo_client, mock_platform_config, test_step_active
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_STOPPING)

        update = backend.sync(test_step_active)

        assert update.status == HelixJobStatus.ACTIVE

    def test_unmapped_phase_name_keeps_status(self, mock_nemo_client, mock_platform_config, test_step_active) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)

        with patch.object(backend_module, "_phase_name", return_value="SANDBOX_PHASE_FROM_A_NEWER_SDK"):
            update = backend.sync(test_step_active)

        assert update.status == HelixJobStatus.ACTIVE

    @pytest.mark.parametrize(("age_seconds", "timed_out"), [(30, False), (7200, True)])
    def test_active_ttl(self, mock_nemo_client, mock_platform_config, test_step_active, age_seconds, timed_out) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, ttl_seconds_active=3600)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
        step = _aged(test_step_active, age_seconds)

        update = backend.sync(step)

        assert update.status == (HelixJobStatus.ERROR if timed_out else HelixJobStatus.ACTIVE)
        assert backend._client.delete.called is timed_out
        if timed_out:
            assert "3600 seconds" in update.status_details["message"]

    @pytest.mark.parametrize(
        ("phase", "timed_out"),
        [
            (pb.SANDBOX_PHASE_PROVISIONING, True),
            # Already running: the step moves to ACTIVE instead of timing out.
            (pb.SANDBOX_PHASE_READY, False),
            # Already finished: the result wins over the timeout.
            (pb.SANDBOX_PHASE_COMPLETED, False),
        ],
    )
    def test_before_active_ttl(
        self, mock_nemo_client, mock_platform_config, test_step_pending, phase, timed_out
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, ttl_seconds_before_active=600)
        backend._client.get_sandbox.return_value = _sandbox(phase, exit_code=0)
        step = _aged(test_step_pending, 1200)

        update = backend.sync(step)

        assert (update.status == HelixJobStatus.ERROR) is timed_out
        assert backend._client.delete.called is timed_out

    def test_cancel_delete_failure_stays_cancelling(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.get_sandbox.return_value = _sandbox(pb.SANDBOX_PHASE_READY)
        backend._client.delete.side_effect = _rpc_error(grpc.StatusCode.UNAVAILABLE, "gateway down")
        step = test_step_pending.model_copy(update={"status": HelixJobStatus.CANCELLING})

        update = backend.sync(step)

        assert update.status == HelixJobStatus.CANCELLING
        assert "gateway down" in update.status_details["message"]


def _aged(step: Any, age_seconds: int) -> Any:
    then = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=age_seconds)
    return step.model_copy(update={"created_at": then, "updated_at": then})


def _step_state(status: str = "error", finished_ago: datetime.timedelta | None = None) -> SimpleNamespace:
    """The step entity cleanup reads: its status and when it last changed."""
    updated_at = datetime.datetime.now(datetime.timezone.utc) - (finished_ago or datetime.timedelta())
    return SimpleNamespace(status=status, updated_at=updated_at)


class TestCleanup:
    def test_deletes_completed_sandbox_when_configured(
        self, mock_nemo_client, mock_platform_config, test_step_completed
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _owned_sandbox(backend, pb.SANDBOX_PHASE_COMPLETED, exit_code=0)
        backend._client.list_all.return_value = [_ref(sandbox)]
        backend.get_step_safe = MagicMock(return_value=_step_state("completed"))  # type: ignore[method-assign]

        backend.cleanup_steps()

        backend.get_step_safe.assert_called_once_with(
            job="test-job-id",
            step_name="test-step",
            workspace="default",
            attempt_id="attempt-1",
        )
        backend._client.delete.assert_called_once_with("nhx-digest", workspace="default", allow_missing=True)
        # The immediate path needs no conditions, so it skips the extra GetSandbox.
        assert not backend._client.get_sandbox.called

    @pytest.mark.parametrize(
        ("finished_ago", "deleted"), [(datetime.timedelta(minutes=5), False), (datetime.timedelta(hours=2), True)]
    )
    def test_errored_sandbox_waits_for_ttl(self, mock_nemo_client, mock_platform_config, finished_ago, deleted) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, ttl_seconds_after_finished=3600)
        sandbox = _owned_sandbox(backend, pb.SANDBOX_PHASE_ERROR, exit_code=1)
        sandbox.status.conditions.add().transition_time.FromDatetime(
            datetime.datetime.now(datetime.timezone.utc) - finished_ago
        )
        backend._client.list_all.return_value = [_ref(sandbox)]
        backend._client.get_sandbox.return_value = sandbox
        # The step went terminal just now; the sandbox's own transition time wins.
        backend.get_step_safe = MagicMock(return_value=_step_state())  # type: ignore[method-assign]

        backend.cleanup_steps()

        assert backend._client.delete.called is deleted

    @pytest.mark.parametrize(
        ("finished_ago", "deleted"), [(datetime.timedelta(minutes=5), False), (datetime.timedelta(hours=2), True)]
    )
    def test_ttl_falls_back_to_step_time_without_transition_time(
        self, mock_nemo_client, mock_platform_config, finished_ago, deleted
    ) -> None:
        """The docker driver stamps no condition transition_time."""
        backend = _backend(mock_nemo_client, mock_platform_config, ttl_seconds_after_finished=3600)
        sandbox = _owned_sandbox(backend, pb.SANDBOX_PHASE_ERROR, exit_code=1)
        sandbox.status.conditions.add(message="no timestamp")
        backend._client.list_all.return_value = [_ref(sandbox)]
        backend._client.get_sandbox.return_value = sandbox
        backend.get_step_safe = MagicMock(return_value=_step_state(finished_ago=finished_ago))  # type: ignore[method-assign]

        backend.cleanup_steps()

        assert backend._client.delete.called is deleted

    @pytest.mark.parametrize("step_state", [_step_state("error"), None], ids=["terminal-step", "missing-step"])
    def test_running_sandbox_under_finished_step_is_reclaimed(
        self, mock_nemo_client, mock_platform_config, step_state
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _owned_sandbox(backend, pb.SANDBOX_PHASE_READY)
        backend._client.list_all.return_value = [_ref(sandbox)]
        backend.get_step_safe = MagicMock(return_value=step_state)  # type: ignore[method-assign]

        backend.cleanup_steps()

        backend._client.delete.assert_called_once_with("nhx-digest", workspace="default", allow_missing=True)

    def test_keeps_sandbox_when_step_not_terminal(
        self, mock_nemo_client, mock_platform_config, test_step_active
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _owned_sandbox(backend, pb.SANDBOX_PHASE_READY)
        backend._client.list_all.return_value = [_ref(sandbox)]
        backend.get_step_safe = MagicMock(return_value=_step_state("active"))  # type: ignore[method-assign]

        backend.cleanup_steps()

        assert not backend._client.delete.called

    def test_list_pager_error_is_logged_not_raised(self, mock_nemo_client, mock_platform_config) -> None:
        from openshell import SandboxError

        backend = _backend(mock_nemo_client, mock_platform_config)
        backend._client.list_all.side_effect = SandboxError("pager received a repeated continuation token")

        backend.cleanup_steps()

        assert not backend._client.delete.called

    def test_terminal_transition_time_is_tz_aware(self, mock_nemo_client, mock_platform_config) -> None:
        """The TTL comparison must not raise on naive vs aware datetimes."""

        from google.protobuf.timestamp_pb2 import Timestamp

        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _sandbox(pb.SANDBOX_PHASE_ERROR, exit_code=1)
        ts = Timestamp()
        ts.FromDatetime(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2))
        sandbox.status.conditions.add().transition_time.CopyFrom(ts)

        finished_at = backend._terminal_transition_time(sandbox)

        assert finished_at is not None
        assert finished_at.tzinfo is not None

    def test_terminal_transition_time_skips_unset_timestamps(self, mock_nemo_client, mock_platform_config) -> None:
        """An unset transition_time must not read as the epoch; fall back to the last real one."""

        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _sandbox(pb.SANDBOX_PHASE_ERROR, exit_code=1)
        recent = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        sandbox.status.conditions.add(message="timestamped").transition_time.FromDatetime(recent)
        sandbox.status.conditions.add(message="no timestamp")

        finished_at = backend._terminal_transition_time(sandbox)

        assert finished_at is not None
        assert abs((finished_at - recent).total_seconds()) < 1


@pytest.mark.parametrize(
    "phase,mode,renew,error",
    [
        (pb.SANDBOX_PHASE_READY, "helix", True, RuntimeError),
        (pb.SANDBOX_PHASE_READY, "helix", True, ProviderAccessRevoked),
        (pb.SANDBOX_PHASE_PROVISIONING, "helix", True, RuntimeError),
        (pb.SANDBOX_PHASE_PROVISIONING, "helix", True, ProviderAccessRevoked),
        (pb.SANDBOX_PHASE_READY, "gateway", False, RuntimeError),
        (pb.SANDBOX_PHASE_STOPPING, "helix", False, RuntimeError),
        (pb.SANDBOX_PHASE_COMPLETED, "helix", False, RuntimeError),
        (pb.SANDBOX_PHASE_ERROR, "helix", False, RuntimeError),
    ],
)
def test_sync_renews_only_active_provider_workloads(
    mock_nemo_client, mock_platform_config, test_step_active, phase, mode, renew, error
):
    backend = _backend(
        mock_nemo_client,
        mock_platform_config,
        gateway_credentials={
            "gateway_id": "gateway",
            "registration_key_file": "/run/key",
            "credential_refresh_verified": True,
            "renewal_mode": mode,
            "profiles": {"service-read": {"workspaces": ["default"], "credential_keys": ["API_TOKEN"]}},
        },
    )
    step = test_step_active
    step.step_spec.gateway_providers = {
        "service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "token"})
    }
    backend._client.get_sandbox.return_value = _sandbox(phase)
    backend._prepare_providers = AsyncMock(side_effect=[error("sensitive RPC detail"), []])
    first = backend.sync(step)
    assert backend._prepare_providers.await_count == int(renew)
    if renew:
        assert first.status == HelixJobStatus.ACTIVE
        assert ("access was removed" if error is ProviderAccessRevoked else "retrying") in first.status_details[
            "message"
        ]
        assert "sensitive" not in str(first)
        second = backend.sync(step)
        assert "renewal failed" not in second.status_details["message"]
        assert backend._prepare_providers.call_args.kwargs == {"renew": True}
    step.status = HelixJobStatus.CANCELLING
    backend._cleanup_step_providers = MagicMock(return_value=True)
    backend._prepare_providers.reset_mock()
    backend.sync(step)
    backend._prepare_providers.assert_not_awaited()


class TestPolicy:
    def test_generated_policy_is_restricted(self, mock_nemo_client, mock_platform_config) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        policy = backend._build_executor_policy()

        assert policy.version == 1
        assert policy.process.run_as_user == "sandbox"
        assert policy.process.run_as_group == "sandbox"
        assert "/app" in policy.filesystem.read_only
        assert "/tmp" in policy.filesystem.read_write
        assert "/var/run/scratch" in policy.filesystem.read_write
        # The only network rule is the platform egress, and only the launcher and
        # the task interpreter may open it. Nothing else gets egress.
        network = policy.network_policies
        assert list(network) == ["nemo_helix"]
        binaries = [b.path for b in network["nemo_helix"].binaries]
        assert binaries == ["/tools/jobs-launcher", "/usr/local/bin/python3*", "/opt/uv/python/*/bin/python3*"]

    def test_platform_egress_null_yields_no_network_rules(self, mock_nemo_client, mock_platform_config) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config, platform_egress=None)
        policy = backend._build_executor_policy()

        assert not policy.network_policies

    def test_policy_path_is_loaded_and_egress_injected(self, mock_nemo_client, mock_platform_config, tmp_path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "version: 1\n"
            "filesystem_policy:\n"
            "  read_only: [/usr]\n"
            "  read_write: [/tmp]\n"
            "process:\n"
            "  run_as_user: worker\n"
            "  run_as_group: sandbox\n"
            "network_policies:\n"
            "  custom:\n"
            "    endpoints:\n"
            "      - host: example.com\n"
            "        port: 443\n"
        )
        backend = _backend(mock_nemo_client, mock_platform_config, policy_path=str(policy_file))
        policy = backend._build_executor_policy()

        # The hand-written rule is kept and the mandatory platform egress is injected.
        assert "custom" in policy.network_policies
        assert "nemo_helix" in policy.network_policies
        # Lists the file declares replace the defaults; sections it omits keep them.
        assert list(policy.filesystem.read_only) == ["/usr"]
        assert list(policy.filesystem.read_write) == ["/tmp"]
        assert policy.filesystem.include_workdir
        assert policy.landlock.compatibility == "best_effort"
        # A scalar the file sets overrides the generated value.
        assert policy.process.run_as_user == "worker"

    def test_network_only_policy_path_keeps_generated_contract(
        self, mock_nemo_client, mock_platform_config, tmp_path
    ) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "landlock: null\n"
            "network_policies:\n"
            "  custom:\n"
            "    endpoints:\n"
            "      - host: example.com\n"
            "        port: 443\n"
        )
        backend = _backend(mock_nemo_client, mock_platform_config, policy_path=str(policy_file))
        policy = backend._build_executor_policy()

        # The launcher's filesystem/process contract survives a file that only adds egress.
        assert "/tools" in policy.filesystem.read_only
        assert "/var/run/scratch" in policy.filesystem.read_write
        assert policy.process.run_as_user == "sandbox"
        # null keeps the generated section rather than dropping confinement.
        assert policy.landlock.compatibility == "best_effort"
        assert set(policy.network_policies) == {"custom", "nemo_helix"}


def test_backend_import_without_optional_openshell_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["openshell"] = None
from nhx.core.jobs.controllers.backends.openshell.backend import _ensure_openshell, CapabilityUnavailableError
try:
    _ensure_openshell()
except CapabilityUnavailableError:
    pass
else:
    raise AssertionError("Selecting OpenShell must report the missing optional SDK")
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("transport", ["plaintext", "system-roots", "mtls"])
async def test_provider_revocation_uses_async_transport_and_timeout(
    mock_nemo_client, mock_platform_config, tmp_path, transport
):
    key = tmp_path / "key"
    key.touch(mode=0o600)
    key.write_bytes(b"k" * 32)
    for name in ("ca", "cert", "tls-key"):
        (tmp_path / name).write_bytes(name.encode())
    backend = _backend(
        mock_nemo_client,
        mock_platform_config,
        gateway_endpoint="http://localhost:17670" if transport == "plaintext" else "https://localhost:17670",
        request_timeout_seconds=7,
        tls=OpenShellJobTLSConfig(
            ca_cert_path=str(tmp_path / "ca"),
            client_cert_path=str(tmp_path / "cert"),
            client_key_path=str(tmp_path / "tls-key"),
        )
        if transport == "mtls"
        else None,
    )
    backend._executor_config.gateway_credentials = GatewayCredentialConfig(
        gateway_id="gateway",
        registration_key_file=key,
        profiles={"service-read": {"workspaces": ["default"], "credential_keys": ["API_TOKEN"]}},
    )
    stub = MagicMock(GetSandbox=AsyncMock(), DetachSandboxProvider=AsyncMock(), DeleteProvider=AsyncMock())
    stub.GetSandbox.return_value = pb.SandboxResponse(
        sandbox=pb.Sandbox(
            metadata={"resource_version": 12},
            spec={"providers": ["provider-1"]},
        )
    )
    channel = MagicMock()
    channel.__aenter__.return_value = channel
    with (
        patch("grpc.aio.insecure_channel", return_value=channel) as insecure,
        patch("grpc.aio.secure_channel", return_value=channel) as secure,
        patch("grpc.ssl_channel_credentials") as credentials,
        patch("nemo_helix_plugin.openshell.provider_client.OpenShellStub", return_value=stub),
    ):
        async with backend._provider_provisioner() as provisioner:
            await provisioner._finish_revocation("provider-1", "sandbox-1")
    channel.__aexit__.assert_awaited_once()
    if transport == "plaintext":
        insecure.assert_called_once_with("localhost:17670")
        secure.assert_not_called()
    else:
        insecure.assert_not_called()
        secure.assert_called_once_with("localhost:17670", credentials.return_value)
        credentials.assert_called_once_with(
            root_certificates=b"ca" if transport == "mtls" else None,
            private_key=b"tls-key" if transport == "mtls" else None,
            certificate_chain=b"cert" if transport == "mtls" else None,
        )
    stub.DetachSandboxProvider.assert_awaited_once_with(
        pb.DetachSandboxProviderRequest(
            workspace_scope={"workspace": backend._workspace},
            sandbox="sandbox-1",
            provider="provider-1",
            expected_resource_version=12,
        ),
        timeout=7,
    )
    stub.DeleteProvider.assert_awaited_once_with(
        pb.DeleteProviderRequest(
            workspace_scope={"workspace": backend._workspace}, name="provider-1", allow_missing=True
        ),
        timeout=7,
    )
