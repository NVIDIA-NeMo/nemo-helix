# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenShell job backend."""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openshell")

import grpc
from nemo_helix_plugin.jobs.execution_profiles import OpenShellJobTLSConfig
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
    _LAUNCHER_PATH,
    _ensure_openshell,
    _sandbox_name,
)
from nhx.core.jobs.controllers.backends.registry import BackendKey, backend_registry
from openshell import SandboxRef, SandboxStatusRef, TlsConfig
from openshell._proto import openshell_pb2 as pb  # ty: ignore[unresolved-import]


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
    def test_creates_sandbox_with_launcher_command(
        self, mock_nemo_client, mock_platform_config, test_step_pending
    ) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        step = test_step_pending
        executor = step.step_spec.executor

        update = backend.schedule(executor, step)

        assert update.status == HelixJobStatus.PENDING
        request = _create_request(backend)
        assert request.spec.command == [_LAUNCHER_PATH, "run", "--fetch-step-config", "--"] + list(_DEFAULT_ENTRYPOINT)
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
        assert env["NEMO_JOB_STEP_CONFIG_STORAGE_PATH"] == "/var/run/scratch/config"
        assert env["PATH"].startswith("/app/.venv/bin")
        assert env["VIRTUAL_ENV"] == "/app/.venv"
        assert env["HOME"] == "/home/sandbox"
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
        assert request.spec.command == [_LAUNCHER_PATH, "run", "--fetch-step-config", "--", "/custom/bin", "--flag"]

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
        backend = _backend(mock_nemo_client, mock_platform_config, platform_egress=None)

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
        assert env["NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH"] == "/var/run/scratch/task"
        assert env["NEMO_JOB_STEP_CONFIG_STORAGE_PATH"] == "/var/run/scratch/config"

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
        import datetime

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
        import datetime

        backend = _backend(mock_nemo_client, mock_platform_config)
        sandbox = _sandbox(pb.SANDBOX_PHASE_ERROR, exit_code=1)
        recent = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        sandbox.status.conditions.add(message="timestamped").transition_time.FromDatetime(recent)
        sandbox.status.conditions.add(message="no timestamp")

        finished_at = backend._terminal_transition_time(sandbox)

        assert finished_at is not None
        assert abs((finished_at - recent).total_seconds()) < 1


class TestPolicy:
    def test_generated_policy_is_restricted(self, mock_nemo_client, mock_platform_config) -> None:
        backend = _backend(mock_nemo_client, mock_platform_config)
        policy = backend._build_executor_policy()

        assert policy.version == 1
        assert policy.process.run_as_user == "sandbox"
        assert policy.process.run_as_group == "sandbox"
        assert "/app" in policy.filesystem.read_only
        assert "/var/run/scratch" in policy.filesystem.read_write
        # The only network rule is the platform egress, and only the launcher and
        # the task interpreter may open it. Nothing else gets egress.
        network = policy.network_policies
        assert list(network) == ["nemo_helix"]
        binaries = [b.path for b in network["nemo_helix"].binaries]
        assert binaries == ["/tools/jobs-launcher", "/usr/local/bin/python3*"]

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
            "  run_as_user: sandbox\n"
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
