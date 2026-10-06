# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the OpenShell job backend against a live gateway.

Mocks only the platform clients (there is no jobs/files API in this setup) and
drives the real backend's ``sync``, ``cleanup_steps``, and sandbox policies
against an OpenShell gateway. Skipped unless the gateway is reachable and the
openshell extra is installed.

Running a job step to COMPLETED through ``schedule`` requires the jobs-launcher
main process and the full platform (step-config fetch, secrets, OTLP export);
that is the ``nemo agents execute --profile openshell`` acceptance, not a
gateway-only test. So this suite creates sandboxes through the backend's own gRPC
stub with the same name and labels ``schedule`` uses, but a trivial command, and
asserts how the backend reads them back.
"""

from __future__ import annotations

import datetime
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

pytest.importorskip("openshell")

import grpc  # noqa: E402
import yaml  # noqa: E402
from nemo_helix_plugin.jobs.execution_profiles import OpenShellJobTLSConfig  # noqa: E402
from nhx.common.jobs.schemas import HelixJobStatus  # noqa: E402
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobStepWithContext  # noqa: E402
from nhx.core.jobs.app.constants import (  # noqa: E402
    JOB_ATTEMPT_ID_LABEL,
    JOB_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider  # noqa: E402
from nhx.core.jobs.app.schemas import HelixJobStepSpec  # noqa: E402
from nhx.core.jobs.controllers.backends.openshell import (  # noqa: E402
    OpenShellJobBackend,
    OpenShellJobExecutionProfileConfig,
)
from nhx.core.jobs.controllers.backends.openshell.backend import _sandbox_name  # noqa: E402
from nhx.core.jobs.controllers.backends.openshell.policy import PLATFORM_EGRESS_KEY  # noqa: E402
from nhx.core.jobs.controllers.backends.registry import BackendKey, backend_registry  # noqa: E402
from openshell._proto import openshell_pb2 as pb  # noqa: E402  ty: ignore[unresolved-import]

GATEWAY = os.environ.get("OPENSHELL_GATEWAY_ENDPOINT", "http://127.0.0.1:17670")
IMAGE = os.environ.get("OPENSHELL_TEST_IMAGE", "ghcr.io/nvidia/openshell-community/sandboxes/base:latest")
# Sandbox scheduling (image pull, supervisor start) dominates; allow a cold node.
TERMINAL_TIMEOUT_SECONDS = float(os.environ.get("OPENSHELL_TEST_TIMEOUT_SECONDS", "300"))
POLL_SECONDS = 2.0
# A public HTTPS host the custom-policy test allows; the default policy denies it.
EGRESS_TEST_HOST = os.environ.get("OPENSHELL_TEST_EGRESS_HOST", "example.com")
# Supervisor HTTP proxy for drivers without per-sandbox networking (docker: http://127.0.0.1:3128).
EGRESS_PROXY = os.environ.get("OPENSHELL_TEST_EGRESS_PROXY") or None
# mTLS material for an https gateway. Unset runs plaintext (or system roots for https).
TLS_CA = os.environ.get("OPENSHELL_TEST_TLS_CA")
TLS_CLIENT_CERT = os.environ.get("OPENSHELL_TEST_TLS_CLIENT_CERT")
TLS_CLIENT_KEY = os.environ.get("OPENSHELL_TEST_TLS_CLIENT_KEY")
TLS = (
    OpenShellJobTLSConfig(ca_cert_path=TLS_CA, client_cert_path=TLS_CLIENT_CERT, client_key_path=TLS_CLIENT_KEY)
    if TLS_CA or TLS_CLIENT_CERT
    else None
)


def _gateway_reachable(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(
        backend_registry.get(BackendKey("cpu", "openshell")) is not OpenShellJobBackend,
        reason="OpenShell backend not registered",
    ),
    pytest.mark.skipif(not _gateway_reachable(GATEWAY), reason=f"OpenShell gateway not reachable at {GATEWAY}"),
]


@pytest.fixture
def make_backend() -> Iterator[Callable[..., OpenShellJobBackend]]:
    """Build backends against the live gateway; deletes their sandboxes and closes them after the test."""
    backends: list[OpenShellJobBackend] = []

    def factory(**config_overrides: object) -> OpenShellJobBackend:
        settings: dict[str, object] = {"tls": TLS, "egress_proxy": EGRESS_PROXY, **config_overrides}
        config = OpenShellJobExecutionProfileConfig(gateway_endpoint=GATEWAY, image=IMAGE, **settings)
        with (
            patch("nhx.core.jobs.controllers.backends.base.client_from_platform"),
            patch(
                "nhx.core.jobs.controllers.backends.openshell.backend._resolve_jobs_controller_instance_id",
                return_value=f"itest-{uuid.uuid4().hex[:8]}",
            ),
        ):
            b = OpenShellJobBackend(MagicMock(), config, profile_name="default")
        # The jobs API is mocked; tests set the step status cleanup should observe.
        b._jobs = MagicMock()
        b._itest_created = []  # ty: ignore[unresolved-attribute]
        backends.append(b)
        return b

    try:
        yield factory
    finally:
        for b in backends:
            for name in b._itest_created:  # ty: ignore[unresolved-attribute]
                b._delete_sandbox_best_effort(name)
            b.shutdown()


@pytest.fixture
def backend(make_backend: Callable[..., OpenShellJobBackend]) -> OpenShellJobBackend:
    return make_backend()


def _step(*, status: HelixJobStatus = HelixJobStatus.PENDING) -> HelixJobStepWithContext:
    # Unique ids per test so sandbox names never collide across tests or reruns.
    suffix = uuid.uuid4().hex[:10]
    return HelixJobStepWithContext(
        id=f"itest-step-{suffix}",
        job=f"itest-job-{suffix}",
        workspace="default",
        attempt_id=f"itest-attempt-{suffix}",
        name="itest-step",
        fileset="itest-logs-fileset",
        step_spec=HelixJobStepSpec(
            name="itest-step",
            executor=CPUExecutionProvider(provider="cpu", profile="default", container=ContainerSpec(image=IMAGE)),
            config={},
            environment=[],
        ),
        status=status,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc),
    )


def _create_sandbox(backend: OpenShellJobBackend, step: HelixJobStepWithContext, command: list[str]) -> str:
    """Create a sandbox with the name and labels ``schedule`` would use, running ``command``."""
    name = _sandbox_name(step.workspace, step.job, step.attempt_id, step.name)
    labels = {
        **backend._base_controller_labels(),
        JOB_WORKSPACE_ID_LABEL: step.workspace,
        JOB_ID_LABEL: step.job,
        JOB_ATTEMPT_ID_LABEL: step.attempt_id,
        JOB_STEP_NAME_LABEL: step.name,
    }
    # Same proxy env schedule() injects, so egress behaves as it would for a real job.
    env = (
        {"HTTP_PROXY": EGRESS_PROXY, "HTTPS_PROXY": EGRESS_PROXY, "NO_PROXY": "127.0.0.1,localhost"}
        if EGRESS_PROXY
        else {}
    )
    spec = pb.SandboxSpec(
        template=pb.SandboxTemplate(image=IMAGE, environment=env),
        environment=env,
        policy=backend._policy,
        command=command,
    )
    backend._client.create(workspace=backend._workspace, spec=spec, name=name, labels=labels)
    backend._itest_created.append(name)  # ty: ignore[unresolved-attribute]
    return name


def _sync_until(backend: OpenShellJobBackend, step: HelixJobStepWithContext, statuses: set[HelixJobStatus]):
    deadline = time.monotonic() + TERMINAL_TIMEOUT_SECONDS
    update = backend.sync(step)
    while update.status not in statuses and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        update = backend.sync(step)
    assert update.status in statuses, f"expected one of {statuses}, last {update.status}: {update.status_details}"
    return update


def _sandbox_gone(backend: OpenShellJobBackend, name: str, *, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sandbox = backend._client.get_sandbox(name, workspace=backend._workspace)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                return True
            raise
        if sandbox.status.phase == pb.SANDBOX_PHASE_DELETING:
            return True
        time.sleep(POLL_SECONDS)
    return False


def test_sync_maps_zero_exit_to_completed(backend: OpenShellJobBackend) -> None:
    step = _step()
    _create_sandbox(backend, step, ["/bin/sh", "-c", "exit 0"])

    update = _sync_until(backend, step, {HelixJobStatus.COMPLETED, HelixJobStatus.ERROR})

    assert update.status == HelixJobStatus.COMPLETED, update.status_details


def test_sync_maps_nonzero_exit_to_error_with_code(backend: OpenShellJobBackend) -> None:
    step = _step()
    _create_sandbox(backend, step, ["/bin/sh", "-c", "exit 37"])

    update = _sync_until(backend, step, {HelixJobStatus.COMPLETED, HelixJobStatus.ERROR})

    assert update.status == HelixJobStatus.ERROR, update.status_details
    assert update.error_details is not None
    assert update.error_details.get("exit_code") == 37


def test_cancel_deletes_running_sandbox(backend: OpenShellJobBackend) -> None:
    step = _step()
    name = _create_sandbox(backend, step, ["/bin/sh", "-c", "sleep 600"])
    _sync_until(backend, step, {HelixJobStatus.ACTIVE})

    step.status = HelixJobStatus.CANCELLING
    update = backend.sync(step)

    assert update.status == HelixJobStatus.CANCELLED, update.status_details
    assert _sandbox_gone(backend, name), f"sandbox {name} still present after cancel"


def test_cleanup_keeps_sandbox_while_step_is_active(backend: OpenShellJobBackend) -> None:
    step = _step()
    name = _create_sandbox(backend, step, ["/bin/sh", "-c", "sleep 600"])
    _sync_until(backend, step, {HelixJobStatus.ACTIVE})
    backend._jobs.get_job_step.return_value.data.return_value = MagicMock(status="active")

    backend.cleanup_steps()

    sandbox = backend._client.get_sandbox(name, workspace=backend._workspace)
    assert sandbox.status.phase != pb.SANDBOX_PHASE_DELETING


def test_cleanup_reclaims_completed_sandbox_for_terminal_step(backend: OpenShellJobBackend) -> None:
    step = _step()
    name = _create_sandbox(backend, step, ["/bin/sh", "-c", "exit 0"])
    _sync_until(backend, step, {HelixJobStatus.COMPLETED})
    backend._jobs.get_job_step.return_value.data.return_value = MagicMock(status="completed")

    backend.cleanup_steps()

    assert _sandbox_gone(backend, name), f"completed sandbox {name} not reclaimed by cleanup"


# Fetches EGRESS_TEST_HOST over HTTPS; exits 0 only when the request goes through.
_FETCH_EGRESS_HOST = [
    "/usr/local/bin/python3",
    "-c",
    "import sys, urllib.request; "
    f"sys.exit(0 if urllib.request.urlopen('https://{EGRESS_TEST_HOST}', timeout=15).status == 200 else 3)",
]


def _write_egress_policy(tmp_path: Path) -> str:
    """A user policy that only adds one network rule; everything else is left to defaults."""
    path = tmp_path / "policy.yaml"
    rule = {
        "name": "egress-test-host",
        "endpoints": [{"host": EGRESS_TEST_HOST, "port": 443, "protocol": "", "tls": "skip", "access": "full"}],
        "binaries": [{"path": "/usr/local/bin/python3*"}],
    }
    path.write_text(yaml.safe_dump({"network_policies": {"egress_test_host": rule}}))
    return str(path)


def test_custom_policy_allows_declared_egress(make_backend, tmp_path: Path) -> None:
    custom = make_backend(policy_path=_write_egress_policy(tmp_path))
    # The user rule is added alongside the platform rule, never in place of it.
    assert set(custom._policy.network_policies) == {"egress_test_host", PLATFORM_EGRESS_KEY}

    step = _step()
    _create_sandbox(custom, step, _FETCH_EGRESS_HOST)
    update = _sync_until(custom, step, {HelixJobStatus.COMPLETED, HelixJobStatus.ERROR})

    assert update.status == HelixJobStatus.COMPLETED, update.status_details


def test_default_policy_denies_undeclared_egress(make_backend, tmp_path: Path) -> None:
    # A custom-policy profile existing alongside must not loosen the default profile.
    make_backend(policy_path=_write_egress_policy(tmp_path))
    default = make_backend()
    assert set(default._policy.network_policies) == {PLATFORM_EGRESS_KEY}

    step = _step()
    _create_sandbox(default, step, _FETCH_EGRESS_HOST)
    update = _sync_until(default, step, {HelixJobStatus.COMPLETED, HelixJobStatus.ERROR})

    assert update.status == HelixJobStatus.ERROR, update.status_details
    assert update.error_details is not None
    assert update.error_details.get("exit_code") not in (None, 0)


@pytest.mark.skipif(TLS is None or TLS.client_cert_path is None, reason="needs an mTLS gateway (OPENSHELL_TEST_TLS_*)")
def test_mtls_gateway_rejects_client_without_certificate(make_backend) -> None:
    # Trusts the gateway but presents no client identity, so the gateway must refuse it.
    anonymous = make_backend(tls=OpenShellJobTLSConfig(ca_cert_path=TLS_CA))

    with pytest.raises(grpc.RpcError) as exc_info:
        anonymous._client.list_all(workspace=anonymous._workspace)

    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
