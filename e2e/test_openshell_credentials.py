# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Secrets, controllers, driver and sandbox requests on an isolated gateway."""

import hashlib
import os
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, Protocol

import grpc
import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.deployments.client import DeploymentsClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.secrets import endpoints as secret_endpoints
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from pydantic import BaseModel, Field, SecretStr

from e2e.openshell_credential_workloads import (
    CredentialProbe,
    CredentialWorkload,
    launch_credential_deployment,
    launch_credential_job,
    remove_credential_workload_metadata,
    stop_credential_workload,
)

pytest.importorskip("openshell")
from openshell._proto import datamodel_pb2 as dm  # noqa: E402
from openshell._proto import openshell_pb2 as pb  # noqa: E402
from openshell._proto.openshell_pb2_grpc import OpenShellStub  # noqa: E402

CONFIG_PATH = os.environ.get("NHX_OPENSHELL_CREDENTIAL_TEST_CONFIG")
pytestmark = [
    pytest.mark.skipif(not CONFIG_PATH, reason="Requires an isolated OpenShell credential E2E installation"),
    pytest.mark.timeout(600),
]


class CredentialTestConfig(BaseModel):
    workspace: str
    image: str
    receiver_url: str
    sandbox_receiver_url: str
    platform_sandbox_url: str
    gateway_endpoint: str
    gateway_workspace: str = "default"
    owned_secrets_workspace: str = "openshell-internal"
    job_profile: str
    deployment_executor: str
    provider_profile: str
    cache_lifetime_seconds: int = Field(ge=1, le=60)
    timeout_seconds: int = Field(default=180, ge=1)
    version: Literal["0.1.6"]
    auth_enabled: Literal[False]


class Observation(BaseModel):
    sequence: int
    received_at: float
    probe_id: str
    pid: int
    boot_id: str
    attempt: int
    placeholder_sha256: str
    forbidden_status: int | None
    forbidden_error: str | None
    public_status: int | None
    result_status: int | None = None
    result_error: str | None = None
    accepted: bool
    version: str | None


class Heartbeat(BaseModel):
    probe_id: str
    pid: int
    boot_id: str
    placeholder_sha256: str
    completed_attempt: int
    probe_status: int | None
    probe_error: str | None


class ObservationPage(BaseModel):
    observations: list[Observation]
    latest_sequence: int
    heartbeats: list[Heartbeat]


class UnaryRpc[Request, Response](Protocol):
    def __call__(self, request: Request, *, timeout: float) -> Response: ...


class Receiver:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def expect(self, tokens: dict[str, str]) -> None:
        self.client.post("/expect", json={"tokens": tokens}).raise_for_status()

    def observations(self, probe_id: str, *, after: int = 0) -> list[Observation]:
        return [row for row in self.read(after=after).observations if row.probe_id == probe_id]

    def heartbeat(self, probe_id: str) -> Heartbeat | None:
        return next((row for row in self.read().heartbeats if row.probe_id == probe_id), None)

    def read(self, *, after: int = 0) -> ObservationPage:
        response = self.client.get("/observations", params={"after": after})
        response.raise_for_status()
        return ObservationPage.model_validate(response.json())


class GatewayObserver:
    """Read runtime evidence; workload controllers own all provider mutations."""

    def __init__(self, channel: grpc.Channel, workspace: str) -> None:
        stub = OpenShellStub(channel)
        self.scope = dm.WorkspaceSelector(workspace=workspace)
        self.list_sandboxes: UnaryRpc[pb.ListSandboxesRequest, pb.ListSandboxesResponse] = stub.ListSandboxes
        self.list_providers: UnaryRpc[pb.ListProvidersRequest, pb.ListProvidersResponse] = stub.ListProviders
        self.get_profile: UnaryRpc[pb.GetProviderProfileRequest, pb.ProviderProfileResponse] = stub.GetProviderProfile

    def require_stable_placeholder(self, profile_id: str) -> None:
        profile = self.get_profile(
            pb.GetProviderProfileRequest(workspace_scope=self.scope, id=profile_id), timeout=10
        ).profile
        credentials = [credential for credential in profile.credentials if "API_TOKEN" in credential.env_vars]
        assert credentials and all(credential.stable_placeholder for credential in credentials), (
            "The synthetic API_TOKEN profile must enable stable_placeholder for externally renewed credentials"
        )

    def sandboxes(self, workload: CredentialWorkload) -> list[pb.Sandbox]:
        selector = (
            f"nhx.nvidia.com/job_id={workload.name}"
            if workload.kind == "job"
            else f"nemo.nvidia.com/deployment-name={workload.name}"
        )
        return list(
            self.list_sandboxes(
                pb.ListSandboxesRequest(workspace_scope=self.scope, label_selector=selector), timeout=10
            ).sandboxes
        )

    def providers(self, names: set[str], *, sandbox_name: str | None = None) -> list[dm.Provider]:
        records: list[dm.Provider] = []
        token = ""
        while True:
            response = self.list_providers(
                pb.ListProvidersRequest(workspace_scope=self.scope, page_size=100, page_token=token), timeout=10
            )
            records.extend(
                provider
                for provider in response.providers
                if provider.metadata.name in names
                or (sandbox_name is not None and provider.metadata.labels.get("nemo-helix/sandbox") == sandbox_name)
            )
            token = response.next_page_token
            if not token:
                return records


def wait_until[T](
    read: Callable[[], T],
    ready: Callable[[T], bool],
    *,
    timeout: float,
    description: str,
    deadline: float | None = None,
) -> T:
    expires = time.monotonic() + timeout
    if deadline is not None:
        expires = min(expires, deadline)
    while True:
        if time.monotonic() >= expires:
            pytest.fail(f"Timed out waiting for {description}")
        result = read()
        if ready(result):
            return result
        time.sleep(max(0, min(1, expires - time.monotonic())))


@pytest.fixture(scope="module")
def credential_config() -> CredentialTestConfig:
    assert CONFIG_PATH is not None
    return CredentialTestConfig.model_validate_json(Path(CONFIG_PATH).read_text())


@pytest.fixture
def receiver(credential_config: CredentialTestConfig) -> Iterator[Receiver]:
    with httpx.Client(base_url=credential_config.receiver_url, timeout=10) as client:
        yield Receiver(client)


@pytest.fixture
def gateway(credential_config: CredentialTestConfig) -> Iterator[GatewayObserver]:
    with grpc.insecure_channel(credential_config.gateway_endpoint) as channel:
        yield GatewayObserver(channel, credential_config.gateway_workspace)


def assert_placeholder(rows: list[Observation], original: Observation, values: tuple[str, ...]) -> None:
    assert rows
    assert all(
        (row.pid, row.boot_id, row.placeholder_sha256) == (original.pid, original.boot_id, original.placeholder_sha256)
        for row in rows
    )
    assert original.placeholder_sha256 not in {hashlib.sha256(value.encode()).hexdigest() for value in values}
    # A 401 only proves missing authentication; socket denial can hide an
    # incorrectly configured probe binary rather than exercise the HTTP policy.
    assert all(row.forbidden_status == 403 for row in rows), "The sandbox must deny direct Secrets access"
    assert all(row.public_status == 200 for row in rows), "The authored external-service policy must remain usable"


def owned_secret_names(secrets_client: SecretsClient, workspace: str) -> set[str]:
    return {
        secret.name
        for secret in secrets_client.list_secrets(workspace=workspace, query_params={"page_size": 100}).items()
    }


def deployment_removed(client: NemoClient, workload: CredentialWorkload) -> bool:
    try:
        DeploymentsClient.from_client(client).get_deployment(workspace=workload.workspace, name=workload.name)
    except NotFoundError:
        return True
    return False


def cleanup_credential_workload(
    client: NemoClient,
    gateway: GatewayObserver,
    workload: CredentialWorkload,
    config: CredentialTestConfig,
    *,
    providers: set[str],
    internal_secrets_before: set[str],
    deadline: float,
) -> None:
    """Confirm runtime cleanup before removing the metadata controllers need."""
    stop_credential_workload(client, workload)
    wait_until(
        lambda: gateway.sandboxes(workload),
        lambda rows: not rows,
        timeout=config.timeout_seconds,
        description="sandbox teardown",
        deadline=deadline,
    )
    secrets = SecretsClient.from_client(client)
    wait_until(
        lambda: owned_secret_names(secrets, config.owned_secrets_workspace),
        lambda names: not names - internal_secrets_before,
        timeout=config.timeout_seconds,
        description="deletion of sandbox-owned keys from Helix Secrets",
        deadline=deadline,
    )
    wait_until(
        lambda: gateway.providers(providers, sandbox_name=workload.sandbox_name),
        lambda rows: not rows,
        timeout=config.timeout_seconds,
        description="provider teardown",
        deadline=deadline,
    )
    if workload.kind == "deployment":
        wait_until(
            lambda: deployment_removed(client, workload),
            bool,
            timeout=config.timeout_seconds,
            description="deployment record deletion",
            deadline=deadline,
        )
    remove_credential_workload_metadata(client, workload)


def workload_observations(
    client: NemoClient, receiver: Receiver, workload: CredentialWorkload, probe_id: str, *, after: int = 0
) -> list[Observation]:
    if workload.kind == "job":
        jobs = JobsClient.from_client(client)
        job = jobs.get_job(workspace=workload.workspace, name=workload.name).data()
        if job.status in {"error", "cancelled", "completed"}:
            step = jobs.get_job_step(workspace=workload.workspace, job=workload.name, name="probe").data()
            pytest.fail(f"Probe job ended: {job.status}, {step.status_details}, {step.error_details}")
    else:
        deployment = (
            DeploymentsClient.from_client(client)
            .get_deployment(workspace=workload.workspace, name=workload.name)
            .data()
        )
        assert deployment.status not in {"FAILED", "LOST", "DELETING", "SUCCEEDED"}, (
            f"Probe deployment ended: {deployment.status}, {deployment.status_message}"
        )
    return receiver.observations(probe_id, after=after)


@pytest.mark.parametrize("backend", ["job", "deployment"])
def test_original_placeholder_survives_rotation_then_secret_deletion_revokes_access(
    client: NemoClient,
    credential_config: CredentialTestConfig,
    receiver: Receiver,
    gateway: GatewayObserver,
    backend: Literal["job", "deployment"],
) -> None:
    config = credential_config
    gateway.require_stable_placeholder(config.provider_profile)
    started = time.monotonic()
    scenario_deadline = started + 360
    cleanup_deadline = started + 540
    client = client.with_options(timeout=10, retry=RetryPolicy(max_retries=0))
    probe_id = f"credential-e2e-{uuid.uuid4().hex[:12]}"
    first, rotated = f"synthetic-first-{uuid.uuid4().hex}", f"synthetic-rotated-{uuid.uuid4().hex}"
    secrets = SecretsClient.from_client(client)
    internal_secrets_before = owned_secret_names(secrets, credential_config.owned_secrets_workspace)
    secrets.create_secret(
        workspace=config.workspace, body=HelixSecretCreateRequest(name=probe_id, value=SecretStr(first))
    )
    workload: CredentialWorkload | None = None
    providers: set[str] = set()
    try:
        receiver.expect({"first": first, "rotated": rotated})
        # The authority is reachable and the value exists; a sandbox denial must
        # come from network isolation rather than a bad route or missing secret.
        assert secrets.access_secret(workspace=config.workspace, name=probe_id).data().value == first
        access_request = secret_endpoints.access_secret(workspace=config.workspace, name=probe_id)
        probe = CredentialProbe(
            image=config.image,
            provider_profile=config.provider_profile,
            secret_ref=f"{config.workspace}/{probe_id}",
            service_url=config.sandbox_receiver_url,
            probe_id=probe_id,
            platform_url=config.platform_sandbox_url,
            forbidden_helix_path=access_request.path_template.format(**access_request.path_params),
        )
        if backend == "job":
            workload = launch_credential_job(
                client, workspace=config.workspace, execution_profile=config.job_profile, probe=probe
            )
        else:
            workload = launch_credential_deployment(
                client,
                workspace=config.workspace,
                entity_store_client=EntitiesClient.from_client(client),
                executor=config.deployment_executor,
                probe=probe,
            )
        rows = wait_until(
            lambda: workload_observations(client, receiver, workload, probe_id),
            lambda rows: any(row.accepted and row.version == "first" for row in rows),
            timeout=config.timeout_seconds,
            description=f"{backend} credential injection",
            deadline=scenario_deadline,
        )
        original = next(row for row in rows if row.accepted)
        assert_placeholder(rows, original, (first, rotated))
        sandboxes = gateway.sandboxes(workload)
        assert len(sandboxes) == 1
        assert owned_secret_names(secrets, credential_config.owned_secrets_workspace) - internal_secrets_before, (
            "Sandbox keys must be stored in Helix Secrets"
        )
        providers = set(sandboxes[0].spec.providers)
        assert providers, "The actual controller must attach a provider"
        if backend == "job":
            assert original.result_status == 201, (
                f"The sandbox could not save a job result: {original.result_status} {original.result_error}"
            )
            saved = (
                JobsClient.from_client(client).list_job_results(workspace=config.workspace, name=workload.name).data()
            )
            assert any(result.name == "credential-probe" for result in saved.data), (
                "The sandbox saved a result but the platform did not record it"
            )

        rows = wait_until(
            lambda: workload_observations(client, receiver, workload, probe_id, after=original.sequence),
            lambda rows: any(
                row.accepted and row.received_at >= original.received_at + 2 * config.cache_lifetime_seconds
                for row in rows
            ),
            timeout=2 * config.cache_lifetime_seconds + config.timeout_seconds,
            description="continued injection beyond the initial credential lifetime",
            deadline=scenario_deadline,
        )
        assert_placeholder(rows, original, (first, rotated))
        assert all(row.accepted for row in rows), "Renewal interrupted requests using the original placeholder"
        assert [row.attempt for row in rows] == list(range(original.attempt + 1, rows[-1].attempt + 1)), (
            "Some probe requests failed before reaching the receiver during renewal"
        )

        secrets.update_secret(
            workspace=config.workspace, name=probe_id, body=HelixSecretUpdateRequest(value=SecretStr(rotated))
        )
        before_rotation = rows[-1].sequence
        rows = wait_until(
            lambda: workload_observations(client, receiver, workload, probe_id, after=before_rotation),
            lambda rows: any(row.accepted and row.version == "rotated" for row in rows),
            timeout=config.timeout_seconds,
            description="rotated secret reaching the original sandbox process",
            deadline=scenario_deadline,
        )
        assert_placeholder(rows, original, (first, rotated))
        receiver.expect({"rotated": rotated})
        rotated_sequence = rows[-1].sequence
        rows = wait_until(
            lambda: workload_observations(client, receiver, workload, probe_id, after=rotated_sequence),
            lambda rows: len(rows) >= 3,
            timeout=config.timeout_seconds,
            description="requests after retiring the original source credential",
            deadline=scenario_deadline,
        )
        assert_placeholder(rows, original, (first, rotated))
        assert all(row.accepted and row.version == "rotated" for row in rows)

        secrets.delete_secret(workspace=config.workspace, name=probe_id)
        revocation_deadline = min(scenario_deadline, time.monotonic() + config.cache_lifetime_seconds + 10)
        wait_until(
            lambda: gateway.providers(providers),
            lambda rows: not rows,
            timeout=config.timeout_seconds,
            description="controller removal of providers after secret deletion",
            deadline=revocation_deadline,
        )
        # Gateway acknowledgement precedes supervisor activation. Observe a
        # completed denied request before testing whether restoration revives it.
        before_restore = wait_until(
            lambda: receiver.heartbeat(probe_id),
            lambda heartbeat: (
                heartbeat is not None
                and heartbeat.completed_attempt > rows[-1].attempt
                and heartbeat.probe_status != 200
            ),
            timeout=config.timeout_seconds,
            description="supervisor enforcement of credential revocation",
            deadline=revocation_deadline,
        )
        assert before_restore is not None
        cutoff = receiver.observations(probe_id)[-1].sequence
        secrets.create_secret(
            workspace=config.workspace, body=HelixSecretCreateRequest(name=probe_id, value=SecretStr(rotated))
        )
        deadline = time.monotonic() + config.cache_lifetime_seconds
        if deadline > scenario_deadline:
            pytest.fail("Scenario deadline leaves insufficient time to observe revoked access")
        while time.monotonic() < deadline:
            assert not gateway.providers(providers), "A revoked workload's provider was recreated"
            sandboxes = gateway.sandboxes(workload)
            assert len(sandboxes) == 1, "The probe must remain alive to verify revoked requests"
            assert not sandboxes[0].spec.providers
            assert not any(row.accepted for row in receiver.observations(probe_id, after=cutoff)), (
                "Restoring a secret revived access for the revoked workload"
            )
            time.sleep(1)
        heartbeat = receiver.heartbeat(probe_id)
        assert heartbeat is not None and heartbeat.completed_attempt >= before_restore.completed_attempt + 2, (
            "The probe stopped making requests after revocation"
        )
        assert (heartbeat.pid, heartbeat.boot_id, heartbeat.placeholder_sha256) == (
            original.pid,
            original.boot_id,
            original.placeholder_sha256,
        )
        assert heartbeat.probe_status != 200, "The revoked credential still reached the receiver successfully"
    finally:
        try:
            if workload is not None:
                cleanup_credential_workload(
                    client,
                    gateway,
                    workload,
                    config,
                    providers=providers,
                    internal_secrets_before=internal_secrets_before,
                    deadline=cleanup_deadline,
                )
        finally:
            try:
                secrets.delete_secret(workspace=config.workspace, name=probe_id)
            except NotFoundError:
                pass
