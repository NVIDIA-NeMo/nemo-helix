# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch credential probes through the real Jobs and Deployments controllers.

Deployment configs are seeded through Entities because provider bindings are an
internal config field. Creation and deletion still use the public Deployments API;
this tests credential integration, not Agent configuration compilation.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from nemo_deployments_plugin.backends.openshell.backend import _sandbox_name as deployment_sandbox_name
from nemo_deployments_plugin.entities import Container, ContainerPort, DeploymentConfig, EnvVar, HTTPGetAction, Probe
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.deployments.client import DeploymentsClient
from nemo_helix_plugin.deployments.types import CreateDeploymentRequest
from nemo_helix_plugin.entities import SyncEntityClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nhx.core.jobs.controllers.backends.openshell.backend import _sandbox_name as job_sandbox_name

_PYTHON = "/usr/local/bin/python3.13"
_SCRIPT = "/app/credential_probe.py"


@dataclass(frozen=True)
class CredentialWorkload:
    kind: Literal["job", "deployment"]
    workspace: str
    name: str
    id: str
    sandbox_name: str


@dataclass(frozen=True)
class CredentialProbe:
    image: str
    provider_profile: str
    secret_ref: str
    service_url: str
    probe_id: str
    platform_url: str
    forbidden_helix_path: str
    interval_seconds: float = 1

    def environment(self) -> dict[str, str]:
        return {
            "SERVICE_URL": self.service_url,
            "PROBE_ID": self.probe_id,
            "NHX_BASE_URL": self.platform_url,
            "FORBIDDEN_HELIX_PATH": self.forbidden_helix_path,
            "PROBE_INTERVAL_SECONDS": str(self.interval_seconds),
            "HTTP_PROXY": "http://127.0.0.1:3128",
            "HTTPS_PROXY": "http://127.0.0.1:3128",
            "NO_PROXY": "127.0.0.1,localhost",
        }

    def providers(self) -> dict[str, GatewayProviderBinding]:
        return {
            "service": GatewayProviderBinding(profile=self.provider_profile, credentials={"API_TOKEN": self.secret_ref})
        }


def launch_credential_job(
    client: NemoClient,
    *,
    workspace: str,
    execution_profile: str,
    probe: CredentialProbe,
) -> CredentialWorkload:
    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                name=f"credential-probe-{uuid4().hex[:12]}",
                source="e2e-openshell-credentials",
                spec={},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStepSpec(
                            name="probe",
                            executor=CPUExecutionProvider(
                                provider="cpu",
                                profile=execution_profile,
                                container=ContainerSpec(image=probe.image, entrypoint=[_PYTHON], command=[_SCRIPT]),
                            ),
                            environment=[
                                HelixJobEnvironmentVariable(name=name, value=value)
                                for name, value in probe.environment().items()
                            ],
                            gateway_providers=probe.providers(),
                        )
                    ]
                ),
            ),
        )
        .data()
    )
    return CredentialWorkload(
        kind="job",
        workspace=workspace,
        name=job.name,
        id=job.id,
        sandbox_name=job_sandbox_name(workspace, job.name, job.attempt_id, "probe"),
    )


def launch_credential_deployment(
    client: NemoClient,
    *,
    entity_store_client: EntitiesClient,
    workspace: str,
    executor: str,
    probe: CredentialProbe,
) -> CredentialWorkload:
    name = f"credential-probe-{uuid4().hex[:12]}"
    entities_client = SyncEntityClient(entity_store_client)
    environment = {**probe.environment(), "PROBE_HEALTH_SERVER": "1"}
    entities_client.create(
        DeploymentConfig(
            name=name,
            workspace=workspace,
            gateway_providers=probe.providers(),
            containers=[
                Container(
                    name="probe",
                    image=probe.image,
                    command=[_PYTHON, _SCRIPT],
                    env=[EnvVar(name=name, value=value) for name, value in environment.items()],
                    ports=[ContainerPort(name="http", containerPort=8000)],
                    readinessProbe=Probe(httpGet=HTTPGetAction(path="/health", port=8000)),
                )
            ],
        )
    )
    try:
        deployment = (
            DeploymentsClient.from_client(client)
            .create_deployment(
                workspace=workspace,
                body=CreateDeploymentRequest(name=name, deployment_config=name, executor=executor),
            )
            .data()
        )
    except Exception:
        entities_client.delete(DeploymentConfig, name=name, workspace=workspace)
        raise
    return CredentialWorkload(
        kind="deployment",
        workspace=workspace,
        name=deployment.name,
        id=deployment.id,
        sandbox_name=deployment_sandbox_name(workspace, deployment.name),
    )


def stop_credential_workload(client: NemoClient, workload: CredentialWorkload) -> None:
    """Request cleanup while retaining the records the controllers need to finish it."""
    try:
        if workload.kind == "job":
            JobsClient.from_client(client).cancel_job(workspace=workload.workspace, name=workload.name)
        else:
            DeploymentsClient.from_client(client).delete_deployment(workspace=workload.workspace, name=workload.name)
    except NotFoundError:
        pass


def remove_credential_workload_metadata(client: NemoClient, workload: CredentialWorkload) -> None:
    """Call after verifying that the controller removed the sandbox and its providers."""
    try:
        if workload.kind == "job":
            JobsClient.from_client(client).delete_job(workspace=workload.workspace, name=workload.name)
        else:
            DeploymentsClient.from_client(client).delete_deployment_config(
                workspace=workload.workspace, name=workload.name
            )
    except NotFoundError:
        pass
