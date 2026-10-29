# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live GPU and image checks of an execution profile, with Docker, Kubernetes and the registry mocked."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from docker.errors import APIError, ContainerError, ImageNotFound, NotFound
from httpx import AsyncClient
from kubernetes.client.exceptions import ApiException
from nemo_helix_plugin.jobs.types import ExecutionProfileGPUs, ExecutionProfileImage
from nhx.core.jobs.app import profile_checks
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.kubernetes import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
)
from nhx.core.jobs.controllers.backends.subprocess import (
    SubprocessJobExecutionProfile,
    SubprocessJobExecutionProfileConfig,
)

H100 = "NVIDIA H100 80GB HBM3"
IMAGE = "nvcr.io/nvidia/nemo:25.09"
MANIFEST = "https://nvcr.io/v2/nvidia/nemo/manifests/25.09"


def _docker_profile() -> DockerJobExecutionProfile:
    return DockerJobExecutionProfile(
        provider="gpu", profile="default", backend="docker", config=DockerJobExecutionProfileConfig()
    )


def _kubernetes_profile(**config: object) -> KubernetesJobExecutionProfile:
    return KubernetesJobExecutionProfile(
        provider="gpu", profile="default", config=KubernetesJobExecutionProfileConfig.model_validate(config)
    )


def _platform_config(*, reserved_gpu_ids: list[int] | None = None, pull_secrets: tuple[str, ...] = ()) -> object:
    return SimpleNamespace(
        docker=SimpleNamespace(get_reserved_gpu_ids=lambda: reserved_gpu_ids),
        image_pull_secrets=[SimpleNamespace(name=name) for name in pull_secrets],
    )


@pytest.fixture
def docker_client() -> Iterator[MagicMock]:
    client = MagicMock()
    with (
        patch.object(profile_checks.docker, "from_env", return_value=client),
        patch.object(profile_checks, "resolve_task_image", return_value="nemo/tasks:latest"),
        patch.object(profile_checks, "get_platform_config", return_value=_platform_config()),
    ):
        yield client


@pytest.fixture
def core_api() -> Iterator[MagicMock]:
    api = MagicMock()
    with (
        patch.object(profile_checks, "load_kubernetes_config"),
        patch.object(profile_checks.k8s_client, "CoreV1Api", return_value=api),
    ):
        yield api


def _node(gpus: int, labels: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(allocatable={"nvidia.com/gpu": str(gpus)}), metadata=SimpleNamespace(labels=labels)
    )


def _pull_secret(auths: dict[str, dict[str, str]]) -> SimpleNamespace:
    config = base64.b64encode(json.dumps({"auths": auths}).encode()).decode()
    return SimpleNamespace(data={".dockerconfigjson": config})


class TestDockerGPUs:
    def test_nvidia_smi_rows_are_counted_by_model(self, docker_client: MagicMock) -> None:
        docker_client.containers.run.return_value = f"{H100}, 9.0\n{H100}, 9.0\nNVIDIA L40S, 8.9\n".encode()

        result = profile_checks.profile_gpus(_docker_profile())

        assert result.status == "available"
        assert [(gpu.name, gpu.compute_capability, gpu.count) for gpu in result.gpus] == [
            (H100, "9.0", 2),
            ("NVIDIA L40S", "8.9", 1),
        ]
        run = docker_client.containers.run.call_args
        assert run.args == ("nemo/tasks:latest",)
        assert run.kwargs["entrypoint"] == ["nvidia-smi"]
        assert run.kwargs["device_requests"][0]["Count"] == -1
        docker_client.close.assert_called_once()

    def test_the_probe_gets_only_the_reserved_gpus(self, docker_client: MagicMock) -> None:
        docker_client.containers.run.return_value = f"{H100}, 9.0\n".encode()
        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config(reserved_gpu_ids=[2])):
            profile_checks.profile_gpus(_docker_profile())

        assert docker_client.containers.run.call_args.kwargs["device_requests"][0]["DeviceIDs"] == ["2"]

    def test_no_reserved_gpus_is_none_without_running_a_container(self, docker_client: MagicMock) -> None:
        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config(reserved_gpu_ids=[])):
            result = profile_checks.profile_gpus(_docker_profile())

        assert result.status == "none"
        assert "reserved_gpu_device_ids" in (result.message or "")
        docker_client.containers.run.assert_not_called()

    def test_a_missing_container_toolkit_is_none(self, docker_client: MagicMock) -> None:
        docker_client.containers.run.side_effect = APIError(
            'could not select device driver "" with capabilities: [[gpu]]'
        )

        result = profile_checks.profile_gpus(_docker_profile())

        assert result.status == "none"
        assert "NVIDIA Container Toolkit" in (result.message or "")

    def test_nvidia_smi_failing_is_none(self, docker_client: MagicMock) -> None:
        docker_client.containers.run.side_effect = ContainerError(
            MagicMock(), 9, "nvidia-smi", "nemo/tasks:latest", b"No devices were found"
        )

        assert profile_checks.profile_gpus(_docker_profile()).status == "none"

    def test_a_missing_probe_image_is_unknown(self, docker_client: MagicMock) -> None:
        docker_client.containers.run.side_effect = ImageNotFound("no such image")

        result = profile_checks.profile_gpus(_docker_profile())

        assert result.status == "unknown"
        assert "nemo/tasks:latest" in (result.message or "")

    def test_an_unreachable_daemon_is_unknown(self) -> None:
        with patch.object(profile_checks.docker, "from_env", side_effect=profile_checks.DockerException("no socket")):
            result = profile_checks.profile_gpus(_docker_profile())

        assert result.status == "unknown"
        assert "no socket" in (result.message or "")


class TestKubernetesGPUs:
    def test_allocatable_gpus_are_summed_by_model_from_feature_discovery_labels(self, core_api: MagicMock) -> None:
        h100 = {
            "nvidia.com/gpu.product": H100,
            "nvidia.com/gpu.compute.major": "9",
            "nvidia.com/gpu.compute.minor": "0",
        }
        core_api.list_node.return_value = SimpleNamespace(
            items=[_node(8, h100), _node(8, h100), _node(0, {}), _node(4, {})]
        )

        result = profile_checks.profile_gpus(_kubernetes_profile(node_selector={"pool": "h100", "arch": "x86"}))

        assert result.status == "available"
        assert [(gpu.name, gpu.compute_capability, gpu.count) for gpu in result.gpus] == [
            (H100, "9.0", 16),
            ("unknown model (no GPU Feature Discovery labels)", None, 4),
        ]
        core_api.list_node.assert_called_once_with(label_selector="arch=x86,pool=h100")

    def test_no_gpu_node_is_none(self, core_api: MagicMock) -> None:
        core_api.list_node.return_value = SimpleNamespace(items=[_node(0, {})])

        result = profile_checks.profile_gpus(_kubernetes_profile())

        assert result.status == "none"
        core_api.list_node.assert_called_once_with(label_selector=None)

    def test_missing_rbac_is_unknown_and_names_the_chart(self, core_api: MagicMock) -> None:
        core_api.list_node.side_effect = ApiException(status=403)

        result = profile_checks.profile_gpus(_kubernetes_profile())

        assert result.status == "unknown"
        assert "RBAC" in (result.message or "")


def test_gpus_are_not_read_for_other_backends() -> None:
    profile = SubprocessJobExecutionProfile(
        provider="subprocess", profile="local", config=SubprocessJobExecutionProfileConfig()
    )

    result = profile_checks.profile_gpus(profile)

    assert result.status == "unknown"
    assert "subprocess" in (result.message or "")


class TestDockerImage:
    def test_an_image_on_the_host_exists(self, docker_client: MagicMock) -> None:
        result = profile_checks.profile_image(_docker_profile(), IMAGE)

        assert (result.status, result.message) == ("exists", "On the Docker host.")
        docker_client.images.get_registry_data.assert_not_called()
        docker_client.close.assert_called_once()

    def test_an_image_only_in_its_registry_exists(self, docker_client: MagicMock) -> None:
        docker_client.images.get.side_effect = ImageNotFound("not here")

        assert profile_checks.profile_image(_docker_profile(), IMAGE).status == "exists"

    def test_an_image_in_neither_is_missing(self, docker_client: MagicMock) -> None:
        docker_client.images.get.side_effect = ImageNotFound("not here")
        docker_client.images.get_registry_data.side_effect = NotFound("manifest unknown")

        assert profile_checks.profile_image(_docker_profile(), IMAGE).status == "missing"

    def test_a_refused_credential_is_unknown(self, docker_client: MagicMock) -> None:
        docker_client.images.get.side_effect = ImageNotFound("not here")
        docker_client.images.get_registry_data.side_effect = APIError(
            "denied", response=SimpleNamespace(status_code=401), explanation="unauthorized"
        )

        result = profile_checks.profile_image(_docker_profile(), IMAGE)

        assert result.status == "unknown"
        assert "unauthorized" in (result.message or "")


class TestRegistryImage:
    @respx.mock
    def test_a_manifest_is_found_with_the_profile_pull_secret(self, core_api: MagicMock) -> None:
        auth = base64.b64encode(b"$oauthtoken:ngc-key").decode()
        core_api.read_namespaced_secret.return_value = _pull_secret({"https://nvcr.io/v1/": {"auth": auth}})
        respx.get("https://nvcr.io/v2/").respond(
            401, headers={"WWW-Authenticate": 'Bearer realm="https://nvcr.io/proxy_auth",service="nvcr.io"'}
        )
        token = respx.get("https://nvcr.io/proxy_auth").respond(200, json={"token": "pull-token"})
        manifest = respx.head(MANIFEST).respond(200)

        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config(pull_secrets=("ngc",))):
            result = profile_checks.profile_image(_kubernetes_profile(namespace="nemo"), IMAGE)

        assert (result.status, result.message) == ("exists", "In nvcr.io.")
        core_api.read_namespaced_secret.assert_called_once_with(name="ngc", namespace="nemo")
        assert token.calls.last.request.url.params["scope"] == "repository:nvidia/nemo:pull"
        assert token.calls.last.request.headers["Authorization"] == f"Basic {auth}"
        assert manifest.calls.last.request.headers["Authorization"] == "Bearer pull-token"

    @respx.mock
    def test_a_missing_manifest_is_missing(self, core_api: MagicMock) -> None:
        respx.get("https://nvcr.io/v2/").respond(200)
        respx.head(MANIFEST).respond(404)

        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config()):
            result = profile_checks.profile_image(_kubernetes_profile(namespace="nemo"), IMAGE)

        assert result.status == "missing"
        core_api.read_namespaced_secret.assert_not_called()

    @respx.mock
    def test_an_anonymous_refusal_says_no_pull_secret_had_credentials(self, core_api: MagicMock) -> None:
        core_api.read_namespaced_secret.side_effect = ApiException(status=404)
        respx.get("https://nvcr.io/v2/").respond(401, headers={"WWW-Authenticate": 'Basic realm="nvcr"'})
        respx.head(MANIFEST).respond(401)

        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config(pull_secrets=("gone",))):
            result = profile_checks.profile_image(_kubernetes_profile(namespace="nemo"), IMAGE)

        assert result.status == "unknown"
        assert "anonymously" in (result.message or "")

    @respx.mock
    def test_an_unreachable_registry_is_unknown(self, core_api: MagicMock) -> None:
        respx.get("https://nvcr.io/v2/").mock(side_effect=httpx.ConnectError("refused"))

        with patch.object(profile_checks, "get_platform_config", return_value=_platform_config()):
            result = profile_checks.profile_image(_kubernetes_profile(namespace="nemo"), IMAGE)

        assert result.status == "unknown"
        assert "Could not reach nvcr.io" in (result.message or "")

    def test_an_invalid_reference_is_missing(self, core_api: MagicMock) -> None:
        assert profile_checks.profile_image(_kubernetes_profile(), "Not A Valid:Ref!").status == "missing"


def test_docker_hub_pull_secret_keys_match_docker_io() -> None:
    assert profile_checks._registry_host("https://index.docker.io/v1/") == "docker.io"
    assert profile_checks._registry_host("nvcr.io") == "nvcr.io"


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #

ENDPOINTS = "nhx.core.jobs.api.v2.jobs.endpoints"


def _cpu_profile() -> DockerJobExecutionProfile:
    return DockerJobExecutionProfile(
        provider="cpu", profile="default", backend="docker", config=DockerJobExecutionProfileConfig()
    )


def _gpus(profile: DockerJobExecutionProfile) -> ExecutionProfileGPUs:
    return ExecutionProfileGPUs(
        provider=profile.provider, profile=profile.profile, backend=profile.backend, status="none"
    )


@pytest.mark.asyncio
async def test_gpus_route_reads_only_gpu_profiles(test_client: AsyncClient) -> None:
    with (
        patch(f"{ENDPOINTS}.profiles", [_cpu_profile(), _docker_profile()]),
        patch(f"{ENDPOINTS}.profile_gpus", side_effect=_gpus) as profile_gpus,
    ):
        response = await test_client.get("/apis/jobs/v2/execution-profiles/gpus")

    assert response.status_code == 200
    assert [(item["provider"], item["status"]) for item in response.json()] == [("gpu", "none")]
    profile_gpus.assert_called_once()


@pytest.mark.asyncio
async def test_gpus_route_404s_when_the_filter_matches_no_gpu_profile(test_client: AsyncClient) -> None:
    with patch(f"{ENDPOINTS}.profiles", [_cpu_profile(), _docker_profile()]):
        response = await test_client.get("/apis/jobs/v2/execution-profiles/gpus", params={"provider": "cpu"})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_image_route_checks_the_named_profile(test_client: AsyncClient) -> None:
    checked = ExecutionProfileImage(provider="gpu", profile="default", image=IMAGE, status="exists")
    with (
        patch(f"{ENDPOINTS}.profiles", [_cpu_profile(), _docker_profile()]),
        patch(f"{ENDPOINTS}.profile_image", return_value=checked) as profile_image,
    ):
        response = await test_client.get(
            "/apis/jobs/v2/execution-profiles/image", params={"provider": "gpu", "profile": "default", "image": IMAGE}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "exists"
    assert profile_image.call_args.args[0].provider == "gpu"
    assert profile_image.call_args.args[1] == IMAGE


@pytest.mark.asyncio
async def test_image_route_404s_for_an_unknown_profile(test_client: AsyncClient) -> None:
    with patch(f"{ENDPOINTS}.profiles", [_cpu_profile()]):
        response = await test_client.get(
            "/apis/jobs/v2/execution-profiles/image", params={"provider": "gpu", "profile": "default", "image": IMAGE}
        )

    assert response.status_code == 404
