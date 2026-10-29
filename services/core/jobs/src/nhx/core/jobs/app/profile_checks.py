# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live checks of what an execution profile can run: its GPUs, and whether an image exists.

Both read the backend on each call, so the answer is current and nothing is stored.
GPUs are the inventory a profile can use, not how many are idle right now.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import Counter
from typing import Any

import httpx
from docker.errors import APIError, ContainerError, DockerException, ImageNotFound, NotFound
from docker.types import DeviceRequest
from kubernetes import client as k8s_client
from kubernetes.client.exceptions import ApiException
from nemo_helix_plugin.images import ImageIdentityError, normalize_reference
from nemo_helix_plugin.jobs.types import (
    ExecutionProfileGPU,
    ExecutionProfileGPUs,
    ExecutionProfileGPUStatus,
    ExecutionProfileImage,
    ExecutionProfileImageStatus,
)
from nemo_helix_plugin.registry_auth import RegistryAuthError, login
from nhx.common.config import get_platform_config
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends.base import resolve_task_image
from nhx.core.jobs.controllers.backends.docker import (
    NEMO_JOBS_IMAGE_REGISTRY,
    NEMO_JOBS_IMAGE_REGISTRY_PASSWORD,
    NEMO_JOBS_IMAGE_REGISTRY_USER_NAME,
)
from nhx.core.jobs.controllers.backends.kubernetes.common import (
    get_namespace_from_environment,
    load_kubernetes_config,
)

import docker

logger = logging.getLogger(__name__)

GPU_PROVIDERS = ("gpu", "gpu_distributed")
_KUBERNETES_BACKENDS = ("kubernetes_job", "volcano_job")
_DOCKER_TIMEOUT_SECONDS = 120
_REGISTRY_TIMEOUT_SECONDS = 15
_GPU_RESOURCE = "nvidia.com/gpu"
_MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def profile_gpus(profile: ExecutionProfileT) -> ExecutionProfileGPUs:
    """The GPUs *profile* can run on, read from its backend now."""
    ids = {"provider": profile.provider, "profile": profile.profile, "backend": profile.backend}
    status: ExecutionProfileGPUStatus
    try:
        if profile.backend == "docker":
            status, gpus, message = _docker_gpus(profile)
        elif profile.backend in _KUBERNETES_BACKENDS:
            status, gpus, message = _kubernetes_gpus(profile)
        else:
            status, gpus, message = "unknown", [], f"GPUs are not read for the {profile.backend} backend."
    except Exception as exc:
        logger.exception("Reading GPUs failed", extra=ids)
        status, gpus, message = "unknown", [], f"Could not read GPUs: {exc}"
    return ExecutionProfileGPUs(**ids, status=status, gpus=gpus, message=message)


def profile_image(profile: ExecutionProfileT, image: str) -> ExecutionProfileImage:
    """Whether *image* exists for *profile*, checked without pulling it."""
    ids = {"provider": profile.provider, "profile": profile.profile, "image": image}
    status: ExecutionProfileImageStatus
    try:
        if profile.backend == "docker":
            status, message = _docker_image(image)
        elif profile.backend in _KUBERNETES_BACKENDS:
            status, message = _registry_image(profile, image)
        else:
            status, message = "unknown", f"Images are not checked for the {profile.backend} backend."
    except Exception as exc:
        logger.exception("Checking the image failed", extra=ids)
        status, message = "unknown", f"Could not check the image: {exc}"
    return ExecutionProfileImage(**ids, status=status, message=message)


# --------------------------------------------------------------------------- #
# Docker                                                                      #
# --------------------------------------------------------------------------- #


def _docker_client() -> docker.DockerClient:
    """A client for the daemon the Docker backend uses, logged in to its registry the same way."""
    client = docker.from_env(timeout=_DOCKER_TIMEOUT_SECONDS)
    if NEMO_JOBS_IMAGE_REGISTRY:
        client.login(
            username=NEMO_JOBS_IMAGE_REGISTRY_USER_NAME,
            password=NEMO_JOBS_IMAGE_REGISTRY_PASSWORD,
            registry=NEMO_JOBS_IMAGE_REGISTRY,
        )
    return client


def _docker_gpus(profile: ExecutionProfileT) -> tuple[ExecutionProfileGPUStatus, list[ExecutionProfileGPU], str | None]:
    """Run ``nvidia-smi`` in a short container given the GPUs a job would get.

    The NVIDIA Container Toolkit puts ``nvidia-smi`` into any container it gives a GPU, so
    the platform's own tasks image serves, and the run also proves GPUs reach containers.
    """
    reserved = get_platform_config().docker.get_reserved_gpu_ids()
    if reserved == []:
        return "none", [], "The platform's Docker config reserves no GPUs (docker.reserved_gpu_device_ids)."
    device = (
        DeviceRequest(count=-1, capabilities=[["gpu"]])
        if reserved is None
        else DeviceRequest(device_ids=[str(gpu_id) for gpu_id in reserved], capabilities=[["gpu"]])
    )
    image = resolve_task_image(None, getattr(profile.config, "default_task_image", None))
    client = _docker_client()
    try:
        output = client.containers.run(
            image,
            entrypoint=["nvidia-smi"],
            command=["--query-gpu=name,compute_cap", "--format=csv,noheader"],
            device_requests=[device],
            environment={"NVIDIA_DRIVER_CAPABILITIES": "utility"},
            remove=True,
            stdout=True,
            stderr=False,
        )
    except ImageNotFound:
        # Before APIError, which it subclasses.
        return "unknown", [], f"The probe image {image} is not on the Docker host and could not be pulled."
    except APIError as exc:
        if "could not select device driver" in str(exc).lower():
            return "none", [], "Docker cannot give containers a GPU: the NVIDIA Container Toolkit is not set up."
        raise
    except ContainerError as exc:
        return "none", [], f"nvidia-smi found no usable GPU in a container: {exc.stderr or exc}"
    finally:
        client.close()
    counts: Counter[tuple[str, str | None]] = Counter()
    for line in output.decode().splitlines():
        if not line.strip():
            continue
        name, _, capability = (part.strip() for part in line.partition(","))
        counts[(name, capability or None)] += 1
    if not counts:
        return "none", [], "nvidia-smi listed no GPUs."
    return "available", _gpu_list(counts), None


def _docker_image(image: str) -> tuple[ExecutionProfileImageStatus, str | None]:
    client = _docker_client()
    try:
        return _docker_image_with(client, image)
    finally:
        client.close()


def _docker_image_with(client: docker.DockerClient, image: str) -> tuple[ExecutionProfileImageStatus, str | None]:
    try:
        client.images.get(image)
        return "exists", "On the Docker host."
    except ImageNotFound:
        pass
    try:
        client.images.get_registry_data(image)
        return "exists", "In its registry; pulled when a job starts."
    except NotFound:
        return "missing", f"Not on the Docker host, and its registry has no {image}."
    except APIError as exc:
        if exc.status_code in (401, 403):
            return "unknown", f"Its registry refused the Docker host's credentials: {exc.explanation}"
        raise
    except DockerException as exc:
        return "unknown", f"The Docker host could not check its registry: {exc}"


# --------------------------------------------------------------------------- #
# Kubernetes                                                                  #
# --------------------------------------------------------------------------- #


def _kubernetes_gpus(
    profile: ExecutionProfileT,
) -> tuple[ExecutionProfileGPUStatus, list[ExecutionProfileGPU], str | None]:
    """Sum allocatable GPUs on the nodes the profile's node selector matches, by GPU model.

    The model and compute capability come from the labels NVIDIA GPU Feature Discovery sets.
    """
    load_kubernetes_config()
    node_selector: dict[str, str] = getattr(profile.config, "node_selector", None) or {}
    selector = ",".join(f"{key}={value}" for key, value in sorted(node_selector.items())) or None
    try:
        nodes = k8s_client.CoreV1Api().list_node(label_selector=selector).items
    except ApiException as exc:
        if exc.status == 403:
            return "unknown", [], "The jobs API may not list nodes. The Helm chart grants this; check its RBAC."
        raise
    counts: Counter[tuple[str, str | None]] = Counter()
    for node in nodes:
        allocatable = int((node.status.allocatable or {}).get(_GPU_RESOURCE, "0") or 0)
        if not allocatable:
            continue
        labels = node.metadata.labels or {}
        major, minor = labels.get("nvidia.com/gpu.compute.major"), labels.get("nvidia.com/gpu.compute.minor")
        name = labels.get("nvidia.com/gpu.product") or "unknown model (no GPU Feature Discovery labels)"
        counts[(name, f"{major}.{minor}" if major and minor else None)] += allocatable
    if not counts:
        where = f"matching node selector {selector}" if selector else "in the cluster"
        return "none", [], f"No node {where} has allocatable {_GPU_RESOURCE}."
    return "available", _gpu_list(counts), None


def _registry_image(profile: ExecutionProfileT, image: str) -> tuple[ExecutionProfileImageStatus, str | None]:
    """Look up the image's manifest in its registry, signed in with the profile's pull secrets."""
    try:
        reference = normalize_reference(image)
    except ImageIdentityError as exc:
        return "missing", str(exc)
    registry = "registry-1.docker.io" if reference.registry == "docker.io" else reference.registry
    username, password = _pull_credentials(profile, reference.registry)
    with httpx.Client(timeout=_REGISTRY_TIMEOUT_SECONDS, follow_redirects=False) as http:
        try:
            authorization = login(registry, username, password, reference.repository, http=http, actions="pull")
            response = http.head(
                f"https://{registry}/v2/{reference.repository}/manifests/{reference.reference}",
                headers={"Accept": _MANIFEST_TYPES, **({"Authorization": authorization} if authorization else {})},
            )
        except (RegistryAuthError, httpx.HTTPError) as exc:
            return "unknown", f"Could not reach {registry}: {exc}"
    if response.status_code == 200:
        return "exists", f"In {registry}."
    if response.status_code == 404:
        return "missing", f"{registry} has no {reference.repository}:{reference.reference}."
    if response.status_code in (401, 403):
        signed_in = (
            "with the profile's pull secrets" if username else "anonymously; no pull secret has credentials for it"
        )
        return "unknown", f"{registry} refused the lookup {signed_in}."
    return "unknown", f"{registry} answered {response.status_code}."


def _pull_credentials(profile: ExecutionProfileT, registry: str) -> tuple[str | None, str | None]:
    """The username and password a pull secret of the profile or platform holds for *registry*."""
    names = [secret.name for secret in get_platform_config().image_pull_secrets]
    names += [secret.name for secret in getattr(profile.config, "image_pull_secrets", None) or []]
    namespace = getattr(profile.config, "namespace", None) or get_namespace_from_environment()
    core = k8s_client.CoreV1Api()
    for name in dict.fromkeys(names):
        try:
            secret = core.read_namespaced_secret(name=name, namespace=namespace)
        except ApiException as exc:
            logger.warning("Could not read image pull secret", extra={"secret": name, "status": exc.status})
            continue
        encoded = (secret.data or {}).get(".dockerconfigjson")
        if not encoded:
            continue
        auths: dict[str, Any] = json.loads(base64.b64decode(encoded)).get("auths", {})
        for host, entry in auths.items():
            if _registry_host(host) != registry:
                continue
            if entry.get("username") and entry.get("password"):
                return entry["username"], entry["password"]
            if entry.get("auth"):
                username, _, password = base64.b64decode(entry["auth"]).decode().partition(":")
                return username, password
    return None, None


def _registry_host(host: str) -> str:
    """A ``.dockerconfigjson`` key as a registry host: ``https://nvcr.io/v1/`` is ``nvcr.io``."""
    host = host.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
    return "docker.io" if host in ("index.docker.io", "registry-1.docker.io", "registry.hub.docker.com") else host


def _gpu_list(counts: Counter[tuple[str, str | None]]) -> list[ExecutionProfileGPU]:
    return [
        ExecutionProfileGPU(name=name, compute_capability=capability, count=count)
        for (name, capability), count in sorted(counts.items(), key=lambda item: item[0][0])
    ]
