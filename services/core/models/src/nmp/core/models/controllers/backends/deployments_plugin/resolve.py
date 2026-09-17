# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve models API objects into compiler inputs."""

from dataclasses import dataclass
from urllib.parse import SplitResult, urljoin, urlsplit

from nemo_platform_plugin.models.types import ModelDeployment, ModelDeploymentConfig, ModelEntity
from nmp.common.config import Runtime, get_platform_config
from nmp.common.config.base import LOOPBACK_ADDRESSES, determine_loopback_override
from nmp.core.models.app import ModelWeightsType, get_model_weights_type, parse_model_name_revision
from nmp.core.models.controllers.backends.common import DeploymentConfigView, deployment_config_view
from nmp.core.models.controllers.context import ModelContext


@dataclass(frozen=True)
class ResolvedPluginDeployment:
    """All API-object data required to compile plugin entities."""

    deployment: ModelDeployment
    config: ModelDeploymentConfig
    model_entity: ModelEntity | None
    view: DeploymentConfigView
    weights_type: ModelWeightsType
    model_namespace: str | None
    model_name: str | None
    model_revision: str | None
    files_hf_url: str
    huggingface_model_puller: str
    runtime: Runtime


def _split_url(url: str) -> SplitResult | None:
    try:
        parsed = urlsplit(url)
        parsed.hostname
        parsed.port
    except ValueError:
        return None
    return parsed


def _format_netloc_with_hostname(parsed: SplitResult, hostname: str) -> str:
    host = hostname[1:-1] if hostname.startswith("[") and hostname.endswith("]") else hostname
    if ":" in host:
        host = f"[{host}]"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"

    netloc = userinfo + host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return netloc


def rewrite_loopback_for_docker_container(
    url: str,
    *,
    runtime: Runtime,
    loopback_address: str | None = None,
) -> str:
    """Rewrite loopback service URLs so Docker deployment containers can reach the host API."""
    if runtime != Runtime.DOCKER:
        return url

    parsed = _split_url(url)
    hostname = (parsed.hostname or "").lower() if parsed is not None else ""
    if parsed is None or hostname not in LOOPBACK_ADDRESSES:
        return url

    override = loopback_address or determine_loopback_override() or "host.docker.internal"
    netloc = _format_netloc_with_hostname(parsed, override)
    return parsed._replace(netloc=netloc).geturl()


def resolve_model_source(
    model_entity: ModelEntity | None, view: DeploymentConfigView
) -> tuple[str | None, str | None, str | None]:
    """Resolve file-set-backed model sources before config fallback."""
    namespace, name, revision = parse_model_name_revision(
        model_namespace=view.model_namespace, model_name=view.model_name, model_revision=view.model_revision
    )
    if model_entity and model_entity.fileset:
        parts = str(model_entity.fileset).removeprefix("hf://").removeprefix("fileset://").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1], revision
    return namespace, name, revision


def resolve_plugin_deployment(ctx: ModelContext, huggingface_model_puller: str) -> ResolvedPluginDeployment:
    """Build compiler input from a model reconciliation context."""
    if ctx.model_deployment is None or ctx.model_deployment_config is None:
        raise ValueError("Model deployment and deployment config are required.")
    view = deployment_config_view(ctx.model_deployment_config)
    namespace, name, revision = resolve_model_source(ctx.model_entity, view)
    platform_config = get_platform_config()
    files_service_url = platform_config.service_discovery.get("files") or platform_config.base_url
    files_hf_url = urljoin(files_service_url.rstrip("/") + "/", "apis/files/v2/hf")
    files_hf_url = rewrite_loopback_for_docker_container(
        files_hf_url,
        runtime=platform_config.runtime,
        loopback_address=platform_config.loopback_address,
    )
    return ResolvedPluginDeployment(
        deployment=ctx.model_deployment,
        config=ctx.model_deployment_config,
        model_entity=ctx.model_entity,
        view=view,
        weights_type=get_model_weights_type(
            model_deployment=ctx.model_deployment,
            model_deployment_config=ctx.model_deployment_config,
            model_entity=ctx.model_entity,
        ),
        model_namespace=namespace,
        model_name=name,
        model_revision=revision,
        files_hf_url=files_hf_url,
        huggingface_model_puller=huggingface_model_puller,
        runtime=platform_config.runtime,
    )
