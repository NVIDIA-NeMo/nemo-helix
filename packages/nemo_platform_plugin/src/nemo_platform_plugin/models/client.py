# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Models service.

Wraps the endpoint functions from ``models.endpoints`` as direct methods using
the ``method()`` descriptor (the Files/Secrets/Jobs pattern), and layers on the
Models-specific ergonomics that used to live on the vendored Stainless
``ModelsResource``:

- OpenAI inference-gateway route builders (``get_openai_route_base_url`` and
  friends) -- pure string builders, safe from sync or async code, and
- deployment/provider status polling (``wait_for_deployment_status`` /
  ``wait_for_provider_status``) driven by the client's own ``get_deployment`` /
  ``get_provider`` methods.

The inference-gateway *readiness* probe (``wait_for_gateway``) lives one layer
up in ``packages/models`` because it targets the separate inference-gateway
service, not Models -- see that module and AIRCORE notes.

Usage::

    from nemo_platform_plugin.models.client import ModelsClient
    from nemo_platform_plugin.models.types import CreateModelEntityRequest

    client = ModelsClient(base_url="...", workspace="default")
    model = client.create_model(body=CreateModelEntityRequest(name="llama")).data()
    for m in client.list_models().items():
        print(m.name)
    client.wait_for_deployment_status("my-deploy", "READY")
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Protocol

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.models import endpoints
from nemo_platform_plugin.models.refs import (
    ResolvedModelReference,
    first_provider_ref,
    model_entity_route_openai_url,
    parse_workspace_name_ref,
    resolved_model_reference,
    warn_provider_host_url_resolution_failure,
)
from nemo_platform_plugin.models.types import ModelDeployment, ModelEntity, ModelProvider

_INFERENCE_GATEWAY_PREFIX = "/apis/inference-gateway/v2/workspaces"


# The OpenAI-route builders only read a couple of attributes, so they accept any
# object exposing them -- the plugin ``ModelProvider`` / ``ModelEntity`` models,
# or the Stainless SDK equivalents that ``packages/models`` passes through.
# Structural typing keeps the plugin free of a dependency on the generated SDK
# types while still accepting them.


class ProviderLike(Protocol):
    """An object identifying a model provider and its upstream host URL."""

    @property
    def workspace(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def host_url(self) -> str: ...


class ModelEntityLike(Protocol):
    """An object identifying a model entity."""

    @property
    def workspace(self) -> str: ...
    @property
    def name(self) -> str: ...


class DeploymentLike(Protocol):
    """An object identifying a deployment and its auto-created provider."""

    @property
    def name(self) -> str: ...
    @property
    def model_provider_id(self) -> str | None: ...


def _split_provider_id(deployment: DeploymentLike) -> tuple[str, str]:
    """Split a deployment's ``model_provider_id`` into ``(workspace, name)``.

    ``model_provider_id`` is a ``workspace/name`` pair. Raise a clear ValueError
    when it is absent or missing the ``workspace/`` prefix, rather than letting an
    opaque unpack error surface.
    """
    provider_id = deployment.model_provider_id
    if not provider_id:
        raise ValueError(f"Deployment '{deployment.name}' has no associated model_provider_id")
    if "/" not in provider_id:
        raise ValueError(
            f"Deployment '{deployment.name}' has malformed model_provider_id '{provider_id}'; expected 'workspace/name'"
        )
    workspace, name = provider_id.split("/", 1)
    return workspace, name


def _served_model_name(provider: ModelProvider, model_entity: ModelEntity) -> str | None:
    """Return the provider model id mapped to this Model Entity."""
    entity_ref = f"{model_entity.workspace}/{model_entity.name}"
    for mapping in provider.served_models or ():
        if mapping.model_entity_id == entity_ref:
            return mapping.served_model_name
    return None


def _seconds_since_creation(entry_timestamp: datetime | str | None, created_at: datetime | None) -> int | None:
    """Seconds from deployment creation to the entry timestamp, or None if not comparable."""
    if created_at is None or entry_timestamp is None:
        return None
    if isinstance(entry_timestamp, str):
        try:
            entry_timestamp = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if not hasattr(entry_timestamp, "timestamp") or not hasattr(created_at, "timestamp"):
        return None
    try:
        return int(entry_timestamp.timestamp() - created_at.timestamp())
    except (TypeError, OSError):
        return None


def _deployment_status(deployment: ModelDeployment) -> tuple[str, str]:
    """Return ``(current_status, status_message)`` for a deployment.

    The API guarantees the last history entry is the current state; fall back to
    the top-level fields when there is no history.
    """
    history = deployment.status_history
    if history:
        last = history[-1]
        return last.status.value, last.status_message or ""
    return deployment.status.value, deployment.status_message or ""


def _print_new_history(deployment: ModelDeployment, last_history_len: int) -> int:
    """Print any status-history entries not yet seen; return the new history length."""
    history = deployment.status_history
    created_at = deployment.created_at
    if len(history) > last_history_len:
        for entry in history[last_history_len:]:
            ts = entry.timestamp
            ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            secs = _seconds_since_creation(ts, created_at)
            part = f"  [{ts_str}] "
            if secs is not None:
                part += f"(+{secs}s) "
            part += f"Status: {entry.status.value}"
            if entry.status_message:
                part += f" - {entry.status_message}"
            print(part)
        return len(history)
    return last_history_len


class _ModelsMethods:
    # Model entities
    create_model = method(endpoints.create_model)
    list_models = method(endpoints.list_models)
    get_model = method(endpoints.get_model)
    update_model = method(endpoints.update_model)
    delete_model = method(endpoints.delete_model)

    # Nested adapters (base model in the path)
    create_model_adapter = method(endpoints.create_model_adapter)
    update_model_adapter = method(endpoints.update_model_adapter)
    delete_model_adapter = method(endpoints.delete_model_adapter)

    # Top-level adapters
    create_adapter = method(endpoints.create_adapter)
    list_adapters = method(endpoints.list_adapters)
    get_adapter = method(endpoints.get_adapter)
    update_adapter = method(endpoints.update_adapter)
    delete_adapter = method(endpoints.delete_adapter)

    # Model providers
    create_provider = method(endpoints.create_provider)
    list_providers = method(endpoints.list_providers)
    get_provider = method(endpoints.get_provider)
    upsert_provider = method(endpoints.upsert_provider)
    update_provider_status = method(endpoints.update_provider_status)
    delete_provider = method(endpoints.delete_provider)

    # Prompts
    create_prompt = method(endpoints.create_prompt)
    list_prompts = method(endpoints.list_prompts)
    get_prompt = method(endpoints.get_prompt)
    update_prompt = method(endpoints.update_prompt)
    delete_prompt = method(endpoints.delete_prompt)

    # Model deployments
    create_deployment = method(endpoints.create_deployment)
    list_deployments = method(endpoints.list_deployments)
    get_deployment = method(endpoints.get_deployment)
    get_deployment_models = method(endpoints.get_deployment_models)
    list_deployment_versions = method(endpoints.list_deployment_versions)
    get_deployment_version = method(endpoints.get_deployment_version)
    update_deployment = method(endpoints.update_deployment)
    update_deployment_status = method(endpoints.update_deployment_status)
    delete_deployment = method(endpoints.delete_deployment)
    delete_deployment_version = method(endpoints.delete_deployment_version)

    # Model deployment configs
    create_deployment_config = method(endpoints.create_deployment_config)
    list_deployment_configs = method(endpoints.list_deployment_configs)
    get_deployment_config = method(endpoints.get_deployment_config)
    list_deployment_config_versions = method(endpoints.list_deployment_config_versions)
    get_deployment_config_version = method(endpoints.get_deployment_config_version)
    update_deployment_config = method(endpoints.update_deployment_config)
    delete_deployment_config = method(endpoints.delete_deployment_config)
    delete_deployment_config_version = method(endpoints.delete_deployment_config_version)


class _ModelsUrlMixin:
    """Pure OpenAI-route URL builders. No I/O -- safe from sync or async code.

    Depends only on the client's ``base_url`` and default ``workspace`` (both
    provided by :class:`BaseNemoClient`).
    """

    base_url: str
    workspace: str | None

    def _resolve_workspace(self, workspace: str | None) -> str:
        ws = workspace or self.workspace
        if not ws:
            raise ValueError("Missing workspace argument; either set a client-level workspace or pass workspace=...")
        return ws

    def get_openai_route_base_url(self, *, workspace: str | None = None) -> str:
        """Base URL for the OpenAI proxy route (routes on the request body ``model`` field)."""
        ws = self._resolve_workspace(workspace)
        return f"{self.base_url}/{_INFERENCE_GATEWAY_PREFIX.lstrip('/')}/{ws}/openai/-/v1"

    def get_provider_route_openai_url(self, provider: ProviderLike) -> str:
        """OpenAI SDK-compatible URL for a provider proxy route.

        Appends ``/v1`` unless the provider's ``host_url`` already ends in ``/v1``.
        """
        route = (
            f"{self.base_url}/{_INFERENCE_GATEWAY_PREFIX.lstrip('/')}/{provider.workspace}/provider/{provider.name}/-"
        )
        if not provider.host_url.rstrip("/").endswith("/v1"):
            route = f"{route}/v1"
        return route

    def get_model_entity_route_openai_url(self, model_entity: ModelEntityLike) -> str:
        """OpenAI SDK-compatible URL for a model-entity proxy route (always ``/v1``)."""
        return model_entity_route_openai_url(
            base_url=self.base_url, workspace=model_entity.workspace, name=model_entity.name
        )


class ModelsClient(_ModelsMethods, _ModelsUrlMixin, NemoClient):
    """Sync client for the Models service API."""

    def resolve_model_reference(self, ref: str) -> ResolvedModelReference:
        """Resolve ``workspace/model`` to inference-gateway route details."""
        workspace, name = parse_workspace_name_ref(ref, label="Model reference", expected_format="workspace/model_name")
        model_entity = self.get_model(name=name, workspace=workspace).data()
        provider = self._try_resolve_model_provider_with_warning(model_entity)
        return resolved_model_reference(
            base_url=self.base_url,
            name=name,
            route_workspace=model_entity.workspace,
            route_model_name=model_entity.name,
            host_url=provider.host_url if provider is not None else None,
            served_model_name=_served_model_name(provider, model_entity) if provider is not None else None,
        )

    def _try_resolve_model_provider_with_warning(self, model_entity: ModelEntity) -> ModelProvider | None:
        """Resolve the Model Entity's first provider, if available."""
        provider_parts = first_provider_ref(model_entity.model_providers)
        if provider_parts is None:
            return None
        provider_ref, provider_workspace, provider_name = provider_parts
        try:
            provider = self.get_provider(name=provider_name, workspace=provider_workspace).data()
        except Exception as exc:
            warn_provider_host_url_resolution_failure(provider_ref, exc, not_found_error_type=NotFoundError)
            return None
        return provider

    def get_provider_route_openai_url_for_deployment(self, deployment: DeploymentLike) -> str:
        """Fetch a deployment's ModelProvider and return its OpenAI route URL."""
        workspace, name = _split_provider_id(deployment)
        provider = self.get_provider(name=name, workspace=workspace).data()
        return self.get_provider_route_openai_url(provider)

    def wait_for_deployment_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
        poll_interval: float = 3.0,
    ) -> bool:
        """Poll a ModelDeployment until it reaches ``desired_status`` (or times out).

        For ``"DELETED"``, waits for the resource to be fully garbage collected
        (404), not merely for the status to read DELETED. Returns False on
        timeout or a terminal ERROR state.
        """
        start = time.time()
        last_status = ""
        last_message = ""
        last_history_len = 0
        print(f"Waiting for status: {desired_status}...\n")

        while time.time() - start < timeout:
            try:
                deployment = self.get_deployment(name=deployment_name, workspace=workspace).data()
            except NotFoundError:
                if desired_status == "DELETED":
                    print(f"Deployment {desired_status}!\n")
                    return True
                print("Deployment not found\n")
                return False

            current_status, status_message = _deployment_status(deployment)
            last_status, last_message = current_status, status_message
            last_history_len = _print_new_history(deployment, last_history_len)

            if current_status == desired_status and desired_status != "DELETED":
                print(f"Deployment reached {desired_status} status!\n")
                return True
            if current_status == "ERROR":
                print(f"Deployment entered ERROR state: {status_message}\n")
                return False
            time.sleep(poll_interval)

        detail = f"Last status: {last_status}"
        if last_message:
            detail += f" - {last_message}"
        print(f"Timeout after {int(time.time() - start)}s. {detail}\n")
        return False

    def wait_for_provider_status(
        self,
        provider_name: str,
        desired_status: str = "READY",
        *,
        workspace: str | None = None,
        timeout: int = 60,
        poll_interval: float = 1.0,
    ) -> bool:
        """Poll a ModelProvider until it reaches ``desired_status`` (or times out)."""
        start = time.time()
        last_status = ""
        print(f"Waiting for provider '{provider_name}' to reach status: {desired_status}...")

        while time.time() - start < timeout:
            try:
                provider = self.get_provider(name=provider_name, workspace=workspace).data()
            except NotFoundError:
                print(f"\nProvider '{provider_name}' not found\n")
                return False

            current_status = provider.status.value
            if current_status != last_status:
                elapsed = int(time.time() - start)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] ({elapsed}s) Status: {current_status}")
                last_status = current_status
            if current_status == desired_status:
                return True
            if current_status == "ERROR":
                print(f"\nProvider entered ERROR state: {provider.status_message}\n")
                return False
            time.sleep(poll_interval)

        print(f"\nProvider timeout after {int(time.time() - start)}s. Last status: {last_status}\n")
        return False


class AsyncModelsClient(_ModelsMethods, _ModelsUrlMixin, AsyncNemoClient):
    """Async client for the Models service API."""

    async def resolve_model_reference(self, ref: str) -> ResolvedModelReference:
        """Resolve ``workspace/model`` to inference-gateway route details."""
        workspace, name = parse_workspace_name_ref(ref, label="Model reference", expected_format="workspace/model_name")
        model_entity = (await self.get_model(name=name, workspace=workspace)).data()
        provider = await self._try_resolve_model_provider_with_warning(model_entity)
        return resolved_model_reference(
            base_url=self.base_url,
            name=name,
            route_workspace=model_entity.workspace,
            route_model_name=model_entity.name,
            host_url=provider.host_url if provider is not None else None,
            served_model_name=_served_model_name(provider, model_entity) if provider is not None else None,
        )

    async def _try_resolve_model_provider_with_warning(self, model_entity: ModelEntity) -> ModelProvider | None:
        """Resolve the Model Entity's first provider, if available."""
        provider_parts = first_provider_ref(model_entity.model_providers)
        if provider_parts is None:
            return None
        provider_ref, provider_workspace, provider_name = provider_parts
        try:
            provider = (await self.get_provider(name=provider_name, workspace=provider_workspace)).data()
        except Exception as exc:
            warn_provider_host_url_resolution_failure(provider_ref, exc, not_found_error_type=NotFoundError)
            return None
        return provider

    async def get_provider_route_openai_url_for_deployment(self, deployment: DeploymentLike) -> str:
        """Fetch a deployment's ModelProvider and return its OpenAI route URL."""
        workspace, name = _split_provider_id(deployment)
        provider = (await self.get_provider(name=name, workspace=workspace)).data()
        return self.get_provider_route_openai_url(provider)

    async def wait_for_deployment_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
        poll_interval: float = 3.0,
    ) -> bool:
        """Async twin of :meth:`ModelsClient.wait_for_deployment_status`."""
        start = time.time()
        last_status = ""
        last_message = ""
        last_history_len = 0
        print(f"Waiting for status: {desired_status}...\n")

        while time.time() - start < timeout:
            try:
                deployment = (await self.get_deployment(name=deployment_name, workspace=workspace)).data()
            except NotFoundError:
                if desired_status == "DELETED":
                    print(f"Deployment {desired_status}!\n")
                    return True
                print("Deployment not found\n")
                return False

            current_status, status_message = _deployment_status(deployment)
            last_status, last_message = current_status, status_message
            last_history_len = _print_new_history(deployment, last_history_len)

            if current_status == desired_status and desired_status != "DELETED":
                print(f"Deployment reached {desired_status} status!\n")
                return True
            if current_status == "ERROR":
                print(f"Deployment entered ERROR state: {status_message}\n")
                return False
            await asyncio.sleep(poll_interval)

        detail = f"Last status: {last_status}"
        if last_message:
            detail += f" - {last_message}"
        print(f"Timeout after {int(time.time() - start)}s. {detail}\n")
        return False

    async def wait_for_provider_status(
        self,
        provider_name: str,
        desired_status: str = "READY",
        *,
        workspace: str | None = None,
        timeout: int = 60,
        poll_interval: float = 1.0,
    ) -> bool:
        """Async twin of :meth:`ModelsClient.wait_for_provider_status`."""
        start = time.time()
        last_status = ""
        print(f"Waiting for provider '{provider_name}' to reach status: {desired_status}...")

        while time.time() - start < timeout:
            try:
                provider = (await self.get_provider(name=provider_name, workspace=workspace)).data()
            except NotFoundError:
                print(f"\nProvider '{provider_name}' not found\n")
                return False

            current_status = provider.status.value
            if current_status != last_status:
                elapsed = int(time.time() - start)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] ({elapsed}s) Status: {current_status}")
                last_status = current_status
            if current_status == desired_status:
                return True
            if current_status == "ERROR":
                print(f"\nProvider entered ERROR state: {provider.status_message}\n")
                return False
            await asyncio.sleep(poll_interval)

        print(f"\nProvider timeout after {int(time.time() - start)}s. Last status: {last_status}\n")
        return False
