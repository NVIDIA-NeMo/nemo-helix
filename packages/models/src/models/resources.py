# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended ModelsResource with high-level helper methods."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from nemo_platform import APIConnectionError, APIStatusError, NotFoundError
from nemo_platform.resources.models import AsyncModelsResource as BaseAsyncModelsResource
from nemo_platform.resources.models import ModelsResource as BaseModelsResource
from nemo_platform.types.inference import ModelDeployment, ModelProvider
from nemo_platform.types.inference.gateway.openai.v1 import OpenAIModelResp
from nemo_platform.types.models import ModelEntity

_T = TypeVar("_T")
_TRANSIENT_GATEWAY_STATUS_CODES = {429, 502, 503, 504}
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedModelReference:
    """Inference route details for a workspace-qualified model reference."""

    url: str
    name: str
    host_url: str | None
    served_model_name: str | None = None


def parse_workspace_name_ref(ref: str, *, label: str, expected_format: str = "workspace/name") -> tuple[str, str]:
    """Parse a strict workspace-qualified reference."""
    workspace, separator, name = ref.partition("/")
    if separator != "/" or not workspace or not name or "/" in name:
        raise ValueError(f"{label} must be in format '{expected_format}'")
    return workspace, name


def first_provider_ref(model_providers: list[str] | None) -> tuple[str, str, str] | None:
    if not model_providers:
        return None

    provider_ref = model_providers[0]
    try:
        provider_workspace, provider_name = parse_workspace_name_ref(provider_ref, label="Provider reference")
    except ValueError:
        _logger.warning("Invalid provider reference format", extra={"provider_ref": provider_ref})
        return None
    return provider_ref, provider_workspace, provider_name


def model_entity_route_openai_url(*, base_url: str, workspace: str, name: str) -> str:
    """OpenAI SDK-compatible URL for a model-entity proxy route."""
    return f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/v1"


def resolved_model_reference(
    *,
    base_url: str,
    name: str,
    route_workspace: str,
    route_model_name: str,
    host_url: str | None,
    served_model_name: str | None = None,
) -> ResolvedModelReference:
    """Build route details for a resolved model entity."""
    return ResolvedModelReference(
        url=model_entity_route_openai_url(base_url=base_url, workspace=route_workspace, name=route_model_name),
        name=name,
        host_url=host_url,
        served_model_name=served_model_name,
    )


def served_model_name_for_entity(provider: ModelProvider, model_entity: ModelEntity) -> str | None:
    """Return the provider model id mapped to this Model Entity."""
    entity_ref = f"{model_entity.workspace}/{model_entity.name}"
    for mapping in getattr(provider, "served_models", None) or ():
        if mapping.model_entity_id == entity_ref:
            return mapping.served_model_name
    return None


def warn_provider_host_url_resolution_failure(
    provider_ref: str,
    exc: Exception,
    *,
    not_found_error_type: type[Exception],
) -> None:
    if isinstance(exc, not_found_error_type):
        _logger.warning("Provider not found during host_url resolution", extra={"provider_ref": provider_ref})
        return
    _logger.warning("Failed to resolve provider host_url", extra={"provider_ref": provider_ref}, exc_info=True)


def _seconds_since_creation(entry_timestamp: datetime | str | None, created_at: datetime | None) -> int | None:
    """Seconds from deployment creation to the entry timestamp. Returns None if either is missing or not comparable."""
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


def _openai_model_route_name(name: str, workspace: str) -> tuple[str, str]:
    model_name = name.removeprefix(f"{workspace}/")
    return model_name, f"{workspace}/{model_name}"


class _PollPending(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _poll_sleep_seconds(*, start_time: float, timeout: float, poll_interval: float) -> float | None:
    remaining_timeout = timeout - (time.time() - start_time)
    if remaining_timeout <= 0:
        return None
    return min(poll_interval, remaining_timeout)


def _poll_until_ready(
    attempt: Callable[[], _T],
    *,
    timeout: float,
    poll_interval: float,
    timeout_message: str,
) -> _T:
    start_time = time.time()
    last_error: Exception | str | None = None

    while time.time() - start_time < timeout:
        try:
            return attempt()
        except _PollPending as exc:
            last_error = exc.reason
        except NotFoundError as exc:
            last_error = exc
        except APIConnectionError as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code not in _TRANSIENT_GATEWAY_STATUS_CODES:
                raise
            last_error = exc

        sleep_seconds = _poll_sleep_seconds(
            start_time=start_time,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        if sleep_seconds is None:
            break
        time.sleep(sleep_seconds)

    raise TimeoutError(f"{timeout_message}. Last error: {last_error}")


async def _async_poll_until_ready(
    attempt: Callable[[], Awaitable[_T]],
    *,
    timeout: float,
    poll_interval: float,
    timeout_message: str,
) -> _T:
    start_time = time.time()
    last_error: Exception | str | None = None

    while time.time() - start_time < timeout:
        try:
            return await attempt()
        except _PollPending as exc:
            last_error = exc.reason
        except NotFoundError as exc:
            last_error = exc
        except APIConnectionError as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code not in _TRANSIENT_GATEWAY_STATUS_CODES:
                raise
            last_error = exc

        sleep_seconds = _poll_sleep_seconds(
            start_time=start_time,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        if sleep_seconds is None:
            break
        await asyncio.sleep(sleep_seconds)

    raise TimeoutError(f"{timeout_message}. Last error: {last_error}")


def _require_openai_model_id(model: OpenAIModelResp, expected_model_id: str) -> OpenAIModelResp:
    if model.id != expected_model_id:
        raise _PollPending(f"expected id {expected_model_id!r}, got {model.id!r}")
    return model


class ModelsResource(BaseModelsResource):
    """Extended ModelsResource with high-level helper methods.

    All existing methods (create, retrieve, list, etc.) work unchanged.
    Adds convenience methods for OpenAI integration and deployment management.

    Example:
        >>> sdk = NeMoPlatform(base_url="http://nmp-host", workspace="default")
        >>> sdk.models.get_openai_route_base_url()
        >>> sdk.models.wait_for_status("my-deployment", "READY")
    """

    def _get_base_url_str(self) -> str:
        """Get the base URL as a string with trailing slash removed."""
        return str(self._client.base_url).rstrip("/")

    def get_openai_route_base_url(self, *, workspace: str | None = None) -> str:
        """
        Generate the base URL for the OpenAI proxy route.

        This route uses the `model` field in the request body for routing,
        formatted as `workspace/model_entity_name`.

        Args:
            workspace: The workspace identifier

        Returns:
            A URL string suitable for use as OpenAI client's base_url

        Example:
            >>> base_url = sdk.models.get_openai_route_base_url()
            >>> # Returns: {base_url}/apis/inference-gateway/v2/workspaces/default/openai/-/v1
            >>> openai_client = OpenAI(base_url=base_url)
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        base_url = self._get_base_url_str()
        return f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"

    def get_client_default_headers(self) -> dict[str, str]:
        """Get string-only default headers for third-party client libraries.

        Use this helper when constructing external clients (for example OpenAI
        SDK or LiteLLM) so auth and identity headers from the SDK are forwarded.
        This is required for successful inference requests when platform auth/
        authorization is enabled.
        """
        return {key: value for key, value in self._client.default_headers.items() if isinstance(value, str)}

    def get_openai_client(self, *, workspace: str | None = None):
        """
        Get a sync OpenAI client configured for NeMo Platform's inference gateway.

        This method returns an OpenAI client with the base_url set to the
        OpenAI proxy route for the specified workspace. The client can be
        used directly with the standard OpenAI SDK interface.

        Args:
            workspace: The workspace identifier

        Returns:
            An OpenAI client configured for the inference gateway

        Example:
            >>> client = sdk.models.get_openai_client()
            >>> response = client.chat.completions.create(
            ...     model="default/meta_llama-3.2-1b-instruct",
            ...     messages=[{"role": "user", "content": "Hello!"}]
            ... )
        """
        import openai

        base_url = self.get_openai_route_base_url(workspace=workspace)
        # Preserve auth and identity headers from the parent SDK client.
        default_headers = self.get_client_default_headers()
        return openai.OpenAI(base_url=base_url, api_key="not-needed", default_headers=default_headers)

    def get_provider_route_openai_url(self, provider: ModelProvider) -> str:
        """
        Generate an OpenAI SDK-compatible URL for the provider proxy route.

        Handles the conditional /v1 suffix based on the provider's host_url:
        - If host_url ends with /v1, no suffix is added
        - Otherwise, /v1 is appended

        Args:
            provider: The ModelProvider object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url

        Example:
            >>> provider = sdk.inference.providers.retrieve("my-provider", workspace="default")
            >>> base_url = sdk.models.get_provider_route_openai_url(provider)
            >>> # Returns: {base_url}/apis/inference-gateway/v2/workspaces/default/provider/my-provider/-/v1
            >>> openai_client = OpenAI(base_url=base_url)
        """
        base_url = self._get_base_url_str()
        route = f"{base_url}/apis/inference-gateway/v2/workspaces/{provider.workspace}/provider/{provider.name}/-"

        host_url_normalized = provider.host_url.rstrip("/")
        if not host_url_normalized.endswith("/v1"):
            route = f"{route}/v1"

        return route

    def get_provider_route_openai_url_for_deployment(self, deployment: ModelDeployment) -> str:
        """
        Generate an OpenAI SDK-compatible URL for a deployment's model provider.

        This is a convenience method that fetches the ModelProvider associated
        with the deployment and returns the provider route URL.

        Args:
            deployment: The ModelDeployment object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url

        Raises:
            ValueError: If the deployment has no associated model_provider_id

        Example:
            >>> deployment = sdk.inference.deployments.retrieve("my-deployment", workspace="default")
            >>> base_url = sdk.models.get_provider_route_openai_url_for_deployment(deployment)
            >>> openai_client = OpenAI(base_url=base_url)
        """
        if not deployment.model_provider_id:
            raise ValueError(f"Deployment '{deployment.name}' has no associated model_provider_id")

        # model_provider_id is in "workspace/name" format
        workspace, name = deployment.model_provider_id.split("/", 1)
        provider = self._client.inference.providers.retrieve(name, workspace=workspace)
        return self.get_provider_route_openai_url(provider)

    def get_model_entity_route_openai_url(self, model_entity: ModelEntity) -> str:
        """
        Generate an OpenAI SDK-compatible URL for the model entity proxy route.

        Always appends /v1 suffix since the client doesn't interact directly
        with the provider's host_url.

        Args:
            model_entity: The ModelEntity object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url

        Example:
            >>> entity = sdk.models.retrieve("my-model", workspace="default")
            >>> base_url = sdk.models.get_model_entity_route_openai_url(entity)
            >>> # Returns: {base_url}/apis/inference-gateway/v2/workspaces/default/model/my-model/-/v1
            >>> openai_client = OpenAI(base_url=base_url)
        """
        return model_entity_route_openai_url(
            base_url=self._get_base_url_str(), workspace=model_entity.workspace, name=model_entity.name
        )

    def resolve_model_reference(self, ref: str) -> ResolvedModelReference:
        """Resolve ``workspace/model`` to inference-gateway route details."""
        workspace, name = parse_workspace_name_ref(ref, label="Model reference", expected_format="workspace/model_name")
        model_entity = self.retrieve(name, workspace=workspace)
        provider = self._try_resolve_model_provider_with_warning(model_entity)
        return resolved_model_reference(
            base_url=self._get_base_url_str(),
            name=name,
            route_workspace=model_entity.workspace,
            route_model_name=model_entity.name,
            host_url=provider.host_url if provider is not None else None,
            served_model_name=served_model_name_for_entity(provider, model_entity) if provider is not None else None,
        )

    def _try_resolve_model_provider_with_warning(self, model_entity: ModelEntity) -> ModelProvider | None:
        """Resolve the model entity's first provider, if available."""
        provider_parts = first_provider_ref(model_entity.model_providers)
        if provider_parts is None:
            return None
        provider_ref, provider_workspace, provider_name = provider_parts
        try:
            provider = self._client.inference.providers.retrieve(provider_name, workspace=provider_workspace)
        except Exception as exc:
            warn_provider_host_url_resolution_failure(provider_ref, exc, not_found_error_type=NotFoundError)
            return None
        return provider

    def wait_for_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
        check_gateway: bool = True,
    ) -> bool:
        """
        Wait for a ModelDeployment to reach the desired status.

        For "DELETED" status, this function waits for the resource to be fully garbage
        collected (404 NotFoundError), not just for the status to show as DELETED.

        Args:
            deployment_name: Name of the deployment
            desired_status: Target status ("READY", "DELETED", etc.)
            workspace: Workspace of the deployment
            timeout: Maximum time to wait in seconds
            check_gateway: When True and desired_status is "READY", verify the
                gateway can route to the provider before returning (default: True).

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        deployment_status = self.wait_for_deployment_status(
            deployment_name, desired_status, workspace=workspace, timeout=timeout
        )
        if not deployment_status:
            return False

        # Verify gateway can route to the provider
        if desired_status == "READY" and check_gateway:
            gateway_ready = self.wait_for_gateway(deployment_name, workspace=workspace, timeout=timeout)
            if not gateway_ready:
                return False

        return True

    def wait_for_gateway(
        self,
        provider_name: str,
        *,
        workspace: str | None = None,
        timeout: int = 60,
    ) -> bool:
        """
        Wait for the inference gateway to be able to route to a provider.

        Polls the gateway's /ready endpoint until it returns success, indicating
        the gateway has refreshed its cache and is aware of the provider.

        Args:
            provider_name: Name of the model provider
            workspace: Workspace of the provider
            timeout: Maximum time to wait in seconds

        Returns:
            True if gateway is ready, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()

        print("Waiting for gateway to be ready...")

        while time.time() - start_time < timeout:
            try:
                self._client.inference.gateway.provider.ready(
                    provider_name,
                    workspace=workspace,
                )
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"  [{timestamp}] Gateway is ready!\n")
                return True
            except NotFoundError:
                # Gateway doesn't know about the provider yet, keep waiting
                time.sleep(1)
            except Exception:
                # Connection error or other issue, keep waiting
                time.sleep(1)

        elapsed = int(time.time() - start_time)
        print(f"Gateway timeout after {elapsed}s\n")
        return False

    def wait_for_openai_model(
        self,
        name: str,
        *,
        workspace: str | None = None,
        timeout: float = 60,
        poll_interval: float = 0.5,
    ) -> OpenAIModelResp:
        """
        Wait for the inference gateway's OpenAI model route to resolve a model.

        This verifies the gateway cache used by OpenAI chat completions when the
        request model is ``workspace/model``. It is more specific than
        :meth:`wait_for_gateway`, which only checks that the provider route is
        ready.

        Args:
            name: Model entity name, with or without a ``workspace/`` prefix
            workspace: Workspace of the model
            timeout: Maximum time to wait in seconds
            poll_interval: Time between route checks in seconds

        Returns:
            The OpenAI model route response once it matches ``workspace/name``.

        Raises:
            TimeoutError: If the route does not resolve the expected model in time.
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        model_name, expected_model_id = _openai_model_route_name(name, workspace)

        def attempt() -> OpenAIModelResp:
            model = self._client.inference.gateway.openai.v1.models.get(name=model_name, workspace=workspace)
            return _require_openai_model_id(model, expected_model_id)

        return _poll_until_ready(
            attempt,
            timeout=timeout,
            poll_interval=poll_interval,
            timeout_message=f"OpenAI model {expected_model_id} not available in inference gateway after {timeout}s",
        )

    def wait_for_provider(
        self,
        provider_name: str,
        desired_status: str = "READY",
        *,
        workspace: str | None = None,
        timeout: int = 60,
        check_gateway: bool = True,
    ) -> bool:
        """
        Wait for a provider to reach the desired status.

        This is useful for external providers (like NVIDIA Build or OpenAI) where
        you need to wait for the provider to be ready before making inference calls.

        Args:
            provider_name: Name of the provider
            desired_status: Target status (default: "READY")
            workspace: Workspace of the provider
            timeout: Maximum time to wait in seconds
            check_gateway: When True and desired_status is "READY", also verify the
                gateway can route to the provider before returning (default: True).

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()
        last_status = ""

        print(f"Waiting for provider '{provider_name}' to reach status: {desired_status}...")

        while time.time() - start_time < timeout:
            try:
                provider = self._client.inference.providers.retrieve(
                    provider_name,
                    workspace=workspace,
                )
                current_status = provider.status

                if current_status != last_status:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    elapsed = int(time.time() - start_time)
                    print(f"  [{timestamp}] ({elapsed}s) Status: {current_status}")
                    last_status = current_status

                if current_status == desired_status:
                    if desired_status == "READY" and check_gateway:
                        return self.wait_for_gateway(provider_name, workspace=workspace, timeout=timeout)
                    print()
                    return True

                if current_status == "ERROR":
                    print(f"\nProvider entered ERROR state: {provider.status_message}\n")
                    return False

            except NotFoundError:
                print(f"\nProvider '{provider_name}' not found\n")
                return False

            time.sleep(1)

        elapsed = int(time.time() - start_time)
        print(f"\nProvider timeout after {elapsed}s. Last status: {last_status}\n")
        return False

    def wait_for_deployment_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
    ) -> bool:
        """
        Wait for a ModelDeployment to reach the desired status.

        For "DELETED" status, this function waits for the resource to be fully garbage
        collected (404 NotFoundError), not just for the status to show as DELETED.

        Args:
            deployment_name: Name of the deployment
            desired_status: Target status ("READY", "DELETED", etc.)
            workspace: Workspace of the deployment
            timeout: Maximum time to wait in seconds

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()
        last_status = ""
        last_message = ""
        last_history_len = 0

        print(f"Waiting for status: {desired_status}...\n")

        while time.time() - start_time < timeout:
            try:
                deployment = self._client.inference.deployments.retrieve(deployment_name, workspace=workspace)
                history = getattr(deployment, "status_history", None)
                created_at = getattr(deployment, "created_at", None)
                # API guarantees last history entry is current state; fall back to top-level fields if no history
                if history and len(history) > 0:
                    last_entry = history[-1]
                    current_status = getattr(last_entry, "status", deployment.status)
                    status_message = getattr(last_entry, "status_message", "") or ""
                else:
                    current_status = deployment.status
                    status_message = deployment.status_message or ""
                last_status = current_status
                last_message = status_message

                # Only print status from history; elapsed shown is seconds since deployment creation
                if history and len(history) > last_history_len:
                    for i in range(last_history_len, len(history)):
                        entry = history[i]
                        ts = getattr(entry, "timestamp", None)
                        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts) if ts else ""
                        st = getattr(entry, "status", "")
                        msg = getattr(entry, "status_message", "") or ""
                        secs = _seconds_since_creation(ts, created_at)
                        part = f"  [{ts_str}] "
                        if secs is not None:
                            part += f"(+{secs}s) "
                        part += f"Status: {st}"
                        if msg:
                            part += f" - {msg}"
                        print(part)
                    last_history_len = len(history)

                # Check if we've reached the desired status
                # For DELETED status, we need to wait for the actual 404 (garbage collection)
                if current_status == desired_status and desired_status != "DELETED":
                    print(f"Deployment reached {desired_status} status!\n")
                    return True

                # Handle error states
                if current_status == "ERROR":
                    print(f"Deployment entered ERROR state: {status_message}\n")
                    return False

            except NotFoundError:
                # For DELETED status, not found means success
                if desired_status == "DELETED":
                    print(f"Deployment {desired_status}!\n")
                    return True
                # For other statuses, not found is an error
                print("Deployment not found\n")
                return False

            time.sleep(3)

        # Timeout reached (wait_elapsed is time since we started polling)
        wait_elapsed = int(time.time() - start_time)
        detail = f"Last status: {last_status}"
        if last_message:
            detail += f" - {last_message}"
        print(f"Timeout after {wait_elapsed}s. {detail}\n")
        return False


class AsyncModelsResource(BaseAsyncModelsResource):
    """Extended AsyncModelsResource with high-level helper methods.

    All existing async methods (create, retrieve, list, etc.) work unchanged.
    Adds convenience methods for OpenAI integration and deployment management.

    URL builder methods are synchronous (no I/O) and safe to call from async code.
    Methods that perform I/O are properly async.

    Example:
        >>> sdk = AsyncNeMoPlatform(base_url="http://nmp-host", workspace="default")
        >>> sdk.models.get_openai_route_base_url()
        >>> await sdk.models.wait_for_status("my-deployment", "READY")
    """

    def _get_base_url_str(self) -> str:
        """Get the base URL as a string with trailing slash removed."""
        return str(self._client.base_url).rstrip("/")

    def get_openai_route_base_url(self, *, workspace: str | None = None) -> str:
        """
        Generate the base URL for the OpenAI proxy route.

        This is a synchronous method (no I/O) and safe to call from async code.

        Args:
            workspace: The workspace identifier

        Returns:
            A URL string suitable for use as OpenAI client's base_url
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        base_url = self._get_base_url_str()
        return f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"

    def get_client_default_headers(self) -> dict[str, str]:
        """Get string-only default headers for third-party client libraries.

        Use this helper when constructing external clients (for example OpenAI
        SDK or LiteLLM) so auth and identity headers from the SDK are forwarded.
        This is required for successful inference requests when platform auth/
        authorization is enabled.
        """
        return {key: value for key, value in self._client.default_headers.items() if isinstance(value, str)}

    def get_async_openai_client(self, *, workspace: str | None = None):
        """
        Get an async OpenAI client configured for NeMo Platform's inference gateway.

        This method returns an AsyncOpenAI client with the base_url set to the
        OpenAI proxy route for the specified workspace.

        Args:
            workspace: The workspace identifier

        Returns:
            An AsyncOpenAI client configured for the inference gateway

        Example:
            >>> client = sdk.models.get_async_openai_client()
            >>> response = await client.chat.completions.create(
            ...     model="default/meta_llama-3.2-1b-instruct",
            ...     messages=[{"role": "user", "content": "Hello!"}]
            ... )
        """
        import openai

        base_url = self.get_openai_route_base_url(workspace=workspace)
        # Preserve auth and identity headers from the parent SDK client.
        default_headers = self.get_client_default_headers()
        return openai.AsyncOpenAI(base_url=base_url, api_key="not-needed", default_headers=default_headers)

    def get_provider_route_openai_url(self, provider: ModelProvider) -> str:
        """
        Generate an OpenAI SDK-compatible URL for the provider proxy route.

        This is a synchronous method (no I/O) and safe to call from async code.

        Args:
            provider: The ModelProvider object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url
        """
        base_url = self._get_base_url_str()
        route = f"{base_url}/apis/inference-gateway/v2/workspaces/{provider.workspace}/provider/{provider.name}/-"

        host_url_normalized = provider.host_url.rstrip("/")
        if not host_url_normalized.endswith("/v1"):
            route = f"{route}/v1"

        return route

    async def get_provider_route_openai_url_for_deployment(self, deployment: ModelDeployment) -> str:
        """
        Generate an OpenAI SDK-compatible URL for a deployment's model provider.

        This is an async method that fetches the ModelProvider associated
        with the deployment and returns the provider route URL.

        Args:
            deployment: The ModelDeployment object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url

        Raises:
            ValueError: If the deployment has no associated model_provider_id

        Example:
            >>> deployment = await sdk.inference.deployments.retrieve("my-deployment", workspace="default")
            >>> base_url = await sdk.models.get_provider_route_openai_url_for_deployment(deployment)
            >>> openai_client = AsyncOpenAI(base_url=base_url)
        """
        if not deployment.model_provider_id:
            raise ValueError(f"Deployment '{deployment.name}' has no associated model_provider_id")

        # model_provider_id is in "workspace/name" format
        workspace, name = deployment.model_provider_id.split("/", 1)
        provider = await self._client.inference.providers.retrieve(name, workspace=workspace)
        return self.get_provider_route_openai_url(provider)

    def get_model_entity_route_openai_url(self, model_entity: ModelEntity) -> str:
        """
        Generate an OpenAI SDK-compatible URL for the model entity proxy route.

        This is a synchronous method (no I/O) and safe to call from async code.

        Args:
            model_entity: The ModelEntity object from the SDK

        Returns:
            A URL string suitable for use as OpenAI client's base_url
        """
        return model_entity_route_openai_url(
            base_url=self._get_base_url_str(), workspace=model_entity.workspace, name=model_entity.name
        )

    async def resolve_model_reference(self, ref: str) -> ResolvedModelReference:
        """Resolve ``workspace/model`` to inference-gateway route details."""
        workspace, name = parse_workspace_name_ref(ref, label="Model reference", expected_format="workspace/model_name")
        model_entity = await self.retrieve(name, workspace=workspace)
        provider = await self._try_resolve_model_provider_with_warning(model_entity)
        return resolved_model_reference(
            base_url=self._get_base_url_str(),
            name=name,
            route_workspace=model_entity.workspace,
            route_model_name=model_entity.name,
            host_url=provider.host_url if provider is not None else None,
            served_model_name=served_model_name_for_entity(provider, model_entity) if provider is not None else None,
        )

    async def _try_resolve_model_provider_with_warning(self, model_entity: ModelEntity) -> ModelProvider | None:
        """Resolve the model entity's first provider, if available."""
        provider_parts = first_provider_ref(model_entity.model_providers)
        if provider_parts is None:
            return None
        provider_ref, provider_workspace, provider_name = provider_parts
        try:
            provider = await self._client.inference.providers.retrieve(provider_name, workspace=provider_workspace)
        except Exception as exc:
            warn_provider_host_url_resolution_failure(provider_ref, exc, not_found_error_type=NotFoundError)
            return None
        return provider

    async def wait_for_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
        check_gateway: bool = True,
    ) -> bool:
        """
        Wait for a ModelDeployment and ModelProvider to reach the desired status.

        For "DELETED" status, this function waits for the resource to be fully garbage
        collected (404 NotFoundError), not just for the status to show as DELETED.

        Args:
            deployment_name: Name of the deployment
            desired_status: Target status ("READY", "DELETED", etc.)
            workspace: Workspace of the deployment
            timeout: Maximum time to wait in seconds
            check_gateway: When True and desired_status is "READY", verify the
                gateway can route to the provider before returning (default: True).

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        deployment_status = await self.wait_for_deployment_status(
            deployment_name, desired_status, workspace=workspace, timeout=timeout
        )
        if not deployment_status:
            return False

        # Verify gateway can route to the provider
        if desired_status == "READY" and check_gateway:
            gateway_ready = await self.wait_for_gateway(deployment_name, workspace=workspace, timeout=timeout)
            if not gateway_ready:
                return False

        return True

    async def wait_for_gateway(
        self,
        provider_name: str,
        *,
        workspace: str | None = None,
        timeout: int = 60,
    ) -> bool:
        """
        Wait for the inference gateway to be able to route to a provider.

        Polls the gateway's /ready endpoint until it returns success, indicating
        the gateway has refreshed its cache and is aware of the provider.

        Args:
            provider_name: Name of the model provider
            workspace: Workspace of the provider
            timeout: Maximum time to wait in seconds

        Returns:
            True if gateway is ready, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()

        print("Waiting for gateway to be ready...")

        while time.time() - start_time < timeout:
            try:
                await self._client.inference.gateway.provider.ready(
                    provider_name,
                    workspace=workspace,
                )
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"  [{timestamp}] Gateway is ready!\n")
                return True
            except NotFoundError:
                # Gateway doesn't know about the provider yet, keep waiting
                await asyncio.sleep(1)
            except Exception:
                # Connection error or other issue, keep waiting
                await asyncio.sleep(1)

        elapsed = int(time.time() - start_time)
        print(f"Gateway timeout after {elapsed}s\n")
        return False

    async def wait_for_openai_model(
        self,
        name: str,
        *,
        workspace: str | None = None,
        timeout: float = 60,
        poll_interval: float = 0.5,
    ) -> OpenAIModelResp:
        """
        Wait for the inference gateway's OpenAI model route to resolve a model.

        This verifies the gateway cache used by OpenAI chat completions when the
        request model is ``workspace/model``. It is more specific than
        :meth:`wait_for_gateway`, which only checks that the provider route is
        ready.

        Args:
            name: Model entity name, with or without a ``workspace/`` prefix
            workspace: Workspace of the model
            timeout: Maximum time to wait in seconds
            poll_interval: Time between route checks in seconds

        Returns:
            The OpenAI model route response once it matches ``workspace/name``.

        Raises:
            TimeoutError: If the route does not resolve the expected model in time.
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        model_name, expected_model_id = _openai_model_route_name(name, workspace)

        async def attempt() -> OpenAIModelResp:
            model = await self._client.inference.gateway.openai.v1.models.get(name=model_name, workspace=workspace)
            return _require_openai_model_id(model, expected_model_id)

        return await _async_poll_until_ready(
            attempt,
            timeout=timeout,
            poll_interval=poll_interval,
            timeout_message=f"OpenAI model {expected_model_id} not available in inference gateway after {timeout}s",
        )

    async def wait_for_provider(
        self,
        provider_name: str,
        desired_status: str = "READY",
        *,
        workspace: str | None = None,
        timeout: int = 60,
        check_gateway: bool = True,
    ) -> bool:
        """
        Wait for a provider to reach the desired status (async version).

        This is useful for external providers (like NVIDIA Build or OpenAI) where
        you need to wait for the provider to be ready before making inference calls.

        Args:
            provider_name: Name of the provider
            desired_status: Target status (default: "READY")
            workspace: Workspace of the provider
            timeout: Maximum time to wait in seconds
            check_gateway: When True and desired_status is "READY", also verify the
                gateway can route to the provider before returning (default: True).

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()
        last_status = ""

        print(f"Waiting for provider '{provider_name}' to reach status: {desired_status}...")

        while time.time() - start_time < timeout:
            try:
                provider = await self._client.inference.providers.retrieve(
                    provider_name,
                    workspace=workspace,
                )
                current_status = provider.status

                if current_status != last_status:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    elapsed = int(time.time() - start_time)
                    print(f"  [{timestamp}] ({elapsed}s) Status: {current_status}")
                    last_status = current_status

                if current_status == desired_status:
                    if desired_status == "READY" and check_gateway:
                        return await self.wait_for_gateway(provider_name, workspace=workspace, timeout=timeout)
                    print()
                    return True

                if current_status == "ERROR":
                    print(f"\nProvider entered ERROR state: {provider.status_message}\n")
                    return False

            except NotFoundError:
                print(f"\nProvider '{provider_name}' not found\n")
                return False

            await asyncio.sleep(1)

        elapsed = int(time.time() - start_time)
        print(f"\nProvider timeout after {elapsed}s. Last status: {last_status}\n")
        return False

    async def wait_for_deployment_status(
        self,
        deployment_name: str,
        desired_status: str,
        *,
        workspace: str | None = None,
        timeout: int = 1200,
    ) -> bool:
        """
        Wait for a ModelDeployment to reach the desired status (async version).

        For "DELETED" status, this function waits for the resource to be fully garbage
        collected (404 NotFoundError), not just for the status to show as DELETED.

        Args:
            deployment_name: Name of the deployment
            desired_status: Target status ("READY", "DELETED", etc.)
            workspace: Workspace of the deployment
            timeout: Maximum time to wait in seconds

        Returns:
            True if desired status reached, False if timeout
        """
        if workspace is None:
            workspace = self._client._get_workspace_path_param()
        start_time = time.time()
        last_status = ""
        last_message = ""
        last_history_len = 0

        print(f"Waiting for status: {desired_status}...\n")

        while time.time() - start_time < timeout:
            try:
                deployment = await self._client.inference.deployments.retrieve(deployment_name, workspace=workspace)
                history = getattr(deployment, "status_history", None)
                created_at = getattr(deployment, "created_at", None)
                # API guarantees last history entry is current state; fall back to top-level fields if no history
                if history and len(history) > 0:
                    last_entry = history[-1]
                    current_status = getattr(last_entry, "status", deployment.status)
                    status_message = getattr(last_entry, "status_message", "") or ""
                else:
                    current_status = deployment.status
                    status_message = deployment.status_message or ""
                last_status = current_status
                last_message = status_message

                # Only print status from history; elapsed shown is seconds since deployment creation
                if history and len(history) > last_history_len:
                    for i in range(last_history_len, len(history)):
                        entry = history[i]
                        ts = getattr(entry, "timestamp", None)
                        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts) if ts else ""
                        st = getattr(entry, "status", "")
                        msg = getattr(entry, "status_message", "") or ""
                        secs = _seconds_since_creation(ts, created_at)
                        part = f"  [{ts_str}] "
                        if secs is not None:
                            part += f"(+{secs}s) "
                        part += f"Status: {st}"
                        if msg:
                            part += f" - {msg}"
                        print(part)
                    last_history_len = len(history)

                # Check if we've reached the desired status
                # For DELETED status, we need to wait for the actual 404 (garbage collection)
                if current_status == desired_status and desired_status != "DELETED":
                    print(f"Deployment reached {desired_status} status!\n")
                    return True

                # Handle error states
                if current_status == "ERROR":
                    print(f"Deployment entered ERROR state: {status_message}\n")
                    return False

            except NotFoundError:
                # For DELETED status, not found means success
                if desired_status == "DELETED":
                    print(f"Deployment {desired_status}!\n")
                    return True
                # For other statuses, not found is an error
                print("Deployment not found\n")
                return False

            await asyncio.sleep(3)

        # Timeout reached (wait_elapsed is time since we started polling)
        wait_elapsed = int(time.time() - start_time)
        detail = f"Last status: {last_status}"
        if last_message:
            detail += f" - {last_message}"
        print(f"Timeout after {wait_elapsed}s. {detail}\n")
        return False
