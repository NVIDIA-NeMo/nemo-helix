# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox backend for the job-level Gym host."""

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from sandboxed_gym.backends._opensandbox_driver import OpenSandboxDriver
    from sandboxed_gym.sandbox_types import SandboxExecResult, SandboxHandle

from sandboxed_gym.config import JOB_ID_METADATA_KEY
from sandboxed_gym.egress import build_egress_policy
from sandboxed_gym.host.models import (
    GymHostBootstrapFailed,
    GymHostHandle,
    GymHostSpec,
    GymHostVolumeMount,
    render_host_error,
)
from sandboxed_gym.opensandbox_policy import create_options_with_policy

LOGGER = logging.getLogger(__name__)

_HEALTH_POLL_S = 2.0
_DEFAULT_RUNTIME_ENTRYPOINT = ("python", "-m", "sandboxed_gym.runtime.gym_host_runtime")


class _HostRoutes(NamedTuple):
    health_url: str
    rollout_url: str
    headers: dict[str, str]


class OpenSandboxGymHostProvider:
    """Provisions the job Gym host through OpenSandbox.

    Delegates create/close to :class:`OpenSandboxDriver`, which drives the OpenSandbox SDK
    directly. Owns PVC mounts, shared egress policy, and proxy URL resolution for the runtime
    HTTP port.
    """

    name = "opensandbox"
    provider_class: type["OpenSandboxDriver"]

    def __init__(
        self,
        connection: Mapping[str, Any] | None = None,
        create: Mapping[str, Any] | None = None,
        probe: Mapping[str, Any] | None = None,
        operations: Mapping[str, Any] | None = None,
    ) -> None:
        from sandboxed_gym.backends._opensandbox_driver import OpenSandboxDriver
        from sandboxed_gym.sandbox_types import SandboxHandle

        self.provider_class = OpenSandboxDriver
        self._connection = dict(connection) if connection else {}
        if "use_server_proxy" not in self._connection:
            self._connection["use_server_proxy"] = True

        self._create_options = dict(create) if create else {}
        # Large training images flake on the SDK's create-time health probe
        # (httpx.ReadError while get_endpoint races). wait_ready() polls /health
        # after routes resolve, so skip the SDK probe by default.
        self._create_options.setdefault("skip_health_check", True)
        self._probe = probe
        self._operations = operations
        self._resource_handles: dict[str, SandboxHandle] = {}

    def _provider_for_spec(self, spec: GymHostSpec) -> "OpenSandboxDriver":
        """Build a provider bound to this spec's egress policy.

        The policy is per-spec rather than per-provider, so it is not stashed on ``self``: one
        provider instance may create several hosts, and a policy read back off the provider would
        describe whichever one happened to be created last.
        """
        egress = build_egress_policy(spec.egress_allowlist)
        return self.provider_class(
            connection=self._connection,
            create=create_options_with_policy(self._create_options, egress),
            probe=self._probe,
            operations=self._operations,
        )

    def _volume_dict(self, mount: GymHostVolumeMount, name: str) -> dict[str, Any]:
        volume: dict[str, Any] = {
            "name": name,
            "mountPath": mount.mount_path,
            "readOnly": mount.read_only,
            "pvc": {
                "claimName": mount.pvc_claim,
                "createIfNotExists": False,
            },
        }
        if mount.sub_path:
            volume["subPath"] = mount.sub_path
        return volume

    def _to_sandbox_spec(self, spec: GymHostSpec):
        from sandboxed_gym.sandbox_types import SandboxResources, SandboxSpec

        volumes = [
            self._volume_dict(spec.environment_mount, "environment"),
            self._volume_dict(spec.workspace_mount, "workspace"),
        ]
        if spec.dataset_mount is not None:
            volumes.append(self._volume_dict(spec.dataset_mount, "dataset"))

        metadata = {
            JOB_ID_METADATA_KEY: spec.job_id,
            **{str(k): str(v) for k, v in spec.labels.items()},
        }

        resources = SandboxResources()
        if spec.resources:
            cpu = spec.resources.get("cpu")
            memory = spec.resources.get("memory") or spec.resources.get("memory_mib")
            resource_kwargs: dict[str, Any] = {}
            if cpu is not None:
                resource_kwargs["cpu"] = float(str(cpu).rstrip("m")) / (1000.0 if str(cpu).endswith("m") else 1.0)
            if memory is not None:
                memory_s = str(memory)
                if memory_s.endswith("Gi"):
                    resource_kwargs["memory_mib"] = int(float(memory_s[:-2]) * 1024)
                elif memory_s.endswith("Mi"):
                    resource_kwargs["memory_mib"] = int(float(memory_s[:-2]))
                else:
                    resource_kwargs["memory_mib"] = int(memory_s)
            resources = SandboxResources.from_mapping(resource_kwargs)

        # OpenSandbox treats an omitted entrypoint as a request for its persistent-sandbox
        # keepalive (`tail -f /dev/null`), rather than preserving the image CMD. Start the standard
        # Gym host explicitly while still allowing runtime images to configure their own command.
        entrypoint = spec.entrypoint or _DEFAULT_RUNTIME_ENTRYPOINT
        return SandboxSpec(
            image=spec.runtime_image,
            ttl_s=spec.ttl_s,
            ready_timeout_s=spec.ready_timeout_s,
            env=dict(spec.bootstrap_env),
            metadata=metadata,
            resources=resources,
            entrypoint=list(entrypoint),
            provider_options={"volumes": volumes},
        )

    async def _resolve_routes(self, sandbox: Any, port: int) -> _HostRoutes:
        endpoint = await sandbox.get_endpoint(port)
        base = self._absolute_url(str(endpoint.endpoint)).rstrip("/")
        return _HostRoutes(
            health_url=f"{base}/health",
            rollout_url=f"{base}/rollouts/run",
            headers={str(k): str(v) for k, v in (endpoint.headers or {}).items()},
        )

    def _absolute_url(self, endpoint: str) -> str:
        """Add http or https to a bare host:port returned by the SDK.

        ``urlopen`` needs a scheme. We default to https because that is what the SDK
        assumes. In-cluster OpenSandbox is http, so the caller must set
        ``connection.protocol`` or the health check will hang until it times out.
        """
        if "://" in endpoint:
            return endpoint
        protocol = str(self._connection.get("protocol") or "https")
        return f"{protocol}://{endpoint.lstrip('/')}"

    async def create_host(self, spec: GymHostSpec) -> "GymHostHandle[OpenSandboxDriver]":
        """Create the job host with mounts and egress, then resolve proxy URLs."""
        provider = self._provider_for_spec(spec)
        resource_handle = await provider.create(self._to_sandbox_spec(spec))
        try:
            routes = await self._resolve_routes(resource_handle.raw, spec.runtime_http_port)
        except Exception:
            await provider.close(resource_handle)
            raise
        self._resource_handles[resource_handle.sandbox_id] = resource_handle
        # The resolved URLs cannot be reconstructed from outside, and every downstream
        # connectivity failure presents only as a timeout against them.
        LOGGER.info(
            "gym host %s ready to probe: health=%s rollouts=%s",
            resource_handle.sandbox_id,
            routes.health_url,
            routes.rollout_url,
        )
        return GymHostHandle(
            host_id=resource_handle.sandbox_id,
            health_url=routes.health_url,
            rollout_url=routes.rollout_url,
            headers=routes.headers,
            provider=provider,
        )

    async def wait_ready(self, handle: "GymHostHandle[OpenSandboxDriver]", timeout_s: float) -> None:
        """Poll ``health_url`` until status is ready."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                body = await asyncio.to_thread(self._get_json, handle.health_url, handle.headers)
                if body.get("status") == "ready":
                    return
                if body.get("status") == "failed":
                    raise GymHostBootstrapFailed(
                        f"job host {handle.host_id} failed to start: {render_host_error(body.get('error'))}"
                    )
                last_error = RuntimeError(f"host not ready: {body!r}")
            except GymHostBootstrapFailed:
                raise
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(_HEALTH_POLL_S)
        # Names the URL: a scheme or port mismatch is otherwise indistinguishable from a slow
        # sandbox, and both present here as a plain timeout.
        raise TimeoutError(
            f"job host {handle.host_id} at {handle.health_url} did not become ready within {timeout_s:g}s"
            + (f": {last_error}" if last_error is not None else "")
        )

    def _get_json(self, url: str, headers: Mapping[str, str]) -> dict[str, Any]:
        request = Request(url, method="GET", headers=dict(headers))
        try:
            with urlopen(request, timeout=10) as response:
                payload = response.read()
        except HTTPError as exc:
            if exc.code == 503:
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                except Exception:
                    body = None
                return body if isinstance(body, dict) else {"status": "starting"}
            raise
        except URLError:
            raise
        return json.loads(payload.decode("utf-8"))

    async def exec_host(
        self,
        handle: "GymHostHandle[OpenSandboxDriver]",
        command: str,
        timeout_s: float | None = None,
    ) -> "SandboxExecResult":
        """Run a diagnostic command through the host's provider."""
        provider, resource_handle = self._provider_state(handle)
        return await provider.exec(resource_handle, command, timeout_s=timeout_s)

    def _require_provider(self, handle: "GymHostHandle[OpenSandboxDriver]") -> "OpenSandboxDriver":
        """Return the handle's provider, or raise if it is not one of ours."""
        provider = handle.provider
        if provider is None or not isinstance(provider, self.provider_class):
            actual = type(provider).__name__ if provider is not None else "None"
            raise TypeError(f"expected {self.provider_class.__name__}, got {actual}")
        return provider

    def _provider_state(
        self, handle: "GymHostHandle[OpenSandboxDriver]"
    ) -> tuple["OpenSandboxDriver", "SandboxHandle"]:
        provider = self._require_provider(handle)
        resource_handle = self._resource_handles.get(handle.host_id)
        if resource_handle is None:
            raise KeyError(f"no resource handle for Gym host {handle.host_id}")
        return provider, resource_handle

    async def destroy_host(self, handle: "GymHostHandle[OpenSandboxDriver]") -> None:
        """Terminate the job host."""
        resource_handle = self._resource_handles.pop(handle.host_id, None)
        if resource_handle is None:
            LOGGER.warning("destroy_host called with no resource handle for %s", handle.host_id)
            return
        provider = self._require_provider(handle)
        try:
            await provider.close(resource_handle)
        except Exception:
            LOGGER.exception("Failed to destroy job host %s", handle.host_id)
