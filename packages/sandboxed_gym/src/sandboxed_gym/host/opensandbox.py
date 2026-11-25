# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox backend for the job-level Gym host."""

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.error import HTTPError, URLError

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
    open_host,
    render_host_error,
)
from sandboxed_gym.opensandbox_policy import create_options_with_policy
from sandboxed_gym.quantity import quantity_bytes, quantity_gib_rounded_up
from sandboxed_gym.runtime.agent_bundle import AGENT_EXECUTABLE_ENV_KEY

LOGGER = logging.getLogger(__name__)

_HEALTH_POLL_S = 2.0
# Isolated (-I): `-m` would otherwise put the working directory, the job's writable workspace, first on the
# import path, and PYTHON* variables would apply too. The host starts as root on the Docker-enabled image.
# Unbuffered (-u), since -I also ignores the image's PYTHONUNBUFFERED.
_DEFAULT_RUNTIME_ENTRYPOINT = ("python", "-I", "-u", "-m", "sandboxed_gym.runtime.gym_host_runtime")
# Match nhx_sandbox.harbor_opensandbox_services, which this package does not depend on.
AGENT_BUNDLE_EXTENSION_KEY = "opensandbox.extensions.nemo-agent-bundle"
INSTALLED_AGENT_DIR = "/installed-agent"
#: The OpenSandbox server's auth header, and the variable the SDK reads its key from when the
#: connection names none. Spelled out so the provider still imports without the SDK.
OPEN_SANDBOX_API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"
_OPEN_SANDBOX_API_KEY_ENV = "OPEN_SANDBOX_API_KEY"


def _host_resource_limits(resources: Mapping[str, str] | None) -> dict[str, str]:
    """Turn the host size into the SDK's limit map.

    ``Sandbox.create`` substitutes ``{"cpu": "1", "memory": "2Gi"}`` when ``resource`` is omitted,
    while this host's ``resources`` are sent as requests. A request above that default is an
    invalid pod: Kubernetes requires requests to be at most the limits, and the create call then
    waits until it times out.
    """
    if not resources:
        return {}
    limits: dict[str, str] = {}
    cpu = resources.get("cpu")
    if cpu is not None and str(cpu) != "":
        limits["cpu"] = str(cpu)
    memory = resources.get("memory")
    memory_mib = resources.get("memory_mib")
    if (memory is None or str(memory) == "") and memory_mib is not None and str(memory_mib) != "":
        memory = f"{memory_mib}Mi"
    if memory is not None and str(memory) != "":
        limits["memory"] = str(memory)
    return limits


EPHEMERAL_STORAGE = "ephemeral-storage"


def host_disk_request_gib(resources: Mapping[str, str] | None, create: Mapping[str, Any]) -> int | None:
    """The host's ephemeral-storage request in whole GiB, rounded up, or ``None`` when it asks for none.

    The sandbox API takes the request in GiB. It is a request only: the scheduler places the host where that
    much node disk is free. An operator's own ``ephemeral-storage`` limit in ``create.resource`` must not be
    below it, or Kubernetes rejects the pod, which surfaces only as a create call that never finishes.
    """
    if not resources or resources.get(EPHEMERAL_STORAGE) in (None, ""):
        return None
    request = str(resources[EPHEMERAL_STORAGE])
    try:
        gib = quantity_gib_rounded_up(request)
    except ValueError as exc:
        raise ValueError(f"sandbox resource {EPHEMERAL_STORAGE}: {exc}") from None
    limit = (create.get("resource") or {}).get(EPHEMERAL_STORAGE)
    if limit is not None and gib * 2**30 > quantity_bytes(str(limit)):
        raise ValueError(
            f"sandbox resource {EPHEMERAL_STORAGE} requests {request} ({gib}Gi, the sandbox API's whole-GiB "
            f"request) but the create options limit it to {limit}"
        )
    return gib


def _create_options_for_host(create: Mapping[str, Any], resources: Mapping[str, str] | None) -> dict[str, Any]:
    """Keep a caller-supplied cap. Otherwise the host size is the limit too."""
    options = dict(create)
    if "resource" not in options:
        limits = _host_resource_limits(resources)
        if limits:
            options["resource"] = limits
    return options


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
        agent_bundle: Mapping[str, str] | None = None,
    ) -> None:
        """``agent_bundle`` is a validated ``SandboxAgentBundle`` dict: an agent CLI installed into the host.

        Gym's agents skip their own install when their CLI is on ``PATH``.
        """
        from sandboxed_gym.backends._opensandbox_driver import OpenSandboxDriver
        from sandboxed_gym.sandbox_types import SandboxHandle

        self._agent_bundle = dict(agent_bundle) if agent_bundle else None
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
            create=_create_options_for_host(
                create_options_with_policy(self._create_options, egress),
                spec.resources,
            ),
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
            disk_gib = host_disk_request_gib(spec.resources, self._create_options)
            if disk_gib is not None:
                resource_kwargs["disk_gib"] = disk_gib
            resources = SandboxResources.from_mapping(resource_kwargs)

        # OpenSandbox treats an omitted entrypoint as a request for its persistent-sandbox
        # keepalive (`tail -f /dev/null`), rather than preserving the image CMD. Start the standard
        # Gym host explicitly while still allowing runtime images to configure their own command.
        entrypoint = list(spec.entrypoint or _DEFAULT_RUNTIME_ENTRYPOINT)
        provider_options: dict[str, Any] = {"volumes": volumes}
        env = dict(spec.bootstrap_env)
        if self._agent_bundle is not None:
            # ponytail: no extension health check. A server without bundle support starts the host with no
            # agent, its --version check fails, and the readiness wait reports it. Harbor checks
            # /v1/nemo-ext/health first; do the same if that timeout turns out to be too slow a signal.
            provider_options["extensions"] = {AGENT_BUNDLE_EXTENSION_KEY: json.dumps(self._agent_bundle)}
            # The runtime checks it with --version and puts its directory first on PATH after dropping any
            # privileges; a shell launcher here would run the agent as root on the Docker-enabled image.
            env[AGENT_EXECUTABLE_ENV_KEY] = f"{INSTALLED_AGENT_DIR}/{self._agent_bundle['entrypoint']}"
        return SandboxSpec(
            image=spec.runtime_image,
            ttl_s=spec.ttl_s,
            ready_timeout_s=spec.ready_timeout_s,
            env=env,
            metadata=metadata,
            resources=resources,
            entrypoint=entrypoint,
            provider_options=provider_options,
        )

    async def _resolve_routes(self, sandbox: Any, port: int) -> _HostRoutes:
        endpoint = await sandbox.get_endpoint(port)
        base = self._absolute_url(str(endpoint.endpoint)).rstrip("/")
        return _HostRoutes(
            health_url=f"{base}/health",
            rollout_url=f"{base}/rollouts/run",
            headers={str(k): str(v) for k, v in (endpoint.headers or {}).items()},
        )

    def _proxy_credential_headers(self) -> dict[str, str]:
        """The API key for requests that reach the host through the OpenSandbox server's proxy.

        A server with tenants checks the key on proxied requests as well as on its own API, and answers
        401 without it, so the host never reports ready. A single-tenant server doesn't check, and
        ignores the header. The server drops it before forwarding, so the host never sees it.

        Same rule as the SDK's own requests to a sandbox: only through the proxy. Direct to the pod,
        the key would reach the sandbox, which runs untrusted environment code.
        """
        if not self._connection.get("use_server_proxy"):
            return {}
        api_key = self._connection.get("api_key") or os.environ.get(_OPEN_SANDBOX_API_KEY_ENV, "")
        return {OPEN_SANDBOX_API_KEY_HEADER: str(api_key)} if api_key else {}

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
            _credentials=self._proxy_credential_headers(),
        )

    async def wait_ready(self, handle: "GymHostHandle[OpenSandboxDriver]", timeout_s: float) -> None:
        """Poll ``health_url`` until status is ready."""
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                body = await asyncio.to_thread(self._get_json, handle)
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

    def _get_json(self, handle: "GymHostHandle[OpenSandboxDriver]") -> dict[str, Any]:
        try:
            with open_host(handle, handle.health_url, timeout=10) as response:
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

    async def destroy_job_sandboxes(self, job_id: str) -> tuple[str, ...]:
        """Destroy every OpenSandbox resource labeled with this job id."""
        if not job_id:
            raise ValueError("sandbox cleanup requires a job id")
        driver = self.provider_class(connection=self._connection)
        return await driver.destroy_sandboxes_matching({JOB_ID_METADATA_KEY: job_id})
