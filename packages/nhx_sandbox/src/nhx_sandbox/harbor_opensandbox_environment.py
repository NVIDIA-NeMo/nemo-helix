# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo's Harbor environment for running scaled-eval trials in OpenSandbox.

This is a ``BaseEnvironment`` written directly on the OpenSandbox SDK (through
``nhx_sandbox.harbor_opensandbox_driver``), not a subclass of Harbor's own OpenSandbox
environment. It uses only the parts of Harbor's environment API that every supported runner
shares, so the same class runs on Harbor 0.13.2 and 0.20.0. What it adds over a plain sandbox:

* Egress comes from a trusted allowlist the runner renders into Harbor's environment kwargs, never
  from ``task.toml``. ``no-network`` tasks get deny-all, ``public`` tasks get the trusted allowlist
  (never unrestricted access), and ``allowlist`` tasks must ask for a subset of it or are rejected
  before any sandbox exists.
* The applied policy is read back after create and the sandbox is killed if it does not match, or
  if it cannot be read at all.
* Every create attempt is labelled, so a sandbox the server created but whose handle was lost to a
  failed or retried request can still be found and killed.
* Each sandbox's writable storage is capped at the task's ``storage_mb`` through an
  ``ephemeral-storage`` limit. OpenSandbox has no disk limit of its own, so without this a task that
  fills its disk fills the node's. Kubernetes evicts a pod that exceeds the limit.
* On an image with a non-root ``USER``, Harbor's ``user="root"`` commands run as the image's user.
  ``execd`` runs as that user and the kernel refuses its switch to uid 0, so asking for root fails
  every such command before it starts, even checks like ``tmux -V``.
* Compose tasks run when the runner passes ``compose_services`` (see
  ``nhx_sandbox.harbor_opensandbox_services``). The server's NeMo extension adds the services to
  the sandbox pod, and Kubernetes starts them before ``main``. This class gives ``main`` its
  entrypoint and env, fails the create early when a service can't start, waits for ``main``'s and
  the services' readiness, and runs per-service operations through the extension's exec route.
* An agent bundle, when the runner passes ``agent_bundle``, is installed into the agent's sandbox
  by the same extension, before ``main`` starts. This class puts it on ``PATH`` and checks it runs,
  so Harbor's installed agents skip their root install and work on non-root images too.

Loaded by Harbor through ``environment.import_path``; see ``NEMO_OPENSANDBOX_IMPORT_PATH`` in
``scaled_evals.dispatch.harbor_opensandbox``. This is the interim home until ``nhx-sandbox`` owns
OpenSandbox access.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import ipaddress
import json
import logging
import math
import os
import random
import shlex
import stat
import tarfile
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

import httpx
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from nhx_sandbox.backends.opensandbox.termination import OpenSandboxTerminationDiagnostics
from nhx_sandbox.compose_env import expand_env
from nhx_sandbox.core.termination import check_termination, write_termination
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from nhx_sandbox.harbor_opensandbox_cleanup import applied_egress_filename
from nhx_sandbox.harbor_opensandbox_driver import (
    CREATE_ATTEMPT_METADATA_KEY,
    Connection,
    CreateRequest,
    OpenSandboxDriver,
    sandbox_id,
)
from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_EXTENSION_KEY,
    AGENT_BUNDLE_FEATURE,
    COMPOSE_FILENAMES,
    COMPOSE_MAIN_FEATURE,
    EXTENSION_KEY,
    MAIN_SERVICE,
    ComposeServices,
    SandboxAgentBundle,
    keep_alive_entrypoint,
    parse_compose_services,
)
from nhx_sandbox.opensandbox_policy import (
    EgressVerificationError,
    EgressVerificationMode,
    canonical_egress_target,
    to_opensandbox_policy,
    verify_applied_egress,
)

try:
    from harbor.environments.base import ServiceOperationsUnsupportedError
except ImportError:
    # Harbor 0.13.2 has no per-service operations, so nothing there catches this error by type.
    ServiceOperationsUnsupportedError = RuntimeError

__all__ = [
    "CREATE_ATTEMPT_METADATA_KEY",
    "ENVIRONMENT_TYPE",
    "MANAGED_BY_METADATA_KEY",
    "MANAGED_BY_METADATA_VALUE",
    "NemoOpenSandboxEnvironment",
]

# Logger for messages that have no environment instance to log through, such as ``preflight``.
LOGGER = logging.getLogger(__name__)

# Harbor environment type this class reports; the value Harbor 0.20's own OpenSandbox environment uses.
ENVIRONMENT_TYPE = "opensandbox"
# Sandbox metadata label marking which NeMo component manages the sandbox.
MANAGED_BY_METADATA_KEY = "nemo-managed-by"
# Value of MANAGED_BY_METADATA_KEY on every sandbox this environment creates.
MANAGED_BY_METADATA_VALUE = "scaled-evals"
# Constructor defaults, the same values Harbor 0.20's own OpenSandbox environment used. They apply
# only when Harbor's environment kwargs leave a setting out; see the class docstring.
# How long the server keeps a sandbox before deleting it. 24 hours (86,400 seconds).
DEFAULT_SANDBOX_TIMEOUT_SEC = 60 * 60 * 24
# How long a new sandbox may take to become healthy. 24 hours (86,400 seconds); in practice Harbor's
# own start timeout ends the wait sooner: the task's build_timeout_sec (600 seconds by default) times
# the job's environment_build_timeout_multiplier.
DEFAULT_READY_TIMEOUT_SEC = 60 * 60 * 24
# How often the readiness wait polls a new sandbox. 2 seconds.
DEFAULT_HEALTH_CHECK_POLL_INTERVAL_SEC = 2.0
# How long the one-time ``id -u`` probe of the image's user may take. 30 seconds.
_IMAGE_UID_PROBE_TIMEOUT_SEC = 30
# How long start() waits for main's readiness check and the services, when neither main's
# readiness timeout_sec nor the spec's services_timeout_sec sets it.
COMPOSE_READY_TIMEOUT_SEC = 300
COMPOSE_READY_POLL_SEC = 2.0
# The server's exec route accepts at most this timeout.
SERVICE_EXEC_MAX_TIMEOUT_SEC = 3600
SERVICE_EXEC_DEFAULT_TIMEOUT_SEC = 600
# The exec route caps output at 16 MiB; an 8 MiB chunk is about 11 MiB once base64-encoded.
SERVICE_DOWNLOAD_BLOCK_BYTES = 1024 * 1024
SERVICE_DOWNLOAD_CHUNK_BLOCKS = 8
# The server's own routes retry transient failures: the extension's 429 (every exec slot busy) and
# the gateway errors a restarting or overloaded server answers with. Delays double from the base
# up to the max, each scaled by a random factor so a burst of trials doesn't retry in step.
SERVER_RETRY_STATUSES = frozenset({429, 502, 503, 504})
SERVER_RETRY_ATTEMPTS = 5
SERVER_RETRY_BASE_SEC = 0.5
SERVER_RETRY_MAX_SEC = 8.0
# The readiness wait polls the services status on every poll, so each read gets only this long.
PROVISIONING_STATUS_BUDGET_SEC = 30
# execd's port, which every sandbox pod listens on; its direct endpoint is ``<pod IP>:<port>``.
EXECD_PORT = 44772
# The pod IP is optional, so its lookup gives up after this long in all, retries included.
SANDBOX_IP_LOOKUP_BUDGET_SEC = 15
# How long each command that activates the agent bundle may take: reading PATH, then ``--version``.
AGENT_BUNDLE_EXEC_TIMEOUT_SEC = 60
# ``stop`` asks the server whether the sandbox stopped early; the check gives up after this long in all.
TERMINATION_CHECK_BUDGET_SEC = 10


def sandbox_role(session_id: str, trial_name: str) -> str:
    """Return ``agent`` or ``verifier`` for a sandbox, from the session ID Harbor gave it.

    Harbor doesn't pass the role to the environment, and names the agent's session differently by
    release: exactly the trial name on 0.13.2, exactly ``<trial>__env`` on 0.20. A separate
    verifier's session is ``<trial>__verifier__<key>`` on both, shortened to end in a hash when too
    long. The trial name can itself contain ``__verifier__``, so only an exact match identifies the
    agent.
    """
    return "agent" if session_id in (trial_name, f"{trial_name}__env") else "verifier"


def _local_write_mode(path: Path) -> int:
    """A file's permission bits as the SDK expects them: octal digits read as decimal (0o755 -> 755).

    Carrying them keeps executable scripts executable after upload.
    """
    return int(format(stat.S_IMODE(path.stat().st_mode), "o"))


#: The runtime the OpenSandbox server starts sandboxes with, which decides how a container's command and env are read.
SandboxRuntime = Literal["kubernetes", "docker"]


def _literal(text: str) -> str:
    """``text`` as Kubernetes passes it on unchanged: a container's command and env expand ``$(VAR)``
    and turn ``$$`` into ``$``, which Docker never does, so every ``$`` is doubled."""
    return text.replace("$", "$$")


class NemoOpenSandboxEnvironment(BaseEnvironment):
    """One OpenSandbox sandbox per Harbor environment, with trusted egress, applied-policy verification, cleanup, and compose services.

    Args:
        domain: OpenSandbox server address; defaults to ``OPENSANDBOX_DOMAIN``.
        api_key: OpenSandbox API key; defaults to ``OPENSANDBOX_API_KEY``. An explicit ``""`` means no auth.
        protocol: ``http`` or ``https``.
        use_server_proxy: Route execd traffic through the server instead of to the pod directly.
        ready_timeout_sec: How long a new sandbox may take to become healthy.
        sandbox_timeout_sec: The sandbox's server-side lifetime.
        sandbox_expires_at: ISO-8601 time, with its UTC offset, no sandbox may outlive. Each create gets ``sandbox_timeout_sec``
            cut to the time left then (at least 1 s), so a sandbox created late (a retry, a separate verifier)
            ends at the same time as the first. ``None`` keeps ``sandbox_timeout_sec`` as is.
        request_timeout_sec: Per-request SDK timeout; ``None`` keeps the SDK default.
        health_check_poll_interval_sec: How often the readiness wait polls.
        metadata: Labels for every sandbox, such as the ownership labels cleanup selects on.
        trusted_allowed_hosts: Destinations trials may reach, as hostnames, ``*.`` wildcard
            hostnames, IP addresses, or CIDRs. Supplied by the runner, not the task.
        resolver_addresses: Nameservers to keep reachable. ``None`` uses the runner's own
            ``/etc/resolv.conf``, which in-cluster is the resolver sandboxes share.
        egress_verification: How strictly to compare the applied policy with the requested one;
            see ``verify_applied_egress``. Readback itself is always required.
        enforce_disk_limit: Cap each sandbox's ephemeral storage at the task's ``storage_mb``. Turn
            this off only on a runtime that doesn't enforce the limit.
        max_disk_mb: Ceiling on the limit, so a task asking for far more than any node holds can't
            make its pod unschedulable. ``None`` applies the task's value as given.
        compose_services: The evaluation profile's ``compose_services``. Applied only when the
            environment directory has a Compose file, so a separate verifier environment (built
            from the task's ``tests/``) gets a plain sandbox. A variable's value may be a
            ``${NEMO_COMPOSE_ENV_<n>}`` template, which this process's environment fills in (see
            ``nhx_sandbox.compose_env``); Harbor then persists the template, not the value.
        agent_bundle: The evaluation's agent bundle, as a ``SandboxAgentBundle``. Only the agent's
            sandbox gets it, never a separate verifier's. The server extension installs it into
            ``/installed-agent`` before the sandbox starts; ``start`` then puts its executable on
            ``PATH``, so Harbor finds the agent already installed and skips its own install,
            which needs root.
        image_entrypoints: Each image's own ``ENTRYPOINT``, by the pinned reference ``task.toml``'s
            ``docker_image`` names, with an empty list for an image that has none. OpenSandbox
            replaces the container's entrypoint, so the sandbox restores its image's, unless the
            compose file gives ``main`` its own. An image missing here starts as before.
        runtime: The OpenSandbox server's runtime. Kubernetes expands ``$`` in a container's
            command and env, so they're escaped for it; Docker passes them on as written. NeMo Evals
            only runs on the in-cluster Kubernetes server; set ``docker`` for a Docker-runtime server.

    Any other keyword argument is Harbor's and goes to ``BaseEnvironment`` unchanged.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        domain: str | None = None,
        api_key: str | None = None,
        protocol: str = "https",
        use_server_proxy: bool = False,
        ready_timeout_sec: float = DEFAULT_READY_TIMEOUT_SEC,
        sandbox_timeout_sec: float = DEFAULT_SANDBOX_TIMEOUT_SEC,
        sandbox_expires_at: str | None = None,
        request_timeout_sec: float | None = None,
        health_check_poll_interval_sec: float = DEFAULT_HEALTH_CHECK_POLL_INTERVAL_SEC,
        metadata: dict[str, str] | None = None,
        trusted_allowed_hosts: list[str] | None = None,
        resolver_addresses: list[str] | None = None,
        egress_verification: EgressVerificationMode = "default_action",
        enforce_disk_limit: bool = True,
        max_disk_mb: int | None = None,
        compose_services: dict[str, Any] | None = None,
        agent_bundle: dict[str, Any] | None = None,
        image_entrypoints: dict[str, list[str]] | None = None,
        runtime: SandboxRuntime = "kubernetes",
        **kwargs: Any,
    ) -> None:
        """Store this environment's settings, then let ``BaseEnvironment`` validate the task against them."""
        # BaseEnvironment.__init__ runs the definition, capability and network policy validators,
        # which read these, so they are set first.
        has_compose_file = any((environment_dir / name).is_file() for name in COMPOSE_FILENAMES)
        # Expanded after validation, so an invalid spec's error shows the templates and never a value.
        self._services: ComposeServices | None = (
            expand_env(parse_compose_services(compose_services), os.environ)
            if compose_services is not None and has_compose_file
            else None
        )

        # The bundle is the agent's alone; a separate verifier's sandbox runs the task image as is.
        is_agent = sandbox_role(session_id, trial_paths.trial_dir.name) == "agent"
        self._agent_bundle: SandboxAgentBundle | None = (
            SandboxAgentBundle.model_validate(agent_bundle) if agent_bundle is not None and is_agent else None
        )

        self._connection = Connection(
            domain=domain if domain is not None else os.environ.get("OPENSANDBOX_DOMAIN"),
            api_key=api_key if api_key is not None else os.environ.get("OPENSANDBOX_API_KEY"),
            protocol=protocol,
            use_server_proxy=use_server_proxy,
            request_timeout_sec=request_timeout_sec,
        )
        self._ready_timeout_sec = ready_timeout_sec
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._sandbox_expires_at = datetime.fromisoformat(sandbox_expires_at) if sandbox_expires_at else None
        self._health_check_poll_interval_sec = health_check_poll_interval_sec
        self._metadata = dict(metadata or {})
        self._trusted_allowed_hosts = tuple(trusted_allowed_hosts or ())
        self._trusted_targets = frozenset(canonical_egress_target(host) for host in self._trusted_allowed_hosts)
        self._resolver_addresses = tuple(resolver_addresses) if resolver_addresses is not None else None
        self._image_entrypoints = dict(image_entrypoints or {})
        self._runtime: SandboxRuntime = runtime
        self._egress_verification: EgressVerificationMode = egress_verification
        self._enforce_disk_limit = enforce_disk_limit
        self._max_disk_mb = max_disk_mb
        self._disk_limit_mb: int | None = None
        self._sandbox: Any | None = None
        # When the server will delete the sandbox, at the latest; see _create_sandbox.
        self._sandbox_ends_at: datetime | None = None
        self._record: dict[str, Any] | None = None
        self._record_path: Path | None = None
        self._image_uid: int | None = None
        self._image_uid_probed = False
        self._image_uid_lock = asyncio.Lock()
        self._warned_root_unavailable = False
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            *args,
            **kwargs,
        )
        self._driver = OpenSandboxDriver(self._connection, logger=self.logger)
        # Compose's environment for main is the base; task and trial env win over it. Set after
        # BaseEnvironment fills in the task's env.
        if self._services is not None:
            self._persistent_env = {**self._services.main.env, **self._persistent_env}

    @staticmethod
    @override
    def type() -> str:
        """Harbor accepts any string from a custom environment; this matches its own OpenSandbox type."""
        return ENVIRONMENT_TYPE

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        """Advertise every allowlist form ``egress_policy`` can enforce, so Harbor accepts those tasks.

        Harbor 0.13 has no per-entry-type allowlist fields (its task model admits hostnames and
        ``*.`` wildcards only), so fields the installed Harbor doesn't define are left out.
        """
        wanted = {
            "gpus": True,
            "disable_internet": True,
            "mounted": False,
            "docker_compose": self._services is not None,
            "network_allowlist": True,
            "network_allowlist_hostnames": True,
            "network_allowlist_wildcard_hostnames": True,
            "network_allowlist_ipv4_addresses": True,
            "network_allowlist_ipv6_addresses": True,
            "network_allowlist_ipv4_cidrs": True,
            "network_allowlist_ipv6_cidrs": True,
        }
        fields = EnvironmentCapabilities.model_fields
        return EnvironmentCapabilities(**{name: value for name, value in wanted.items() if name in fields})

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        """CPU and memory are sent as limits; OpenSandbox has no separate request for them here."""
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @classmethod
    @override
    def preflight(cls) -> None:
        """Warn early when the server address is unset; it can still come from the ``domain`` kwarg."""
        if not os.environ.get("OPENSANDBOX_DOMAIN"):
            LOGGER.warning(
                "OPENSANDBOX_DOMAIN is not set. Provide the OpenSandbox server address through the "
                "environment's `domain` kwarg or OPENSANDBOX_DOMAIN, or trials will fail at start()."
            )

    @override
    def _validate_definition(self) -> None:
        """Require a prebuilt image; a Compose file only with its services configured, since OpenSandbox runs no Compose."""
        has_compose_file = any((self.environment_dir / name).exists() for name in COMPOSE_FILENAMES)
        if has_compose_file and self._services is None:
            raise ValueError(
                "NemoOpenSandboxEnvironment runs docker-compose tasks only when the evaluation profile "
                "lists their services under environment.kwargs.compose_services."
            )
        if self.task_env_config.docker_image:
            return
        if (self.environment_dir / "Dockerfile").exists():
            raise ValueError("NemoOpenSandboxEnvironment requires a prebuilt image via task.environment.docker_image.")
        raise FileNotFoundError(
            "NemoOpenSandboxEnvironment requires a prebuilt image. Set task.environment.docker_image."
        )

    def _task_disk_limit_mb(self) -> int | None:
        """The task's ``storage_mb`` capped at ``max_disk_mb``, or ``None`` when no limit applies."""
        requested = self._effective_storage_mb
        if not self._enforce_disk_limit or requested is None:
            return None
        return min(requested, self._max_disk_mb) if self._max_disk_mb is not None else requested

    @override
    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        """Run Harbor's checks, then reject ``allowlist`` tasks that ask for hosts outside the trusted allowlist."""
        super().validate_network_policy_support(network_policy)
        network_policy = network_policy or self._network_policy
        if network_policy.network_mode != NetworkMode.ALLOWLIST:
            return
        outside = sorted(
            host for host in network_policy.allowed_hosts if canonical_egress_target(host) not in self._trusted_targets
        )
        if outside:
            raise ValueError(
                f"Task requests egress to {outside}, which the trusted allowlist for this run does not "
                "include. Tasks may narrow the trusted allowlist but not extend it."
            )

    def egress_policy(self) -> EgressPolicy:
        """Return the egress policy for the current network mode."""
        mode = self._network_policy.network_mode
        if mode == NetworkMode.NO_NETWORK:
            return EgressPolicy()
        targets = (
            tuple(self._network_policy.allowed_hosts) if mode == NetworkMode.ALLOWLIST else self._trusted_allowed_hosts
        )
        return build_egress_policy(EgressAllowlist(targets=targets, resolver_addresses=self._resolver_addresses))

    def _resource(self) -> dict[str, str]:
        """CPU, memory, GPU and disk limits from the task.

        OpenSandbox has no disk limit of its own; its Kubernetes runtime passes ``ephemeral-storage``
        through to the pod. Built on every create attempt, so a retry sends the same limit.
        """
        # _resource_limit_value returns None when the task asks for a request rather than a limit,
        # so a reservation is never silently turned into a hard cap.
        cpu = self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT)
        memory = self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT)
        resource: dict[str, str] = {}
        if cpu is not None:
            resource["cpu"] = str(cpu)
        if memory is not None:
            resource["memory"] = f"{memory}Mi"
        if self._effective_gpus > 0:
            resource["gpu"] = str(self._effective_gpus)
        self._disk_limit_mb = self._task_disk_limit_mb()
        if self._disk_limit_mb is not None:
            resource["ephemeral-storage"] = f"{self._disk_limit_mb}Mi"
        return resource

    def _extensions(self) -> dict[str, str]:
        """Server extensions: the task's GPU types, under the key Harbor 0.20 used, the agent bundle, and the compose services."""
        extensions: dict[str, str] = {}
        if self.task_env_config.gpu_types:
            extensions["harbor.gpu_types"] = ",".join(self.task_env_config.gpu_types)
        # The NeMo extension reads these two and adds the bundle's init container and the services to the pod.
        if self._agent_bundle is not None:
            extensions[AGENT_BUNDLE_EXTENSION_KEY] = self._agent_bundle.to_json()
        if self._services is not None:
            extensions[EXTENSION_KEY] = self._services.to_json()
        return extensions

    def _as_written(self, text: str) -> str:
        """``text`` escaped so the runtime starts the container with it exactly as written."""
        return _literal(text) if self._runtime == "kubernetes" else text

    async def _create_sandbox(self) -> Any:
        """Create the sandbox, then verify its egress; it is killed if the policy can't be read or doesn't match."""
        uses_extension = self._services is not None or self._agent_bundle is not None
        if uses_extension:
            await self._require_extension()

        egress_policy = self.egress_policy()
        entrypoint = self._sandbox_entrypoint()
        # The server starts the lifetime when the create reaches it, so it ends no earlier than this.
        # stop tells a sandbox whose lifetime ran out from one deleted early by comparing with it.
        requested_at = datetime.now(UTC)
        lifetime_ends_at = requested_at + timedelta(seconds=self._sandbox_timeout_sec)
        self._sandbox_ends_at = (
            min(lifetime_ends_at, self._sandbox_expires_at) if self._sandbox_expires_at else lifetime_ends_at
        )
        sandbox = await self._driver.create(
            CreateRequest(
                # _validate_definition guarantees the task names a prebuilt image.
                image=str(self.task_env_config.docker_image),
                # Applied at create so the entrypoint sees it too, not only exec'd commands. Escaped
                # here only: commands run in the sandbox later get the env as is, unexpanded.
                env={name: self._as_written(value) for name, value in self._persistent_env.items()},
                metadata={
                    **self._metadata,
                    "session_id": self.session_id,
                    MANAGED_BY_METADATA_KEY: MANAGED_BY_METADATA_VALUE,
                },
                resource=self._resource(),
                network_policy=to_opensandbox_policy(egress_policy),
                extensions=self._extensions(),
                sandbox_timeout_sec=self._sandbox_timeout_sec,
                expires_at=self._sandbox_expires_at,
                ready_timeout_sec=self._ready_timeout_sec,
                health_check_poll_interval_sec=self._health_check_poll_interval_sec,
                entrypoint=[self._as_written(argument) for argument in entrypoint] if entrypoint is not None else None,
            ),
            # Kubernetes starts main only after the bundle and services, so without this a bundle that
            # can't install, or a crashing service, would leave the create waiting for main until its
            # ready timeout.
            failure_check=self._provisioning_failure if uses_extension else None,
        )
        started_at = datetime.now(UTC)
        recorded_id = sandbox_id(sandbox)
        try:
            await verify_applied_egress(
                sandbox,
                egress_policy,
                mode=self._egress_verification,
                label=f"OpenSandbox sandbox {sandbox_id(sandbox)} (session {self.session_id})",
                require_readback=True,
            )
            ip = await self._sandbox_ip(recorded_id) if recorded_id else None
        except BaseException as exc:
            await self._driver.safe_kill(sandbox)
            if isinstance(exc, EgressVerificationError):
                raise RuntimeError(str(exc)) from exc
            raise
        self._record_applied_egress(recorded_id, egress_policy, ip=ip, started_at=started_at)
        return sandbox

    def _sandbox_entrypoint(self) -> list[str] | None:
        """The sandbox container's entrypoint: the compose file's for ``main``, else its image's own; None for neither."""
        image_entrypoint = self._image_entrypoints.get(str(self.task_env_config.docker_image)) or None
        if self._services is not None:
            return self._services.sandbox_entrypoint(image_entrypoint)
        return keep_alive_entrypoint(image_entrypoint)

    async def _sandbox_ip(self, recorded_id: str) -> str | None:
        """The pod IP from execd's direct endpoint, or ``None`` if the server doesn't report one."""
        try:
            async with asyncio.timeout(SANDBOX_IP_LOOKUP_BUDGET_SEC):
                response = await self._server().get(
                    f"/sandboxes/{recorded_id}/endpoints/{EXECD_PORT}",
                    timeout=SANDBOX_IP_LOOKUP_BUDGET_SEC,
                    deadline=time.monotonic() + SANDBOX_IP_LOOKUP_BUDGET_SEC,
                )
            response.raise_for_status()
            endpoint = str(response.json()["endpoint"])
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            self.logger.warning(
                "Could not read sandbox %s's pod IP; it is recorded without one", recorded_id, exc_info=True
            )
            return None
        host, _, port = endpoint.rpartition(":")
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            ip = None
        if ip is None or port != str(EXECD_PORT):
            self.logger.warning(
                "Sandbox %s's execd endpoint %r is not <pod IP>:%d; it is recorded without an IP",
                recorded_id,
                endpoint,
                EXECD_PORT,
            )
            return None
        return str(ip)

    def _record_applied_egress(
        self, recorded_id: str | None, policy: EgressPolicy, *, ip: str | None, started_at: datetime
    ) -> None:
        """Write the verified policy, its hash, and the pod IP and start time into the trial directory.

        The agent and a separate verifier share the trial directory, so each sandbox writes its own file.
        ``stop`` fills in ``stopped_at``.
        """
        payload = to_opensandbox_policy(policy)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        record = {
            "sandbox_id": recorded_id,
            "session_id": self.session_id,
            "role": sandbox_role(self.session_id, self.trial_paths.trial_dir.name),
            "network_mode": self._network_policy.network_mode.value,
            "policy_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "policy": payload,
            "disk_requested_mb": self._effective_storage_mb,
            "disk_limit_mb": self._disk_limit_mb,
            "ip": ip,
            "started_at": started_at.isoformat(),
            "stopped_at": None,
        }
        self._record = record
        self._record_path = self.trial_paths.trial_dir / applied_egress_filename(recorded_id or self.session_id)
        try:
            self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
            self._write_record(self._record_path, record)
        except OSError:
            self.logger.warning("Could not record the applied egress policy", exc_info=True)

    def _record_stopped(self) -> None:
        """Fill in the record's ``stopped_at``."""
        if self._record is None or self._record_path is None:
            return
        self._record["stopped_at"] = datetime.now(UTC).isoformat()
        try:
            self._write_record(self._record_path, self._record)
        except OSError:
            self.logger.warning("Could not record when the sandbox stopped", exc_info=True)
        self._record = None
        self._record_path = None

    @staticmethod
    def _write_record(path: Path, record: dict[str, Any]) -> None:
        """Replace *path* with *record* in one rename, so a failed write leaves the previous record whole."""
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    @override
    async def start(self, force_build: bool) -> None:
        """Create the sandbox, prepare the log directories, and upload the task's environment directory.

        With an agent bundle, also put it on ``PATH``. With compose services, also wait for
        ``main``'s readiness check and every service.
        """
        if force_build:
            self.logger.warning("NemoOpenSandboxEnvironment ignores force_build; it runs prebuilt images only.")
        self._sandbox = await self._create_sandbox()

        log_dirs = [
            str(EnvironmentPaths.agent_dir),
            str(EnvironmentPaths.verifier_dir),
            str(EnvironmentPaths.artifacts_dir),
        ]
        # Only create a working directory the task asked for; otherwise the image's WORKDIR is used.
        workdir = self.task_env_config.workdir
        await self._driver.create_directories(
            self._sandbox, [*log_dirs, *([workdir] if workdir and workdir != "/" else [])]
        )

        # The directories were created as root, so non-root agents and verifiers need them opened up.
        chmod = await self.exec("chmod 777 " + " ".join(shlex.quote(path) for path in log_dirs), cwd="/", user="root")
        if chmod.return_code != 0:
            self.logger.warning(
                "Failed to chmod 777 log directories (exit %s): %s", chmod.return_code, chmod.stderr or chmod.stdout
            )

        # Supplementary files (scripts, data, configs) for prebuilt-image tasks go into the workdir.
        await self._upload_environment_dir_after_start()

        # Harbor installs the agent after start returns, so the bundle must be on PATH by then.
        if self._agent_bundle is not None:
            await self._activate_agent_bundle(self._agent_bundle)

        if self._services is None:
            return

        # One deadline covers both waits, like a Compose healthcheck's overall start period.
        readiness = self._services.main.readiness
        if readiness is not None:
            timeout_sec = readiness.timeout_sec
        else:
            timeout_sec = self._services.services_timeout_sec or COMPOSE_READY_TIMEOUT_SEC

        # Wait for main to be ready, then for all services to be ready.
        deadline = time.monotonic() + timeout_sec
        if readiness is not None:
            await self._wait_for_main(readiness.exec_, deadline)
        # after_main services start with main, so they may still be coming up.
        await self._wait_for_services(deadline)

    async def _activate_agent_bundle(self, bundle: SandboxAgentBundle) -> None:
        """Put the bundle's executable first on ``PATH`` for every command, then check it runs.

        Some of Harbor's installed agents (ex. claude-code) skip their install when they find the agent
        on ``PATH``. The image's own ``PATH`` is read as the default user and kept after the bundle's directory;
        ``PATH`` set at create would replace it. The runtime always gives a container a ``PATH``,
        so failing to read one means the sandbox can't run commands, and the trial fails here.
        """
        # Read the image's PATH as the default user, the same user Harbor's agents run as.
        path_probe = await self.exec('printf %s "$PATH"', cwd="/", timeout_sec=AGENT_BUNDLE_EXEC_TIMEOUT_SEC)
        entries = [entry for entry in (path_probe.stdout or "").split(":") if entry]
        if path_probe.return_code != 0 or not entries:
            raise RuntimeError(
                f"Agent bundle setup failed: could not read the sandbox's PATH (exit {path_probe.return_code}): "
                f"{(path_probe.stderr or '').strip()[-2000:]}"
            )

        # The bundle's directory goes first, so it wins over any copy of the agent the image already has.
        # Every later exec, including Harbor's install check and the agent's run, merges in _persistent_env.
        bundle_dir = bundle.executable.rsplit("/", 1)[0]
        self._persistent_env["PATH"] = ":".join([bundle_dir, *(entry for entry in entries if entry != bundle_dir)])

        # Run the agent once, so a bundle built for another platform fails here with a clear cause.
        version_check = await self.exec(
            f"{shlex.quote(bundle.executable)} --version", cwd="/", timeout_sec=AGENT_BUNDLE_EXEC_TIMEOUT_SEC
        )
        output = ((version_check.stdout or "") + (version_check.stderr or "")).strip()
        if version_check.return_code != 0:
            raise RuntimeError(
                f"Agent bundle check failed: {bundle.executable} --version exited {version_check.return_code}: "
                f"{output[-2000:]}. The bundle may not match this image (it needs glibc on linux/amd64)."
            )
        self.logger.info("Agent bundle ready: %s --version: %s", bundle.executable, output[:200])

    def _server(self) -> _ServerClient:
        """A client for the extension's routes, on the same server and API key as the driver's SDK calls."""
        return _ServerClient.from_connection_config(self._driver.connection_config())

    async def _require_extension(self) -> None:
        """Fail before creating anything if the server can't run this sandbox's services or agent bundle.

        A stock server ignores unknown extensions and would create a sandbox without them. So would
        an extension from before agent bundles, which reports no ``features``.
        """
        try:
            response = await self._server().get("/nemo-ext/health", timeout=30)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach the OpenSandbox compose services extension: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                "The OpenSandbox server does not run the NeMo compose services extension; "
                f"GET /v1/nemo-ext/health returned {response.status_code}. See "
                "k8s/helm/examples/opensandbox/README.md, 'Compose services'."
            )

        features = response.json().get("features") or []
        # An extension from before agent bundles answers the health check too, but would drop the bundle.
        if self._agent_bundle is not None and AGENT_BUNDLE_FEATURE not in features:
            raise RuntimeError(
                "The OpenSandbox server's NeMo compose services extension doesn't support agent bundles; upgrade it to "
                "the version in plugins/_temporary-scaled-evals/opensandbox_ext."
            )
        # An extension from before them would refuse a spec with no services, or main's command or working directory.
        if self._services is not None and self._services.uses_main_settings() and COMPOSE_MAIN_FEATURE not in features:
            raise RuntimeError(
                "The OpenSandbox server's NeMo compose services extension doesn't support compose files that only "
                "change main, or main's command and working directory; upgrade it to the version in "
                "plugins/_temporary-scaled-evals/opensandbox_ext."
            )

    async def _services_status(self, sandbox_id: str, deadline: float) -> dict[str, Any]:
        """The extension's view of the sandbox's services: ``{"failure": str | None, "not_ready": [names]}``.

        The read, with its retries, stops at ``deadline`` (a ``time.monotonic()`` value).
        """
        response = await self._server().get(f"/nemo-ext/sandboxes/{sandbox_id}/services", timeout=30, deadline=deadline)
        if response.status_code != 200:
            raise RuntimeError(f"reading compose services failed ({response.status_code}): {_error_text(response)}")
        return response.json()

    async def _provisioning_failure(self, sandbox: Any) -> str | None:
        """The driver's ``failure_check``: an agent bundle or compose service that can't start fails the create now, with its logs."""
        try:
            status = await self._services_status(
                str(sandbox_id(sandbox)), deadline=time.monotonic() + PROVISIONING_STATUS_BUDGET_SEC
            )
        except Exception:
            # The pod may not exist yet, or the read hit a transient error; the next poll asks again.
            return None
        if status.get("failure"):
            return f"OpenSandbox sandbox {sandbox_id(sandbox)}: {status['failure']}"
        return None

    async def _wait_for_main(self, argv: list[str], deadline: float) -> None:
        """Run ``argv`` in the sandbox until it exits 0, like a Compose healthcheck of ``main``.

        Kubernetes only knows when the sandbox container started, not when the processes its
        entrypoint launched are up.
        """
        command = shlex.join(argv)
        sandbox = self._require_sandbox()
        while True:
            # Docker runs a healthcheck as the image's user from its WORKDIR, so no user or directory is sent:
            # not root, and not the task's default user or workdir, which `exec` would apply.
            remaining = max(1, int(deadline - time.monotonic()))
            last = await self._driver.run(sandbox, command, env=self._merge_env(None), timeout_sec=min(30, remaining))
            if last.exit_code == 0:
                return
            # Out of time: report the check's last output, which usually says what's missing.
            if time.monotonic() >= deadline:
                output = ((last.stdout or "") + (last.stderr or ""))[-2000:]
                raise RuntimeError(f"main was not ready: {command!r} last exited {last.exit_code}: {output}")
            await asyncio.sleep(COMPOSE_READY_POLL_SEC)

    async def _wait_for_services(self, deadline: float) -> None:
        """Poll the services route until every service is ready; raise on a failure or at ``deadline``.

        A failure is raised as soon as the route reports one, with the service's log lines.
        """
        current = str(sandbox_id(self._require_sandbox()))
        while True:
            status = await self._services_status(current, deadline)
            if status["failure"]:
                raise RuntimeError(f"OpenSandbox sandbox {current}: {status['failure']}")
            if not status["not_ready"]:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"compose services not ready: {', '.join(status['not_ready'])}")
            await asyncio.sleep(COMPOSE_READY_POLL_SEC)

    def _service_transport(self, service: str) -> _ServiceExecTransport:
        """The transport for ``service``'s operations; only the profile's services qualify.

        ``main`` is the sandbox itself, which callers reach through the regular methods.
        """
        if self._services is None:
            raise ServiceOperationsUnsupportedError(
                f"{self.type()} environment is not running a docker-compose task, so it cannot target "
                f"compose service {service!r}."
            )
        if service not in self._services.names():
            raise ServiceOperationsUnsupportedError(
                f"compose service {service!r} is not one of this sandbox's services {self._services.names()}"
                + (" (main is the sandbox itself)" if service == MAIN_SERVICE else "")
            )
        return _ServiceExecTransport(self._server(), str(sandbox_id(self._require_sandbox())))

    # Harbor 0.20's per-service operations. Harbor 0.13.2 never calls them. ``service=None`` or
    # ``main`` is the sandbox itself; any other service goes through the extension's exec route.

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run ``command`` in ``service``; a service's command doesn't get main's workdir, user or env."""
        if service is None or service == MAIN_SERVICE:
            return await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        return await self._service_transport(service).service_exec(
            command, service=service, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )

    async def service_download_file(
        self, source_path: str, target_path: Path | str, *, service: str | None = None
    ) -> None:
        """Copy ``source_path`` in ``service`` to ``target_path`` here."""
        if service is None or service == MAIN_SERVICE:
            await self.download_file(source_path, target_path)
            return
        await self._service_transport(service).service_download_file(source_path, target_path, service=service)

    async def service_download_dir(
        self, source_dir: str, target_dir: Path | str, *, service: str | None = None
    ) -> None:
        """Copy the directory ``source_dir`` in ``service`` to ``target_dir`` here."""
        if service is None or service == MAIN_SERVICE:
            await self.download_dir(source_dir, target_dir)
            return
        await self._service_transport(service).service_download_dir(source_dir, target_dir, service=service)

    async def stop_service(self, service: str) -> None:
        """Always unsupported: every service, ``main`` included, is a container in one running pod."""
        self._service_transport(service)  # Raises first for main and services this sandbox doesn't have.
        raise ServiceOperationsUnsupportedError(
            f"compose service {service!r} runs as a container in the sandbox pod and can't be stopped on its own"
        )

    @override
    async def stop(self, delete: bool) -> None:
        """Record why the sandbox stopped if it already had, kill it when ``delete`` is set, and always release the client.

        Harbor calls ``stop`` for every sandbox it started, whether the trial passed, failed or was
        cancelled, and before it writes the trial's ``result.json``. So this one check sees every
        sandbox that died under its trial, whichever operation hit it first.
        """
        if not self._sandbox:
            return
        await self._record_termination(self._sandbox)
        self._record_stopped()
        try:
            if delete:
                await self._sandbox.kill()
        finally:
            try:
                await self._sandbox.close()
            finally:
                self._sandbox = None

    async def _record_termination(self, sandbox: Any) -> None:
        """Write why ``sandbox`` stopped into the trial directory, if it had; never raises, so ``stop`` always goes on."""
        termination = await check_termination(
            OpenSandboxTerminationDiagnostics(self._server(), timeout_sec=TERMINATION_CHECK_BUDGET_SEC),
            str(sandbox_id(sandbox)),
            ends_at=self._sandbox_ends_at,
            role=sandbox_role(self.session_id, self.trial_paths.trial_dir.name),
            budget_sec=TERMINATION_CHECK_BUDGET_SEC,
            logger=self.logger,
        )
        if termination is None:
            return
        self.logger.warning(
            "Sandbox %s stopped before the trial finished with it: %s", termination.sandbox_id, termination.detail
        )
        try:
            write_termination(self.trial_paths.trial_dir, termination)
        except OSError:
            self.logger.warning("Could not record why sandbox %s stopped", termination.sandbox_id, exc_info=True)

    def _require_sandbox(self) -> Any:
        """The live sandbox; raises if ``start`` hasn't created one or ``stop`` already released it."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload one file, keeping its permission bits, and create its parent directory."""
        sandbox = self._require_sandbox()
        source_path = Path(source_path)
        parent = str(PurePosixPath(target_path).parent)
        if parent not in (".", "/"):
            await self._driver.create_directories(sandbox, [parent])
        await self._driver.write_files(
            sandbox, [(target_path, source_path.read_bytes(), _local_write_mode(source_path))]
        )

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory tree in one write, keeping each file's permission bits."""
        sandbox = self._require_sandbox()
        source_dir = Path(source_dir)
        files: list[tuple[str, bytes, int]] = []
        directories: set[str] = {target_dir}
        for file_path in source_dir.rglob("*"):
            remote_path = str(PurePosixPath(target_dir) / file_path.relative_to(source_dir).as_posix())
            if file_path.is_dir():
                directories.add(remote_path)
                continue
            directories.add(str(PurePosixPath(remote_path).parent))
            files.append((remote_path, file_path.read_bytes(), _local_write_mode(file_path)))
        await self._driver.create_directories(sandbox, sorted(directories))
        if files:
            await self._driver.write_files(sandbox, files)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download one file, creating its local parent directory."""
        sandbox = self._require_sandbox()
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(await self._driver.read_bytes(sandbox, source_path))

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory tree; unreadable and special entries are skipped with a warning."""
        sandbox = self._require_sandbox()
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for relative, remote_path, entry_type in await self._driver.list_tree(sandbox, source_dir):
            local_path = target_dir if relative == PurePosixPath(".") else target_dir / relative

            # The search result usually says what each entry is; ask the sandbox only when it doesn't.
            is_directory = entry_type.endswith("directory")
            is_regular = entry_type.endswith("file")
            if not (is_directory or is_regular):
                is_directory = await self.is_dir(remote_path)
                is_regular = not is_directory and await self.is_file(remote_path)

            if is_directory:
                local_path.mkdir(parents=True, exist_ok=True)
            elif is_regular:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    data = await self._driver.read_bytes(sandbox, remote_path)
                except Exception as exc:
                    # For example a dangling symlink; one bad entry must not abort artifact collection.
                    self.logger.warning("Skipping unreadable entry %s during download_dir: %s", remote_path, exc)
                    continue
                local_path.write_bytes(data)
            else:
                self.logger.warning(
                    "Skipping non-regular entry %s (type=%r) during download_dir.", remote_path, entry_type or "unknown"
                )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run a command; explicit ``cwd``, then the task's workdir, then ``main``'s compose ``working_dir``, then the image's WORKDIR."""
        sandbox = self._require_sandbox()
        requested_uid = self._numeric_uid(self._resolve_user(user))
        # Learn the image's uid before the first root command, so it can be mapped below.
        if not self._image_uid_probed and requested_uid == 0:
            await self._probe_image_uid()
        result = await self._driver.run(
            sandbox,
            command,
            cwd=cwd or self.task_env_config.workdir or self._compose_working_dir() or None,
            env=self._merge_env(env),
            timeout_sec=timeout_sec,
            uid=self._uid_for(requested_uid),
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    def _compose_working_dir(self) -> str | None:
        """``main``'s ``working_dir`` from the compose file, or None."""
        return self._services.main.working_dir if self._services is not None else None

    def _numeric_uid(self, user: str | int | None) -> int | None:
        """The uid execd should switch to: integers, digit strings and ``root``; ``None`` for anything else.

        Other names would need the image's passwd database, so they run as the default user.
        """
        if user is None:
            return None
        if isinstance(user, int):
            return user
        if user.isdigit():
            return int(user)
        if user == "root":
            return 0
        self.logger.debug("OpenSandbox runs commands as a numeric uid; running %r as the sandbox default user.", user)
        return None

    def _uid_for(self, uid: int | None) -> int | None:
        """The uid to send for a requested one: unchanged, except that root on a non-root image runs as the image's user."""
        if uid != 0 or self._image_uid in (None, 0):
            return uid
        if not self._warned_root_unavailable:
            self._warned_root_unavailable = True
            self.logger.warning(
                "Sandbox image runs as uid %s and OpenSandbox can't switch it to root; running root "
                "commands as uid %s instead. Commands that need root, such as package installs, will fail.",
                self._image_uid,
                self._image_uid,
            )
        return None

    async def _probe_image_uid(self) -> None:
        """Run ``id -u`` once as the sandbox default user; on failure, root requests stay uid 0."""
        async with self._image_uid_lock:
            if self._image_uid_probed or self._sandbox is None:
                return
            self._image_uid_probed = True
            try:
                result = await self._driver.run(
                    self._sandbox, "id -u", cwd="/", timeout_sec=_IMAGE_UID_PROBE_TIMEOUT_SEC
                )
                self._image_uid = int(result.stdout.strip())
            except Exception:
                self.logger.warning(
                    "Could not read the sandbox image's uid; root commands are sent as uid 0", exc_info=True
                )


def _error_text(response: httpx.Response) -> str:
    """A short error message from a server response: ``code: message`` when it has the server's error shape."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:2000]
    detail = body.get("detail", body) if isinstance(body, dict) else None
    if isinstance(detail, dict) and "message" in detail:
        return f"{detail.get('code', 'error')}: {detail['message']}"
    return response.text[:2000]


class _ServerClient:
    """Requests to the OpenSandbox server's own API, authenticated like the SDK's."""

    def __init__(self, base_url: str, headers: dict[str, str]) -> None:
        """``base_url`` ends in the API version (``.../v1``); ``headers`` carry the API key."""
        self.base_url = base_url.rstrip("/")
        self.headers = headers

    @classmethod
    def from_connection_config(cls, config: Any) -> _ServerClient:
        """Build a client from the SDK's ``ConnectionConfig``, so it reaches the same server with the same key."""
        headers = dict(getattr(config, "headers", None) or {})
        api_key = config.get_api_key()
        if api_key:
            headers["OPEN-SANDBOX-API-KEY"] = api_key
        return cls(config.get_base_url(), headers)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        retry_statuses: frozenset[int],
        retry_errors: tuple[type[httpx.HTTPError], ...],
        body: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> httpx.Response:
        """Send one request, retrying a ``retry_statuses`` answer or a ``retry_errors`` failure with backoff.

        After the last attempt the response is returned as is, or the error raised, so callers
        map failures exactly as they would without retries. With a ``deadline`` (a
        ``time.monotonic()`` value), no attempt waits past it by more than a second and none
        starts after it, and a response that does arrive is returned whenever it came.
        """
        for attempt in range(1, SERVER_RETRY_ATTEMPTS + 1):
            # Always at least one second, so a deadline that already passed still gets a real try.
            attempt_timeout = timeout if deadline is None else max(1.0, min(timeout, deadline - time.monotonic()))
            error: httpx.HTTPError | None = None
            response: httpx.Response | None = None

            try:
                async with httpx.AsyncClient(timeout=attempt_timeout) as client:
                    response = await client.request(method, f"{self.base_url}{path}", json=body, headers=self.headers)
            except retry_errors as exc:
                error = exc

            is_final_attempt = attempt == SERVER_RETRY_ATTEMPTS or (
                deadline is not None and time.monotonic() >= deadline
            )
            if response is not None and (is_final_attempt or response.status_code not in retry_statuses):
                return response
            if error is not None and is_final_attempt:
                raise error

            pause = min(SERVER_RETRY_MAX_SEC, SERVER_RETRY_BASE_SEC * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            if deadline is not None:
                pause = min(pause, deadline - time.monotonic())

            await asyncio.sleep(pause)
        raise AssertionError("unreachable")  # the last attempt returns or raises

    async def get(self, path: str, *, timeout: float, deadline: float | None = None) -> httpx.Response:
        """``GET base_url + path``; the response is returned whatever its status.

        A read changes nothing, so any transport error or transient status is retried, until
        ``deadline`` if there is one.
        """
        return await self._request(
            "GET",
            path,
            timeout=timeout,
            retry_statuses=SERVER_RETRY_STATUSES,
            retry_errors=(httpx.TransportError,),
            deadline=deadline,
        )

    async def post(self, path: str, body: dict[str, Any], *, timeout: float) -> httpx.Response:
        """``POST base_url + path`` with a JSON ``body``; the response is returned whatever its status.

        A POST may have run its command, so only what proves it didn't is retried: a 429, which
        the server answers before running anything, and a connection that never opened.
        """
        return await self._request(
            "POST",
            path,
            timeout=timeout,
            retry_statuses=frozenset({429}),
            retry_errors=(httpx.ConnectError, httpx.ConnectTimeout),
            body=body,
        )


class _ServiceExecTransport:
    """Harbor's per-service operations, run through the extension's exec route.

    Commands run as the service container's own user; ``user`` is ignored because the
    container's image decides which users exist. Downloads stream through exec output (base64) in
    chunks that fit the route's output cap, so the service image needs ``sh``, ``wc``, ``dd``,
    ``base64`` and, for directories, ``mktemp`` and ``tar``.
    """

    def __init__(self, client: _ServerClient, sandbox_id: str) -> None:
        """Operations go to the services of ``sandbox_id`` through ``client``."""
        self._client = client
        self._sandbox_id = sandbox_id

    async def _exec(self, service: str, argv: list[str], timeout_sec: int | None) -> ExecResult:
        """Run ``argv`` in ``service``; a timeout is exit code 124, like ``timeout(1)``."""
        timeout = min(timeout_sec or SERVICE_EXEC_DEFAULT_TIMEOUT_SEC, SERVICE_EXEC_MAX_TIMEOUT_SEC)
        try:
            # The HTTP timeout leaves the server time to report its own timeout first.
            response = await self._client.post(
                f"/nemo-ext/sandboxes/{self._sandbox_id}/services/{service}/exec",
                {"command": argv, "timeout": timeout},
                timeout=timeout + 30,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"exec in compose service {service!r} failed: {exc}") from exc

        # The extension's 504 is the command's own timeout, which Harbor expects as a result, not an
        # error. Any other 504 (a gateway's, say) says nothing about the command, so it raises below.
        if response.status_code == 504 and _error_text(response).startswith("NEMO::EXEC_TIMEOUT"):
            return ExecResult(stdout="", stderr=f"timed out after {timeout}s", return_code=124)

        if response.status_code != 200:
            raise RuntimeError(
                f"exec in compose service {service!r} failed ({response.status_code}): {_error_text(response)}"
            )

        body = response.json()

        return ExecResult(stdout=body["stdout"], stderr=body["stderr"], return_code=body["exit_code"])

    async def service_exec(
        self,
        command: str,
        *,
        service: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Harbor's ``exec`` for a service: run the shell ``command`` with ``cwd`` and ``env`` applied."""
        # The exec route takes an argv and has no cwd or env, so both go into the script.
        script = command
        if cwd:
            script = f"cd {shlex.quote(cwd)} && {script}"
        if env:
            script = "".join(f"export {name}={shlex.quote(value)}; " for name, value in env.items()) + script
        return await self._exec(service, ["sh", "-c", script], timeout_sec)

    async def _run(self, service: str, script: str, what: str) -> str:
        """Run a download helper ``script`` and return its stdout; any non-zero exit fails reading ``what``."""
        result = await self._exec(service, ["sh", "-c", script], None)
        if result.return_code != 0:
            raise RuntimeError(
                f"could not read {what} from compose service {service!r}: {(result.stderr or '').strip()}"
            )
        return result.stdout or ""

    async def _download(self, service: str, source_path: str, target: Path, what: str) -> None:
        """Copy one file out of ``service`` to ``target``, one base64 chunk per exec, and check the byte count."""
        # The size up front tells how many chunks to read and lets the result be checked.
        quoted = shlex.quote(source_path)
        size = int((await self._run(service, f"wc -c < {quoted}", what)).strip())
        chunk_bytes = SERVICE_DOWNLOAD_BLOCK_BYTES * SERVICE_DOWNLOAD_CHUNK_BLOCKS
        target.parent.mkdir(parents=True, exist_ok=True)

        # Each exec reads one chunk with dd; base64 keeps binary data intact in the JSON response.
        written = 0
        with target.open("wb") as out:
            for index in range(math.ceil(size / chunk_bytes)):
                script = (
                    f"dd if={quoted} bs={SERVICE_DOWNLOAD_BLOCK_BYTES} "
                    f"skip={index * SERVICE_DOWNLOAD_CHUNK_BLOCKS} count={SERVICE_DOWNLOAD_CHUNK_BLOCKS} "
                    "2>/dev/null | base64"
                )
                written += out.write(base64.b64decode(await self._run(service, script, what)))
        # A file that changed while being read would otherwise download silently truncated or mixed.
        if written != size:
            raise RuntimeError(f"could not read {what} from compose service {service!r}: got {written} of {size} bytes")

    async def service_download_file(self, source_path: str, target_path: Path | str, *, service: str) -> None:
        """Harbor's ``download_file`` for a service: copy ``source_path`` in the container to ``target_path`` here."""
        await self._download(service, source_path, Path(target_path), source_path)

    async def service_download_dir(self, source_dir: str, target_dir: Path | str, *, service: str) -> None:
        """Harbor's ``download_dir`` for a service: tar the directory in the container, download the archive, and extract it here."""
        # One archive keeps it to a single download, whatever the number of files.
        script = f't=$(mktemp) && tar -C {shlex.quote(source_dir)} -czf "$t" . && echo "$t"'
        remote = (await self._run(service, script, source_dir)).strip()
        try:
            with tempfile.TemporaryDirectory() as scratch:
                local = Path(scratch) / "dir.tgz"
                await self._download(service, remote, local, source_dir)
                target = Path(target_dir)
                target.mkdir(parents=True, exist_ok=True)
                # The "data" filter refuses absolute paths, links out of the target, and device files.
                with tarfile.open(local, mode="r:gz") as archive:
                    archive.extractall(target, filter="data")
        finally:
            # Best effort: a failed cleanup must not replace the download's own error.
            with contextlib.suppress(Exception):
                await self._exec(service, ["rm", "-f", remote], None)

    async def stop_service(self, service: str) -> None:
        """Always unsupported: a container in a running pod can't be stopped on its own."""
        raise ServiceOperationsUnsupportedError(
            f"compose service {service!r} runs as a container in the sandbox pod and can't be stopped on its own"
        )
