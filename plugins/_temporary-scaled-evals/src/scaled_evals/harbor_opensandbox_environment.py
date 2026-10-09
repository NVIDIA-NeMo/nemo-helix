# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo's Harbor environment for running scaled-eval trials in OpenSandbox.

Harbor 0.20's ``OpenSandboxEnvironment`` creates the sandbox, waits for it, and runs the trial.
This subclass keeps all of that and changes what the sandbox is allowed to reach and who owns it:

* Egress comes from a trusted allowlist the runner renders into Harbor's environment kwargs, never
  from ``task.toml``. ``no-network`` tasks get deny-all, ``public`` tasks get the trusted allowlist
  (never unrestricted access), and ``allowlist`` tasks must ask for a subset of it or are rejected
  before any sandbox exists.
* The applied policy is read back after create and the sandbox is killed if it does not match, or
  if it cannot be read at all.
* Every create attempt is labelled, so a sandbox the server created but whose handle was lost to a
  failed or retried request can still be found and killed.
* On an image with a non-root ``USER``, Harbor's ``user="root"`` commands run as the image's user.
  ``execd`` runs as that user and the kernel refuses its switch to uid 0, so asking for root fails
  every such command before it starts, even checks like ``tmux -V``.
* Compose tasks run when the runner passes ``compose_services`` (see
  ``scaled_evals.harbor_opensandbox_services``). The server's NeMo extension adds the services to
  the sandbox pod, and Kubernetes starts them before ``main``. This class gives ``main`` its
  entrypoint and env, fails the create early when a service can't start, waits for ``main``'s and
  the services' readiness, and runs per-service operations through the extension's exec route.

Loaded by Harbor through ``environment.import_path``; see ``NEMO_OPENSANDBOX_IMPORT_PATH`` in
``scaled_evals.dispatch.harbor_opensandbox``. This is the interim home until ``nhx-sandbox`` owns
OpenSandbox access.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import math
import shlex
import tarfile
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, override
from uuid import uuid4

import httpx
from harbor.environments.base import ExecResult, ServiceOperationsUnsupportedError
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.compose_service_ops import ComposeServiceOpsMixin, ComposeServiceTransport
from harbor.environments.opensandbox import OpenSandboxEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from nhx_sandbox.opensandbox_policy import (
    EgressVerificationError,
    EgressVerificationMode,
    canonical_egress_target,
    to_opensandbox_policy,
    verify_applied_egress,
)

from scaled_evals.harbor_opensandbox_cleanup import APPLIED_EGRESS_FILENAME
from scaled_evals.harbor_opensandbox_services import (
    COMPOSE_FILENAMES,
    EXTENSION_KEY,
    MAIN_SERVICE,
    ComposeServices,
    parse_compose_services,
)

# Sandbox metadata label marking which NeMo component manages the sandbox.
MANAGED_BY_METADATA_KEY = "nemo-managed-by"
# Value of MANAGED_BY_METADATA_KEY on every sandbox this environment creates.
MANAGED_BY_METADATA_VALUE = "scaled-evals"
# Sandbox metadata label shared by every sandbox one create call made, including Harbor's retries.
CREATE_ATTEMPT_METADATA_KEY = "nemo-scaled-evals-create-attempt"
_IMAGE_UID_PROBE_TIMEOUT = timedelta(seconds=30)
# How long start() waits for main's readiness check and the services, when main has no timeout_sec.
COMPOSE_READY_TIMEOUT_SEC = 300
COMPOSE_READY_POLL_SEC = 2.0
# The server's exec route accepts at most this timeout.
SERVICE_EXEC_MAX_TIMEOUT_SEC = 3600
SERVICE_EXEC_DEFAULT_TIMEOUT_SEC = 600
# The exec route caps output at 16 MiB; an 8 MiB chunk is about 11 MiB once base64-encoded.
SERVICE_DOWNLOAD_BLOCK_BYTES = 1024 * 1024
SERVICE_DOWNLOAD_CHUNK_BLOCKS = 8


class NemoOpenSandboxEnvironment(ComposeServiceOpsMixin, OpenSandboxEnvironment):
    """``OpenSandboxEnvironment`` with trusted egress, applied-policy verification, cleanup, and compose services.

    Args:
        trusted_allowed_hosts: Destinations trials may reach, as hostnames, ``*.`` wildcard
            hostnames, IP addresses, or CIDRs. Supplied by the runner, not the task.
        resolver_addresses: Nameservers to keep reachable. ``None`` uses the runner's own
            ``/etc/resolv.conf``, which in-cluster is the resolver sandboxes share.
        egress_verification: How strictly to compare the applied policy with the requested one;
            see ``verify_applied_egress``. Readback itself is always required.
        compose_services: The evaluation profile's ``compose_services``. Applied only when the
            environment directory has a Compose file, so a separate verifier environment (built
            from the task's ``tests/``) gets a plain sandbox.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        trusted_allowed_hosts: list[str] | None = None,
        resolver_addresses: list[str] | None = None,
        egress_verification: EgressVerificationMode = "default_action",
        compose_services: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment.__init__ validates network policy support, the definition and the
        # capabilities, which read these.
        has_compose_file = any((environment_dir / name).is_file() for name in COMPOSE_FILENAMES)
        self._services: ComposeServices | None = (
            parse_compose_services(compose_services) if compose_services is not None and has_compose_file else None
        )
        self._trusted_allowed_hosts = tuple(trusted_allowed_hosts or ())
        self._trusted_targets = frozenset(canonical_egress_target(host) for host in self._trusted_allowed_hosts)
        self._resolver_addresses = tuple(resolver_addresses) if resolver_addresses is not None else None
        self._egress_verification: EgressVerificationMode = egress_verification
        self._expected_egress: EgressPolicy | None = None
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
        # Harbor's create call sends these attributes, so the services ride along with every create.
        if self._services is not None:
            self._extensions[EXTENSION_KEY] = self._services.to_json()
            self._entrypoint = self._services.sandbox_entrypoint() or self._entrypoint
            # Compose's environment for main is the base; task and trial env win over it.
            self._persistent_env = {**self._services.main.env, **self._persistent_env}

    @override
    def _validate_definition(self) -> None:
        """Accept the task's Compose file when its services were configured; still require the prebuilt image."""
        if self._services is None:
            super()._validate_definition()
            return
        if not self.task_env_config.docker_image:
            raise FileNotFoundError(
                "OpenSandboxEnvironment requires a prebuilt image. Set task.environment.docker_image."
            )

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        """Advertise every allowlist form ``egress_policy`` can enforce, so Harbor accepts those tasks."""
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=True,
            mounted=False,
            docker_compose=self._services is not None,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv6_addresses=True,
            network_allowlist_ipv4_cidrs=True,
            network_allowlist_ipv6_cidrs=True,
        )

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

    @override
    def _build_network_policy(self, sdk: dict[str, Any]) -> Any:
        """Send the policy fixed for this create call, so the policy verified afterwards is the one sent."""
        expected = self._expected_egress or self.egress_policy()
        return sdk["NetworkPolicy"].model_validate(to_opensandbox_policy(expected))

    @override
    async def _create_sandbox(self, sdk: dict[str, Any]) -> Any:
        """Create the sandbox through Harbor, kill any orphans from the attempt, then verify its egress.

        The sandbox is killed and the trial fails if the applied policy can't be read or doesn't match.
        """
        if self._services is not None:
            await self._require_compose_extension(sdk)

        # Harbor's _create_sandbox retries transient failures internally. One attempt ID spans all
        # of those retries, so the sweeps below also find sandboxes an earlier retry orphaned.
        attempt_id = uuid4().hex
        caller_metadata = dict(self._metadata)
        self._metadata = {
            **caller_metadata,
            MANAGED_BY_METADATA_KEY: MANAGED_BY_METADATA_VALUE,
            CREATE_ATTEMPT_METADATA_KEY: attempt_id,
        }
        self._expected_egress = self.egress_policy()
        try:
            sandbox = await super()._create_sandbox(sdk)
        except BaseException:
            await self._kill_create_attempt(sdk, attempt_id, keep=None)
            raise
        finally:
            self._metadata = caller_metadata

        await self._kill_create_attempt(sdk, attempt_id, keep=_sandbox_id(sandbox))
        try:
            await verify_applied_egress(
                sandbox,
                self._expected_egress,
                mode=self._egress_verification,
                label=f"OpenSandbox sandbox {_sandbox_id(sandbox)} (session {self.session_id})",
                require_readback=True,
            )
        except BaseException as exc:
            await self._safe_kill(sandbox)
            if isinstance(exc, EgressVerificationError):
                raise RuntimeError(str(exc)) from exc
            raise
        self._record_applied_egress(_sandbox_id(sandbox), self._expected_egress)
        return sandbox

    def _server(self) -> _ServerClient:
        """A client for the extension's routes, on the same server and API key as Harbor's SDK calls."""
        return _ServerClient.from_connection_config(self._build_connection_config(self._load_opensandbox()))

    async def _require_compose_extension(self, sdk: dict[str, Any]) -> None:
        """Fail before creating anything if the server can't run Compose services.

        A stock server ignores unknown extensions and would create a sandbox without the services.
        """
        client = _ServerClient.from_connection_config(self._build_connection_config(sdk))
        try:
            response = await client.get("/nemo-ext/health", timeout=30)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach the OpenSandbox compose services extension: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                "The OpenSandbox server does not run the NeMo compose services extension; "
                f"GET /v1/nemo-ext/health returned {response.status_code}. See "
                "k8s/helm/examples/opensandbox/README.md, 'Compose services'."
            )

    async def _services_status(self, sandbox_id: str) -> dict[str, Any]:
        """The extension's view of the sandbox's services: ``{"failure": str | None, "not_ready": [names]}``."""
        response = await self._server().get(f"/nemo-ext/sandboxes/{sandbox_id}/services", timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"reading compose services failed ({response.status_code}): {_error_text(response)}")
        return response.json()

    @override
    async def _terminal_provisioning_failure(self, sandbox: Any) -> str | None:
        """Harbor's check, plus a compose service that can't start, so the create fails now with its logs.

        Kubernetes starts ``main`` only after the services, so without this a crashing service
        would leave Harbor waiting for ``main`` until its ready timeout.
        """
        failure = await super()._terminal_provisioning_failure(sandbox)
        if failure is not None or self._services is None:
            return failure
        try:
            status = await self._services_status(str(_sandbox_id(sandbox)))
        except Exception:
            # The pod may not exist yet, or the read hit a transient error; Harbor polls again.
            return None
        if status.get("failure"):
            return f"OpenSandbox sandbox {_sandbox_id(sandbox)}: {status['failure']}"
        return None

    @override
    async def start(self, force_build: bool) -> None:
        """Start the sandbox; with compose services, also wait for ``main``'s readiness check and every service."""
        await super().start(force_build)
        if self._services is None:
            return

        # One deadline covers both waits, like a Compose healthcheck's overall start period.
        readiness = self._services.main.readiness
        deadline = time.monotonic() + (readiness.timeout_sec if readiness else COMPOSE_READY_TIMEOUT_SEC)
        if readiness is not None:
            await self._wait_for_main(readiness.exec_, deadline)
        # after_main services start with main, so they may still be coming up.
        await self._wait_for_services(deadline)

    async def _wait_for_main(self, argv: list[str], deadline: float) -> None:
        """Run ``argv`` in the sandbox until it exits 0, like a Compose healthcheck of ``main``.

        Kubernetes only knows when the sandbox container started, not when the processes its
        entrypoint launched are up.
        """
        command = shlex.join(argv)
        while True:
            # As root from /, like a Compose healthcheck, whatever the task's default user and workdir.
            remaining = max(1, int(deadline - time.monotonic()))
            last = await self.exec(command, cwd="/", user="root", timeout_sec=min(30, remaining))
            if last.return_code == 0:
                return
            # Out of time: report the check's last output, which usually says what's missing.
            if time.monotonic() >= deadline:
                output = ((last.stdout or "") + (last.stderr or ""))[-2000:]
                raise RuntimeError(f"main was not ready: {command!r} last exited {last.return_code}: {output}")
            await asyncio.sleep(COMPOSE_READY_POLL_SEC)

    async def _wait_for_services(self, deadline: float) -> None:
        """Poll the services route until every service is ready; raise on a failure or at ``deadline``.

        A failure is raised as soon as the route reports one, with the service's log lines.
        """
        sandbox_id = str(_sandbox_id(self._sandbox))
        while True:
            status = await self._services_status(sandbox_id)
            if status["failure"]:
                raise RuntimeError(f"OpenSandbox sandbox {sandbox_id}: {status['failure']}")
            if not status["not_ready"]:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"compose services not ready: {', '.join(status['not_ready'])}")
            await asyncio.sleep(COMPOSE_READY_POLL_SEC)

    @override
    def _compose_service_transport(self, service: str | None) -> ComposeServiceTransport:
        """The transport Harbor's per-service operations (exec, download) use for ``service``.

        Only the profile's services qualify; ``main`` is the sandbox itself and goes through execd.
        """
        if self._services is None:
            raise self._compose_unsupported(service)
        if service not in self._services.names():
            raise ServiceOperationsUnsupportedError(
                f"compose service {service!r} is not one of this sandbox's services {self._services.names()}"
                + (" (main is the sandbox itself)" if service == MAIN_SERVICE else "")
            )
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return _ServiceExecTransport(self._server(), str(_sandbox_id(self._sandbox)))

    def _record_applied_egress(self, sandbox_id: str | None, policy: EgressPolicy) -> None:
        """Write the verified policy and its hash into the trial directory for the supervisor."""
        payload = to_opensandbox_policy(policy)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        record = {
            "sandbox_id": sandbox_id,
            "session_id": self.session_id,
            "network_mode": self._network_policy.network_mode.value,
            "policy_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "policy": payload,
        }
        try:
            self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
            (self.trial_paths.trial_dir / APPLIED_EGRESS_FILENAME).write_text(json.dumps(record, indent=2))
        except OSError:
            self.logger.warning("Could not record the applied egress policy", exc_info=True)

    async def _kill_create_attempt(self, sdk: dict[str, Any], attempt_id: str, *, keep: str | None) -> None:
        """Best-effort kill of every sandbox labelled with ``attempt_id`` except ``keep``."""
        try:
            manager = await sdk["SandboxManager"].create(connection_config=self._build_connection_config(sdk))
        except Exception:
            self.logger.warning("Could not reconcile OpenSandbox create attempt %s", attempt_id, exc_info=True)
            return
        try:
            page = 1
            orphans: list[str] = []
            while True:
                result = await manager.list_sandbox_infos(
                    sdk["SandboxFilter"](metadata={CREATE_ATTEMPT_METADATA_KEY: attempt_id}, page=page)
                )
                orphans.extend(info.id for info in result.sandbox_infos if info.id != keep)
                if not result.pagination.has_next_page:
                    break
                page += 1
            for sandbox_id in orphans:
                try:
                    await manager.kill_sandbox(sandbox_id)
                except Exception:
                    self.logger.warning("Could not kill orphaned OpenSandbox sandbox %s", sandbox_id, exc_info=True)
            if orphans:
                self.logger.warning(
                    "Killed OpenSandbox sandboxes left by create attempt %s: %s", attempt_id, ", ".join(orphans)
                )
        except Exception:
            self.logger.warning("Could not reconcile OpenSandbox create attempt %s", attempt_id, exc_info=True)
        finally:
            try:
                await manager.close()
            except Exception:
                pass

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Learn the image's uid before the first root command, so ``_resolve_uid`` can map it."""
        if not self._image_uid_probed and self._requests_root(user):
            await self._probe_image_uid()
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    @override
    def _resolve_uid(self, user: str | int) -> int | None:
        """Drop a uid 0 request on a non-root image, so the command runs as the image's user."""
        uid = super()._resolve_uid(user)
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

    def _requests_root(self, user: str | int | None) -> bool:
        resolved = self._resolve_user(user)
        return resolved is not None and super()._resolve_uid(resolved) == 0

    async def _probe_image_uid(self) -> None:
        """Run ``id -u`` once as the sandbox default user; on failure, root requests stay uid 0."""
        async with self._image_uid_lock:
            if self._image_uid_probed or self._sandbox is None:
                return
            self._image_uid_probed = True
            sdk = self._load_opensandbox()
            try:
                execution = await self._sandbox.commands.run(
                    "id -u",
                    opts=sdk["RunCommandOpts"](working_directory="/", timeout=_IMAGE_UID_PROBE_TIMEOUT),
                )
                self._image_uid = int("".join(message.text for message in execution.logs.stdout).strip())
            except Exception:
                self.logger.warning(
                    "Could not read the sandbox image's uid; root commands are sent as uid 0", exc_info=True
                )

    @override
    def _load_opensandbox(self) -> dict[str, Any]:
        """Add the SDK classes ``_kill_create_attempt`` needs to the ones Harbor loads."""
        sdk = super()._load_opensandbox()
        from opensandbox.manager import SandboxManager  # ty: ignore[unresolved-import]  # Harbor 0.20 venv only
        from opensandbox.models.sandboxes import SandboxFilter  # ty: ignore[unresolved-import]

        return {**sdk, "SandboxManager": SandboxManager, "SandboxFilter": SandboxFilter}


def _sandbox_id(sandbox: Any) -> str | None:
    """Return the sandbox's ID from whichever of ``id`` or ``sandbox_id`` the SDK object exposes."""
    return getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)


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

    async def get(self, path: str, *, timeout: float) -> httpx.Response:
        """``GET base_url + path``; the response is returned whatever its status."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(f"{self.base_url}{path}", headers=self.headers)

    async def post(self, path: str, body: dict[str, Any], *, timeout: float) -> httpx.Response:
        """``POST base_url + path`` with a JSON ``body``; the response is returned whatever its status."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(f"{self.base_url}{path}", json=body, headers=self.headers)


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

        # The server's 504 is the command's own timeout, which Harbor expects as a result, not an error.
        if response.status_code == 504:
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
