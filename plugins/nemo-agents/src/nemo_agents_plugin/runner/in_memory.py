# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InMemoryRunnerBackend — spawns local agent-server subprocesses.

Agent processes run as local subprocesses on the same machine as the platform
server (``deployment_mode=subprocess``).  Process state is tracked in memory;
it is lost if the platform restarts (orphaned processes must be cleaned up
manually or via OS process management).

Container modes (docker/k8s) use
:class:`~nemo_agents_plugin.runner.deployments_backend.DeploymentsRunnerBackend`
instead.

Module-level helpers ``system_dir`` and ``log_path_for_deployment``
encode the on-disk layout convention so out-of-process callers (e.g. the
``nemo agents logs`` CLI) can locate a deployment's log file without
instantiating the backend.  The convention is intentionally narrow: it is
correct only for the local in-memory backend on the same host.  Once a
remote backend (Docker/K8s) lands, log retrieval should move to a server-
side endpoint that streams content over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml
from nemo_agents_plugin.config import AgentsConfig, ControllerConfig
from nemo_agents_plugin.entities import (
    AGENT_CONFIG_FILENAME,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    ComputeResources,
    DeploymentMode,
)
from nemo_agents_plugin.fabric.gateway_credentials import platform_gateway_credential_env
from nemo_agents_plugin.runner.backend import DeploymentInfo, LocalLog, LogLocation, NotYetAvailable, RunnerBackend
from nemo_agents_plugin.runner.fabric_artifact_staging import stage_fabric_ethos_dir
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.sdk_provider import get_async_platform_sdk

# Match characters not safe for filesystem paths.  Deployment names are
# normally URL-safe identifiers, but we sanitise defensively to ensure we
# never spawn temp files outside the system dir.
_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(name: str) -> str:
    """Coerce a deployment name into a safe single filename component.

    Replaces filesystem-unsafe characters with ``_`` and falls back to
    ``"deployment"`` if the result strips to empty.  Collision-safe so long
    as the agents API enforces URL-safe deployment names upstream (which it
    currently does — names that differ only by unsafe characters cannot
    coexist).
    """
    cleaned = _FILENAME_UNSAFE_RE.sub("_", name).strip("._-") or "deployment"
    return cleaned


def _resolve_nat_bin() -> str:
    """Locate the ``nat`` executable to spawn for an agent deployment.

    Resolution order:

    1. ``shutil.which("nat")`` — handles activated venvs, the agentic
       container (which prepends ``/app/.venv/bin`` to ``PATH``), and any
       setup where the user has ``nat`` on their shell ``PATH``.
    2. Sibling of ``sys.executable`` — covers ``uv tool install
       nemo-platform``, where the tool venv's ``bin/`` is **not** prepended
       to ``PATH`` (uv only symlinks the declared ``[project.scripts]`` into
       ``~/.local/bin``; the rest of the tool venv stays off ``PATH``).
       ``nat`` is co-installed with ``nemo`` in that same venv, so picking
       it up next to ``sys.executable`` is the canonical way to find it.
    3. ``/app/.venv/bin/nat`` — last-resort fallback for the official
       agentic container if the ``PATH`` lookup somehow fails. Kept for
       backwards compatibility with prior behavior.
    """
    if found := shutil.which("nat"):
        return found
    sibling = Path(sys.executable).parent / "nat"
    if sibling.is_file():
        return str(sibling)
    return "/app/.venv/bin/nat"


async def validate_platform_agent_config(config: dict[str, Any], *, base_dir: Path) -> Any:
    """Validate Fabric configs lazily so NAT/container deployments do not import Fabric."""
    # TODO(AIRCORE-902): Hoist once Fabric runtime is installed in the default Platform image.
    from nemo_agents_plugin.fabric.validation import validate_platform_agent_config as _validate_platform_agent_config

    return await _validate_platform_agent_config(config, base_dir=base_dir)


def system_dir(workspace_dir: Path | None = None) -> Path:
    """Return the directory holding rendered configs and per-deployment logs.

    When *workspace_dir* is omitted, the default is read from
    :class:`AgentsConfig` so out-of-process callers (CLI) resolve the same
    path the running platform uses.
    """
    if workspace_dir is None:
        workspace_dir = AgentsConfig.get().controller.workspace_dir
    return workspace_dir.resolve() / "system"


def log_path_for_deployment(workspace: str, name: str, workspace_dir: Path | None = None) -> Path:
    """``<system_dir>/<workspace>/<name>.log`` — also used by the ``nemo agents logs`` CLI."""
    base = system_dir(workspace_dir) / _sanitize_filename(workspace)
    return base / f"{_sanitize_filename(name)}.log"


def config_path_for_deployment(workspace: str, name: str, workspace_dir: Path | None = None) -> Path:
    """Return the absolute rendered-config path for a deployment named *name*."""
    base = system_dir(workspace_dir) / _sanitize_filename(workspace)
    return base / f"{_sanitize_filename(name)}.yaml"


def _write_yaml_config(path: Path, config: dict[str, Any]) -> Path:
    """Atomically write a YAML config to an absolute path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    tmp_path.replace(path)
    return path


logger = logging.getLogger(__name__)


class InMemoryRunnerBackend(RunnerBackend):
    """Manages agent processes as local subprocesses.

    Process state is stored in instance-level dicts.  All public methods are
    async; blocking subprocess calls are dispatched via ``asyncio.to_thread``
    to avoid blocking the event loop.
    """

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._workspace_root: Path = config.workspace_dir.resolve()
        self._processes: dict[tuple[str, str], subprocess.Popen[bytes]] = {}
        self._deployments: dict[tuple[str, str], DeploymentInfo] = {}
        self._next_port: int = config.port_range_start
        self._temp_files: dict[tuple[str, str], Path] = {}
        self._http_client: httpx.AsyncClient | None = None

    @property
    def output_base_dir(self) -> Path:
        """Backend artifact root: ``workspace_dir`` itself (configs/logs live in ``system/``).

        Defaults to ``nmp_user_data_dir() / "agents"`` (e.g.
        ``~/.local/share/nemo/agents``) so artifacts persist across reboots
        and live in a documented, user-accessible location instead of inside
        the plugin source tree.
        """
        return self._workspace_root

    @property
    def system_dir(self) -> Path:
        """Directory under ``output_base_dir`` for rendered configs and log files."""
        return system_dir(self._workspace_root)

    def log_path_for(self, workspace: str, name: str) -> Path:
        """Instance-level convenience wrapper around :func:`log_path_for_deployment`."""
        return log_path_for_deployment(workspace, name, self._workspace_root)

    def get_log_location(self, workspace: str, name: str) -> LogLocation:
        path = log_path_for_deployment(workspace, name, self._workspace_root)
        return LocalLog(path=path) if path.exists() else NotYetAvailable()

    def config_path_for(self, workspace: str, name: str) -> Path:
        """Instance-level convenience wrapper around :func:`config_path_for_deployment`."""
        return config_path_for_deployment(workspace, name, self._workspace_root)

    def allocate_port(self) -> int:
        """Return the next free port in [port_range_start, port_range_end].

        Scans forward from the current position, wrapping around, and probes
        each candidate with ``socket.bind`` to confirm it is not already in
        use.  This handles restarts with orphaned processes, third-party
        processes occupying ports in the range, and reuse of ports freed by
        deleted deployments.

        Raises:
            RuntimeError: If no free port is found in the entire range.
        """
        start = self._config.port_range_start
        end = self._config.port_range_end
        span = end - start + 1
        for offset in range(span):
            candidate = start + (self._next_port - start + offset) % span
            if self._is_port_free(candidate):
                self._next_port = candidate + 1 if candidate < end else start
                return candidate
        raise RuntimeError(
            f"No free port available in range [{start}, {end}]. "
            "Consider adjusting NMP_AGENTS_CONTROLLER_PORT_RANGE_START / _END."
        )

    @staticmethod
    def _is_port_free(port: int) -> bool:
        """Return ``True`` if nothing is currently bound to *port* on loopback."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    async def create_deployment(
        self,
        workspace: str,
        name: str,
        config: dict[str, Any],
        port: int,
        *,
        agent: str = "",
        image: str | None = None,
        deployment_mode: DeploymentMode = "subprocess",
        created_by: str | None = None,
        auth_context: AuthContext | None = None,
        resources: ComputeResources | None = None,
        secrets: dict[str, str] | None = None,
        use_image_entrypoint: bool = False,
    ) -> DeploymentInfo:
        """Start a local deployment for NAT workflows or Platform-owned agents."""
        # created_by drives on-behalf-of delegation only for container modes (via
        # the auth-proxy sidecar). Subprocess deployments run in-process on the
        # platform host and do not use the sidecar, so it does not apply here.
        # resources (compute spec) and secrets (secret-env refs) only apply to
        # container modes; subprocess runs in-process on the platform host with no
        # resource isolation and injects no managed secrets.
        del image, deployment_mode, created_by, auth_context, resources, secrets, use_image_entrypoint
        if config.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            return await self._create_fabric_deployment(workspace, name, config, port, agent=agent)

        key = (workspace, name)
        config_path = await asyncio.to_thread(self._write_config, workspace, name, config)
        log_path = self.log_path_for(workspace, name)
        try:
            proc = await asyncio.to_thread(self._spawn, name, config_path, log_path, port)
        except Exception:
            config_path.unlink(missing_ok=True)
            raise

        info = DeploymentInfo(
            name=name,
            status="starting",
            port=port,
            pid=proc.pid,
            endpoint=f"http://127.0.0.1:{port}",
            log_path=str(log_path),
        )
        self._processes[key] = proc
        self._deployments[key] = info
        self._temp_files[key] = config_path
        logger.info(
            "Spawned agent process for '%s/%s' (pid=%d, port=%d, log=%s)",
            workspace,
            name,
            proc.pid,
            port,
            log_path,
        )
        return info

    async def _create_fabric_deployment(
        self,
        workspace: str,
        name: str,
        config: dict[str, Any],
        port: int,
        *,
        agent: str = "",
    ) -> DeploymentInfo:
        """Validate and start a Platform-owned Fabric-backed agent server."""
        key = (workspace, name)
        base_dir = self._fabric_base_dir_for(workspace, name)
        await asyncio.to_thread(base_dir.mkdir, parents=True, exist_ok=True)
        try:
            await self._stage_ethos(workspace, agent, config, base_dir)
            config_path = await asyncio.to_thread(self._write_fabric_config, base_dir, config)
            await validate_platform_agent_config(config, base_dir=base_dir)
            log_path = self.log_path_for(workspace, name)
            credential_env = platform_gateway_credential_env(config)
            proc = await asyncio.to_thread(
                self._spawn_fabric,
                name,
                config_path,
                log_path,
                port,
                credential_env,
            )
        except Exception:
            await asyncio.to_thread(shutil.rmtree, base_dir, ignore_errors=True)
            raise

        info = DeploymentInfo(
            name=name,
            status="starting",
            port=port,
            pid=proc.pid,
            endpoint=f"http://127.0.0.1:{port}",
            log_path=str(log_path),
            extra={"base_dir": str(base_dir)},
        )
        self._processes[key] = proc
        self._deployments[key] = info
        logger.info(
            "Spawned Fabric-backed deployment for '%s/%s' (pid=%d, port=%d, base_dir=%s, log=%s)",
            workspace,
            name,
            proc.pid,
            port,
            base_dir,
            log_path,
        )
        return info

    async def get_deployment_status(self, workspace: str, name: str) -> DeploymentInfo | None:
        key = (workspace, name)
        info = self._deployments.get(key)
        if info is None:
            return None

        proc = self._processes.get(key)
        if proc is not None and proc.poll() is not None:
            info.status = "failed"
            info.error = f"Process exited with code {proc.returncode}"
        return info

    async def delete_deployment(self, workspace: str, name: str) -> bool:
        key = (workspace, name)
        proc = self._processes.pop(key, None)
        info = self._deployments.pop(key, None)
        config_path = self._temp_files.pop(key, None)

        if proc is None and info is None:
            # Already gone — safe for the controller to remove the entity.
            return True

        if proc is not None:
            await asyncio.to_thread(self._terminate, name, proc)

        if config_path is not None:
            config_path.unlink(missing_ok=True)

        if info is not None and (base_dir := info.extra.get("base_dir")):
            await asyncio.to_thread(shutil.rmtree, base_dir, ignore_errors=True)

        logger.info("Deleted agent deployment '%s/%s'", workspace, name)
        return True

    async def list_deployments(self, workspace: str | None = None) -> list[DeploymentInfo]:
        if workspace is None:
            return list(self._deployments.values())
        return [info for (ws, _), info in self._deployments.items() if ws == workspace]

    async def health_check(self, endpoint: str) -> bool:
        url = endpoint.rstrip("/") + "/health"
        try:
            client = self._get_http_client()
            resp = await client.get(url)
            return resp.status_code < 400
        except Exception:
            return False

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def shutdown(self) -> None:
        """Terminate all managed processes (best-effort)."""
        items = list(self._processes.items())
        fabric_base_dirs: list[Path] = []
        for info in self._deployments.values():
            base_dir = info.extra.get("base_dir")
            if base_dir is not None:
                fabric_base_dirs.append(Path(base_dir))
        labels = [f"{ws}/{nm}" for (ws, nm), _ in items]
        results = await asyncio.gather(
            *(asyncio.to_thread(self._terminate, f"{ws}/{nm}", proc) for (ws, nm), proc in items),
            return_exceptions=True,
        )
        self._processes.clear()
        for label, result in zip(labels, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Error terminating '%s' during shutdown", label, exc_info=result)
        self._deployments.clear()
        for path in self._temp_files.values():
            path.unlink(missing_ok=True)
        self._temp_files.clear()
        await asyncio.gather(*(asyncio.to_thread(shutil.rmtree, path, ignore_errors=True) for path in fabric_base_dirs))
        if self._http_client is not None and not self._http_client.is_closed:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.warning("Error closing HTTP client during shutdown", exc_info=True)
            self._http_client = None
        logger.info("InMemoryRunnerBackend shut down — all processes terminated.")

    def _write_config(self, workspace: str, name: str, config: dict[str, Any]) -> Path:
        return _write_yaml_config(self.config_path_for(workspace, name), config)

    def _fabric_base_dir_for(self, workspace: str, name: str) -> Path:
        """Return the local base directory used by a Fabric-backed deployment."""
        return self.system_dir / _sanitize_filename(workspace) / f"{_sanitize_filename(name)}-fabric"

    def _write_fabric_config(self, base_dir: Path, config: dict[str, Any]) -> Path:
        """Write a Platform-owned agent config into its deployment directory."""
        return _write_yaml_config(base_dir / AGENT_CONFIG_FILENAME, config)

    async def _stage_ethos(
        self,
        workspace: str,
        agent: str,
        config: dict[str, Any],
        base_dir: Path,
    ) -> None:
        """Deliver the agent's Ethos fileset into *base_dir*."""
        sdk = get_async_platform_sdk(as_service="agents", internal=True) if agent else None
        await stage_fabric_ethos_dir(
            workspace=workspace,
            agent_name=agent,
            agent_config=config,
            base_dir=base_dir,
            files_client=client_from_platform(sdk, AsyncFilesClient) if sdk else None,
        )

    def _spawn(
        self,
        name: str,
        config_path: Path,
        log_path: Path,
        port: int,
    ) -> subprocess.Popen[bytes]:
        """Spawn ``nat start fastapi`` bound to 127.0.0.1 (platform-internal only).

        ``--host`` is intentionally omitted: processes bind to the default
        (127.0.0.1) so they are not directly reachable externally.  All
        traffic reaches them through the agents gateway proxy.
        """
        # config_path is an absolute path — pass it as-is so nat can find it
        # regardless of the working directory it inherits.
        nat_bin = _resolve_nat_bin()
        cmd = [
            nat_bin,
            "start",
            "fastapi",
            "--config_file",
            str(config_path),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Spawning: %s  (log: %s)", " ".join(cmd), log_path)
        log_file = log_path.open("w")
        try:
            return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        finally:
            # Parent closes its copy of the fd; the child retains its inherited
            # copy for the lifetime of the subprocess.
            log_file.close()

    def _spawn_fabric(
        self,
        name: str,
        config_path: Path,
        log_path: Path,
        port: int,
        credential_env: dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes]:
        """Spawn the Platform-owned Fabric server on a loopback port."""
        cmd = [
            sys.executable,
            "-m",
            "nemo_agents_plugin.fabric.server",
            "--agent-config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Spawning: %s  (log: %s)", " ".join(cmd), log_path)
        log_file = log_path.open("w")
        child_env = os.environ.copy()
        child_env.update(credential_env or {})
        try:
            return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=child_env)
        finally:
            log_file.close()

    def _terminate(self, name: str, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        logger.debug("Terminated process for '%s' (pid=%d)", name, proc.pid)
