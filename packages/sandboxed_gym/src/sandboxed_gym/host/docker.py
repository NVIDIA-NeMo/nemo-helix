# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A job-host provider that runs the Gym host in a local Docker container.

The OpenSandbox provider needs a Kubernetes control plane and PVC claims, so until one is available
the sandboxed path cannot be *executed* at all -- only its wiring tested with the provider replaced.
This provider closes that gap: same runtime image contract, same bootstrap environment, same HTTP
health and rollout endpoints, on a laptop.

**It is not an isolation boundary.** A Docker container is not kata-qemu, this provider enforces no
egress policy, and it mounts host directories rather than cluster-managed volumes. The
sandboxed-GRPO RFC's threat model is not satisfied by any of that. What it is for is executing and
debugging the path -- and for that, running the real Gym is worth far more than a stub.

``egress_allow`` is therefore accepted and *recorded* rather than applied, and reading it back off
the handle is how a test can assert what a cluster provider would have enforced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from sandboxed_gym.host.models import GymHostHandle, GymHostSpec, GymHostVolumeMount

LOGGER = logging.getLogger(__name__)

_HEALTH_POLL_S = 1.0
_CONTAINER_PREFIX = "nhx-gym-host-"


class DockerHostError(RuntimeError):
    """A docker command failed while provisioning or tearing down the host."""


class DockerGymHostProvider:
    """Provision the Gym host as a container, with host directories standing in for PVCs.

    Args:
        root_dir: Host directory the volume claims resolve under. A mount's ``pvc_claim`` and
            ``sub_path`` become ``<root_dir>/<pvc_claim>/<sub_path>``, so one claim with two
            sub-paths stays two directories, exactly as it would on a cluster.
        docker: Docker executable.
        network: Optional docker network to attach the container to.
    """

    name = "docker"

    def __init__(
        self,
        root_dir: str | None = None,
        docker: str | None = None,
        network: str | None = None,
    ) -> None:
        self._root = Path(root_dir or "/tmp/nhx-gym-host").expanduser()
        self._docker = docker or shutil.which("docker") or "docker"
        self._network = network
        self._containers: dict[str, str] = {}
        self._egress: dict[str, tuple[tuple[str, int], ...]] = {}

    async def _run(self, *argv: str, timeout_s: float = 120.0) -> str:
        process = await asyncio.create_subprocess_exec(
            self._docker, *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            process.kill()
            raise DockerHostError(f"docker {argv[0]} timed out after {timeout_s:g}s") from None
        if process.returncode != 0:
            raise DockerHostError(f"docker {argv[0]} failed ({process.returncode}): {stderr.decode().strip()}")
        return stdout.decode().strip()

    def _host_path(self, mount: GymHostVolumeMount) -> Path:
        path = self._root / mount.pvc_claim
        if mount.sub_path:
            path = path / mount.sub_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _mount_args(self, spec: GymHostSpec) -> list[str]:
        args: list[str] = []
        mounts = [spec.environment_mount, spec.workspace_mount]
        if spec.dataset_mount is not None:
            mounts.append(spec.dataset_mount)
        for mount in mounts:
            source = self._host_path(mount)
            suffix = ":ro" if mount.read_only else ""
            args += ["-v", f"{source}:{mount.mount_path}{suffix}"]
        return args

    async def create_host(self, spec: GymHostSpec) -> GymHostHandle:
        """Start the container and return its published health and rollout URLs."""
        name = _CONTAINER_PREFIX + uuid.uuid4().hex[:12]
        argv = ["run", "-d", "--name", name, "-P", "--expose", str(spec.runtime_http_port)]
        if self._network:
            argv += ["--network", self._network]
        argv += self._mount_args(spec)
        for key, value in spec.bootstrap_env.items():
            argv += ["-e", f"{key}={value}"]
        # The runtime reads its port from the environment; publishing alone would not move it.
        argv += ["-e", f"NHX_RUNTIME_HTTP_PORT={spec.runtime_http_port}"]
        for key, value in (spec.resources or {}).items():
            if key == "cpu":
                argv += ["--cpus", str(value)]
            elif key in {"memory", "memory_mib"}:
                argv += ["--memory", str(value) if key == "memory" else f"{value}m"]
        argv.append(spec.runtime_image)
        # `entrypoint` defaults to the NeMo-RL image's launcher script, which this image does not
        # have; its own CMD starts the runtime, so an unset-for-this-provider entrypoint is right.
        if spec.entrypoint and spec.entrypoint[0] != "/bin/sh":
            argv += list(spec.entrypoint)

        await self._run(*argv)
        self._containers[name] = name
        self._egress[name] = tuple((rule.host, rule.port) for rule in spec.egress_allow)

        try:
            # `docker port` answers e.g. "0.0.0.0:55003", but exits 0 and says nothing at all when
            # the port was never published -- inside the try so that case is cleaned up too.
            published = await self._run("port", name, str(spec.runtime_http_port))
            lines = published.splitlines()
            if not lines:
                raise DockerHostError(f"docker port published no mapping for {spec.runtime_http_port} on {name}")
        except DockerHostError:
            await self._force_remove(name)
            raise
        port = lines[0].rsplit(":", 1)[-1].strip()
        base = f"http://127.0.0.1:{port}"
        LOGGER.info("Gym host %s listening on %s (container %s)", spec.job_id, base, name)
        return GymHostHandle(
            host_id=name,
            health_url=f"{base}/health",
            rollout_url=f"{base}/rollouts/run",
            headers={},
            provider=self,
        )

    async def wait_ready(self, handle: GymHostHandle, timeout_s: float) -> None:
        """Poll ``/health`` until the runtime reports ready.

        Gym starts a Ray cluster and several uvicorn servers before it answers, so a cold start is
        slow. On timeout the container's logs are attached to the error: the useful diagnosis is
        almost always in them, and the container is about to be removed.
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        last: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            # Check liveness before health. A container that failed to bootstrap is never going to
            # answer, and polling it until the ready timeout turns a crash that took two seconds
            # into a fifteen-minute wait with the traceback sitting unread in `docker logs`.
            if not await self._is_running(handle.host_id):
                logs = await self._logs(handle.host_id)
                raise DockerHostError(
                    f"Gym host {handle.host_id} exited before becoming ready"
                    + (f"\n--- container logs ---\n{logs}" if logs else "")
                )
            try:
                body = await asyncio.to_thread(self._get_json, handle.health_url)
                if body.get("status") == "ready":
                    return
                last = RuntimeError(f"host not ready: {body!r}")
            except Exception as exc:
                last = exc
            await asyncio.sleep(_HEALTH_POLL_S)
        logs = await self._logs(handle.host_id)
        raise TimeoutError(
            f"Gym host {handle.host_id} was not ready within {timeout_s:g}s"
            + (f" ({last})" if last else "")
            + (f"\n--- container logs ---\n{logs}" if logs else "")
        )

    async def _is_running(self, name: str) -> bool:
        try:
            return (await self._run("inspect", "-f", "{{.State.Running}}", name)).strip() == "true"
        except DockerHostError:
            return False

    @staticmethod
    def _get_json(url: str) -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())

    async def _logs(self, name: str, tail: int = 80) -> str:
        try:
            return await self._run("logs", "--tail", str(tail), name)
        except DockerHostError:
            return ""

    async def exec_host(self, handle: GymHostHandle, command: str, timeout_s: float | None = None) -> str:
        return await self._run("exec", handle.host_id, "sh", "-c", command, timeout_s=timeout_s or 120.0)

    def egress_recorded_for(self, handle: GymHostHandle) -> tuple[tuple[str, int], ...]:
        """Egress a cluster provider would have enforced for this host. Not applied here."""
        return self._egress.get(handle.host_id, ())

    async def _force_remove(self, name: str) -> None:
        try:
            await self._run("rm", "-f", name)
        except DockerHostError:
            LOGGER.warning("Could not remove Gym host container %s", name)

    async def destroy_host(self, handle: GymHostHandle) -> None:
        await self._force_remove(handle.host_id)
        self._containers.pop(handle.host_id, None)
        self._egress.pop(handle.host_id, None)
