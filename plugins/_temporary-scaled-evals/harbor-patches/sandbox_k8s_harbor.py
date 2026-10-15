# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor framework adapter for sandbox-k8s.

Implements Harbor's ``BaseEnvironment`` interface backed by the K8s agent-sandbox
(CRD or Claim mode). This lets Harbor orchestrators use K8s sandboxes via the
standard ``import_path`` mechanism::

    # In a Harbor task config:
    [environment]
    import_path = "sandbox_k8s.harbor:K8sSandboxEnvironment"

Workarounds
-----------
* **EnvironmentType**: Harbor's ``EnvironmentType`` enum has no ``KUBERNETES``
  member. We return ``EnvironmentType.DOCKER`` from ``type()`` as a stand-in.
  Upstream fix: add ``KUBERNETES = "kubernetes"`` to the Harbor enum.

* **TrialPaths**: Harbor passes a ``TrialPaths`` dataclass that assumes a
  host-local ``trial_dir``.  We create the directory structure on ``start()``
  but the sandbox itself runs in a K8s pod, so these paths are only used for
  local log/artifact collection via ``download_dir``.

Install with::

    pip install sandbox-k8s[harbor]
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import math
import os
import secrets
import shutil
import tarfile
import tempfile
import textwrap
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from sandbox_k8s.exceptions import SandboxConfigError
from sandbox_k8s.sandbox import K8sSandbox
from sandbox_k8s.sidecars import sanitize_k8s_name
from sandbox_k8s.types import (
    CommandResult,
    EmptyDirVolume,
    ResourceRequirements,
    Toleration,
    Volume,
    VolumeMount,
)

logger = logging.getLogger(__name__)

_DEFAULT_HARBOR_AGENT_DIRS = (
    "/tmp/agent-home/.claude/skills",
    "/tmp/agent-home/.local/bin",
    "/tmp/agent-home/sessions/debug",
    "/tmp/agent-home/sessions/projects/-app",
    "/tmp/agent-home/sessions/shell-snapshots",
    "/tmp/agent-home/sessions/statsig",
    "/tmp/agent-home/sessions/todos",
    "/tmp/agent-home/sessions/skills",
)

_HARBOR_PATH_ALLOW_PATTERNS = (
    r"^/logs(/|$)",
    r"^/tests(/|$)",
    r"^/solution(/|$)",
    r"^/installed-agent(/|$)",
    r"^/installed-tools(/|$)",
    r"^/git(/|$)",
    # Harbor's conventional task working directory. Separate verifier mode moves
    # declared artifacts out of the agent container and back into the verifier's,
    # and benchmarks overwhelmingly declare them under /app.
    r"^/app(/|$)",
    # A task's working tree under the container HOME. Reverse-engineering and
    # exploit tasks conventionally root their subject binary at /root/<name>
    # and declare artifacts there, which the base deny list refuses wholesale.
    r"^/root/",
    # Cross-service evidence directories. Multi-service tasks put the record a
    # verifier grades -- an append-only action log, a decision journal -- on a
    # volume shared with the service that writes it, and declare it as a
    # per-service artifact. The base list already allows /shared, but only with
    # a trailing slash, so a task declaring the directory itself is refused.
    r"^/audit(/|$)",
    r"^/shared(/|$)",
)

# The base deny list groups /root with /etc, /proc and /sys as a system path,
# but in this runtime /root is the task container's HOME: it holds task-owned
# files, not host state. What actually needs protecting there is agent state --
# credentials and caches, which live in dotfiles. So the blanket prefix deny is
# replaced by two narrower ones: the home directory itself stays unreadable, so
# no task can declare it as one bulk artifact, and every dotfile child stays
# denied in both directions. A nested dotdir (/root/<task>/.git) is not covered
# and does not need to be; it belongs to the task's own tree.
_HARBOR_PATH_DENY_REPLACEMENTS = {
    r"^/root": (r"^/root/?$", r"^/root/\."),
}

_SANDBOX_LOG_POLL_INTERVAL_SECONDS = 5.0
_SANDBOX_LOG_POLL_TIMEOUT_SECONDS = 15.0
_SANDBOX_LOG_MAX_BYTES_PER_FILE = 64 * 1024
_SANDBOX_LOG_MAX_BYTES_PER_POLL = 256 * 1024
_PIP_ZIPAPP = Path("/opt/harbor/pip.pyz")
_SANDBOX_LOG_MAX_LINE_CHARS = 16 * 1024
_AUTOSCALER_SAFE_TO_EVICT_ANNOTATION = "cluster-autoscaler.kubernetes.io/safe-to-evict"
_SANDBOX_LOG_POLL_SCRIPT = r"""
import base64
import json
import os
import pathlib
import sys

offsets = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
remaining = int(sys.argv[2])
per_file = int(sys.argv[3])
root = pathlib.Path("/logs")
if root.is_dir():
    for path in sorted(root.rglob("*")):
        if remaining <= 0:
            break
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
            previous = int(offsets.get(relative, 0))
            reset = previous > size
            offset = 0 if reset else previous
            limit = min(per_file, remaining)
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(limit)
        except OSError:
            continue
        if not data:
            continue
        remaining -= len(data)
        print(json.dumps({
            "path": relative,
            "offset": offset,
            "next_offset": offset + len(data),
            "reset": reset,
            "data": base64.b64encode(data).decode(),
        }, separators=(",", ":")))
"""


def _is_sandbox_creation_conflict(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "SandboxCreationError" in text and ("Conflict" in text or "already exists" in text.lower())


async def _delete_stale_sandbox(sandbox: Any) -> None:
    """Delete a conflicting CR without permanently stopping the new SDK object."""
    backend = sandbox._backend
    if hasattr(backend, "_template_name"):
        await backend._run_sync(
            backend._client.delete_claim,
            backend._name,
            backend._namespace,
        )
        return

    policy_name = _sanitize_k8s_name(f"{backend._name}-network-policy")
    with suppress(Exception):
        await backend._run_sync(
            backend._client.delete_network_policy,
            policy_name,
            backend._namespace,
        )
    await backend._run_sync(
        backend._client.delete_sandbox,
        backend._name,
        backend._namespace,
        wait=True,
        wait_timeout=60.0,
    )


_HARBOR_CRD_EMPTY_DIR_MOUNTS = (
    ("harbor-logs", "/logs"),
    ("harbor-tests", "/tests"),
    ("harbor-solution", "/solution"),
    ("harbor-installed-agent", "/installed-agent"),
)
_ROOTLESS_CRD_EMPTY_DIR_MOUNTS = (("rootless-git", "/git"),)

_ROOTLESS_CONTAINER_COMMAND = [
    "/bin/sh",
    "-c",
    "while [ ! -x /installed-tools/bin/rootless-supervisor ]; do sleep 1; done; "
    "exec /installed-tools/bin/rootless-supervisor",
]

# Directory whose ``start`` file the supervisor polls to begin serving. The path
# is a handshake with the rootless-supervisor binary shipped in the tools image,
# so a tools image using a different location overrides it here.
_ROOTLESS_SIGNAL_DIR = os.environ.get("SANDBOX_K8S_ROOTLESS_SIGNAL_DIR", "/tmp/rootless-supervisor")

# ---------------------------------------------------------------------------
# Lazy imports for Harbor types -- these are only available when the ``harbor``
# extra is installed.  We defer the import so that the rest of sandbox_k8s
# works without harbor.
# ---------------------------------------------------------------------------

_HARBOR_IMPORTED = False
_BaseEnvironment: Any = None
_ExecResult: Any = None
_EnvironmentType: Any = None
_EnvironmentConfig: Any = None
_TrialPaths: Any = None


def _import_harbor_types() -> Any | None:
    global _HARBOR_IMPORTED, _BaseEnvironment, _ExecResult  # noqa: PLW0603
    global _EnvironmentType, _EnvironmentConfig, _TrialPaths  # noqa: PLW0603
    if _HARBOR_IMPORTED:
        return _BaseEnvironment
    try:
        from harbor.environments.base import BaseEnvironment, ExecResult
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.task.config import EnvironmentConfig
        from harbor.models.trial.paths import TrialPaths
    except ImportError:
        return None

    _BaseEnvironment = BaseEnvironment
    _ExecResult = ExecResult
    _EnvironmentType = EnvironmentType
    _EnvironmentConfig = EnvironmentConfig
    _TrialPaths = TrialPaths
    _HARBOR_IMPORTED = True
    return BaseEnvironment


def _ensure_harbor() -> None:
    if _import_harbor_types() is None:
        raise ImportError(
            "Harbor framework is required for the Harbor adapter. Install with: pip install sandbox-k8s[harbor]"
        )


def _service_operations_unsupported(message: str) -> RuntimeError:
    try:
        from harbor.environments.base import ServiceOperationsUnsupportedError
    except ImportError:
        return RuntimeError(message)
    return ServiceOperationsUnsupportedError(message)


def _add_harbor_crd_mounts(
    volumes: list[Volume],
    volume_mounts: list[VolumeMount],
) -> None:
    """Add writable emptyDir mounts Harbor expects when CRD mode has no template."""
    mounted_paths = {mount.mount_path for mount in volume_mounts}
    volume_names = {volume.name for volume in volumes}
    for name, mount_path in _HARBOR_CRD_EMPTY_DIR_MOUNTS:
        if mount_path in mounted_paths:
            continue
        volume_name = name
        if volume_name in volume_names:
            volume_name = f"{name}-{secrets.token_hex(3)}"
        volumes.append(Volume(name=volume_name, empty_dir=EmptyDirVolume()))
        volume_mounts.append(VolumeMount(name=volume_name, mount_path=mount_path))
        volume_names.add(volume_name)
        mounted_paths.add(mount_path)


def _add_rootless_crd_mounts(
    volumes: list[Volume],
    volume_mounts: list[VolumeMount],
) -> None:
    """Add writable roots that cannot be created on a read-only task image."""
    mounted_paths = {mount.mount_path for mount in volume_mounts}
    volume_names = {volume.name for volume in volumes}
    for name, mount_path in _ROOTLESS_CRD_EMPTY_DIR_MOUNTS:
        if mount_path in mounted_paths:
            continue
        volume_name = name
        if volume_name in volume_names:
            volume_name = f"{name}-{secrets.token_hex(3)}"
        volumes.append(Volume(name=volume_name, empty_dir=EmptyDirVolume()))
        volume_mounts.append(VolumeMount(name=volume_name, mount_path=mount_path))
        volume_names.add(volume_name)
        mounted_paths.add(mount_path)


def _coerce_volume(value: Volume | Mapping[str, Any]) -> Volume:
    """Convert profile YAML volume mappings to sandbox-k8s value objects."""
    if isinstance(value, Volume):
        return value
    if not isinstance(value, Mapping):
        raise SandboxConfigError("volumes entries must be Volume objects or mappings")

    config = dict(value)
    empty_dir = config.get("empty_dir")
    if isinstance(empty_dir, Mapping):
        config["empty_dir"] = EmptyDirVolume(**dict(empty_dir))
    try:
        return Volume(**config)
    except TypeError as exc:
        raise SandboxConfigError(f"invalid volume config: {exc}") from exc


def _coerce_volume_mount(value: VolumeMount | Mapping[str, Any]) -> VolumeMount:
    """Convert profile YAML volume-mount mappings to sandbox-k8s value objects."""
    if isinstance(value, VolumeMount):
        return value
    if not isinstance(value, Mapping):
        raise SandboxConfigError("volume_mounts entries must be VolumeMount objects or mappings")
    try:
        return VolumeMount(**dict(value))
    except TypeError as exc:
        raise SandboxConfigError(f"invalid volume mount config: {exc}") from exc


def _normalize_image_pull_secrets(value: Any) -> list[str] | None:
    """Normalize Harbor CLI/env image pull secret overrides for K8sSandbox."""
    if value is None:
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        if not all(isinstance(item, str) for item in value):
            raise SandboxConfigError("image_pull_secrets must be a string or a sequence of strings")
        values = [item.strip() for item in value]
    else:
        raise SandboxConfigError("image_pull_secrets must be a string or a sequence of strings")
    secrets = [item for item in values if item]
    return secrets or None


def _sandbox_annotations(value: Any) -> dict[str, str]:
    annotations = dict(value or {})
    # The task filesystem and active agent session cannot resume in a replacement pod.
    annotations[_AUTOSCALER_SAFE_TO_EVICT_ANNOTATION] = "false"
    return annotations


def _normalize_gpu_count(value: Any) -> int:
    """Treat Harbor's omitted/null GPU field as the CPU-only default."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise SandboxConfigError("task environment gpus must be an integer")
    if value < 0:
        raise SandboxConfigError("task environment gpus must be zero or a positive integer")
    return value


def _resolve_command_timeout(command_timeout: float | None, startup_timeout: int) -> float:
    """Keep command execution independent from the sandbox readiness budget."""
    if command_timeout is None:
        return float(startup_timeout)
    if (
        isinstance(command_timeout, bool)
        or not isinstance(command_timeout, int | float)
        or not math.isfinite(command_timeout)
        or command_timeout <= 0
    ):
        raise SandboxConfigError("command_timeout must be a positive number")
    return float(command_timeout)


def _validate_writable_root_request(
    *,
    read_only_root_filesystem: bool,
    requested_uid: int | None,
    root_authorized: bool,
    writable_root_authorized: bool,
    template_name: str | None,
    pod_spec: Any | None,
) -> None:
    if read_only_root_filesystem:
        return
    if template_name:
        raise SandboxConfigError("writable root requires sandbox_k8s direct CRD mode")
    if pod_spec is not None:
        raise SandboxConfigError("writable root is incompatible with full pod_spec overrides")
    if requested_uid != 0 or not root_authorized or not writable_root_authorized:
        raise SandboxConfigError(
            "read_only_root_filesystem=false requires operator-authorized run_as_user=0 for the exact task image digest"
        )


def _gpu_resources(resources: Any, gpu_count: int) -> ResourceRequirements | Any:
    """Add an explicit Harbor GPU request to sandbox container resources.

    CPU tasks retain the caller's resource object unchanged. GPU tasks require
    equal Kubernetes requests and limits so admission cannot accept a request
    that later runs without a device. An environment-level resource override
    may add CPU/memory, but it may not contradict the task's GPU declaration.
    """
    if gpu_count < 0:
        raise SandboxConfigError("task environment gpus must be zero or a positive integer")
    if gpu_count == 0:
        return resources

    if resources is None:
        parsed = ResourceRequirements()
    elif isinstance(resources, ResourceRequirements):
        parsed = resources.model_copy(deep=True)
    elif isinstance(resources, dict):
        parsed = ResourceRequirements.model_validate(resources)
    else:
        raise SandboxConfigError("resources must be sandbox_k8s ResourceRequirements or a mapping")

    expected = str(gpu_count)
    requests = dict(parsed.requests or {})
    limits = dict(parsed.limits or {})
    for resource_set, values in (("requests", requests), ("limits", limits)):
        configured = values.get("nvidia.com/gpu")
        if configured is not None and str(configured) != expected:
            raise SandboxConfigError(
                f"task requests {expected} GPU(s), but resources.{resource_set} sets nvidia.com/gpu={configured!r}"
            )
        values["nvidia.com/gpu"] = expected
    parsed.requests = requests
    parsed.limits = limits
    return parsed


def _gpu_tolerations(tolerations: Any, gpu_count: int) -> Any:
    """Add the conventional NVIDIA GPU-node toleration for GPU tasks only."""
    if gpu_count <= 0:
        return tolerations

    parsed: list[Toleration] = []
    for value in tolerations or []:
        if isinstance(value, Toleration):
            parsed.append(value)
        elif isinstance(value, dict):
            parsed.append(Toleration.model_validate(value))
        else:
            raise SandboxConfigError("tolerations must contain sandbox_k8s Toleration values or mappings")
    if not any(
        value.key == "nvidia.com/gpu" and value.operator == "Exists" and value.effect == "NoSchedule"
        for value in parsed
    ):
        parsed.append(Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"))
    return parsed


def _task_config_docker_image(task_env_config: Any) -> str | None:
    """Return Harbor's task-local docker_image/image setting when present."""
    raw: Any = None
    if isinstance(task_env_config, Mapping):
        raw = task_env_config.get("docker_image") or task_env_config.get("image")
    else:
        raw = getattr(task_env_config, "docker_image", None) or getattr(task_env_config, "image", None)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _is_verifier_sandbox(session_id: str) -> bool:
    """Detect Harbor's separate verifier sandbox.

    The phase is in the session id (``task__aBcDeF__verifier__trial``) and only
    there. ``environment_name`` is the task's environment directory, so it reads
    the same for the agent sandbox and the verifier sandbox and can never tell
    them apart -- checking it silently classifies every verifier as an agent.
    """
    return "__verifier__" in session_id


def _effective_sandbox_image(
    *,
    session_id: str,
    task_env_config: Any,
    profile_image: str | None,
) -> str | None:
    """Choose the K8s image without letting profile image mask verifier images.

    scaled-evals binds the finalized task image into the profile-level
    ``image`` slot so primary agent sandboxes keep using the admitted signed
    task image. DeepSWE-style tasks also set a separate verifier
    ``docker_image`` in task.toml; Harbor carries that on ``task_env_config``
    when it starts the verifier sandbox. Prefer that task-local image only for
    verifier sandboxes so separate verifier images keep their hidden tests.
    """
    task_image = _task_config_docker_image(task_env_config)
    if task_image and _is_verifier_sandbox(session_id):
        return task_image
    return profile_image or task_image


# ---------------------------------------------------------------------------
# Helper: translate CommandResult → ExecResult
# ---------------------------------------------------------------------------


def _to_exec_result(result: CommandResult) -> Any:
    """Convert a sandbox_k8s CommandResult to a Harbor ExecResult."""
    _ensure_harbor()
    return _ExecResult(
        stdout=result.stdout,
        stderr=result.stderr,
        return_code=result.exit_code,
    )


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

if TYPE_CHECKING:

    class _HarborBaseEnvironment:
        pass
else:
    _HarborBaseEnvironment = _import_harbor_types() or object


class K8sSandboxEnvironment(_HarborBaseEnvironment):
    """Harbor ``BaseEnvironment`` implementation backed by K8s agent-sandbox.

    The adapter delegates all container operations to :class:`K8sSandbox` while
    satisfying Harbor's interface contract (``start``, ``stop``, ``exec``,
    ``upload_file``, ``download_file``, etc.).

    Extra ``**kwargs`` passed through the Harbor factory are forwarded to the
    underlying ``K8sSandbox`` constructor.  Useful keys include:

    * ``template_name`` – enables Claim mode (sub-second startup)
    * ``kubeconfig_path`` / ``context`` / ``in_cluster`` – K8s auth
    * ``image`` – container image override
    * ``resources`` – CPU/memory requests/limits
    * ``namespace`` – K8s namespace (also derived from ``task_env_config``)
    * ``command_timeout`` – default command budget, independent of sandbox readiness
    * ``harbor_agent_dirs`` – extra writable directories needed by agent images
    """

    # -- Harbor protocol attributes (set in __init__) -----------------------
    environment_dir: Path
    environment_name: str
    session_id: str
    trial_paths: Any  # TrialPaths
    task_env_config: Any  # EnvironmentConfig
    logger: logging.Logger
    default_user: str | int | None

    def __init__(
        self,
        environment_dir: Path | str,
        environment_name: str,
        session_id: str,
        trial_paths: Any,
        task_env_config: Any,
        logger: logging.Logger | None = None,
        # Resource overrides (Harbor factory passes these)
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        suppress_override_warnings: bool = False,
        persistent_env: dict[str, str] | None = None,
        # K8sSandbox-specific kwargs
        namespace: str | None = None,
        template_name: str | None = None,
        kubeconfig_path: str | None = None,
        context: str | None = None,
        in_cluster: bool = False,
        verify_ssl: bool = True,
        image: str | None = None,
        working_dir: str = "/workspace",
        timeout: int = 300,
        command_timeout: float | None = None,
        lifecycle_timeout: int = 3600,
        skip_network_policy_check: bool = False,
        harbor_agent_dirs: Sequence[str] | None = None,
        pod_spec: Any | None = None,
        default_container: str | None = "sandbox",
        network_policy: Any | None = None,
        k8s_network_policy: Any | None = None,
        sidecar_wait_ports: list[tuple[str, int]] | None = None,
        sidecar_wait_timeout: int = 120,
        sidecar_log_containers: list[str] | None = None,
        setup_command: str | None = None,
        rootless_overlay: bool = False,
        enable_ipv6_loopback: bool = False,
        read_only_root_filesystem: bool = True,
        rootless_start_timeout: int = 600,
        _scaled_evals_root_authorized: bool = False,
        _scaled_evals_writable_root_authorized: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_harbor()

        # Sidecar containers get the same name sanitization the SDK applies when
        # it builds the pod, so a task's compose service name resolves to the
        # container that service operations must target. Computed before
        # ``super().__init__`` because Harbor's capability validators read
        # ``capabilities`` during base initialization, and this set is what the
        # ``docker_compose`` capability is derived from.
        self._sidecar_containers = {
            sanitize_k8s_name(name)
            for name in (
                sidecar.get("name") if isinstance(sidecar, Mapping) else getattr(sidecar, "name", None)
                for sidecar in kwargs.get("sidecars") or []
            )
            if name
        }

        # The separate verifier sandbox gets no copy of the services. Under
        # Docker a verifier joins the running compose project, so the services
        # exist once; giving the verifier pod its own set is not that, it is a
        # second empty instance. And because every container in a pod shares one
        # network namespace, that copy takes the ports the verifier's own stack
        # needs: a task that boots postgres on 5432 and its app on 8069 to grade
        # a restored snapshot finds both already bound, silently grades the
        # empty copy instead, and reports a plausible wrong score. Dropping them
        # also halves what the verifier phase requests, which matters where the
        # agent and verifier pods are alive together under one namespace quota.
        #
        # ``_sidecar_containers`` is deliberately left intact: it is what the
        # ``docker_compose`` capability is derived from, and a verifier that
        # reported no compose support would fail Harbor's validation for any
        # task that declares services.
        dropped_verifier_sidecars = 0
        self._withheld_sidecar_containers: set[str] = set()
        if _is_verifier_sandbox(session_id) and kwargs.get("sidecars"):
            dropped_verifier_sidecars = len(kwargs.pop("sidecars"))
            self._withheld_sidecar_containers = set(self._sidecar_containers)
            sidecar_wait_ports = None
            sidecar_log_containers = None

        # Let Harbor initialize its internal execution context. Reimplementing
        # this constructor left newer fields such as `_exec_env_overlays`
        # absent when Harbor evolved, causing trials to fail before agent
        # setup. BaseEnvironment accepts provider-specific extras via kwargs.
        super().__init__(
            environment_dir=Path(environment_dir),
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_storage_mb=override_storage_mb,
            override_gpus=override_gpus,
            suppress_override_warnings=suppress_override_warnings,
            persistent_env=persistent_env,
            network_policy=network_policy,
        )
        self.logger = self.logger.getChild("k8s")
        # Which services a sandbox ended up with, said out loud. A pod shares one
        # network namespace, so an unexpected copy of a service silently takes the
        # port the sandbox's own stack wants and the run reports a low score
        # rather than an error -- indistinguishable, from the outside, from an
        # agent that did the work badly.
        if self._sidecar_containers or dropped_verifier_sidecars:
            self.logger.info(
                "Sandbox %s: task declares %d service(s); %s",
                environment_name,
                len(self._sidecar_containers),
                (
                    f"{dropped_verifier_sidecars} withheld so the verifier boots its own"
                    if dropped_verifier_sidecars
                    else "running them alongside the task container"
                ),
            )
        self._working_dir = working_dir
        self._harbor_agent_dirs = tuple(_DEFAULT_HARBOR_AGENT_DIRS if harbor_agent_dirs is None else harbor_agent_dirs)
        if (
            isinstance(sidecar_wait_timeout, bool)
            or not isinstance(sidecar_wait_timeout, int)
            or sidecar_wait_timeout <= 0
        ):
            raise SandboxConfigError("sidecar_wait_timeout must be a positive integer")
        self._sidecar_wait_ports = sidecar_wait_ports or []
        self._sidecar_wait_timeout = sidecar_wait_timeout
        self._sidecar_log_containers = sidecar_log_containers or []
        self._setup_command = setup_command
        self._live_log_task: asyncio.Task[None] | None = None
        self._live_log_offsets: dict[str, int] = {}
        self._live_log_buffers: dict[str, str] = {}
        self._live_log_warned = False
        self._timeout = timeout
        self._command_timeout = _resolve_command_timeout(command_timeout, timeout)
        if not isinstance(rootless_overlay, bool):
            raise SandboxConfigError("rootless_overlay must be a boolean")
        if not isinstance(enable_ipv6_loopback, bool):
            raise SandboxConfigError("enable_ipv6_loopback must be a boolean")
        if not isinstance(read_only_root_filesystem, bool):
            raise SandboxConfigError("read_only_root_filesystem must be a boolean")
        if (
            isinstance(rootless_start_timeout, bool)
            or not isinstance(rootless_start_timeout, int)
            or rootless_start_timeout <= 0
        ):
            raise SandboxConfigError("rootless_start_timeout must be a positive integer")
        if rootless_overlay and template_name:
            raise SandboxConfigError(
                "rootless_overlay requires sandbox_k8s direct CRD mode; template_name is unsupported"
            )
        if rootless_overlay and pod_spec is not None:
            raise SandboxConfigError("rootless_overlay is incompatible with full pod_spec overrides")
        if rootless_overlay and kwargs.get("container_command") is not None:
            raise SandboxConfigError("rootless_overlay owns container_command; remove the profile override")
        if rootless_overlay:
            kwargs["container_command"] = list(_ROOTLESS_CONTAINER_COMMAND)
        self._rootless_overlay = rootless_overlay
        self._rootless_active = False
        self._rootless_start_timeout = rootless_start_timeout

        gpu_count = _normalize_gpu_count(getattr(self.task_env_config, "gpus", None))
        gpu_types = getattr(self.task_env_config, "gpu_types", None)
        if gpu_count > 0 and gpu_types:
            raise SandboxConfigError(
                "sandbox_k8s does not yet support task environment gpu_types; "
                "omit gpu_types to accept any available GPU"
            )
        if gpu_count > 0 and template_name:
            raise SandboxConfigError(
                "GPU task requests require sandbox_k8s direct CRD mode; configure "
                "GPU resources on the SandboxTemplate before using claim mode"
            )
        if gpu_count > 0:
            kwargs["resources"] = _gpu_resources(kwargs.get("resources"), gpu_count)
            kwargs["tolerations"] = _gpu_tolerations(kwargs.get("tolerations"), gpu_count)

        # Never trust a process/env-file value. Only the private config marker
        # that scaled-evals overwrites after profile merging may unlock root.
        os.environ.pop("SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED", None)
        os.environ.pop("SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS", None)
        os.environ.pop("SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK", None)
        if rootless_overlay:
            os.environ["SANDBOX_K8S_INTERNAL_ROOTLESS_LOW_PORTS"] = "true"
        # Tasks that bind [::1] -- GKE disables IPv6 in every pod sandbox on an
        # IPv4-only cluster, so the sandbox has no usable loopback without this.
        if enable_ipv6_loopback:
            os.environ["SANDBOX_K8S_INTERNAL_IPV6_LOOPBACK"] = "true"
        requested_uid = kwargs.get("run_as_user")
        if requested_uid == 0:
            if _scaled_evals_root_authorized is not True:
                raise SandboxConfigError(
                    "run_as_user=0 requires operator authorization for the exact task image digest"
                )
            # The scaled-evals runner is one process per evaluation. The
            # compatibility patch in sandbox-k8s reads this private marker;
            # task/profile environment variables are never copied here.
            os.environ["SANDBOX_K8S_INTERNAL_ROOT_AUTHORIZED"] = "true"
        _validate_writable_root_request(
            read_only_root_filesystem=read_only_root_filesystem,
            requested_uid=requested_uid,
            root_authorized=_scaled_evals_root_authorized is True,
            writable_root_authorized=_scaled_evals_writable_root_authorized is True,
            template_name=template_name,
            pod_spec=pod_spec,
        )

        # Build sandbox kwargs
        sandbox_kwargs: dict[str, Any] = {
            "name": _sanitize_k8s_name(f"hbr-{session_id[:50]}"),
            "namespace": namespace or "default",
            "working_dir": working_dir,
            "timeout": timeout,
            "lifecycle_timeout": lifecycle_timeout,
            "kubeconfig_path": kubeconfig_path,
            "context": context,
            "in_cluster": in_cluster,
            "verify_ssl": verify_ssl,
            "skip_network_policy_check": skip_network_policy_check,
            "default_container": default_container,
            "read_only_root_filesystem": read_only_root_filesystem,
            "_scaled_evals_writable_root_authorized": (_scaled_evals_writable_root_authorized is True),
            "annotations": _sandbox_annotations(kwargs.pop("annotations", None)),
        }
        if pod_spec is not None:
            sandbox_kwargs["pod_spec"] = pod_spec
        # Harbor reserves ``network_policy`` for its provider-neutral
        # NetworkPolicy model and passes it explicitly from the task baseline.
        # That overwrites a raw Kubernetes policy placed under the same key in
        # environment.kwargs.  Keep a deliberately Kubernetes-specific escape
        # hatch for open-book runs that need concrete DNS/port rules.
        # Other runs omit this and rely on namespace default-deny.
        if k8s_network_policy is not None:
            sandbox_kwargs["network_policy"] = k8s_network_policy
        if template_name:
            sandbox_kwargs["template_name"] = template_name
        effective_image = _effective_sandbox_image(
            session_id=session_id,
            task_env_config=task_env_config,
            profile_image=image,
        )
        if effective_image:
            sandbox_kwargs["image"] = effective_image
        if self._persistent_env:
            sandbox_kwargs["env"] = self._persistent_env
        if "path_validator" not in kwargs:
            sandbox_kwargs["path_validator"] = _harbor_path_validator()
        if not template_name:
            volumes = [_coerce_volume(value) for value in (kwargs.pop("volumes", []) or [])]
            volume_mounts = [_coerce_volume_mount(value) for value in (kwargs.pop("volume_mounts", []) or [])]
            _add_harbor_crd_mounts(volumes, volume_mounts)
            if rootless_overlay:
                _add_rootless_crd_mounts(volumes, volume_mounts)
            sandbox_kwargs["volumes"] = volumes
            sandbox_kwargs["volume_mounts"] = volume_mounts

        # Forward any remaining K8sSandbox-compatible kwargs
        _K8S_SANDBOX_PARAMS = {
            "mode",
            "labels",
            "annotations",
            "dependencies",
            "volumes",
            "volume_mounts",
            "resources",
            "runtime_class_name",
            "tolerations",
            "node_selector",
            "service_account_name",
            "run_as_user",
            "run_as_group",
            "fs_group",
            "checkpoint_config",
            "container_command",
            "path_validator",
            "expected_image_digest",
            "image_pull_policy",
            "image_pull_secrets",
            "pod_spec",
            "default_container",
            "network_policy",
            "sidecars",
        }
        for k, v in kwargs.items():
            if k in _K8S_SANDBOX_PARAMS:
                sandbox_kwargs[k] = v
        if "image_pull_secrets" in sandbox_kwargs:
            normalized_image_pull_secrets = _normalize_image_pull_secrets(sandbox_kwargs["image_pull_secrets"])
            if normalized_image_pull_secrets is None:
                sandbox_kwargs.pop("image_pull_secrets")
            elif template_name:
                raise SandboxConfigError(
                    "image_pull_secrets is only supported in direct CRD mode; "
                    "configure image pull secrets on the SandboxTemplate for claim mode"
                )
            else:
                sandbox_kwargs["image_pull_secrets"] = normalized_image_pull_secrets

        self._sandbox = K8sSandbox(**sandbox_kwargs)
        self._started = False

    # -- Harbor abstract: type() --------------------------------------------

    @staticmethod
    def type() -> Any:
        """Return the environment type.

        **Workaround**: Harbor's ``EnvironmentType`` enum does not include a
        ``KUBERNETES`` variant.  We return ``EnvironmentType.DOCKER`` as the
        closest available stand-in.  When used via ``import_path`` in the
        Harbor factory, ``type()`` is not consulted for dispatch, so this
        value is cosmetic only.

        Proposed upstream fix: add ``KUBERNETES = "kubernetes"`` to
        ``harbor.models.environment_type.EnvironmentType``.
        """
        _ensure_harbor()
        return _EnvironmentType.DOCKER

    # -- Harbor abstract: properties ----------------------------------------

    @property
    def is_mounted(self) -> bool:
        """K8s sandbox does not mount host logging directories."""
        return False

    @property
    def supports_gpus(self) -> bool:
        """Direct CRD mode maps explicit task GPU counts to K8s resources."""
        return True

    @property
    def can_disable_internet(self) -> bool:
        """K8s supports NetworkPolicy-based internet isolation."""
        return True

    @property
    def capabilities(self) -> Any:
        """Harbor environment capabilities supported by the K8s adapter."""
        from harbor.environments.capabilities import EnvironmentCapabilities

        return EnvironmentCapabilities(
            gpus=self.supports_gpus,
            disable_internet=self.can_disable_internet,
            mounted=self.is_mounted,
            # Derived from the profile rather than fixed, because this runtime has
            # no compose builder: it can only serve a compose task when the
            # operator has supplied that task's services as sidecars. Declaring
            # the capability unconditionally would make Harbor accept a compose
            # task whose services nothing ever schedules, trading its clear
            # up-front rejection for an obscure failure mid-trial.
            docker_compose=bool(self._sidecar_containers),
        )

    @property
    def task_os(self) -> Any:
        """Target OS from Harbor's task environment config."""
        from harbor.models.task.config import TaskOS

        return getattr(self.task_env_config, "os", TaskOS.LINUX)

    @property
    def env_paths(self) -> Any:
        """Container-side Harbor paths used by Harbor trial cleanup."""
        from harbor.models.trial.paths import EnvironmentPaths

        return EnvironmentPaths.for_os(self.task_os)

    # -- Harbor abstract: validation ----------------------------------------

    def _validate_definition(self) -> None:
        """No-op: K8s sandboxes are defined by CRD/Claim, not local files."""

    # -- Harbor abstract: lifecycle -----------------------------------------

    async def start(self, force_build: bool = False) -> None:
        """Start the K8s sandbox.

        Args:
            force_build: Ignored (no image build step in K8s sandbox; images
                are pre-built and pulled from a registry).
        """
        if self._started:
            self.logger.warning("Environment already started")
            return

        self.logger.info("Starting K8s sandbox environment: %s", self.session_id)

        # Create trial paths directories on the host for log collection
        if self.trial_paths is not None:
            try:
                self.trial_paths.mkdir()
            except Exception as exc:
                self.logger.warning("Could not create trial directories: %s", exc)

        try:
            await self._sandbox.start()
        except Exception as exc:
            if not _is_sandbox_creation_conflict(exc):
                raise
            self.logger.warning(
                "K8s sandbox %s already exists; deleting stale resource before retry",
                self._sandbox.name,
            )
            await _delete_stale_sandbox(self._sandbox)
            await self._sandbox.start()
        self._started = True
        self.logger.info("K8s sandbox environment started: %s", self._sandbox.name)

        await self._prepare_harbor_dirs()
        self._live_log_task = asyncio.create_task(
            self._stream_sandbox_logs(),
            name=f"sandbox-logs-{self._sandbox.name}",
        )
        await self._upload_environment_public_files()
        await self._upload_environment_skills()
        await self._wait_for_sidecar_ports()
        await self._run_setup_command()
        await self._activate_rootless_overlay()

    async def _activate_rootless_overlay(self) -> None:
        """Start and probe the persistent PRoot command supervisor."""
        if not self._rootless_overlay:
            return
        signal_result = await self._sandbox.run_command(
            [
                "/bin/sh",
                "-c",
                f"mkdir -p {_ROOTLESS_SIGNAL_DIR} && touch {_ROOTLESS_SIGNAL_DIR}/start",
            ],
            timeout=10,
            workdir=self._working_dir,
        )
        if signal_result.exit_code != 0:
            raise RuntimeError(f"could not signal rootless supervisor: {signal_result.stderr or signal_result.stdout}")

        deadline = time.monotonic() + self._rootless_start_timeout
        last_detail = "rootless supervisor probe did not complete"
        while time.monotonic() < deadline:
            probe = await self._sandbox.run_command(
                [
                    "/installed-tools/bin/rootless-client",
                    "--probe",
                    "--timeout",
                    "5",
                ],
                timeout=10,
                workdir=self._working_dir,
            )
            if probe.exit_code == 0:
                self._rootless_active = True
                self.logger.info("Persistent rootless tools overlay is ready")
                return
            last_detail = probe.stderr or probe.stdout or f"exit {probe.exit_code}"
            await asyncio.sleep(2)
        raise RuntimeError(
            f"rootless tools overlay did not become ready within {self._rootless_start_timeout}s: {last_detail}"
        )

    async def _upload_environment_public_files(self) -> None:
        """Overlay task-declared public runtime files into the working directory.

        Prebuilt environments should contain runtime dependencies, not one image
        layer per benchmark case. Tasks may place agent-visible files under
        ``environment/public/``; Harbor uploads those files after the sandbox
        starts and before setup or agent execution. Verifier-only ``tests/``
        remains outside this path and retains Harbor's normal isolation.
        """
        candidates = [
            self.environment_dir / "public",
            self.environment_dir / "environment" / "public",
        ]
        public_src = next((path for path in candidates if path.is_dir()), None)
        if public_src is None:
            self.logger.debug(
                "No public/ dir found near %s — skipping runtime overlay",
                self.environment_dir,
            )
            return
        if not any(path.is_file() for path in public_src.rglob("*")):
            self.logger.debug("environment/public/ is empty — skipping runtime overlay")
            return
        self.logger.info("Uploading public task files to sandbox %s/", self._working_dir)
        await self.upload_dir(public_src, self._working_dir)

    async def _run_setup_command(self) -> None:
        """Run the optional per-task setup command in the sandbox before the agent.

        Lets a task seed its writable working_dir from a read-only baked copy
        (e.g. craft-ipcamera's seed-repo.sh populating /repo from /opt/task-repo)
        so non-oracle agents start with a populated working tree. The image
        ENTRYPOINT does not run here (the CR sets command=sleep), and only
        the oracle runs solve.sh, so this is the general seeding hook. No-op when
        ``setup_command`` is unset.
        """
        if not self._setup_command:
            return
        self.logger.info("Running sandbox setup_command before agent")
        result = await self._sandbox.run_command(
            ["bash", "-c", self._setup_command],
            timeout=600,
            workdir=self._working_dir,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"sandbox setup_command failed (exit {result.exit_code}): {result.stderr or result.stdout}"
            )

    async def _wait_for_sidecar_ports(self) -> None:
        """Wait until sidecar ports are reachable without one long Kubernetes exec.

        A sidecar may legitimately need several minutes to load a large dataset.
        Keeping one exec/WebSocket open for that entire interval is brittle on
        hosted clusters, so use bounded probes and sleep in the runner process.
        """
        if not self._sidecar_wait_ports:
            return
        payload = json.dumps(self._sidecar_wait_ports)
        script = f"""
python_bin="$(command -v python3 || command -v python || true)"
if [ -z "$python_bin" ]; then
  echo "python is required to wait for sidecar ports" >&2
  exit 1
fi
"$python_bin" - <<'PY'
import json
import socket
import sys

ports = json.loads({payload!r})
for host, port in ports:
    try:
        with socket.create_connection((str(host), int(port)), timeout=2):
            pass
    except OSError as exc:
        print(f"{{host}}:{{port}}: {{exc}}", file=sys.stderr)
        sys.exit(1)
print("Sidecars ready: " + ", ".join(f"{{h}}:{{p}}" for h, p in ports))
PY
"""
        deadline = time.monotonic() + self._sidecar_wait_timeout
        last_detail = "no probe completed"
        while time.monotonic() < deadline:
            result = await self._sandbox.run_command(
                ["bash", "-c", textwrap.dedent(script)],
                timeout=10,
                workdir=self._working_dir,
            )
            if result.exit_code == 0:
                return
            last_detail = result.stderr or result.stdout or f"exit {result.exit_code}"
            await asyncio.sleep(2)
        ports = ", ".join(f"{host}:{port}" for host, port in self._sidecar_wait_ports)
        raise RuntimeError(
            f"K8s sandbox sidecars did not become ready within {self._sidecar_wait_timeout}s ({ports}): {last_detail}"
        )

    async def _upload_environment_skills(self) -> None:
        """Upload environment_dir/skills/ into the pod's /workspace/skills/.

        In Docker mode, Harbor's Dockerfile has ``COPY skills/ /workspace/skills/``
        which bakes skills into the image at build time.  In K8s sandbox mode
        we use pre-built images, so skills must be uploaded at runtime.

        Also copies skills into common agent discovery paths (~/.claude/skills,
        ~/.config/opencode/skills, etc.) so installed agents can find them.
        """
        # Harbor passes environment_dir pointing at either the task dir or the
        # task/environment/ subdir.  Check both locations to be robust.
        candidates = [
            self.environment_dir / "skills",  # if env_dir is task/environment/
            self.environment_dir / "environment" / "skills",  # if env_dir is task/
        ]
        skills_src = next((p for p in candidates if p.is_dir()), None)
        if skills_src is None:
            self.logger.debug(
                "No skills/ dir found near %s — skipping skill upload",
                self.environment_dir,
            )
            return

        skill_dirs = [d for d in skills_src.iterdir() if d.is_dir()]
        if not skill_dirs:
            self.logger.debug("environment/skills/ is empty — skipping")
            return

        # Primary upload target: /tmp/agent-home/.claude/skills is pre-created
        # by _prepare_harbor_dirs and is writable (tmpfs).  The root FS may be
        # read-only (e.g. OpenShift), so /workspace/skills and /skills cannot
        # be used as mkdir targets.
        agent_home = PurePosixPath(_DEFAULT_HARBOR_AGENT_DIRS[0]).parents[1]
        primary_target = str(agent_home / ".claude/skills")
        self.logger.info("Uploading %d skill(s) to pod %s/", len(skill_dirs), primary_target)
        try:
            await self._sandbox.run_command(["bash", "-c", f"mkdir -p {_shell_quote(primary_target)}"])
            await self.upload_dir(skills_src, primary_target)

            # Mirror into remaining writable agent skill-discovery paths (best-effort).
            # Only /tmp/ paths are safe on a read-only root filesystem.
            mirror_targets = (
                str(agent_home / ".config/opencode/skills"),
                str(agent_home / "sessions/skills"),
            )
            quoted_targets = " ".join(_shell_quote(path) for path in mirror_targets)
            mirror_cmd = (
                f"for d in {quoted_targets}; do "
                f'  mkdir -p "$d" 2>/dev/null && '
                f'cp -r {_shell_quote(primary_target)}/. "$d/" 2>/dev/null || true; '
                "done; true"
            )
            await self._sandbox.run_command(["bash", "-c", mirror_cmd])
            self.logger.debug("Skills uploaded to %s and mirrored to agent discovery paths", primary_target)
        except Exception as exc:
            self.logger.warning("Could not upload skills to pod: %s", exc)

    async def _prepare_harbor_dirs(self) -> None:
        """Pre-create directories Harbor agents expect to be writable.

        Harbor expects /logs, /tests, /solution, /installed-agent to exist.
        On K8s these may not be in the image or may be read-only.  Create
        them under /tmp and symlink from the expected paths.  This approach
        works even on OpenShift where the root filesystem is read-only but
        /tmp and $HOME are writable.
        """
        setup_script = r"""
mkdir -p /tmp/harbor-dirs/logs/agent/sessions /tmp/harbor-dirs/logs/agent/setup \
  /tmp/harbor-dirs/logs/verifier /tmp/harbor-dirs/logs/artifacts \
  /tmp/harbor-dirs/tests \
  /tmp/harbor-dirs/solution /tmp/harbor-dirs/installed-agent

for d in /logs /tests /solution /installed-agent; do
  if [ ! -e "$d" ]; then
    ln -sf /tmp/harbor-dirs"$d" "$d" 2>/dev/null || true
  elif [ -d "$d" ] && [ "$d" = "/logs" ]; then
    mkdir -p "$d/agent/sessions" "$d/agent/setup" "$d/verifier" "$d/artifacts" 2>/dev/null || true
  fi
done
"""
        if self._harbor_agent_dirs:
            quoted_dirs = " ".join(_shell_quote(path) for path in self._harbor_agent_dirs)
            setup_script += f"\nmkdir -p {quoted_dirs} 2>/dev/null\n"
        setup_script += "true\n"
        try:
            await self._sandbox.run_command(["bash", "-c", setup_script])
            if _PIP_ZIPAPP.is_file():
                await self._sandbox.upload_file(_PIP_ZIPAPP.read_bytes(), "/tmp/pip.pyz")
            self.logger.debug("Pre-created Harbor directories in sandbox")
        except Exception as exc:
            self.logger.warning("Could not pre-create Harbor dirs: %s", exc)

    async def stop(self, delete: bool = True) -> None:
        """Stop the K8s sandbox.

        Args:
            delete: When True (default) the sandbox resources are cleaned up.
                When False the pod is left running for debugging.
        """
        if not self._started:
            return

        self.logger.info("Stopping K8s sandbox environment: %s", self._sandbox.name)
        await self._stop_sandbox_log_stream()
        try:
            await self._collect_pod_artifacts()
        except Exception as exc:
            self.logger.warning("Could not collect K8s pod artifacts before cleanup: %s", exc)
        try:
            await self._collect_sidecar_logs()
        except Exception as exc:
            self.logger.warning("Could not collect sidecar logs before cleanup: %s", exc)
        if delete:
            try:
                await self._sandbox.stop()
            except Exception as exc:
                # Cleanup is retried by the evaluation runtime using the
                # evaluation label. Do not overwrite a completed, scored trial
                # when an admission webhook or API call transiently fails here.
                self.logger.warning("K8s sandbox cleanup deferred to evaluation runtime: %s", exc)
        self._started = False

    async def _stream_sandbox_logs(self) -> None:
        """Forward newly appended in-pod Harbor logs to the live runner log."""
        while self._started:
            try:
                await self._poll_sandbox_logs_once()
                self._live_log_warned = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._live_log_warned:
                    self.logger.warning("Could not stream sandbox logs: %s", exc)
                    self._live_log_warned = True
                else:
                    self.logger.debug("Could not stream sandbox logs: %s", exc)
            await asyncio.sleep(_SANDBOX_LOG_POLL_INTERVAL_SECONDS)

    async def _stop_sandbox_log_stream(self) -> None:
        task = self._live_log_task
        self._live_log_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        try:
            await self._poll_sandbox_logs_once()
        except Exception as exc:
            self.logger.warning("Could not collect final live sandbox logs: %s", exc)
        for path, buffered in self._live_log_buffers.items():
            if buffered:
                self._emit_sandbox_log_line(path, buffered)
        self._live_log_buffers.clear()

    async def _poll_sandbox_logs_once(self) -> None:
        encoded_offsets = base64.urlsafe_b64encode(
            json.dumps(self._live_log_offsets, separators=(",", ":")).encode()
        ).decode()
        result = await self._sandbox.run_command(
            [
                "python3",
                "-c",
                _SANDBOX_LOG_POLL_SCRIPT,
                encoded_offsets,
                str(_SANDBOX_LOG_MAX_BYTES_PER_POLL),
                str(_SANDBOX_LOG_MAX_BYTES_PER_FILE),
            ],
            timeout=_SANDBOX_LOG_POLL_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            detail = result.stderr or result.stdout or f"exit {result.exit_code}"
            raise RuntimeError(f"sandbox log poll failed: {detail}")
        for raw_record in (result.stdout or "").splitlines():
            try:
                record = json.loads(raw_record)
                path = str(record["path"])
                offset = int(record["offset"])
                next_offset = int(record["next_offset"])
                data = base64.b64decode(record["data"])
            except (KeyError, TypeError, ValueError, binascii.Error, json.JSONDecodeError) as exc:
                raise RuntimeError("sandbox log poll returned an invalid record") from exc
            if record.get("reset") or offset != self._live_log_offsets.get(path, 0):
                self._live_log_buffers.pop(path, None)
            self._live_log_offsets[path] = next_offset
            if b"\x00" in data:
                continue
            self._emit_sandbox_log_text(path, data.decode("utf-8", errors="replace"))

    def _emit_sandbox_log_text(self, path: str, text: str) -> None:
        combined = self._live_log_buffers.pop(path, "") + text
        lines = combined.splitlines(keepends=True)
        for line in lines:
            if line.endswith(("\n", "\r")):
                self._emit_sandbox_log_line(path, line.rstrip("\r\n"))
            else:
                while len(line) > _SANDBOX_LOG_MAX_LINE_CHARS:
                    self.logger.info(
                        "[sandbox %s] %s... [continued]",
                        path,
                        line[:_SANDBOX_LOG_MAX_LINE_CHARS],
                    )
                    line = line[_SANDBOX_LOG_MAX_LINE_CHARS:]
                self._live_log_buffers[path] = line

    def _emit_sandbox_log_line(self, path: str, line: str) -> None:
        if len(line) > _SANDBOX_LOG_MAX_LINE_CHARS:
            line = f"{line[:_SANDBOX_LOG_MAX_LINE_CHARS]}... [truncated]"
        self.logger.info("[sandbox %s] %s", path, line)

    async def _collect_pod_artifacts(self) -> None:
        """Collect Kubernetes pod evidence into Harbor artifacts before cleanup."""
        if self.trial_paths is None:
            return
        artifacts_dir = getattr(self.trial_paths, "artifacts_dir", None)
        if artifacts_dir is None:
            return
        k8s_dir = Path(artifacts_dir) / "k8s"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        pod_name = getattr(self._sandbox, "pod_name", None)
        if pod_name:
            (k8s_dir / "pod-name.txt").write_text(str(pod_name) + "\n", encoding="utf-8")
        for filename, getter in (
            ("pod-status.json", getattr(self._sandbox, "get_pod_status", None)),
            ("pod-manifest.json", getattr(self._sandbox, "get_pod_manifest", None)),
        ):
            if getter is None:
                continue
            try:
                payload = await getter()
            except Exception as exc:
                payload = {"error": f"Could not collect {filename}: {exc}"}
            try:
                text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
            except Exception as exc:
                text = (
                    json.dumps(
                        {"error": f"Could not serialize {filename}: {exc}"},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            (k8s_dir / filename).write_text(text, encoding="utf-8")

    async def _collect_sidecar_logs(self) -> None:
        """Collect configured sidecar container logs into Harbor artifacts."""
        if not self._sidecar_log_containers or self.trial_paths is None:
            return
        artifacts_dir = getattr(self.trial_paths, "artifacts_dir", None)
        if artifacts_dir is None:
            return
        log_dir = Path(artifacts_dir) / "sidecar-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for container in self._sidecar_log_containers:
            try:
                logs = await self._sandbox.get_logs(container=container)
            except Exception as exc:
                logs = f"Could not collect logs for sidecar {container}: {exc}\n"
            (log_dir / f"{container}.log").write_text(logs or "", encoding="utf-8")

    # -- Harbor abstract: exec ----------------------------------------------

    async def _run_sandbox_command(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
    ) -> CommandResult:
        """Route commands through the persistent rootless namespace once active."""
        if not self._rootless_active:
            return await self._sandbox.run_command(
                command,
                timeout=timeout,
                workdir=workdir,
            )
        if len(command) != 3 or command[:2] != ["bash", "-c"]:
            raise RuntimeError("rootless command routing requires an explicit bash -c command")
        effective_timeout = timeout or self._command_timeout
        return await self._sandbox.run_command(
            [
                "/installed-tools/bin/rootless-client",
                "--cwd",
                workdir or self._working_dir,
                "--timeout",
                str(effective_timeout),
                "--command",
                command[2],
            ],
            timeout=effective_timeout + 20,
            workdir=self._working_dir,
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        """Execute a command in the K8s sandbox.

        Args:
            command: Shell command string.
            cwd: Working directory inside the container.
            env: Per-command environment variables.
            timeout_sec: Command timeout in seconds.
            user: Username/UID to run as (currently ignored; K8s pods run as
                the configured security context user).

        Returns:
            Harbor ``ExecResult`` with stdout, stderr, return_code.
        """
        _ensure_harbor()

        # Merge persistent env with per-exec env
        merged_env = self._merge_env(env)

        # Build the command with env prefix and cwd
        full_cmd = self._build_command(command, cwd=cwd, env=merged_env, user=user)

        # Always exec via bash so that bash builtins (source, etc.) work.
        # K8sSandbox.run_command wraps strings in ["sh", "-c", ...] which
        # breaks agents that rely on bash features.
        bash_cmd = ["bash", "-c", full_cmd]
        effective_timeout = float(timeout_sec) if timeout_sec else self._command_timeout

        result: CommandResult = await self._run_sandbox_command(
            bash_cmd,
            timeout=effective_timeout,
            workdir=cwd or self._working_dir,
        )

        # -- Workaround 1: privilege escalation failures (exec_as_root) --
        # On K8s (especially OpenShift), pods run as non-root and sudo is
        # blocked.  Retry without user switch; some package-manager
        # failures can be treated as non-fatal only after verifying the
        # relevant tool is already present in the sandbox image.
        if result.exit_code != 0 and user is not None:
            self.logger.warning(
                "Command failed with user=%s (exit %d). Retrying without privilege escalation.",
                user,
                result.exit_code,
            )
            retry_cmd = self._build_command(command, cwd=cwd, env=merged_env, user=None)
            retry_bash = ["bash", "-c", retry_cmd]
            result = await self._run_sandbox_command(
                retry_bash,
                timeout=effective_timeout,
                workdir=cwd or self._working_dir,
            )
            if result.exit_code != 0:
                check_cmd = _setup_tool_check_command(command) if _is_setup_command(command) else None
                tool_available = False
                if check_cmd:
                    check = await self._run_sandbox_command(
                        ["bash", "-c", check_cmd],
                        workdir=self._working_dir,
                    )
                    tool_available = check.exit_code == 0

                if tool_available:
                    self.logger.warning(
                        "Install/setup command still failed (exit %d) — "
                        "verified required tool is already present in sandbox image.",
                        result.exit_code,
                    )
                    result = CommandResult(
                        stdout=check.stdout or result.stdout or "",
                        stderr=result.stderr or "",
                        exit_code=0,
                    )
                else:
                    self.logger.warning(
                        "Install/setup command still failed (exit %d), and no matching "
                        "pre-installed tool could be verified. Preserving failure.",
                        result.exit_code,
                    )

        # -- Workaround 2: permission-denied on mkdir/setup (exec_as_agent) --
        # Harbor creates dirs under paths that may not be writable on K8s
        # (e.g. /logs/agent/sessions on OpenShift).  Only treat failures as
        # non-fatal when the command is a simple mkdir for Harbor-managed paths
        # and those paths are already writable due to _prepare_harbor_dirs().
        if result.exit_code != 0 and user is None:
            combined_lower = ((result.stderr or "") + (result.stdout or "")).lower()
            if "permission denied" in combined_lower and "mkdir" in command.lower():
                check_cmd = _harbor_mkdir_check_command(command)
                if check_cmd:
                    check = await self._run_sandbox_command(
                        ["bash", "-c", check_cmd],
                        workdir=self._working_dir,
                    )
                    if check.exit_code == 0:
                        self.logger.warning(
                            "Harbor mkdir failed with permission denied (exit %d), "
                            "but requested paths already exist and are writable.",
                            result.exit_code,
                        )
                        result = CommandResult(stdout=result.stdout or "", stderr="", exit_code=0)

        # -- Workaround 3: network-blocked install commands (exec_as_agent) --
        # Harbor agents unconditionally download tools (e.g. curl claude.ai)
        # even when they're pre-installed.  On K8s pods with restricted
        # egress, these downloads fail.  Treat them as non-fatal only when a
        # concrete tool named by the failed command is already on PATH.
        _network_errors = (
            "could not resolve host",
            "connection refused",
            "could not connect",
            "network is unreachable",
            "name resolution",
            "timed out",
            "ssl",
            "getaddrinfo",
            "failed to connect",
        )
        if result.exit_code != 0 and user is None:
            combined = ((result.stderr or "") + (result.stdout or "")).lower()
            _install_markers = (
                "install.sh",
                "npm install -g",
                "curl -f",
                "curl https://",
                "wget ",
                "/install -f",
                "/install | bash",
                "uv tool install",
            )
            is_install = any(m in command.lower() for m in _install_markers)
            is_network = any(e in combined for e in _network_errors)
            check_cmd = _setup_tool_check_command(command) if is_install and is_network else None
            if check_cmd:
                self.logger.warning(
                    "Install command failed (exit %d) due to network restrictions. "
                    "Checking if the requested tool is already available in the image...",
                    result.exit_code,
                )
                check = await self._run_sandbox_command(
                    ["bash", "-c", check_cmd],
                    workdir=self._working_dir,
                )
                if check.exit_code == 0 and check.stdout and check.stdout.strip():
                    self.logger.warning(
                        "Tool found at %s — treating install failure as non-fatal.",
                        check.stdout.strip(),
                    )
                    result = CommandResult(
                        stdout=check.stdout or "",
                        stderr="",
                        exit_code=0,
                    )

        return _to_exec_result(result)

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge persistent env with per-exec env."""
        if not self._persistent_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        return merged or None

    @staticmethod
    def _build_command(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> str:
        """Build a shell command string with optional env, cwd, and user prefix.

        The ``user`` parameter is intentionally ignored in K8s sandbox mode.
        In Docker, containers typically run as root and Harbor uses
        ``exec_as_root`` / ``exec_as_agent`` to switch users via sudo.
        In K8s (especially OpenShift), pods run as a fixed non-root UID
        enforced by the SecurityContext, and ``sudo`` is blocked by the
        "no new privileges" kernel flag.  Since the pod's user is already
        set by the SandboxTemplate, privilege escalation is neither
        possible nor necessary.
        """
        parts: list[str] = []

        if env:
            for k, v in env.items():
                parts.append(f"export {_validate_env_var_name(k)}={_shell_quote(v)};")

        if cwd:
            parts.append(f"cd {_shell_quote(cwd)} &&")

        parts.append(command)

        return " ".join(parts)

    # -- Harbor abstract: file transfer -------------------------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a local file to the K8s sandbox.

        Args:
            source_path: Local file path on the host.
            target_path: Destination path inside the container.
        """
        source = Path(source_path)
        content = source.read_bytes()
        await self._sandbox.upload_file(content, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a local directory to the K8s sandbox.

        Streams the directory as a single tarball to avoid one K8s exec/upload
        round trip per file.
        """
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for child in source.iterdir():
                tar.add(child, arcname=child.name)
        tar_bytes = tar_buffer.getvalue()

        remote_tar = f"/tmp/sandbox-k8s-upload-{secrets.token_hex(8)}.tar"
        await self._sandbox.upload_file(tar_bytes, remote_tar)

        extract_cmd = (
            f"mkdir -p {_shell_quote(target_dir)} && "
            f"REMOTE_TAR={_shell_quote(remote_tar)} "
            f"TARGET={_shell_quote(target_dir)} "
            "python3 -c 'import os,tarfile; "
            'archive=tarfile.open(os.environ["REMOTE_TAR"], "r"); '
            'data_filter=getattr(tarfile, "data_filter", None); '
            'archive.extractall(os.environ["TARGET"], **({"filter": data_filter} if data_filter else {}))\' '
            "&& status=0 || status=$?\n"
            f"rm -f {_shell_quote(remote_tar)}\n"
            "exit ${status:-0}"
        )
        result = await self._sandbox.run_command(["bash", "-c", extract_cmd])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to extract uploaded directory to {target_dir}: {result.stderr}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the K8s sandbox to the local host.

        Args:
            source_path: Path inside the container.
            target_path: Local destination path.
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = await self._sandbox.download_file(source_path)
        target.write_bytes(content)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the K8s sandbox to the local host.

        Uses ``tar`` inside the container to stream the directory, then
        extracts locally.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        # Base64 keeps the binary tar stream intact across CommandResult.stdout,
        # which is text in the K8s exec layer.
        tar_cmd = (
            f"python3 -c 'import tarfile,io,os,base64,sys; buf=io.BytesIO(); src=sys.argv[1]; "
            'tf=tarfile.open(fileobj=buf, mode="w"); '
            "[tf.add(os.path.join(r,f), arcname=os.path.relpath(os.path.join(r,f), src)) "
            "for r,_,fs in os.walk(src) for f in fs]; tf.close(); "
            'print(base64.b64encode(buf.getvalue()).decode(), end="")\' '
            f"{_shell_quote(source_dir)}"
        )
        result = await self._sandbox.run_command(
            ["bash", "-c", tar_cmd],
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to tar remote directory {source_dir}: {result.stderr}")
        if not result.stdout:
            raise RuntimeError(f"No tar data returned for remote directory {source_dir}")

        # Write tar to temp file and extract
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
            encoded = result.stdout.encode("ascii") if isinstance(result.stdout, str) else result.stdout
            try:
                tmp.write(base64.b64decode(encoded))
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError(f"Failed to decode tar stream from {source_dir}") from exc
            tmp.flush()

            # The tar comes back from the sandbox, so it is untrusted here on the host.
            # 3.12/3.13 still default to the fully-trusted filter, which honours absolute
            # paths, `..` and symlinks; "data" is the default only from 3.14.
            shutil.unpack_archive(tmp.name, target, "tar", filter="data")

    # -- Harbor abstract: compose service operations -------------------------
    #
    # A compose sidecar becomes another container in the same pod, so a service
    # operation is an ordinary sandbox call with ``container=`` set. The base
    # class routes ``service=None``/"main" to the main container itself, so only
    # the sidecar branch needs implementing; ``service_download_dir`` and
    # ``service_download_dir_with_exclusions`` are generic in Harbor and build on
    # these two.

    @staticmethod
    def is_main_service(service: str | None) -> bool:
        """Return whether a compose service refers to the agent container."""
        return service is None or service == "main"

    def _sidecar_container(self, service: str) -> str:
        """Resolve a compose service name to its container name in this pod."""
        container = sanitize_k8s_name(service)

        if container in getattr(self, "_withheld_sidecar_containers", set()):
            raise _service_operations_unsupported(
                f"compose service {service!r} was withheld from the separate verifier pod",
            )
        if container in self._sidecar_containers:
            return container
        declared = ", ".join(sorted(self._sidecar_containers)) or "none"
        raise _service_operations_unsupported(
            f"compose service {service!r} has no sidecar container in this pod "
            f"(declared sidecars: {declared}). Add it to the Harbor profile's "
            "environment.kwargs.sidecars list.",
        )

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        """Execute a command in a compose service (default: the main container)."""
        if self.is_main_service(service):
            return await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        _ensure_harbor()
        container = self._sidecar_container(service)  # ty: ignore[invalid-argument-type]
        # Sidecar execs deliberately do not inherit the main container's workdir,
        # user or persistent env -- those are main-specific. `sh` rather than
        # `bash` because sidecars are arbitrary third-party images and minimal
        # ones (alpine variants) ship no bash; this matches Harbor's own Docker
        # provider. A `user` switch has no equivalent here: the pod's security
        # context owns the uid, so it is ignored rather than silently faked.
        script = command
        if env:
            exports = " ".join(f"{key}={_shell_quote(value)}" for key, value in env.items())
            script = f"export {exports}\n{script}"
        result = await self._sandbox.run_command(
            ["sh", "-c", script],
            timeout=float(timeout_sec) if timeout_sec else self._command_timeout,
            container=container,
            # An explicit workdir, because every layer below coalesces a missing
            # one to the *main* container's working directory -- so a sidecar exec
            # would open with `cd /app` and die in any image that has no /app,
            # which is most of them. A stock postgres or redis image fails every
            # collect hook this way before the command runs. Root always exists,
            # and a caller with a real directory in mind passes cwd.
            workdir=cwd or "/",
        )
        exec_result = _to_exec_result(result)
        # A broken sidecar exec is logged here because nothing else prints it:
        # Harbor reports a failed collect hook as an exit code alone, so a hook
        # that cannot reach its database looks exactly like one that wrote an
        # empty dump, and the run grades whatever the verifier finds and reports
        # a low score instead of an error.
        #
        # Only a command that wrote to stderr is reported. Harbor also probes
        # sidecars with predicates (`test -d` on an artifact path), where a
        # non-zero exit is the answer and not a fault; those are silent, so
        # stderr is what separates a question from a failure. The exit code
        # stays the caller's to interpret either way.
        stderr = getattr(exec_result, "stderr", "") or ""
        if getattr(exec_result, "return_code", 0) != 0 and stderr.strip():
            self.logger.warning(
                "Service exec in %r (container %s) exited %s: %s\n%s",
                service,
                container,
                getattr(exec_result, "return_code", "?"),
                command,
                stderr[-2000:].strip(),
            )
        return exec_result

    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        """Download a file from a compose service (default: the main container)."""
        if self.is_main_service(service):
            await self.download_file(source_path, target_path)
            return
        container = self._sidecar_container(service)  # ty: ignore[invalid-argument-type]
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self._sandbox.download_file(source_path, container=container))

    async def service_download_dir(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        service: str | None = None,
    ) -> None:
        """Download a directory from a compose service (default: the main container).

        Harbor's base class raises for the sidecar case, but its exclusion-aware
        sibling does not: that one is written against ``service_exec`` plus
        ``service_download_file``, both of which work on sidecars here. An empty
        exclude list produces ``tar czf ... -C <src> .``, which is a plain
        directory download, so delegating is the same operation rather than an
        approximation of it.

        Without this, a task that collects a directory from a service -- an
        emptyDir shared between sidecars, say -- fails the whole trial at
        collection time.
        """
        await self.service_download_dir_with_exclusions(
            source_dir=source_dir,
            target_dir=target_dir,
            exclude=[],
            service=service,
        )

    async def service_download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
        service: str | None = None,
    ) -> None:
        """Download a compose-service directory while excluding matching paths."""
        remote_tar = f"/tmp/harbor-transfer-{secrets.token_hex(8)}.tar.gz"
        exclude_flags = " ".join(f"--exclude={_shell_quote(pattern)}" for pattern in exclude)
        try:
            result = await self.service_exec(
                f"tar czf {_shell_quote(remote_tar)} {exclude_flags} -C {_shell_quote(source_dir)} .",
                service=service,
                timeout_sec=120,
                user="root",
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "no output"
                raise RuntimeError(f"Failed to create transfer archive for {source_dir!r}: {output}")

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as host_tar:
                await self.service_download_file(remote_tar, host_tar.name, service=service)
                with tarfile.open(host_tar.name, "r:gz") as archive:
                    archive.extractall(path=target, filter="data")
        finally:
            try:
                await self.service_exec(
                    f"rm -f {_shell_quote(remote_tar)}",
                    service=service,
                    timeout_sec=120,
                    user="root",
                )
            except Exception as exc:
                self.logger.warning("Could not remove transfer archive %s: %s", remote_tar, exc)

    # -- Harbor optional: is_dir / is_file ----------------------------------

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a directory."""
        import shlex as _shlex

        result = await self.exec(
            f"test -d {_shlex.quote(path)}",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a regular file."""
        import shlex as _shlex

        result = await self.exec(
            f"test -f {_shlex.quote(path)}",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    # -- Harbor optional: preflight -----------------------------------------

    @classmethod
    def preflight(cls) -> None:
        """Verify K8s connectivity before queueing trials."""
        try:
            from .client import get_k8s_client

            get_k8s_client()
        except Exception as exc:
            import sys

            print(f"K8s preflight check failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    # -- Convenience accessors -----------------------------------------------

    @property
    def sandbox(self) -> K8sSandbox:
        """Access the underlying K8sSandbox instance."""
        return self._sandbox

    @property
    def pod_name(self) -> str | None:
        """Resolved pod name (available after start)."""
        return self._sandbox.pod_name


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sanitize_k8s_name(name: str, max_len: int = 63) -> str:
    """Sanitize a string into a valid Kubernetes resource name (RFC 1123 DNS label).

    K8s names must be lowercase, alphanumeric or ``-``, start/end with
    alphanumeric, and be at most 63 characters.
    """
    import re

    sanitized = name.lower()
    sanitized = re.sub(r"[^a-z0-9-]", "-", sanitized)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-")
    return sanitized[:max_len].rstrip("-")


def _harbor_path_validator() -> Any:
    """Return a path validator that permits Harbor-managed mount points."""
    from sandbox_k8s.config import get_settings
    from sandbox_k8s.validators import SandboxPathValidator

    settings = get_settings()
    allow_patterns = list(settings.path_allow_patterns)
    allow_patterns.extend(pattern for pattern in _HARBOR_PATH_ALLOW_PATTERNS if pattern not in allow_patterns)
    deny_patterns: list[str] = []
    for pattern in settings.path_deny_patterns:
        deny_patterns.extend(_HARBOR_PATH_DENY_REPLACEMENTS.get(pattern, (pattern,)))
    return SandboxPathValidator(
        max_path_length=settings.max_path_length,
        allow_patterns=allow_patterns,
        deny_patterns=deny_patterns,
    )


def _is_setup_command(command: str) -> bool:
    """Return True if a command looks like Harbor setup/install boilerplate."""
    command_lower = command.lower()
    setup_markers = (
        "apt-get",
        "apk",
        "yum",
        "dnf",
        "pip install",
        "npm install",
        "curl",
        "wget",
        "mkdir -p",
        "install.sh",
        "uv tool install",
    )
    return any(marker in command_lower for marker in setup_markers)


_HARBOR_MKDIR_PREFIXES = (
    "/logs",
    "/tests",
    "/solution",
    "/installed-agent",
    "/tmp/harbor-dirs",
    "/tmp/agent-home",
    "/workspace/skills",
)


def _harbor_mkdir_check_command(command: str) -> str | None:
    """Return a command that verifies failed Harbor mkdir targets are satisfied."""
    import shlex

    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or tokens[0] != "mkdir":
        return None

    targets: list[str] = []
    skip_next = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in {"-m", "--mode", "-Z", "--context"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        targets.append(token)

    if not targets:
        return None
    if not all(any(_is_same_or_child_path(target, prefix) for prefix in _HARBOR_MKDIR_PREFIXES) for target in targets):
        return None

    return " && ".join(f"test -d {_shell_quote(target)} && test -w {_shell_quote(target)}" for target in targets)


def _is_same_or_child_path(path: str, prefix: str) -> bool:
    try:
        candidate = PurePosixPath(path)
    except ValueError:
        return False
    parent = PurePosixPath(prefix)
    return candidate == parent or parent in candidate.parents


def _setup_tool_check_command(command: str) -> str | None:
    """Build a shell check that verifies a failed setup command is already satisfied."""
    command_lower = command.lower()
    checks: list[str] = []
    skip_tools: set[str] = set()
    if "pip install" in command_lower:
        skip_tools.update({"pip", "pip3", "python", "python3"})
    if "npm install" in command_lower:
        skip_tools.update({"npm", "node"})
    if "curl" in command_lower and " install " not in command_lower:
        skip_tools.add("curl")
    if "wget" in command_lower and " install " not in command_lower:
        skip_tools.add("wget")
    known_tools = (
        "claude",
        "cursor-agent",
        "mini-swe-agent",
        "openhands",
        "uv",
        "node",
        "npm",
        "python3",
        "python",
        "pip3",
        "pip",
        "curl",
        "wget",
        "git",
        "bash",
        "mkdir",
    )

    for tool in known_tools:
        if tool not in skip_tools and tool in command_lower:
            checks.append(f"command -v {_shell_quote(tool)}")

    # Package-manager setup frequently installs runtime tools. If no concrete
    # tool is visible, stay conservative and let the original failure surface.
    unique_checks = list(dict.fromkeys(checks))
    if not unique_checks:
        return None
    return " || ".join(f"({check})" for check in unique_checks)


def _shell_quote(s: str) -> str:
    """Shell-quote a string for use in commands."""
    import shlex

    return shlex.quote(s)


def _validate_env_var_name(name: str) -> str:
    """Return a safe POSIX shell env-var name or raise ``ValueError``."""
    if (
        not name
        or not name.isascii()
        or not (name[0].isalpha() or name[0] == "_")
        or any(not (ch.isalnum() or ch == "_") for ch in name)
    ):
        raise ValueError(f"Invalid environment variable name: {name!r}")
    return name
