# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile the persistent ClickHouse container owned by local Intake.

Lifecycle contract:
- Intake startup reconciles one container owned by the resolved data directory.
- Managed lifecycle assumes one active Intake platform runner per data directory.
- Matching running containers are reused; matching stopped containers are started.
- Containers for an earlier incarnation of the data directory are replaced.
- Unsafe image, credential, mount, or ownership mismatches fail closed.
- Graceful Intake shutdown stops the container; the next startup restarts it.
- Hard process termination can leave the detached container running.
- Explicit removal stops/removes the container but never deletes its data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final, cast
from uuid import UUID, uuid4

import clickhouse_connect
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from nhx.common.config import nhx_user_data_dir
from nhx.intake.config import DEFAULT_CLICKHOUSE_IMAGE, DEFAULT_CLICKHOUSE_VERSION, IntakeConfig
from nhx.intake.spans.clickhouse_client import ClickHouseSettings
from nhx.intake.spans.clickhouse_migrations import parse_clickhouse_url
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

import docker
from docker import DockerClient

logger = logging.getLogger(__name__)

CLICKHOUSE_VERSION: Final = DEFAULT_CLICKHOUSE_VERSION
CLICKHOUSE_HTTP_PORT: Final = 8123
CLICKHOUSE_HTTP_PORT_KEY: Final = f"{CLICKHOUSE_HTTP_PORT}/tcp"
CLICKHOUSE_NATIVE_PORT: Final = 9000
CLICKHOUSE_NATIVE_PORT_KEY: Final = f"{CLICKHOUSE_NATIVE_PORT}/tcp"
CLICKHOUSE_DATA_PATH: Final = "/var/lib/clickhouse"
LEGACY_CONTAINER_NAME: Final = "nhx-intake-clickhouse"

_MANAGED_BY_LABEL = "nhx.nvidia.com/managed-by"
_COMPONENT_LABEL = "nhx.nvidia.com/component"
_DATA_DIR_LABEL = "nhx.nvidia.com/data-directory-sha256"
_DATA_INSTANCE_LABEL = "nhx.nvidia.com/data-instance-id"
_MANAGED_BY_VALUE = "nemo-helix"
_COMPONENT_VALUE = "intake-clickhouse"
_DATA_IDENTITY_FILE = ".nhx-clickhouse-identity"
_DATA_INSTANCE_FILE_PREFIX = f"{_DATA_IDENTITY_FILE}-"
_READINESS_TIMEOUT_SECONDS = 60.0
_READINESS_POLL_SECONDS = 0.5
_CLICKHOUSE_HTTP_LOGGER_NAME = "clickhouse_connect.driver.httpclient"
_TRANSIENT_HTTP_WARNING = "Unexpected Http Driver Exception"
_DOCKER_UNAVAILABLE_GUIDANCE = (
    "Docker daemon is unavailable. Start Docker Desktop on macOS/Windows or the Docker service on Linux, "
    "then rerun `nemo setup` or restart `nemo services run`. To use an externally managed ClickHouse instead, "
    "set NHX_INTAKE_CLICKHOUSE_URL."
)


class LocalClickHouseProvisioningError(RuntimeError):
    """Raised when Intake cannot safely reconcile or remove its local ClickHouse."""


class DockerUnavailableError(LocalClickHouseProvisioningError):
    """Raised when local ClickHouse cannot start because Docker is unavailable."""


class _TransientClickHouseStartupFilter(logging.Filter):
    """Hide the driver's expected connection warning while readiness is polling."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _TRANSIENT_HTTP_WARNING


class ProvisioningMode(Enum):
    """Container layout selected by the normal or compatibility entry point."""

    MANAGED = auto()
    LEGACY_COMPATIBILITY = auto()


class ContainerKind(Enum):
    """Ownership model of a discovered container candidate."""

    MANAGED = auto()
    LEGACY_COMPATIBILITY = auto()


class ContainerDisposition(Enum):
    """Safe reconciliation outcome for an existing container."""

    REUSABLE = auto()
    STALE_DATA_INSTANCE = auto()


@dataclass(frozen=True)
class LocalClickHouseDesiredState:
    """Container configuration and data identity Intake wants to be running."""

    settings: ClickHouseSettings
    image: str
    data_dir: Path
    data_instance_id: str
    mode: ProvisioningMode

    @property
    def container_name(self) -> str:
        if self.mode is ProvisioningMode.LEGACY_COMPATIBILITY:
            return LEGACY_CONTAINER_NAME
        return _managed_container_name(self.data_dir)

    @property
    def container_kind(self) -> ContainerKind:
        if self.mode is ProvisioningMode.LEGACY_COMPATIBILITY:
            return ContainerKind.LEGACY_COMPATIBILITY
        return ContainerKind.MANAGED


@dataclass(frozen=True)
class ContainerCandidate:
    """An existing container and the compatibility rules that apply to it."""

    container: Container
    kind: ContainerKind


async def reconcile_local_clickhouse(
    settings: ClickHouseSettings,
    *,
    image: str = DEFAULT_CLICKHOUSE_IMAGE,
    data_dir: Path | None = None,
) -> str:
    """Reconcile Intake's data-directory-owned ClickHouse and return its HTTP URL."""

    return await asyncio.to_thread(
        _reconcile_local_clickhouse,
        settings,
        image=image,
        data_dir=data_dir,
    )


async def stop_local_clickhouse(*, data_dir: Path | None = None) -> bool:
    """Stop Intake's data-directory-owned ClickHouse without removing it or its data."""

    return await asyncio.to_thread(_stop_local_clickhouse, data_dir=data_dir)


async def check_local_clickhouse_data_directory(*, data_dir: Path | None = None) -> None:
    """Verify that the managed container can write its ClickHouse data directory."""

    await asyncio.to_thread(_check_local_clickhouse_data_directory, data_dir=data_dir)


def _reconcile_local_clickhouse(
    settings: ClickHouseSettings,
    *,
    image: str = DEFAULT_CLICKHOUSE_IMAGE,
    data_dir: Path | None = None,
    mode: ProvisioningMode = ProvisioningMode.MANAGED,
) -> str:
    resolved_data_dir = _resolve_data_dir(data_dir)
    client = _connect_docker()
    try:
        data_instance_id = _ensure_data_directory_identity(
            resolved_data_dir,
            manage_permissions=data_dir is None,
        )
        desired = LocalClickHouseDesiredState(
            settings=settings,
            image=image,
            data_dir=resolved_data_dir,
            data_instance_id=data_instance_id,
            mode=mode,
        )
        candidate = _find_container_candidate(client, desired)
        container = _reconcile_container(client, desired, candidate)
        return _prepare_and_wait_until_ready(container, settings)
    except LocalClickHouseProvisioningError:
        raise
    except Exception as exc:
        raise LocalClickHouseProvisioningError(f"Failed to reconcile local ClickHouse: {exc}") from exc
    finally:
        client.close()


def _container_targets(client: DockerClient, data_dir: Path) -> list[tuple[str, Container]]:
    targets: list[tuple[str, Container]] = []
    managed_container_name = _managed_container_name(data_dir)
    if (managed_container := _get_container(client, managed_container_name)) is not None:
        targets.append((managed_container_name, managed_container))
    if (legacy_container := _get_container(client, LEGACY_CONTAINER_NAME)) is not None:
        legacy_container.reload()
        legacy_attrs = cast(dict[str, Any], legacy_container.attrs or {})
        if _mounted_clickhouse_data_dir(legacy_attrs) == data_dir:
            targets.append((LEGACY_CONTAINER_NAME, legacy_container))
    return targets


def _validated_container_targets(
    client: DockerClient,
    data_dir: Path,
    *,
    action: str,
) -> list[tuple[str, Container]]:
    targets = _container_targets(client, data_dir)
    for container_name, container in targets:
        _validate_lifecycle_target(
            container,
            expected_name=container_name,
            data_dir=data_dir,
            action=action,
            allow_unlabeled_legacy=container_name == LEGACY_CONTAINER_NAME,
        )
    return targets


def _stop_local_clickhouse(*, data_dir: Path | None = None) -> bool:
    resolved_data_dir = _resolve_data_dir(data_dir)
    client = _connect_docker()
    try:
        containers = _validated_container_targets(client, resolved_data_dir, action="stop")
        for container_name, container in containers:
            _stop_container(container)
            logger.info("Stopped managed local ClickHouse container %s", container_name)
        return bool(containers)
    except LocalClickHouseProvisioningError:
        raise
    except Exception as exc:
        raise LocalClickHouseProvisioningError(f"Failed to stop local ClickHouse: {exc}") from exc
    finally:
        client.close()


def _check_local_clickhouse_data_directory(*, data_dir: Path | None = None) -> None:
    resolved_data_dir = _resolve_data_dir(data_dir)
    client = _connect_docker()
    try:
        container_name = _managed_container_name(resolved_data_dir)
        container = _get_container(client, container_name)
        if container is None:
            container_name = LEGACY_CONTAINER_NAME
            container = _get_container(client, container_name)
        if container is None:
            raise LocalClickHouseProvisioningError(
                f"Managed ClickHouse container for data directory {resolved_data_dir} does not exist"
            )
        _validate_lifecycle_target(
            container,
            expected_name=container_name,
            data_dir=resolved_data_dir,
            action="check readiness for",
            allow_unlabeled_legacy=container_name == LEGACY_CONTAINER_NAME,
        )
        if container.status != "running":
            raise LocalClickHouseProvisioningError(f"Managed ClickHouse container {container_name} is not running")
        _check_clickhouse_data_directory_access(container)
    except LocalClickHouseProvisioningError:
        raise
    except Exception as exc:
        raise LocalClickHouseProvisioningError(f"Failed to check local ClickHouse data storage: {exc}") from exc
    finally:
        client.close()


def remove_local_clickhouse(
    *,
    data_dir: Path | None = None,
    restore_data_ownership: bool = False,
) -> bool:
    """Remove managed ClickHouse containers for a local NeMo data directory."""

    resolved_data_dir = _resolve_data_dir(data_dir)
    client = _connect_docker()
    try:
        containers = _validated_container_targets(client, resolved_data_dir, action="remove")
        for container_name, container in containers:
            if restore_data_ownership:
                _restore_data_dir_ownership(container)
            _stop_and_remove_container(container)
            logger.info("Removed managed local ClickHouse container %s", container_name)
        return bool(containers)
    except LocalClickHouseProvisioningError:
        raise
    except Exception as exc:
        raise LocalClickHouseProvisioningError(f"Failed to remove local ClickHouse: {exc}") from exc
    finally:
        client.close()


def _connect_docker() -> DockerClient:
    client: DockerClient | None = None
    try:
        client = docker.from_env(timeout=10)
        client.ping()
        return client
    except (DockerException, RequestsConnectionError, RequestsTimeout, OSError) as exc:
        if client is not None:
            client.close()
        raise DockerUnavailableError(_DOCKER_UNAVAILABLE_GUIDANCE) from exc


def _resolve_data_dir(data_dir: Path | None) -> Path:
    return (data_dir.expanduser() if data_dir is not None else nhx_user_data_dir() / "intake-clickhouse").resolve()


def _get_container(client: DockerClient, name: str) -> Container | None:
    try:
        return client.containers.get(name)
    except NotFound:
        return None


def _find_container_candidate(
    client: DockerClient,
    desired: LocalClickHouseDesiredState,
) -> ContainerCandidate | None:
    container = _get_container(client, desired.container_name)
    if container is not None:
        return ContainerCandidate(container=container, kind=desired.container_kind)

    if desired.mode is ProvisioningMode.LEGACY_COMPATIBILITY:
        return None

    legacy_container = _get_container(client, LEGACY_CONTAINER_NAME)
    if legacy_container is None:
        return None
    legacy_container.reload()
    legacy_attrs = cast(dict[str, Any], legacy_container.attrs or {})
    if _mounted_clickhouse_data_dir(legacy_attrs) != desired.data_dir:
        return None
    return ContainerCandidate(container=legacy_container, kind=ContainerKind.LEGACY_COMPATIBILITY)


def _reconcile_container(
    client: DockerClient,
    desired: LocalClickHouseDesiredState,
    candidate: ContainerCandidate | None,
) -> Container:
    if candidate is None:
        return _create_container(client, desired)

    disposition = _classify_existing_container(candidate, desired)
    if disposition is ContainerDisposition.STALE_DATA_INSTANCE:
        logger.warning(
            "Replacing stale managed ClickHouse container %s after its data directory was recreated",
            candidate.container.name,
        )
        _stop_and_remove_container(candidate.container)
        return _create_container(client, desired)

    _disable_automatic_restart(candidate.container)
    candidate.container.reload()
    if candidate.container.status != "running":
        candidate.container.start()
    return candidate.container


def _create_container(
    client: DockerClient,
    desired: LocalClickHouseDesiredState,
) -> Container:
    try:
        client.images.get(desired.image)
    except ImageNotFound:
        logger.warning(
            "Local ClickHouse image %s is not installed; pulling it before Intake starts. "
            "The first pull may take several minutes.",
            desired.image,
        )
        client.images.pull(desired.image)
        logger.info("Finished pulling local ClickHouse image %s", desired.image)

    run_kwargs = {
        "image": desired.image,
        "name": desired.container_name,
        "detach": True,
        "environment": {
            "CLICKHOUSE_USER": desired.settings.user,
            "CLICKHOUSE_PASSWORD": desired.settings.password,
            "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT": "1",
            "CLICKHOUSE_SKIP_USER_SETUP": "1",
        },
        "ports": (
            {
                CLICKHOUSE_HTTP_PORT_KEY: ("127.0.0.1", CLICKHOUSE_HTTP_PORT),
                CLICKHOUSE_NATIVE_PORT_KEY: ("127.0.0.1", CLICKHOUSE_NATIVE_PORT),
            }
            if desired.mode is ProvisioningMode.LEGACY_COMPATIBILITY
            else {CLICKHOUSE_HTTP_PORT_KEY: ("127.0.0.1", 0)}
        ),
        "volumes": {str(desired.data_dir): {"bind": CLICKHOUSE_DATA_PATH, "mode": "rw"}},
        "labels": _expected_labels(desired.data_dir, desired.data_instance_id),
        "restart_policy": {"Name": "no"},
    }
    try:
        return client.containers.run(**run_kwargs)
    except APIError as exc:
        if exc.status_code != 409:
            raise
        container = _get_container(client, desired.container_name)
        if container is None:
            raise
        return _reconcile_container(
            client,
            desired,
            ContainerCandidate(container=container, kind=desired.container_kind),
        )


def _classify_existing_container(
    candidate: ContainerCandidate,
    desired: LocalClickHouseDesiredState,
) -> ContainerDisposition:
    container = candidate.container
    expected_name = container.name
    container.reload()
    attrs = cast(dict[str, Any], container.attrs or {})
    container_config = cast(dict[str, Any], attrs.get("Config") or {})
    actual_image = str(container_config.get("Image", ""))
    if actual_image != desired.image:
        raise LocalClickHouseProvisioningError(
            f"Container {expected_name} uses {actual_image or 'an unknown image'}, but Intake requires {desired.image}. "
            "Preserve any data you need and remove or rename the container before retrying."
        )

    if _mounted_clickhouse_data_dir(attrs) != desired.data_dir:
        raise LocalClickHouseProvisioningError(
            f"Container {expected_name} does not use the expected data directory {desired.data_dir}"
        )

    if candidate.kind is ContainerKind.MANAGED:
        labels = container.labels or {}
        identity_labels = {
            _MANAGED_BY_LABEL: _MANAGED_BY_VALUE,
            _COMPONENT_LABEL: _COMPONENT_VALUE,
            _DATA_DIR_LABEL: _data_dir_identity(desired.data_dir),
        }
        if any(labels.get(key) != value for key, value in identity_labels.items()):
            raise LocalClickHouseProvisioningError(
                f"Container name collision: {expected_name} is not the matching Intake-managed ClickHouse"
            )
        if labels.get(_DATA_INSTANCE_LABEL) != desired.data_instance_id:
            return ContainerDisposition.STALE_DATA_INSTANCE
    environment = _container_environment(attrs)
    missing_credential_keys = [key for key in ("CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD") if key not in environment]
    if missing_credential_keys:
        raise LocalClickHouseProvisioningError(
            f"Container {expected_name} was not provisioned by Intake with explicit ClickHouse credentials "
            f"(missing {', '.join(missing_credential_keys)}). Remove or rename the container before retrying."
        )
    if (
        environment.get("CLICKHOUSE_USER") != desired.settings.user
        or environment.get("CLICKHOUSE_PASSWORD") != desired.settings.password
    ):
        raise LocalClickHouseProvisioningError(
            f"ClickHouse credentials changed for container {expected_name}. Remove the container to re-provision "
            "it with NHX_INTAKE_CLICKHOUSE_USER and NHX_INTAKE_CLICKHOUSE_PASSWORD, or restore the previous values."
        )

    return ContainerDisposition.REUSABLE


def _validate_lifecycle_target(
    container: Container,
    *,
    expected_name: str,
    data_dir: Path,
    action: str,
    allow_unlabeled_legacy: bool = False,
) -> None:
    container.reload()
    attrs = cast(dict[str, Any], container.attrs or {})
    if _mounted_clickhouse_data_dir(attrs) != data_dir:
        raise LocalClickHouseProvisioningError(
            f"Refusing to {action} container {expected_name}: it does not mount {data_dir}"
        )

    labels = container.labels or {}
    expected_labels = {
        _MANAGED_BY_LABEL: _MANAGED_BY_VALUE,
        _COMPONENT_LABEL: _COMPONENT_VALUE,
        _DATA_DIR_LABEL: _data_dir_identity(data_dir),
    }
    if allow_unlabeled_legacy and not any(label in labels for label in expected_labels):
        return
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise LocalClickHouseProvisioningError(
            f"Refusing to {action} container {expected_name}: it is not owned by Intake for {data_dir}"
        )


def _stop_container(container: Container) -> None:
    container.reload()
    if container.status == "running":
        container.stop(timeout=30)


def _disable_automatic_restart(container: Container) -> None:
    container.reload()
    attrs = cast(dict[str, Any], container.attrs or {})
    host_config = cast(dict[str, Any], attrs.get("HostConfig") or {})
    restart_policy = cast(dict[str, Any], host_config.get("RestartPolicy") or {})
    if restart_policy.get("Name") != "no":
        container.update(restart_policy={"Name": "no"})


def _stop_and_remove_container(container: Container) -> None:
    _stop_container(container)
    container.remove()


def _restore_data_dir_ownership(container: Container) -> None:
    if sys.platform == "win32":
        return
    container.reload()
    if container.status != "running":
        container.start()
    owner = f"{os.getuid()}:{os.getgid()}"
    result = container.exec_run(
        ["chown", "-R", owner, CLICKHOUSE_DATA_PATH],
        user="root",
    )
    if result.exit_code != 0:
        output = result.output.decode(errors="replace") if isinstance(result.output, bytes) else str(result.output)
        raise LocalClickHouseProvisioningError(
            f"Could not restore host ownership of ClickHouse data in {container.name}: {output.strip()}"
        )


def _data_dir_identity(data_dir: Path) -> str:
    return hashlib.sha256(str(data_dir).encode()).hexdigest()


def _managed_container_name(data_dir: Path) -> str:
    return f"nhx-intake-clickhouse-{_data_dir_identity(data_dir)[:12]}"


def _container_environment(attrs: dict[str, Any]) -> dict[str, str]:
    container_config = cast(dict[str, Any], attrs.get("Config") or {})
    entries = cast(list[str], container_config.get("Env") or [])
    return {key: value for entry in entries if "=" in entry for key, value in [entry.split("=", 1)]}


def _mounted_clickhouse_data_dir(attrs: dict[str, Any]) -> Path | None:
    mounts = cast(list[dict[str, Any]], attrs.get("Mounts") or [])
    source = next(
        (mount.get("Source") for mount in mounts if mount.get("Destination") == CLICKHOUSE_DATA_PATH),
        None,
    )
    return Path(str(source)).resolve() if source is not None else None


def _expected_labels(data_dir: Path, data_instance_id: str) -> dict[str, str]:
    return {
        _MANAGED_BY_LABEL: _MANAGED_BY_VALUE,
        _COMPONENT_LABEL: _COMPONENT_VALUE,
        _DATA_DIR_LABEL: _data_dir_identity(data_dir),
        _DATA_INSTANCE_LABEL: data_instance_id,
    }


def _ensure_data_directory_identity(data_dir: Path, *, manage_permissions: bool) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    if manage_permissions:
        for path in (data_dir, tmp_dir):
            try:
                path.chmod(0o755)
            except PermissionError:
                # A rootless Docker bind mount can leave this path owned by a
                # container-mapped UID after a previous run. The container-side
                # reconciliation repairs that ownership before ClickHouse starts.
                logger.debug("Could not update permissions for existing ClickHouse data path %s", path)

    # Keep the identity inside the bind mount intentionally: it belongs to this
    # data incarnation and must disappear with the ClickHouse data during a wipe.
    # The real-image integration test verifies ClickHouse tolerates this file.
    identity_path = data_dir / _DATA_IDENTITY_FILE
    if not identity_path.exists():
        data_instance_id = uuid4().hex
        instance_path = data_dir / f"{_DATA_INSTANCE_FILE_PREFIX}{data_instance_id}"
        file_descriptor = os.open(instance_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(file_descriptor)
        try:
            os.link(instance_path, identity_path)
        except FileExistsError:
            # Another provisioner won the atomic link race. Its record is the
            # identity; this unmatched candidate is harmless and ignored.
            pass
        else:
            return data_instance_id

    identity_stat = identity_path.stat()
    matching_instance_ids: list[str] = []
    for instance_path in data_dir.glob(f"{_DATA_INSTANCE_FILE_PREFIX}*"):
        try:
            instance_id = UUID(instance_path.name.removeprefix(_DATA_INSTANCE_FILE_PREFIX)).hex
            instance_stat = instance_path.stat()
        except (FileNotFoundError, ValueError):
            continue
        if os.path.samestat(identity_stat, instance_stat):
            matching_instance_ids.append(instance_id)

    if len(matching_instance_ids) == 1:
        return matching_instance_ids[0]

    raise LocalClickHouseProvisioningError(
        f"ClickHouse data identity marker {identity_path} has no unique instance record. "
        "Preserve any data you need and remove the marker before re-provisioning."
    )


def _check_clickhouse_data_directory_access(container: Container) -> None:
    result = container.exec_run(
        [
            "sh",
            "-c",
            "mkdir -p /var/lib/clickhouse/tmp && "
            'probe=$(mktemp /var/lib/clickhouse/tmp/.nhx-write-probe.XXXXXX) && rm -f "$probe"',
        ],
        user="clickhouse",
    )
    if result.exit_code != 0:
        output = result.output.decode(errors="replace") if isinstance(result.output, bytes) else str(result.output)
        raise LocalClickHouseProvisioningError(
            f"ClickHouse data storage in {container.name} is not writable by the clickhouse user: {output.strip()}"
        )


def _prepare_and_wait_until_ready(container: Container, settings: ClickHouseSettings) -> str:
    url = _container_http_url(container)
    _wait_until_ready(replace(settings, url=url))
    _check_clickhouse_data_directory_access(container)
    logger.info("Local ClickHouse is ready at %s", url, extra={"container": container.name})
    return url


def _container_http_url(container: Container) -> str:
    container.reload()
    bindings = (container.ports or {}).get(CLICKHOUSE_HTTP_PORT_KEY)
    if not bindings:
        raise LocalClickHouseProvisioningError(
            f"Container {container.name} does not publish ClickHouse HTTP port {CLICKHOUSE_HTTP_PORT}"
        )
    host_port = bindings[0].get("HostPort")
    if not host_port:
        raise LocalClickHouseProvisioningError(f"Container {container.name} has no assigned ClickHouse host port")
    return f"http://127.0.0.1:{int(host_port)}"


def _wait_until_ready(settings: ClickHouseSettings) -> None:
    deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
    last_error: Exception | None = None
    driver_logger = logging.getLogger(_CLICKHOUSE_HTTP_LOGGER_NAME)
    transient_warning_filter = _TransientClickHouseStartupFilter()
    driver_logger.addFilter(transient_warning_filter)
    try:
        while time.monotonic() < deadline:
            try:
                _ping_clickhouse(settings)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(_READINESS_POLL_SECONDS)
    finally:
        driver_logger.removeFilter(transient_warning_filter)
    raise LocalClickHouseProvisioningError(
        f"ClickHouse at {settings.url} did not become ready within {_READINESS_TIMEOUT_SECONDS:g} seconds"
    ) from last_error


def _ping_clickhouse(settings: ClickHouseSettings) -> None:
    parsed = parse_clickhouse_url(settings.url)
    client = clickhouse_connect.get_client(
        host=parsed.host,
        port=parsed.port,
        secure=parsed.secure,
        username=settings.user,
        password=settings.password,
        database="default",
        connect_timeout=2,
    )
    try:
        client.command("SELECT 1")
    finally:
        client.close()


def main() -> int:
    """Provision or remove local ClickHouse for compatibility and teardown commands."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true", help="remove the data-directory-owned ClickHouse container")
    parser.add_argument("--data-dir", type=Path, help="ClickHouse data directory to provision or clean up")
    parser.add_argument(
        "--legacy-script-mode",
        action="store_true",
        help="use the compatibility container name and fixed localhost ports",
    )
    args = parser.parse_args()
    config = IntakeConfig()
    settings = ClickHouseSettings.from_config(config)
    clickhouse_data_dir = args.data_dir or config.clickhouse_config.data_dir
    try:
        if args.remove:
            restore_data_ownership = _resolve_data_dir(clickhouse_data_dir).is_relative_to(
                nhx_user_data_dir().resolve()
            )
            removed = remove_local_clickhouse(
                data_dir=clickhouse_data_dir,
                restore_data_ownership=restore_data_ownership,
            )
            print(
                "Removed managed local ClickHouse container"
                if removed
                else "No managed local ClickHouse container found"
            )
            return 0
        mode = ProvisioningMode.LEGACY_COMPATIBILITY if args.legacy_script_mode else ProvisioningMode.MANAGED
        url = _reconcile_local_clickhouse(
            settings,
            image=config.clickhouse_config.image,
            data_dir=clickhouse_data_dir,
            mode=mode,
        )
    except LocalClickHouseProvisioningError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Local ClickHouse is ready at {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
