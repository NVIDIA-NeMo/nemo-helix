# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a Harbor task's ``compose_services`` from the ``docker-compose.yaml`` in its archive.

A compose task runs helper services (a database, a mock API) next to ``main``. OpenSandbox can't run Compose, so
``NemoOpenSandboxEnvironment`` takes the services as a ``ComposeServices`` spec, and the OpenSandbox server's
extension adds them to the sandbox pod as extra containers. This package builds that spec from the task itself:

* :mod:`~nemo_evals.harbor.compose.resolve` merges the task's compose file over Harbor's base file and interpolates it.
* :func:`parse_compose` (this module) reads each service of the result into a :class:`ComposeTask`, using the
  per-option readers in :mod:`~nemo_evals.harbor.compose.service_options` and
  :mod:`~nemo_evals.harbor.compose.images`. A :class:`ComposeTask` says where every image comes from and how each
  service runs, without needing a registry or a sandbox.
* :mod:`~nemo_evals.harbor.compose.spec` maps a :class:`ComposeTask` to ``ComposeServices``.

The compose file is task content, not trusted input. It is parsed with ``yaml.safe_load`` under a size cap and a cap
on how many values its aliases may expand to, and it is never executed. Anything the sandbox would run differently
from Compose is rejected with a typed :class:`ComposeError` that lists every offending path, rather than being
dropped silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_evals.harbor.compose.images import (
    dockerfile_path,
    parse_build_source,
    parse_image_source,
    read_dockerfile_entrypoint,
)
from nemo_evals.harbor.compose.models import (
    ComposeBuild,
    ComposeError,
    ComposeMainSettings,
    ComposeProblems,
    ComposeService,
    ComposeTask,
    ComposeVolumeMount,
)
from nemo_evals.harbor.compose.resolve import (
    COMPOSE_FILENAME,
    ENVIRONMENT_DIR,
    find_compose_file,
    read_env_files,
    resolve_compose,
)
from nemo_evals.harbor.compose.service_options import (
    parse_argument_list,
    parse_capabilities,
    parse_dependency_conditions,
    parse_entrypoint,
    parse_environment,
    parse_memory_quantity,
    parse_platform,
    parse_ports,
    parse_readiness,
    parse_resources,
    parse_volume_mounts,
)
from nhx_sandbox.harbor_opensandbox_services import MAIN_SERVICE

_COMPLETED = "service_completed_successfully"

# Service keys that a pod already behaves as, or that images prebuilt elsewhere make moot.
_NO_EFFECT_KEYS = frozenset({"pull_policy", "restart", "container_name", "hostname", "labels", "logging"})

_COMMON_KEYS = frozenset(
    {
        "image",
        "build",
        "depends_on",
        "healthcheck",
        "entrypoint",
        "command",
        "environment",
        "env_file",
        "expose",
        "ports",
        "volumes",
        "network_mode",
        "networks",
        "platform",
    }
)
_SERVICE_KEYS = _COMMON_KEYS | {"shm_size", "cpus", "mem_limit", "deploy"}
_MAIN_KEYS = _COMMON_KEYS | {"cap_add"}


def parse_compose(task_dir: Path, *, read_entrypoint: bool = True) -> ComposeTask | None:
    """Read a task directory's compose file the way Harbor's Docker path does; None for a single-container task.

    Harbor runs ``docker compose -f <base> -f environment/docker-compose.yaml``, where the base is the
    one that builds ``main`` from ``environment/`` or, when ``task.toml`` names a ``docker_image``, runs
    that image. The task's file is laid over the base, so it may leave ``main`` out or override parts of it.

    A file that defines only ``main`` parses to a task with no services; ``to_compose_services`` refuses it.
    ``read_entrypoint=False`` leaves ``main``'s Dockerfile unread, for callers that plan images and don't need
    its ``ENTRYPOINT``.

    Raises:
        ComposeError: The file can't run as sandbox services. The error's ``code`` says why.
    """
    compose_file = find_compose_file(task_dir)
    if compose_file is None:
        return None

    compose_path = f"{task_dir.name}/{ENVIRONMENT_DIR}/{COMPOSE_FILENAME}"

    # Problems are collected through every step below and raised together, so one error lists all of them.
    problems = ComposeProblems()

    # Merge the task's file over Harbor's base file and interpolate it, as `docker compose` would.
    resolved = resolve_compose(task_dir, compose_file, compose_path, problems)

    # Volumes are declared once at the top level, so read them before any service mounts one.
    declared_volumes = _declared_volumes(resolved.document, problems)
    _check_networks(resolved.document, problems)

    # Parse every service before raising, so a task with several problems reports all of them.
    service_parser = ServiceParser(task_dir, declared_volumes, problems)
    parsed_services = {name: service_parser.service(name, raw) for name, raw in resolved.services.items()}
    main, main_build = service_parser.main(
        resolved.main, resolved.harbor_main, prebuilt=resolved.prebuilt, read_entrypoint=read_entrypoint
    )

    # Roles and start order read `depends_on`, which is only meaningful once every service parsed cleanly.
    problems.raise_if_any(compose_path)

    services = _with_roles(
        {name: service for name, service in parsed_services.items() if service is not None},
        main.depends_on,
        compose_path=compose_path,
    )

    # Only a prebuilt task runs the image named in the compose file; a built `main` comes from its Dockerfile.
    main_image = resolved.main.get("image")
    return ComposeTask(
        compose_path=compose_path,
        main_dockerfile=dockerfile_path(main_build),
        main_build=main_build,
        main_image=str(main_image) if resolved.prebuilt and main_image else None,
        services=services,
        main=main,
        volumes=_mounted_volumes(resolved.document, services, main),
    )


class ServiceParser:
    """Reads the raw services of one compose file into models, recording what a pod can't run in ``problems``."""

    def __init__(self, task_dir: Path, declared_volumes: set[str], problems: ComposeProblems) -> None:
        """Parse services of the task at ``task_dir``, whose compose file declares ``declared_volumes``."""
        self._task_dir = task_dir
        self._declared_volumes = declared_volumes
        self._problems = problems

    def service(self, name: str, raw_service: dict[str, Any]) -> ComposeService | None:
        """Read service ``name`` (any service but ``main``); None when it has no usable image source.

        In that case the problem is recorded in ``problems`` instead of raised.
        """
        path = f"services.{name}"
        self._check_keys(name, raw_service, _SERVICE_KEYS)
        source = parse_image_source(name, raw_service, self._problems)
        if source is None:
            return None

        environment = self._environment(raw_service, path)
        limits, requests = parse_resources(raw_service, path, self._problems)
        return ComposeService(
            name=name,
            source=source,
            platform=parse_platform(raw_service.get("platform"), f"{path}.platform", self._problems),
            depends_on=parse_dependency_conditions(raw_service.get("depends_on"), f"{path}.depends_on", self._problems),
            command=parse_entrypoint(raw_service.get("entrypoint"), f"{path}.entrypoint", self._problems),
            args=parse_argument_list(raw_service.get("command"), f"{path}.command", self._problems),
            env=environment,
            ports=parse_ports(raw_service, path, self._problems),
            readiness=parse_readiness(raw_service.get("healthcheck"), f"{path}.healthcheck", self._problems),
            limits=limits,
            requests=requests,
            volume_mounts=self._volume_mounts(raw_service, path),
            shm_size=parse_memory_quantity(raw_service.get("shm_size"), f"{path}.shm_size", self._problems),
        )

    def main(
        self,
        raw_main: dict[str, Any],
        harbor_main: Mapping[str, Any],
        *,
        prebuilt: bool,
        read_entrypoint: bool,
    ) -> tuple[ComposeMainSettings, ComposeBuild | None]:
        """Read ``main`` as the task's file lays it over Harbor's own, as its settings and what builds it.

        What builds ``main`` is None for a prebuilt task. ``main``'s entrypoint is the compose override if there is
        one, otherwise (for a built ``main``, and when ``read_entrypoint`` is set) the last ``ENTRYPOINT`` of its
        Dockerfile, since the sandbox would otherwise replace it.
        """
        path = f"services.{MAIN_SERVICE}"
        self._check_keys(MAIN_SERVICE, raw_main, _MAIN_KEYS)
        if raw_main.get("command") != harbor_main.get("command"):
            # The sandbox keeps ``main`` alive itself; a task's own ``command`` for it would be dropped.
            self._problems.unsupported(f"{path}.command")

        environment = self._environment(raw_main, path)
        entrypoint = parse_entrypoint(raw_main.get("entrypoint"), f"{path}.entrypoint", self._problems)
        build: ComposeBuild | None = None
        # A prebuilt task has no build to read: `main` runs the image `task.toml` names.
        if not prebuilt:
            build = parse_build_source(MAIN_SERVICE, raw_main.get("build"), self._problems)
            # The sandbox replaces `main`'s command, so carry over the image's own ENTRYPOINT from its Dockerfile.
            if build is not None and entrypoint is None and read_entrypoint:
                entrypoint = read_dockerfile_entrypoint(
                    self._task_dir / ENVIRONMENT_DIR, str(dockerfile_path(build)), f"{path}.build", self._problems
                )

        settings = ComposeMainSettings(
            depends_on=parse_dependency_conditions(raw_main.get("depends_on"), f"{path}.depends_on", self._problems),
            env=environment,
            readiness=parse_readiness(raw_main.get("healthcheck"), f"{path}.healthcheck", self._problems),
            add_capabilities=parse_capabilities(raw_main, path, self._problems),
            volume_mounts=self._volume_mounts(raw_main, path),
            entrypoint=entrypoint,
            platform=parse_platform(raw_main.get("platform"), f"{path}.platform", self._problems),
        )
        return settings, build

    def _environment(self, raw_service: Mapping[str, Any], path: str) -> dict[str, str]:
        """Read a service's variables: its ``env_file`` entries, overridden by its ``environment``, as compose does."""
        return {
            **read_env_files(raw_service.get("env_file"), f"{path}.env_file", self._task_dir, self._problems),
            **parse_environment(raw_service.get("environment"), f"{path}.environment", self._problems),
        }

    def _volume_mounts(self, raw_service: Mapping[str, Any], path: str) -> list[ComposeVolumeMount]:
        """Read a service's ``volumes`` as mounts of the compose file's declared named volumes."""
        return parse_volume_mounts(
            raw_service.get("volumes"), f"{path}.volumes", self._declared_volumes, self._problems
        )

    def _check_keys(self, name: str, raw_service: Mapping[str, Any], allowed_keys: frozenset[str]) -> None:
        """Record every key of service ``name`` that a pod can't honor: unknown keys, other networks, other network modes."""
        for key in raw_service:
            if key not in allowed_keys and key not in _NO_EFFECT_KEYS and not str(key).startswith("x-"):
                self._problems.unsupported(f"services.{name}.{key}")

        network_mode = raw_service.get("network_mode")
        if network_mode is not None and not (isinstance(network_mode, str) and network_mode.startswith("service:")):
            self._problems.unsupported(f"services.{name}.network_mode")

        networks = raw_service.get("networks")
        network_names = {str(network) for network in networks} if isinstance(networks, (list, dict)) else set()
        if network_names - {"default"}:
            self._problems.unsupported(f"services.{name}.networks")


def _declared_volumes(document: Mapping[str, Any], problems: ComposeProblems) -> set[str]:
    """Read the names of the top-level ``volumes``; options on a volume aren't supported."""
    volumes = document.get("volumes")
    if volumes is None:
        return set()
    if not isinstance(volumes, dict):
        problems.invalid("volumes", "expected a mapping")
        return set()

    for name, options in volumes.items():
        if options not in (None, {}):
            problems.unsupported(f"volumes.{name}: options")
    return {str(name) for name in volumes}


def _check_networks(document: Mapping[str, Any], problems: ComposeProblems) -> None:
    """Record any top-level network besides the default: a pod has one network."""
    networks = document.get("networks")
    if isinstance(networks, dict) and set(networks) - {"default"}:
        problems.unsupported("networks")


def _mounted_volumes(
    document: Mapping[str, Any], services: Mapping[str, ComposeService], main: ComposeMainSettings
) -> list[str]:
    """List the declared volumes that ``main`` or any service mounts, in the order the file declares them."""
    declared = document.get("volumes")
    declared_names = [str(name) for name in declared] if isinstance(declared, dict) else []

    mounts = [mount for service in services.values() for mount in service.volume_mounts] + main.volume_mounts
    mounted_names = {mount.name for mount in mounts}
    return [name for name in declared_names if name in mounted_names]


def _check_dependencies(
    services: Mapping[str, ComposeService], main_depends_on: Mapping[str, str], compose_path: str
) -> None:
    """Raise unless every service that a service or ``main`` waits for exists."""
    problems = ComposeProblems()
    for name, service in services.items():
        for dependency in service.depends_on:
            if dependency != MAIN_SERVICE and dependency not in services:
                problems.invalid(f"services.{name}.depends_on", f"{dependency!r} is not a service")
    for dependency in main_depends_on:
        if dependency not in services:
            problems.invalid("services.main.depends_on", f"{dependency!r} is not a service")
    problems.raise_if_any(compose_path)


def _after_main_services(services: Mapping[str, ComposeService]) -> set[str]:
    """Find the services that start only once ``main`` is up: those that wait for ``main``, or for such a service.

    With ``worker`` waiting for ``main`` and ``reporter`` waiting for ``worker``, both are after ``main``.
    """
    after_main: set[str] = set()
    found_new = True
    # Repeat until no service joins, so a chain (reporter waits for worker, worker for main) is followed to its end.
    while found_new:
        found_new = False
        for name, service in services.items():
            waits_for_main = MAIN_SERVICE in service.depends_on or after_main & set(service.depends_on)
            if name not in after_main and waits_for_main:
                after_main.add(name)
                found_new = True
    return after_main


def _start_order(services: Mapping[str, ComposeService], compose_path: str) -> list[str]:
    """Order the service names so each comes after its dependencies; ``main`` is not part of the order.

    Raises:
        ComposeError: The services depend on each other in a cycle.
    """
    ordered: list[str] = []
    remaining = list(services)
    while remaining:
        # A service is ready to start once every dependency other than `main` is already placed.
        ready = [
            name
            for name in remaining
            if all(dependency in ordered for dependency in services[name].depends_on if dependency != MAIN_SERVICE)
        ]
        if not ready:
            raise ComposeError(
                "compose_invalid",
                "the services depend on each other in a cycle",
                compose_path=compose_path,
                paths=tuple(remaining),
            )

        ordered.append(ready[0])
        remaining.remove(ready[0])
    return ordered


def _run_once_services(services: Mapping[str, ComposeService], main_depends_on: Mapping[str, str]) -> set[str]:
    """Find the services that something waits on to finish (``service_completed_successfully``).

    Such a service is a one-time step, like a migration: it runs to completion before its dependents start.
    """
    run_once = {dependency for dependency, condition in main_depends_on.items() if condition == _COMPLETED}
    run_once |= {
        dependency
        for service in services.values()
        for dependency, condition in service.depends_on.items()
        if condition == _COMPLETED
    }
    return run_once


def _with_roles(
    services: dict[str, ComposeService], main_depends_on: Mapping[str, str], *, compose_path: str
) -> dict[str, ComposeService]:
    """Give each service its role and put the services in start order (a dependency before its dependents).

    A service is ``after_main`` if it waits for ``main``, ``run_once`` if something waits for it to finish, and a
    ``sidecar`` otherwise. A ``run_once`` service loses its readiness check, since its dependents wait for it to exit.

    Raises:
        ComposeError: A dependency doesn't exist, the services form a cycle, or a service and ``main`` wait for each other.
    """
    _check_dependencies(services, main_depends_on, compose_path)

    after_main = _after_main_services(services)
    # `main` can't wait for a service that waits for `main`: neither could start.
    if clash := sorted(after_main & set(main_depends_on)):
        raise ComposeError(
            "compose_invalid",
            "a service and main depend on each other",
            compose_path=compose_path,
            paths=tuple(clash),
        )

    run_once = _run_once_services(services, main_depends_on)
    result: dict[str, ComposeService] = {}
    for name in _start_order(services, compose_path):
        service = services[name]
        # A service that must finish first never reports ready, so its readiness check is dropped.
        role = "after_main" if name in after_main else "run_once" if name in run_once else "sidecar"
        result[name] = service.model_copy(
            update={"role": role, "readiness": None if role == "run_once" else service.readiness}
        )
    return result
