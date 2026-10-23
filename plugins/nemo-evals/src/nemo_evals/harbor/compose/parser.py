# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a Harbor task's ``docker-compose.yaml`` into a :class:`ComposeTask`, one service at a time.

The package works in three steps: :mod:`~nemo_evals.harbor.compose.resolve` merges the task's file over Harbor's base
file and fills in its variables, :func:`parse_compose` (this module) reads each service, and
:mod:`~nemo_evals.harbor.compose.spec` maps the result to the sandbox's ``ComposeServices``. The file is untrusted, so
it's never run, and anything a pod would run differently from Compose is refused with a :class:`ComposeError` that
lists every offending key.
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
    parse_dependency_conditions,
    parse_entrypoint,
    parse_environment,
    parse_ports,
    parse_readiness,
)
from nhx_sandbox.harbor_opensandbox_services import MAIN_SERVICE

_COMPLETED = "service_completed_successfully"

# Service keys that a pod already behaves as, or that images prebuilt elsewhere make moot.
_NO_EFFECT_KEYS = frozenset({"pull_policy", "restart", "container_name", "hostname", "labels", "logging"})

# The service keys read so far; any other key is recorded as unsupported.
_SERVICE_KEYS = frozenset(
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
        "network_mode",
    }
)


def parse_compose(task_dir: Path, *, read_entrypoint: bool = True) -> ComposeTask | None:
    """Read a task's compose file as Harbor does; None for a task without one.

    1. Find ``environment/docker-compose.yaml`` (see :func:`find_compose_file`).
    2. Merge it over Harbor's base file and fill in its variables (see :func:`resolve_compose`).
    3. Read every service other than ``main`` into a :class:`ComposeService`.
    4. Read ``main``: its build or image, and the settings the sandbox carries over.
    5. Raise every problem found so far, together.
    6. Give each service a role from ``depends_on`` and put them in start order.

    The task's file may leave ``main`` out or change only part of it, since Harbor's base file defines it. A file that
    defines only ``main`` gives a task with no services. ``read_entrypoint=False`` skips ``main``'s Dockerfile, for
    image planning.

    Raises:
        ComposeError: The file can't run as sandbox services; ``code`` says why.
    """
    compose_file = find_compose_file(task_dir)
    if compose_file is None:
        return None

    compose_path = f"{task_dir.name}/{ENVIRONMENT_DIR}/{COMPOSE_FILENAME}"

    # Problems are collected through every step below and raised together, so one error lists all of them.
    problems = ComposeProblems()

    # Merge the task's file over Harbor's base file and interpolate it, as `docker compose` would.
    resolved = resolve_compose(task_dir, compose_file, compose_path, problems)

    # Parse every service before raising, so a task with several problems reports all of them.
    service_parser = ServiceParser(task_dir, problems)
    parsed_services = {
        name: service_parser.service(name, raw_service) for name, raw_service in resolved.services.items()
    }
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
    )


class ServiceParser:
    """Reads the raw services of one compose file into models, recording what a pod can't run in ``problems``."""

    def __init__(self, task_dir: Path, problems: ComposeProblems) -> None:
        """Parse services of the task at ``task_dir``."""
        self._task_dir = task_dir
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

        return ComposeService(
            name=name,
            source=source,
            depends_on=parse_dependency_conditions(raw_service.get("depends_on"), f"{path}.depends_on", self._problems),
            command=parse_entrypoint(raw_service.get("entrypoint"), f"{path}.entrypoint", self._problems),
            args=parse_argument_list(raw_service.get("command"), f"{path}.command", self._problems),
            env=self._environment(raw_service, path),
            ports=parse_ports(raw_service, path, self._problems),
            readiness=parse_readiness(raw_service.get("healthcheck"), f"{path}.healthcheck", self._problems),
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
        self._check_keys(MAIN_SERVICE, raw_main, _SERVICE_KEYS)
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
            entrypoint=entrypoint,
        )
        return settings, build

    def _environment(self, raw_service: Mapping[str, Any], path: str) -> dict[str, str]:
        """Read a service's variables: its ``env_file`` entries, overridden by its ``environment``, as compose does."""
        return {
            **read_env_files(raw_service.get("env_file"), f"{path}.env_file", self._task_dir, self._problems),
            **parse_environment(raw_service.get("environment"), f"{path}.environment", self._problems),
        }

    def _check_keys(self, name: str, raw_service: Mapping[str, Any], allowed_keys: frozenset[str]) -> None:
        """Record every key of service ``name`` that a pod can't honor: unknown keys and other network modes."""
        for key in raw_service:
            if key not in allowed_keys and key not in _NO_EFFECT_KEYS and not str(key).startswith("x-"):
                self._problems.unsupported(f"services.{name}.{key}")

        if _shares_a_network_namespace(raw_service) is False:
            self._problems.unsupported(f"services.{name}.network_mode")


def _shares_a_network_namespace(raw_service: Mapping[str, Any]) -> bool | None:
    """Whether a service sets ``network_mode: service:<other>``; None without a ``network_mode``, False for any other.

    Sharing another service's network namespace is what every container in a pod already does.
    """
    network_mode = raw_service.get("network_mode")
    if network_mode is None:
        return None
    return isinstance(network_mode, str) and network_mode.startswith("service:")


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
    run_once: set[str] = set()
    for depends_on in [main_depends_on, *(service.depends_on for service in services.values())]:
        for dependency, condition in depends_on.items():
            if condition == _COMPLETED:
                run_once.add(dependency)
    return run_once


def _with_roles(
    services: dict[str, ComposeService], main_depends_on: Mapping[str, str], *, compose_path: str
) -> dict[str, ComposeService]:
    """Give each service its role and put the services in start order, each after its dependencies.

    ``after_main`` waits for ``main``; ``run_once`` is something others wait on to finish, so it loses its readiness
    check; anything else is a ``sidecar``.

    Raises:
        ComposeError: A dependency doesn't exist, the services form a cycle, or a service and ``main`` wait for
            each other.
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
