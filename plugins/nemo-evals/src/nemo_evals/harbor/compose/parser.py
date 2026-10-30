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
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.models.trial.paths import EnvironmentPaths
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
    ComposeSecurity,
    ComposeService,
    ComposeTask,
    ComposeTmpfs,
    ComposeVolumeMount,
)
from nemo_evals.harbor.compose.resolve import (
    COMPOSE_FILENAME,
    ENVIRONMENT_DIR,
    HARBOR_LOG_DIRECTORIES,
    Interpolator,
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
    parse_security,
    parse_tmpfs,
    parse_volume_mounts,
    parse_working_dir,
)
from nhx_sandbox.harbor_opensandbox_services import MAIN_SERVICE, MAX_SERVICES, quantity_value

_COMPLETED = "service_completed_successfully"
_HEALTHY = "service_healthy"
_STARTED = "service_started"

# Service keys that a pod already behaves as, or that images prebuilt elsewhere make moot. Two do differ a little:
# another service can't reach one by its `container_name`, only by its service name, and `hostname` prints the pod's
# name. They're accepted anyway, since the tasks that set them reach every service by its service name. `restart` is
# not among them: a pod can't give each service its own restart policy. `tty` and `stdin_open` attach a terminal to
# the container's own process; Harbor runs every command through exec, which never uses it.
_NO_EFFECT_KEYS = frozenset({"pull_policy", "container_name", "hostname", "labels", "logging", "tty", "stdin_open"})

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
# Keys that only narrow what a container may do; see :func:`parse_security` and :func:`parse_tmpfs`.
_SECURITY_KEYS = frozenset({"user", "read_only", "cap_drop", "security_opt", "tmpfs"})
_SERVICE_KEYS = _COMMON_KEYS | _SECURITY_KEYS | {"shm_size", "cpus", "mem_limit", "deploy"}
_MAIN_KEYS = _COMMON_KEYS | _SECURITY_KEYS | {"cap_add", "cpus", "mem_limit", "deploy", "working_dir"}

# Where Harbor writes in `main`: the log directories `start()` creates, and where it uploads the tests and the
# solution. A read-only `main` must mount something writable over each.
_PATHS_HARBOR_WRITES = tuple(
    str(path)
    for path in (
        EnvironmentPaths.agent_dir,
        EnvironmentPaths.verifier_dir,
        EnvironmentPaths.artifacts_dir,
        EnvironmentPaths.tests_dir,
        EnvironmentPaths.solution_dir,
    )
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
    # The file is untrusted and the sandbox runs at most `MAX_SERVICES`, so a larger one is refused before any of the
    # per-service work below, whose start order grows with the cube of the number of services.
    if len(resolved.services) > MAX_SERVICES:
        raise ComposeError(
            "compose_unsupported",
            f"the file defines {len(resolved.services)} services besides main; a sandbox runs at most {MAX_SERVICES}",
            compose_path=compose_path,
        )

    # Volumes are declared once at the top level, so read them before any service mounts one.
    declared_volumes = _declared_volumes(resolved.document, problems)
    for name in sorted(declared_volumes & {volume for _path, volume in HARBOR_LOG_DIRECTORIES.values()}):
        problems.invalid(f"volumes.{name}", "this name is reserved for Harbor's log directories")
    _check_networks(resolved.document, {MAIN_SERVICE: resolved.main, **resolved.services}, problems)
    _check_main_network_mode(resolved.main, resolved.services, resolved.no_network_in_every_phase, problems)

    # Parse every service before raising, so a task with several problems reports all of them.
    service_parser = ServiceParser(task_dir, declared_volumes, resolved.interpolator, problems)
    parsed_services = {
        name: service_parser.service(name, raw_service, prebuilt=resolved.prebuilt)
        for name, raw_service in resolved.services.items()
    }
    main, main_build = service_parser.main(
        resolved.main,
        resolved.harbor_main,
        prebuilt=resolved.prebuilt,
        read_entrypoint=read_entrypoint,
        task_limits=resolved.task_limits,
    )

    # Roles and start order read `depends_on`, which is only meaningful once every service parsed cleanly.
    problems.raise_if_any(compose_path)

    services = _with_roles(
        {name: service for name, service in parsed_services.items() if service is not None},
        main.depends_on,
        compose_path=compose_path,
    )
    services, main, log_volumes = _share_harbor_log_mounts(services, main, resolved.harbor_log_mounts)
    _check_harbor_paths_writable(main, resolved.workdir, compose_path)

    # Only a prebuilt task runs the image named in the compose file; a built `main` comes from its Dockerfile.
    main_image = resolved.main.get("image")
    return ComposeTask(
        compose_path=compose_path,
        main_dockerfile=dockerfile_path(main_build),
        main_build=main_build,
        main_image=str(main_image) if resolved.prebuilt and main_image else None,
        services=services,
        main=main,
        volumes=[
            *_mounted_volumes(resolved.document, services, main),
            *service_parser.anonymous_volumes,
            *log_volumes,
        ],
    )


class ServiceParser:
    """Reads the raw services of one compose file into models, recording what a pod can't run in ``problems``."""

    def __init__(
        self, task_dir: Path, declared_volumes: set[str], interpolator: Interpolator, problems: ComposeProblems
    ) -> None:
        """Parse services of the task at ``task_dir``, whose compose file declares ``declared_volumes``.

        ``env_file`` and value-less entries are read with ``interpolator``.
        """
        self._task_dir = task_dir
        self._declared_volumes = declared_volumes
        self._interpolator = interpolator
        self._problems = problems
        # Every anonymous volume a service mounts, in parse order: each is its own pod volume.
        self.anonymous_volumes: list[str] = []

    def service(self, name: str, raw_service: dict[str, Any], *, prebuilt: bool) -> ComposeService | None:
        """Read service ``name`` (any service but ``main``) of a task that is ``prebuilt`` or not.

        None when it has no usable image source; the problem is then recorded in ``problems`` instead of raised.
        """
        path = f"services.{name}"
        self._check_keys(name, raw_service, _SERVICE_KEYS)
        source = parse_image_source(name, raw_service, self._interpolator, self._problems, prebuilt=prebuilt)
        if source is None:
            return None

        environment = self._environment(raw_service, path)
        limits, requests = parse_resources(raw_service, path, self._problems)
        security, tmpfs = self._security_and_tmpfs(raw_service, path)
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
            volume_mounts=self._volume_mounts(name, raw_service, path),
            shm_size=parse_memory_quantity(raw_service.get("shm_size"), f"{path}.shm_size", self._problems),
            security=security,
            tmpfs=tmpfs,
        )

    def main(
        self,
        raw_main: dict[str, Any],
        harbor_main: Mapping[str, Any],
        *,
        prebuilt: bool,
        read_entrypoint: bool,
        task_limits: Mapping[str, str],
    ) -> tuple[ComposeMainSettings, ComposeBuild | None]:
        """Read ``main`` as the task's file lays it over Harbor's own, as its settings and what builds it.

        What builds ``main`` is None for a prebuilt task. ``main``'s entrypoint is the compose override if there is
        one, otherwise (for a built ``main``, and when ``read_entrypoint`` is set) the last ``ENTRYPOINT`` of its
        Dockerfile, since the sandbox would otherwise replace it. ``task_limits`` are ``task.toml``'s CPUs and
        memory, which ``main``'s own limits may only repeat.
        """
        path = f"services.{MAIN_SERVICE}"
        self._check_keys(MAIN_SERVICE, raw_main, _MAIN_KEYS)
        self._check_main_limits(raw_main, task_limits, path)

        # Harbor's own command only keeps `main` alive, which the sandbox does too; a task's own replaces it.
        command = None
        if raw_main.get("command") != harbor_main.get("command"):
            command = parse_argument_list(raw_main.get("command"), f"{path}.command", self._problems)
            # `command: null` and `command: []` both mean Docker runs the image's own CMD, which isn't known here.
            # Accepting them would run the sandbox's keep-alive instead, and the CMD's process would never start.
            if raw_main.get("command") is None:
                self._problems.unsupported(f"{path}.command: null")
            elif command == []:
                self._problems.unsupported(f"{path}.command: an empty command")

        environment = self._environment(raw_main, path)
        entrypoint = parse_entrypoint(raw_main.get("entrypoint"), f"{path}.entrypoint", self._problems)
        build: ComposeBuild | None = None
        # A prebuilt task has no build to read: `main` runs the image `task.toml` names.
        if not prebuilt:
            build = parse_build_source(MAIN_SERVICE, raw_main.get("build"), self._interpolator, self._problems)
            # The sandbox replaces `main`'s command, so carry over the image's own ENTRYPOINT from its Dockerfile.
            if build is not None and entrypoint is None and read_entrypoint:
                entrypoint = read_dockerfile_entrypoint(
                    self._task_dir, str(dockerfile_path(build)), f"{path}.build", self._problems
                )

        security, tmpfs = self._security_and_tmpfs(raw_main, path)
        settings = ComposeMainSettings(
            depends_on=parse_dependency_conditions(raw_main.get("depends_on"), f"{path}.depends_on", self._problems),
            env=environment,
            readiness=parse_readiness(raw_main.get("healthcheck"), f"{path}.healthcheck", self._problems),
            add_capabilities=parse_capabilities(raw_main, path, self._problems),
            volume_mounts=self._volume_mounts(MAIN_SERVICE, raw_main, path),
            entrypoint=entrypoint,
            ports=parse_ports(raw_main, path, self._problems),
            command=command or None,
            working_dir=parse_working_dir(raw_main.get("working_dir"), f"{path}.working_dir", self._problems),
            platform=parse_platform(raw_main.get("platform"), f"{path}.platform", self._problems),
            security=security,
            tmpfs=tmpfs,
        )
        return settings, build

    def _check_main_limits(self, raw_main: Mapping[str, Any], task_limits: Mapping[str, str], path: str) -> None:
        """Record resources on ``main`` that the sandbox wouldn't apply.

        The sandbox sizes ``main`` from ``task.toml``, and many files set ``main``'s limits from Harbor's ``${CPUS}``
        and ``${MEMORY}``, which only repeats it. Limits that say something else, and reservations (requests, GPUs),
        would be silently ignored, so they are unsupported.
        """
        limits, requests = parse_resources(raw_main, path, self._problems)
        # Read `task.toml`'s values the same way and compare by value, so `2g` matches `memory_mb = 2048`.
        expected, _ = parse_resources(task_limits, path, self._problems)
        if requests:
            self._problems.unsupported(f"{path}.deploy.resources.reservations")
        if any(
            resource not in expected or quantity_value(expected[resource]) != quantity_value(quantity)
            for resource, quantity in limits.items()
        ):
            self._problems.unsupported(f"{path}: resource limits {limits} differ from task.toml's {expected}")

    def _environment(self, raw_service: Mapping[str, Any], path: str) -> dict[str, str]:
        """Read a service's variables: its ``env_file`` entries, overridden by its ``environment``, as compose does.

        A value-less ``environment`` entry overrides an ``env_file`` too, so it's filled in only after the two merge.
        """
        environment: dict[str, str | None] = {
            **read_env_files(
                raw_service.get("env_file"), f"{path}.env_file", self._task_dir, self._interpolator, self._problems
            ),
            **parse_environment(raw_service.get("environment"), f"{path}.environment", self._problems),
        }
        return self._interpolator.fill_in_value_less_entries(environment, f"{path}.environment")

    def _security_and_tmpfs(
        self, raw_service: Mapping[str, Any], path: str
    ) -> tuple[ComposeSecurity | None, list[ComposeTmpfs]]:
        """Read a service's restrictions and its ``tmpfs``, whose ``nosuid`` and ``nodev`` depend on them."""
        security = parse_security(raw_service, path, self._problems)
        return security, parse_tmpfs(raw_service.get("tmpfs"), f"{path}.tmpfs", security, self._problems)

    def _volume_mounts(self, name: str, raw_service: Mapping[str, Any], path: str) -> list[ComposeVolumeMount]:
        """Read service ``name``'s ``volumes`` as mounts of the file's declared volumes, or of anonymous ones.

        Each anonymous volume is recorded in ``anonymous_volumes``. One whose generated name the file also declares
        is refused, since the two would share a pod volume.
        """
        mounts = parse_volume_mounts(
            raw_service.get("volumes"), f"{path}.volumes", name, self._declared_volumes, self._problems
        )
        for mount in mounts:
            if not mount.anonymous:
                continue
            if mount.name in self._declared_volumes:
                self._problems.invalid(f"volumes.{mount.name}", "this name is reserved for an anonymous volume")
            self.anonymous_volumes.append(mount.name)
        return mounts

    def _check_keys(self, name: str, raw_service: Mapping[str, Any], allowed_keys: frozenset[str]) -> None:
        """Record every key of service ``name`` that a pod can't honor: unknown keys and other network modes.

        Its ``networks`` are checked across all services, by :func:`_check_networks`.
        """
        for key in raw_service:
            if key not in allowed_keys and key not in _NO_EFFECT_KEYS and not str(key).startswith("x-"):
                self._problems.unsupported(f"services.{name}.{key}")

        # `main`'s `network_mode: none` is checked by `_check_main_network_mode`.
        main_without_network = name == MAIN_SERVICE and raw_service.get("network_mode") == "none"
        if _shares_a_network_namespace(raw_service) is False and not main_without_network:
            self._problems.unsupported(f"services.{name}.network_mode")


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


def _check_main_network_mode(
    raw_main: Mapping[str, Any],
    raw_services: Mapping[str, Mapping[str, Any]],
    no_network_in_every_phase: bool,
    problems: ComposeProblems,
) -> None:
    """Record ``network_mode: none`` on ``main`` unless the sandbox would isolate ``main`` as much.

    A pod always has a network, which execd needs to take Harbor's commands, so ``main`` can't go without one. What
    a pod can do is deny all egress, which the sandbox already does when ``task.toml`` is ``no-network``; Harbor
    itself runs a ``no-network`` task by setting this same ``network_mode: none`` on ``main``. So it's accepted when
    ``task.toml`` already says so for every phase. It's refused alongside any other service: under Compose those
    can't reach ``main``, but in a pod they share its network.
    """
    if raw_main.get("network_mode") != "none":
        return
    if not no_network_in_every_phase:
        problems.unsupported(
            f'services.{MAIN_SERVICE}.network_mode: none, unless task.toml sets network_mode = "no-network" '
            "for every phase"
        )
    if raw_services:
        problems.unsupported(f"services.{MAIN_SERVICE}.network_mode: none, alongside other services")


def _check_harbor_paths_writable(main: ComposeMainSettings, workdir: str | None, compose_path: str) -> None:
    """Raise if Harbor couldn't write somewhere it needs to in ``main``.

    Harbor writes its logs, and uploads the tests and the solution, into ``main``. The sandbox also creates
    ``task.toml``'s ``workdir`` there when it starts, unless it's ``/``. Without this check such a task would only fail
    later, when the sandbox starts or the verifier runs.

    What decides is the closest mount over each path, as in the container: a read-only mount inside a writable one
    wins over it. A path no mount covers is on the image's own filesystem, which is writable unless ``main`` is
    ``read_only``.
    """
    main_is_read_only = main.security is not None and main.security.read_only
    # Each mount over main, and whether it can be written.
    mounts = [(PurePosixPath(mount.mount_path), True) for mount in main.tmpfs]
    mounts += [(PurePosixPath(mount.mount_path), not mount.read_only) for mount in main.volume_mounts]

    paths_written = [*_PATHS_HARBOR_WRITES, *([workdir] if workdir and workdir != "/" else [])]
    unwritable_paths = []
    for path in paths_written:
        covering_mounts = [
            (mount_path, writable) for mount_path, writable in mounts if PurePosixPath(path).is_relative_to(mount_path)
        ]
        if covering_mounts:
            _closest_path, writable = max(covering_mounts, key=lambda mount: len(mount[0].parts))
        else:
            writable = not main_is_read_only
        if not writable:
            unwritable_paths.append(path)

    if unwritable_paths:
        raise ComposeError(
            "compose_unsupported",
            "Harbor writes to these paths in main, but they aren't writable; mount a writable tmpfs or volume over each",
            compose_path=compose_path,
            paths=tuple(unwritable_paths),
        )


def _shares_a_network_namespace(raw_service: Mapping[str, Any]) -> bool | None:
    """Whether a service sets ``network_mode: service:<other>``; None without a ``network_mode``, False for any other.

    Sharing another service's network namespace is what every container in a pod already does.
    """
    network_mode = raw_service.get("network_mode")
    if network_mode is None:
        return None
    return isinstance(network_mode, str) and network_mode.startswith("service:")


def _check_networks(
    document: Mapping[str, Any], raw_services: Mapping[str, Mapping[str, Any]], problems: ComposeProblems
) -> None:
    """Record any use of networks that a pod, which has exactly one network, couldn't reproduce.

    Every container in a pod shares one network, so a file is accepted only when Compose would also connect every
    service to every other: when they all, ``main`` included, join the same set of networks, and those networks are
    plain bridges. Separate network sets would isolate services from each other, and per-service settings (a fixed
    ``ipv4_address``, extra ``aliases``) have no meaning in a pod, so both are refused.

    ``raw_services`` holds every service by name, ``main`` included.
    """
    declared = document.get("networks") or {}
    if not isinstance(declared, dict):
        problems.invalid("networks", "expected a mapping")
        declared = {}

    for network, options in declared.items():
        if options is None:
            continue
        if not isinstance(options, dict):
            problems.invalid(f"networks.{network}", "expected a mapping")
            continue
        for option, value in options.items():
            # A plain bridge is what a pod's one network behaves as; anything else (a fixed subnet, an internal-only
            # network, an external one) changes what the services can reach.
            if (option, value) not in (("driver", "bridge"), ("internal", False)) and option != "name":
                problems.unsupported(f"networks.{network}.{option}")

    owners = _network_namespace_owners(raw_services, problems)
    network_sets: dict[str, frozenset[str]] = {}
    for name, raw_service in raw_services.items():
        # A service in another's network namespace has no networks of its own.
        if _shares_a_network_namespace(raw_service):
            continue
        networks = _service_networks(name, raw_service.get("networks"), problems)
        for network in networks - {"default"} - set(declared):
            problems.invalid(f"services.{name}.networks", f"network {network!r} is not declared")
        network_sets[name] = networks

    # `main` in another service's namespace has that service's networks.
    reference = network_sets.get(owners.get(MAIN_SERVICE, MAIN_SERVICE), frozenset({"default"}))
    for name, networks in network_sets.items():
        if networks != reference:
            problems.unsupported(f"services.{name}.networks: not the same networks as main")


def _network_namespace_owners(
    raw_services: Mapping[str, Mapping[str, Any]], problems: ComposeProblems
) -> dict[str, str]:
    """Map each service to the one whose network namespace it runs in, following ``network_mode: service:`` links.

    ``{a: {network_mode: "service:b"}, b: {}}`` gives ``{"a": "b", "b": "b"}``. As in Compose, a link to an undefined
    service and a loop of links are invalid; a service in a loop gets no owner.
    """
    links: dict[str, str] = {}
    for name, raw_service in raw_services.items():
        if _shares_a_network_namespace(raw_service):
            target = raw_service["network_mode"].removeprefix("service:")
            if target in raw_services:
                links[name] = target
            else:
                problems.invalid(f"services.{name}.network_mode", f"depends on undefined service {target!r}")

    order = list(raw_services)
    owners: dict[str, str] = {}
    for name in raw_services:
        chain = [name]
        while chain[-1] in links and links[chain[-1]] not in chain:
            chain.append(links[chain[-1]])
        if chain[-1] not in links:
            owners[name] = chain[-1]
            continue
        # The chain looped back. Report the loop once, from its first service in the file.
        loop = chain[chain.index(links[chain[-1]]) :]
        if name == min(loop, key=order.index):
            problems.invalid(
                f"services.{name}.network_mode", f"dependency cycle detected: {' -> '.join([*loop, name])}"
            )
    return owners


def _service_networks(name: str, value: Any, problems: ComposeProblems) -> frozenset[str]:
    """Read the networks a service joins: ``default`` when it names none, as in Compose.

    ``[backend]`` and ``{backend: {}}`` both give ``{"backend"}``. A per-network setting such as ``ipv4_address`` is
    recorded as unsupported.
    """
    path = f"services.{name}.networks"
    if value is None:
        return frozenset({"default"})
    if isinstance(value, dict):
        for network, settings in value.items():
            if settings is None:
                continue
            if not isinstance(settings, dict):
                problems.invalid(f"{path}.{network}", "expected a mapping")
                continue
            for setting in settings:
                problems.unsupported(f"{path}.{network}.{setting}")
        return frozenset(str(network) for network in value)
    if isinstance(value, list) and all(isinstance(network, str) for network in value):
        return frozenset(value)

    problems.unsupported(path)
    return frozenset()


def _share_harbor_log_mounts(
    services: dict[str, ComposeService], main: ComposeMainSettings, log_mounts: Mapping[str, Mapping[str, bool]]
) -> tuple[dict[str, ComposeService], ComposeMainSettings, list[str]]:
    """Give each service that mounted a Harbor log directory a pod volume there, shared with ``main``.

    Under Harbor such a mount binds one host directory into several containers, so what a service writes there (kumo's
    game server writes the agent's step log to ``/logs/verifier``) reaches ``main``, where the verifier and Harbor
    read it. A shared pod volume keeps that. ``main``'s own log mounts need nothing: the sandbox creates the
    directories in it.

    Returns the services, ``main``, and the names of the volumes added, in :data:`HARBOR_LOG_DIRECTORIES` order.
    """
    shared: set[str] = set()
    for name, directories in log_mounts.items():
        service = services.get(name)
        if service is None:
            continue
        mounts = [
            ComposeVolumeMount(name=volume, mount_path=path, read_only=read_only)
            for directory, read_only in directories.items()
            for path, volume in [HARBOR_LOG_DIRECTORIES[directory]]
        ]
        services[name] = service.model_copy(update={"volume_mounts": [*service.volume_mounts, *mounts]})
        shared.update(mount.name for mount in mounts)

    # `main` mounts each shared volume read-write, where its own directory would be.
    added = [(path, volume) for path, volume in HARBOR_LOG_DIRECTORIES.values() if volume in shared]
    main_mounts = [ComposeVolumeMount(name=volume, mount_path=path) for path, volume in added]
    main = main.model_copy(update={"volume_mounts": [*main.volume_mounts, *main_mounts]})
    return services, main, [volume for _path, volume in added]


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


def _check_waits(
    services: Mapping[str, ComposeService], main_depends_on: Mapping[str, str], after_main: set[str], compose_path: str
) -> None:
    """Raise for every wait on another service that the pod would start without honoring.

    * ``service_healthy`` on a service with no ``test`` in the compose file: nothing in the pod runs an image's own
      ``HEALTHCHECK``, so the dependent would start at once. Compose waits for the image's check, or refuses the file
      when there is none.
    * ``service_healthy`` or ``service_completed_successfully`` between two ``after_main`` services: those are regular
      containers, which Kubernetes starts in order but never holds on another's health or exit. ``service_started``
      is honored by that order.
    """
    problems = ComposeProblems()
    waits = [(MAIN_SERVICE, main_depends_on), *((name, service.depends_on) for name, service in services.items())]
    for name, depends_on in waits:
        for dependency, condition in depends_on.items():
            if dependency == MAIN_SERVICE:
                continue
            path = f"services.{name}.depends_on.{dependency}"
            if condition == _HEALTHY and services[dependency].readiness is None:
                problems.unsupported(f"{path}: {dependency} has no healthcheck test in the compose file")
            elif name in after_main and dependency in after_main and condition != _STARTED:
                problems.unsupported(f"{path}: {condition} between two services that start after main")
    problems.raise_if_any(compose_path)


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
        ComposeError: A dependency doesn't exist, the services form a cycle, a service and ``main`` wait for each
            other, or a wait is one the pod can't honor (see :func:`_check_waits`).
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
    _check_waits(services, main_depends_on, after_main, compose_path)

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
