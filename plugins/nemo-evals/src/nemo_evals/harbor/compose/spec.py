# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a parsed :class:`ComposeTask` to the ``ComposeServices`` spec that ``NemoOpenSandboxEnvironment`` takes."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_evals.harbor.compose.models import (
    ComposeBuild,
    ComposeError,
    ComposeImage,
    ComposeMainSettings,
    ComposeService,
    ComposeTask,
    ComposeVolumeMount,
)
from nemo_evals.harbor.compose.parser import parse_compose
from nhx_sandbox.harbor_opensandbox_services import (
    MAIN_SERVICE,
    RESERVED_NAMES,
    RESERVED_PORTS,
    ComposeServices,
    egress_api_port_for,
)
from pydantic import ValidationError

# The bounds of ``MainReadiness.timeout_sec`` in ``ComposeServices``.
MIN_READY_TIMEOUT_SEC = 300
MAX_READY_TIMEOUT_SEC = 3600

# The digest that stands in for a built service's image before the image exists.
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64


def to_compose_services(task: ComposeTask, images: Mapping[str, str] | None = None) -> ComposeServices:
    """Translate a parsed compose file into the ``ComposeServices`` spec the sandbox builds its pod from.

    The input comes from :func:`parse_compose` and still speaks Compose: services by name, ``entrypoint`` and
    ``command``, Compose healthcheck timings. OpenSandbox can't run Compose, so each service becomes an extra
    container in the sandbox pod. A file that only changes ``main`` gives a spec with no services. The task Job
    passes the result to ``NemoOpenSandboxEnvironment`` as ``compose_services``, and submit checks it early through
    :func:`check_compose_task`.

    Args:
        task: The parsed compose file.
        images: A full image reference per service, such as a digest-pinned one. A published image without an
            entry runs by its normalized tag; a built service needs one.

    Raises:
        ComposeError: A service has no image, or the spec breaks a rule of ``ComposeServices``
            (``compose_unsupported``: the parser has already refused what Compose itself would).
    """
    _check_reserved(task)

    # One wait budget covers every service, since the sandbox waits for them together.
    wait_budget_sec = _wait_budget_sec(task)
    volume_names = _pod_volume_names(task.volumes, task.compose_path)
    spec: dict[str, Any] = {
        "services": [
            _service_spec(name, service, images or {}, volume_names, task.compose_path)
            for name, service in task.services.items()
        ],
        "main": _main_spec(task.main, wait_budget_sec, volume_names),
    }
    if task.volumes:
        spec["volumes"] = [volume_names[volume] for volume in task.volumes]
    if task.main.readiness is None and wait_budget_sec > MIN_READY_TIMEOUT_SEC:
        # With no readiness check on `main`, there's nowhere else to carry a wait longer than the default.
        spec["services_timeout_sec"] = _clamped_timeout_sec(wait_budget_sec)

    # The egress sidecar's policy API listens on 18080 too. If `main` or a service needs that port, move the API
    # instead. `main` counts as well: every container in the pod shares one network.
    container_ports = [port.port for port in task.main.ports]
    for service in task.services.values():
        container_ports.extend(port.port for port in service.ports)
    try:
        egress_api_port = egress_api_port_for(container_ports)
    except ValueError as error:
        raise ComposeError("compose_reserved_port", str(error), compose_path=task.compose_path) from error
    if egress_api_port is not None:
        spec["egress_api_port"] = egress_api_port

    # `ComposeServices` is the sandbox's own model, so whatever it rejects would also fail on the server. The parser
    # has already refused what Compose would, so anything rejected here is a limit of the sandbox, not of Compose.
    try:
        return ComposeServices.model_validate(spec)
    except ValidationError as error:
        raise ComposeError(
            "compose_unsupported", _validation_detail(error, list(task.services)), compose_path=task.compose_path
        ) from error


def compose_services_kwarg(
    task_dir: Path, images: Mapping[str, str] | None = None, *, read_entrypoint: bool = True
) -> dict[str, Any] | None:
    """Build the ``environment.kwargs.compose_services`` value for a task directory; None for a single-container task.

    The task Job calls this on the materialized task with the images its run prepared. From compose file to spec:

    1. :func:`find_compose_file` finds ``environment/docker-compose.yaml``.
    2. :func:`resolve_compose` merges it over Harbor's base file and fills in its variables, as Harbor's
       ``docker compose`` call would.
    3. :func:`parse_compose` reads each service, and ``main``, into a :class:`ComposeTask`, refusing what a pod
       can't run.
    4. It gives each service a role (``sidecar``, ``run_once`` or ``after_main``) and a start order.
    5. :func:`to_compose_services` maps that to ``ComposeServices``: a container per service, ``main``'s readiness
       wait, and each service's image from ``images``.
    6. The spec is returned as the plain data ``NemoOpenSandboxEnvironment`` takes.

    ``read_entrypoint=False`` leaves ``main``'s entrypoint to the compose file alone, for a caller that knows the
    image's own ``ENTRYPOINT`` from its config; otherwise it's guessed from ``main``'s Dockerfile.
    """
    task = parse_compose(task_dir, read_entrypoint=read_entrypoint)
    if task is None:
        return None
    return to_compose_services(task, images).model_dump(mode="json", by_alias=True, exclude_defaults=True)


def check_compose_task(task: ComposeTask) -> ComposeServices:
    """Map a parsed task to its spec with a placeholder for each image still to be built, to refuse it at submit.

    Raises:
        ComposeError: The task can't run as a pod.
    """
    images = {
        name: f"registry.invalid/{name}@{_PLACEHOLDER_DIGEST}"
        for name, service in task.services.items()
        if isinstance(service.source, ComposeBuild)
    }
    return to_compose_services(task, images)


def compose_provenance(services: ComposeServices) -> dict[str, str]:
    """Map each service of a spec to the image it runs, as the run's provenance records them."""
    return {service.name: service.image for service in services.services}


def _check_reserved(task: ComposeTask) -> None:
    """Raise if a container uses a port the sandbox pod needs, ``main`` shares a port with a service, or a service
    takes the name of one of the pod's containers.

    Compose gives each service its own network, so two services can listen on the same port. In the sandbox every
    container shares the pod's network, so the second one to listen on a port fails to start.
    """
    ports_by_container = {MAIN_SERVICE: task.main.ports}
    for name, service in task.services.items():
        ports_by_container[name] = service.ports

    reserved_ports = []
    for name, ports in ports_by_container.items():
        for port in ports:
            if port.port in RESERVED_PORTS:
                reserved_ports.append(f"{name}.{port.source} = {port.port}")
    if reserved_ports:
        raise ComposeError(
            "compose_reserved_port",
            "these ports are used by the sandbox pod",
            compose_path=task.compose_path,
            paths=tuple(reserved_ports),
        )

    # `ComposeServices` already refuses two services on one port, but it never sees `main`'s ports, so check those here.
    service_ports = set()
    for service in task.services.values():
        service_ports.update(port.port for port in service.ports)
    shared_ports = [
        f"{MAIN_SERVICE}.{port.source} = {port.port}" for port in task.main.ports if port.port in service_ports
    ]
    if shared_ports:
        raise ComposeError(
            "compose_unsupported",
            "`main` and a service use the same port; every container in the sandbox pod shares one network",
            compose_path=task.compose_path,
            paths=tuple(shared_ports),
        )

    if reserved_names := [name for name in task.services if name in RESERVED_NAMES]:
        raise ComposeError(
            "compose_reserved_name",
            "these names are reserved by the sandbox pod",
            compose_path=task.compose_path,
            paths=tuple(reserved_names),
        )


def _wait_budget_sec(task: ComposeTask) -> float:
    """Find the longest any readiness check may take to pass: the services' and ``main``'s, whichever is greater.

    A ``run_once`` service has no readiness check to wait for, since its dependents wait for it to finish.
    """
    service_budgets = (
        service.readiness.budget_sec
        for service in task.services.values()
        if service.readiness is not None and service.role != "run_once"
    )
    main_budget = task.main.readiness.budget_sec if task.main.readiness is not None else 0.0
    return max(main_budget, *service_budgets, 0.0)


def _clamped_timeout_sec(seconds: float) -> int:
    """Round ``seconds`` up to a whole second and keep it within the 300 to 3600 bounds ``ComposeServices`` allows."""
    return min(MAX_READY_TIMEOUT_SEC, max(MIN_READY_TIMEOUT_SEC, math.ceil(seconds)))


def _pod_volume_names(volumes: list[str], compose_path: str) -> dict[str, str]:
    """Give each compose volume a name the pod accepts: lowercase, with ``_`` and ``.`` as ``-``.

    Compose allows names like ``agent1_out``, but a pod volume's name is a Kubernetes DNS label, so ``agent1_out``
    becomes ``agent1-out``. Returns each compose name's pod name.

    Raises:
        ComposeError: Two volumes would end up with the same name, as ``data_1`` and ``data.1`` would.
    """
    names = {volume: re.sub(r"[_.]", "-", volume.lower()) for volume in volumes}
    clashing = [name for name, count in Counter(names.values()).items() if count > 1]
    if clashing:
        raise ComposeError(
            "compose_invalid",
            "these volumes would have the same name in the pod",
            compose_path=compose_path,
            paths=tuple(volume for volume, name in names.items() if name in clashing),
        )
    return names


def _mount_spec(mount: ComposeVolumeMount, volume_names: Mapping[str, str]) -> dict[str, Any]:
    """Render a volume mount as the plain data ``ComposeServices`` takes, naming its volume as the pod does."""
    return {"name": volume_names[mount.name], "mount_path": mount.mount_path, "read_only": mount.read_only}


def _main_spec(main: ComposeMainSettings, wait_budget_sec: float, volume_names: Mapping[str, str]) -> dict[str, Any]:
    """Render ``main``'s part of the spec; ``wait_budget_sec`` becomes its readiness timeout when it has a readiness check."""
    spec: dict[str, Any] = {}
    if main.entrypoint:
        spec["entrypoint"] = main.entrypoint
    if main.command:
        spec["command"] = main.command
    if main.working_dir:
        spec["working_dir"] = main.working_dir
    if main.env:
        spec["env"] = main.env
    if main.volume_mounts:
        spec["volume_mounts"] = [_mount_spec(mount, volume_names) for mount in main.volume_mounts]
    if main.add_capabilities:
        spec["add_capabilities"] = main.add_capabilities
    if main.readiness is not None:
        spec["readiness"] = {"exec": main.readiness.exec_, "timeout_sec": _clamped_timeout_sec(wait_budget_sec)}
    return spec


def _service_spec(
    name: str,
    service: ComposeService,
    images: Mapping[str, str],
    volume_names: Mapping[str, str],
    compose_path: str,
) -> dict[str, Any]:
    """Render one service as the plain data ``ComposeServices`` takes, on the image ``images`` names for it.

    Raises:
        ComposeError: The service is built from the archive and ``images`` has no image for it.
    """
    # A prepared image (pinned by digest) wins; otherwise a published image runs by its normalized tag.
    if name in images:
        image = images[name]
    elif isinstance(service.source, ComposeImage):
        image = service.source.reference
    else:
        raise ComposeError(
            "compose_build_not_prepared",
            f"service {name} is built from {service.source.context_path}; its image hasn't been built",
            compose_path=compose_path,
        )

    # Each string the pod runs or sets is escaped, since Kubernetes expands it and Compose doesn't (see `_literal`).
    spec: dict[str, Any] = {"name": name, "image": image, "role": service.role}
    if service.command is not None:
        spec["command"] = [_literal(argument) for argument in service.command]
    if service.args is not None:
        spec["args"] = [_literal(argument) for argument in service.args]
    if service.env:
        spec["env"] = {variable: _literal(value) for variable, value in service.env.items()}
    if service.ports:
        spec["ports"] = [port.port for port in service.ports]
    if service.readiness is not None and service.role != "run_once":
        spec["readiness"] = {"exec": [_literal(argument) for argument in service.readiness.exec_]}
    if service.limits or service.requests:
        spec["resources"] = {
            key: value for key, value in (("limits", service.limits), ("requests", service.requests)) if value
        }
    if service.volume_mounts:
        spec["volume_mounts"] = [_mount_spec(mount, volume_names) for mount in service.volume_mounts]
    if service.shm_size is not None:
        spec["shm_size"] = service.shm_size
    return spec


def _literal(text: str) -> str:
    """Escape ``text`` so a pod passes it on exactly as Compose would.

    Kubernetes rewrites a container's ``command``, ``args`` and ``env`` values, and its exec probes: ``$(NAME)``
    becomes that variable's value, and ``$$`` becomes ``$``. Compose does neither, so ``echo $(TOKEN)`` must print
    ``$(TOKEN)``. Doubling every ``$`` makes Kubernetes give back the original text.
    """
    return text.replace("$", "$$")


def _validation_detail(error: ValidationError, service_names: list[str]) -> str:
    """Join the spec's validation errors, naming a service by its compose name instead of its list position.

    ``services.1.ports`` becomes ``services.db.ports`` when ``db`` is the second service.
    """
    details = []
    for problem in error.errors():
        location = [
            service_names[part]
            if index == 1 and problem["loc"][0] == "services" and isinstance(part, int)
            else str(part)
            for index, part in enumerate(problem["loc"])
        ]
        details.append(f"{'.'.join(location) or 'spec'}: {problem['msg']}")
    return "; ".join(details)
