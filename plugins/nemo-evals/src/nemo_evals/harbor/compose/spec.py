# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a parsed :class:`ComposeTask` to the ``ComposeServices`` spec that ``NemoOpenSandboxEnvironment`` takes."""

from __future__ import annotations

import math
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
)
from nemo_evals.harbor.compose.parser import parse_compose
from nhx_sandbox.harbor_opensandbox_services import RESERVED_NAMES, RESERVED_PORTS, ComposeServices
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
    container in the sandbox pod. The task Job passes the result to ``NemoOpenSandboxEnvironment`` as
    ``compose_services``, and submit checks it early through :func:`check_compose_task`.

    Args:
        task: The parsed compose file.
        images: A full image reference per service, such as a digest-pinned one. A published image without an
            entry runs by its normalized tag; a built service needs one.

    Raises:
        ComposeError: The file defines only ``main``, a service has no image, or the spec breaks a rule of
            ``ComposeServices`` (``compose_unsupported``: the parser has already refused what Compose itself would).
    """
    if not task.services:
        raise ComposeError(
            "compose_override_only",
            "the file defines only `main`, which Harbor already provides; there are no services to run",
            compose_path=task.compose_path,
        )
    _check_reserved(task)

    # One wait budget covers every service, since the sandbox waits for them together.
    wait_budget_sec = _wait_budget_sec(task)
    spec: dict[str, Any] = {
        "services": [
            _service_spec(name, service, images or {}, task.compose_path) for name, service in task.services.items()
        ],
        "main": _main_spec(task.main, wait_budget_sec),
    }

    # `ComposeServices` is the sandbox's own model, so whatever it rejects would also fail on the server. The parser
    # has already refused what Compose would, so anything rejected here is a limit of the sandbox, not of Compose.
    try:
        return ComposeServices.model_validate(spec)
    except ValidationError as error:
        raise ComposeError(
            "compose_unsupported", _validation_detail(error, list(task.services)), compose_path=task.compose_path
        ) from error


def compose_services_kwarg(task_dir: Path, images: Mapping[str, str] | None = None) -> dict[str, Any] | None:
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
    """
    task = parse_compose(task_dir)
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
    """Raise if a service exposes a port the sandbox pod uses, or takes the name of one of the pod's containers."""
    reserved_ports = [
        f"{name}.{port.source} = {port.port}"
        for name, service in task.services.items()
        for port in service.ports
        if port.port in RESERVED_PORTS
    ]
    if reserved_ports:
        raise ComposeError(
            "compose_reserved_port",
            "these ports are used by the sandbox pod",
            compose_path=task.compose_path,
            paths=tuple(reserved_ports),
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


def _main_spec(main: ComposeMainSettings, wait_budget_sec: float) -> dict[str, Any]:
    """Render ``main``'s part of the spec; ``wait_budget_sec`` becomes its readiness timeout when it has a readiness check."""
    spec: dict[str, Any] = {}
    if main.entrypoint:
        spec["entrypoint"] = main.entrypoint
    if main.env:
        spec["env"] = main.env
    if main.readiness is not None:
        spec["readiness"] = {"exec": main.readiness.exec_, "timeout_sec": _clamped_timeout_sec(wait_budget_sec)}
    return spec


def _service_spec(name: str, service: ComposeService, images: Mapping[str, str], compose_path: str) -> dict[str, Any]:
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
