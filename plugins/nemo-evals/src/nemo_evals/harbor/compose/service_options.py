# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read one option of a compose service (``environment``, ``ports``, ``healthcheck``, ...) as one typed value.

Every ``parse_*`` function takes the option's raw YAML value and the ``path`` it came from, such as
``services.db.ports``. A value Compose itself would refuse is recorded with :meth:`ComposeProblems.invalid`, and a
feature a pod can't run with :meth:`ComposeProblems.unsupported`. Neither raises, so one error can list every problem.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping
from typing import Any, Literal

from nemo_evals.harbor.compose.models import ComposePort, ComposeProblems, ComposeReadiness, ComposeVolumeMount
from nhx_sandbox.harbor_opensandbox_services import quantity_value

_CONDITIONS = ("service_started", "service_healthy", "service_completed_successfully")

# Compose's defaults for a healthcheck: ``interval`` 30 s, ``retries`` 3, ``start_period`` 0 s.
_DEFAULT_INTERVAL_SECONDS = 30.0
_DEFAULT_RETRIES = 3
_HEALTHCHECK_KEYS = frozenset({"test", "interval", "timeout", "retries", "start_period", "start_interval", "disable"})

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNIT_SECONDS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}

# The strings Compose reads as a boolean, in any case: `"true"`, which a variable fills in as, is True.
_TRUE_STRINGS = frozenset({"true", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"false", "no", "n", "off"})

# Compose sizes are binary (`512m` is 512 MiB), so they map to Kubernetes' binary suffixes, not `M` or `G`.
_MEMORY_SUFFIXES = {None: "", "b": "", "k": "Ki", "kb": "Ki", "m": "Mi", "mb": "Mi", "g": "Gi", "gb": "Gi"}


def parse_boolean(value: Any, path: str, problems: ComposeProblems) -> bool | None:
    """Read a compose boolean as Compose does: ``true`` or a string like ``"yes"``; None when it isn't set."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _TRUE_STRINGS | _FALSE_STRINGS:
        return value.lower() in _TRUE_STRINGS
    problems.invalid(path, f"not a boolean: {value!r}")
    return None


def _scalar(value: Any) -> str | None:
    """Render a YAML scalar as compose passes it on (``True`` is ``"true"``, ``8`` is ``"8"``); None for any other value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def parse_environment(value: Any, path: str, problems: ComposeProblems) -> dict[str, str | None]:
    """Read ``environment`` (or a build's ``args``) as ``{name: value}`` from either of compose's forms.

    ``{A: 1}`` and ``["A=1"]`` both give ``{"A": "1"}``. A name with no value, such as ``["A"]`` or ``{A: null}``,
    gives ``{"A": None}``: Compose looks it up in the variables it interpolates with (see
    :meth:`Interpolator.variable_value`), and when none has it, the container doesn't get it, even if an ``env_file``
    set it.
    """
    if value is None:
        return {}

    # Both forms become (name, value) pairs first; a value of None means the entry named a variable but gave no value.
    pairs: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        pairs = [(str(name), entry_value) for name, entry_value in value.items()]
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            text = _scalar(entry)
            if text is None:
                problems.invalid(f"{path}[{index}]", "expected KEY=value")
                continue
            # Split at the first `=` only, so a value may itself contain `=`: "URL=http://a?b=c".
            name, equals_sign, entry_value = text.partition("=")
            pairs.append((name, entry_value if equals_sign else None))
    else:
        problems.invalid(path, "expected a mapping or a list")
        return {}

    environment: dict[str, str | None] = {}
    for name, entry_value in pairs:
        # Kept as None, so it's still looked up, and still overrides an `env_file`, once the service's files are read.
        if entry_value is None:
            environment[name] = None
            continue
        text = _scalar(entry_value)
        if text is None:
            problems.invalid(f"{path}.{name}", "expected a string, number or boolean")
        else:
            environment[name] = text
    return environment


def parse_argument_list(value: Any, path: str, problems: ComposeProblems) -> list[str] | None:
    """Read ``command`` or ``entrypoint`` as an argument list; a string is split as a shell would.

    ``"python -m app"`` and ``["python", "-m", "app"]`` both give ``["python", "-m", "app"]``.
    """
    if value is None:
        return None

    if isinstance(value, list):
        arguments = [_scalar(item) for item in value]
        if any(argument is None for argument in arguments):
            problems.invalid(path, "expected a list of strings")
            return None
        return [argument for argument in arguments if argument is not None]

    # Compose splits a string command into arguments with shell quoting rules, but runs no shell.
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError as error:
            problems.invalid(path, f"can't split the command: {error}")
            return None

    problems.invalid(path, "expected a string or a list")
    return None


def parse_entrypoint(value: Any, path: str, problems: ComposeProblems) -> list[str] | None:
    """Read ``entrypoint`` as an argument list; an empty one is unsupported, since a pod can't clear an image's entrypoint."""
    arguments = parse_argument_list(value, path, problems)
    if arguments == []:
        problems.unsupported(f"{path}: empty (it would clear the image's entrypoint)")
        return None
    return arguments


def parse_depends_on(value: Any, path: str, problems: ComposeProblems) -> dict[str, dict[str, Any]]:
    """Read ``depends_on`` as ``{service: options}``; the list form means ``service_started`` for each name.

    ``["db"]`` gives ``{"db": {"condition": "service_started"}}``.
    """
    if value is None:
        return {}
    if isinstance(value, list):
        return {str(name): {"condition": "service_started"} for name in value}
    if not isinstance(value, dict):
        problems.invalid(path, "expected a list or a mapping")
        return {}

    dependencies: dict[str, dict[str, Any]] = {}
    for name, options in value.items():
        if options is not None and not isinstance(options, dict):
            problems.invalid(f"{path}.{name}", "expected a mapping such as {condition: service_healthy}")
            continue
        dependencies[str(name)] = dict(options or {})
    return dependencies


def parse_dependency_conditions(value: Any, path: str, problems: ComposeProblems) -> dict[str, str]:
    """Read ``depends_on`` as ``{service: condition}``, checking that each condition is one compose knows.

    ``{db: {condition: service_healthy}}`` gives ``{"db": "service_healthy"}``.
    """
    conditions: dict[str, str] = {}
    for name, options in parse_depends_on(value, path, problems).items():
        # A dependency without a condition only waits for the container to start, as in Compose.
        condition = options.get("condition", "service_started")
        if condition not in _CONDITIONS:
            problems.invalid(f"{path}.{name}", f"unknown condition {condition!r}")
        conditions[name] = str(condition)
    return conditions


def _duration_seconds(value: Any, default: float, path: str, problems: ComposeProblems) -> float:
    """Read a compose duration in seconds: ``"1m30s"`` is 90.0 and ``"5s"`` is 5.0.

    Compose requires a string, so a bare number (``5``, ``.inf``) is invalid, as it is there.
    """
    if value is None:
        return default

    if isinstance(value, str):
        text = value.replace(" ", "")
        parts = _DURATION_PART.findall(text)
        # `findall` skips text it can't match, so the parts must rebuild the whole string; "5sx" is not a duration.
        if parts and "".join(f"{number}{unit}" for number, unit in parts) == text:
            return sum(float(number) * _DURATION_UNIT_SECONDS[unit] for number, unit in parts)

    problems.invalid(path, f"not a duration: {value!r}")
    return default


def _healthcheck_command(test: Any, path: str, problems: ComposeProblems) -> list[str] | None:
    """Read the command a healthcheck ``test`` runs; None when it runs nothing (``NONE``) or isn't valid.

    ``"curl -f localhost"`` and ``["CMD-SHELL", "curl -f localhost"]`` both run ``sh -c "curl -f localhost"``;
    ``["CMD", "pg_isready"]`` runs ``pg_isready`` directly.
    """
    if isinstance(test, str):
        return ["sh", "-c", test]

    if not isinstance(test, list) or not test:
        problems.invalid(path, "expected a string or a list")
        return None

    parts = [_scalar(part) for part in test]
    if any(part is None for part in parts):
        problems.invalid(path, "expected a list of strings")
        return None

    # The first item says how to run the rest: `CMD` runs them as a command, `CMD-SHELL` runs one string through
    # `sh -c`, and `NONE` turns off a healthcheck the image defines.
    run_mode, arguments = parts[0], [part for part in parts[1:] if part is not None]
    if run_mode == "NONE":
        return None
    if run_mode == "CMD" and arguments:
        return arguments
    if run_mode == "CMD-SHELL" and len(arguments) == 1:
        return ["sh", "-c", arguments[0]]

    problems.invalid(path, "expected CMD, CMD-SHELL or NONE first, with a command after it")
    return None


def parse_readiness(value: Any, path: str, problems: ComposeProblems) -> ComposeReadiness | None:
    """Read a ``healthcheck`` as a readiness command and its wait budget; None when it declares no command.

    The budget is the longest Compose would wait for the service to turn healthy, ``start_period + interval × retries``.
    ``{test: [CMD, pg_isready], interval: 5s, retries: 10}`` gives a 50 second budget.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        problems.invalid(path, "expected a mapping")
        return None

    for key in value:
        if key not in _HEALTHCHECK_KEYS:
            problems.unsupported(f"{path}.{key}")

    if value.get("disable") is True:
        return None
    # With no `test`, the healthcheck only tunes the image's own `HEALTHCHECK`, which nothing in the pod runs.
    if "test" not in value:
        problems.unsupported(f"{path}: no `test` (it would tune the image's own HEALTHCHECK)")
        return None

    command = _healthcheck_command(value["test"], f"{path}.test", problems)
    if command is None:
        return None

    # Compose waits up to `start_period + interval * retries` for a service to turn healthy.
    start_period = _duration_seconds(value.get("start_period"), 0.0, f"{path}.start_period", problems)
    interval = _duration_seconds(value.get("interval"), _DEFAULT_INTERVAL_SECONDS, f"{path}.interval", problems)
    retries = value.get("retries", _DEFAULT_RETRIES)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        problems.invalid(f"{path}.retries", "expected a whole number")
        retries = _DEFAULT_RETRIES
    return ComposeReadiness(exec=command, budget_sec=start_period + interval * retries)


def _container_port(value: Any, path: str, problems: ComposeProblems) -> int | None:
    """Read the container side of one ``ports`` or ``expose`` entry: ``"8080:80/tcp"``, ``80`` and ``{target: 80}`` are all 80."""
    if isinstance(value, dict):
        value = value.get("target")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        problems.invalid(path, "expected a port")
        return None

    # Only the container side matters in a pod: drop the protocol (`/tcp`), then any host IP and host port before it.
    text = str(value).split("/", 1)[0].rsplit(":", 1)[-1]
    # `isdigit` alone also takes digits like `²`, which `int` can't read.
    if not (text.isascii() and text.isdigit()):
        problems.unsupported(f"{path}: port ranges and names")
        return None
    return int(text)


def parse_ports(raw_service: Mapping[str, Any], path: str, problems: ComposeProblems) -> list[ComposePort]:
    """Read the container ports a service ``expose``s or publishes with ``ports``, each once.

    A port found twice keeps the first key that named it.
    """
    ports: list[ComposePort] = []
    for key in ("expose", "ports"):
        entries = raw_service.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            problems.invalid(f"{path}.{key}", "expected a list")
            continue

        for index, entry in enumerate(entries):
            port = _container_port(entry, f"{path}.{key}[{index}]", problems)
            # Exposing and publishing the same port is common and harmless; only clashes between services matter.
            if port is not None and all(existing.port != port for existing in ports):
                ports.append(ComposePort(port=port, source=f"{key}[{index}]"))
    return ports


def parse_memory_quantity(
    value: Any, path: str, problems: ComposeProblems, *, strings_only: bool = False
) -> str | None:
    """Read a compose size as a Kubernetes quantity: ``"512m"`` is ``"512Mi"``, ``"1gb"`` is ``"1Gi"``, ``1024`` is ``"1024"``.

    ``strings_only`` is for ``deploy.resources``, where Compose refuses a number.
    """
    if isinstance(value, bool) or value is None:
        return None
    if strings_only and isinstance(value, (int, float)):
        problems.invalid(path, "must be a string")
        return None
    # Compose ignores a float `mem_limit` or `shm_size` (`2.0`, `.inf` and `.nan` alike) as if it weren't set.
    if isinstance(value, float):
        return None

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(b|k|kb|m|mb|g|gb)?", str(value).strip().lower())
    if match is None:
        problems.invalid(path, f"not a size: {value!r}")
        return None

    number, unit = match.groups()
    return f"{number}{_MEMORY_SUFFIXES[unit]}"


def _cpu_quantity(value: Any, path: str, problems: ComposeProblems) -> str | None:
    """Read a compose CPU count as a Kubernetes quantity; a whole number drops its decimal point (``2.0`` is ``"2"``)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        problems.invalid(path, "expected a number")
        return None

    # `float()` also accepts "inf", "nan" and "1e3", none of which Compose takes as a CPU count.
    if isinstance(value, str) and not re.fullmatch(r"(?:\d+\.?\d*|\.\d+)", value.strip()):
        problems.invalid(path, f"not a number: {value!r}")
        return None

    try:
        number = float(value)
    # An integer too large for a float, which Compose refuses as out of range, raises OverflowError.
    except (ValueError, OverflowError):
        problems.invalid(path, f"not a number: {value!r}")
        return None
    if not math.isfinite(number) or number <= 0:
        problems.invalid(path, f"not a positive number: {value!r}")
        return None
    return str(int(number)) if number == int(number) else str(number)


def _resource_section(value: Any, path: str, problems: ComposeProblems) -> dict[str, str]:
    """Read one ``deploy.resources`` section (``limits`` or ``reservations``) as Kubernetes ``cpu`` and ``memory``.

    ``{cpus: "0.5", memory: 512M}`` gives ``{"cpu": "0.5", "memory": "512Mi"}``.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.invalid(path, "expected a mapping")
        return {}

    # Kubernetes sizes a container by CPU and memory only; `pids` and `devices` (such as GPUs) have no equivalent here.
    for key in value:
        if key not in ("cpus", "memory"):
            problems.unsupported(f"{path}.{key}")

    quantities: dict[str, str] = {}
    if (cpu := _cpu_quantity(value.get("cpus"), f"{path}.cpus", problems)) is not None:
        quantities["cpu"] = cpu
    if (
        memory := parse_memory_quantity(value.get("memory"), f"{path}.memory", problems, strings_only=True)
    ) is not None:
        quantities["memory"] = memory
    return quantities


def parse_resources(
    raw_service: Mapping[str, Any], path: str, problems: ComposeProblems
) -> tuple[dict[str, str], dict[str, str]]:
    """Read a service's resources as ``(limits, requests)`` from ``cpus``, ``mem_limit`` and ``deploy.resources``.

    ``cpus`` and ``mem_limit`` are limits. ``deploy.resources.limits`` adds to them, and
    ``deploy.resources.reservations`` become the requests.
    """
    limits: dict[str, str] = {}
    requests: dict[str, str] = {}
    if (cpu := _cpu_quantity(raw_service.get("cpus"), f"{path}.cpus", problems)) is not None:
        limits["cpu"] = cpu
    if (memory := parse_memory_quantity(raw_service.get("mem_limit"), f"{path}.mem_limit", problems)) is not None:
        limits["memory"] = memory

    deploy = raw_service.get("deploy")
    if deploy is None:
        return limits, requests
    if not isinstance(deploy, dict):
        problems.invalid(f"{path}.deploy", "expected a mapping")
        return limits, requests

    # The rest of `deploy` (replicas, placement, restart policy) is for Swarm and has no meaning in one pod.
    for key in deploy:
        if key != "resources":
            problems.unsupported(f"{path}.deploy.{key}")

    resources = deploy.get("resources")
    if resources is not None and not isinstance(resources, dict):
        problems.invalid(f"{path}.deploy.resources", "expected a mapping")
        return limits, requests

    resources = resources or {}
    deploy_limits = _resource_section(resources.get("limits"), f"{path}.deploy.resources.limits", problems)
    # Compose refuses a `cpus` or `mem_limit` that differs from `deploy.resources.limits`; the same value twice is fine.
    for resource, key, deploy_key in (("cpu", "cpus", "cpus"), ("memory", "mem_limit", "memory")):
        limit, deploy_limit = limits.get(resource), deploy_limits.get(resource)
        if limit is not None and deploy_limit is not None and quantity_value(limit) != quantity_value(deploy_limit):
            problems.invalid(
                f"{path}.{key}", f"can't set distinct values on '{key}' and 'deploy.resources.limits.{deploy_key}'"
            )
    limits.update(deploy_limits)
    requests.update(_resource_section(resources.get("reservations"), f"{path}.deploy.resources.reservations", problems))
    return limits, requests


def _mount_entry(
    entry: Any, path: str, declared_volumes: set[str], problems: ComposeProblems
) -> ComposeVolumeMount | None:
    """Read one ``volumes`` entry as a mount of a declared named volume; bind mounts and anonymous volumes are refused.

    ``"data:/var/lib/db:ro"`` and ``{type: volume, source: data, target: /var/lib/db, read_only: true}`` both
    mount ``data`` read-only; ``"./data:/db"`` is a bind mount and ``"/db"`` an anonymous volume.
    """
    source: Any
    target: Any
    read_only = False

    # Short form `source:target[:ro]`, or long form `{type, source, target, read_only}`.
    if isinstance(entry, str):
        parts = entry.split(":")
        # A lone path, such as "/data", asks Compose for a fresh unnamed volume there.
        if len(parts) == 1:
            problems.unsupported(f"{path}: anonymous volume")
            return None
        # The third part is a comma-separated list of options, such as "ro,z"; only `ro` changes anything here.
        source, target = parts[0], parts[1]
        read_only = len(parts) > 2 and "ro" in parts[2].split(",")
    elif isinstance(entry, dict):
        if entry.get("type", "volume") != "volume":
            problems.unsupported(f"{path}: {entry.get('type')} mount")
            return None
        source, target = entry.get("source"), entry.get("target")
        read_only = bool(parse_boolean(entry.get("read_only"), f"{path}.read_only", problems))
        # A pod's volume starts empty, as `nocopy` asks; any other option, such as `subpath`, would mount something else.
        options = entry.get("volume") or {}
        if not isinstance(options, dict):
            problems.invalid(f"{path}.volume", "expected a mapping")
        else:
            for option in options:
                if option != "nocopy":
                    problems.unsupported(f"{path}.volume.{option}")
    else:
        problems.invalid(path, "expected a string or a mapping")
        return None

    # What a pod can't do: a volume with no name, a host path, or a volume the file never declared.
    if not isinstance(source, str) or not isinstance(target, str) or not source:
        problems.unsupported(f"{path}: anonymous volume")
    elif source.startswith((".", "/", "~")):
        problems.unsupported(f"{path}: bind mount {source}")
    elif source not in declared_volumes:
        problems.invalid(path, f"volume {source!r} is not declared under the top-level volumes")
    else:
        return ComposeVolumeMount(name=source, mount_path=target, read_only=read_only)
    return None


def parse_volume_mounts(
    value: Any, path: str, declared_volumes: set[str], problems: ComposeProblems
) -> list[ComposeVolumeMount]:
    """Read a service's ``volumes`` as mounts of the top-level named volumes in ``declared_volumes``."""
    if value is None:
        return []
    if not isinstance(value, list):
        problems.invalid(path, "expected a list")
        return []

    mounts = (_mount_entry(entry, f"{path}[{index}]", declared_volumes, problems) for index, entry in enumerate(value))
    return [mount for mount in mounts if mount is not None]


def parse_platform(value: Any, path: str, problems: ComposeProblems) -> str | None:
    """Read a service's ``platform``, such as ``linux/amd64``, as written; None when it names none.

    Image planning pins the service's image to it, so the image is built or pulled for the platform Compose would
    use. Planning also checks that it names a real platform. An empty value, as ``platform: ${PLATFORM}`` gives
    when the variable is unset, names none: Compose then uses its default platform.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        problems.invalid(path, "expected a platform such as linux/amd64")
        return None
    return value or None


def parse_working_dir(value: Any, path: str, problems: ComposeProblems) -> str | None:
    """Read ``working_dir`` as an absolute path in the container; None when it sets none.

    A relative one would be resolved against the image's ``WORKDIR``, which isn't known here, so it's unsupported.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        problems.invalid(path, "expected a path")
        return None
    if not value.startswith("/"):
        problems.unsupported(f"{path}: a relative working directory")
        return None
    return value


def parse_capabilities(
    raw_service: Mapping[str, Any], path: str, problems: ComposeProblems
) -> list[Literal["SYS_PTRACE"]]:
    """Read the capabilities ``main`` adds with ``cap_add``; only ``SYS_PTRACE`` is supported."""
    capabilities = raw_service.get("cap_add")
    if capabilities is None:
        return []
    if not isinstance(capabilities, list):
        problems.invalid(f"{path}.cap_add", "expected a list")
        return []

    # SYS_PTRACE only lets `main` debug its own processes. Anything broader, such as NET_ADMIN, could let the agent
    # rewrite the pod's network rules and get around the egress policy.
    for capability in capabilities:
        if capability != "SYS_PTRACE":
            problems.unsupported(f"{path}.cap_add: {capability}")
    return ["SYS_PTRACE"] if "SYS_PTRACE" in capabilities else []
