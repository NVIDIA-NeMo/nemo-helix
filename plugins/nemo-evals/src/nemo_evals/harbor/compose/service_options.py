# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read one option of a compose service (``environment``, ``ports``, ``healthcheck``, ...) as one typed value.

Every ``parse_*`` function takes the option's raw YAML value and the ``path`` it came from, such as
``services.db.ports``. A value Compose itself would refuse is recorded with :meth:`ComposeProblems.invalid`, and a
feature a pod can't run with :meth:`ComposeProblems.unsupported`. Neither raises, so one error can list every problem.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any

from nemo_evals.harbor.compose.models import ComposePort, ComposeProblems, ComposeReadiness

_CONDITIONS = ("service_started", "service_healthy", "service_completed_successfully")

# Compose's defaults for a healthcheck: ``interval`` 30 s, ``retries`` 3, ``start_period`` 0 s.
_DEFAULT_INTERVAL_SECONDS = 30.0
_DEFAULT_RETRIES = 3
_HEALTHCHECK_KEYS = frozenset({"test", "interval", "timeout", "retries", "start_period", "start_interval", "disable"})

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNIT_SECONDS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


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
    """Read a compose duration in seconds: ``"1m30s"`` is 90.0, ``"5s"`` is 5.0, and a bare number is seconds already."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

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

    # With no `test`, the healthcheck only tunes timings of one the image defines, which the pod never runs.
    if value.get("disable") is True or "test" not in value:
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
    if not text.isdigit():
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
