# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a Harbor task's ``compose_services`` from the ``docker-compose.yaml`` in its archive.

A compose task runs helper services (a database, a mock API) next to ``main``. OpenSandbox can't run
Compose, so ``NemoOpenSandboxEnvironment`` takes the services as a ``ComposeServices`` spec and the
OpenSandbox server's extension adds them to the sandbox pod as extra containers. This module builds
that spec from the task itself, in two steps:

* :func:`parse_compose` reads the task directory the way Harbor's Docker path does: the task's
  ``environment/docker-compose.yaml`` laid over Harbor's own base file, with compose's interpolation
  applied. It returns a :class:`ComposeTask`, which says where each service's image comes from and
  how it runs, without needing a registry or a sandbox.
* :func:`to_compose_services` maps a :class:`ComposeTask` to ``ComposeServices``, with the image
  references supplied by the caller.

The compose file is task content, not trusted input. It is parsed with ``yaml.safe_load`` under a size
cap and a cap on how many values its aliases may expand to, and it is never executed. Anything the
sandbox would run differently from Compose is rejected with a typed :class:`ComposeError` that lists
every offending path, rather than being dropped silently.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
import shlex
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from harbor.environments.docker import COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH
from nemo_builder_plugin.identity import ImageIdentityError, normalize_reference
from nhx_sandbox.harbor_opensandbox_services import (
    COMPOSE_FILENAMES,
    MAIN_SERVICE,
    RESERVED_NAMES,
    RESERVED_PORTS,
    ComposeServices,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# The only compose file name Harbor reads, in the task's ``environment/`` directory.
COMPOSE_FILENAME = "docker-compose.yaml"
ENVIRONMENT_DIR = "environment"

# The size of a compose file, ``.env`` file or ``env_file``, and how many values the compose file may expand to
# once YAML aliases are resolved. Both bound the work a hostile file can cause.
MAX_COMPOSE_BYTES = 256 * 1024
MAX_EXPANDED_VALUES = 50_000

# Compose's defaults for a healthcheck: ``interval`` 30 s, ``retries`` 3, ``start_period`` 0 s.
_DEFAULT_INTERVAL_SEC = 30.0
_DEFAULT_RETRIES = 3

# The bounds of ``MainReadiness.timeout_sec`` in ``ComposeServices``.
MIN_READY_TIMEOUT_SEC = 300
MAX_READY_TIMEOUT_SEC = 3600

# Names that image planning gives to images besides ``main``'s, so a service can't take them.
_IMAGE_ROLE_NAMES = frozenset({"verifier"})

_CONDITIONS = ("service_started", "service_healthy", "service_completed_successfully")
_COMPLETED = "service_completed_successfully"

# Service keys that a pod already behaves as, or that images prebuilt elsewhere make moot.
_NO_EFFECT_KEYS = frozenset({"pull_policy", "restart", "container_name", "hostname", "labels", "logging"})
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
        "volumes",
        "shm_size",
        "cpus",
        "mem_limit",
        "deploy",
        "network_mode",
        "networks",
    }
)
_MAIN_KEYS = frozenset(
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
        "cap_add",
        "network_mode",
        "networks",
    }
)
_BUILD_KEYS = frozenset({"context", "dockerfile", "args", "target", "no_cache"})
_HEALTHCHECK_KEYS = frozenset({"test", "interval", "timeout", "retries", "start_period", "start_interval", "disable"})
_TOP_LEVEL_KEYS = frozenset({"services", "volumes", "networks", "version", "name"})

# Compose merges these sequences by appending; every other sequence is replaced.
_APPENDED_KEYS = frozenset({"expose", "ports", "cap_add", "env_file"})


class ComposeError(ValueError):
    """A task whose compose file can't run as services in a sandbox. ``code`` is stable and shown to users."""

    def __init__(self, code: str, detail: str, *, compose_path: str | None = None, paths: tuple[str, ...] = ()) -> None:
        """Build the message ``<code>: <detail> (<paths>) in <compose_path>``, and keep each part on the error."""
        message = f"{code}: {detail}"
        if paths:
            message += f" ({', '.join(paths)})"
        if compose_path:
            message += f" in {compose_path}"
        super().__init__(message)

        self.code = code
        self.detail = detail
        self.compose_path = compose_path
        self.paths = paths


class ComposeImage(BaseModel):
    """A service that runs a published image, as ``image: postgres:18``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["image"] = "image"
    ref: str = Field(description="The full reference, as Docker would pull it: `docker.io/library/postgres:18`.")


class ComposeBuild(BaseModel):
    """A service built from the archive, as ``build: {context: ./api}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["build"] = "build"
    context_path: str = Field(description="Inside the task directory, and under `environment/`.")
    dockerfile: str = Field(default="Dockerfile", description="Relative to `context_path`.")
    args: dict[str, str] = Field(default_factory=dict)
    target: str | None = None


ComposeImageSource = Annotated[ComposeImage | ComposeBuild, Field(discriminator="kind")]


class ComposePort(BaseModel):
    """A container port, and the compose key it came from (for error messages)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int
    source: str


class ComposeReadiness(BaseModel):
    """A healthcheck command, and the longest Compose would let the service take to become healthy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exec_: list[str] = Field(alias="exec", min_length=1)
    budget_sec: float = Field(description="`start_period + interval * retries`.")


class ComposeVolumeMount(BaseModel):
    """A named volume mounted into a container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    mount_path: str
    read_only: bool = False


class ComposeService(BaseModel):
    """One service other than ``main``: its image source, how it starts, and how it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source: ComposeImageSource
    role: Literal["sidecar", "run_once", "after_main"] = "sidecar"
    depends_on: dict[str, str] = Field(
        default_factory=dict, description="Each service this one waits for, `main` included, and the compose condition."
    )
    command: list[str] | None = Field(default=None, description="Compose `entrypoint`; Kubernetes calls it `command`.")
    args: list[str] | None = Field(default=None, description="Compose `command`; Kubernetes calls it `args`.")
    env: dict[str, str] = Field(default_factory=dict)
    ports: list[ComposePort] = Field(default_factory=list)
    readiness: ComposeReadiness | None = None
    limits: dict[str, str] = Field(default_factory=dict)
    requests: dict[str, str] = Field(default_factory=dict)
    volume_mounts: list[ComposeVolumeMount] = Field(default_factory=list)
    shm_size: str | None = None


class ComposeMainSettings(BaseModel):
    """What the task's compose file and Dockerfile say about ``main``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    depends_on: dict[str, str] = Field(
        default_factory=dict, description="Each service `main` waits for, and the condition."
    )
    env: dict[str, str] = Field(default_factory=dict)
    readiness: ComposeReadiness | None = None
    add_capabilities: list[Literal["SYS_PTRACE"]] = Field(default_factory=list)
    volume_mounts: list[ComposeVolumeMount] = Field(default_factory=list)
    entrypoint: list[str] | None = Field(
        default=None, description="The last exec-form `ENTRYPOINT` of `main`'s Dockerfile, or the compose override."
    )


class ComposeTask(BaseModel):
    """A task's compose file, merged with Harbor's base and interpolated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compose_path: str = Field(description="`<task>/environment/docker-compose.yaml`.")
    main_dockerfile: str | None = Field(
        description=(
            "`main`'s Dockerfile, relative to `environment/`: `Dockerfile` unless the task's `main.build` says "
            "otherwise. None when the task runs a prebuilt image."
        )
    )
    main_build: ComposeBuild | None = Field(
        default=None,
        description="What builds `main`: its context and Dockerfile. None when the task runs a prebuilt image.",
    )
    main_image: str | None = Field(
        default=None,
        description=(
            "The image `main` runs, for a prebuilt task: `task.toml`'s `docker_image` unless `main.image` replaces it."
        ),
    )
    services: dict[str, ComposeService] = Field(description="By name, in start order.")
    main: ComposeMainSettings
    volumes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------------
# Problems


class _Problems:
    """What a compose file does that sandboxes can't, collected so one error lists all of it."""

    def __init__(self) -> None:
        """Start with nothing found."""
        self._unsupported: list[str] = []
        self._invalid: list[str] = []

    def unsupported(self, path: str) -> None:
        """Record a feature at ``path`` that Compose has and a pod doesn't."""
        self._unsupported.append(path)

    def invalid(self, path: str, why: str) -> None:
        """Record a value at ``path`` that Compose itself would refuse, and why."""
        self._invalid.append(f"{path}: {why}")

    def raise_any(self, compose_path: str) -> None:
        """Raise one error for everything recorded, invalid values first; do nothing when nothing was."""
        if self._invalid:
            raise ComposeError("compose_invalid", "; ".join(self._invalid), compose_path=compose_path)
        if self._unsupported:
            raise ComposeError(
                "compose_unsupported",
                "these can't run as sandbox services",
                compose_path=compose_path,
                paths=tuple(dict.fromkeys(self._unsupported)),
            )


# ---------------------------------------------------------------------------------------------------------------------
# Interpolation

# A bare ``$NAME``, and the inside of ``${NAME}``, ``${NAME:-fallback}`` or ``${NAME-fallback}``.
_NAMED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BRACED = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])(.*))?", re.DOTALL)


def _closing_brace(text: str, start: int) -> int | None:
    """The index of the ``}`` that closes a ``${`` whose body starts at ``start``, allowing nested ``${...}``."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            if depth == 0:
                return index
            depth -= 1
    return None


def interpolate(text: str, variables: Mapping[str, str], *, path: str, problems: _Problems) -> str:
    """Apply compose's variable substitution to one string.

    ``$$`` is a literal ``$``. ``$VAR`` and ``${VAR}`` take a value from ``variables``; ``${VAR:-d}`` and
    ``${VAR-d}`` fall back to ``d``, which may itself use variables. A variable with no value is reported
    against ``path``, since Harbor would read it from the host's environment, which a sandbox doesn't have.
    A ``$`` followed by anything else stays as written.
    """
    pieces: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "$":
            pieces.append(text[index])
            index += 1
            continue

        next_char = text[index + 1 : index + 2]
        if next_char == "$":
            pieces.append("$")
            index += 2
        elif next_char == "{" and (end := _closing_brace(text, index + 2)) is not None:
            pieces.append(_expand_braced(text[index + 2 : end], text[index : end + 1], variables, path, problems))
            index = end + 1
        elif next_char != "{" and (bare_name := _NAMED.match(text, index + 1)) is not None:
            pieces.append(_lookup(bare_name.group(0), variables, path, problems, text[index : bare_name.end()]))
            index = bare_name.end()
        else:
            pieces.append("$")
            index += 1
    return "".join(pieces)


def _lookup(name: str, variables: Mapping[str, str], path: str, problems: _Problems, written: str) -> str:
    """The value of ``name``; when it has none, record it against ``path`` and keep the text as ``written``."""
    value = variables.get(name)
    if value is None:
        problems.unsupported(f"{path}: interpolation: {name}")
        return written
    return value


def _expand_braced(body: str, written: str, variables: Mapping[str, str], path: str, problems: _Problems) -> str:
    """Expand the inside of one ``${...}``; text that isn't a variable reference stays as ``written``."""
    match = _BRACED.fullmatch(body)
    if match is None:
        return written

    name, operator, fallback = match.groups()
    value = variables.get(name)
    if operator in ("?", ":?", "+", ":+"):
        problems.unsupported(f"{path}: interpolation operator {operator!r} on {name}")
        return written

    if (operator == ":-" and not value) or (operator == "-" and value is None):
        return interpolate(fallback, variables, path=path, problems=problems)
    return _lookup(name, variables, path, problems, written)


def _interpolate_tree(node: Any, variables: Mapping[str, str], path: str, problems: _Problems) -> Any:
    """``node`` with every string interpolated; ``path`` names where each string sits, for error messages."""
    if isinstance(node, str):
        return interpolate(node, variables, path=path, problems=problems)
    if isinstance(node, dict):
        return {
            key: _interpolate_tree(value, variables, f"{path}.{key}" if path else str(key), problems)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_interpolate_tree(value, variables, f"{path}[{index}]", problems) for index, value in enumerate(node)]
    return node


# ---------------------------------------------------------------------------------------------------------------------
# Reading the task


def _expanded_size(node: Any, sizes: dict[int, int]) -> int:
    """How many values ``node`` holds once YAML aliases are resolved, counting a shared value once per use.

    ``sizes`` remembers each container's size by identity, so a file of nested aliases is counted in time
    proportional to its length, not to the size it would expand to.
    """
    if not isinstance(node, (dict, list)):
        return 1

    key = id(node)
    if key not in sizes:
        children = node.values() if isinstance(node, dict) else node
        sizes[key] = 1 + sum(_expanded_size(child, sizes) for child in children)
    return sizes[key]


def _read_yaml(path: Path, compose_path: str) -> dict[str, Any]:
    """A YAML mapping from ``path``, refusing a file that is too large, malformed, or expands to too many values."""
    try:
        size = path.stat().st_size
        if size > MAX_COMPOSE_BYTES:
            raise ComposeError(
                "compose_invalid",
                f"the file is {size} bytes; the limit is {MAX_COMPOSE_BYTES}",
                compose_path=compose_path,
            )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))

        if _expanded_size(document, {}) > MAX_EXPANDED_VALUES:
            raise ComposeError(
                "compose_invalid",
                f"the file expands to more than {MAX_EXPANDED_VALUES} values once its aliases are resolved",
                compose_path=compose_path,
            )
    except (OSError, UnicodeDecodeError) as exc:
        raise ComposeError("compose_invalid", f"can't read the file: {exc}", compose_path=compose_path) from exc
    except yaml.YAMLError as exc:
        raise ComposeError("compose_invalid", f"not valid YAML: {exc}", compose_path=compose_path) from exc
    except RecursionError as exc:
        raise ComposeError("compose_invalid", "the file is nested too deeply", compose_path=compose_path) from exc

    if not isinstance(document, dict):
        raise ComposeError("compose_invalid", "not a YAML mapping", compose_path=compose_path)
    return document


def _read_data_file(path: Path, label: str) -> str:
    """The text of a data file from the archive, refusing a symlink, an oversized file or bytes that aren't UTF-8.

    Raises:
        ComposeError: The file can't be read as data; its detail names ``label``.
    """
    if path.is_symlink():
        raise ComposeError("compose_invalid", f"{label} is a symlink")

    try:
        if path.stat().st_size > MAX_COMPOSE_BYTES:
            raise ComposeError("compose_invalid", f"{label} is over the {MAX_COMPOSE_BYTES} byte limit")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ComposeError("compose_invalid", f"can't read {label}: {exc}") from exc


def _task_environment(task_dir: Path) -> dict[str, Any]:
    """The ``[environment]`` table of ``task.toml``; empty when the task has none."""
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return {}

    try:
        environment = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("environment", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ComposeError("compose_invalid", f"can't read task.toml: {exc}") from exc
    return environment if isinstance(environment, dict) else {}


def _project_dotenv(task_dir: Path) -> dict[str, str]:
    """``environment/.env``, which compose reads for interpolation from Harbor's project directory.

    The path is fixed and the file is read as data, never run.
    """
    file = task_dir / ENVIRONMENT_DIR / ".env"
    if not file.is_file() and not file.is_symlink():
        return {}

    compose_path = f"{task_dir.name}/{ENVIRONMENT_DIR}/.env"
    try:
        return _dotenv(_read_data_file(file, "environment/.env"))
    except ComposeError as exc:
        raise ComposeError(exc.code, exc.detail, compose_path=compose_path) from exc


def _infra_variables(task_dir: Path, environment: Mapping[str, Any]) -> dict[str, str]:
    """The variables compose interpolates with, lowest priority first.

    ``environment/.env``, then the task's own ``[environment] env``, then the variables Harbor's Docker path sets.
    The host's environment is never read: a sandbox doesn't have it, and the dispatcher's must not leak into a spec.
    """
    variables = _project_dotenv(task_dir)

    task_env = environment.get("env")
    if isinstance(task_env, dict):
        variables.update({str(key): str(value) for key, value in task_env.items()})

    variables.update({"CONTEXT_DIR": ".", "MAIN_IMAGE_NAME": f"hb__{task_dir.name}"})
    if environment.get("docker_image"):
        variables["PREBUILT_IMAGE_NAME"] = str(environment["docker_image"])
    if environment.get("cpus") is not None:
        variables["CPUS"] = str(environment["cpus"])
    if environment.get("memory_mb") is not None:
        variables["MEMORY"] = f"{environment['memory_mb']}M"
    return variables


# ---------------------------------------------------------------------------------------------------------------------
# Merging with Harbor's base file


def _merge(base: Any, over: Any, key: str = "") -> Any:
    """Lay ``over`` on ``base`` as compose merges files: mappings merge, appended sequences append, the rest replace."""
    if isinstance(base, dict) and isinstance(over, dict):
        merged = dict(base)
        for name, value in over.items():
            merged[name] = _merge(base[name], value, name) if name in base else value
        return merged

    if isinstance(base, list) and isinstance(over, list) and key in _APPENDED_KEYS:
        return [*base, *(item for item in over if item not in base)]
    return over


def _depends_on_mapping(value: Any, path: str, problems: _Problems) -> dict[str, dict[str, Any]]:
    """``depends_on`` as a mapping of service to options; the list form means ``service_started``."""
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


def _conditions(value: Any, path: str, problems: _Problems) -> dict[str, str]:
    """``depends_on`` as service to condition, checking each condition is one compose knows."""
    conditions: dict[str, str] = {}
    for name, options in _depends_on_mapping(value, path, problems).items():
        condition = options.get("condition", "service_started")
        if condition not in _CONDITIONS:
            problems.invalid(f"{path}.{name}", f"unknown condition {condition!r}")
        conditions[name] = str(condition)
    return conditions


def _normalize_service(raw: Any, path: str, problems: _Problems) -> dict[str, Any]:
    """A service as a mapping with ``environment`` and ``depends_on`` in mapping form, so they merge by key."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        problems.invalid(path, "expected a mapping")
        return {}

    service = dict(raw)
    if "depends_on" in service:
        service["depends_on"] = _depends_on_mapping(service["depends_on"], f"{path}.depends_on", problems)
    if "environment" in service:
        service["environment"] = _environment(service["environment"], f"{path}.environment", problems)
    if isinstance(service.get("build"), str):
        service["build"] = {"context": service["build"]}
    return service


def _load_services(document: Mapping[str, Any], problems: _Problems, *, what: str) -> dict[str, dict[str, Any]]:
    """The ``services`` of a compose document, each in mapping form; ``what`` names the document in errors."""
    services = document.get("services")
    if not isinstance(services, dict):
        raise ComposeError("compose_invalid", f"{what} has no `services` mapping")
    return {str(name): _normalize_service(raw, f"services.{name}", problems) for name, raw in services.items()}


# ---------------------------------------------------------------------------------------------------------------------
# Value helpers


def _scalar(value: Any) -> str | None:
    """A YAML scalar as the string compose would pass on."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def _environment(value: Any, path: str, problems: _Problems) -> dict[str, str]:
    """``environment`` as a mapping, from the map form or the ``KEY=value`` list form."""
    if value is None:
        return {}

    pairs: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        pairs = [(str(key), item) for key, item in value.items()]
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            text = _scalar(entry)
            if text is None:
                problems.invalid(f"{path}[{index}]", "expected KEY=value")
                continue
            key, equals, item = text.partition("=")
            pairs.append((key, item if equals else None))
    else:
        problems.invalid(path, "expected a mapping or a list")
        return {}

    variables: dict[str, str] = {}
    for key, item in pairs:
        text = _scalar(item)
        if text is None:
            # Compose would read it from the host's environment, which a sandbox doesn't have.
            problems.unsupported(f"{path}.{key}: no value")
        else:
            variables[key] = text
    return variables


def _argv(value: Any, path: str, problems: _Problems) -> list[str] | None:
    """``command`` or ``entrypoint`` as an argument list; a string is split as a shell would."""
    if value is None:
        return None

    if isinstance(value, list):
        arguments = [_scalar(item) for item in value]
        if any(argument is None for argument in arguments):
            problems.invalid(path, "expected a list of strings")
            return None
        return [argument for argument in arguments if argument is not None]

    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError as exc:
            problems.invalid(path, f"can't split the command: {exc}")
            return None

    problems.invalid(path, "expected a string or a list")
    return None


def _entrypoint(value: Any, path: str, problems: _Problems) -> list[str] | None:
    """``entrypoint`` as an argument list; an empty one is unsupported, since a pod can't clear an image's entrypoint."""
    arguments = _argv(value, path, problems)
    if arguments == []:
        problems.unsupported(f"{path}: empty (it would clear the image's entrypoint)")
        return None
    return arguments


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def _duration_sec(value: Any, default: float, path: str, problems: _Problems) -> float:
    """A compose duration (``1m30s``, ``5s``) in seconds; a bare number is seconds."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    if isinstance(value, str):
        text = value.replace(" ", "")
        parts = _DURATION_PART.findall(text)
        if parts and "".join(f"{number}{unit}" for number, unit in parts) == text:
            return sum(float(number) * _DURATION_UNITS[unit] for number, unit in parts)

    problems.invalid(path, f"not a duration: {value!r}")
    return default


def _healthcheck_command(test: Any, path: str, problems: _Problems) -> list[str] | None:
    """The command a healthcheck ``test`` runs; None when it runs nothing (``NONE``) or isn't valid."""
    if isinstance(test, str):
        return ["sh", "-c", test]

    if not isinstance(test, list) or not test:
        problems.invalid(path, "expected a string or a list")
        return None

    items = [_scalar(item) for item in test]
    if any(item is None for item in items):
        problems.invalid(path, "expected a list of strings")
        return None

    kind, rest = items[0], [item for item in items[1:] if item is not None]
    if kind == "NONE":
        return None
    if kind == "CMD" and rest:
        return rest
    if kind == "CMD-SHELL" and len(rest) == 1:
        return ["sh", "-c", rest[0]]

    problems.invalid(path, "expected CMD, CMD-SHELL or NONE first, with a command after it")
    return None


def _readiness(value: Any, path: str, problems: _Problems) -> ComposeReadiness | None:
    """A compose ``healthcheck`` as a readiness command and its wait budget; None when it declares no command."""
    if value is None:
        return None
    if not isinstance(value, dict):
        problems.invalid(path, "expected a mapping")
        return None

    for key in value:
        if key not in _HEALTHCHECK_KEYS:
            problems.unsupported(f"{path}.{key}")
    if value.get("disable") is True or "test" not in value:
        return None

    command = _healthcheck_command(value["test"], f"{path}.test", problems)
    if command is None:
        return None

    start_period = _duration_sec(value.get("start_period"), 0.0, f"{path}.start_period", problems)
    interval = _duration_sec(value.get("interval"), _DEFAULT_INTERVAL_SEC, f"{path}.interval", problems)
    retries = value.get("retries", _DEFAULT_RETRIES)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        problems.invalid(f"{path}.retries", "expected a whole number")
        retries = _DEFAULT_RETRIES
    return ComposeReadiness(exec=command, budget_sec=start_period + interval * retries)


def _container_port(value: Any, path: str, problems: _Problems) -> int | None:
    """The container side of a ``ports`` or ``expose`` entry."""
    if isinstance(value, dict):
        value = value.get("target")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        problems.invalid(path, "expected a port")
        return None

    text = str(value).split("/", 1)[0].rsplit(":", 1)[-1]
    if not text.isdigit():
        problems.unsupported(f"{path}: port ranges and names")
        return None
    return int(text)


def _ports(raw: Mapping[str, Any], path: str, problems: _Problems) -> list[ComposePort]:
    """The container ports a service ``expose``s or publishes with ``ports``, each once."""
    found: list[ComposePort] = []
    for key in ("expose", "ports"):
        entries = raw.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            problems.invalid(f"{path}.{key}", "expected a list")
            continue

        for index, entry in enumerate(entries):
            port = _container_port(entry, f"{path}.{key}[{index}]", problems)
            if port is not None and all(existing.port != port for existing in found):
                found.append(ComposePort(port=port, source=f"{key}[{index}]"))
    return found


def _quantity(value: Any, path: str, problems: _Problems) -> str | None:
    """A compose size (``512m``, ``1gb``, ``1073741824``) as a Kubernetes quantity."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and value == int(value):
        value = int(value)

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(b|k|kb|m|mb|g|gb)?", str(value).strip().lower())
    if match is None:
        problems.invalid(path, f"not a size: {value!r}")
        return None

    number, unit = match.groups()
    suffix = {None: "", "b": "", "k": "Ki", "kb": "Ki", "m": "Mi", "mb": "Mi", "g": "Gi", "gb": "Gi"}[unit]
    return f"{number}{suffix}"


def _cpus(value: Any, path: str, problems: _Problems) -> str | None:
    """A compose CPU count as a Kubernetes quantity; a whole number is written without its decimal point."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        problems.invalid(path, "expected a number")
        return None

    if isinstance(value, str) and not re.fullmatch(r"(?:\d+\.?\d*|\.\d+)", value.strip()):
        problems.invalid(path, f"not a number: {value!r}")
        return None

    try:
        number = float(value)
    except ValueError:
        problems.invalid(path, f"not a number: {value!r}")
        return None
    if not math.isfinite(number) or number <= 0:
        problems.invalid(path, f"not a positive number: {value!r}")
        return None
    return str(int(number)) if number == int(number) else str(number)


def _resource_section(values: Any, path: str, problems: _Problems) -> dict[str, str]:
    """One ``deploy.resources`` section (``limits`` or ``reservations``) as Kubernetes ``cpu`` and ``memory``."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        problems.invalid(path, "expected a mapping")
        return {}

    for key in values:
        if key not in ("cpus", "memory"):
            problems.unsupported(f"{path}.{key}")

    quantities: dict[str, str] = {}
    if (cpu := _cpus(values.get("cpus"), f"{path}.cpus", problems)) is not None:
        quantities["cpu"] = cpu
    if (memory := _quantity(values.get("memory"), f"{path}.memory", problems)) is not None:
        quantities["memory"] = memory
    return quantities


def _resources(raw: Mapping[str, Any], path: str, problems: _Problems) -> tuple[dict[str, str], dict[str, str]]:
    """``(limits, requests)`` from ``cpus``, ``mem_limit`` and ``deploy.resources``."""
    limits: dict[str, str] = {}
    requests: dict[str, str] = {}
    if (cpu := _cpus(raw.get("cpus"), f"{path}.cpus", problems)) is not None:
        limits["cpu"] = cpu
    if (memory := _quantity(raw.get("mem_limit"), f"{path}.mem_limit", problems)) is not None:
        limits["memory"] = memory

    deploy = raw.get("deploy")
    if deploy is None:
        return limits, requests
    if not isinstance(deploy, dict):
        problems.invalid(f"{path}.deploy", "expected a mapping")
        return limits, requests

    for key in deploy:
        if key != "resources":
            problems.unsupported(f"{path}.deploy.{key}")

    resources = deploy.get("resources")
    if resources is not None and not isinstance(resources, dict):
        problems.invalid(f"{path}.deploy.resources", "expected a mapping")
        return limits, requests

    resources = resources or {}
    limits.update(_resource_section(resources.get("limits"), f"{path}.deploy.resources.limits", problems))
    requests.update(_resource_section(resources.get("reservations"), f"{path}.deploy.resources.reservations", problems))
    return limits, requests


def _mount_entry(entry: Any, path: str, declared: set[str], problems: _Problems) -> ComposeVolumeMount | None:
    """One ``volumes`` entry as a mount of a declared named volume; bind mounts and anonymous volumes are refused."""
    source: Any
    target: Any
    read_only = False

    if isinstance(entry, str):
        parts = entry.split(":")
        if len(parts) == 1:
            problems.unsupported(f"{path}: anonymous volume")
            return None
        source, target = parts[0], parts[1]
        read_only = len(parts) > 2 and "ro" in parts[2].split(",")
    elif isinstance(entry, dict):
        if entry.get("type", "volume") != "volume":
            problems.unsupported(f"{path}: {entry.get('type')} mount")
            return None
        source, target, read_only = entry.get("source"), entry.get("target"), bool(entry.get("read_only"))
    else:
        problems.invalid(path, "expected a string or a mapping")
        return None

    if not isinstance(source, str) or not isinstance(target, str) or not source:
        problems.unsupported(f"{path}: anonymous volume")
    elif source.startswith((".", "/", "~")):
        problems.unsupported(f"{path}: bind mount {source}")
    elif source not in declared:
        problems.invalid(path, f"volume {source!r} is not declared under the top-level volumes")
    else:
        return ComposeVolumeMount(name=source, mount_path=target, read_only=read_only)
    return None


def _volume_mounts(raw: Any, path: str, declared: set[str], problems: _Problems) -> list[ComposeVolumeMount]:
    """The named-volume mounts in ``volumes``; bind mounts and anonymous volumes are rejected."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.invalid(path, "expected a list")
        return []

    mounts = (_mount_entry(entry, f"{path}[{index}]", declared, problems) for index, entry in enumerate(raw))
    return [mount for mount in mounts if mount is not None]


def _dotenv(text: str) -> dict[str, str]:
    """The ``KEY=value`` lines of a dotenv file, with values as written.

    Blank lines and ``#`` comments are skipped. A quoted value ends at its closing quote, and anything after that is
    ignored; an unquoted value ends at a `` #`` comment. References to other variables inside a value are not
    expanded, and a quoted value that continues on the next line is refused.

    Raises:
        ComposeError: A quoted value doesn't end on its line.
    """
    variables: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()

        if value[:1] in ("'", '"'):
            closing_quote = value.find(value[0], 1)
            if closing_quote == -1:
                raise ComposeError(
                    "compose_invalid", f"{key} is quoted but doesn't end on its line (multi-line values)"
                )
            value = value[1:closing_quote]
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        variables[key] = value
    return variables


def _env_file(raw: Any, path: str, environment_dir: Path, problems: _Problems) -> dict[str, str]:
    """The variables from ``env_file`` entries, read from the archive; paths must stay under ``environment/``."""
    if raw is None:
        return {}

    variables: dict[str, str] = {}
    entries = raw if isinstance(raw, list) else [raw]
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
        required = True
        if isinstance(entry, dict):
            required = entry.get("required", True) is not False
            entry = entry.get("path")
        if not isinstance(entry, str):
            problems.invalid(entry_path, "expected a path")
            continue

        resolved = posixpath.normpath(posixpath.join(ENVIRONMENT_DIR, entry))
        if resolved != ENVIRONMENT_DIR and not resolved.startswith(f"{ENVIRONMENT_DIR}/"):
            problems.unsupported(f"{entry_path}: {entry} is outside environment/")
            continue

        file = environment_dir.parent / resolved
        if not file.is_file():
            if required:
                problems.invalid(entry_path, f"{entry} is not in the task")
            continue

        try:
            variables.update(_dotenv(_read_data_file(file, entry)))
        except ComposeError as exc:
            problems.invalid(entry_path, exc.detail)
    return variables


def normalize_image(ref: str) -> str:
    """Docker's own normalization: ``postgres:18`` is ``docker.io/library/postgres:18``, ``redis`` is ``:latest``.

    A digest is kept, and so is a tag written next to it, as ``docker pull`` accepts both.
    """
    try:
        parsed = normalize_reference(ref)
    except ImageIdentityError as exc:
        raise ComposeError("compose_invalid", f"image {ref!r}: {exc}") from exc

    name, digest_separator, _ = ref.strip().partition("@")
    _, tag_separator, tag = name.rpartition(":")
    if digest_separator and tag_separator and "/" not in tag:
        return f"{parsed.registry}/{parsed.repository}:{tag}@{parsed.digest}"
    return str(parsed)


def _image_source(name: str, raw: Mapping[str, Any], problems: _Problems) -> ComposeImage | ComposeBuild | None:
    """Where a service's image comes from; ``build`` wins when both are set, as in Compose."""
    path = f"services.{name}"
    build = raw.get("build")
    if build is not None:
        return _build_source(name, build, problems)

    image = raw.get("image")
    if not isinstance(image, str) or not image:
        problems.invalid(path, "has neither `image` nor `build`")
        return None

    try:
        return ComposeImage(ref=normalize_image(image))
    except ComposeError as exc:
        problems.invalid(f"{path}.image", exc.detail)
        return None


def _build_source(name: str, build: Any, problems: _Problems, *, default_context: str = ".") -> ComposeBuild | None:
    """A service's ``build`` section as a context under ``environment/`` and a Dockerfile inside it."""
    path = f"services.{name}.build"
    if not isinstance(build, dict):
        problems.invalid(path, "expected a mapping or a path")
        return None

    for key in build:
        if key not in _BUILD_KEYS:
            problems.unsupported(f"{path}.{key}")

    context = build.get("context", default_context)
    dockerfile = build.get("dockerfile", "Dockerfile")
    if not isinstance(context, str) or not isinstance(dockerfile, str):
        problems.invalid(path, "`context` and `dockerfile` must be strings")
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://|^git@", context):
        problems.unsupported(f"{path}.context: remote context")
        return None

    context_path = posixpath.normpath(posixpath.join(ENVIRONMENT_DIR, context))
    if context_path != ENVIRONMENT_DIR and not context_path.startswith(f"{ENVIRONMENT_DIR}/"):
        problems.unsupported(f"{path}.context: {context} is outside environment/")
        return None
    if posixpath.isabs(dockerfile) or ".." in dockerfile.split("/"):
        problems.unsupported(f"{path}.dockerfile: {dockerfile} is outside its context")
        return None

    target = build.get("target")
    return ComposeBuild(
        context_path=context_path,
        dockerfile=dockerfile,
        args=_environment(build.get("args"), f"{path}.args", problems),
        target=target if isinstance(target, str) else None,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Dockerfile


def dockerfile_entrypoint(text: str) -> list[str] | None:
    """The ``ENTRYPOINT`` the last build stage ends with: exec form as written, shell form as ``sh -c``.

    A new ``FROM`` starts over, since the stage's own base image's entrypoint isn't known here.
    """
    entrypoint: list[str] | None = None
    continued = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.endswith("\\"):
            continued += stripped[:-1] + " "
            continue

        instruction = (continued + stripped).strip()
        continued = ""

        keyword, _, rest = instruction.partition(" ")
        if keyword.upper() == "FROM":
            entrypoint = None
        elif keyword.upper() == "ENTRYPOINT":
            entrypoint = _entrypoint_value(rest.strip())
    return entrypoint


def _entrypoint_value(rest: str) -> list[str] | None:
    """The argument list of an ``ENTRYPOINT`` instruction's arguments: JSON for exec form, else ``sh -c``."""
    if rest.startswith("["):
        try:
            parsed = json.loads(rest)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed or None
    return ["sh", "-c", rest] if rest else None


# ---------------------------------------------------------------------------------------------------------------------
# Parsing


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
    environment_dir = task_dir / ENVIRONMENT_DIR
    compose_file = environment_dir / COMPOSE_FILENAME
    compose_path = f"{task_dir.name}/{ENVIRONMENT_DIR}/{COMPOSE_FILENAME}"
    if not compose_file.is_file():
        misnamed = [
            name for name in COMPOSE_FILENAMES if name != COMPOSE_FILENAME and (environment_dir / name).is_file()
        ]
        if misnamed:
            raise ComposeError(
                "compose_unsupported_name",
                f"Harbor reads only {COMPOSE_FILENAME}; rename {misnamed[0]} to {COMPOSE_FILENAME}",
                compose_path=f"{task_dir.name}/{ENVIRONMENT_DIR}/{misnamed[0]}",
            )
        return None

    document = _read_yaml(compose_file, compose_path)
    if not isinstance(document.get("services"), dict):
        raise ComposeError("compose_invalid", "no `services` mapping", compose_path=compose_path)

    task_environment = _task_environment(task_dir)
    prebuilt = bool(task_environment.get("docker_image"))
    variables = _infra_variables(task_dir, task_environment)
    problems = _Problems()

    base_path = COMPOSE_PREBUILT_PATH if prebuilt else COMPOSE_BUILD_PATH
    base_document = _interpolate_tree(_read_yaml(base_path, "Harbor's base compose file"), variables, "base", problems)
    document = _interpolate_tree(document, variables, "", problems)
    for key in document:
        if key not in _TOP_LEVEL_KEYS and not str(key).startswith("x-"):
            problems.unsupported(str(key))

    base_services = _load_services(base_document, problems, what="Harbor's base compose file")
    task_services = _load_services(document, problems, what="the compose file")
    harbor_main = base_services.get(MAIN_SERVICE, {})
    main_raw = _merge(harbor_main, task_services.get(MAIN_SERVICE, {}))
    other_services = {name: service for name, service in task_services.items() if name != MAIN_SERVICE}

    declared_volumes = _declared_volumes(document, problems)
    _check_networks(document, problems)
    parsed = {
        name: _parse_service(name, raw, environment_dir, declared_volumes, problems)
        for name, raw in other_services.items()
    }
    main, main_build = _parse_main(
        main_raw, harbor_main, prebuilt, environment_dir, declared_volumes, problems, read_entrypoint
    )
    problems.raise_any(compose_path)

    services = _with_roles(
        {name: service for name, service in parsed.items() if service is not None},
        main.depends_on,
        compose_path=compose_path,
    )
    used_volumes = _used_volumes(services, main)
    main_image = main_raw.get("image")
    return ComposeTask(
        compose_path=compose_path,
        main_dockerfile=_relative_dockerfile(main_build),
        main_build=main_build,
        main_image=str(main_image) if prebuilt and main_image else None,
        services=services,
        main=main,
        volumes=[name for name in _declared_in_order(document) if name in used_volumes],
    )


def _relative_dockerfile(build: ComposeBuild | None) -> str | None:
    """A build's Dockerfile relative to ``environment/``; None without a build."""
    if build is None:
        return None
    return posixpath.relpath(posixpath.join(build.context_path, build.dockerfile), ENVIRONMENT_DIR)


def _declared_in_order(document: Mapping[str, Any]) -> list[str]:
    """The named volumes the compose file declares, in the order it declares them."""
    volumes = document.get("volumes")
    return [str(name) for name in volumes] if isinstance(volumes, dict) else []


def _used_volumes(services: Mapping[str, ComposeService], main: ComposeMainSettings) -> set[str]:
    """The names of the volumes that any service or ``main`` mounts."""
    service_mounts = (mount for service in services.values() for mount in service.volume_mounts)
    return {mount.name for mount in service_mounts} | {mount.name for mount in main.volume_mounts}


def _declared_volumes(document: Mapping[str, Any], problems: _Problems) -> set[str]:
    """The names of the top-level ``volumes``; options on a volume aren't supported."""
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


def _check_networks(document: Mapping[str, Any], problems: _Problems) -> None:
    """Record any top-level network besides the default: a pod has one network."""
    networks = document.get("networks")
    if isinstance(networks, dict) and set(networks) - {"default"}:
        problems.unsupported("networks")


def _check_service_keys(name: str, raw: Mapping[str, Any], allowed: frozenset[str], problems: _Problems) -> None:
    """Record every key of a service that a pod can't honor: unknown keys, other networks, other network modes."""
    for key in raw:
        if key not in allowed and key not in _NO_EFFECT_KEYS and not str(key).startswith("x-"):
            problems.unsupported(f"services.{name}.{key}")

    network_mode = raw.get("network_mode")
    if network_mode is not None and not (isinstance(network_mode, str) and network_mode.startswith("service:")):
        problems.unsupported(f"services.{name}.network_mode")

    networks = raw.get("networks")
    network_names = {str(network) for network in networks} if isinstance(networks, (list, dict)) else set()
    if network_names - {"default"}:
        problems.unsupported(f"services.{name}.networks")


def _parse_service(
    name: str, raw: dict[str, Any], environment_dir: Path, declared_volumes: set[str], problems: _Problems
) -> ComposeService | None:
    """One service other than ``main``; None when it has no usable image source (the problem is recorded)."""
    path = f"services.{name}"
    _check_service_keys(name, raw, _SERVICE_KEYS, problems)
    source = _image_source(name, raw, problems)
    if source is None:
        return None

    env = {
        **_env_file(raw.get("env_file"), f"{path}.env_file", environment_dir, problems),
        **_environment(raw.get("environment"), f"{path}.environment", problems),
    }
    limits, requests = _resources(raw, path, problems)
    shm_size = raw.get("shm_size")
    return ComposeService(
        name=name,
        source=source,
        depends_on=_conditions(raw.get("depends_on"), f"{path}.depends_on", problems),
        command=_entrypoint(raw.get("entrypoint"), f"{path}.entrypoint", problems),
        args=_argv(raw.get("command"), f"{path}.command", problems),
        env=env,
        ports=_ports(raw, path, problems),
        readiness=_readiness(raw.get("healthcheck"), f"{path}.healthcheck", problems),
        limits=limits,
        requests=requests,
        volume_mounts=_volume_mounts(raw.get("volumes"), f"{path}.volumes", declared_volumes, problems),
        shm_size=_quantity(shm_size, f"{path}.shm_size", problems),
    )


def _capabilities(raw: Mapping[str, Any], path: str, problems: _Problems) -> list[Literal["SYS_PTRACE"]]:
    """The capabilities ``main`` adds; only ``SYS_PTRACE`` is supported."""
    capabilities = raw.get("cap_add")
    if capabilities is None:
        return []
    if not isinstance(capabilities, list):
        problems.invalid(f"{path}.cap_add", "expected a list")
        return []

    for capability in capabilities:
        if capability != "SYS_PTRACE":
            problems.unsupported(f"{path}.cap_add: {capability}")
    return ["SYS_PTRACE"] if "SYS_PTRACE" in capabilities else []


def _main_dockerfile_entrypoint(
    environment_dir: Path, relative: str, path: str, problems: _Problems
) -> list[str] | None:
    """The ``ENTRYPOINT`` of ``main``'s Dockerfile at ``relative`` under ``environment/``.

    A Dockerfile that isn't there or can't be read is recorded against ``path``.
    """
    dockerfile = environment_dir / relative
    if not dockerfile.is_file():
        problems.invalid(path, f"main's Dockerfile {relative} is not in the task")
        return None

    try:
        return dockerfile_entrypoint(dockerfile.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        problems.invalid(path, f"can't read main's Dockerfile: {exc}")
        return None


def _parse_main(
    raw: dict[str, Any],
    harbor_main: Mapping[str, Any],
    prebuilt: bool,
    environment_dir: Path,
    declared_volumes: set[str],
    problems: _Problems,
    read_entrypoint: bool = True,
) -> tuple[ComposeMainSettings, ComposeBuild | None]:
    """``main``'s settings and what builds it (None for a prebuilt task), from ``main`` as laid over Harbor's own."""
    path = f"services.{MAIN_SERVICE}"
    _check_service_keys(MAIN_SERVICE, raw, _MAIN_KEYS, problems)
    if raw.get("command") != harbor_main.get("command"):
        # The sandbox keeps ``main`` alive itself; a task's own ``command`` for it would be dropped.
        problems.unsupported(f"{path}.command")

    env = {
        **_env_file(raw.get("env_file"), f"{path}.env_file", environment_dir, problems),
        **_environment(raw.get("environment"), f"{path}.environment", problems),
    }
    entrypoint = _entrypoint(raw.get("entrypoint"), f"{path}.entrypoint", problems)

    build: ComposeBuild | None = None
    if not prebuilt:
        build = _build_source(MAIN_SERVICE, raw.get("build"), problems, default_context=".")
        if build is not None and entrypoint is None and read_entrypoint:
            relative = str(_relative_dockerfile(build))
            entrypoint = _main_dockerfile_entrypoint(environment_dir, relative, f"{path}.build", problems)

    settings = ComposeMainSettings(
        depends_on=_conditions(raw.get("depends_on"), f"{path}.depends_on", problems),
        env=env,
        readiness=_readiness(raw.get("healthcheck"), f"{path}.healthcheck", problems),
        add_capabilities=_capabilities(raw, path, problems),
        volume_mounts=_volume_mounts(raw.get("volumes"), f"{path}.volumes", declared_volumes, problems),
        entrypoint=entrypoint,
    )
    return settings, build


def _check_dependencies(
    services: Mapping[str, ComposeService], main_depends: Mapping[str, str], compose_path: str
) -> None:
    """Raise unless every service a service or ``main`` waits for exists."""
    problems = _Problems()
    for name, service in services.items():
        for dependency in service.depends_on:
            if dependency != MAIN_SERVICE and dependency not in services:
                problems.invalid(f"services.{name}.depends_on", f"{dependency!r} is not a service")
    for dependency in main_depends:
        if dependency not in services:
            problems.invalid("services.main.depends_on", f"{dependency!r} is not a service")
    problems.raise_any(compose_path)


def _after_main_services(services: Mapping[str, ComposeService]) -> set[str]:
    """The services that start only once ``main`` is up: those that wait for ``main``, or for such a service."""
    after_main: set[str] = set()
    grew = True
    while grew:
        grew = False
        for name, service in services.items():
            waits_for_main = MAIN_SERVICE in service.depends_on or after_main & set(service.depends_on)
            if name not in after_main and waits_for_main:
                after_main.add(name)
                grew = True
    return after_main


def _start_order(services: Mapping[str, ComposeService], compose_path: str) -> list[str]:
    """The service names with each dependency before its dependents; ``main`` is not part of the order."""
    ordered: list[str] = []
    remaining = list(services)
    while remaining:
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


def _with_roles(
    services: dict[str, ComposeService], main_depends: Mapping[str, str], *, compose_path: str
) -> dict[str, ComposeService]:
    """Give each service its role and put the services in start order (a dependency before its dependents)."""
    _check_dependencies(services, main_depends, compose_path)

    after_main = _after_main_services(services)
    if clash := sorted(after_main & set(main_depends)):
        raise ComposeError(
            "compose_invalid", "a service and main depend on each other", compose_path=compose_path, paths=tuple(clash)
        )

    # A service another one waits on to finish is a one-time step: it runs to completion before its dependents start.
    run_once = {dependency for dependency, condition in main_depends.items() if condition == _COMPLETED}
    run_once |= {
        dependency
        for service in services.values()
        for dependency, condition in service.depends_on.items()
        if condition == _COMPLETED
    }

    result: dict[str, ComposeService] = {}
    for name in _start_order(services, compose_path):
        service = services[name]
        role = "after_main" if name in after_main else "run_once" if name in run_once else "sidecar"
        result[name] = service.model_copy(
            update={"role": role, "readiness": None if role == "run_once" else service.readiness}
        )
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Mapping to ComposeServices


def _check_reserved(task: ComposeTask) -> None:
    """Raise if a service exposes a port the sandbox pod uses, or takes a name the pod or image planning uses."""
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

    if reserved_names := [name for name in task.services if name in RESERVED_NAMES | _IMAGE_ROLE_NAMES]:
        raise ComposeError(
            "compose_reserved_name",
            "these names are reserved by the sandbox pod and the run's images",
            compose_path=task.compose_path,
            paths=tuple(reserved_names),
        )


def _wait_budget_sec(task: ComposeTask) -> float:
    """The longest any readiness check may take to pass: the services' and ``main``'s, whichever is greater.

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
    """``seconds`` rounded up to a whole second and kept within the bounds ``ComposeServices`` allows."""
    return min(MAX_READY_TIMEOUT_SEC, max(MIN_READY_TIMEOUT_SEC, math.ceil(seconds)))


def _mount_spec(mount: ComposeVolumeMount) -> dict[str, Any]:
    """A volume mount as the plain data ``ComposeServices`` takes."""
    return {"name": mount.name, "mount_path": mount.mount_path, "read_only": mount.read_only}


def _main_spec(main: ComposeMainSettings, wait_budget_sec: float) -> dict[str, Any]:
    """``main``'s part of the spec; ``wait_budget_sec`` becomes its readiness timeout when it has a readiness check."""
    spec: dict[str, Any] = {}
    if main.entrypoint:
        spec["entrypoint"] = main.entrypoint
    if main.env:
        spec["env"] = main.env
    if main.volume_mounts:
        spec["volume_mounts"] = [_mount_spec(mount) for mount in main.volume_mounts]
    if main.add_capabilities:
        spec["add_capabilities"] = main.add_capabilities
    if main.readiness is not None:
        spec["readiness"] = {"exec": main.readiness.exec_, "timeout_sec": _clamped_timeout_sec(wait_budget_sec)}
    return spec


def _service_spec(name: str, service: ComposeService, images: Mapping[str, str], compose_path: str) -> dict[str, Any]:
    """One service as the plain data ``ComposeServices`` takes, on the image ``images`` names for it."""
    if name in images:
        image = images[name]
    elif isinstance(service.source, ComposeImage):
        image = service.source.ref
    else:
        raise ComposeError(
            "compose_build_not_prepared",
            f"service {name} is built from {service.source.context_path}; its image hasn't been built",
            compose_path=compose_path,
        )

    spec: dict[str, Any] = {"name": name, "image": image, "role": service.role}
    if service.command is not None:
        spec["command"] = service.command
    if service.args is not None:
        spec["args"] = service.args
    if service.env:
        spec["env"] = service.env
    if service.ports:
        spec["ports"] = [port.port for port in service.ports]
    if service.readiness is not None and service.role != "run_once":
        spec["readiness"] = {"exec": service.readiness.exec_}
    if service.limits or service.requests:
        spec["resources"] = {
            key: value for key, value in (("limits", service.limits), ("requests", service.requests)) if value
        }
    if service.volume_mounts:
        spec["volume_mounts"] = [_mount_spec(mount) for mount in service.volume_mounts]
    if service.shm_size is not None:
        spec["shm_size"] = service.shm_size
    return spec


def _validation_detail(error: ValidationError, service_names: list[str]) -> str:
    """The spec's validation errors, with a service named by its compose name instead of its list position."""
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


def to_compose_services(task: ComposeTask, images: Mapping[str, str] | None = None) -> ComposeServices:
    """Map a parsed task to the ``ComposeServices`` spec for ``NemoOpenSandboxEnvironment``.

    Args:
        task: The task from :func:`parse_compose`.
        images: A full image reference for each service by name, such as a digest-pinned one. A service
            that runs a published image without an entry here runs by its normalized tag. A service built
            from the archive needs an entry, since nothing here builds images.

    Raises:
        ComposeError: The file defines only ``main``, a service has no image to run, or the spec breaks a rule of
            ``ComposeServices``.
    """
    if not task.services:
        raise ComposeError(
            "compose_override_only",
            "the file defines only `main`, which Harbor already provides; there are no services to run",
            compose_path=task.compose_path,
        )
    _check_reserved(task)

    wait_budget_sec = _wait_budget_sec(task)
    spec: dict[str, Any] = {
        "services": [
            _service_spec(name, service, images or {}, task.compose_path) for name, service in task.services.items()
        ],
        "main": _main_spec(task.main, wait_budget_sec),
    }
    if task.volumes:
        spec["volumes"] = task.volumes
    if task.main.readiness is None and wait_budget_sec > MIN_READY_TIMEOUT_SEC:
        # With no readiness check on ``main`` there is nowhere else to carry the services' wait.
        spec["services_timeout_sec"] = _clamped_timeout_sec(wait_budget_sec)

    try:
        return ComposeServices.model_validate(spec)
    except ValidationError as exc:
        raise ComposeError(
            "compose_invalid", _validation_detail(exc, list(task.services)), compose_path=task.compose_path
        ) from exc


def compose_services_kwarg(task_dir: Path, images: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The ``environment.kwargs.compose_services`` value for a task directory; None for a single-container task.

    The task Job calls this on the materialized task with the images its run prepared: it parses the compose file,
    maps it with :func:`to_compose_services`, and returns the spec as the plain data ``NemoOpenSandboxEnvironment``
    takes.
    """
    task = parse_compose(task_dir)
    if task is None:
        return None
    return to_compose_services(task, images).model_dump(mode="json", by_alias=True, exclude_defaults=True)


_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64


def check_compose_task(task: ComposeTask) -> ComposeServices:
    """The spec a parsed task maps to, before any image exists.

    This is submit's check: a service built from the archive gets a placeholder image, so everything but the real
    image is validated, and what the pod can't run is refused before any build starts.

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
    """Each service and the image it runs, as the run's provenance records them."""
    return {service.name: service.image for service in services.services}
