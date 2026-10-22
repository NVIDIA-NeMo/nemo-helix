# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the compose document ``docker compose`` would run for a Harbor task, without Docker.

Harbor runs ``docker compose -f <Harbor's base file> -f environment/docker-compose.yaml``. Compose merges the two
files and fills in their ``${VAR}`` references, so the task's file alone doesn't say what runs. :func:`resolve_compose`
repeats those steps and returns the result.

The task's files are untrusted. Everything here reads them as data, under size caps, and never runs them.
"""

from __future__ import annotations

import posixpath
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.environments.docker import COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH
from harbor.models.trial.paths import EnvironmentPaths
from nemo_evals.harbor.compose.models import ComposeError, ComposeProblems
from nemo_evals.harbor.compose.service_options import parse_depends_on, parse_environment
from nhx_sandbox.harbor_opensandbox_services import COMPOSE_FILENAMES, MAIN_SERVICE

# The only compose file name Harbor reads, in the task's ``environment/`` directory.
COMPOSE_FILENAME = "docker-compose.yaml"
ENVIRONMENT_DIR = "environment"

# The size of a compose file, ``.env`` file or ``env_file``, and how many values the compose file may expand to
# once YAML aliases are resolved. Both bound the work a hostile file can cause.
MAX_COMPOSE_BYTES = 256 * 1024
MAX_EXPANDED_VALUES = 50_000

_TOP_LEVEL_KEYS = frozenset({"services", "volumes", "networks", "version", "name"})

# Compose merges these sequences by appending; every other sequence is replaced.
_APPENDED_KEYS = frozenset({"expose", "ports", "cap_add", "env_file"})

# Harbor's log directories, which older task templates mount themselves as `${HOST_X_PATH}:${ENV_X_PATH}`: each `X`,
# the directory it is in every container, and the pod volume that stands in for the host directory Harbor binds.
HARBOR_LOG_DIRECTORIES = {
    "VERIFIER_LOGS": (str(EnvironmentPaths.verifier_dir), "harbor-verifier-logs"),
    "AGENT_LOGS": (str(EnvironmentPaths.agent_dir), "harbor-agent-logs"),
    "ARTIFACTS": (str(EnvironmentPaths.artifacts_dir), "harbor-artifacts"),
}

# `${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}`, optionally `:ro` or `:rw`, braces optional; the same
# directory on both sides.
_LOG_MOUNT = re.compile(
    r"\$\{?HOST_(?P<directory>VERIFIER_LOGS|AGENT_LOGS|ARTIFACTS)_PATH\}?"
    r":\$\{?ENV_(?P=directory)_PATH\}?"
    r"(?::(?P<mode>ro|rw))?"
)


@dataclass(frozen=True)
class ResolvedCompose:
    """A task's compose file after merging and interpolation, with every service still in raw YAML form."""

    document: dict[str, Any]
    """The task's own file, interpolated: its top-level ``volumes`` and ``networks`` are read from here."""

    prebuilt: bool
    """Whether ``main`` runs the prebuilt image ``task.toml`` names, rather than one built from ``environment/``."""

    harbor_main: Mapping[str, Any]
    """``main`` as Harbor's base file defines it, to tell what the task changed."""

    main: dict[str, Any]
    """``main`` after the task's file is laid over Harbor's."""

    services: dict[str, dict[str, Any]]
    """Every service of the task's file other than ``main``, by name."""

    harbor_log_mounts: dict[str, dict[str, bool]]
    """The Harbor log directories each service mounted itself, taken out of its ``volumes``: by service, each
    directory's key in :data:`HARBOR_LOG_DIRECTORIES` and whether the mount was read-only."""

    task_limits: dict[str, str]
    """``task.toml``'s CPUs and memory as compose would write them on ``main`` (``cpus``, ``mem_limit``); the
    sandbox already sizes ``main`` from them."""


def resolve_compose(
    task_dir: Path, compose_file: Path, compose_path: str, problems: ComposeProblems
) -> ResolvedCompose:
    """Build the compose document ``docker compose`` would run for this task.

    1. Read the task's compose file, under the size and alias caps.
    2. Take out the services' own mounts of Harbor's log directories (see :func:`_take_harbor_log_mounts`).
    3. Pick Harbor's base file: the prebuilt one when ``task.toml`` names a ``docker_image``, else the build one.
    4. Collect the variables Harbor would give Compose (see :meth:`Interpolator.for_task`).
    5. Fill in ``${VAR}`` references in both files.
    6. Record top-level keys a sandbox has no equivalent for, such as ``secrets``.
    7. Merge Harbor's ``main`` with the task's, as Compose would (see :func:`merge_documents`).

    Each service is returned still as raw YAML, for the parser to read. ``compose_path`` names the file in errors,
    and problems found along the way are recorded in ``problems``.

    Raises:
        ComposeError: A file is too large, isn't valid YAML, or has no ``services`` mapping.
    """
    task_document = _read_yaml(compose_file, compose_path)
    if not isinstance(task_document.get("services"), dict):
        raise ComposeError("compose_invalid", "no `services` mapping", compose_path=compose_path)

    # Harbor's own log mounts are recognized by their variable names, so take them out before those are filled in.
    harbor_log_mounts = _take_harbor_log_mounts(task_document)

    # Harbor picks its base file by whether `task.toml` names a prebuilt image.
    task_environment = _task_environment(task_dir)
    prebuilt = bool(task_environment.get("docker_image"))
    base_path = COMPOSE_PREBUILT_PATH if prebuilt else COMPOSE_BUILD_PATH
    interpolator = Interpolator.for_task(task_dir, task_environment, problems)

    base_document = interpolator.tree(_read_yaml(base_path, "Harbor's base compose file"), "base")
    document = interpolator.tree(task_document)

    # Top-level keys other than the ones compose defines (and `x-` extensions) have no sandbox equivalent.
    for key in document:
        if key not in _TOP_LEVEL_KEYS and not str(key).startswith("x-"):
            problems.unsupported(str(key))

    # Both files' services are put in mapping form first, so `main` merges key by key.
    base_services = _load_services(base_document, problems, what="Harbor's base compose file")
    task_services = _load_services(document, problems, what="the compose file")
    harbor_main = base_services.get(MAIN_SERVICE, {})
    return ResolvedCompose(
        document=document,
        prebuilt=prebuilt,
        harbor_main=harbor_main,
        main=merge_documents(harbor_main, task_services.get(MAIN_SERVICE, {})),
        services={name: service for name, service in task_services.items() if name != MAIN_SERVICE},
        harbor_log_mounts=harbor_log_mounts,
        task_limits=_task_limits(task_environment),
    )


def find_compose_file(task_dir: Path) -> Path | None:
    """Return the task's ``environment/docker-compose.yaml``, or None if it has none.

    Harbor reads only that exact name. A compose file under another name Compose accepts (``compose.yaml``,
    ``docker-compose.yml``) would be ignored by Harbor, so it's refused with a hint to rename it.

    Raises:
        ComposeError: The task has a compose file under a name Harbor doesn't read.
    """
    environment_dir = task_dir / ENVIRONMENT_DIR
    compose_file = environment_dir / COMPOSE_FILENAME
    if compose_file.is_file():
        return compose_file

    misnamed = [name for name in COMPOSE_FILENAMES if name != COMPOSE_FILENAME and (environment_dir / name).is_file()]
    if misnamed:
        raise ComposeError(
            "compose_unsupported_name",
            f"Harbor reads only {COMPOSE_FILENAME}; rename {misnamed[0]} to {COMPOSE_FILENAME}",
            compose_path=f"{task_dir.name}/{ENVIRONMENT_DIR}/{misnamed[0]}",
        )
    return None


def merge_documents(base: Any, override: Any, key: str = "") -> Any:
    """Merge ``override`` into ``base`` by Compose's rules for multiple ``-f`` files.

    Mappings merge key by key. ``ports``, ``expose``, ``cap_add`` and ``env_file`` append. Anything else in
    ``override`` replaces what's in ``base``. ``key`` is the key the two values sit under.

    ``{ports: [80]}`` merged with ``{ports: [81]}`` gives ``{ports: [80, 81]}``; ``{command: [a]}`` merged with
    ``{command: [b]}`` gives ``{command: [b]}``.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for name, value in override.items():
            merged[name] = merge_documents(base[name], value, name) if name in base else value
        return merged

    # An entry the base already has isn't added twice, so `ports: [80]` in both files stays one port.
    if isinstance(base, list) and isinstance(override, list) and key in _APPENDED_KEYS:
        return [*base, *(item for item in override if item not in base)]
    return override


class Interpolator:
    """Fills in ``${VAR}`` references the way ``docker compose`` does.

    Variables come only from the fixed set it's built with; a variable outside that set is empty, as when Compose
    runs on a host that doesn't define it. The host's own environment is never read.
    """

    # A bare ``$NAME``, and the inside of ``${NAME}``, ``${NAME:-fallback}`` or ``${NAME-fallback}``.
    _BARE_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    _BRACED_REFERENCE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])(.*))?", re.DOTALL)

    def __init__(self, variables: Mapping[str, str], problems: ComposeProblems) -> None:
        """Fill in from ``variables``; record in ``problems`` any reference Compose itself would refuse."""
        self._variables = variables
        self._problems = problems

    @classmethod
    def for_task(cls, task_dir: Path, task_environment: Mapping[str, Any], problems: ComposeProblems) -> Interpolator:
        """Build an interpolator with the variables Harbor gives Compose for this task.

        Lowest priority first: ``environment/.env``, then ``task.toml``'s ``[environment] env``, then Harbor's own
        (``CONTEXT_DIR``, ``MAIN_IMAGE_NAME``, ``PREBUILT_IMAGE_NAME``, ``CPUS``, ``MEMORY``, ``ENV_*_PATH``).
        """
        variables = cls._project_dotenv(task_dir)

        task_variables = task_environment.get("env")
        if isinstance(task_variables, dict):
            variables.update({str(name): str(value) for name, value in task_variables.items()})

        # Harbor's own variables win over anything the task defines. `ENV_*_PATH` name its log directories, which
        # Harbor still defines for older templates; their `HOST_*_PATH` partners name host paths a sandbox doesn't have.
        variables.update({"CONTEXT_DIR": ".", "MAIN_IMAGE_NAME": f"hb__{task_dir.name}"})
        variables.update({f"ENV_{key}_PATH": path for key, (path, _volume) in HARBOR_LOG_DIRECTORIES.items()})
        if task_environment.get("docker_image"):
            variables["PREBUILT_IMAGE_NAME"] = str(task_environment["docker_image"])

        limits = _task_limits(task_environment)
        if "cpus" in limits:
            variables["CPUS"] = limits["cpus"]
        if "mem_limit" in limits:
            variables["MEMORY"] = limits["mem_limit"]
        return cls(variables, problems)

    def tree(self, node: Any, path: str = "") -> Any:
        """Return a copy of a YAML document, or any part of one, with every string filled in.

        ``path`` is where ``node`` sits, such as ``services.db.image``, so a problem names the key it came from.
        """
        if isinstance(node, str):
            return self.string(node, path)
        if isinstance(node, dict):
            return {key: self.tree(value, f"{path}.{key}" if path else str(key)) for key, value in node.items()}
        if isinstance(node, list):
            return [self.tree(value, f"{path}[{index}]") for index, value in enumerate(node)]
        return node

    def string(self, text: str, path: str) -> str:
        """Fill in the references in one string found at ``path``.

        - ``$$`` is a literal ``$``.
        - ``$VAR`` and ``${VAR}`` are the variable's value, or empty.
        - ``${VAR:-d}`` and ``${VAR-d}`` fall back to ``d``.
        - ``${VAR:+a}`` and ``${VAR+a}`` give ``a`` only when the variable is set.
        - ``${VAR:?e}`` and ``${VAR?e}`` require the variable.
        - Any other ``$`` stays as written.

        With ``REGISTRY=mirror``, ``"${REGISTRY:-docker.io}/redis"`` becomes ``"mirror/redis"``.
        """
        pieces: list[str] = []
        index = 0
        while index < len(text):
            # `$$` is an escaped `$`, `${...}` and `$NAME` are references, and any other `$` stays as written.
            if text[index] != "$":
                pieces.append(text[index])
                index += 1
                continue

            next_character = text[index + 1 : index + 2]
            if next_character == "$":
                pieces.append("$")
                index += 2
            elif next_character == "{" and (end := self._closing_brace(text, index + 2)) is not None:
                pieces.append(self._expand_braced(text[index + 2 : end], text[index : end + 1], path))
                index = end + 1
            elif next_character != "{" and (bare_name := self._BARE_REFERENCE.match(text, index + 1)) is not None:
                pieces.append(self._variables.get(bare_name.group(0)) or "")
                index = bare_name.end()
            else:
                pieces.append("$")
                index += 1
        return "".join(pieces)

    def _expand_braced(self, body: str, written: str, path: str) -> str:
        """Expand the inside of one ``${...}``; text that isn't a valid reference stays as ``written``.

        A colon makes an empty variable count as unset: ``${V:-d}`` falls back when ``V`` is empty, ``${V-d}`` only
        when it's undefined. The text after an operator is itself filled in.
        """
        match = self._BRACED_REFERENCE.fullmatch(body)
        if match is None:
            return written

        name, operator, operand = match.groups()
        value = self._variables.get(name)
        # With a colon, an empty variable counts as unset.
        if operator is not None and operator.startswith(":") and not value:
            value = None

        if operator in (":-", "-"):
            return value if value is not None else self.string(operand, path)
        if operator in (":+", "+"):
            return self.string(operand, path) if value is not None else ""
        if operator in (":?", "?"):
            if value is not None:
                return value
            # Compose itself refuses to run a file whose required variable is missing.
            self._problems.invalid(path, f"{name} is required: {operand or 'it is not set'}")
            return written

        # A plain `${NAME}`: unset is empty, as on a host that doesn't define it.
        return value or ""

    @staticmethod
    def _closing_brace(text: str, start: int) -> int | None:
        """Return the index of the ``}`` closing a ``${`` whose body starts at ``start``, or None if it never closes.

        Braces are counted, since a fallback can hold its own reference: ``${IMAGE:-${REGISTRY}/app}``.
        """
        depth = 0
        for index in range(start, len(text)):
            # Each nested `{` must close before the outer reference can.
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                if depth == 0:
                    return index
                depth -= 1
        return None

    @staticmethod
    def _project_dotenv(task_dir: Path) -> dict[str, str]:
        """Read ``environment/.env``, which Compose loads automatically because Harbor runs it from there."""
        file = task_dir / ENVIRONMENT_DIR / ".env"
        if not file.is_file() and not file.is_symlink():
            return {}

        try:
            return _parse_dotenv(_read_data_file(file, "environment/.env"))
        except ComposeError as error:
            raise ComposeError(
                error.code, error.detail, compose_path=f"{task_dir.name}/{ENVIRONMENT_DIR}/.env"
            ) from error


def _read_yaml(path: Path, compose_path: str) -> dict[str, Any]:
    """Load a YAML mapping from an untrusted file, refusing one that's too large, malformed, or expands too far.

    YAML aliases let a small file describe billions of values, so the expanded size is checked before anything walks
    the document. Every failure becomes a :class:`ComposeError` that names ``compose_path``.
    """

    def expanded_size(node: Any, sizes: dict[int, int]) -> int:
        """Count the values ``node`` holds with aliases expanded; ``sizes`` caches each shared container."""
        if not isinstance(node, (dict, list)):
            return 1
        if id(node) not in sizes:
            children = node.values() if isinstance(node, dict) else node
            sizes[id(node)] = 1 + sum(expanded_size(child, sizes) for child in children)
        return sizes[id(node)]

    try:
        size = path.stat().st_size
        if size > MAX_COMPOSE_BYTES:
            raise ComposeError(
                "compose_invalid",
                f"the file is {size} bytes; the limit is {MAX_COMPOSE_BYTES}",
                compose_path=compose_path,
            )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))

        # Aliases share objects, so the file stays small in memory but a walk over it would not; cap the expanded size.
        if expanded_size(document, {}) > MAX_EXPANDED_VALUES:
            raise ComposeError(
                "compose_invalid",
                f"the file expands to more than {MAX_EXPANDED_VALUES} values once its aliases are resolved",
                compose_path=compose_path,
            )
    except (OSError, UnicodeDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read the file: {error}", compose_path=compose_path) from error
    except yaml.YAMLError as error:
        raise ComposeError("compose_invalid", f"not valid YAML: {error}", compose_path=compose_path) from error
    except RecursionError as error:
        raise ComposeError("compose_invalid", "the file is nested too deeply", compose_path=compose_path) from error

    if not isinstance(document, dict):
        raise ComposeError("compose_invalid", "not a YAML mapping", compose_path=compose_path)
    return document


def _task_environment(task_dir: Path) -> dict[str, Any]:
    """Read ``task.toml``'s ``[environment]`` table, or an empty one if the task has none."""
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return {}

    try:
        environment = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("environment", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read task.toml: {error}") from error
    return environment if isinstance(environment, dict) else {}


def _task_limits(task_environment: Mapping[str, Any]) -> dict[str, str]:
    """Write ``task.toml``'s ``cpus`` and ``memory_mb`` as compose service options, the way Harbor passes them.

    Harbor gives them to compose as ``CPUS`` and ``MEMORY`` (``2048M``), so a file that sets ``main``'s limits from
    those variables is only repeating them. ``{cpus = 2, memory_mb = 2048}`` gives ``{"cpus": "2", "mem_limit":
    "2048M"}``.
    """
    limits: dict[str, str] = {}
    if task_environment.get("cpus") is not None:
        limits["cpus"] = str(task_environment["cpus"])
    if task_environment.get("memory_mb") is not None:
        limits["mem_limit"] = f"{task_environment['memory_mb']}M"
    return limits


def _take_harbor_log_mounts(task_document: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Remove Harbor's own log mounts from each service's ``volumes``, and return them by service.

    Older task templates mount Harbor's log directories themselves, as ``${HOST_VERIFIER_LOGS_PATH}:
    ${ENV_VERIFIER_LOGS_PATH}``; Harbor still defines those variables so the mounts keep working. They bind a host
    directory a sandbox doesn't have, so they are taken out here, and the parser gives each one an equivalent: on
    ``main`` the sandbox already creates the directories, and another service gets a volume shared with ``main``.
    Only a mount written exactly this way is taken; any other bind mount is still refused.

    ``{"verifier": {"VERIFIER_LOGS": False}}`` means the ``verifier`` service mounted the verifier logs read-write.
    """
    mounts: dict[str, dict[str, bool]] = {}
    services = task_document["services"]
    for name, service in list(services.items()):
        volumes = service.get("volumes") if isinstance(service, dict) else None
        if not isinstance(volumes, list):
            continue

        kept = []
        for entry in volumes:
            # The long form is rebuilt as the short one, so both are recognized by the same pattern.
            if isinstance(entry, dict) and entry.get("type", "bind") == "bind":
                suffix = ":ro" if entry.get("read_only") is True else ""
                entry_text = f"{entry.get('source')}:{entry.get('target')}{suffix}"
            else:
                entry_text = entry if isinstance(entry, str) else ""

            match = _LOG_MOUNT.fullmatch(entry_text.strip())
            if match is None:
                kept.append(entry)
            else:
                mounts.setdefault(str(name), {})[match["directory"]] = match["mode"] == "ro"
        # A new mapping, not an edit: a YAML alias can share one mapping between services.
        services[name] = {**service, "volumes": kept}
    return mounts


def _load_services(document: Mapping[str, Any], problems: ComposeProblems, *, what: str) -> dict[str, dict[str, Any]]:
    """Return a compose document's services, each with its short forms written out (see :func:`_normalize_service`).

    ``what`` names the document in errors: Harbor's base file or the task's.
    """
    services = document.get("services")
    if not isinstance(services, dict):
        raise ComposeError("compose_invalid", f"{what} has no `services` mapping")
    return {
        str(name): _normalize_service(raw_service, f"services.{name}", problems)
        for name, raw_service in services.items()
    }


def _normalize_service(raw_service: Any, path: str, problems: ComposeProblems) -> dict[str, Any]:
    """Write a service's ``environment``, ``depends_on`` and ``build`` as mappings, so two files merge them by key.

    Compose accepts a list or a mapping for these, and the two files may use different forms:
    ``environment: ["A=1"]`` becomes ``{A: "1"}``, and ``build: ./api`` becomes ``{context: ./api}``.
    """
    if raw_service is None:
        return {}
    if not isinstance(raw_service, dict):
        problems.invalid(path, "expected a mapping")
        return {}

    service = dict(raw_service)
    if "depends_on" in service:
        service["depends_on"] = parse_depends_on(service["depends_on"], f"{path}.depends_on", problems)
    if "environment" in service:
        service["environment"] = parse_environment(service["environment"], f"{path}.environment", problems)
    if isinstance(service.get("build"), str):
        service["build"] = {"context": service["build"]}
    return service


# ---------------------------------------------------------------------------------------------------------------------
# Shared with the service readers: paths under environment/, and dotenv files.


def path_in_environment(relative_path: str) -> str | None:
    """Return a compose path as a path from the task root, or None if it leaves ``environment/``.

    Compose paths are relative to ``environment/``. Leaving it is refused, since ``tests/`` or ``solution/`` would
    hand the agent the answers. ``"./api"`` gives ``"environment/api"``; ``"../tests"`` gives None.
    """
    resolved = posixpath.normpath(posixpath.join(ENVIRONMENT_DIR, relative_path))
    if resolved == ENVIRONMENT_DIR or resolved.startswith(f"{ENVIRONMENT_DIR}/"):
        return resolved
    return None


def read_env_files(value: Any, path: str, task_dir: Path, problems: ComposeProblems) -> dict[str, str]:
    """Read a service's ``env_file`` entries from the task into one ``{name: value}`` mapping; later files win.

    Compose reads these files when the container starts; the sandbox can't, so they're read now. ``env_file: .env``
    and ``env_file: [{path: .env, required: false}]`` both read ``environment/.env``. A path outside
    ``environment/`` is unsupported, and a missing file is invalid unless marked ``required: false``.
    """
    if value is None:
        return {}

    variables: dict[str, str] = {}
    entries = value if isinstance(value, list) else [value]
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
        # The long form is `{path: .env, required: false}`; the short form is just the path.
        required = True
        if isinstance(entry, dict):
            required = entry.get("required", True) is not False
            entry = entry.get("path")
        if not isinstance(entry, str):
            problems.invalid(entry_path, "expected a path")
            continue

        # `env_file` paths are relative to `environment/` and must not leave it.
        resolved = path_in_environment(entry)
        if resolved is None:
            problems.unsupported(f"{entry_path}: {entry} is outside environment/")
            continue

        file = task_dir / resolved
        if not file.is_file():
            if required:
                problems.invalid(entry_path, f"{entry} is not in the task")
            continue

        try:
            variables.update(_parse_dotenv(_read_data_file(file, entry)))
        except ComposeError as error:
            problems.invalid(entry_path, error.detail)
    return variables


def _read_data_file(path: Path, label: str) -> str:
    """Read a text file from the task, refusing a symlink, an oversized file, or bytes that aren't UTF-8.

    A symlink could point outside the task, so it's refused rather than followed.

    Raises:
        ComposeError: The file can't be read as data; the detail names ``label``.
    """
    if path.is_symlink():
        raise ComposeError("compose_invalid", f"{label} is a symlink")

    try:
        if path.stat().st_size > MAX_COMPOSE_BYTES:
            raise ComposeError("compose_invalid", f"{label} is over the {MAX_COMPOSE_BYTES} byte limit")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read {label}: {error}") from error


def _parse_dotenv(text: str) -> dict[str, str]:
    """Parse a dotenv file's ``KEY=value`` lines, as Compose reads ``.env`` and ``env_file``.

    - Blank lines and ``#`` comments are skipped, and an ``export`` prefix is dropped.
    - A quoted value ends at its closing quote; an unquoted one ends at a `` #`` comment.
    - References to other variables aren't expanded.

    ``export A="x y" # note`` gives ``{"A": "x y"}``; ``B=value # note`` gives ``{"B": "value"}``.

    Raises:
        ComposeError: A quoted value continues onto the next line.
    """
    variables: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.strip()

        # A quoted value is taken literally up to its closing quote; an unquoted one stops at a ` #` comment.
        if value[:1] in ("'", '"'):
            closing_quote = value.find(value[0], 1)
            if closing_quote == -1:
                raise ComposeError(
                    "compose_invalid", f"{name} is quoted but doesn't end on its line (multi-line values)"
                )
            value = value[1:closing_quote]
        else:
            value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        variables[name] = value
    return variables
