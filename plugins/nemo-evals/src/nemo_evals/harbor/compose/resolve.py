# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a task's compose file into the document Harbor would hand to ``docker compose``.

Harbor runs ``docker compose -f <base> -f environment/docker-compose.yaml``: the task's file is laid over one of
Harbor's own base files (one that builds ``main`` from ``environment/``, or one that runs the ``docker_image`` named
in ``task.toml``), and ``${VAR}`` references are filled in from Harbor's variables, ``task.toml`` and
``environment/.env``. :func:`resolve_compose` does the same without Docker and returns the merged services.

The task is untrusted content. Everything here reads it as data, under a size cap, and never runs it.
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

# A bare ``$NAME``, and the inside of ``${NAME}``, ``${NAME:-fallback}`` or ``${NAME-fallback}``.
_BARE_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BRACED_REFERENCE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])(.*))?", re.DOTALL)


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


class Interpolator:
    """Fills in ``${VAR}`` references the way ``docker compose`` does before it reads a file.

    Compose files are templates: Harbor's base file says ``context: ${CONTEXT_DIR}``, and task files use variables
    from ``task.toml`` and ``environment/.env``. Until those are filled in, an image name or a port is just a
    placeholder, so every value must pass through here before anything reads it. The variables are a fixed set that
    :func:`_interpolation_variables` builds; the dispatcher's own environment is never consulted.

    A variable outside that set is empty, as it is when Compose runs on a machine that doesn't define it.
    """

    def __init__(self, variables: Mapping[str, str], problems: ComposeProblems) -> None:
        """Substitute from ``variables``, and record in ``problems`` each reference Compose itself would refuse."""
        self._variables = variables
        self._problems = problems

    def string(self, text: str, path: str) -> str:
        """Apply compose's variable substitution to one string found at ``path``.

        ``$$`` is a literal ``$``. ``$VAR`` and ``${VAR}`` take a value from the variables, or are empty when it has
        none. ``${VAR:-d}`` and ``${VAR-d}`` fall back to ``d``, ``${VAR:+a}`` and ``${VAR+a}`` give ``a`` only when
        the variable is set, and ``${VAR:?e}`` and ``${VAR?e}`` require it; ``d`` and ``a`` may use variables too.
        A ``$`` followed by anything else stays as written.

        With ``REGISTRY=mirror``: ``"${REGISTRY:-docker.io}/redis"`` becomes ``"mirror/redis"``.
        """
        pieces: list[str] = []
        index = 0
        while index < len(text):
            if text[index] != "$":
                pieces.append(text[index])
                index += 1
                continue

            # `$$` is an escaped `$`, `${...}` and `$NAME` are references, and any other `$` stays as written.
            next_character = text[index + 1 : index + 2]
            if next_character == "$":
                pieces.append("$")
                index += 2
            elif next_character == "{" and (end := _closing_brace(text, index + 2)) is not None:
                pieces.append(self._expand_braced(text[index + 2 : end], text[index : end + 1], path))
                index = end + 1
            elif next_character != "{" and (bare_name := _BARE_REFERENCE.match(text, index + 1)) is not None:
                pieces.append(self._lookup(bare_name.group(0)))
                index = bare_name.end()
            else:
                pieces.append("$")
                index += 1
        return "".join(pieces)

    def tree(self, node: Any, path: str = "") -> Any:
        """Return a copy of a whole YAML document (or any part of it) with every string interpolated.

        Compose substitutes variables anywhere in a file, in keys' values at any depth, so a document is interpolated
        once, as a whole, before it is merged or read. ``path`` tracks where each string sits, such as
        ``services.db.image``, so a reference Compose would refuse is reported against the key that used it.
        """
        if isinstance(node, str):
            return self.string(node, path)
        if isinstance(node, dict):
            return {key: self.tree(value, f"{path}.{key}" if path else str(key)) for key, value in node.items()}
        if isinstance(node, list):
            return [self.tree(value, f"{path}[{index}]") for index, value in enumerate(node)]
        return node

    def _lookup(self, name: str) -> str:
        """Return the value of variable ``name``, or an empty string when it has none.

        Compose does the same (with a warning) for a variable the host doesn't define, and Harbor runs these files on
        hosts that define none of a task's own variables. So a task that passes ``TEST_DIR=${TEST_DIR}`` gets an
        empty ``TEST_DIR`` here, as it does under Harbor.
        """
        return self._variables.get(name) or ""

    def _expand_braced(self, body: str, written: str, path: str) -> str:
        """Expand the inside of one ``${...}``, which unlike a bare ``$NAME`` may carry an operator.

        Compose files commonly write ``${REGISTRY:-docker.io}`` so they work without the variable set; the text after
        an operator is itself interpolated. Text that isn't a variable reference stays as ``written``, as Compose
        leaves it. A colon makes an empty variable count as unset: ``${V:-d}`` falls back when ``V`` is empty,
        ``${V-d}`` only when it is undefined.
        """
        match = _BRACED_REFERENCE.fullmatch(body)
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
        return value or ""


def _closing_brace(text: str, start: int) -> int | None:
    """Find the ``}`` that closes a ``${`` whose body starts at ``start``; None if it is never closed.

    A fallback can hold its own reference, as in ``${IMAGE:-${REGISTRY}/app}``, so the first ``}`` isn't always the
    closing one: nested braces are counted.
    """
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            if depth == 0:
                return index
            depth -= 1
    return None


def find_compose_file(task_dir: Path) -> Path | None:
    """Find the task's ``environment/docker-compose.yaml``; None for a task without a compose file.

    Whether this file exists is what makes a task a compose task. Harbor reads only this exact name, so a file under
    another name Compose accepts (``compose.yaml``, ``docker-compose.yml``) would be silently ignored by Harbor; it
    is reported instead, since the author almost certainly meant it to be used.

    Raises:
        ComposeError: The task has a compose file under a name Harbor doesn't read, such as ``docker-compose.yml``.
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


def resolve_compose(
    task_dir: Path, compose_file: Path, compose_path: str, problems: ComposeProblems
) -> ResolvedCompose:
    """Merge the task's compose file over Harbor's base file and interpolate both, as ``docker compose`` would.

    ``compose_path`` names the file in errors. Unsupported top-level keys and unresolvable variables are recorded in
    ``problems``; a file that can't be read at all raises.

    Raises:
        ComposeError: A file is too large, isn't valid YAML, or has no ``services`` mapping.
    """
    task_document = _read_yaml(compose_file, compose_path)
    if not isinstance(task_document.get("services"), dict):
        raise ComposeError("compose_invalid", "no `services` mapping", compose_path=compose_path)

    # Harbor picks its base file by whether `task.toml` names a prebuilt image.
    task_environment = _task_environment(task_dir)
    prebuilt = bool(task_environment.get("docker_image"))
    interpolator = Interpolator(_interpolation_variables(task_dir, task_environment), problems)

    base_path = COMPOSE_PREBUILT_PATH if prebuilt else COMPOSE_BUILD_PATH
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
    )


def merge_documents(base: Any, override: Any, key: str = "") -> Any:
    """Lay ``override`` on ``base`` as compose merges files: mappings merge, appended sequences append, the rest replace.

    Harbor never runs a task's compose file alone: it passes Harbor's base file first and the task's file second, so
    the task's ``main`` is a set of changes to Harbor's ``main``. Reading ``main`` correctly means merging the two
    the way Compose does.

    ``key`` is the mapping key the values sit under; ``ports``, ``expose``, ``cap_add`` and ``env_file`` append, so
    merging ``{ports: [80]}`` with ``{ports: [81]}`` gives ``{ports: [80, 81]}`` but ``{command: [a]}`` with
    ``{command: [b]}`` gives ``{command: [b]}``.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for name, value in override.items():
            merged[name] = merge_documents(base[name], value, name) if name in base else value
        return merged

    if isinstance(base, list) and isinstance(override, list) and key in _APPENDED_KEYS:
        return [*base, *(item for item in override if item not in base)]
    return override


def path_in_environment(relative_path: str) -> str | None:
    """Return ``relative_path`` as a normalized path from the task root, or None if it leaves ``environment/``.

    Paths in the compose file are relative to ``environment/``, but builds and file reads work from the task root,
    so paths are translated here. Only ``environment/`` is the task's environment: a path into ``tests/`` or
    ``solution/`` would hand the agent the answers, and one above the task reaches the host. ``"./api"`` gives
    ``"environment/api"``, and ``"../tests"`` gives None.
    """
    resolved = posixpath.normpath(posixpath.join(ENVIRONMENT_DIR, relative_path))
    if resolved == ENVIRONMENT_DIR or resolved.startswith(f"{ENVIRONMENT_DIR}/"):
        return resolved
    return None


def read_env_files(value: Any, path: str, task_dir: Path, problems: ComposeProblems) -> dict[str, str]:
    """Read a service's ``env_file`` entries from the archive as one ``{name: value}`` mapping; later files win.

    Compose reads these files on the host when the container starts. The sandbox has no host, so their contents
    are read now, from the task archive, and carried in the spec as plain variables.

    ``env_file: .env`` and ``env_file: [{path: .env, required: false}]`` both read ``environment/.env``. A path that
    leaves ``environment/`` is unsupported, and a missing file is invalid unless it is marked ``required: false``.
    """
    if value is None:
        return {}

    variables: dict[str, str] = {}
    entries = value if isinstance(value, list) else [value]
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
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


def _expanded_size(node: Any, sizes: dict[int, int]) -> int:
    """Count the values ``node`` holds once YAML aliases are resolved, counting a shared value once per use.

    YAML anchors and aliases let a few hundred bytes describe billions of values (the "billion laughs" attack). A
    compose file is untrusted task content, so its expanded size is checked before anything walks it.

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
    """Load a YAML mapping from ``path``, refusing a file that is too large, malformed, or expands to too many values.

    The one place a compose file is loaded, so every limit on untrusted YAML is applied here, and every way it can
    fail becomes a :class:`ComposeError` that names the file.
    """
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
        if _expanded_size(document, {}) > MAX_EXPANDED_VALUES:
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


def _read_data_file(path: Path, label: str) -> str:
    """Read a data file from the archive as text, refusing a symlink, an oversized file or bytes that aren't UTF-8.

    Used for ``.env`` and ``env_file`` files, which come from the untrusted task archive. A symlink could point
    outside the task (at the dispatcher's own files), so it is refused rather than followed.

    Raises:
        ComposeError: The file can't be read as data; its detail names ``label``.
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
    """Parse the ``KEY=value`` lines of a dotenv file, with values as written.

    Compose reads two kinds of dotenv file: ``environment/.env``, whose variables fill in ``${VAR}`` references, and
    a service's ``env_file``, whose variables go into its container. This reads both, following the subset of
    Compose's dotenv rules that task files use.

    Blank lines and ``#`` comments are skipped. A quoted value ends at its closing quote, and anything after that is
    ignored; an unquoted value ends at a `` #`` comment. References to other variables inside a value are not
    expanded, and a quoted value that continues on the next line is refused.

    ``export A="x y" # note`` gives ``{"A": "x y"}`` and ``B=value # note`` gives ``{"B": "value"}``.

    Raises:
        ComposeError: A quoted value doesn't end on its line.
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


def _task_environment(task_dir: Path) -> dict[str, Any]:
    """Read the ``[environment]`` table of ``task.toml``; empty when the task has none.

    Two of its settings change how the compose file is read: ``docker_image`` picks which of Harbor's base files the
    task's file is laid over, and ``env``, ``cpus`` and ``memory_mb`` become variables the file can reference.
    """
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return {}

    try:
        environment = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("environment", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read task.toml: {error}") from error
    return environment if isinstance(environment, dict) else {}


def _project_dotenv(task_dir: Path) -> dict[str, str]:
    """Read ``environment/.env``, the file of default variables Compose reads for ``${VAR}`` references.

    Harbor runs Compose from ``environment/``, so Compose picks up a ``.env`` file there automatically, and task
    authors rely on that. The path is fixed and the file is read as data, never run.
    """
    file = task_dir / ENVIRONMENT_DIR / ".env"
    if not file.is_file() and not file.is_symlink():
        return {}

    compose_path = f"{task_dir.name}/{ENVIRONMENT_DIR}/.env"
    try:
        return _parse_dotenv(_read_data_file(file, "environment/.env"))
    except ComposeError as error:
        raise ComposeError(error.code, error.detail, compose_path=compose_path) from error


def _interpolation_variables(task_dir: Path, task_environment: Mapping[str, Any]) -> dict[str, str]:
    """Build the variables that ``${VAR}`` references are filled in from, as Harbor would set them for Compose.

    Under Harbor, Compose sees three sources, applied lowest priority first: ``environment/.env``, the task's own
    ``[environment] env``, and the variables Harbor sets itself (``CONTEXT_DIR``, ``MAIN_IMAGE_NAME``, and so on).
    The host's environment, Compose's fourth source, is never read: a sandbox doesn't have one, and the
    dispatcher's environment must not leak into a spec.
    """
    variables = _project_dotenv(task_dir)

    task_variables = task_environment.get("env")
    if isinstance(task_variables, dict):
        variables.update({str(name): str(value) for name, value in task_variables.items()})

    # Harbor's own variables win over anything the task defines.
    variables.update({"CONTEXT_DIR": ".", "MAIN_IMAGE_NAME": f"hb__{task_dir.name}"})
    if task_environment.get("docker_image"):
        variables["PREBUILT_IMAGE_NAME"] = str(task_environment["docker_image"])
    if task_environment.get("cpus") is not None:
        variables["CPUS"] = str(task_environment["cpus"])
    if task_environment.get("memory_mb") is not None:
        variables["MEMORY"] = f"{task_environment['memory_mb']}M"
    return variables


def _normalize_service(raw_service: Any, path: str, problems: ComposeProblems) -> dict[str, Any]:
    """Put a service's ``environment``, ``depends_on`` and ``build`` in mapping form, so two files merge them by key.

    Compose accepts a list or a mapping for these keys, and merges them by key whichever form each file uses. Harbor's
    base file and a task's file often use different forms, so both are converted to mappings before
    :func:`merge_documents` runs. ``environment: ["A=1"]`` becomes ``{A: "1"}`` and ``build: ./api`` becomes
    ``build: {context: ./api}``.
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


def _load_services(document: Mapping[str, Any], problems: ComposeProblems, *, what: str) -> dict[str, dict[str, Any]]:
    """Return the ``services`` of a compose document, each in mapping form, ready to merge or read.

    Used for both Harbor's base file and the task's file; ``what`` names which one in errors.
    """
    services = document.get("services")
    if not isinstance(services, dict):
        raise ComposeError("compose_invalid", f"{what} has no `services` mapping")
    return {
        str(name): _normalize_service(raw_service, f"services.{name}", problems)
        for name, raw_service in services.items()
    }
