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
from collections import ChainMap
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.environments.docker import COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH
from harbor.models.task.config import NetworkMode, TaskConfig
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import get_required_host_vars, is_env_template
from nemo_evals.harbor.compose.models import ComposeError, ComposeProblems
from nemo_evals.harbor.compose.service_options import parse_depends_on, parse_environment
from nhx_sandbox.harbor_opensandbox_services import COMPOSE_FILENAMES, MAIN_SERVICE
from yaml.constructor import ConstructorError

# The only compose file name Harbor reads, in the task's ``environment/`` directory.
# TODO: accept the other COMPOSE_FILENAMES and run-level overlays, via Harbor's extra_docker_compose.
COMPOSE_FILENAME = "docker-compose.yaml"
ENVIRONMENT_DIR = "environment"

# The size of a compose file, ``.env`` file or ``env_file``, and how many values the compose file may expand to
# once YAML aliases are resolved. Both bound the work a hostile file can cause.
MAX_COMPOSE_BYTES = 256 * 1024
MAX_EXPANDED_VALUES = 50_000

# How deeply ``${...}`` references may nest in each other's fallbacks. Real files nest two or three deep; the cap keeps
# a hostile one from exhausting the stack.
MAX_REFERENCE_DEPTH = 32

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

    interpolator: Interpolator
    """The variables Compose has for this task, to read ``env_file`` and look up value-less ``environment`` entries."""

    network_disabled: bool
    """Whether ``task.toml`` makes every phase ``no-network``, so the sandbox already denies all egress."""

    workdir: str | None
    """``task.toml``'s ``workdir``, which the sandbox creates in ``main`` when it starts; None when it sets none."""


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
        ComposeError: A file is a symlink, is too large, isn't valid YAML, or has no ``services`` mapping.
    """
    if _reached_through_symlink(compose_file, task_dir):
        raise ComposeError(
            "compose_invalid", "the compose file is a symlink or sits under one", compose_path=compose_path
        )
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

    base_document = interpolator.interpolate_tree(_read_yaml(base_path, "Harbor's base compose file"), "base")
    document = interpolator.interpolate_tree(task_document)

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
        interpolator=interpolator,
        network_disabled=_network_disabled(task_dir),
        workdir=str(task_environment["workdir"]) if task_environment.get("workdir") else None,
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
    runs on a host that doesn't define it. The host's own environment is never read, so a ``task.toml`` variable that
    Harbor would read from the host has no value here: using one is recorded as unsupported.
    """

    # A bare ``$NAME``, and the inside of ``${NAME}``, ``${NAME:-fallback}`` or ``${NAME-fallback}``.
    _BARE_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    _BRACED_REFERENCE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])(.*))?", re.DOTALL)

    def __init__(
        self, variables: Mapping[str, str], problems: ComposeProblems, host_only: Mapping[str, str] | None = None
    ) -> None:
        """Fill in from ``variables``; record in ``problems`` any reference Compose itself would refuse.

        ``host_only`` maps each ``task.toml`` variable Harbor would read from the host to the host variable it reads.
        """
        self._variables = variables
        self._problems = problems
        self._host_only = host_only or {}

    @classmethod
    def for_task(cls, task_dir: Path, task_environment: Mapping[str, Any], problems: ComposeProblems) -> Interpolator:
        """Build an interpolator with the variables Harbor gives Compose for this task.

        Lowest priority first: ``environment/.env``, then ``task.toml``'s ``[environment] env``, then Harbor's own
        (``CONTEXT_DIR``, ``MAIN_IMAGE_NAME``, ``PREBUILT_IMAGE_NAME``, ``CPUS``, ``MEMORY``, ``ENV_*_PATH``).
        Values in ``.env`` may refer to its earlier lines and to the other two.
        """
        # Harbor fills in a `task.toml` value that is all `${VAR}` or `${VAR:-default}` from the host before Compose
        # sees it (`resolve_env_vars`). The host isn't known here, so it takes its default, and without one it's
        # host-only.
        task_variables: dict[str, str] = {}
        # Maps a `task.toml` variable to the host variable Harbor would read for it: `TAG = "${HOST_TAG}"` gives
        # `{"TAG": "HOST_TAG"}`. Such a variable gets no value here; `variable_value` refuses it wherever the file
        # uses it.
        host_only: dict[str, str] = {}
        raw_task_variables = task_environment.get("env")
        if isinstance(raw_task_variables, dict):
            for raw_name, raw_value in raw_task_variables.items():
                name, value = str(raw_name), str(raw_value)
                # Harbor's own helpers decide what's a template, so this matches Harbor exactly. A value with text
                # around the reference, such as `x-${A}`, isn't one and stays as written.
                if is_env_template(value):
                    # One template gives one (host variable, default) pair; the default is None without `:-`.
                    ((host_name, default),) = get_required_host_vars({name: value})
                    if default is None:
                        host_only[name] = host_name
                        continue
                    value = default
                task_variables[name] = value

        # Harbor's own variables win over anything the task defines. `ENV_*_PATH` name its log directories, which
        # Harbor still defines for older templates; their `HOST_*_PATH` partners name host paths a sandbox doesn't have.
        harbor_variables = {"CONTEXT_DIR": ".", "MAIN_IMAGE_NAME": f"hb__{task_dir.name}"}
        harbor_variables.update({f"ENV_{key}_PATH": path for key, (path, _volume) in HARBOR_LOG_DIRECTORIES.items()})
        if task_environment.get("docker_image"):
            harbor_variables["PREBUILT_IMAGE_NAME"] = str(task_environment["docker_image"])

        limits = _task_limits(task_environment)
        if "cpus" in limits:
            harbor_variables["CPUS"] = limits["cpus"]
        if "mem_limit" in limits:
            harbor_variables["MEMORY"] = limits["mem_limit"]
        # The variables that override `.env`; Harbor's come last, so they win over the task's.
        later_variables = {**task_variables, **harbor_variables}

        # `.env` is read with an interpolator of its own, because its values may refer to `task.toml`'s and Harbor's
        # variables (`IMAGE=${REGISTRY}/redis:$TAG`), and its lines are not yet variables themselves.
        dotenv = cls._project_dotenv(task_dir, cls(later_variables, problems, host_only))

        # Lowest priority first: `.env`, then the task's and Harbor's. A host-only variable would override `.env`
        # with the host's value, so a `.env` line of the same name is dropped rather than used in its place.
        variables = {name: value for name, value in dotenv.items() if name not in host_only}
        variables.update(later_variables)
        return cls(variables, problems, host_only)

    def with_dotenv_lines(self, lines: dict[str, str]) -> Interpolator:
        """Return an interpolator for a dotenv file that looks in this one's variables first, then in its ``lines``.

        Compose fills in a dotenv line from the variables it already has before the file's earlier lines: with
        ``REGISTRY`` set in ``task.toml``, ``.env``'s ``REGISTRY=other`` then ``IMAGE=${REGISTRY}/redis`` still uses
        ``task.toml``'s registry. ``lines`` is read as it changes, so each line of the file can see the lines above it.
        """
        # A ChainMap looks a name up in each mapping in turn, and holds `lines` itself rather than a copy, so a line
        # added to it later is found too.
        return Interpolator(ChainMap(dict(self._variables), lines), self._problems, self._host_only)

    def variable_value(self, name: str, path: str) -> str | None:
        """Return the value Compose has for variable ``name``, or None if it has none.

        A ``task.toml`` variable Harbor would read from the host is recorded as unsupported at ``path``: its value is
        known only when the task runs.
        """
        value = self._variables.get(name)
        # Checked only when nothing else defines the name: one of Harbor's own variables, say, still wins.
        if value is None and name in self._host_only:
            self._problems.unsupported(f"{path}: task.toml sets {name} from the host's {self._host_only[name]}")
        return value

    def fill_in_value_less_entries(self, entries: Mapping[str, str | None], path: str) -> dict[str, str]:
        """Return ``entries`` with each value-less one (None) given the value Compose has for that variable.

        One Compose has no value for is left out, so the container doesn't get it, even if an ``env_file`` set it:
        ``{TOKEN: None}`` gives ``{TOKEN: "from-dotenv"}`` when ``.env`` sets it, and ``{}`` when nothing does.
        """
        filled: dict[str, str] = {}
        for name, value in entries.items():
            if value is None:
                value = self.variable_value(name, f"{path}.{name}")
            if value is not None:
                filled[name] = value
        return filled

    def interpolate_tree(self, node: Any, path: str = "") -> Any:
        """Return a copy of a YAML document, or any part of one, with every string filled in.

        ``path`` is where ``node`` sits, such as ``services.db.image``, so a problem names the key it came from.
        """
        if isinstance(node, str):
            return self.interpolate_string(node, path)
        if isinstance(node, dict):
            return {
                key: self.interpolate_tree(value, f"{path}.{key}" if path else str(key)) for key, value in node.items()
            }
        if isinstance(node, list):
            return [self.interpolate_tree(value, f"{path}[{index}]") for index, value in enumerate(node)]
        return node

    def interpolate_string(self, text: str, path: str, *, depth: int = 0) -> str:
        """Fill in the references in one string found at ``path``.

        - ``$$`` is a literal ``$``.
        - ``$VAR`` and ``${VAR}`` are the variable's value, or empty.
        - ``${VAR:-d}`` and ``${VAR-d}`` fall back to ``d``.
        - ``${VAR:+a}`` and ``${VAR+a}`` give ``a`` only when the variable is set.
        - ``${VAR:?e}`` and ``${VAR?e}`` require the variable.
        - A ``${`` that never closes, or a ``${...}`` in none of these forms, is invalid, as in Compose.
        - Any other ``$`` stays as written.

        With ``REGISTRY=mirror``, ``"${REGISTRY:-docker.io}/redis"`` becomes ``"mirror/redis"``. ``depth`` is how many
        ``${...}`` the text sits inside.
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
            elif next_character == "{":
                # `text[index]` is the `$` and `index + 1` the `{`, so the body of `${...}` starts at `index + 2`.
                end = self._closing_brace(text, index + 2)
                if end is None:
                    # Compose refuses the whole value. Stopping here also means a run of unclosed `${` isn't rescanned
                    # once per `${`.
                    self._problems.invalid(path, "invalid interpolation format: a `${` never closes (write $$ for $)")
                    pieces.append(text[index:])
                    break
                # Pass the body (`A:-x` in `${A:-x}`), and the whole reference as written for when it's invalid.
                pieces.append(self._expand_braced(text[index + 2 : end], text[index : end + 1], path, depth))
                index = end + 1
            # A `$` followed by a name, such as `$TAG`: the name runs to the first character a name can't hold.
            elif (bare_name := self._BARE_REFERENCE.match(text, index + 1)) is not None:
                pieces.append(self.variable_value(bare_name.group(0), path) or "")
                index = bare_name.end()
            else:
                pieces.append("$")
                index += 1
        return "".join(pieces)

    def _expand_braced(self, body: str, written: str, path: str, depth: int) -> str:
        """Expand the inside of one ``${...}`` found ``depth`` references deep; an invalid one stays as ``written``.

        A colon makes an empty variable count as unset: ``${V:-d}`` falls back when ``V`` is empty, ``${V-d}`` only
        when it's undefined. The text after an operator is itself filled in.
        """
        match = self._BRACED_REFERENCE.fullmatch(body)
        if match is None:
            self._problems.invalid(path, f"invalid interpolation format {written!r} (write $$ for $)")
            return written
        # Each fallback is filled in by a nested call, so a cap keeps a hostile file from exhausting the stack.
        if depth >= MAX_REFERENCE_DEPTH:
            self._problems.invalid(path, f"references nest more than {MAX_REFERENCE_DEPTH} deep")
            return written

        # `A:-x` gives ("A", ":-", "x"); a plain `A` gives ("A", None, None).
        name, operator, operand = match.groups()
        value = self.variable_value(name, path)
        # With a colon, an empty variable counts as unset.
        if operator is not None and operator.startswith(":") and not value:
            value = None

        # The operand may hold references of its own (`${A:-${B}}`), so it's filled in one level deeper.
        if operator in (":-", "-"):
            return value if value is not None else self.interpolate_string(operand, path, depth=depth + 1)
        if operator in (":+", "+"):
            return self.interpolate_string(operand, path, depth=depth + 1) if value is not None else ""
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
    def _project_dotenv(task_dir: Path, interpolator: Interpolator) -> dict[str, str]:
        """Read ``environment/.env``, which Compose loads automatically because Harbor runs it from there.

        Its references are filled in from ``interpolator`` (see :func:`_parse_dotenv`).
        """
        file = task_dir / ENVIRONMENT_DIR / ".env"
        if not file.is_file() and not file.is_symlink():
            return {}

        try:
            return _parse_dotenv(read_data_file(file, "environment/.env", task_dir), interpolator, "environment/.env")
        except ComposeError as error:
            raise ComposeError(
                error.code, error.detail, compose_path=f"{task_dir.name}/{ENVIRONMENT_DIR}/.env"
            ) from error


# The tags YAML names its types by: PyYAML's `resolve` picks one for each unquoted value (`8080` gets the int tag),
# and the value is then built as that type. `_ComposeLoader` checks and swaps these tags to read values as Compose
# does: it turns PyYAML's bool tag on `on` and its int tag on `22:22` into the str tag, and skips the merge tag (`<<`)
# when it looks for duplicate keys. PyYAML exports no names for them, so they're spelled out once here.
_BOOL_TAG = "tag:yaml.org,2002:bool"
_INT_TAG = "tag:yaml.org,2002:int"
_FLOAT_TAG = "tag:yaml.org,2002:float"
_STR_TAG = "tag:yaml.org,2002:str"
_MERGE_TAG = "tag:yaml.org,2002:merge"


class _ComposeLoader(yaml.SafeLoader):
    """A safe loader that reads a file as Compose's YAML parser does, where PyYAML's YAML 1.1 rules differ.

    - Only ``true`` and ``false`` are booleans; ``on``, ``off``, ``yes`` and ``no`` stay strings.
    - Numbers aren't read in base 60, so ``ports: [22:22]`` stays ``"22:22"`` rather than becoming ``1342``.
    - A key written twice in one mapping is refused; PyYAML would keep the last value without a word.
    """

    def resolve(self, kind: type[yaml.Node], value: Any, implicit: Any) -> str:
        """Return the type Compose's YAML 1.2 parser gives an unquoted value, where PyYAML's YAML 1.1 rules differ."""
        tag = super().resolve(kind, value, implicit)
        # YAML 1.1 also reads `on`, `off`, `yes` and `no` as booleans; YAML 1.2 only `true` and `false`.
        if tag == _BOOL_TAG and value.lower() not in ("true", "false"):
            return _STR_TAG
        # Only YAML 1.1's base-60 numbers contain a colon: `22:22` would become 1342.
        if tag in (_INT_TAG, _FLOAT_TAG) and ":" in value:
            return _STR_TAG
        return tag

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        """Build a mapping, refusing a key that appears twice in it, as Compose does."""
        # `node.value` is the mapping's (key node, value node) pairs as written in the file. They're checked before
        # the base class runs, because it copies a `<<` merge's keys in, and a key written next to `<<` legitimately
        # overrides a merged one.
        seen: set[Any] = set()
        for key_node, _ in node.value:
            # PyYAML has no constructor for `<<` itself; the base class handles it.
            if key_node.tag == _MERGE_TAG:
                continue
            # Build the key as Python sees it, so `1` and `"1"` stay different keys, as they are in YAML.
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError:
                # A list or mapping as a key; the base class refuses it with a YAML error.
                continue
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
                )
        return super().construct_mapping(node, deep)


def _read_yaml(path: Path, compose_path: str) -> dict[str, Any]:
    """Load a YAML mapping from an untrusted file, refusing one that's too large, malformed, or expands too far.

    YAML aliases let a small file describe billions of values. A merge key (``<<: [*a, *a]``) also makes PyYAML copy
    the merged keys while it builds the document, so a few hundred bytes of nested merges can take minutes and
    gigabytes to build, yet leave mappings small enough that a count afterwards sees nothing wrong. So the file is
    first read as a graph of nodes, where every alias is one shared node, and its expanded size is checked before
    anything is built. Every failure becomes a :class:`ComposeError` that names ``compose_path``.
    """

    def expanded_size(node: yaml.Node, sizes: dict[int, int]) -> int:
        """Count the nodes under ``node``, keys included, with aliases expanded; ``sizes`` caches each shared node."""
        # A mapping node's value is its (key, value) pairs; a sequence node's is its items; a scalar counts as one.
        if isinstance(node, yaml.MappingNode):
            children = [child for pair in node.value for child in pair]
        elif isinstance(node, yaml.SequenceNode):
            children = node.value
        else:
            return 1
        # Every alias of an anchor is the same node object, so each node is counted once and its total reused: the
        # walk stays as small as the file even when the total is in the billions.
        if id(node) not in sizes:
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

        # PyYAML loads in two steps, run separately here: parse the text into nodes, then build Python objects.
        loader = _ComposeLoader(path.read_text(encoding="utf-8"))
        try:
            # Step 1: parse only. Cheap even for a hostile file, since nothing is copied yet.
            node = loader.get_single_node()
            # A merge key's value is a sequence of aliases at this stage, so merges are counted like any other alias.
            if node is not None and expanded_size(node, {}) > MAX_EXPANDED_VALUES:
                raise ComposeError(
                    "compose_invalid",
                    f"the file expands to more than {MAX_EXPANDED_VALUES} values once its aliases and merge keys "
                    "are resolved",
                    compose_path=compose_path,
                )
            # Step 2: build the document, now known to be a safe size. An empty file has no node, as `safe_load` gives
            # None for it.
            document = loader.construct_document(node) if node is not None else None
        finally:
            loader.dispose()
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
    if not task_toml.is_file() and not task_toml.is_symlink():
        return {}
    if _reached_through_symlink(task_toml, task_dir):
        raise ComposeError("compose_invalid", "task.toml is a symlink")

    try:
        environment = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("environment", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read task.toml: {error}") from error
    return environment if isinstance(environment, dict) else {}


def _network_disabled(task_dir: Path) -> bool:
    """Whether ``task.toml`` sets ``no-network`` for the environment and no phase overrides it with anything else.

    Read with Harbor's own model, so the deprecated ``allow_internet = false`` counts too. A ``task.toml`` that
    Harbor can't read counts as networked: the runner refuses it anyway.
    """
    task_toml = task_dir / "task.toml"
    try:
        config = TaskConfig.model_validate_toml(read_data_file(task_toml, "task.toml", task_dir))
    except (ComposeError, ValueError):
        return False

    if config.environment.network_mode != NetworkMode.NO_NETWORK:
        return False
    phases = [config.agent, config.verifier]
    phases += [phase for step in config.steps or [] for phase in (step.agent, step.verifier)]
    return all(phase.network_mode in (None, NetworkMode.NO_NETWORK) for phase in phases)


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


def read_env_files(
    value: Any, path: str, task_dir: Path, interpolator: Interpolator, problems: ComposeProblems
) -> dict[str, str]:
    """Read a service's ``env_file`` entries from the task into one ``{name: value}`` mapping; later files win.

    Compose reads these files when the container starts; the sandbox can't, so they're read now, with references
    filled in from ``interpolator``. ``env_file: .env`` and ``env_file: [{path: .env, required: false}]`` both read
    ``environment/.env``. A path outside ``environment/`` is unsupported, as is a long-form key other than ``path``
    and ``required`` (such as ``format: raw``), and a missing file is invalid unless marked ``required: false``.
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
            # Other keys change how Compose reads the file (`format: raw` keeps quotes), which this reader doesn't do.
            other_keys = [str(key) for key in entry if key not in ("path", "required")]
            for key in other_keys:
                problems.unsupported(f"{entry_path}.{key}")
            if other_keys:
                continue
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
            variables.update(_parse_dotenv(read_data_file(file, entry, task_dir), interpolator, entry_path))
        except ComposeError as error:
            problems.invalid(entry_path, error.detail)
    return variables


def _reached_through_symlink(path: Path, task_dir: Path) -> bool:
    """Whether reaching ``path`` from ``task_dir`` follows a symlink: the file itself or any folder above it.

    Checking only the file misses ``environment/sub -> ../tests`` with ``env_file: sub/answers``, which reads the
    tests while the path still looks like it's inside ``environment/``.
    """
    # `resolve()` follows every symlink on the path. With none, the result is just the path under the resolved task
    # directory; with `sub -> ../tests`, `environment/sub/answers` resolves to `tests/answers` and the two differ.
    return path.resolve() != task_dir.resolve() / path.relative_to(task_dir)


def read_data_file(path: Path, label: str, task_dir: Path) -> str:
    """Read a text file from ``task_dir``, refusing a symlink, an oversized file, or bytes that aren't UTF-8.

    A symlink, or a folder above the file that is one, could point at the task's tests or outside the task, so it's
    refused rather than followed.

    Raises:
        ComposeError: The file can't be read as data; the detail names ``label``.
    """
    if _reached_through_symlink(path, task_dir):
        raise ComposeError("compose_invalid", f"{label} is a symlink or sits under one")

    try:
        if path.stat().st_size > MAX_COMPOSE_BYTES:
            raise ComposeError("compose_invalid", f"{label} is over the {MAX_COMPOSE_BYTES} byte limit")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ComposeError("compose_invalid", f"can't read {label}: {error}") from error


def _parse_dotenv(text: str, interpolator: Interpolator, label: str) -> dict[str, str]:
    """Parse a dotenv file's ``KEY=value`` lines, as Compose reads ``.env`` and ``env_file``.

    - Blank lines and ``#`` comments are skipped, and an ``export`` prefix is dropped.
    - A single-quoted value is taken literally up to its closing quote.
    - A double-quoted value ends at its closing quote, and its ``\\n``, ``\\t``, ``\\r``, ``\\\\``, ``\\"`` and
      ``\\$`` escapes are decoded.
    - An unquoted value ends at a `` #`` comment.
    - References in unquoted and double-quoted values are filled in from the file's earlier lines, then from
      ``interpolator``. A problem with one names ``label`` and the variable.

    ``export A="x y" # note`` gives ``{"A": "x y"}``; with ``USER=alice`` above it, ``URL=pg://${USER}@db # note``
    gives ``{"URL": "pg://alice@db"}``.

    Raises:
        ComposeError: A quoted value continues onto the next line.
    """
    variables: dict[str, str] = {}
    # Reads `variables` as it fills up, so each line sees the ones above it.
    file_interpolator = interpolator.with_dotenv_lines(variables)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.strip()

        # The value's first character picks the rule; anything after a closing quote, like a ` # note`, is dropped.
        if value[:1] == "'":
            # Single quotes: everything up to the closing quote, as written. No escapes, no references.
            closing_quote = value.find("'", 1)
            if closing_quote == -1:
                raise _multi_line_value_error(name)
            value = value[1:closing_quote]
        elif value[:1] == '"':
            # Double quotes: decode the escapes first, then fill in references.
            value = file_interpolator.interpolate_string(_decode_double_quoted_value(value, name), f"{label}.{name}")
        else:
            # Unquoted: cut a ` #` comment off, then fill in references. A `#` with no space before it is kept.
            value = file_interpolator.interpolate_string(
                re.split(r"\s#", value, maxsplit=1)[0].rstrip(), f"{label}.{name}"
            )
        # Added after its own value is read, so a line can't refer to itself, only to the lines above it.
        variables[name] = value
    return variables


# What each backslash escape in a double-quoted dotenv value stands for. `\$` becomes `$$`, which interpolation then
# turns into a literal `$`, so an escaped `$` isn't read as a reference.
_DOTENV_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "$": "$$"}


def _decode_double_quoted_value(value: str, name: str) -> str:
    """Return the inside of a dotenv value that starts with ``"``, with its escapes decoded.

    ``"{\\"ok\\":true}" # note`` gives ``{"ok":true}``: an escaped quote doesn't end the value.

    Raises:
        ComposeError: The closing quote isn't on this line.
    """
    pieces: list[str] = []
    # Start after the opening quote, and stop at the first quote that no backslash escapes.
    index = 1
    while index < len(value):
        if value[index] == '"':
            return "".join(pieces)
        # A backslash and the character after it are read as one escape. An escape not in the table, such as `\x`,
        # is kept as written.
        if value[index] == "\\" and index + 1 < len(value):
            pieces.append(_DOTENV_ESCAPES.get(value[index + 1], value[index : index + 2]))
            index += 2
            continue
        pieces.append(value[index])
        index += 1
    raise _multi_line_value_error(name)


def _multi_line_value_error(name: str) -> ComposeError:
    """The error for a quoted dotenv value that doesn't close on its line: Compose allows one, this reader doesn't."""
    return ComposeError("compose_invalid", f"{name} is quoted but doesn't end on its line (multi-line values)")
