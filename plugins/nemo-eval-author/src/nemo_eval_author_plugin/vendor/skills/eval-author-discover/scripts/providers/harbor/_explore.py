# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Explore phase: find the Harbor artifacts a repository owns.

Standard library only, and safe to import when Harbor is absent, so an inventory
survives to orient in a repository that Judge cannot run on.

Everything is read from the local checkout: no client, no workspace, no trace
probe, and the agent doctrine comes from a local ``ETHOS.md``.

Everything here observes rather than proves. Finding a config file says nothing
about whether Harbor accepts it, which is why Judge, in ``_judge.py``, runs next
whenever Harbor is importable.

``yaml`` is used when available and is not a dependency of this skill: Harbor
depends on PyYAML, so a repository with Harbor installed always has it. Without
it, config detection falls back to a top-level key scan and every candidate is
marked unparsed. A file PyYAML rejects falls back to that same scan, so a config
with broken syntax is reported rather than silently missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _checks import ADVISORY, FAIL, PASS, WARN, CheckResult, check

try:
    import yaml
except ModuleNotFoundError:  # ships with Harbor; absent only when Harbor is
    yaml = None

# Malformed YAML raises yaml.YAMLError, which is not a ValueError. Empty without
# PyYAML, so the except clause naming these stays valid either way.
_PARSE_ERRORS: tuple[type[BaseException], ...] = () if yaml is None else (yaml.YAMLError,)

_CONFIG_SUFFIXES = (".yaml", ".yml", ".json")
_MAX_CONFIG_DEPTH = 4
_WORK_KEYS = ("datasets", "tasks")
# Matches a top-level `datasets:` or `tasks:` key, for the no-PyYAML fallback.
_WORK_KEY_PATTERN = re.compile(r"^(?:{}):".format("|".join(_WORK_KEYS)), re.MULTILINE)
_PRUNE_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        ".cache",
        "site-packages",
        "vendor",
        "cache",
        "dist",
        "build",
        "eval-and-optimize",
        ".nemo-optimizer",
        "jobs",
    }
)


@dataclass(frozen=True)
class ConfigCandidate:
    """A repository-owned Harbor config file.

    ``data`` is empty when the file could not be parsed, either because PyYAML is
    absent or because the syntax is broken. The ladder needs parsed data, so an
    unparsed candidate is reported and skipped.
    """

    path: Path
    data: dict[str, Any]
    parsed: bool

    @property
    def name(self) -> str:
        """Return the declared job name or the file name."""
        job_name = self.data.get("job_name")
        return job_name.strip() if isinstance(job_name, str) and job_name.strip() else self.path.name


@dataclass
class Dataset:
    """A directory Harbor can run as one dataset, with the tasks it holds.

    A directory is a dataset when a config points at it or when every
    subdirectory is a task. Otherwise each task in it is its own one-task dataset,
    so a folder mixing unrelated environments is not reported as one dataset.
    """

    path: Path
    task_paths: list[Path]
    theme: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class RepositoryScan:
    """The repository facts the validation ladder needs."""

    configs: list[ConfigCandidate]
    excluded_configs: list[ConfigCandidate]
    datasets: list[Dataset]
    task_paths: list[Path]
    ethos_path: str | None
    fingerprint: str
    input_file_count: int
    checks: list[CheckResult]


def _check(name: str, status: str, message: str, **kwargs: Any) -> CheckResult:
    return check(name, "repository", status, message, **kwargs)


def walk_dirs(root: Path, *, max_depth: int | None = None) -> Iterator[Path]:
    """Yield repository directories and skip generated trees."""
    for current, dir_names, _ in os.walk(root):
        directory = Path(current)
        depth = len(directory.relative_to(root).parts)
        dir_names[:] = sorted(
            name for name in dir_names if name not in _PRUNE_DIR_NAMES and (max_depth is None or depth < max_depth)
        )
        yield directory


def scan_repository(
    repo_root: Path,
    *,
    exclude_config: Callable[[ConfigCandidate], bool] | None = None,
    never_group: Callable[[Path], bool] | None = None,
) -> RepositoryScan:
    """Find repo-owned configs, Harbor datasets, and task directories.

    ``exclude_config`` removes files another provider owns, such as a Gym
    manifest, which also declares a ``datasets`` list. ``never_group`` names
    directories whose task children are each their own dataset, such as Gym's
    ``environments/`` root.
    """
    repo_root = repo_root.resolve()
    candidates = _config_candidates(repo_root)
    excluded = [candidate for candidate in candidates if exclude_config is not None and exclude_config(candidate)]
    configs = [candidate for candidate in candidates if candidate not in excluded]
    checks: list[CheckResult] = []
    tasks = _task_paths(repo_root)

    if not configs and not tasks:
        checks.append(
            _check(
                "config",
                FAIL,
                "No repository-owned Harbor config file or task directory exists.",
                hint="Add a YAML, YML, or JSON config with a nonempty datasets or tasks list, or a task directory.",
            )
        )
    elif not configs:
        checks.append(
            _check(
                "config",
                WARN,
                "No repository-owned Harbor config file exists; each dataset runs directly by path.",
                severity=ADVISORY,
                hint="Run a dataset with `harbor run -p <dataset> -a <agent>`, or add a job config to pin the agent.",
            )
        )
    else:
        count = len(configs)
        plural = "s" if count != 1 else ""
        checks.append(_check("config", PASS, "Found {} repository-owned Harbor config file{}.".format(count, plural)))

    unparsed = [candidate for candidate in configs if not candidate.parsed]
    if unparsed:
        names = ", ".join(candidate.path.name for candidate in unparsed)
        checks.append(
            _check(
                "config-parse",
                FAIL,
                "Cannot read {} config file{}: {}.".format(len(unparsed), "s" if len(unparsed) != 1 else "", names),
                hint=(
                    "Install PyYAML, which arrives with Harbor, to read YAML configs."
                    if yaml is None
                    else "Fix the YAML syntax in each file this message names."
                ),
            )
        )

    ethos: tuple[str, bytes] | None = None
    ethos_file = repo_root / "ETHOS.md"
    if ethos_file.is_file():
        try:
            ethos = ("ETHOS.md", ethos_file.read_bytes())
        except OSError as exc:
            checks.append(
                _check(
                    "ethos",
                    WARN,
                    "ETHOS.md exists but cannot be read: {}.".format(exc.strerror or exc),
                    severity=ADVISORY,
                    hint="Make ETHOS.md readable to record the agent doctrine.",
                )
            )
        else:
            checks.append(_check("ethos", PASS, "ETHOS.md defines the agent doctrine.", severity=ADVISORY))
    else:
        checks.append(
            _check(
                "ethos",
                WARN,
                "ETHOS.md does not exist at the repository root.",
                severity=ADVISORY,
                hint="Add ETHOS.md to define the agent doctrine.",
            )
        )

    datasets = _datasets(repo_root, tasks, configs, never_group)
    if tasks:
        checks.append(
            _check(
                "tasks-on-disk",
                PASS,
                "Found {} task {} in {} {}.".format(
                    len(tasks),
                    "directory" if len(tasks) == 1 else "directories",
                    len(datasets),
                    "dataset" if len(datasets) == 1 else "datasets",
                ),
                severity=ADVISORY,
                proven=False,
            )
        )
    else:
        checks.append(
            _check(
                "tasks-on-disk",
                WARN,
                "No task directories exist. A Harbor task directory holds a task.toml.",
                severity=ADVISORY,
                proven=False,
            )
        )

    fingerprint, count = _fingerprint(
        repo_root,
        [config.path for config in [*configs, *excluded]],
        ethos,
        [dataset.path for dataset in datasets],
    )
    return RepositoryScan(
        configs=configs,
        excluded_configs=excluded,
        datasets=datasets,
        task_paths=tasks,
        ethos_path=ethos[0] if ethos else None,
        fingerprint=fingerprint,
        input_file_count=count,
        checks=checks,
    )


def _config_candidates(repo_root: Path) -> list[ConfigCandidate]:
    candidates: list[ConfigCandidate] = []
    for directory in walk_dirs(repo_root, max_depth=_MAX_CONFIG_DEPTH):
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in _CONFIG_SUFFIXES:
                continue
            candidate = _candidate(path)
            if candidate is not None:
                candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda candidate: (
            len(candidate.path.relative_to(repo_root).parts) - 1,
            candidate.path.relative_to(repo_root).as_posix(),
        ),
    )


def _candidate(path: Path) -> ConfigCandidate | None:
    """Return a candidate when the file declares Harbor work, else None."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None

    is_json = path.suffix.lower() == ".json"
    if is_json or yaml is not None:
        try:
            if is_json:
                data = json.loads(text)
            else:
                assert yaml is not None
                data = yaml.safe_load(text)
        except (json.JSONDecodeError, ValueError, *_PARSE_ERRORS):
            pass  # unparseable, so fall back to the key scan
        else:
            if not isinstance(data, dict) or not _has_work(data):
                return None
            return ConfigCandidate(path=path, data=data, parsed=True)

    if not _WORK_KEY_PATTERN.search(text):
        return None
    return ConfigCandidate(path=path, data={}, parsed=False)


def _has_work(data: dict[str, Any]) -> bool:
    return any(isinstance(data.get(name), list) and data[name] for name in _WORK_KEYS)


def _task_paths(repo_root: Path) -> list[Path]:
    return sorted(
        directory
        for directory in walk_dirs(repo_root)
        if directory != repo_root and directory.name != "task_template" and (directory / "task.toml").is_file()
    )


def _datasets(
    repo_root: Path,
    tasks: list[Path],
    configs: list[ConfigCandidate],
    never_group: Callable[[Path], bool] | None,
) -> list[Dataset]:
    task_set = set(tasks)
    referenced = _config_dataset_paths(repo_root, configs)
    grouped: dict[Path, list[Path]] = {}
    # Sibling tasks share a parent, so decide once per parent to keep grouping linear in the task count.
    groups_by_parent: dict[Path, bool] = {}
    for task in tasks:
        parent = task.parent
        if parent not in groups_by_parent:
            groupable = parent != repo_root and not (never_group is not None and never_group(parent))
            groups_by_parent[parent] = parent in referenced or (groupable and _holds_only_tasks(parent, task_set))
        grouped.setdefault(parent if groups_by_parent[parent] else task, []).append(task)
    return [Dataset(path, members, _theme(members)) for path, members in sorted(grouped.items())]


def _config_dataset_paths(repo_root: Path, configs: list[ConfigCandidate]) -> set[Path]:
    paths: set[Path] = set()
    for config in configs:
        for dataset in config.data.get("datasets") or []:
            path = dataset.get("path") if isinstance(dataset, dict) else None
            if isinstance(path, str):
                paths.add((repo_root / path).resolve())
    return paths


def _holds_only_tasks(directory: Path, tasks: set[Path]) -> bool:
    try:
        children = [child for child in directory.iterdir() if child.is_dir()]
    except OSError:
        return False
    # Check the cheap conditions first: _holds_files walks the child's whole subtree.
    return all(
        child in tasks
        or child.name in _PRUNE_DIR_NAMES
        or child.name == "task_template"
        or child.name.startswith(".")
        or not _holds_files(child)
        for child in children
    )


def _holds_files(directory: Path) -> bool:
    """Return whether a directory holds any file outside generated trees such as ``__pycache__``."""
    for current in walk_dirs(directory):
        try:
            if any(entry.is_file() for entry in current.iterdir()):
                return True
        except OSError:
            continue
    return False


def _theme(tasks: list[Path]) -> dict[str, dict[str, int]]:
    """Count the categories, tags, and keywords the tasks' ``task.toml`` files declare."""
    counts: dict[str, Counter[str]] = {"categories": Counter(), "tags": Counter(), "keywords": Counter()}
    for task in tasks:
        try:
            config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            continue
        metadata = config.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        header = config.get("task")
        header = header if isinstance(header, dict) else {}
        category = metadata.get("category")
        if isinstance(category, str) and category:
            counts["categories"][category] += 1
        for key, source in (("tags", metadata.get("tags")), ("keywords", header.get("keywords"))):
            if isinstance(source, list):
                counts[key].update(item for item in source if isinstance(item, str) and item)
    return {key: dict(counter.most_common()) for key, counter in counts.items()}


def _fingerprint(
    repo_root: Path,
    config_paths: list[Path],
    ethos: tuple[str, bytes] | None,
    datasets: list[Path],
) -> tuple[str, int]:
    files = {path for path in [*config_paths, repo_root / "optimizer.yaml"] if path.is_file()}
    for dataset in datasets:
        if not dataset.is_relative_to(repo_root):
            continue
        for directory in walk_dirs(dataset):
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            files.update(path for path in entries if path.is_file() and path.resolve().is_relative_to(repo_root))
    files.discard(repo_root / "ETHOS.md")

    digest = hashlib.sha256()
    counted = 0
    for path in sorted(files):
        body = _file_digest(path)
        if body is None:
            continue
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(b"\0")
        digest.update(body)
        digest.update(b"\0")
        counted += 1
    if ethos is not None:
        digest.update(ethos[0].encode() + b"\0" + ethos[1] + b"\0")
    return digest.hexdigest(), counted + (ethos is not None)


def _file_digest(path: Path) -> bytes | None:
    """Return the file's digest, or None when it cannot be read.

    Hashing each file separately keeps a file the fingerprint cannot read out of
    the digest entirely, rather than contributing the bytes read before the
    failure. ``file_digest`` reads in chunks, so a large repository-owned dataset
    never lands in memory whole.
    """
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").digest()
    except OSError:
        return None
