# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Judge phase: make Harbor judge each repository-owned config and dataset.

The checks form a ladder of rungs, from schema to host variables. Every rung asks Harbor's own validators for a verdict, so each recorded fact is
proved rather than observed. Nothing here reimplements a Harbor rule.

This module imports Harbor at module scope. Import it only after ``_probe`` reports
Harbor available, so that a repository without Harbor still gets an inventory
instead of an ImportError.

``from harbor... import`` here reaches the installed Harbor library, not the
directory holding this file. Python 3 imports are absolute, and ``scripts/`` holds
no top-level ``harbor`` name for the enclosing directory to be found under.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from _checks import ADVISORY, FAIL, PASS, REQUIRED, WARN, CheckResult, check
from harbor.agents.factory import AgentFactory
from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task
from harbor.utils.env import get_required_host_vars
from harbor.utils.import_path import import_class
from providers.harbor._explore import ConfigCandidate
from pydantic import ValidationError

_ROUND_TRIP_TIMEOUT_SEC = 120


@dataclass
class RequiredEnvVar:
    """A host variable required by a Harbor config."""

    name: str
    default: str | None
    declared_in: Path


@dataclass
class ValidationOutcome:
    """Results from one Harbor preflight."""

    checks: list[CheckResult] = field(default_factory=list)
    required_env_vars: list[RequiredEnvVar] = field(default_factory=list)


def _check(name: str, status: str, message: str, **kwargs: object) -> CheckResult:
    return check(name, "validation", status, message, **kwargs)  # ty: ignore[invalid-argument-type]


async def judge_config(candidate: ConfigCandidate, repo_root: Path) -> ValidationOutcome:
    """Run the complete preflight without caching or skipping any check."""
    outcome = ValidationOutcome()
    if not candidate.parsed:
        outcome.checks.append(
            _check(
                "schema",
                FAIL,
                "Cannot read {} to validate it.".format(candidate.path.name),
                # Harbor is importable here, so it brought PyYAML: the syntax is the fault.
                hint="Fix the file's YAML syntax, which PyYAML could not parse.",
            )
        )
        return outcome

    with contextlib.chdir(repo_root):
        try:
            config = JobConfig.model_validate(candidate.data)
            config.validate_agent_concurrency_limits()
        except ValidationError as exc:
            errors = exc.errors(include_url=False, include_input=False)
            outcome.checks.append(_check("schema", FAIL, "Harbor rejected the job config: {}".format(errors)))
            return outcome
        except ValueError as exc:
            outcome.checks.append(_check("schema", FAIL, "Harbor rejected the job config: {}".format(exc)))
            return outcome
        outcome.checks.append(_check("schema", PASS, "Harbor accepts the job config schema."))

        job = await _resolve(config, outcome)
        _check_agent(config, outcome)
        _check_backend(config, outcome)
        outcome.checks.append(check_config_file(candidate.path, repo_root))
        if job is None:
            return outcome

        resolved = _resolved_task_paths(job)
        if resolved is None:
            outcome.checks.append(
                _check(
                    "compatibility",
                    FAIL,
                    "This Harbor version does not expose Job._task_configs.",
                    hint="Install a Harbor version that exposes the resolved task list.",
                )
            )
            return outcome
        task_dirs = _check_tasks(resolved, outcome)
        _check_coverage(config, resolved, outcome)
        _check_required_env_vars(config, task_dirs, outcome)
    return outcome


async def _resolve(config: JobConfig, outcome: ValidationOutcome) -> Job | None:
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="eval-author-jobs-") as scratch:
            job = await Job.create(config.model_copy(update={"jobs_dir": Path(scratch)}))
            # Cleanup on a private API, and it has to precede the scratch removal.
            # Resolution already succeeded, so a Harbor rename here is not a
            # resolution failure and must not be reported as one.
            with contextlib.suppress(Exception):
                job._close_logger_handlers()
    except Exception as exc:
        outcome.checks.append(
            _check(
                "resolution",
                FAIL,
                "Harbor could not resolve the job: {}: {}".format(type(exc).__name__, exc),
                hint="This error occurs before Harbor starts a container.",
            )
        )
        return None
    outcome.checks.append(_check("resolution", PASS, "Harbor resolved the job."))
    return job


def _resolved_task_paths(job: Job) -> list[Path] | None:
    task_configs = getattr(job, "_task_configs", None)
    if task_configs is None:
        return None
    paths: list[Path] = []
    for task_config in task_configs:
        try:
            paths.append(task_config.get_local_path().resolve())
        except ValueError:
            continue
    return paths


def _check_tasks(resolved: list[Path], outcome: ValidationOutcome) -> list[Path]:
    valid = [path for path in resolved if Task.is_valid_dir(path)]
    if not resolved:
        outcome.checks.append(_check("tasks", FAIL, "The config resolves to zero tasks."))
        return []
    outcome.checks.append(
        _check(
            "tasks",
            FAIL if len(valid) != len(resolved) else PASS,
            "{} of {} task dirs are valid Harbor tasks.".format(len(valid), len(resolved)),
        )
    )
    return valid


def _check_coverage(config: JobConfig, resolved: list[Path], outcome: ValidationOutcome) -> None:
    """Report task dirs Harbor dropped.

    Harbor skips task directories it cannot parse without raising, so a per-task
    check alone reports every survivor valid while one silently vanishes.
    """
    resolved_set = {path.resolve() for path in resolved}
    dropped_any = False
    for dataset in config.datasets:
        if dataset.path is None or not dataset.path.is_dir():
            continue
        on_disk = [
            child
            for child in sorted(dataset.path.iterdir())
            if child.is_dir() and child.name != "task_template" and (child / "task.toml").is_file()
        ]
        dropped = [child for child in on_disk if child.resolve() not in resolved_set]
        if not dropped:
            continue
        dropped_any = True
        selected_dropped = [
            path
            for path in dropped
            if (not dataset.task_names or any(fnmatchcase(path.name, pattern) for pattern in dataset.task_names))
            and not any(fnmatchcase(path.name, pattern) for pattern in dataset.exclude_task_names or [])
        ]
        required = bool(selected_dropped) and dataset.n_tasks is None
        filtered = bool(dataset.task_names or dataset.exclude_task_names)
        reported = selected_dropped if required else dropped
        names = ", ".join(path.name for path in reported)
        outcome.checks.append(
            _check(
                "coverage",
                FAIL if required else WARN,
                "Harbor did not resolve {} task dirs: {}.".format(len(reported), names),
                severity=REQUIRED if required else ADVISORY,
                hint=(
                    "Harbor skipped a task selected by the dataset filters."
                    if required and filtered
                    else "Harbor skips these task dirs silently."
                    if required
                    else "The dataset filters or n_tasks select a task subset."
                ),
            )
        )
    if not dropped_any:
        outcome.checks.append(_check("coverage", PASS, "Harbor dropped no local task dirs."))


def _check_required_env_vars(config: JobConfig, task_dirs: list[Path], outcome: ValidationOutcome) -> None:
    required: dict[str, RequiredEnvVar] = {}

    def collect(env: dict[str, str], declared_in: Path) -> None:
        for name, default in get_required_host_vars(env):
            required.setdefault(name, RequiredEnvVar(name, default, declared_in))

    for task_dir in task_dirs:
        task_config = _task_config(task_dir)
        if task_config is None:
            continue
        path = TaskPaths(task_dir).config_path
        collect(task_config.environment.env, path)
        collect(task_config.verifier.env, path)
        collect(task_config.solution.env, path)
    collect(config.environment.env, Path("<job config>"))
    collect(config.verifier.env, Path("<job config>"))
    for agent in config.agents:
        collect(agent.env, Path("<job config>"))
    outcome.required_env_vars = sorted(required.values(), key=lambda item: item.name)
    names = ", ".join(item.name for item in outcome.required_env_vars)
    outcome.checks.append(
        _check(
            "credentials",
            PASS,
            "{} host variables required".format(len(required)) + (": {}.".format(names) if names else "."),
        )
    )


@dataclass(frozen=True)
class TaskUnderJudgment:
    """A task for Harbor to judge: ``path`` is what Harbor reads, ``source`` is where it lives in the repository."""

    name: str
    path: Path
    source: Path


def judge_tasks(label: str, tasks: list[TaskUnderJudgment]) -> ValidationOutcome:
    """Ask Harbor whether each task in one dataset is valid, and which host variables it needs."""
    outcome = ValidationOutcome()
    invalid = [task.name for task in tasks if not Task.is_valid_dir(task.path)]
    outcome.checks.append(
        _check(
            "dataset-tasks",
            FAIL if invalid or not tasks else PASS,
            "{} of {} tasks in {} are valid Harbor tasks{}".format(
                len(tasks) - len(invalid),
                len(tasks),
                label,
                ": invalid {}.".format(", ".join(invalid)) if invalid else ".",
            ),
            hint=(
                "A Harbor task needs a parseable task.toml, an environment/ directory, and a tests/test.sh."
                if invalid or not tasks
                else None
            ),
        )
    )
    required: dict[str, RequiredEnvVar] = {}
    for task in tasks:
        task_config = _task_config(task.path)
        if task_config is None:
            continue
        for env in (task_config.environment.env, task_config.verifier.env, task_config.solution.env):
            for name, default in get_required_host_vars(env):
                required.setdefault(name, RequiredEnvVar(name, default, task.source / "task.toml"))
    outcome.required_env_vars = sorted(required.values(), key=lambda item: item.name)
    return outcome


def _task_config(task_dir: Path) -> TaskConfig | None:
    try:
        return TaskConfig.model_validate_toml(TaskPaths(task_dir).config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError):
        return None


def _check_agent(config: JobConfig, outcome: ValidationOutcome) -> None:
    for agent in config.agents:
        if agent.import_path is not None:
            with _evict_module_tree(agent.import_path):
                try:
                    imported = import_class(agent.import_path, label="agent")
                except (Exception, SystemExit) as exc:
                    outcome.checks.append(
                        _check(
                            "agent",
                            FAIL,
                            "Cannot import agent {}: {}: {}".format(agent.import_path, type(exc).__name__, exc),
                        )
                    )
                else:
                    outcome.checks.append(
                        _check("agent", PASS, "Agent {} imports as a class.".format(imported.__name__))
                    )
        elif agent.name is not None:
            try:
                AgentFactory.get_agent_class(AgentName(agent.name))
            except Exception as exc:
                outcome.checks.append(
                    _check(
                        "agent",
                        FAIL,
                        "Cannot load Harbor agent {}: {}: {}".format(agent.name, type(exc).__name__, exc),
                    )
                )
            else:
                outcome.checks.append(_check("agent", PASS, "Built-in agent {} is available.".format(agent.name)))


def _check_backend(config: JobConfig, outcome: ValidationOutcome) -> None:
    outcome.checks.append(backend_check(config.environment.type, config.environment.import_path))


def backend_check(environment_type: EnvironmentType | None = None, import_path: str | None = None) -> CheckResult:
    """Preflight one environment backend; the defaults are the Docker backend `harbor run -p` uses.

    Harbor's preflight silently passes when given no type, so an unset type is spelled out as Docker.
    """
    if environment_type is None and import_path is None:
        environment_type = EnvironmentType.DOCKER
    label = import_path or (environment_type.value if environment_type else "docker")
    try:
        EnvironmentFactory.run_preflight(environment_type, import_path)
    except (Exception, SystemExit) as exc:
        return _check(
            "backend",
            FAIL,
            "Environment backend {} is not ready: {}: {}".format(label, type(exc).__name__, exc),
        )
    return _check("backend", PASS, "Environment backend {} passed preflight.".format(label))


def check_config_file(config_path: Path, repo_root: Path) -> CheckResult:
    """Check the bytes that Harbor receives from its CLI."""
    harbor = _harbor_executable()
    if harbor is None:
        return _check(
            "round-trip",
            WARN,
            "The Harbor CLI round trip did not run.",
            severity=ADVISORY,
            hint="No harbor executable exists on PATH.",
        )
    try:
        completed = subprocess.run(
            [harbor, "job", "start", "--print-config", "-c", str(config_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_ROUND_TRIP_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _check("round-trip", WARN, "The Harbor CLI round trip failed: {}".format(exc), severity=ADVISORY)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return _check(
            "round-trip",
            FAIL,
            "The Harbor CLI rejected the config: {}.".format(detail[-1] if detail else "no output"),
        )
    return _check("round-trip", PASS, "The config file loads through the Harbor CLI.")


def _harbor_executable() -> str | None:
    local = Path(sys.executable).parent / "harbor"
    return str(local) if local.is_file() else shutil.which("harbor")


@contextlib.contextmanager
def _evict_module_tree(import_path: str) -> Iterator[None]:
    """Import without cached modules from another repository."""
    module = import_path.split(":", 1)[0].split(".", 1)[0]
    previous = {
        name: cached
        for name, cached in list(sys.modules.items())
        if name == module or name.startswith("{}.".format(module))
    }
    for name in previous:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == module or name.startswith("{}.".format(module)):
                sys.modules.pop(name)
        sys.modules.update(previous)
