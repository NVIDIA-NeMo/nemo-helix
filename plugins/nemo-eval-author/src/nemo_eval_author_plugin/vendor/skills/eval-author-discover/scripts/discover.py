#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record whether a repository's Harbor and NeMo Gym evaluations are ready to run.

Discovery helps the user orient themselves in an unfamiliar repository. The output
helps a later model run the repository's evals without re-deriving how. This command
finds the repository's Harbor and Gym artifacts and then validates them through four phases:

1. **Probe**, using only the standard library:
    - Is Harbor importable, and is its CLI on PATH?
    - Is NeMo Gym importable, and is its CLI on PATH?
2. **Explore**: take stock of the inventory of Harbor and Gym tasks
    - Which config files and datasets exist in the repository?
    - How many tasks are there in each dataset? How many datasets?
    - Is there a common theme or pattern to the different datasets?
3. **Judge.** Validate the contents of the inventory, each with the runtime that owns it:
    - Harbor judges configs and datasets, after converting any Gym extension tasks
      to Harbor tasks: schema, job resolution, agent, environment backend, CLI round
      trip, per-task validity, dropped-task coverage, and required host variables.
    - NeMo Gym judges manifests with ``gym env validate``: config, components, and
      data files. A manifest whose data Explore found missing is skipped.
4. **Solve.** Prove a sample of the evals actually runs:
    - Run up to 4 datasets and up to 4 tasks from each, picked the same way every run.
    - Harbor tasks run with Harbor's oracle agent, which needs Harbor and Docker.
    - Gym manifests run their verifier fixture with ``gym env test``, which needs Gym.
    - A task runs if it finishes and earns a reward; the reward's value is advisory.

Each provider's phases live under ``providers/<provider>/``. This module owns only
argument parsing, phase order, report assembly, and the exit code.

Explore runs whenever the Probe finds Harbor or Gym. Judge and Solve each run the
half the installed runtimes allow: Harbor's half needs Harbor, and Gym's half needs
Gym. Without Harbor, every Harbor finding is marked ``"proven": false`` and the
report names the next steps to get Harbor working. Solve skips Gym extension tasks,
because Gym has no runner for them yet.

This skill carries no dependency of its own. Harbor is the one import beyond the
standard library, and a repository holding Harbor evaluations has Harbor by
construction. PyYAML, pydantic, and toml arrive with it. NeMo Gym is probed but
never imported.

Usage:
    discover.py [--repo PATH] [--compact]

    --repo PATH    Repository to inspect. Defaults to the working directory.
    --compact      Emit single-line JSON.

Prints a JSON report on stdout. Converted Gym tasks and Harbor job output go to
temporary directories that are removed before exit. The one write into the
repository is Gym's: ``gym env test`` builds a resources server's venv at
``resources_servers/<name>/.venv``. ``SKILL.md`` tells the agent where to save the
report.

Exit codes:
    0  every config and dataset passed Judge, and nothing Solve ran failed
    1  a required check failed, a runtime is missing, or the path is unusable

WARNING: Run this only against a trusted repository. Validating an agent's
import path executes module top-level code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Every bundled module resolves against this directory, so put it on the path
# before importing one. Note that it holds no directory named after a provider
# package: a `harbor/` directory here would satisfy `find_spec("harbor")` on a
# machine without Harbor and make the probe claim an install that is not there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _other_evals  # noqa: E402
from _checks import ADVISORY, FAIL, PASS, WARN, CheckResult, check, required_failures  # noqa: E402
from providers.gym import _explore as gym_explore  # noqa: E402
from providers.gym import _judge as gym_judge  # noqa: E402
from providers.gym import _probe as gym_probe  # noqa: E402
from providers.gym import _solve as gym_solve  # noqa: E402
from providers.harbor import _probe  # noqa: E402
from providers.harbor import _solve as harbor_solve  # noqa: E402
from providers.harbor._explore import Dataset, RepositoryScan, scan_repository  # noqa: E402

_SCHEMA_VERSION = 2
_MIN_PYTHON = (3, 11)
_HARBOR_FORMAT = "harbor"
_MAX_SOLVE_DATASETS = 4
_MAX_SOLVE_TASKS = 4
# `gym env validate` takes a few seconds per manifest, so validate several at a time.
_GYM_JUDGE_WORKERS = 8
# Phase verdicts, then the Solve status of one dataset or manifest.
_VALID, _INVALID, _SKIPPED = "valid", "invalid", "skipped"
_PASSED, _FAILED, _NOT_SAMPLED = "passed", "failed", "not sampled"
# A manifest Judge could not check because Gym itself is missing counts against Judge, unlike an ordinary skip.
_BLOCKED = "blocked"


def _unproven(checks: list[CheckResult]) -> list[CheckResult]:
    """Mark observed findings so they cannot read as evidence."""
    for result in checks:
        result.proven = False
    return checks


async def _judge(scan: RepositoryScan, repo_root: Path) -> list[dict]:
    """Run the validation ladder against each config Harbor can read.

    Imported here rather than at module scope because ``_judge`` imports Harbor,
    which a repository without Harbor does not have.
    """
    from providers.harbor import _judge as harbor_judge

    configs: list[dict] = []
    for candidate in scan.configs:
        outcome = await harbor_judge.judge_config(candidate, repo_root)
        configs.append(
            {
                "name": candidate.name,
                "path": _display(candidate.path, repo_root),
                "runnable": not required_failures(outcome.checks),
                "required_env_vars": _env_vars(outcome.required_env_vars, repo_root),
                "checks": [result.as_dict() for result in outcome.checks],
                "_checks": outcome.checks,
            }
        )
    return configs


def _unjudged(scan: RepositoryScan, repo_root: Path) -> list[dict]:
    """Describe each config without claiming anything about it."""
    return [
        {
            "name": candidate.name,
            "path": _display(candidate.path, repo_root),
            "runnable": False,
            "required_env_vars": [],
            "checks": [],
            "_checks": [],
        }
        for candidate in scan.configs
    ]


def _describe_dataset(dataset: Dataset, repo_root: Path) -> dict:
    """Explore one dataset: its task count per format and its theme."""
    formats: dict[str, int] = {}
    for task in dataset.task_paths:
        kind = gym_explore.extension_kind(task) or _HARBOR_FORMAT
        count = gym_explore.jsonl_rows(task / "tasks.jsonl") if kind == gym_explore.PARAMETERIZED else 1
        formats[kind] = formats.get(kind, 0) + count
    return {
        "path": _display(dataset.path, repo_root),
        "task_count": sum(formats.values()),
        "formats": formats,
        "theme": dataset.theme,
        "runnable": False,
        "run_command": None,
        "required_env_vars": [],
        "solve": None,
        "checks": [],
        "_checks": [],
    }


def _judge_datasets(datasets: list[Dataset], described: list[dict], repo_root: Path) -> None:
    """Convert each dataset's Gym extension tasks, then have Harbor judge every task."""
    from providers.harbor import _judge as harbor_judge

    backend = harbor_judge.backend_check() if datasets else None
    with tempfile.TemporaryDirectory(prefix="eval-author-gym-") as scratch_root:
        for index, (dataset, entry) in enumerate(zip(datasets, described, strict=True)):
            label = entry["path"]
            tasks: list[harbor_judge.TaskUnderJudgment] = []
            checks: list[CheckResult] = []
            for position, task in enumerate(dataset.task_paths):
                kind = gym_explore.extension_kind(task)
                if kind is None:
                    tasks.append(harbor_judge.TaskUnderJudgment(task.name, task, task))
                    continue
                scratch = Path(scratch_root) / str(index) / str(position)
                try:
                    converted = gym_judge.convert(task, kind, scratch)
                except (gym_judge.ConversionError, OSError) as exc:
                    checks.append(
                        check(
                            "gym-conversion",
                            "validation",
                            FAIL,
                            "Cannot convert Gym task {}: {}".format(_display(task, repo_root), exc),
                            hint="Fix the Gym task files this message names.",
                        )
                    )
                    continue
                tasks.extend(harbor_judge.TaskUnderJudgment(item.name, item.path, task) for item in converted)
            outcome = harbor_judge.judge_tasks(label, tasks)
            checks.extend(outcome.checks)
            if backend is not None:
                checks.append(backend)
            entry["_judged"] = not required_failures(checks)
            entry["required_env_vars"] = _env_vars(outcome.required_env_vars, repo_root)
            entry["_checks"] = checks


def _judge_manifests(
    manifests: list[gym_explore.GymManifest], entries: list[dict], runtime: dict, repo_root: Path
) -> list[CheckResult]:
    """Have NeMo Gym validate every manifest whose data Explore found, several at a time.

    Records a ``judge`` block on each entry. A manifest Gym cannot look up, or whose
    data is missing, is skipped with the reason, since validating it would only
    rediscover what Explore already knows. Without a usable Gym, every manifest is
    blocked instead, and a required check fails so the repo cannot pass unchecked.
    """
    missing = None if gym_probe.is_available(runtime) else gym_probe.unavailable_reason(runtime)
    if missing and manifests:
        for entry in entries:
            entry["judge"] = {"status": _BLOCKED, "reason": missing, "error": None}
        return [
            check(
                "gym-unavailable",
                "validation",
                FAIL,
                "{} Discovery could not check the repo's {} Gym manifest{}.".format(
                    missing, len(manifests), "" if len(manifests) == 1 else "s"
                ),
                hint=gym_probe.INSTALL_HINT,
            )
        ]
    ready: list[tuple[dict, gym_explore.GymManifest]] = []
    for item, entry in zip(manifests, entries, strict=True):
        if not gym_explore.is_cataloged(item):
            reason = "Gym's catalog does not list this manifest, so Gym cannot find it by name."
        elif item.missing_data and item.kind == "benchmark":
            reason = "Its data is not downloaded ({}). Prepare it with `gym eval prepare --benchmark {}`.".format(
                ", ".join(item.missing_data), item.name
            )
        elif item.missing_data:
            reason = "Its data files are missing: {}.".format(", ".join(item.missing_data))
        else:
            reason = None
            ready.append((entry, item))
        entry["judge"] = {"status": _SKIPPED if reason else _NOT_SAMPLED, "reason": reason, "error": None}

    def judge_one(pair: tuple[dict, gym_explore.GymManifest]) -> gym_judge.GymRun:
        item = pair[1]
        assert item.name is not None and item.kind is not None
        return gym_judge.validate(runtime["gym_cli"], item.name, item.kind, repo_root)

    checks: list[CheckResult] = []
    with ThreadPoolExecutor(max_workers=_GYM_JUDGE_WORKERS) as pool:
        for (entry, item), run in zip(ready, pool.map(judge_one, ready), strict=True):
            assert item.name is not None and item.kind is not None
            entry["judge"] = {
                "status": _PASSED if run.completed else _FAILED,
                "reason": None,
                "error": run.error,
                "command": gym_judge.validate_command(repo_root, item.name, item.kind),
            }
            if not run.completed:
                checks.append(
                    check(
                        "gym-validate",
                        "validation",
                        FAIL,
                        "NeMo Gym rejected {}: {}".format(entry["path"], run.error),
                        hint="Reproduce it with `{}` and fix what Gym reports.".format(entry["judge"]["command"]),
                    )
                )
    if ready and not checks:
        checks.append(
            check(
                "gym-validate",
                "validation",
                PASS,
                "NeMo Gym validated all {} manifest{} with data.".format(len(ready), "" if len(ready) == 1 else "s"),
            )
        )
    return checks


def _solve(
    datasets: list[Dataset],
    described: list[dict],
    manifests: list[gym_explore.GymManifest],
    manifest_entries: list[dict],
    runtime: dict,
    repo_root: Path,
) -> list[CheckResult]:
    """Run a sample of the judged evals end to end, each with the runtime that owns it.

    Picks at most ``_MAX_SOLVE_DATASETS`` datasets and manifests, sharing the picks
    between Harbor and Gym when both have some, and at most ``_MAX_SOLVE_TASKS``
    tasks from each dataset. Returns the checks that belong to no dataset.
    """
    harbor_ready: list[tuple[dict, list[Path]]] = []
    for dataset, entry in zip(datasets, described, strict=True):
        tasks = [task for task in dataset.task_paths if harbor_solve.has_solution(task)]
        if entry["formats"].keys() - {_HARBOR_FORMAT}:
            reason = "NeMo Gym cannot run Harbor-format extension tasks yet."
        elif not _probe.is_available(runtime):
            reason = "Harbor is not installed."
        elif not entry.get("_judged"):
            reason = "Judge found problems to fix first."
        elif runtime["harbor_cli"] is None:
            reason = "No harbor executable is on PATH."
        elif not tasks:
            reason = "No task has a solution/solve.sh for Harbor's oracle agent to run."
        else:
            reason = None
            harbor_ready.append((entry, tasks))
        entry["solve"] = _solve_block(_SKIPPED if reason else _NOT_SAMPLED, reason)

    gym_ready: list[tuple[dict, gym_explore.GymManifest]] = []
    for item, entry in zip(manifests, manifest_entries, strict=True):
        judged = entry["judge"]
        if judged["status"] in {_SKIPPED, _BLOCKED}:
            reason = judged["reason"]
        elif judged["status"] == _FAILED:
            reason = "Judge found problems to fix first."
        else:
            reason = None
            gym_ready.append((entry, item))
        entry["solve"] = _solve_block(_SKIPPED if reason else _NOT_SAMPLED, reason)

    harbor_picks = min(len(harbor_ready), max(_MAX_SOLVE_DATASETS // 2, _MAX_SOLVE_DATASETS - len(gym_ready)))
    gym_picks = min(len(gym_ready), _MAX_SOLVE_DATASETS - harbor_picks)
    harbor_plan = [(entry, _sample(tasks, _MAX_SOLVE_TASKS)) for entry, tasks in _sample(harbor_ready, harbor_picks)]
    gym_plan = _sample(gym_ready, gym_picks)
    total = sum(len(tasks) for _, tasks in harbor_plan) + len(gym_plan)
    started = 0

    def announce(label: str, runner: str) -> None:
        # Solve can take many minutes, so tell whoever is watching stderr that discovery is still working.
        nonlocal started
        started += 1
        print("solve {}/{}: {} with {}".format(started, total, label, runner), file=sys.stderr, flush=True)

    checks: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="eval-author-solve-") as jobs_dir:
        for entry, tasks in harbor_plan:
            runs = []
            for task in tasks:
                announce(_display(task, repo_root), "Harbor's oracle agent")
                runs.append(harbor_solve.run_oracle(runtime["harbor_cli"], task, repo_root, Path(jobs_dir)))
            entry["_checks"].extend(_oracle_checks(entry["path"], runs, repo_root))
            entry["solve"] = _solve_block(
                _PASSED if all(run.completed for run in runs) else _FAILED,
                results=[
                    {
                        "task": _display(run.task, repo_root),
                        "rewards": run.rewards,
                        "error": run.error,
                        "command": harbor_solve.command(repo_root, _display(run.task, repo_root)),
                    }
                    for run in runs
                ],
            )
    for entry, item in gym_plan:
        assert item.name is not None and item.kind is not None
        announce(entry["path"], "`gym env test`")
        run = gym_solve.run_fixture(runtime["gym_cli"], item.name, item.kind, repo_root)
        cases = gym_solve.cases(run)
        reproduce = gym_solve.command(repo_root, item.name, item.kind)
        checks.append(
            check(
                "solve",
                "solve",
                PASS if run.completed else FAIL,
                "NeMo Gym scored all {} verifier cases for {}.".format(len(cases), entry["path"])
                if run.completed
                else "NeMo Gym could not test {}: {}".format(entry["path"], run.error),
                hint=None if run.completed else "Reproduce it with `{}` and fix what Gym reports.".format(reproduce),
            )
        )
        entry["solve"] = _solve_block(
            _PASSED if run.completed else _FAILED,
            results=[{"cases": cases, "error": run.error, "command": reproduce}],
        )
    return checks


def _gym_runner_checks(described: list[dict]) -> None:
    """Note each judged dataset holding Gym extension tasks, which nothing can run as written yet.

    Advisory: the tasks pass Judge, and the user can do nothing about the missing runner.
    """
    for entry in described:
        if entry.get("_judged") and entry["formats"].keys() - {_HARBOR_FORMAT}:
            entry["_checks"].append(
                check(
                    "gym-runner",
                    "solve",
                    WARN,
                    "{} holds Gym extension tasks. They pass Harbor's checks once converted, "
                    "but NeMo Gym has no runner for them yet.".format(entry["path"]),
                    severity=ADVISORY,
                    hint="Give each task a tests/test.sh so Harbor runs it as written, "
                    "or wait for NeMo Gym's Harbor task runner.",
                )
            )


def _oracle_checks(label: str, runs: list[harbor_solve.TaskRun], repo_root: Path) -> list[CheckResult]:
    failed = [run for run in runs if not run.completed]
    checks = [
        check(
            "solve",
            "solve",
            FAIL if failed else PASS,
            "Harbor's oracle agent could not run {} of {} sampled tasks in {}: {}".format(
                len(failed),
                len(runs),
                label,
                "; ".join("{}: {}".format(_display(run.task, repo_root), run.brief_error) for run in failed),
            )
            if failed
            else "Harbor's oracle agent ran {} sampled task{} in {} end to end.".format(
                len(runs), "" if len(runs) == 1 else "s", label
            ),
            hint="Reproduce one with `{}` and fix what Harbor reports.".format(
                harbor_solve.command(repo_root, _display(failed[0].task, repo_root))
            )
            if failed
            else None,
        )
    ]
    unrewarded = [run for run in runs if run.completed and not any(value > 0 for value in (run.rewards or {}).values())]
    if unrewarded:
        checks.append(
            check(
                "solve-reward",
                "solve",
                WARN,
                "The reference solution earned no reward on {} in {}.".format(
                    ", ".join(_display(run.task, repo_root) for run in unrewarded), label
                ),
                severity=ADVISORY,
                hint="The eval runs, but solution/solve.sh and the tests disagree; check both.",
            )
        )
    return checks


def _solve_block(status: str, reason: str | None = None, results: list[dict] | None = None) -> dict:
    return {"status": status, "reason": reason, "results": results or []}


def _sample(items: list, limit: int) -> list:
    """Pick up to ``limit`` items spread evenly across the list, first and last included, the same ones every run."""
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return items[:limit]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(index * step)] for index in range(limit)]


def _gym_manifest_checks(manifests: list[gym_explore.GymManifest]) -> list[CheckResult]:
    if not manifests:
        return []
    return [
        check(
            "gym-manifests",
            "repository",
            WARN,
            "Found {} NeMo Gym manifest{} for resources-server environments or benchmarks{}. "
            "Harbor cannot judge them, so Solve tests a sample with NeMo Gym's own CLI.".format(
                len(manifests),
                "" if len(manifests) == 1 else "s",
                "; {} have no data downloaded".format(sum(1 for item in manifests if item.task_count == 0))
                if any(item.task_count == 0 for item in manifests)
                else "",
            ),
            severity=ADVISORY,
            hint="Test any of them with `gym env test <name>`.",
            proven=False,
        )
    ]


def _env_vars(items: list, repo_root: Path) -> list[dict]:
    return [
        {"name": item.name, "default": item.default, "declared_in": _display(item.declared_in, repo_root)}
        for item in items
    ]


def _display(path: Path, repo_root: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_command(repo_root: Path, configs: list[dict]) -> str | None:
    """Return the Harbor command, only when exactly one config is runnable."""
    runnable = [config for config in configs if config["runnable"]]
    if len(configs) != 1 or len(runnable) != 1:
        return None
    return "cd {} && harbor job start -c {}".format(repo_root, runnable[0]["path"])


def _fail(message: str, hint: str) -> int:
    json.dump({"error": message, "hint": hint}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record whether a repository's Harbor and NeMo Gym evaluations are ready to run."
    )
    parser.add_argument("--repo", type=Path, default=Path(), help="Repository to inspect.")
    parser.add_argument("--compact", action="store_true", help="Emit single-line JSON.")
    args = parser.parse_args(argv)

    if sys.version_info < _MIN_PYTHON:
        return _fail(
            "Discovery needs Python {}.{} or later; this is {}.".format(
                _MIN_PYTHON[0], _MIN_PYTHON[1], ".".join(str(part) for part in sys.version_info[:3])
            ),
            "Re-run with a newer interpreter, for example `python3.12 discover.py --repo .`.",
        )

    repo_root = args.repo.expanduser()
    if not repo_root.is_dir():
        return _fail(
            "Not a directory: {}".format(repo_root),
            "Pass the repository that holds your Harbor configs, Gym environments, or task directories.",
        )
    repo_root = repo_root.resolve()

    # Probe: Harbor or Gym must be installed to go further.
    runtime = {**_probe.probe(), **gym_probe.probe()}
    proven = _probe.is_available(runtime)
    probe_valid = proven or gym_probe.is_available(runtime)
    phases = {
        "probe": _VALID if probe_valid else _INVALID,
        "explore": _SKIPPED,
        "judge": _SKIPPED,
        "solve": _SKIPPED,
    }
    runtime_checks = [*_probe.probe_checks(runtime), *gym_probe.probe_checks(runtime, report_missing=not probe_valid)]
    report: dict = {
        "schema_version": _SCHEMA_VERSION,
        "repo_root": repo_root.as_posix(),
        "provider": _probe.PROVIDER,
        "proven": proven,
        "runnable": False,
        "phases": phases,
        "runtime": runtime,
        "configs": [],
        "datasets": [],
        "task_count": 0,
        "gym_manifests": [],
        "other_eval_candidates": [],
        "other_eval_scan": None,
        "ethos_path": None,
        "fingerprint": None,
        "input_file_count": 0,
        "run_command": None,
    }
    if not probe_valid:
        return _emit(report, runtime_checks, args.compact)

    # Explore: which configs, datasets, and tasks exist, and what are they about?
    scan = scan_repository(
        repo_root,
        exclude_config=gym_explore.is_manifest,
        never_group=gym_explore.is_environments_root,
    )
    manifests = [gym_explore.manifest(candidate, repo_root) for candidate in scan.excluded_configs]
    datasets = [_describe_dataset(dataset, repo_root) for dataset in scan.datasets]
    # A manifest owns its environment folder and its resources server, whose code is the workload, not another eval.
    owned = [
        *scan.task_paths,
        *(candidate.path.parent for candidate in scan.excluded_configs),
        *(repo_root / "resources_servers" / item.name for item in manifests if item.name),
    ]
    manifest_entries: list[dict] = [
        {
            "path": _display(item.path, repo_root),
            "name": item.name,
            "kind": item.kind,
            "domain": item.domain,
            "dataset_count": item.dataset_count,
            "task_count": item.task_count,
            "missing_data": list(item.missing_data),
            "judge": None,
            "solve": None,
        }
        for item in manifests
    ]
    repository_checks = [*scan.checks, *_gym_manifest_checks(manifests)]
    phases["explore"] = _VALID if scan.configs or datasets or manifests else _INVALID

    # Judge: Harbor judges configs and datasets, NeMo Gym judges manifests, and either can run alone.
    harbor_found = bool(scan.configs or datasets)
    if not proven and not harbor_found and manifests and gym_probe.is_available(runtime):
        # Nothing here needs Harbor: the evals are Gym manifests, and Gym is installed to judge and run them.
        runtime_checks = [item for item in runtime_checks if item.name not in {"harbor", "harbor-cli"}]
    verdicts: list[bool] = []
    if proven and harbor_found:
        configs = await _judge(scan, repo_root)
        _judge_datasets(scan.datasets, datasets, repo_root)
        verdicts += [config["runnable"] for config in configs] + [entry["_judged"] for entry in datasets]
    else:
        configs = _unjudged(scan, repo_root)
        if not proven:
            _unproven(repository_checks)
    gym_judge_checks = _judge_manifests(manifests, manifest_entries, runtime, repo_root)
    # A skipped manifest is left out, but a blocked one counts as failed: Gym is missing, not its data.
    verdicts += [
        entry["judge"]["status"] == _PASSED for entry in manifest_entries if entry["judge"]["status"] != _SKIPPED
    ]
    if verdicts:
        phases["judge"] = _VALID if all(verdicts) else _INVALID

    # Solve: run a sample of the datasets and manifests that passed Judge end to end, Harbor tasks with Harbor
    # and Gym manifests with Gym. One failing dataset does not hold back the others; with none passing, Solve waits.
    _gym_runner_checks(datasets)
    solve_checks: list[CheckResult] = []
    judged = [entry for entry in datasets if entry.get("_judged")] + [
        entry for entry in manifest_entries if entry["judge"]["status"] == _PASSED
    ]
    if not judged:
        for entry in [*datasets, *manifest_entries]:
            entry["solve"] = _solve_block(_SKIPPED, "Solve waits until Judge passes.")
    else:
        solve_checks = _solve(scan.datasets, datasets, manifests, manifest_entries, runtime, repo_root)
        statuses = [entry["solve"]["status"] for entry in [*datasets, *manifest_entries]]
        if _FAILED in statuses:
            phases["solve"] = _INVALID
        elif _PASSED in statuses:
            phases["solve"] = _VALID
        else:
            reasons = dict.fromkeys(entry["solve"]["reason"] for entry in [*datasets, *manifest_entries])
            solve_checks.append(
                check(
                    "solve",
                    "solve",
                    WARN,
                    "Solve ran nothing end to end. {}".format(" ".join(reason for reason in reasons if reason)),
                    severity=ADVISORY,
                    hint="Fix what this names, then rerun discovery to prove the evals run.",
                    proven=False,
                )
            )
    for entry in datasets:
        entry["runnable"] = proven and not required_failures(entry["_checks"])
        # Harbor cannot run Gym extension tasks as written, so only a plain Harbor dataset gets a run command.
        if entry["runnable"] and not entry["formats"].keys() - {_HARBOR_FORMAT}:
            entry["run_command"] = "cd {} && harbor run -p {} -a <agent>".format(repo_root, entry["path"])
        entry["checks"] = [result.as_dict() for result in entry["_checks"]]
        entry.pop("_judged", None)

    # Harbor configs and datasets need Harbor's verdict; manifests need Gym's.
    runnable = (
        phases["judge"] == _VALID
        and phases["solve"] != _INVALID
        and (proven or not harbor_found)
        and all(config["runnable"] for config in configs)
        and all(dataset["runnable"] for dataset in datasets)
    )
    other_evals, other_eval_scan = _other_evals.find(repo_root, owned)
    report.update(
        {
            "runnable": runnable,
            "configs": configs,
            "datasets": datasets,
            "task_count": sum(dataset["task_count"] for dataset in datasets),
            "gym_manifests": manifest_entries,
            "other_eval_candidates": other_evals,
            "other_eval_scan": other_eval_scan,
            "ethos_path": scan.ethos_path,
            "fingerprint": "sha256:{}".format(scan.fingerprint),
            "input_file_count": scan.input_file_count,
            "run_command": _run_command(repo_root, configs),
        }
    )
    judged = [item for entry in [*configs, *datasets] for item in entry.pop("_checks")]
    # Datasets share one backend preflight, so list each check object once.
    checks = [
        *runtime_checks,
        *repository_checks,
        *{id(item): item for item in judged}.values(),
        *gym_judge_checks,
        *solve_checks,
    ]
    return _emit(report, checks, args.compact)


def _emit(report: dict, checks: list[CheckResult], compact: bool) -> int:
    report["discovered_at"] = datetime.now(timezone.utc).isoformat()
    report["checks"] = [result.as_dict() for result in checks]
    json.dump(report, sys.stdout, indent=None if compact else 2)
    sys.stdout.write("\n")
    return 0 if report["runnable"] else 1


def main(argv: list[str] | None = None) -> int:
    """Run discovery and print the JSON report."""
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    sys.exit(main())
