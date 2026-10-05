#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render discovery results for people, with the original evidence below."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _required_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in report.get("checks", []) if c.get("status") == "fail" and c.get("severity") == "required"]


def _docker_failed(report: dict[str, Any]) -> bool:
    """Whether Harbor's Docker preflight failed, which blocks every Harbor eval at once."""
    return any(
        c.get("name") == "backend" and "docker" in str(c.get("message", "")).lower() for c in _required_failures(report)
    )


def _action(check: dict[str, Any]) -> str:
    name = check.get("name")
    message = str(check.get("message", "")).lower()
    if name == "backend" and "docker" in message:
        return (
            "Launch Docker if it isn't running, then rerun discovery. If Docker is not installed, "
            "[install Docker](https://docs.docker.com/get-started/get-docker/) first. If `docker info` works "
            "but discovery still fails, this session may be sandboxed: rerun discovery with permission to access "
            "Docker in that same environment."
        )
    return {
        "backend": "Check the evaluation environment's setup, then rerun the readiness check.",
        "config": "Confirm where the Harbor configuration lives; discovery searches up to four directories deep.",
        "dataset-tasks": "Repair the task files identified in the diagnostic details below.",
        "gym-conversion": "Fix the Gym task files identified in the diagnostic details below.",
        "gym-runner": "Give each Gym extension task a `tests/test.sh`, or wait for NeMo Gym's Harbor task runner.",
        "gym-validate": "Rerun the rejected manifest with the command in the diagnostic details below, and fix what Gym reports.",
        "solve": "Rerun the failing eval with the command in the diagnostic details below, and fix what it reports.",
        "config-parse": "Check the configuration's syntax and Python dependencies; see the diagnostic details below.",
        "schema": "Correct the configuration fields identified in the diagnostic details below.",
        "resolution": "Check the dataset paths and job settings in the affected configuration, then rerun the readiness check.",
        "credentials": "Set the required environment variables for the affected configuration, then rerun the readiness check.",
        "agent": "Check that the configured agent is available in this Python environment.",
        "tasks": "Repair the task files identified in the diagnostic details below.",
        "coverage": "Check the task directories Harbor skipped before running the evaluations.",
        "round-trip": "Correct the configuration that the Harbor CLI rejected; see the diagnostic details below.",
        "compatibility": "Use a Harbor version that supports the validation checks, then check again.",
    }.get(name, "Review the diagnostic details below and resolve the reported problem before checking again.")


def _harbor_setup_guidance(report: dict[str, Any]) -> str:
    runtime = report.get("runtime", {})
    if runtime.get("harbor_importable") is not False:
        return ""
    preparation = (
        "We can still inspect your eval material and clarify requirements and grading rules. "
        "Creating native task files, validating them, and running evals need a working Harbor installation."
    )
    if runtime.get("harbor_cli"):
        return (
            "A Harbor command was found, but Harbor is unavailable in the environment checked. "
            "Check the existing Harbor installation and use its working environment before rerunning discovery. "
            + preparation
        )
    return (
        "Harbor is unavailable in the environment checked. "
        + preparation
        + " If you already have Harbor installed elsewhere, use that environment. Otherwise, follow "
        "[Harbor's setup guide](https://www.harborframework.com/docs/getting-started): "
        "with uv available, run `uv tool install harbor`, then `harbor --help`. "
        "If uv is missing, use the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). "
        "After setup, we'll verify Harbor and resume from the saved findings."
    )


_GYM_SETUP = "To work from NeMo Gym environments, install `nemo-gym>=0.6.0` (Python 3.13.14 or newer)."
_MAX_SOLVED = 4
_RUNTIMES = (("Harbor", "harbor_importable"), ("NeMo Gym", "gym_available"))


_CELEBRATION = "🎉🎉🎉 Every phase passed and every eval Solve ran earned a reward: your evals are operational! 🎉🎉🎉"


def _fully_operational(report: dict[str, Any]) -> bool:
    """Whether all four phases passed and no eval Solve ran came back with a zero reward."""
    phases = report.get("phases") or {}
    return (
        bool(phases)
        and all(status == "valid" for status in phases.values())
        and bool(report.get("runnable"))
        and not any(c.get("name") == "solve-reward" and c.get("status") == "warn" for c in report.get("checks", []))
    )


def _judge_cell(entry: dict[str, Any]) -> str:
    """Describe what Gym's Judge did with one manifest."""
    judge = entry.get("judge") or {}
    status = judge.get("status")
    if status == "passed":
        return "Valid"
    if status == "failed":
        return "Invalid"
    if status == "blocked":
        return f"Blocked: {judge['reason']}"
    return f"Skipped: {judge['reason']}" if judge.get("reason") else "Not run"


def _solve_cell(entry: dict[str, Any]) -> str:
    """Describe what Solve did with one dataset or manifest."""
    solve = entry.get("solve")
    if not solve:
        return "Not run"
    status = solve.get("status")
    results = solve.get("results") or []
    if status == "passed":
        return f"Ran {len(results)} sampled" if results and "task" in results[0] else "Verifier ran"
    if status == "failed":
        return "Failed"
    if status == "not sampled":
        return "Not sampled"
    return f"Skipped: {solve.get('reason')}" if solve.get("reason") else "Skipped"


def _other_evals_note(report: dict[str, Any]) -> str:
    candidates = report.get("other_eval_candidates") or []
    if not candidates:
        return ""
    return (
        "I also found code that looks like evaluations in another format: "
        + ", ".join(f"`{c['path']}`" for c in candidates)
        + ". Do you want to convert any of it into Harbor tasks?"
        + _partial_scan_note(report)
    )


def _partial_scan_note(report: dict[str, Any]) -> str:
    """Say when the other-evals scan stopped early on a large repository, so there may be more."""
    scan = report.get("other_eval_scan") or {}
    if scan.get("complete", True):
        return ""
    return f" I stopped looking after {scan['files_scanned']:,} files, so there may be more."


def _with_note(text: str, report: dict[str, Any]) -> str:
    note = _other_evals_note(report)
    return f"{text}\n\n{note}" if note else text


def _units_noun(configs: list[dict[str, Any]], datasets: list[dict[str, Any]]) -> str:
    if configs and datasets:
        return "configurations and datasets"
    return "configurations" if configs else "datasets"


def render_summary(report: dict[str, Any]) -> str:
    """Return the short user reply, also used at the top of the saved report."""
    if "error" in report:
        return "\n\n".join(
            ["I could not inspect this repository.", str(report["error"]), str(report.get("hint", ""))]
        ).strip()
    configs = report.get("configs", [])
    datasets = report.get("datasets", [])
    manifests = report.get("gym_manifests") or []
    found = bool(configs or report.get("task_count") or datasets or manifests)
    proven = bool(report.get("proven"))
    setup = _harbor_setup_guidance(report)
    if report.get("phases", {}).get("probe") == "invalid":
        return (
            "Neither [Harbor](https://www.harborframework.com/docs) nor NeMo Gym is installed in the "
            "environment I checked, so I stopped before looking for evals.\n\n"
            + (setup + " " if setup else "")
            + _GYM_SETUP
        )
    if not found:
        runtime = report.get("runtime", {})
        installed = [name for name, key in _RUNTIMES if runtime.get(key)]
        opening = (
            f"You have {' and '.join(installed)} installed, but it doesn't look like you have any evals "
            "in the locations I checked."
            if installed
            else "It doesn't look like you have any evals in the locations I checked."
        )
        note = _other_evals_note(report)
        return (
            opening
            + " You may still have other kinds of evaluations we can work from.\n\n"
            + (setup + "\n\n" if setup else "")
            + (note + "\n\n" if note else "")
            + "Do you already have evals in any form, such as tests, scripts, a dataset, a notebook, "
            "or a manual checklist? Can you point me to them?"
        )
    if not (configs or datasets):
        return _with_note(_gym_readiness(report, manifests), report)
    if not proven:
        return _with_note(
            "I found possible Harbor evals, but readiness has not been checked.\n\n"
            + (
                setup
                or "Use the Python environment for this suite with Harbor installed, then rerun the readiness check."
            )
            + (f"\n\n{_gym_readiness(report, manifests)}" if manifests else ""),
            report,
        )
    readiness = _readiness(report, configs, datasets)
    return _with_note(readiness + (f"\n\n{_gym_readiness(report, manifests)}" if manifests else ""), report)


def _gym_readiness(report: dict[str, Any], manifests: list[dict[str, Any]]) -> str:
    """Summarize a repository's NeMo Gym manifests, which Judge and Solve check with Gym."""

    def with_status(stage: str, *statuses: str) -> list[dict[str, Any]]:
        return [m for m in manifests if (m.get(stage) or {}).get("status") in statuses]

    headline = f"This repo has {len(manifests)} NeMo Gym environment or benchmark manifest{'s' if len(manifests) != 1 else ''}."
    judged = with_status("judge", "passed", "failed")
    blocked = with_status("judge", "blocked")
    if blocked:
        reason = (blocked[0].get("judge") or {}).get("reason") or "NeMo Gym is not available."
        return f"{headline} I could not check {'it' if len(manifests) == 1 else 'them'}: {reason}\n\n{_GYM_SETUP}"
    if not judged:
        return (
            f"{headline} I could not check any of them; the most common reason is missing data. "
            "The NeMo Gym Manifests table below gives each one's reason."
        )
    parts = [headline]
    rejected = with_status("judge", "failed")
    parts.append(
        f"NeMo Gym validated {len(judged) - len(rejected)} of the {len(judged)} that have their data"
        + (": it rejected " + ", ".join(f"`{m['path']}`" for m in rejected) + "." if rejected else ".")
    )
    solved = with_status("solve", "passed", "failed")
    broken = with_status("solve", "failed")
    if solved:
        parts.append(
            f"I ran {len(solved)} of them end to end with `gym env test`"
            + (", and " + ", ".join(f"`{m['path']}`" for m in broken) + " failed." if broken else "; all ran.")
        )
    if len(judged) < len(manifests):
        parts.append(f"{len(manifests) - len(judged)} were not checked; the table below says why.")
    return " ".join(parts)


def _readiness(report: dict[str, Any], configs: list[dict[str, Any]], datasets: list[dict[str, Any]]) -> str:
    units = [*configs, *datasets]
    noun = _units_noun(configs, datasets)
    ready = [u for u in units if u.get("runnable")]
    if len(ready) == len(units):
        headline = "This repo has Harbor evals, and they are ready to run."
        if report.get("run_command") and not datasets:
            return f"{headline}\n\nRun the evals with:\n\n```bash\n{report['run_command']}\n```"
        parts = [headline]
        if configs:
            parts.append(
                "Each discovered config is ready. Pick the config you want to run:\n\n"
                + "\n".join(f"- `{c['path']}`" for c in configs)
            )
        if datasets:
            parts.append(
                "Each dataset runs directly by path. Pick a dataset and an agent:\n\n"
                + "\n".join(f"- `{d['run_command']}`" for d in datasets)
            )
        return "\n\n".join(parts)
    if ready:
        headline = f"This repo has Harbor evals: {len(ready)} of {len(units)} {noun} are ready to run."
        headline += "\n\nYou can choose a ready one: " + ", ".join(f"`{u['path']}`" for u in ready) + "."
        headline += "\n\nFor the blocked ones:"
    elif _docker_failed(report):
        headline = (
            "This repo has Harbor evals, but I could not verify readiness because the Docker preflight check failed."
        )
    else:
        headline = f"This repo has Harbor evals, but none of the {len(units)} {noun} is ready to run yet."
    # Group common actions so one unavailable service does not produce a wall of failures.
    # The NeMo Gym paragraph that follows covers a missing Gym.
    actions = list(dict.fromkeys(_action(c) for c in _required_failures(report) if c.get("name") != "gym-unavailable"))
    return headline + "\n\n" + "\n".join(f"- {action}" for action in actions)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _phase_detail(name: str, status: str, report: dict[str, Any]) -> str:
    """Say in a few words what one phase found; a skipped phase did nothing."""
    if status == "skipped":
        return "N/A"
    valid = status == "valid"
    if name == "probe":
        installed = [runtime for runtime, key in _RUNTIMES if (report.get("runtime") or {}).get(key)]
        return f"found {' and '.join(installed)} installed" if valid else "found neither Harbor nor NeMo Gym installed"
    if name == "explore":
        if not valid:
            return "found no evals"
        tasks = report.get("task_count", 0) + sum(m.get("task_count", 0) for m in report.get("gym_manifests", []))
        if tasks:
            return f"found {_plural(tasks, 'task')}"
        # No task rows yet, such as a config for a remote dataset or an unprepared benchmark.
        found = [
            _plural(len(report.get(key) or []), noun)
            for key, noun in (("configs", "config"), ("gym_manifests", "Gym manifest"))
            if report.get(key)
        ]
        return f"found {' and '.join(found)} but no tasks"
    if name == "judge":
        verdicts = _judge_verdicts(report)
        # A failed Docker preflight blocks every Harbor eval, and the summary says how to fix it.
        if _docker_failed(report):
            return "the Docker preflight check failed"

        def counted(passed: bool) -> str:
            return " and ".join(
                _plural(results.count(passed), f"{runtime} eval")
                for runtime, results in verdicts.items()
                if passed in results
            )

        validated, failed = counted(True), counted(False)
        if not failed:
            return f"validated {validated}"
        return (
            f"validated {validated}, but {failed} could not be validated"
            if validated
            else f"{failed} could not be validated"
        )
    if name == "solve":
        runs = [
            result.get("error") is None
            for entry in [*report.get("datasets", []), *report.get("gym_manifests", [])]
            for result in (entry.get("solve") or {}).get("results") or []
        ]
        return f"{runs.count(True)} of {len(runs)} sampled evals ran"
    return ""


def _judge_verdicts(report: dict[str, Any]) -> dict[str, list[bool]]:
    """Whether each eval Judge checked passed, by format: Harbor configs and datasets when proven, and Gym manifests."""
    harbor: list[bool] = []
    if report.get("proven"):
        harbor += [bool(c.get("runnable")) for c in report.get("configs", [])]
        # Solve's checks, including gym-runner, are not Judge verdicts.
        harbor += [
            not any(
                c.get("status") == "fail" and c.get("severity") == "required" and c.get("group") != "solve"
                for c in d.get("checks", [])
            )
            for d in report.get("datasets", [])
        ]
    gym = [
        (m.get("judge") or {}).get("status") == "passed"
        for m in report.get("gym_manifests", [])
        if (m.get("judge") or {}).get("status") in {"passed", "failed", "blocked"}
    ]
    return {"Harbor": harbor, "Gym": gym}


def _phase_line(name: str, status: str, report: dict[str, Any]) -> str:
    """Render one phase as a checklist line, checked only when the phase is valid."""
    box = "x" if status == "valid" else " "
    return f"- [{box}] **{name.capitalize()}** ({status}) - {_phase_detail(name, status, report)}"


def render_report(report: dict[str, Any], *, evidence: str | None = None) -> str:
    """Return the saved report, preserving input JSON bytes when supplied by the CLI."""
    lines = ["# Eval Discovery", "", render_summary(report), ""]
    phases = report.get("phases")
    if phases:
        lines.extend([*(_phase_line(name, status, report) for name, status in phases.items()), ""])
    if _fully_operational(report):
        lines.extend([_CELEBRATION, ""])
    explored = (phases or {}).get("explore") != "skipped" or (phases or {}).get("probe") != "invalid"
    if "error" not in report and explored:
        proven = bool(report.get("proven"))
        configs = report.get("configs", [])
        datasets = report.get("datasets", [])
        # Each section appears only when it has rows; the summary already says when nothing was found.
        if configs:
            lines.extend(["## Configs", "", "| Configuration | Readiness | Required host variables |", "|---|---|---|"])
            for config in configs:
                status = ("Ready" if config.get("runnable") else "Blocked") if proven else "Not checked"
                credentials_checked = proven and any(c.get("name") == "credentials" for c in config.get("checks", []))
                env = ", ".join(f"`{v['name']}`" for v in config.get("required_env_vars", []))
                env = (env or "None") if credentials_checked else "Not checked"
                lines.append(f"| `{config['path']}` | {status} | {env} |")
            lines.append("")
        if datasets:
            lines.extend(
                [
                    "## Datasets",
                    "",
                    "| Dataset | Format | Tasks | Readiness | Solve | Theme |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for dataset in datasets:
                status = ("Ready" if dataset.get("runnable") else "Blocked") if proven else "Not checked"
                formats = ", ".join(f"{name} ({count})" for name, count in dataset.get("formats", {}).items())
                theme = dataset.get("theme", {})
                labels = [*theme.get("categories", {}), *theme.get("tags", {}), *theme.get("keywords", {})]
                summary = ", ".join(dict.fromkeys(labels)) or "None declared"
                lines.append(
                    f"| `{dataset['path']}` | {formats} | {dataset.get('task_count', 0)} | {status} "
                    f"| {_solve_cell(dataset)} | {summary} |"
                )
            task_count = report.get("task_count", 0)
            lines.extend(
                [
                    "",
                    f"Found {task_count} task{'s' if task_count != 1 else ''} in {len(datasets)} "
                    f"dataset{'s' if len(datasets) != 1 else ''} on disk.",
                    "",
                ]
            )
        manifests = report.get("gym_manifests") or []
        if manifests:
            lines.extend(
                [
                    "## NeMo Gym Manifests",
                    "",
                    "NeMo Gym judges these with `gym env validate`, and Solve runs up to "
                    f"{_MAX_SOLVED} of them with `gym env test`.",
                    "",
                    "| Manifest | Kind | Domain | Tasks | Judge | Solve |",
                    "|---|---|---|---|---|---|",
                ]
            )
            lines.extend(
                f"| `{m['path']}` | {m.get('kind') or 'Unknown'} | {m.get('domain') or 'None declared'} "
                f"| {m.get('task_count', 0)}{' (data missing)' if m.get('missing_data') else ''} "
                f"| {_judge_cell(m)} | {_solve_cell(m)} |"
                for m in manifests
            )
            lines.append("")
        candidates = report.get("other_eval_candidates") or []
        if candidates:
            lines.extend(["## Other Possible Evals", ""])
            lines.extend(f"- `{c['path']}`: {', '.join(c['signals'])} ({', '.join(c['files'])})" for c in candidates)
            if _partial_scan_note(report):
                lines.extend(["", _partial_scan_note(report).strip()])
            lines.append("")
        # With nothing found, the summary already says so, and every check stays in the evidence below.
        found = bool(configs or datasets or manifests or report.get("task_count"))
        if found and (proven or (phases or {}).get("judge", "skipped") != "skipped"):
            # Top-level checks also contain config checks. Render each distinct diagnostic once with its owners.
            groups: dict[tuple[str, str, str], list[str]] = {}
            for unit in [*configs, *datasets]:
                for check in _required_failures(unit):
                    key = (check.get("name", "unknown"), check.get("message", ""), check.get("hint") or "")
                    groups.setdefault(key, []).append(unit["path"])
            for check in _required_failures(report):
                key = (check.get("name", "unknown"), check.get("message", ""), check.get("hint") or "")
                groups.setdefault(key, [])
            if groups:
                lines.extend(["## Diagnostic Details", ""])
            for (name, message, hint), paths in groups.items():
                lines.append(f"- `{name}`: {message}")
                if paths:
                    lines.append("  Affects: " + ", ".join(f"`{p}`" for p in dict.fromkeys(paths)))
                if hint:
                    lines.append(f"  Hint: {hint}")
            if groups:
                lines.append("")
            advisories = list(
                dict.fromkeys(
                    (c.get("name", "unknown"), c.get("message", ""), c.get("hint") or "")
                    for c in report.get("checks", [])
                    if c.get("severity") == "advisory" and c.get("status") in {"warn", "fail"}
                )
            )
            if advisories:
                lines.extend(["## Advisories", ""])
            for name, message, hint in advisories:
                lines.append(f"- `{name}`: {message}")
                if hint:
                    lines.append(f"  Hint: {hint}")
            if advisories:
                lines.append("")
    lines.extend(
        [
            "## Evidence JSON",
            "",
            "```json",
            evidence if evidence is not None else json.dumps(report, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", help="JSON file; defaults to stdin.")
    parser.add_argument("--summary", action="store_true", help="Print only the user-facing reply.")
    args = parser.parse_args(argv)
    source = Path(args.report).read_text(encoding="utf-8") if args.report else sys.stdin.read()
    report = json.loads(source)
    sys.stdout.write(render_summary(report) + "\n" if args.summary else render_report(report, evidence=source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
