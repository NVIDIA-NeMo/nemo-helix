#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _action(check: dict[str, Any]) -> str:
    name = check.get("name")
    message = str(check.get("message", "")).lower()
    if name == "backend" and "docker" in message:
        return (
            "Check Docker access with `docker info`. If this session is sandboxed, retry with permission to access "
            "Docker, then rerun discovery in that same environment. Start Docker only if it is confirmed stopped."
        )
    return {
        "backend": "Check the evaluation environment's setup, then rerun the readiness check.",
        "config": "Confirm where the Harbor configuration lives; discovery searches up to four directories deep.",
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


def render_summary(report: dict[str, Any]) -> str:
    """Return the short user reply, also used at the top of the saved report."""
    if "error" in report:
        return "\n\n".join(
            ["I could not inspect this repository.", str(report["error"]), str(report.get("hint", ""))]
        ).strip()
    configs = report.get("configs", [])
    found = bool(configs or report.get("task_count") or report.get("dataset_paths"))
    proven = bool(report.get("proven"))
    setup = _harbor_setup_guidance(report)
    if not found:
        return (
            "Eval Author uses [Harbor](https://www.harborframework.com/docs) to run evals. "
            "It gives your agent a task, checks the result, and lets you repeat the same "
            "test after changes to see how your agent is doing.\n\n"
            "It doesn't look like you have any Harbor evals in the locations I checked. "
            "You may still have other kinds of evaluations we can work from. "
            "Let's first make sure we're starting with the right material.\n\n"
            + (setup + "\n\n" if setup else "")
            + "Do you already have evals in any form, such as tests, scripts, a dataset, a notebook, "
            "or a manual checklist? Can you point me to them?"
        )
    if not proven:
        return "I found possible Harbor evals, but readiness has not been checked.\n\n" + (
            setup or "Use the Python environment for this suite with Harbor installed, then rerun the readiness check."
        )
    if not configs:
        return (
            "I found Harbor task or dataset files, but no run configuration.\n\n"
            "Confirm where the Harbor configuration lives; discovery searches up to four directories deep."
        )
    ready = [c for c in configs if c.get("runnable")]
    if len(ready) == len(configs):
        headline = "This repo has Harbor evals, and they are ready to run."
        if report.get("run_command"):
            return f"{headline}\n\nRun the evals with:\n\n```bash\n{report['run_command']}\n```"
        return (
            headline
            + "\n\nEach discovered config is ready. Pick the config you want to run:\n\n"
            + "\n".join(f"- `{c['path']}`" for c in ready)
        )
    if ready:
        headline = f"This repo has Harbor evals: {len(ready)} of {len(configs)} configurations are ready to run."
        headline += "\n\nYou can choose a ready configuration: " + ", ".join(f"`{c['path']}`" for c in ready) + "."
        headline += "\n\nFor the blocked configurations:"
    elif any(
        c.get("name") == "backend" and "docker" in str(c.get("message", "")).lower() for c in _required_failures(report)
    ):
        headline = (
            "This repo has Harbor evals, but I could not verify readiness because the Docker preflight check failed."
        )
    else:
        headline = f"This repo has Harbor evals, but none of the {len(configs)} configurations is ready to run yet."
    # Group common actions so one unavailable service does not produce a wall of failures.
    actions = list(dict.fromkeys(_action(c) for c in _required_failures(report)))
    return headline + "\n\n" + "\n".join(f"- {action}" for action in actions)


def render_report(report: dict[str, Any], *, evidence: str | None = None) -> str:
    """Return the saved report, preserving input JSON bytes when supplied by the CLI."""
    lines = ["# Eval Discovery", "", render_summary(report), ""]
    if "error" not in report:
        proven = bool(report.get("proven"))
        configs = report.get("configs", [])
        lines.extend(["## Configs", "", "| Configuration | Readiness | Required host variables |", "|---|---|---|"])
        for config in configs:
            status = ("Ready" if config.get("runnable") else "Blocked") if proven else "Not checked"
            credentials_checked = proven and any(c.get("name") == "credentials" for c in config.get("checks", []))
            env = ", ".join(f"`{v['name']}`" for v in config.get("required_env_vars", []))
            env = (env or "None") if credentials_checked else "Not checked"
            lines.append(f"| `{config['path']}` | {status} | {env} |")
        if not configs:
            lines.append("| None found | Not checked | Not checked |")
        lines.extend(
            [
                "",
                f"Found {report.get('task_count', 0)} task directories and {len(report.get('dataset_paths', []))} dataset directories on disk.",
                "",
            ]
        )
        if proven:
            lines.extend(["## Diagnostic Details", ""])
            # Top-level checks also contain config checks. Render each distinct diagnostic once with its owners.
            groups: dict[tuple[str, str, str], list[str]] = {}
            for config in configs:
                for check in _required_failures(config):
                    key = (check.get("name", "unknown"), check.get("message", ""), check.get("hint") or "")
                    groups.setdefault(key, []).append(config["path"])
            for check in _required_failures(report):
                key = (check.get("name", "unknown"), check.get("message", ""), check.get("hint") or "")
                groups.setdefault(key, [])
            for (name, message, hint), paths in groups.items():
                lines.append(f"- `{name}`: {message}")
                if paths:
                    lines.append("  Affects: " + ", ".join(f"`{p}`" for p in dict.fromkeys(paths)))
                if hint:
                    lines.append(f"  Hint: {hint}")
            if not groups:
                lines.append("No required failures.")
            lines.extend(["", "## Advisories", ""])
            advisories = list(
                dict.fromkeys(
                    (c.get("name", "unknown"), c.get("message", ""), c.get("hint") or "")
                    for c in report.get("checks", [])
                    if c.get("severity") == "advisory" and c.get("status") in {"warn", "fail"}
                )
            )
            for name, message, hint in advisories:
                lines.append(f"- `{name}`: {message}")
                if hint:
                    lines.append(f"  Hint: {hint}")
            if not advisories:
                lines.append("None.")
    lines.extend(
        [
            "",
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
