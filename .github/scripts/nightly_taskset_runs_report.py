# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nightly taskset-runs e2e's report.

Keeps a single GitHub issue for the nightly test and posts each night's result there:

- Test failed -> open the issue, or comment on it if already open. Then rerun the same commit once: the rerun
  labels the failure ``flaky`` (passed this time) or ``regression`` (failed again).
- Test passed, but a step was slow -> comment on the issue, opening it if needed.
- Test passed -> comment "Green again" and close the issue; nothing to do if none is open.

Reads the workflow's environment (``NEEDS_JSON``, ``SOURCE_SHA``, ``CLASSIFY_FOR``, ``ARTIFACTS`` and GitHub's
defaults) and calls the API through ``gh api``, which authenticates with ``GH_TOKEN``. Exits 1 on a red report.

Under nektos/act (``ACT`` set) it calls nothing: each call is printed and answered with a harmless stand-in, so
the report runs locally without a token or side effects.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LABEL = "nightly-taskset-runs-e2e"
WORKFLOW = "nightly-taskset-runs-e2e.yaml"
SLOW_SHARE = 0.5  # a green run that used more than this share of a step's cap gets a warning
FAILING = "Nightly taskset runs e2e is failing"
SLOWING = "Nightly taskset runs e2e is slowing down"
EVIDENCE_NAME = re.compile(r"taskset-runs-e2e-[0-9a-f]{32}\.json$")
STAGE_IN_FAILURE = re.compile(r'<failure[^>]*message="[^"]*?\[([a-z][a-z-]*)\]')


class GitHub(Protocol):
    def __call__(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any: ...


def gh_api(
    method: str,
    path: str,
    body: Mapping[str, Any] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Any:
    """One REST call through ``gh api``; its JSON response, or ``None`` when it has no body."""
    out = run(
        ["gh", "api", "--method", method, path, *(["--input", "-"] if body is not None else [])],
        input=json.dumps(body) if body is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:  # CalledProcessError would hide gh's stderr, which holds the API's reason
        raise RuntimeError(f"gh api {method} {path} exited {out.returncode}: {out.stderr.strip()}")
    return json.loads(out.stdout) if out.stdout.strip() else None


def dry_run_api(method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
    """Print the call instead of making it: no open issue, and a created issue is number 0."""
    print(f"[dry run] {method} {path}" + (f" {json.dumps(body)}" if body is not None else ""))
    if method == "GET":
        return []
    return {"number": 0} if method == "POST" and path.endswith("/issues") else None


@dataclass(frozen=True)
class Run:
    """This workflow run, as the report sees it."""

    repo: str  # owner/name
    server_url: str
    run_id: str
    source_sha: str
    needs: Mapping[str, Mapping[str, Any]]
    classify_for: str = ""
    dispatch_ref: str = "main"

    def url(self, run_id: str) -> str:
        return f"{self.server_url}/{self.repo}/actions/runs/{run_id}"


def failure_stage(needs: Mapping[str, Mapping[str, Any]], junit: str) -> str:
    """The job that failed before the test, or the ``[stage]`` the test's own failure message names."""
    if (needs.get("build-images") or {}).get("result") != "success":
        return "build-images"
    if match := STAGE_IN_FAILURE.search(junit):
        return match.group(1)
    return "test" if junit else "setup"


def run_ids(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    return [e[step]["run_id"] for e in evidence for step in ("test_1", "test_2") if (e.get(step) or {}).get("run_id")]


def slow_steps(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    """Steps of green runs that used more than SLOW_SHARE of their cap: ``test_1.run: 610 of 900 s``."""
    slow = []
    for e in evidence:
        for step in ("test_1", "test_2"):
            caps = (e.get(step) or {}).get("caps_s") or {}
            for name, seconds in ((e.get(step) or {}).get("durations_s") or {}).items():
                if caps.get(name) and seconds > SLOW_SHARE * caps[name]:
                    slow.append(f"{step}.{name}: {seconds} of {caps[name]} s")
    return slow


def report(github: GitHub, run: Run, evidence: Sequence[Mapping[str, Any]], junit: str) -> dict[str, Any]:
    """File this run's outcome on the tracking issue; returns what it did."""
    green = all(job.get("result") == "success" for job in run.needs.values())
    stage = None if green else failure_stage(run.needs, junit)
    ids = run_ids(evidence)
    facts = "\n".join(
        [f"Run: {run.url(run.run_id)}", f"Commit: {run.source_sha}"]
        + ([f"Stage: `[{stage}]`"] if stage else [])
        + ([f"Taskset runs: {', '.join(ids)}"] if ids else [])
    )
    issues = github("GET", f"repos/{run.repo}/issues?labels={LABEL}&state=open&per_page=10") or []
    issue = issues[0]["number"] if issues else None  # the newest: the API lists newest first
    if len(issues) > 1:
        others = ", ".join(f"#{other['number']}" for other in issues[1:])
        print(f"::warning::{len(issues)} open issues are labelled {LABEL}; reporting on #{issue}, not {others}")

    def comment(title: str, body: str) -> int:
        """Comment on the open issue, or open one; returns its number."""
        if issue is not None:
            github("POST", f"repos/{run.repo}/issues/{issue}/comments", {"body": body})
            return issue
        return github("POST", f"repos/{run.repo}/issues", {"title": title, "body": body, "labels": [LABEL]})["number"]

    def label(number: int, labels: list[str]) -> None:
        github("POST", f"repos/{run.repo}/issues/{number}/labels", {"labels": labels})

    if run.classify_for:
        # The second run only labels the first red; it never closes the issue.
        verdict = "flaky" if green else "regression"
        again = "passed" if green else "failed"
        target = comment(
            FAILING,
            f"Classification of {run.url(run.classify_for)}: **{verdict}** (the same commit {again} again).\n\n{facts}",
        )
        label(target, [verdict])
        return {"action": "classified", "verdict": verdict}

    if green:
        if slow := slow_steps(evidence):
            # Still needs attention before it becomes a timeout: keep (or open) the issue rather than closing it.
            listed = "\n".join(f"- {s}" for s in slow)
            share = f"{SLOW_SHARE:.0%}"
            comment(SLOWING, f"Green, but steps used more than {share} of their caps:\n\n{listed}\n\n{facts}")
            return {"action": "slow", "slow": slow}
        if issue is None:
            return {"action": "none"}
        github("POST", f"repos/{run.repo}/issues/{issue}/comments", {"body": f"Green again.\n\n{facts}"})
        github("PATCH", f"repos/{run.repo}/issues/{issue}", {"state": "closed"})
        return {"action": "closed"}

    target = comment(FAILING, f"Red.\n\n{facts}")
    label(target, [f"stage:{stage}"])
    # One more run of the same commit decides flaky or regression; the red above stands either way.
    github(
        "POST",
        f"repos/{run.repo}/actions/workflows/{WORKFLOW}/dispatches",
        {"ref": run.dispatch_ref, "inputs": {"classify_for": run.run_id, "source_sha": run.source_sha}},
    )
    return {"action": "reported", "stage": stage}


def read_artifacts(root: Path) -> tuple[list[dict[str, Any]], str]:
    """The evidence JSONs and the JUnit XML, concatenated, under ``root``; none when it doesn't exist."""
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    evidence = [json.loads(path.read_text(encoding="utf-8")) for path in files if EVIDENCE_NAME.search(path.name)]
    junit = "\n".join(path.read_text(encoding="utf-8") for path in files if path.suffix == ".xml")
    return evidence, junit


def main(env: Mapping[str, str] = os.environ, github: GitHub | None = None) -> int:
    github = github or (dry_run_api if env.get("ACT") else gh_api)
    run = Run(
        repo=env["GITHUB_REPOSITORY"],
        server_url=env.get("GITHUB_SERVER_URL", "https://github.com"),
        run_id=env["GITHUB_RUN_ID"],
        source_sha=env["SOURCE_SHA"],
        needs=json.loads(env["NEEDS_JSON"]),
        classify_for=env.get("CLASSIFY_FOR", ""),
        dispatch_ref=env.get("DISPATCH_REF") or "main",
    )
    evidence, junit = read_artifacts(Path(env["ARTIFACTS"])) if env.get("ARTIFACTS") else ([], "")
    outcome = report(github, run, evidence, junit)
    print(json.dumps(outcome))
    if outcome["action"] == "reported":
        print(f"::error::nightly taskset runs e2e failed at [{outcome['stage']}]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
