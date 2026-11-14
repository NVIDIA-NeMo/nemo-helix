# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the nightly taskset-runs e2e report.

Checks that red runs open or update the tracking issue with the failure stage and taskset run IDs, then
redispatch the same commit once. Classification runs label the issue ``flaky`` or ``regression`` without
closing it or dispatching again. Green runs close an open issue unless a step used more than half its cap.

Also checks artifact loading, missing-evidence fallbacks, the exit code for red reports, and ``gh api`` error
messages. Uses a fake GitHub client and an injected subprocess runner; no real issues or workflows are changed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "nightly_taskset_runs_report.py"
_spec = importlib.util.spec_from_file_location("nightly_taskset_runs_report", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
report_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = report_module  # its dataclass looks itself up there
_spec.loader.exec_module(report_module)

SHA = "a" * 40
REPO = "NVIDIA-NeMo/nemo-helix"
GREEN = {"build-images": {"result": "success"}, "kind-taskset-runs-e2e": {"result": "success"}}
RED_TEST = {"build-images": {"result": "success"}, "kind-taskset-runs-e2e": {"result": "failure"}}
RED_BUILD = {"build-images": {"result": "failure"}, "kind-taskset-runs-e2e": {"result": "skipped"}}
JUNIT_DISPATCH = '<testcase><failure message="AssertionError: [dispatch] run run-1 still running after 900 s"/>'


class FakeGitHub:
    """Records each call; the issue search returns ``open`` and creating an issue returns number 7."""

    def __init__(self, open_issue: int | None = None, older_open: tuple[int, ...] = ()) -> None:
        self.open_issue = open_issue
        self.older_open = older_open
        self.calls: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def __call__(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        self.calls.append((method, path, body))
        if method == "GET":
            numbers = ([self.open_issue] if self.open_issue is not None else []) + list(self.older_open)
            return [{"number": number} for number in numbers]
        if path == f"repos/{REPO}/issues":
            return {"number": 7}
        return None

    def writes(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _ in self.calls if method != "GET"]

    def last_body(self) -> Mapping[str, Any]:
        body = self.calls[-1][2]
        assert body is not None
        return body


def make_run(needs: Mapping[str, Any], classify_for: str = "") -> Any:
    return report_module.Run(
        repo=REPO,
        server_url="https://github.com",
        run_id="100",
        source_sha=SHA,
        needs=needs,
        classify_for=classify_for,
    )


def test_failure_stage_names_the_failed_job_before_the_test_else_the_tests_own_stage() -> None:
    assert report_module.failure_stage(RED_BUILD, "") == "build-images"
    assert report_module.failure_stage(RED_TEST, "") == "setup"
    assert report_module.failure_stage(RED_TEST, JUNIT_DISPATCH) == "dispatch"
    assert report_module.failure_stage(RED_TEST, '<failure message="boom"/>') == "test"


def test_slow_steps_flags_steps_over_half_their_cap() -> None:
    evidence = [{"test_1": {"durations_s": {"submit": 100, "run": 600}, "caps_s": {"submit": 900, "run": 900}}}]

    assert report_module.slow_steps(evidence) == ["test_1.run: 600 of 900 s"]
    assert report_module.slow_steps([{}]) == []


def test_a_green_scheduled_run_closes_the_open_issue() -> None:
    github = FakeGitHub(open_issue=3)

    assert report_module.report(github, make_run(GREEN), [], "") == {"action": "closed"}
    assert github.writes() == [("POST", f"repos/{REPO}/issues/3/comments"), ("PATCH", f"repos/{REPO}/issues/3")]
    assert github.last_body() == {"state": "closed"}


def test_a_green_run_with_no_issue_does_nothing() -> None:
    github = FakeGitHub()

    assert report_module.report(github, make_run(GREEN), [], "") == {"action": "none"}
    assert github.writes() == []


def test_a_slow_green_run_warns_on_the_issue_instead_of_closing_it() -> None:
    github = FakeGitHub(open_issue=3)
    evidence = [{"test_2": {"durations_s": {"run": 400}, "caps_s": {"run": 600}}}]

    assert report_module.report(github, make_run(GREEN), evidence, "")["action"] == "slow"
    assert github.writes() == [("POST", f"repos/{REPO}/issues/3/comments")]
    assert "test_2.run: 400 of 600 s" in github.last_body()["body"]


def test_a_red_scheduled_run_opens_the_issue_with_the_stage_and_dispatches_one_classification_run() -> None:
    github = FakeGitHub()
    evidence = [{"test_1": {"run_id": "run-1"}}]

    outcome = report_module.report(github, make_run(RED_TEST), evidence, JUNIT_DISPATCH)

    assert outcome == {"action": "reported", "stage": "dispatch"}
    assert github.writes() == [
        ("POST", f"repos/{REPO}/issues"),
        ("POST", f"repos/{REPO}/issues/7/labels"),
        ("POST", f"repos/{REPO}/actions/workflows/nightly-taskset-runs-e2e.yaml/dispatches"),
    ]
    _, _, opened = github.calls[1]
    assert opened is not None
    assert "Stage: `[dispatch]`" in opened["body"] and "Taskset runs: run-1" in opened["body"]
    assert github.calls[2][2] == {"labels": ["stage:dispatch"]}
    assert github.last_body() == {"ref": "main", "inputs": {"classify_for": "100", "source_sha": SHA}}


def test_a_red_run_without_evidence_still_reports_the_build_stage() -> None:
    github = FakeGitHub(open_issue=3)

    assert report_module.report(github, make_run(RED_BUILD), [], "")["stage"] == "build-images"
    assert github.writes() == [
        ("POST", f"repos/{REPO}/issues/3/comments"),
        ("POST", f"repos/{REPO}/issues/3/labels"),
        ("POST", f"repos/{REPO}/actions/workflows/nightly-taskset-runs-e2e.yaml/dispatches"),
    ]


def test_a_green_classification_run_labels_the_issue_flaky_and_never_closes_it() -> None:
    github = FakeGitHub(open_issue=3)

    assert report_module.report(github, make_run(GREEN, "99"), [], "") == {"action": "classified", "verdict": "flaky"}
    assert github.writes() == [("POST", f"repos/{REPO}/issues/3/comments"), ("POST", f"repos/{REPO}/issues/3/labels")]
    assert github.last_body() == {"labels": ["flaky"]}


def test_a_red_classification_run_labels_the_issue_a_regression_and_dispatches_nothing() -> None:
    github = FakeGitHub(open_issue=3)

    outcome = report_module.report(github, make_run(RED_TEST, "99"), [], "")

    assert outcome == {"action": "classified", "verdict": "regression"}
    assert github.last_body() == {"labels": ["regression"]}
    assert not any(path.endswith("/dispatches") for _, path in github.writes())


def test_main_reads_the_artifacts_and_fails_a_red_report(tmp_path: Path) -> None:
    nested = tmp_path / "kind-taskset-runs-e2e-1-kubernetes-artifacts"
    nested.mkdir()
    (nested / f"taskset-runs-e2e-{'0' * 32}.json").write_text(json.dumps({"test_1": {"run_id": "run-1"}}))
    (nested / "junit.xml").write_text(JUNIT_DISPATCH)
    (nested / "other.json").write_text("{}")  # not evidence
    github = FakeGitHub()
    env = {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "100",
        "SOURCE_SHA": SHA,
        "NEEDS_JSON": json.dumps(RED_TEST),
        "ARTIFACTS": str(tmp_path),
    }

    assert report_module.main(env, github) == 1
    _, _, opened = github.calls[1]
    assert opened is not None
    assert "Stage: `[dispatch]`" in opened["body"] and "Taskset runs: run-1" in opened["body"]


JUNIT_DENIED = (
    '<testsuite><testcase><failure message="AssertionError: [image-build] task-image-build-x-1: '
    "PermissionDeniedError: HTTP 403: Not allowed to read entities in workspace 'ws'\"/></testcase></testsuite>"
)
KNOWN = {"row": "auth-workload", "bug": "NVBug 7083853", "url": "https://nvbugspro.nvidia.com/bug/7083853"}


def _artifact(root: Path, name: str, junit: str, known: bool = False) -> None:
    nested = root / name
    nested.mkdir()
    (nested / "junit.xml").write_text(junit)
    if known:
        (nested / "known-failure.json").write_text(json.dumps(KNOWN))


def test_a_known_auth_workload_failure_does_not_hide_an_auth_off_regression(tmp_path: Path) -> None:
    """The known row's JUnit names [image-build] first; the report must name auth-off's own stage."""
    _artifact(tmp_path, "kind-taskset-runs-e2e-auth-workload-1-kubernetes-artifacts", JUNIT_DENIED, known=True)
    _artifact(tmp_path, "kind-taskset-runs-e2e-auth-off-1-kubernetes-artifacts", JUNIT_DISPATCH)
    github = FakeGitHub()
    env = {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "100",
        "SOURCE_SHA": SHA,
        "NEEDS_JSON": json.dumps(RED_TEST),  # the auth-off leg failed the matrix job
        "ARTIFACTS": str(tmp_path),
    }

    assert report_module.main(env, github) == 1
    _, _, opened = github.calls[1]
    assert opened is not None
    assert "Stage: `[dispatch]`" in opened["body"]
    assert "Known failure, reported only: auth-workload: [NVBug 7083853]" in opened["body"]


def test_a_green_run_with_a_known_failure_lists_it_when_closing(tmp_path: Path) -> None:
    _artifact(tmp_path, "kind-taskset-runs-e2e-auth-workload-1-kubernetes-artifacts", JUNIT_DENIED, known=True)
    evidence, junit, known = report_module.read_artifacts(tmp_path)
    github = FakeGitHub(open_issue=3)

    assert junit == "" and known == [KNOWN]
    assert report_module.report(github, make_run(GREEN), evidence, junit, known) == {"action": "closed"}
    assert "Known failure, reported only: auth-workload" in github.calls[1][2]["body"]


def test_main_without_artifacts_reports_the_setup_stage() -> None:
    github = FakeGitHub()
    env = {"GITHUB_REPOSITORY": REPO, "GITHUB_RUN_ID": "100", "SOURCE_SHA": SHA, "NEEDS_JSON": json.dumps(RED_TEST)}

    assert report_module.main(env, github) == 1
    assert github.calls[2][2] == {"labels": ["stage:setup"]}


def test_gh_api_failure_names_the_api_error() -> None:
    def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, "", "gh: Resource not accessible by integration (HTTP 403)")

    with pytest.raises(RuntimeError, match=r"POST repos/o/r/issues exited 1: .*HTTP 403"):
        report_module.gh_api("POST", "repos/o/r/issues", {"title": "t"}, run=failing_run)


def test_more_than_one_open_issue_warns_and_reports_on_the_newest(capsys: pytest.CaptureFixture[str]) -> None:
    github = FakeGitHub(open_issue=5, older_open=(3,))

    report_module.report(github, make_run(RED_TEST), [], "")

    assert github.writes()[0] == ("POST", f"repos/{REPO}/issues/5/comments")
    assert "::warning::2 open issues are labelled nightly-taskset-runs-e2e; reporting on #5, not #3" in (
        capsys.readouterr().out
    )


def test_under_act_main_prints_the_calls_instead_of_making_them(capsys: pytest.CaptureFixture[str]) -> None:
    env = {
        "ACT": "true",
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "100",
        "SOURCE_SHA": SHA,
        "NEEDS_JSON": json.dumps(RED_TEST),
    }

    assert report_module.main(env) == 1  # same exit status as a real red report

    out = capsys.readouterr().out
    assert f"[dry run] GET repos/{REPO}/issues?labels=nightly-taskset-runs-e2e" in out
    assert f"[dry run] POST repos/{REPO}/issues/0/labels" in out
    assert f"[dry run] POST repos/{REPO}/actions/workflows/nightly-taskset-runs-e2e.yaml/dispatches" in out


def test_dry_run_api_answers_like_an_empty_repo() -> None:
    assert report_module.dry_run_api("GET", "repos/o/r/issues?state=open") == []
    assert report_module.dry_run_api("POST", "repos/o/r/issues", {"title": "t"}) == {"number": 0}
    assert report_module.dry_run_api("PATCH", "repos/o/r/issues/0", {"state": "closed"}) is None


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
                "-o",
                "addopts=",
                "-c",
                os.devnull,
                "-p",
                "no:cacheprovider",
                "--confcutdir",
                str(Path(__file__).parent),
            ]
        )
    )
