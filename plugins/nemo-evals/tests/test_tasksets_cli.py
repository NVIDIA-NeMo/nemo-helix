# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals tasksets``: create, build-images (request, --status, --wait), and the build loop they share."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import typer
from nemo_evals.image_builds.models import BuildImage, ImageBuild
from nemo_evals.runs_cli import (
    BUILD_READ_FAILURES,
    BUILD_REPAIR_SECONDS,
    CreatedTaskset,
    parse_dataset,
    qualify_ref,
    wait_for_builds,
)
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.tasksets_cli import STATUS_BUILDING, add_tasksets_commands
from nemo_helix_plugin.client.errors import NemoTransportError
from typer.testing import CliRunner

DIGEST = "1" * 64
T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _build(name: str, build_id: str, status: str = "pending", platform: str | None = None) -> ImageBuild:
    return ImageBuild(
        id=build_id,
        task=f"default/{name}",
        revision=DIGEST,
        status=status,  # ty: ignore[invalid-argument-type]
        job=f"task-image-build-{name}-{build_id}",
        allowed_registries=[],
        platform=platform,
        images=[],
        created_at=T0,
        updated_at=T0,
    )


class FakeBuilds:
    """``evaluator.tasks`` for image builds, following ``request_build``'s rules.

    Each attempt goes ``pending`` → ``running`` on its first read, then ends on the next as its task's script says:
    ``ok`` (ready), ``fail`` (failed), or ``dead`` (its Job died: it stays ``running`` until a request finds the Job
    ended, records the attempt failed and starts the next). Attempts with no script entry end ``ok``.
    """

    def __init__(self, outcomes: dict[str, list[str]] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.attempts: dict[str, list[ImageBuild]] = {}
        self.posts: list[str] = []
        self.gets = 0

    def _outcome(self, name: str, build_id: str) -> str:
        script = self.outcomes.get(name, [])
        index = int(build_id) - 1
        return script[index] if index < len(script) else "ok"

    def _start(self, name: str) -> ImageBuild:
        attempts = self.attempts.setdefault(name, [])
        attempts.append(_build(name, str(len(attempts) + 1)))
        return attempts[-1]

    def build_images(self, name, *, revision, workspace):
        self.posts.append(name)
        attempts = self.attempts.get(name)
        if not attempts or attempts[-1].status == "failed":
            return self._start(name)
        latest = attempts[-1]
        if latest.status == "running" and self._outcome(name, latest.id) == "dead":
            attempts[-1] = latest.model_copy(update={"status": "failed", "error": "Job ended (error) without a result"})
            return self._start(name)
        return latest

    def get_image_build(self, name, build_id, *, revision, workspace):
        self.gets += 1
        attempts = self.attempts[name]
        index = int(build_id) - 1
        build = attempts[index]
        if build.status in ("ready", "failed"):
            return build
        outcome = self._outcome(name, build_id)
        status = (
            "running" if build.status == "pending" else {"ok": "ready", "fail": "failed", "dead": "running"}[outcome]
        )
        update: dict = {"status": status, "updated_at": T0 + timedelta(minutes=2, seconds=14)}
        if status == "ready":
            update["images"] = [BuildImage(type="main", source="prebuilt")]
        if status == "failed":
            update["error"] = "no Dockerfile"
        attempts[index] = build.model_copy(update=update)
        return attempts[index]

    def list_image_builds(self, name, *, revision, workspace):
        return SimpleNamespace(data=list(reversed(self.attempts.get(name, []))))


def evaluator(builds: FakeBuilds, names=("a", "b")) -> Evaluator:
    members = [f"default/{name}#{DIGEST}" for name in names]
    tasksets = SimpleNamespace(retrieve=lambda name, revision, workspace: SimpleNamespace(tasks=members))
    return cast(Evaluator, SimpleNamespace(tasks=builds, tasksets=tasksets))


def wait(builds: FakeBuilds, names=("a", "b"), **kwargs):
    lines: list[str] = []
    refs = [f"default/{name}#{DIGEST}" for name in names]
    result = wait_for_builds(
        evaluator(builds, names), refs, poll_seconds=0, echo=lines.append, sleep=lambda _: None, **kwargs
    )
    return result, lines


# --- The --wait loop -------------------------------------------------------------------------------------


def test_wait_requests_once_then_only_reads_by_id():
    builds = FakeBuilds()
    result, lines = wait(builds)
    assert builds.posts == ["a", "b"]  # one request per task; every poll after that is a GET
    assert {ref: b.status for ref, b in result.items()} == dict.fromkeys(result, "ready")
    assert lines[0] == "Image builds for 2 tasks: 0 ready, 2 building, 0 failed. Ctrl-C is safe; builds keep going."
    assert lines[1:] == [
        f"[0/2] building  default/a#{DIGEST}",
        f"[0/2] building  default/b#{DIGEST}",
        f"[1/2] ready     default/a#{DIGEST} (1 image, 2m14s)",
        f"[2/2] ready     default/b#{DIGEST} (1 image, 2m14s)",
        "Image builds: 2 of 2 tasks ready (2 images).",
    ]


def test_a_task_whose_newest_build_is_ready_gets_no_request():
    builds = FakeBuilds()
    builds.attempts["a"] = [_build("a", "1", "ready")]
    result, lines = wait(builds)
    assert builds.posts == ["b"]  # only the task with no ready build
    assert {ref: b.status for ref, b in result.items()} == dict.fromkeys(result, "ready")
    assert lines[0] == "Image builds for 2 tasks: 1 ready, 1 building, 0 failed. Ctrl-C is safe; builds keep going."


def test_a_failed_attempt_is_retried_once_then_the_task_settles_failed():
    builds = FakeBuilds({"a": ["fail", "fail"], "b": ["fail", "ok"]})
    result, lines = wait(builds)
    assert result[f"default/a#{DIGEST}"].status == "failed" and result[f"default/b#{DIGEST}"].status == "ready"
    assert builds.posts == ["a", "b", "a", "b"]  # the requests, then one retry each
    assert f"[0/2] retrying  default/a#{DIGEST}  attempt 1 failed: no Dockerfile" in lines
    assert f"[1/2] failed    default/a#{DIGEST}  attempt 2 failed: no Dockerfile" in lines
    assert lines[-1] == "Image builds: 1 of 2 tasks ready (1 images), 1 failed."


def test_a_retry_request_that_fails_does_not_use_up_the_retry():
    builds = FakeBuilds({"a": ["fail"]})
    real_request, failures = builds.build_images, [0]

    def request(name, *, revision, workspace):
        if builds.attempts.get(name) and builds.attempts[name][-1].status == "failed" and not failures[0]:
            failures[0] += 1  # the retry's first request doesn't get through
            raise NemoTransportError(httpx.ConnectError("connection refused"))
        return real_request(name, revision=revision, workspace=workspace)

    builds.build_images = request  # ty: ignore[invalid-assignment]
    result, lines = wait(builds, names=("a",))
    assert result[f"default/a#{DIGEST}"].status == "ready"  # the next poll retried after all
    assert sum("retrying" in line for line in lines) == 1


def test_a_dead_job_is_noticed_by_the_repair_request_and_counts_as_an_ended_attempt():
    builds = FakeBuilds({"a": ["dead"]})
    now = [0.0]

    def clock() -> float:
        now[0] += BUILD_REPAIR_SECONDS / 4  # a repair request is due every fourth poll
        return now[0]

    result, lines = wait(builds, names=("a",), clock=clock)
    assert result[f"default/a#{DIGEST}"].status == "ready"
    assert builds.posts == ["a", "a"]  # the request, then one repair that started attempt 2
    assert f"[0/1] retrying  default/a#{DIGEST}  attempt 1 failed: Job ended (error) without a result" in lines


def test_a_second_dead_job_settles_the_task_and_names_the_attempt_it_no_longer_follows():
    builds = FakeBuilds({"a": ["dead", "dead"]})
    now = [0.0]

    def clock() -> float:
        now[0] += BUILD_REPAIR_SECONDS / 4
        return now[0]

    result, lines = wait(builds, names=("a",), clock=clock)
    assert result[f"default/a#{DIGEST}"].status == "failed"
    assert lines[-2].endswith("attempt 2 failed: Job ended (error) without a result; attempt 3 started, not followed")


def test_reads_that_fail_keep_the_last_state_until_too_many_in_a_row():
    builds = FakeBuilds()
    real_get, failures = builds.get_image_build, [BUILD_READ_FAILURES - 1]

    def flaky(*args, **kwargs):
        if failures[0]:
            failures[0] -= 1
            raise NemoTransportError(httpx.ConnectError("connection refused"))
        return real_get(*args, **kwargs)

    builds.get_image_build = flaky  # ty: ignore[invalid-assignment]
    result, lines = wait(builds, names=("a",))
    assert result[f"default/a#{DIGEST}"].status == "ready"
    assert sum("Reading the image builds failed" in line for line in lines) == BUILD_READ_FAILURES - 1

    def down(*args, **kwargs):
        raise NemoTransportError(httpx.ConnectError("connection refused"))

    builds = FakeBuilds()
    builds.get_image_build = down  # ty: ignore[invalid-assignment]
    with pytest.raises(NemoTransportError):
        wait(builds, names=("a",))


# --- The commands ----------------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    """The ``tasksets`` commands over a fake evaluator; ``cli(builds, *args)`` invokes one."""

    def invoke(builds: FakeBuilds, *args: str, names=("a", "b")):
        monkeypatch.setattr(
            "nemo_evals.tasksets_cli.Evaluator", SimpleNamespace(from_client=lambda _: evaluator(builds, names))
        )
        client = SimpleNamespace(resolve_workspace=lambda workspace=None: workspace or "default")
        app = typer.Typer()
        add_tasksets_commands(app, lambda: client)  # ty: ignore[invalid-argument-type]
        return CliRunner().invoke(app, ["tasksets", *args])

    return invoke


def test_build_images_requests_each_task_once_and_returns(cli):
    builds = FakeBuilds()
    result = cli(builds, "build-images", "suite#" + "e" * 64)  # no workspace: the configured one
    assert result.exit_code == 0, result.output
    assert builds.posts == ["a", "b"] and builds.gets == 0
    assert result.output.splitlines() == [
        "Image builds for 2 tasks: 0 ready, 2 building, 0 failed.",
        "Check with --status, follow with --wait, or submit once all are ready.",
    ]


def test_status_only_reads_and_its_exit_code_says_where_the_builds_are(cli):
    builds = FakeBuilds({"b": ["fail"]})
    assert cli(builds, "build-images", "default/suite").exit_code == 0
    building = cli(builds, "build-images", "default/suite", "--status")
    assert building.exit_code == STATUS_BUILDING and builds.posts == ["a", "b"]
    assert "  building      default/a#" in building.output and "(no change for " in building.output
    for name in ("a", "b"):  # finish both: a ends ready, b failed
        builds.get_image_build(name, "1", revision=DIGEST, workspace="default")
        builds.get_image_build(name, "1", revision=DIGEST, workspace="default")
    failed = cli(builds, "build-images", "default/suite", "--status")
    assert failed.exit_code == 1 and "  failed        default/b#" in failed.output and "no Dockerfile" in failed.output
    assert builds.posts == ["a", "b"]  # --status never requests


def test_status_reads_the_default_platform_build_and_names_tasks_never_requested(cli):
    builds = FakeBuilds()
    builds.attempts["a"] = [_build("a", "1", "ready"), _build("a", "2", "running", platform="linux/arm64")]
    result = cli(builds, "build-images", "default/suite", "--status")
    assert result.exit_code == 1  # b has no build
    assert result.output.splitlines()[0] == "Image builds for 2 tasks: 1 ready, 0 building, 0 failed, 1 not requested."


def test_wait_and_status_do_not_combine(cli):
    assert cli(FakeBuilds(), "build-images", "default/suite", "--wait", "--status").exit_code == 2


def test_wait_exits_1_when_a_build_failed_and_points_to_submit_when_all_are_ready(cli):
    failed = cli(FakeBuilds({"a": ["fail", "fail"]}), "build-images", "default/suite", "--wait", "--poll-seconds", "0")
    assert failed.exit_code == 1 and failed.output.splitlines()[-1] == "Rerun this command to retry the failed builds."
    ready = cli(FakeBuilds(), "build-images", "default/suite", "--wait", "--poll-seconds", "0")
    assert ready.exit_code == 0
    assert ready.output.splitlines()[-1] == "Submit with: nemo evals runs submit --taskset default/suite"


def test_ctrl_c_during_wait_leaves_the_builds_running(cli):
    builds = FakeBuilds()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    builds.get_image_build = interrupted  # ty: ignore[invalid-assignment]
    result = cli(builds, "build-images", "default/suite", "--wait", "--poll-seconds", "0")
    assert result.exit_code == 130 and "the builds keep going" in result.output


def test_create_prints_the_dataset_flag_and_the_ref_last(cli, monkeypatch):
    created = CreatedTaskset(
        "ac9/deep-swe-162c7ea7#" + "2" * 64, [f"ac9/a#{DIGEST}"], 113, ("datacurve/deep-swe", "sha256:" + "a" * 64)
    )
    monkeypatch.setattr("nemo_evals.tasksets_cli.create_taskset", lambda *a, **k: created)
    result = cli(FakeBuilds(), "create", "--harbor-hub", "datacurve/deep-swe@latest", "--workspace", "ac9")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"Dataset for Intake: --dataset datacurve/deep-swe@sha256:{'a' * 64}",
        created.ref,
    ]
    as_json = json.loads(cli(FakeBuilds(), "create", "--harbor-hub", "x/y@1", "--json").output)
    assert as_json["taskset_ref"] == created.ref and as_json["dataset"] == f"datacurve/deep-swe@sha256:{'a' * 64}"


def test_create_needs_exactly_one_source(cli, tmp_path: Path):
    assert cli(FakeBuilds(), "create").exit_code == 2
    assert cli(FakeBuilds(), "create", "--harbor-dir", str(tmp_path), "--harbor-hub", "x/y@1").exit_code == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("datacurve/deep-swe@sha256:" + "a" * 64, ("datacurve/deep-swe", "sha256:" + "a" * 64)),
        ("harbor-framework/terminal-bench@21ae84bf", ("harbor-framework/terminal-bench", "21ae84bf")),
        ("my-dataset", ("my-dataset", None)),
    ],
)
def test_dataset_flag_parses_name_and_version(value, expected):
    assert parse_dataset(value) == expected


def test_a_ref_without_a_workspace_takes_the_configured_one():
    assert qualify_ref("suite#" + "e" * 64, "ac9") == "ac9/suite#" + "e" * 64
    assert qualify_ref("other/suite", "ac9") == "other/suite"
