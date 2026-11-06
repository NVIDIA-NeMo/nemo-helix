# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals runs submit``: client-side task selection and the run request it sends."""

import io
import json
import shlex
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import typer
from nemo_evals import runs_cli
from nemo_evals.api.fields import TaskRef, TasksetRef
from nemo_evals.api.schemas import AgentEvalResult
from nemo_evals.api.taskset_run_schemas import TasksetRunStatus, TasksetRunTask
from nemo_evals.harbor.remote.profile import SandboxSelection
from nemo_evals.image_builds.models import BuildImage, ImageBuild
from nemo_evals.runs_cli import (
    BundleJobView,
    Candidate,
    CreatedTaskset,
    ScoreRollup,
    SelectionError,
    _progress_line,
    _results_table,
    _select_from_spec,
    _status_table,
    _workspace_flag,
    add_runs_commands,
    check_git_source,
    download_git_tasks,
    download_hub_tasks,
    job_error,
    newest_bundle_job,
    publish_tasks,
    rollup_scores,
    run_request,
    select_tasks,
    selection_suffix,
    task_state,
)
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.sdk.taskset_run_resources import (
    NamedTask,
    TaskResult,
    TasksetRunResults,
    TasksetRunSubmitError,
)
from nemo_evals.taskset_runs.progress import TasksetRunProgress
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError, NotFoundError, raise_for_status
from nemo_helix_plugin.jobs.types import HelixJobResponse
from typer.testing import CliRunner

CANDIDATES = [
    Candidate(0, "debug-dataset-debug-agent-runtime-error", "hello/debug-agent-runtime-error"),
    Candidate(1, "debug-dataset-greet-universe", "hello/greet-universe"),
    Candidate(2, "debug-dataset-sum-three", "hello/sum-three"),
]


@pytest.mark.parametrize(
    ("tokens", "indexes"),
    [
        (None, [0, 1, 2]),
        ("0,2", [0, 2]),
        ("greet-universe,sum-three", [1, 2]),
        ("hello/sum-three, 1", [2, 1]),
    ],
)
def test_selects_by_index_or_name(tokens, indexes):
    assert [c.index for c in select_tasks(CANDIDATES, tokens)] == indexes


@pytest.mark.parametrize(
    ("tokens", "indexes"),
    [
        ("0-2", [0, 1, 2]),
        ("2-2", [2]),
        ("2,0-1", [2, 0, 1]),
        ("0-0,sum-three", [0, 2]),
        (" 00-01 , ,sum-three,", [0, 1, 2]),
    ],
)
def test_selects_inclusive_ranges(tokens, indexes):
    assert [c.index for c in select_tasks(CANDIDATES, tokens)] == indexes


def test_selects_ranges_mixed_with_indexes_and_names():
    candidates = [Candidate(i, f"task-{i}", f"task-{i}") for i in range(8)]
    candidates.append(Candidate(8, "anko-example", "anko-example"))
    assert [c.index for c in select_tasks(candidates, "0-4,7,anko-example")] == [0, 1, 2, 3, 4, 7, 8]


@pytest.mark.parametrize(
    ("tokens", "reason"),
    [
        ("2-1", "reversed"),
        ("0,2-1", "reversed"),
        ("0-3", "out of range"),
        ("3-3", "out of range"),
        ("3-5", "out of range"),
        ("0-99999999999999999999", "out of range"),
    ],
)
def test_invalid_ranges_identify_the_range(tokens, reason):
    with pytest.raises(SelectionError, match=reason) as error:
        select_tasks(CANDIDATES, tokens)
    assert repr(tokens.split(",")[-1]) in str(error.value)


def test_explicit_range_on_an_empty_source_is_rejected():
    assert select_tasks([], None) == []
    with pytest.raises(SelectionError, match="out of range"):
        select_tasks([], "0-0")


def test_range_endpoint_exceeding_the_interpreter_limit_is_a_selection_error():
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("integer-string conversion limit is disabled")
    with pytest.raises(SelectionError, match="invalid task range"):
        select_tasks(CANDIDATES, "0-" + "9" * (limit + 1))


@pytest.mark.parametrize(
    "tokens",
    ["0-1,1", "1,0-1", "0-1,1-2", "0-1,0-1", "1-2,greet-universe", "greet-universe,1-2"],
)
def test_range_overlaps_and_aliases_are_rejected(tokens):
    with pytest.raises(SelectionError, match="selected twice"):
        select_tasks(CANDIDATES, tokens)


@pytest.mark.parametrize("tokens", ["1--2", "1-", "-1-2", "0 - 2", "0:2", "1-2-3"])
def test_malformed_range_without_a_matching_name_is_rejected(tokens):
    with pytest.raises(SelectionError, match="no task named"):
        select_tasks(CANDIDATES, tokens)


@pytest.mark.parametrize("token", ["7-days", "suite-0-2", "bench/0-2", "1--2"])
def test_hyphenated_names_are_not_ranges(token):
    candidate = (
        Candidate(0, "suite-0-2", "bench/0-2") if token in ("suite-0-2", "bench/0-2") else Candidate(0, token, token)
    )
    assert select_tasks([candidate], token) == [candidate]


def test_numeric_pair_names_are_selected_by_individual_index():
    candidates = [Candidate(0, "first", "first"), Candidate(1, "0-2", "0-2"), Candidate(2, "last", "last")]
    assert select_tasks(candidates, "0-2") == candidates
    assert select_tasks(candidates, "1") == [candidates[1]]


@pytest.mark.parametrize("tokens", ["", " ", ",", " , , "])
def test_empty_selection_is_rejected(tokens):
    with pytest.raises(SelectionError, match="selects no task"):
        select_tasks(CANDIDATES, tokens)


@pytest.mark.parametrize(
    ("tokens", "match"),
    [("3", "out of range"), ("nope", "no task named"), ("2,sum-three", "selected twice"), (",", "selects no task")],
)
def test_bad_selection_fails_before_any_upload(tokens, match):
    with pytest.raises(SelectionError, match=match):
        select_tasks(CANDIDATES, tokens)


def test_ambiguous_name_is_rejected():
    candidates = [Candidate(0, "a-sum", "x/sum"), Candidate(1, "b-sum", "y/sum")]
    with pytest.raises(SelectionError, match="matches several tasks"):
        select_tasks(candidates, "sum")


def test_selection_suffix_ignores_order():
    assert selection_suffix(["a#1", "b#2"]) == selection_suffix(["b#2", "a#1"]) != selection_suffix(["a#1"])


STORED = "ac9/deep-swe-162c7ea7#" + "2" * 64
MEMBERS: list[str] = [f"ac9/deep-swe-{name}#" + "1" * 64 for name in ("a", "b")]


@pytest.fixture
def submitting(monkeypatch):
    """``runs submit`` over a fake evaluator and submit; returns what it published, requested and sent."""
    seen = SimpleNamespace(bodies=[], published=[], published_members=[], posts=[], created=[], members=list(MEMBERS))
    builds = SimpleNamespace(
        build_images=lambda name, revision, workspace: seen.posts.append(name) or _ready(name),
        get_image_build=lambda name, build_id, revision, workspace: _ready(name),
    )
    tasks = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(revision=1, spec=SimpleNamespace()),
        build_images=builds.build_images,
        get_image_build=builds.get_image_build,
    )

    def replace_taskset(name, taskset, workspace):
        seen.published.append(name)
        seen.published_members.append([member.root for member in taskset.tasks])

    tasksets = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(tasks=[TaskRef(m) for m in seen.members]),
        replace=replace_taskset,
        list_revisions=lambda name, workspace, page_size: SimpleNamespace(
            data=[SimpleNamespace(content_hash="3" * 64)]
        ),
    )
    seen.taskset_runs = SimpleNamespace(submit=lambda request, workspace: seen.submit(request, workspace=workspace))
    fake = SimpleNamespace(tasks=tasks, tasksets=tasksets, taskset_runs=seen.taskset_runs)
    monkeypatch.setattr("nemo_evals.runs_cli.Evaluator", SimpleNamespace(from_client=lambda _: fake))

    def submit(request, *, workspace):
        seen.bodies.append(request.model_dump(mode="json", exclude_none=True))
        if seen.refuse is not None:
            raise seen.refuse
        return SimpleNamespace(run_id="run-1", phase=seen.phase, warnings=[])

    seen.refuse = None
    seen.phase = "accepted"
    seen.submit = submit
    client = SimpleNamespace(resolve_workspace=lambda workspace=None: workspace or "ac9")

    def invoke(*args: str):
        app = typer.Typer()
        add_runs_commands(app, lambda: client)  # ty: ignore[invalid-argument-type]
        return CliRunner().invoke(app, ["runs", "submit", *args])

    seen.invoke = invoke
    return seen


@pytest.fixture
def local_submission(submitting, monkeypatch, tmp_path):
    for name in ("sum-three", "debug-agent-runtime-error", "greet-universe"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "task.toml").write_text(f'[task]\nname = "hello/{name}"\n')
    submitting.directory = tmp_path
    submitting.uploaded = []
    submitting.build_refs = []

    def publish(evaluator, task_dirs, *, prefix, workspace, echo):
        submitting.uploaded.extend(task_dirs)
        return [f"{workspace}/{directory.name}#" + "1" * 64 for directory in task_dirs]

    def build(evaluator, refs, **kwargs):
        submitting.build_refs.extend(refs)
        return {ref: _ready(ref) for ref in refs}

    monkeypatch.setattr(runs_cli, "publish_tasks", publish)
    monkeypatch.setattr(runs_cli, "wait_for_builds", build)
    return submitting


def test_submit_local_range_selection(local_submission):
    seen = local_submission
    result = seen.invoke("--harbor-dir", str(seen.directory), "--tasks", "sum-three,0-0")
    assert result.exit_code == 0, result.output
    refs = [f"ac9/{name}#" + "1" * 64 for name in ("sum-three", "debug-agent-runtime-error")]
    assert seen.uploaded == [seen.directory / "sum-three", seen.directory / "debug-agent-runtime-error"]
    assert seen.published_members == [refs]
    assert seen.build_refs == refs
    assert seen.bodies[0]["spec"]["tasks"] == f"ac9/{seen.published[0]}#" + "3" * 64
    assert "2 of 3 tasks" in result.output


@pytest.mark.parametrize("tokens", ["0,2-1", "0-3", "0-1,1"])
def test_submit_invalid_range_writes_nothing(local_submission, tokens):
    seen = local_submission
    result = seen.invoke("--harbor-dir", str(seen.directory), "--tasks", tokens)
    assert result.exit_code == 1 and "Error:" in result.output
    assert seen.uploaded == seen.published == seen.build_refs == seen.posts == seen.bodies == []


def test_submit_listing_ignores_an_invalid_range(local_submission):
    seen = local_submission
    result = seen.invoke("--harbor-dir", str(seen.directory), "--tasks", "2-1", "--list-tasks")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "  #  task",
        "  0  hello/debug-agent-runtime-error",
        "  1  hello/greet-universe",
        "  2  hello/sum-three",
    ]
    assert seen.uploaded == seen.published == seen.build_refs == seen.bodies == []


@pytest.mark.parametrize("from_spec", [False, True], ids=["taskset", "spec-file"])
@pytest.mark.parametrize("tokens", ["0-1", "0,2-1", "0-3", "0-1,1"])
def test_submit_stored_range_selection(submitting, tmp_path, from_spec, tokens):
    members = [f"ac9/deep-swe-{name}#" + "1" * 64 for name in ("c", "a", "b")]
    submitting.members = members
    if from_spec:
        spec = tmp_path / "run.json"
        spec.write_text(
            json.dumps(
                {
                    "spec": {"tasks": STORED, "target": {"kind": "harbor", "source": {"name": "oracle"}}},
                    "execution": {},
                }
            )
        )
        source = ["--spec-file", str(spec)]
    else:
        source = ["--taskset", STORED]
    result = submitting.invoke(*source, "--tasks", tokens)
    assert submitting.posts == []
    if tokens == "0-1":
        assert result.exit_code == 0, result.output
        assert submitting.published_members == [[members[1], members[2]]]
        assert submitting.bodies[0]["spec"]["tasks"] == f"ac9/{submitting.published[0]}#" + "3" * 64
    else:
        assert result.exit_code == 1 and "Error:" in result.output
        assert submitting.published == submitting.bodies == []


def test_submit_help_explains_ranges_without_a_client():
    result = CliRunner().invoke(_runs_app(), ["runs", "submit", "--help"], terminal_width=160)
    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "Zero-based" in help_text and "inclusive ranges" in help_text
    assert "0-4,7,anko-example" in help_text


def submit_error(status: int, detail: str, key: str = "k-1") -> TasksetRunSubmitError:
    response = httpx.Response(status, json={"detail": detail}, request=httpx.Request("POST", "http://platform.test"))
    return TasksetRunSubmitError(raise_error(response), key)


def raise_error(response: httpx.Response) -> NemoHTTPError:
    try:
        raise_for_status(response)
    except NemoHTTPError as exc:
        return exc
    raise AssertionError("not an error status")


def _image_build(build_id: str = "1", status: str = "ready") -> ImageBuild:
    return ImageBuild(
        id=build_id,
        task="ac9/deep-swe-a",
        revision="1" * 64,
        status=status,  # ty: ignore[invalid-argument-type]
        job=f"task-image-build-{build_id}",
        allowed_registries=[],
        images=[BuildImage(type="main", source="prebuilt")] if status == "ready" else [],
        error="no Dockerfile" if status == "failed" else None,
    )


def _ready(name):
    return _image_build()


def test_submit_with_a_taskset_uploads_nothing_and_requests_no_builds(submitting):
    result = submitting.invoke(
        "--taskset", STORED, "--experiment", "m4", "--intake-dataset", "datacurve/deep-swe@sha256:" + "a" * 64
    )
    assert result.exit_code == 0, result.output
    assert submitting.published == [] and submitting.posts == []
    (body,) = submitting.bodies
    assert body["spec"]["tasks"] == STORED
    assert body["publication"]["intake"] == {
        "experiment": "m4",
        "dataset_name": "datacurve/deep-swe",
        "dataset_version": "sha256:" + "a" * 64,
    }
    assert "Run run-1 accepted: 2 tasks, queued" in result.output
    assert "Rerun these tasks" not in result.output  # --taskset already names it


@pytest.mark.parametrize("phase", ["running", "cancelled"])
def test_a_resubmit_that_replays_a_started_run_says_it_started_nothing(submitting, phase):
    submitting.phase = phase
    result = submitting.invoke("--taskset", STORED, "--idempotency-key", "k")
    assert result.exit_code == 0, result.output
    assert f"Run run-1 was already submitted with this key and is {phase}; nothing new started" in result.output
    assert "starts their task Jobs" not in result.output


def test_submit_narrows_a_taskset_by_publishing_a_selection_without_uploading_tasks(submitting):
    result = submitting.invoke("--taskset", STORED, "--tasks", "1")
    assert result.exit_code == 0, result.output
    assert len(submitting.published) == 1 and submitting.published[0].startswith("deep-swe-162c7ea7-sel-")
    assert submitting.posts == [] and submitting.bodies[0]["spec"]["tasks"].endswith("#" + "3" * 64)


@pytest.mark.parametrize("workspace", ["ac9", "team"], ids=["configured", "other"])
def test_a_refused_taskset_submit_says_how_to_check_the_builds(submitting, workspace):
    """The suggested command must run as printed: build-images reads the workspace from the ref."""
    submitting.refuse = submit_error(422, "tasks without images: ac9/deep-swe-b#...")
    result = submitting.invoke("--taskset", "deep-swe-162c7ea7#" + "2" * 64, "--workspace", workspace)
    assert result.exit_code == 1
    stored = f"{workspace}/deep-swe-162c7ea7#" + "2" * 64
    assert f"Check with: nemo evals tasksets build-images {shlex.quote(stored)} --status\n" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--taskset", STORED, "--harbor-hub", "x/y@1"],
        ["--taskset", STORED, "--intake-dataset", "x/y"],  # no --experiment
        ["--harbor-hub", "x/y@1", "--experiment", "e", "--intake-dataset", "x/y"],  # the source sets the dataset
    ],
)
def test_taskset_flags_that_do_not_combine(submitting, args):
    assert submitting.invoke(*args).exit_code == 2 and submitting.bodies == []


@pytest.mark.parametrize(("failures", "submitted"), [(0, True), (1, True), (2, False)])
def test_the_wrapper_follows_one_retry_and_never_submits_after_a_failed_build(
    submitting, monkeypatch, failures, submitted
):
    created = CreatedTaskset(STORED, MEMBERS[:1], 1, ("datacurve/deep-swe", None))
    monkeypatch.setattr("nemo_evals.runs_cli.create_taskset", lambda *a, **k: created)
    attempts = []

    def build_images(name, revision, workspace):
        attempts.append(name)
        return _image_build(str(len(attempts)), "pending")

    def get_image_build(name, build_id, revision, workspace):
        return _image_build(build_id, "failed" if int(build_id) <= failures else "ready")

    submitting_tasks = SimpleNamespace(
        build_images=build_images,
        get_image_build=get_image_build,
        list_image_builds=lambda name, revision, workspace: SimpleNamespace(data=[]),
    )
    monkeypatch.setattr(
        "nemo_evals.runs_cli.Evaluator",
        SimpleNamespace(
            from_client=lambda _: SimpleNamespace(tasks=submitting_tasks, taskset_runs=submitting.taskset_runs)
        ),
    )
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--experiment", "m4", "--poll-seconds", "0")
    assert bool(submitting.bodies) is submitted, result.output
    assert len(attempts) == min(failures, 1) + 1  # at most one retry
    assert "Submit later" not in result.output
    if submitted:
        rerun = f"Rerun these tasks: nemo evals runs submit --taskset {shlex.quote(STORED)} --experiment m4 --intake-dataset datacurve/deep-swe"
        assert rerun in result.output  # the same Experiment and Intake label
        assert submitting.bodies[0]["publication"]["intake"]["dataset_name"] == "datacurve/deep-swe"
    else:
        assert result.exit_code == 1 and "image builds failed" in result.output


def test_build_hints_outside_the_configured_workspace_name_it_only_where_the_command_takes_it(submitting, monkeypatch):
    other = "team/deep-swe-162c7ea7#" + "2" * 64
    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(other, MEMBERS[:1], 1, None)
    )

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("nemo_evals.runs_cli.wait_for_builds", interrupted)
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--workspace", "team")
    # build-images has no --workspace: the qualified ref carries it
    assert f"Check them with: nemo evals tasksets build-images {shlex.quote(other)} --status\n" in result.output
    assert f"submit with: nemo evals runs submit --taskset {shlex.quote(other)} --workspace team\n" in result.output


def test_ctrl_c_during_the_build_wait_says_how_to_submit_later(submitting, monkeypatch):
    created = CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    monkeypatch.setattr("nemo_evals.runs_cli.create_taskset", lambda *a, **k: created)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("nemo_evals.runs_cli.wait_for_builds", interrupted)
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--agent", "openhands")
    assert result.exit_code == 130 and submitting.bodies == []
    assert f"Check them with: nemo evals tasksets build-images {shlex.quote(STORED)} --status" in result.output
    assert (
        f"Once they're ready, submit with: nemo evals runs submit --taskset {shlex.quote(STORED)} --agent openhands"
        in result.output
    )


@pytest.mark.parametrize(
    ("options", "kept"),
    [
        (
            ["--agent", "openhands", "--sandbox-profile", "harbor-web", "--cpus", "4", "--experiment", "my exp"],
            " --agent openhands --sandbox-profile harbor-web --cpus 4 --experiment 'my exp'",
        ),
        (["--agent", "oracle", "--dispatcher-profile", "gpu"], " --dispatcher-profile gpu"),  # oracle: the default
    ],
)
def test_the_rerun_command_keeps_the_run_settings(submitting, monkeypatch, options, kept):
    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    )
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--no-build-images", *options)
    assert f"Rerun these tasks: nemo evals runs submit --taskset {shlex.quote(STORED)}{kept}\n" in result.output, (
        result.output
    )


def test_the_rerun_command_keeps_the_spec_file_not_the_default_agent(submitting, monkeypatch, tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {},
            }
        )
    )
    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    )
    result = submitting.invoke(
        "--harbor-hub", "datacurve/deep-swe@latest", "--no-build-images", "--spec-file", str(spec)
    )
    rerun = f"Rerun these tasks: nemo evals runs submit --taskset {shlex.quote(STORED)} --spec-file {shlex.quote(str(spec))}\n"
    assert rerun in result.output, result.output
    assert "idempotency_key" not in result.output


def test_the_rerun_command_warns_when_the_spec_file_pins_the_key(submitting, monkeypatch, tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {},
                "idempotency_key": "pinned",
            }
        )
    )
    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    )
    args = ["--harbor-hub", "datacurve/deep-swe@latest", "--no-build-images", "--spec-file", str(spec)]
    assert "spec file's idempotency_key; pass a new --idempotency-key" in submitting.invoke(*args).output
    # The rerun drops --idempotency-key (repeating k2 would return this run), so it falls back to the file's key
    explicit = submitting.invoke(*args, "--idempotency-key", "k2").output
    assert "spec file's idempotency_key; pass a new --idempotency-key" in explicit
    assert "k2" not in explicit.split("Rerun these tasks:", 1)[1].splitlines()[0]


def test_submit_reads_the_spec_file_once(submitting, monkeypatch, tmp_path):
    """A stream such as ``--spec-file /dev/stdin`` can't be read again after the request is built."""
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {},
                "idempotency_key": "pinned",
            }
        )
    )
    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    )
    submit = submitting.submit

    def consumed(*args, **kwargs):
        spec.unlink()  # like a stream already read
        return submit(*args, **kwargs)

    submitting.submit = consumed
    result = submitting.invoke(
        "--harbor-hub", "datacurve/deep-swe@latest", "--no-build-images", "--spec-file", str(spec)
    )
    assert result.exit_code == 0, result.output
    assert "spec file's idempotency_key; pass a new --idempotency-key" in result.output
    assert "--spec-file '<spec file>'" in result.output  # the stream's path is gone by then


def test_a_spec_file_alone_is_read_once(submitting, monkeypatch, tmp_path):
    """With no source or --taskset, the spec's taskset is selected from the same read as the request."""
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": STORED, "target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {},
            }
        )
    )
    select = runs_cli.select_from_taskset

    def consumed(*args, **kwargs):
        spec.unlink()  # like a stream already read
        return select(*args, **kwargs)

    monkeypatch.setattr("nemo_evals.runs_cli.select_from_taskset", consumed)
    result = submitting.invoke("--spec-file", str(spec), "--tasks", "0")
    assert result.exit_code == 0, result.output
    assert submitting.bodies and submitting.bodies[0]["spec"]["target"]["source"]["name"] == "openhands"


def test_ctrl_c_during_the_submit_says_to_retry_the_published_taskset_with_the_same_key(submitting, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "nemo_evals.runs_cli.create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS[:1], 1, None)
    )
    submitting.submit = interrupted
    result = submitting.invoke(
        "--harbor-hub", "datacurve/deep-swe@latest", "--no-build-images", "--idempotency-key", "k-$USER"
    )
    assert result.exit_code == 130
    assert "the run may still be accepted" in result.output
    # The published taskset, not the source, which may have moved since; the key is quoted, so the shell keeps it.
    retry = (
        f"submit with the same key: nemo evals runs submit --taskset {shlex.quote(STORED)} --idempotency-key 'k-$USER'"
    )
    assert retry in result.output
    assert "image builds" not in result.output and "--harbor-hub" not in result.output


def test_ctrl_c_before_the_submit_names_the_taskset(submitting, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("nemo_evals.runs_cli.run_request", interrupted)
    result = submitting.invoke("--taskset", STORED)
    assert result.exit_code == 130
    assert "Interrupted. Submit with: nemo evals runs submit --taskset " + shlex.quote(STORED) in result.output


@pytest.mark.parametrize("interrupt_at", ["wait_for_builds", "run_request"])
@pytest.mark.parametrize("spec_key", [None, "file-key"])
@pytest.mark.parametrize("cli_key", [None, "cli key-$USER"])
def test_ctrl_c_before_submit_keeps_the_effective_key(
    submitting, monkeypatch, tmp_path, interrupt_at, spec_key, cli_key
):
    """A printed recovery command must not replace the CLI override with the file's key."""
    spec = tmp_path / "run settings.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {},
                "idempotency_key": spec_key,
            }
        )
    )
    monkeypatch.setattr(runs_cli, "create_taskset", lambda *a, **k: CreatedTaskset(STORED, MEMBERS, 2, None))
    original = getattr(runs_cli, interrupt_at)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runs_cli, interrupt_at, interrupted)
    options = ["--idempotency-key", cli_key] if cli_key is not None else []
    if interrupt_at == "run_request":
        options.append("--no-build-images")
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--spec-file", str(spec), *options)
    assert result.exit_code == 130 and submitting.bodies == [], result.output
    command = next(
        line[line.index("nemo evals runs submit") :]
        for line in result.output.splitlines()
        if "nemo evals runs submit" in line
    )
    retry = shlex.split(command)
    monkeypatch.setattr(runs_cli, interrupt_at, original)
    resumed = submitting.invoke(*retry[4:])
    assert resumed.exit_code == 0, resumed.output
    (body,) = submitting.bodies
    if cli_key is not None or spec_key is not None:
        assert body["idempotency_key"] == (cli_key or spec_key)
    else:
        assert "--idempotency-key" not in retry and body["idempotency_key"]


@pytest.mark.parametrize("key_source", ["cli", "spec", "generated"])
def test_ctrl_c_during_submit_replays_the_original_request(submitting, monkeypatch, tmp_path, key_source):
    """Recovery preserves the effective key and publication without rediscovering the source."""
    spec = tmp_path / "run settings.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"target": {"kind": "harbor", "source": {"name": "openhands"}}},
                "execution": {"sandbox": {"profile": "harbor-web", "cpus": 4}, "max_concurrent_task_jobs": 3},
                "timeout_seconds": 7200,
                "idempotency_key": "file-key" if key_source == "spec" else None,
            }
        )
    )
    created = CreatedTaskset(STORED, MEMBERS, 2, ("datacurve/deep-swe", "sha256:" + "a" * 64))
    monkeypatch.setattr(runs_cli, "create_taskset", lambda *a, **k: created)
    original_submit = submitting.submit
    attempted = []

    def interrupted(request, *, workspace):
        attempted.append((workspace, request.model_dump(mode="json", exclude_none=True)))
        raise KeyboardInterrupt

    submitting.submit = interrupted
    options = ["--idempotency-key", "cli key-$USER"] if key_source == "cli" else []
    result = submitting.invoke(
        "--harbor-hub",
        "datacurve/deep-swe@latest",
        "--no-build-images",
        "--spec-file",
        str(spec),
        "--workspace",
        "team",
        "--experiment",
        "my experiment",
        *options,
    )
    assert result.exit_code == 130, result.output
    command = next(
        line[line.index("nemo evals runs submit") :]
        for line in result.output.splitlines()
        if "nemo evals runs submit" in line
    )
    retry = shlex.split(command)
    assert retry.count("--idempotency-key") == 1
    assert "--harbor-hub" not in retry
    sent = []

    def resumed_submit(request, *, workspace):
        sent.append((workspace, request.model_dump(mode="json", exclude_none=True)))
        return original_submit(request, workspace=workspace)

    def changed_source(*args, **kwargs):
        raise AssertionError("recovery must use the already published taskset")

    monkeypatch.setattr(runs_cli, "create_taskset", changed_source)
    submitting.submit = resumed_submit
    resumed = submitting.invoke(*retry[4:])
    assert resumed.exit_code == 0, resumed.output
    assert sent == attempted


def test_spec_file_keeps_its_settings_and_takes_the_selected_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "oracle"}}},
                "execution": {"sandbox": {"profile": "harbor-open"}, "max_concurrent_task_jobs": 3},
            }
        )
    )
    request = run_request(
        "default/full-sel-1234abcd#" + "a" * 64,
        spec_file=spec,
        agent="ignored",
        dispatcher_profile=None,
        sandbox=SandboxSelection(profile="ignored", cpus=64),
    )
    assert isinstance(request.spec.tasks, TasksetRef)
    assert request.spec.tasks.root.startswith("default/full-sel-1234abcd#")
    assert request.execution.sandbox == SandboxSelection(profile="harbor-open")
    assert request.execution.max_concurrent_task_jobs == 3
    assert request.idempotency_key


def test_experiment_publishes_the_run_to_intake(tmp_path):
    plain = run_request("default/t#" + "a" * 64, spec_file=None, agent="codex", dispatcher_profile=None)
    assert plain.publication is None
    request = run_request(
        "default/t#" + "a" * 64,
        spec_file=None,
        agent="codex",
        dispatcher_profile=None,
        experiment="swe-agent-tuning",
    )
    assert request.publication is not None and request.publication.intake.experiment == "swe-agent-tuning"
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"spec": {"tasks": "default/t", "target": {"kind": "harbor"}}, "execution": {}}))
    # With --harbor-hub/--harbor-dir the dataset supplies the tasks, so the file may leave spec.tasks out.
    no_tasks = tmp_path / "no-tasks.json"
    no_tasks.write_text(json.dumps({"spec": {"target": {"kind": "harbor"}}, "execution": {}}))
    selected = run_request("default/t#" + "a" * 64, spec_file=no_tasks, agent="", dispatcher_profile=None)
    assert isinstance(selected.spec.tasks, TasksetRef) and selected.spec.tasks.root == "default/t#" + "a" * 64
    # --experiment replaces the file's publication before validation, so an invalid one doesn't block it.
    no_tasks.write_text(
        json.dumps(
            {
                "spec": {"target": {"kind": "harbor"}},
                "execution": {},
                "publication": {"intake": {"experiment": ""}},
            }
        )
    )
    overridden = run_request(
        "default/t#" + "a" * 64,
        spec_file=no_tasks,
        agent="",
        dispatcher_profile=None,
        experiment="x",
    )
    assert overridden.publication is not None and overridden.publication.intake.experiment == "x"
    for bad in ([], {"spec": ["not", "an", "object"]}):
        no_tasks.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="expected a TasksetRunInput JSON object"):
            run_request("default/t#" + "a" * 64, spec_file=no_tasks, agent="", dispatcher_profile=None)
    from_file = run_request("default/t#" + "a" * 64, spec_file=spec, agent="", dispatcher_profile=None, experiment="x")
    assert from_file.publication is not None and from_file.publication.intake.experiment == "x"
    assert from_file.publication.intake.dataset_name is None  # no Hub dataset: the server uses the taskset
    hub = run_request(
        "default/t#" + "a" * 64,
        spec_file=spec,
        agent="",
        dispatcher_profile=None,
        experiment="x",
        dataset=("datacurve/deep-swe", "sha256:" + "b" * 64),
    )
    assert hub.publication is not None
    assert (hub.publication.intake.dataset_name, hub.publication.intake.dataset_version) == (
        "datacurve/deep-swe",
        "sha256:" + "b" * 64,
    )


@pytest.mark.parametrize(
    ("sandbox", "expected"),
    [
        (SandboxSelection(cpus=4, storage_mb=20_480), {"cpus": 4, "storage_mb": 20_480}),
        (SandboxSelection(profile="harbor-open"), {"profile": "harbor-open"}),
        (None, None),
    ],
)
def test_agent_request_carries_the_sandbox_selection(sandbox, expected):
    request = run_request(
        "default/full#" + "a" * 64,
        spec_file=None,
        agent="oracle",
        dispatcher_profile=None,
        sandbox=sandbox,
    )
    assert request.model_dump(mode="json", exclude_none=True)["execution"].get("sandbox") == expected


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--sandbox-profile", "harbor-open"), ("--cpus", "4"), ("--memory-mb", "4"), ("--storage-mb", "4")],
)
def test_sandbox_flags_are_rejected_with_a_spec_file(tmp_path, flag, value):
    spec = tmp_path / "run.json"
    spec.write_text("{}")
    app = typer.Typer()

    def no_client() -> NemoClient:
        raise AssertionError("no request is made")

    add_runs_commands(app, no_client)
    result = CliRunner().invoke(app, ["runs", "submit", "--spec-file", str(spec), flag, value])
    assert result.exit_code == 2
    assert "don't apply with --spec-file" in " ".join(result.output.replace("│", " ").split())


TB = "https://github.com/harbor-framework/terminal-bench.git"
SHA = "2b0442c3" + "0" * 32  # what Harbor's ls-remote gives an annotated tag: the tag object
COMMIT = "9f1e7a55" + "0" * 32  # the commit the download checks out


class FakeGitRegistry:
    """Harbor's ``GitRepoRegistryClient``: three tasks under ``tasks/`` at ``SHA``; records what it downloads."""

    downloads: list[list[str]] = []
    commits: list[str | None] | None = None  # per selected task; default COMMIT for each

    repos: list = []

    def __init__(self, repo):
        self.repo = repo
        self._task_client = self
        FakeGitRegistry.repos.append(repo)

    async def get_dataset_metadata(self, name):
        from harbor.models.task.id import GitTaskId

        names = ["sum-three", "bun-sourcemap-leak", "greet-universe"]
        return SimpleNamespace(
            task_ids=[GitTaskId(git_url=self.repo.git_url, git_commit_id=SHA, path=Path("tasks") / n) for n in names]
        )

    async def download_tasks(self, task_ids, output_dir, export):
        FakeGitRegistry.downloads.append([t.get_name() for t in task_ids])
        paths = [output_dir / t.get_name() for t in task_ids]
        commits = FakeGitRegistry.commits or [COMMIT] * len(paths)
        results = [SimpleNamespace(path=p, resolved_git_commit_id=c) for p, c in zip(paths, commits, strict=True)]
        return SimpleNamespace(paths=paths, results=results)


@pytest.fixture
def git_registry(monkeypatch):
    from harbor.registry.client import git_repo

    FakeGitRegistry.downloads = []
    FakeGitRegistry.repos = []
    FakeGitRegistry.commits = None
    monkeypatch.setattr(git_repo, "GitRepoRegistryClient", FakeGitRegistry)
    return FakeGitRegistry


@pytest.mark.parametrize(
    ("tokens", "names"),
    [
        ("sum-three,0", ["sum-three", "bun-sourcemap-leak"]),
        ("1-2", ["greet-universe", "sum-three"]),
        ("1-2,0", ["greet-universe", "sum-three", "bun-sourcemap-leak"]),
    ],
)
def test_git_source_downloads_only_the_selected_tasks(git_registry, tmp_path, tokens, names):
    name, sha, dirs, total = download_git_tasks(f"{TB}@v3.0.0", tokens, tmp_path)
    # The checked-out commit, not the tag object an annotated tag's ls-remote gives.
    assert (name, sha, total) == ("harbor-framework/terminal-bench", COMMIT, 3)
    assert dirs == [tmp_path / task for task in names]  # name order: index 0 is bun-…
    assert git_registry.downloads == [names]


@pytest.mark.parametrize("tokens", ["0,2-1", "0-3", "0,0-1"])
def test_git_invalid_range_downloads_nothing(git_registry, tmp_path, tokens):
    with pytest.raises(SelectionError):
        download_git_tasks(f"{TB}@v3.0.0", tokens, tmp_path)
    assert git_registry.downloads == []


@pytest.fixture
def hub_registry(monkeypatch):
    downloads = []

    async def metadata(dataset):
        names = ["hello/sum-three", "hello/debug-agent-runtime-error", "hello/greet-universe"]
        return SimpleNamespace(
            name="hello/suite",
            version="sha256:" + "a" * 64,
            task_ids=[SimpleNamespace(get_name=lambda n=name: n) for name in names],
        )

    async def download_tasks(task_ids, output_dir, export):
        names = [task.get_name() for task in task_ids]
        downloads.append(names)
        return SimpleNamespace(paths=[output_dir / name.rpartition("/")[2] for name in names])

    client = SimpleNamespace(get_dataset_metadata=metadata, _task_client=SimpleNamespace(download_tasks=download_tasks))
    monkeypatch.setattr(runs_cli, "PackageDatasetClient", lambda: client)
    return downloads


@pytest.mark.parametrize(
    ("tokens", "names"),
    [("1-2", ["greet-universe", "sum-three"]), ("1-2,0", ["greet-universe", "sum-three", "debug-agent-runtime-error"])],
)
def test_hub_range_downloads_only_selected_tasks(hub_registry, tmp_path, tokens, names):
    dataset, version, paths = download_hub_tasks("hello/suite@latest", tokens, tmp_path)
    assert (dataset, version) == ("hello/suite", "sha256:" + "a" * 64)
    assert hub_registry == [[f"hello/{name}" for name in names]]
    assert paths == [tmp_path / name for name in names]


@pytest.mark.parametrize("tokens", ["0,2-1", "0-3", "0,0-1"])
def test_hub_invalid_range_downloads_nothing(hub_registry, tmp_path, tokens):
    with pytest.raises(SelectionError):
        download_hub_tasks("hello/suite@latest", tokens, tmp_path)
    assert hub_registry == []


@pytest.mark.parametrize("second", ["a" * 40, None], ids=["another-commit", "unresolved"])
def test_git_source_needs_one_commit_for_every_task(git_registry, tmp_path, second):
    git_registry.commits = [COMMIT, second]
    with pytest.raises(RuntimeError, match="not one commit"):
        download_git_tasks(f"{TB}@v3.0.0", "sum-three,0", tmp_path)


def test_git_source_needs_a_ref():
    check_git_source(f"{TB}@v3.0.0")
    check_git_source("https://github.com/harbor-framework/terminal-bench/tree/2b0442c3/tasks")
    with pytest.raises(ValueError, match="names no tag or commit"):
        check_git_source(TB)
    result = CliRunner().invoke(_runs_app(), ["runs", "submit", "--harbor-git", TB])
    assert result.exit_code == 2
    assert "names no tag or commit" in " ".join(result.output.replace("│", " ").split())


@pytest.mark.parametrize(
    ("other", "flags"),
    [
        (["--harbor-dir", "."], "--harbor-dir and --harbor-git"),
        (["--harbor-hub", "a/b@1"], "--harbor-hub and --harbor-git"),
    ],
)
def test_git_source_excludes_the_other_sources(other, flags):
    result = CliRunner().invoke(_runs_app(), ["runs", "submit", *other, "--harbor-git", f"{TB}@v3.0.0"])
    assert result.exit_code == 2
    assert f"not {flags}" in " ".join(result.output.replace("│", " ").split())


def test_list_tasks_reads_the_git_repo(git_registry):
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "submit", "--harbor-git", f"{TB}@v3.0.0", "--list-tasks"])
    assert result.exit_code == 0, result.output
    assert result.output.split("\n")[1:4] == ["  0  bun-sourcemap-leak", "  1  greet-universe", "  2  sum-three"]
    assert git_registry.downloads == []


def test_a_tree_url_reads_tasks_from_its_directory(git_registry):
    result = CliRunner().invoke(
        _listing_app(),
        ["runs", "submit", "--harbor-git", "https://github.com/acme/evals/tree/v2/suites/smoke", "--list-tasks"],
    )
    assert result.exit_code == 0, result.output
    (repo,) = git_registry.repos
    assert (repo.git_url, repo.ref, repo.subdir) == ("https://github.com/acme/evals", "v2", "suites/smoke")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.CalledProcessError(128, ["git", "ls-remote", TB, "v9"], b"", b"fatal: repository not found\n"),
            "Error: git ls-remote failed: fatal: repository not found",
        ),
        (FileNotFoundError(2, "No such file or directory", "git"), "Error: --harbor-git needs git on PATH"),
    ],
)
def test_a_failed_git_command_is_an_error_not_a_traceback(monkeypatch, error, message):
    from harbor.registry.client import git_repo

    class UnreachableRepo(FakeGitRegistry):
        async def get_dataset_metadata(self, name):
            raise error

    monkeypatch.setattr(git_repo, "GitRepoRegistryClient", UnreachableRepo)
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "submit", "--harbor-git", f"{TB}@v9", "--list-tasks"])
    assert result.exit_code == 1
    assert message in result.output
    assert isinstance(result.exception, SystemExit)  # typer.Exit, not the git error


def _listing_app() -> typer.Typer:
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return app


def _runs_app() -> typer.Typer:
    def no_client() -> NemoClient:
        raise AssertionError("no request is made")

    app = typer.Typer()
    add_runs_commands(app, no_client)
    return app


@pytest.mark.parametrize(
    ("status", "advised"),
    [(403, True), (500, True), (None, True), (400, False), (404, False), (409, False), (422, False), (429, False)],
    ids=["403", "5xx", "transport", "400", "404", "409", "422", "429"],
)
def test_a_failed_submit_names_its_key_and_advises_a_rerun_only_when_the_run_may_be_accepted(
    submitting, status, advised
):
    if status is None:
        cause = NemoTransportError(httpx.ReadTimeout("t", request=httpx.Request("POST", "http://platform.test")))
        submitting.refuse = TasksetRunSubmitError(cause, "k-1")
    else:
        submitting.refuse = submit_error(status, "nope")
    result = submitting.invoke("--taskset", STORED, "--idempotency-key", "k-1")
    assert result.exit_code == 1
    assert "Error: " in result.output and "this submit used idempotency_key=k-1" in result.output
    retry = f"submit with the same key: nemo evals runs submit --taskset {shlex.quote(STORED)} --idempotency-key k-1"
    assert (retry in result.output) is advised, result.output


@pytest.fixture
def publishing(monkeypatch):
    """``publish_tasks`` over fake discovery and a fake evaluator holding one stored task, ``deep-swe-a``."""
    expected = SimpleNamespace(spec="spec-v1", metadata=[])
    monkeypatch.setattr(
        "nhx_evals_sdk.agent_eval.runtimes.harbor.runtime.discover_harbor_tasks", lambda path: [path.name]
    )
    monkeypatch.setattr("nemo_evals.sdk.task_preparation.describe_harbor_task", lambda source, workspace: expected)
    seen = SimpleNamespace(replaced=[], stored={"deep-swe-a": SimpleNamespace(spec="spec-v1", metadata=[], revision=3)})

    def retrieve(name, workspace):
        if name not in seen.stored:
            raise NotFoundError(httpx.Response(404, json={"detail": "no task"}))
        return seen.stored[name]

    def replace(name, task, workspace):
        seen.replaced.append(name)
        return SimpleNamespace(revision=4)

    revisions = lambda name, workspace, page_size: SimpleNamespace(  # noqa: E731
        data=[SimpleNamespace(revision=r, content_hash=f"{r}" * 64) for r in (3, 4)]
    )
    seen.evaluator = SimpleNamespace(
        tasks=SimpleNamespace(retrieve=retrieve, replace=replace, list_revisions=revisions)
    )
    return seen


def test_an_unchanged_task_is_pinned_without_publishing(publishing, tmp_path):
    lines = []
    refs = publish_tasks(publishing.evaluator, [tmp_path / "a"], prefix="deep-swe", workspace="ac9", echo=lines.append)
    assert publishing.replaced == [] and refs == ["ac9/deep-swe-a#" + "3" * 64]
    assert lines == ["Unchanged task ac9/deep-swe-a#" + "3" * 64]


@pytest.mark.parametrize("stored", [None, SimpleNamespace(spec="spec-v0", metadata=[], revision=3)])
def test_a_new_or_changed_task_is_published(publishing, tmp_path, stored):
    if stored is None:
        publishing.stored.clear()
    else:
        publishing.stored["deep-swe-a"] = stored
    lines = []
    refs = publish_tasks(publishing.evaluator, [tmp_path / "a"], prefix="deep-swe", workspace="ac9", echo=lines.append)
    assert publishing.replaced == ["deep-swe-a"] and refs == ["ac9/deep-swe-a#" + "4" * 64]
    assert lines == ["Published task ac9/deep-swe-a#" + "4" * 64]


@pytest.mark.parametrize("spec_key", [None, "from-file"])
def test_an_explicit_idempotency_key_wins(tmp_path, spec_key):
    ref = "default/t#" + "a" * 64
    assert (
        run_request(
            ref, spec_file=None, agent="oracle", dispatcher_profile=None, idempotency_key="k-123"
        ).idempotency_key
        == "k-123"
    )
    spec = tmp_path / "run.json"
    body = {
        "spec": {"tasks": "default/full", "target": {"kind": "harbor"}},
        "execution": {},
        "idempotency_key": spec_key,
    }
    spec.write_text(json.dumps(body))
    explicit = run_request(ref, spec_file=spec, agent="", dispatcher_profile=None, idempotency_key="k-123")
    assert explicit.idempotency_key == "k-123"
    default = run_request(ref, spec_file=spec, agent="", dispatcher_profile=None)
    assert default.idempotency_key == (spec_key or default.idempotency_key) and default.idempotency_key


def test_spec_selection_ignores_a_publication_that_experiment_replaces(tmp_path):
    spec = tmp_path / "run.json"
    body = {
        "spec": {"tasks": "default/suite#" + "e" * 64, "target": {"kind": "harbor"}},
        "execution": {"sandbox_policy": "p"},
        "publication": {"intake": {"experiment": ""}},  # invalid on its own; --experiment replaces it
    }
    spec.write_text(json.dumps(body))
    member = "default/a#" + "1" * 64
    tasks = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(revision=1, spec=SimpleNamespace())
    )
    tasksets = SimpleNamespace(retrieve=lambda name, revision, workspace: SimpleNamespace(tasks=[TaskRef(member)]))
    evaluator = cast(Evaluator, SimpleNamespace(tasks=tasks, tasksets=tasksets))
    assert _select_from_spec(evaluator, spec, None, "default") == ([member], 1, "suite")


def test_spec_selection_needs_a_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps({"spec": {"tasks": ["default/a#" + "1" * 64], "target": {"kind": "harbor"}}, "execution": {}})
    )
    with pytest.raises(ValueError, match="must name a taskset"):
        _select_from_spec(cast(Evaluator, SimpleNamespace()), spec, None, "default")


# --- runs status / runs results ----------------------------------------------------------------------------

RUN = {
    "run_id": "run-0123456789abcdef",
    "workspace": "default",
    "taskset_ref": "default/debug-1f9c2a7e#" + "e" * 64,
    "phase": "accepted",
    "result_state": "pending",
    "accepted_at": "2026-10-16T12:00:00Z",
    "timeout_seconds": 3600,
    "execution_deadline_at": "2026-10-16T13:00:00Z",
    "finalization_deadline_at": "2026-10-16T14:00:00Z",
}
NOW = "2026-10-16T12:00:00Z"
RUN_ID = "run-0123456789abcdef"


def run_status(**overrides) -> TasksetRunStatus:
    return TasksetRunStatus.model_validate(RUN | overrides)


def named(index, name, phase=None, job=None, job_status=None, error=None, publication=None) -> NamedTask:
    """A task of a run; with no ``phase`` its ``/tasks`` row doesn't exist yet."""
    row = None
    if phase is not None:
        row = TasksetRunTask(
            task=f"task-{index}",
            task_index=index,
            phase=phase,
            job=job,
            job_status=job_status,
            error=error,
            publication=publication,
        )
    return NamedTask(index=index, name=name, task=row)


def score(name, mean, count, nan=0):
    return {"name": name, "count": count, "nan_count": nan, "mean": mean}


def result_record(job_id, scores, trials=1, errors=0):
    return {
        "id": f"r-{job_id}",
        "name": job_id,
        "workspace": "default",
        "job_id": job_id,
        "scores": {"scores": scores},
        "bundle_ref": f"default/{job_id}#bundle",
        "summary": {"task_count": 1, "trial_count": trials, "score_count": len(scores), "error_count": errors},
        "created_at": NOW,
        "updated_at": NOW,
    }


def job(name, status, task_id):
    return {
        "id": f"id-{name}",
        "attempt_id": "a",
        "name": name,
        "workspace": "default",
        "source": "evals.harbor-task",
        "spec": {"spec": {"tasks": [{"id": task_id}]}},
        "platform_spec": {
            "steps": [{"name": "run", "executor": {"provider": "cpu", "container": {"image": "x"}}, "config": {}}]
        },
        "fileset": "default/f",
        "status": status,
    }


def page(items):
    n = len(items)
    pagination = {"page": 1, "page_size": 100, "current_page_size": n, "total_pages": 1, "total_results": n}
    return {"data": items, "pagination": pagination}


def results_platform(*, run=None, rows=None, taskset_status=200):
    """A 3-task run: its pinned taskset, its ``/tasks`` rows and task 0's result record. Jobs is never read."""
    run = run if run is not None else RUN | {"phase": "running"}
    if rows is None:
        rows = [
            {
                "task": "task-0",
                "task_index": 0,
                "phase": "succeeded",
                "job": f"{RUN_ID}-task-0",
                "job_status": "completed",
            },
            {
                "task": "task-1",
                "task_index": 1,
                "phase": "registered",
                "job": f"{RUN_ID}-task-1",
                "job_status": "active",
            },
        ]
    results = {
        f"{RUN_ID}-task-0": result_record(
            f"{RUN_ID}-task-0",
            [score("harbor_reward.reward", 1.0, 2), score("harbor_reward.style", 0.5, 2, nan=1)],
            trials=2,
        )
    }
    taskset = {
        "id": "ts",
        "name": "debug-1f9c2a7e",
        "workspace": "default",
        "tasks": ["default/greet#" + "1" * 64, "default/sum#" + "2" * 64, "default/sort#" + "3" * 64],
        "revision": 1,
        "tags": {},
        "created_at": NOW,
        "updated_at": NOW,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}"):
            return httpx.Response(200, json=run)
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}/tasks"):
            return httpx.Response(200, json=page(rows))
        if "/tasksets/debug-1f9c2a7e" in path:
            return httpx.Response(taskset_status, json=taskset if taskset_status == 200 else {"detail": "x"})
        if path.endswith("/agent-eval-results"):
            found = [r for r in results.values() if request.url.params.get("filter[job_id]") == r["job_id"]]
            return httpx.Response(200, json=page(found))
        if path.endswith(f"/agent-eval-results/{RUN_ID}"):  # the run's merged record
            return httpx.Response(404, json={"detail": f"Result not found: default/{RUN_ID}"})
        if "/jobs/" in path and "-bundle-" in path:
            return httpx.Response(404, json={"detail": "x"})  # no bundle Job yet
        raise AssertionError(f"unexpected {request.method} {request.url}")

    return NemoClient(base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def invoke_on(client, *args):
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", *args, RUN_ID, "--workspace", "default"])


def test_results_show_completed_tasks_scores_by_their_taskset_names():
    result = invoke_on(results_platform(), "results")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == [
        "#",
        "task",
        "status",
        "trials",
        "errors",
        "harbor_reward.reward",
        "harbor_reward.style",
    ]
    assert lines[1].split() == ["0", "greet", "succeeded", "2", "0", "1.00", "0.50"]
    assert lines[2].split() == ["1", "sum", "registered", "(active)", "-", "-"]
    assert lines[3].split() == ["2", "sort", "waiting", "-", "-"]
    assert "1/3 tasks with results, 2 trials" in result.output


def test_results_name_what_is_missing():
    """A missing run reads as one; a missing merged record of a run that exists reads as the server says."""
    gone = httpx.MockTransport(lambda request: httpx.Response(404, json={"detail": "Taskset run not found"}))
    result = invoke_on(NemoClient(base_url="http://platform.test", http_client=httpx.Client(transport=gone)), "results")
    assert result.exit_code == 1 and result.output == f"Error: run default/{RUN_ID} not found\n"

    ready = RUN | {"phase": "completed", "result_state": "ready"}
    result = invoke_on(results_platform(run=ready), "results")
    assert result.exit_code == 1
    assert result.output == f"Error: HTTP 404: Result not found: default/{RUN_ID}\n"


def test_results_json_shares_the_status_task_shape():
    result = invoke_on(results_platform(), "results", "--json")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert set(body) == {"run", "tasks"}  # no merged record yet
    assert [(t["index"], t["name"], t["state"]) for t in body["tasks"]] == [
        (0, "greet", "succeeded"),
        (1, "sum", "registered (active)"),
        (2, "sort", "waiting"),
    ]
    assert body["tasks"][0]["task"]["job_status"] == "completed" and "result" in body["tasks"][0]
    assert "task" not in body["tasks"][2] and "result" not in body["tasks"][2]


@pytest.mark.parametrize("command", ["status", "results"])
def test_an_active_run_without_rows_shows_every_task_waiting(command):
    result = invoke_on(results_platform(rows=[]), command)
    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.output.splitlines() if line[:3].strip().isdigit()]
    assert [(r[1], r[2]) for r in rows] == [("greet", "waiting"), ("sum", "waiting"), ("sort", "waiting")]
    assert ("0/3 done" if command == "status" else "0/3 tasks with results") in result.output


@pytest.mark.parametrize("command", ["status", "results"])
def test_a_run_without_rows_or_a_readable_taskset_has_no_tasks_yet(command):
    result = invoke_on(results_platform(rows=[], taskset_status=403), command)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == "no tasks yet"


@pytest.mark.parametrize(
    ("rows", "states"),
    [
        ([], ["not run", "not run", "not run"]),
        ([{"task": "task-1", "task_index": 1, "phase": "failed", "error": "plan"}], ["not run", "failed", "not run"]),
    ],
    ids=["no-rows", "one-row"],
)
def test_a_finished_run_shows_tasks_without_rows_as_not_run(rows, states):
    result = invoke_on(results_platform(run=RUN | {"phase": "failed", "result_state": "none"}, rows=rows), "status")
    assert result.exit_code == 0, result.output
    table = [line for line in result.output.splitlines() if line[:3].strip().isdigit()]
    assert [_state_of(line) for line in table] == states


def _state_of(line: str) -> str:
    cells = [cell for cell in line.split("  ") if cell.strip()]
    return cells[2].strip()


@pytest.mark.parametrize("command", ["status", "results"])
def test_a_reader_who_cannot_read_the_taskset_sees_member_names(command):
    result = invoke_on(results_platform(taskset_status=403), command)
    assert result.exit_code == 0, result.output
    names = [line.split()[1] for line in result.output.splitlines() if line[:3].strip().isdigit()]
    assert names == ["task-0", "task-1"]


def test_rollup_weights_task_means_by_count():
    first = AgentEvalResult.model_validate(result_record("j0", [score("harbor_reward.reward", 1.0, 3)]))
    second = AgentEvalResult.model_validate(
        result_record("j1", [score("harbor_reward.reward", 0.0, 1, nan=1), score("harbor_reward.extra", 0.5, 1)])
    )
    rollup = {s.name: s for s in rollup_scores([first, second])}
    assert rollup["harbor_reward.reward"] == ScoreRollup("harbor_reward.reward", 0.75, 4, 1, 2)
    assert rollup["harbor_reward.extra"].tasks == 1


def test_status_shows_what_each_task_published():
    done = {"evaluation_id": "e"}
    tasks = [
        named(0, "a", "succeeded", "j0", publication=done | {"status": "completed", "trial_count": 2}),
        named(1, "b", "succeeded", "j1", publication=done | {"status": "error", "error": "intake 503"}),
        named(2, "c", "registered", "j2", job_status="active"),
    ]
    lines = _status_table(run_status(), tasks)
    assert lines[-5].endswith("published")
    assert lines[-4].endswith("✓ 2 trials") and lines[-3].endswith("✗ intake 503") and lines[-2].endswith("  -")


def test_status_counts_done_failed_and_expired_apart():
    tasks = [
        named(0, "a", "succeeded", "j0"),
        named(1, "b", "failed", "j1", error="job_failed: exit 1"),
        named(2, "c", "expired"),
        named(3, "d", "registered", "j3", job_status="active"),
        named(4, "e", "image_ready"),
    ]
    assert _status_table(run_status(), tasks)[-1] == "3/5 done (1 succeeded, 1 failed, 1 expired)"


def test_status_counts_settled_tasks_whose_state_shows_an_unexpected_job_status():
    tasks = [
        named(0, "a", "succeeded", "j0"),
        named(1, "b", "failed", "j1", job_status="completed", error="no_result"),
        named(2, "c", "cancelled", "j2", job_status="completed"),
    ]
    lines = _status_table(run_status(result_state="failed"), tasks)
    assert lines[-1] == "3/3 done (1 succeeded, 1 failed, 1 cancelled)"
    assert any(line.startswith("  completed tasks' scores:") for line in lines)


@pytest.mark.parametrize(
    ("task", "state"),
    [
        ({"phase": "image_ready"}, "queued"),
        ({"phase": "image_ready", "job": "run-x-task-0"}, "queued (creating)"),
        ({"phase": "registered", "job": "run-x-task-0", "job_status": "active"}, "registered (active)"),
        ({"phase": "registered", "job": "run-x-task-0", "job_status": "pending"}, "registered (pending)"),
        ({"phase": "registered", "job": "run-x-task-0"}, "registered"),  # no status read yet
        ({"phase": "failed", "job": "run-x-task-0", "job_status": "error", "error": "job_failed"}, "failed"),
        ({"phase": "expired"}, "expired"),
        # A settled task shows its Job's status only when the phase doesn't imply it.
        ({"phase": "succeeded", "job": "run-x-task-0", "job_status": "completed"}, "succeeded"),
        ({"phase": "cancelled", "job": "run-x-task-0", "job_status": "completed"}, "cancelled (completed)"),
        ({"phase": "expired", "job": "run-x-task-0", "job_status": "cancelled"}, "expired"),
        (
            {"phase": "failed", "job": "run-x-task-0", "job_status": "completed", "error": "no_result"},
            "failed (completed)",
        ),
    ],
)
def test_task_state_names_the_queue_and_the_jobs_status(task, state):
    assert task_state(TasksetRunTask.model_validate({"task": "task-0", "task_index": 0} | task)) == state


@pytest.mark.parametrize(
    ("progress", "line"),
    [
        (None, "no tasks yet; the run controller creates them on its next pass"),
        (
            {
                "tasks": 113,
                "members": {"image_ready": 60, "registered": 20, "succeeded": 30, "failed": 3},
                "queued": 60,
                "task_jobs": {"pending": 2, "active": 18},
            },
            "113 tasks: 60 queued, 20 running (18 active, 2 pending), 30 succeeded, 3 failed",
        ),
        (
            {"tasks": 4, "members": {"image_ready": 3, "registered": 1}, "queued": 1, "task_jobs": {}},
            "4 tasks: 1 queued, 2 creating, 1 running",
        ),
        (
            {"tasks": 1, "members": {"cancelled": 1}, "queued": 0, "task_jobs": {}},
            "1 task: 1 cancelled",
        ),
    ],
)
def test_progress_line_counts_queued_creating_running_and_settled(progress, line):
    assert _progress_line(progress and TasksetRunProgress.model_validate(progress)) == line


def results_of(*tasks: TaskResult, record: AgentEvalResult | None = None) -> TasksetRunResults:
    return TasksetRunResults(run=run_status(phase="completed"), tasks=list(tasks), result=record)


def completed_task(name: str = "a") -> TaskResult:
    task = named(0, name, "succeeded", "run-x-task-0", job_status="completed")
    return TaskResult(index=task.index, name=task.name, task=task.task)


def test_completed_job_without_a_record_is_flagged():
    assert _results_table(results_of(completed_task()))[-2] == (
        "warning: completed without a result record: run-x-task-0"
    )


def test_results_use_the_runs_own_record_when_ready():
    record = AgentEvalResult.model_validate(
        result_record(
            "run-x-bundle-1", [score("harbor_reward.reward", 0.5, 4), score("harbor_reward.reward.pass@1", 0.5, 2)]
        )
        | {"name": "run-x", "bundle_ref": "default/job-fileset-run-x-bundle-1#results/a/agent-eval-results"}
    )
    lines = _results_table(results_of(completed_task(), record=record))
    assert lines[-5].startswith("run (merged bundle default/job-fileset-run-x-bundle-1#")
    assert lines[-4] == "  harbor_reward.reward: 0.500 (n=4, nan=0)"
    assert lines[-1] == "Download the merged bundle: nemo evals runs download run-x"
    lines = _results_table(results_of(completed_task(), record=record), workspace_flag=" --workspace b")
    assert lines[-1] == "Download the merged bundle: nemo evals runs download run-x --workspace b"


def test_printed_commands_name_the_workspace_only_when_it_is_not_the_configured_one():
    assert _workspace_flag("team", "team") == ""
    assert _workspace_flag("other", "team") == " --workspace other"


def test_status_shows_result_state_and_bundle():
    run = run_status(phase="completed", result_state="ready")
    bundle = BundleJobView("run-x-bundle-1", 1, "completed")
    lines = _status_table(run, [named(0, "a", "succeeded", "j0")], bundle)
    assert lines[3] == "  status    completed; the run's result is ready (phase completed, result ready)"
    assert lines[5] == "  bundle    run-x-bundle-1 (attempt 1 of 3): completed"


MODULE_ERROR = "/app/.venv/bin/python: No module named nemo_evals.tasks.taskset_bundle"


def test_a_failed_bundle_says_tasks_finished_and_shows_its_own_error():
    progress = {"tasks": 1, "members": {"succeeded": 1}, "queued": 0, "task_jobs": {}}
    error = "bundle Job run-x-bundle-3 ended error"
    run = run_status(phase="completed", result_state="failed", error=error, progress=progress)
    bundle = BundleJobView("run-x-bundle-3", 3, "error", MODULE_ERROR)
    lines = _status_table(run, [named(0, "a", "succeeded", "j0")], bundle)
    assert lines[3:9] == [
        "  status    tasks finished, but merging their results into the run's result failed"
        " (phase completed, result failed)",
        "  progress  1 task: 1 succeeded",
        "  bundle    run-x-bundle-3 (attempt 3 of 3): error",
        f"            {MODULE_ERROR}",
        "            logs: nemo jobs get-logs run-x-bundle-3",
        "  completed tasks' scores: nemo evals runs results run-0123456789abcdef",
    ]
    assert not any(line.startswith("  error") for line in lines)  # "bundle Job … ended error" only repeats it


@pytest.mark.parametrize("status", ["active", "cancelled"])
def test_a_deadline_error_shows_whatever_the_bundle_did(status):
    error = "finalization deadline passed before the result was saved"
    run = run_status(phase="completed", result_state="failed", error=error)
    lines = _status_table(run, [named(0, "a", "succeeded", "j0")], BundleJobView("run-x-bundle-1", 1, status))
    assert f"  error     {error}" in lines
    assert not any("no error reported" in line for line in lines)
    assert any(line.strip().startswith("logs:") for line in lines) is (status == "cancelled")


def test_the_status_screen_shows_how_long_the_run_queued():
    started = run_status(phase="running", started_at="2026-10-16T12:50:12Z")
    assert "  started   2026-10-16 12:50:12 (queued 50m12s)" in _status_table(started, [])
    waiting = _status_table(run_status(), [], now=datetime(2026, 10, 16, 13, 10, 5, tzinfo=UTC))
    assert waiting[3] == "  queued    1h10m so far"
    never_started = run_status(phase="expired", error="no task started within the queue timeout (48 h)")
    assert not any(line.startswith(("  started", "  queued")) for line in _status_table(never_started, []))


@pytest.mark.parametrize("phase", ["expired", "cancelled"])
@pytest.mark.parametrize(
    "result,words",
    [
        ("ready", "a result over the tasks that finished is ready"),
        ("failed", "merging the finished tasks' results failed"),
        ("none", None),
    ],
)
def test_a_stopped_run_says_whether_its_finished_tasks_have_a_result(phase, result, words):
    run = run_status(phase=phase, result_state=result, started_at=RUN["accepted_at"])
    state = f"{phase}; {words}" if words else phase
    assert f"  status    {state} (phase {phase}, result {result})" in _status_table(run, [])


def test_a_running_run_whose_bundle_job_runs_reads_as_merging():
    run = run_status(phase="running", started_at=RUN["accepted_at"])
    lines = _status_table(run, [named(0, "a", "succeeded", "j0")], BundleJobView("run-x-bundle-1", 1, "active"))
    assert "  status    merging task results (phase running, result pending)" in lines


FAILED_POD = {
    "phase": "Failed",
    "containers": [
        {
            "name": "nemo-job-task",
            "terminated": {
                "exit_code": 1,
                "reason": "Error",
                "message": "2026/10/16 17:31:44 INFO using OTLP log authentication auth_mechanism=x\n"
                "[launcher] 2026/10/16 17:31:44 Running main process: python [-m nemo_evals.tasks.taskset_bundle]\n"
                f"{MODULE_ERROR}\n"
                "[launcher] 2026/10/16 17:31:44 Process exited with error: exit status 1\n",
            },
        }
    ],
}
BACKOFF = {"message": "Job failed: Job has reached the specified backoff limit"}


@pytest.mark.parametrize(
    ("status_details", "expected"),
    [(FAILED_POD, MODULE_ERROR), ({}, BACKOFF["message"])],
    ids=["container", "fallback"],
)
def test_job_error_prefers_the_failed_process_over_the_retry_summary(status_details, expected):
    failed = job("run-x-bundle-1", "error", "t") | {"status_details": status_details, "error_details": BACKOFF}
    assert job_error(HelixJobResponse.model_validate(failed)) == expected


@pytest.mark.parametrize("first", ["error", None], ids=["failed", "unreadable"])
def test_newest_bundle_job_is_the_last_attempt_with_its_error(first):
    """The controller skips a name its create left unreadable (404), so the search goes on past it."""
    jobs = {"run-x-bundle-2": job("run-x-bundle-2", "error", "t") | {"status_details": FAILED_POD}}
    if first is not None:
        jobs["run-x-bundle-1"] = job("run-x-bundle-1", first, "t")

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, json=jobs[name]) if name in jobs else httpx.Response(404, json={"detail": "x"})

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert newest_bundle_job(client, "run-x", workspace="default") == BundleJobView(
        "run-x-bundle-2", 2, "error", MODULE_ERROR
    )


@pytest.mark.parametrize(
    "command",
    [
        ["status", "run-0123456789abcdef"],
        ["tasks", "run-0123456789abcdef"],
        ["results", "run-0123456789abcdef"],
        ["download", "run-0123456789abcdef"],
        ["list"],
    ],
)
def test_run_commands_default_to_the_configured_workspace(command):
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(404, json={"detail": "not found"})

    client = NemoClient(
        base_url="http://platform.test",
        workspace="team",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", *command])
    assert result.exit_code == 1
    assert paths and paths[0].startswith("/apis/evals/v2/workspaces/team/")


@pytest.mark.parametrize("command", ["status", "results"])
def test_an_sdk_error_exits_with_a_message_not_a_traceback(command):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agent-evaluate/runs/run-0123456789abcdef"):
            return httpx.Response(200, json=RUN)
        return httpx.Response(500, json={"detail": "Internal server error"})  # the taskset read, through the SDK

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", command, "run-0123456789abcdef", "--workspace", "default"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert result.output.startswith("Error: ")


RECORDED_TASKS = [  # GET …/runs/{id}/tasks, as the API returns it (nulls omitted)
    {
        "task": "task-0",
        "task_index": 0,
        "phase": "succeeded",
        "job": f"{RUN_ID}-task-0",
        "job_status": "completed",
        "publication": {
            "status": "completed",
            "evaluation_id": "e",
            "trial_count": 2,
            "evaluator_result_count": 0,
            "skipped": [],
        },
    },
    {"task": "task-1", "task_index": 1, "phase": "registered", "job": f"{RUN_ID}-task-1", "job_status": "active"},
    {"task": "task-2", "task_index": 2, "phase": "image_ready", "job": f"{RUN_ID}-task-2"},
    {"task": "task-3", "task_index": 3, "phase": "image_ready"},
    {
        "task": "task-4",
        "task_index": 4,
        "phase": "failed",
        "job": f"{RUN_ID}-task-4",
        "job_status": "error",
        "error": "job_failed: exit 1\ntraceback",
    },
]
RECORDED_PROGRESS = {
    "tasks": 5,
    "members": {"succeeded": 1, "registered": 1, "image_ready": 2, "failed": 1},
    "queued": 1,
    "task_jobs": {"active": 1},
}


def recorded_platform(tasks_page_size=2):
    """Recorded API responses for a running run; returns the client and every request it made."""
    requests: list[httpx.Request] = []
    taskset = {
        "id": "ts",
        "name": "debug-1f9c2a7e",
        "workspace": "default",
        "tasks": [f"default/{name}#" + "1" * 64 for name in ["greet", "sum", "sort", "fizz", "buzz"]],
        "revision": 1,
        "tags": {},
        "created_at": NOW,
        "updated_at": NOW,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}"):
            return httpx.Response(200, json=RUN | {"phase": "running", "progress": RECORDED_PROGRESS})
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}/tasks"):
            number = int(request.url.params["page"])
            data = RECORDED_TASKS[(number - 1) * tasks_page_size : number * tasks_page_size]
            pages = -(-len(RECORDED_TASKS) // tasks_page_size)
            return httpx.Response(
                200, json=page(data) | {"pagination": page(data)["pagination"] | {"page": number, "total_pages": pages}}
            )
        if "/tasksets/debug-1f9c2a7e" in path:
            return httpx.Response(200, json=taskset)
        if "/jobs/" in path:
            return httpx.Response(404, json={"detail": "x"})  # no bundle Job yet
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return client, requests


def test_status_reads_the_run_and_its_tasks_with_no_jobs_call_per_task():
    client, requests = recorded_platform()
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "status", RUN_ID, "--workspace", "default"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[3:5] == [
        "  status    running (phase running, result pending)",
        "  progress  5 tasks: 1 queued, 1 creating, 1 running (1 active), 1 succeeded, 1 failed",
    ]
    assert [line.split() for line in lines[6:12]] == [
        ["#", "task", "state", "Job", "published"],
        ["0", "greet", "succeeded", f"{RUN_ID}-task-0", "✓", "2", "trials"],
        ["1", "sum", "registered", "(active)", f"{RUN_ID}-task-1", "-"],
        ["2", "sort", "queued", "(creating)", f"{RUN_ID}-task-2", "-"],
        ["3", "fizz", "queued", "-", "-"],
        ["4", "buzz", "failed", f"{RUN_ID}-task-4", "-", "job_failed:", "exit", "1"],
    ]
    assert lines[-1] == "2/5 done (1 succeeded, 1 failed)"
    task_pages = [r.url.params["page"] for r in requests if r.url.path.endswith("/tasks")]
    assert task_pages == ["1", "2", "3"]
    # What task-0 published comes from /tasks, as the controller recorded it: no task Job is read.
    jobs_read = [r.url.path.rsplit("/", 1)[1] for r in requests if "/jobs/" in r.url.path]
    assert [name for name in jobs_read if "-task-" in name] == []


def test_status_json_lists_each_tasks_state():
    client, _ = recorded_platform()
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "status", RUN_ID, "--workspace", "default", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["run"]["progress"] == RECORDED_PROGRESS
    assert body["tasks"][1]["task"]["job_status"] == "active"  # the raw status, beside the state cell
    assert [(t["index"], t["name"], t["state"]) for t in body["tasks"]] == [
        (0, "greet", "succeeded"),
        (1, "sum", "registered (active)"),
        (2, "sort", "queued (creating)"),
        (3, "fizz", "queued"),
        (4, "buzz", "failed"),
    ]


def invoke_tasks(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "tasks", RUN_ID, "--workspace", "default", *args]), requests


def test_tasks_lists_the_api_page_with_each_tasks_state_and_no_jobs_call():
    body = run_page(RECORDED_TASKS, total_pages=3, total=45)
    result, requests = invoke_tasks(
        ["--phase", "registered", "--page", "1", "--page-size", "5"], httpx.Response(200, json=body)
    )
    assert result.exit_code == 0, result.output
    (request,) = requests
    assert request.url.path == f"/apis/evals/v2/workspaces/default/agent-evaluate/runs/{RUN_ID}/tasks"
    assert dict(request.url.params) == {"phase": "registered", "page": "1", "page_size": "5"}
    lines = result.output.splitlines()
    assert lines[0].split() == ["#", "task", "state", "Job"]
    assert lines[1].split() == ["0", "task-0", "succeeded", f"{RUN_ID}-task-0"]
    assert lines[2].split() == ["1", "task-1", "registered", "(active)", f"{RUN_ID}-task-1"]
    assert lines[3].split() == ["2", "task-2", "queued", "(creating)", f"{RUN_ID}-task-2"]
    assert lines[4].split() == ["3", "task-3", "queued", "-"]
    assert lines[5].split() == ["4", "task-4", "failed", f"{RUN_ID}-task-4", "job_failed:", "exit", "1"]
    assert lines[6] == "page 1 of 3 (45 tasks); next: --page 2"


def test_tasks_without_members_and_as_json():
    result, _ = invoke_tasks([], httpx.Response(200, json=run_page([])))
    assert result.exit_code == 0 and result.output == "No tasks.\n"
    body = run_page(RECORDED_TASKS)
    result, requests = invoke_tasks(["--json"], httpx.Response(200, json=body))
    assert json.loads(result.output) == body
    assert dict(requests[0].url.params) == {"page": "1", "page_size": "100"}


def test_tasks_of_an_unknown_run_exits_with_a_message():
    result, _ = invoke_tasks([], httpx.Response(404, json={"detail": "Taskset run not found"}))
    assert result.exit_code == 1 and result.output == f"Error: run default/{RUN_ID} not found\n"


DIGEST = "1f9c2a7e" + "0" * 56


def listed_run(run_id, phase="running", created_by="user:alice"):
    return {
        "run_id": run_id,
        "workspace": "default",
        "taskset_ref": f"default/debug-dataset#{DIGEST}",
        "phase": phase,
        "result_state": "none",
        "accepted_at": "2026-10-17T12:04:11.123456Z",
        "timeout_seconds": 3600,
        "execution_deadline_at": "2026-10-17T13:04:11Z",
        "finalization_deadline_at": "2026-10-17T14:04:11Z",
        "created_by": created_by,
    }


def run_page(runs, *, page=1, total_pages=1, total=None):
    pagination = {
        "page": page,
        "page_size": 20,
        "current_page_size": len(runs),
        "total_pages": total_pages,
        "total_results": len(runs) if total is None else total,
    }
    return {"data": runs, "pagination": pagination}


def invoke_list(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "list", "--workspace", "default", *args]), requests


def test_list_passes_its_filters_and_page_to_the_api():
    result, requests = invoke_list(
        ["--active", "--mine", "--page", "2", "--page-size", "5"], httpx.Response(200, json=run_page([]))
    )
    assert result.exit_code == 0
    (request,) = requests
    assert request.url.path == "/apis/evals/v2/workspaces/default/agent-evaluate/runs"
    assert dict(request.url.params) == {"active": "true", "mine": "true", "page": "2", "page_size": "5"}


def test_list_shows_one_row_per_run_and_points_to_the_next_page():
    runs = [listed_run("run-aaaa"), listed_run("run-bbbb", phase="completed", created_by=None)]
    result, _ = invoke_list([], httpx.Response(200, json=run_page(runs, total_pages=3, total=45)))
    lines = result.output.splitlines()
    assert lines[0].split() == ["run", "phase", "result", "taskset", "accepted", "by"]
    assert lines[1].split() == [
        "run-aaaa",
        "running",
        "none",
        "default/debug-dataset#1f9c2a7e",
        "2026-10-17",
        "12:04:11",
        "user:alice",
    ]
    assert lines[2].split()[-1] == "-"  # no creator recorded
    assert lines[3] == "page 1 of 3 (45 runs); next: --page 2"

    result, _ = invoke_list([], httpx.Response(200, json=run_page(runs)))
    assert result.output.splitlines()[-1] == "2 runs"


def test_list_says_so_when_there_are_no_runs():
    result, _ = invoke_list([], httpx.Response(200, json=run_page([])))
    assert result.exit_code == 0 and result.output == "No runs.\n"


def test_list_json_prints_the_api_page_unchanged():
    body = run_page([listed_run("run-aaaa")])
    result, _ = invoke_list(["--json"], httpx.Response(200, json=body))
    assert json.loads(result.output) == body


def test_list_failure_exits_with_the_api_detail():
    result, _ = invoke_list([], httpx.Response(500, json={"detail": "Internal server error"}))
    assert result.exit_code == 1
    assert "Error: HTTP 500: Internal server error" in result.output


def invoke_cancel(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "cancel", "run-aaaa", "--workspace", "ac9", *args]), requests


def test_cancel_posts_the_intent_and_says_how_to_follow_the_run():
    run = listed_run("run-aaaa") | {"cancel_requested_at": "2026-10-17T12:30:00Z"}
    result, requests = invoke_cancel([], httpx.Response(200, json=run))
    assert result.exit_code == 0
    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/apis/evals/v2/workspaces/ac9/agent-evaluate/runs/run-aaaa/cancel"
    assert result.output.splitlines() == [
        "Cancel requested for run-aaaa at 2026-10-17T12:30:00+00:00 (phase running).",
        "Follow it with: nemo evals runs status run-aaaa --workspace ac9",
    ]
    result, _ = invoke_cancel(["--json"], httpx.Response(200, json=run))
    assert json.loads(result.output) == run


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, json={"detail": "Taskset run not found"}), "run ac9/run-aaaa not found"),
        (
            httpx.Response(409, json={"detail": "run run-aaaa already finished: completed"}),
            "Error: HTTP 409: run run-aaaa already finished: completed",
        ),
    ],
)
def test_cancel_failure_exits_with_the_reason(response, message):
    result, _ = invoke_cancel([], response)
    assert result.exit_code == 1 and message in result.output


def bundle_archive(*members: tuple[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


GOOD_BUNDLE = bundle_archive(("agent-eval-results/report.html", b"<html>"), ("agent-eval-results/summary.json", b"{}"))


def downloading(tmp_path, archive, *, result_state="ready", download_status=200):
    """Invoke ``runs download run-x -o tmp_path/run-x`` against a run whose record names ``run-x-bundle-2``."""
    requests: list[str] = []
    run = RUN | {"run_id": "run-x", "phase": "completed", "result_state": result_state}
    record = result_record("run-x-bundle-2", []) | {"name": "run-x"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if path.endswith("/agent-evaluate/runs/run-x"):
            return httpx.Response(200, json=run)
        if path.endswith("/agent-eval-results/run-x"):
            return httpx.Response(200, json=record)
        if path.endswith("/results/agent-eval-results/download"):
            return httpx.Response(download_status, stream=httpx.ByteStream(archive))
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    args = ["runs", "download", "run-x", "--workspace", "default", "-o", str(tmp_path / "run-x")]
    return CliRunner().invoke(app, args), requests


def test_download_unpacks_the_bundle_of_the_job_the_record_names(tmp_path):
    result, requests = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 0, result.output
    assert "/apis/jobs/v2/workspaces/default/jobs/run-x-bundle-2/results/agent-eval-results/download" in requests
    bundle = tmp_path / "run-x" / "agent-eval-results"
    assert (bundle / "report.html").read_bytes() == b"<html>"
    assert f"{bundle / 'report.html'}" in result.output
    assert [p.name for p in (tmp_path / "run-x").iterdir()] == ["agent-eval-results"]  # no scratch left


@pytest.mark.parametrize("state", ["failed", "pending"])
def test_download_refuses_a_run_without_a_merged_result(tmp_path, state):
    result, requests = downloading(tmp_path, GOOD_BUNDLE, result_state=state)
    assert result.exit_code == 1
    assert f"has no merged result (phase completed, result {state})" in result.output
    assert not any(path.endswith("/download") for path in requests)


def test_download_never_overwrites_an_existing_bundle(tmp_path):
    existing = tmp_path / "run-x" / "agent-eval-results"
    existing.mkdir(parents=True)
    (existing / "notes.txt").write_text("mine")
    result, _ = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 1 and "already exists" in result.output
    assert [p.name for p in existing.iterdir()] == ["notes.txt"]


def test_an_escaping_archive_writes_nothing_and_a_retry_still_works(tmp_path):
    evil = bundle_archive(("agent-eval-results/report.html", b"<html>"), ("../evil", b"x"))
    result, _ = downloading(tmp_path, evil)
    assert result.exit_code == 1 and "unsafe tar member" in result.output
    assert not (tmp_path / "evil").exists()
    assert list((tmp_path / "run-x").iterdir()) == []

    result, _ = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(("archive", "status"), [(b"not found", 404), (b"not a tarball", 200)], ids=["404", "corrupt"])
def test_a_failed_download_is_an_error_not_a_traceback(tmp_path, archive, status):
    result, _ = downloading(tmp_path, archive, download_status=status)
    assert result.exit_code == 1 and result.output.startswith("Error: ")
    assert not (tmp_path / "run-x" / "agent-eval-results").exists()


@pytest.mark.parametrize("path", ["/dev/stdin", "/dev/fd/63", "/proc/self/fd/0"])
def test_a_spec_file_under_dev_or_proc_is_a_stream_even_when_redirected_from_a_file(path):
    assert runs_cli._is_stream(Path(path))  # `--spec-file /dev/stdin < spec.json`: is_file() is true


def test_a_regular_spec_file_is_not_a_stream(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    assert not runs_cli._is_stream(spec)


@pytest.mark.parametrize("path", ["/dev/shm/run.json", "/proc/self/cwd/spec.json"])
def test_a_regular_file_under_dev_or_proc_is_not_a_stream(monkeypatch, path):
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert not runs_cli._is_stream(Path(path))
