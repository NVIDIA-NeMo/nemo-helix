# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The taskset-run E2E's driver and checkers, without a cluster.

The driver's timeout, interrupt and cleanup paths run only after a failure, so a green cluster run never exercises
them; the checker tests prove each assertion can still fail.

The checker inputs are built by the factories below, as the CLI prints them, and parsed through the same boundary as
the e2e (``parse_status``, ``parse_results``, ``read_bundle``). A test can't build an input the platform's models
reject, and a model change breaks the factories on the PR that makes it. Their values follow a green run on a
deployed namespace (run-238e586d56bcfd4b, in AALGO-1034's evidence).
"""

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_helix_plugin.client import client as client_module
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.jobs.client import JobsClient
from nhx_evals_sdk.agent_eval.scores import AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial
from pydantic import BaseModel

from e2e.taskset_runs_driver import (
    BUNDLE_FILES,
    CLEANUP,
    EXPECTED_REWARDS,
    FORMAT_OK,
    REWARD,
    BuildView,
    Evidence,
    JobView,
    RunDriver,
    Snapshot,
    check_bundle,
    check_run,
    check_scores,
    check_snapshot,
    check_snapshot_unchanged,
    check_task_jobs,
    cli_contexts,
    login_hint,
    parse_results,
    parse_status,
    read_bundle,
)

RUN = "run-0123456789abcdef"
Respond = Callable[[tuple[str, ...], int | None], subprocess.CompletedProcess[str]]


class Clock:
    """A monotonic clock that only moves when a fake call or sleep moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeNemo:
    """The ``nemo_run`` fixture: records each call's args and timeout, and answers with ``respond``."""

    def __init__(self, respond: Respond) -> None:
        self.respond = respond
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[int | None] = []

    def __call__(
        self, *args: str, workspace: str | None = None, timeout: int | None = 60
    ) -> subprocess.CompletedProcess[str]:
        assert workspace == "ws"
        self.calls.append(args)
        self.timeouts.append(timeout)
        return self.respond(args, timeout)

    def commands(self) -> list[str]:
        """Each call as ``runs <verb>`` (``runs cancel run-…`` for a cancel)."""
        return [" ".join(call[1:3] + (call[3:4] if call[2] in {"cancel", "status"} else ())) for call in self.calls]


def done(stdout: Any = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    text = stdout if isinstance(stdout, str) else json.dumps(stdout)
    return subprocess.CompletedProcess(["nemo"], returncode, text, stderr)


def page(*runs: dict[str, str], pages: int = 1) -> dict[str, Any]:
    return {"data": list(runs), "pagination": {"page": 1, "total_pages": pages, "total_results": len(runs)}}


def make_driver(
    tmp_path: Path,
    respond: Respond,
    clock: Clock,
    *,
    time_left: float = 2400,
    client: NemoClient | None = None,
    **kwargs: Any,
) -> tuple[RunDriver, FakeNemo]:
    nemo = FakeNemo(respond)
    driver = RunDriver(
        nemo,
        client or NemoClient(base_url="http://platform.test"),
        "ws",
        Evidence(tmp_path / "evidence", invocation="inv"),
        end=clock() + time_left,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return driver, nemo


def evidence_file(driver: RunDriver) -> dict[str, Any]:
    return json.loads(driver.evidence.path.read_text())


# --- Auth --------------------------------------------------------------------------------------------------------

_CLI_CONFIG = """\
current_context: local
clusters:
  - {name: local-cluster, base_url: "http://127.0.0.1:18080"}
  - {name: aire-cluster, base_url: "https://nemo-helix.example.com/"}
users:
  - {name: me, type: oauth}
contexts:
  - {name: local, cluster: local-cluster, user: me}
  - {name: aire, cluster: aire-cluster, user: me}
  - {name: dangling, cluster: missing-cluster, user: me}
"""


def test_cli_contexts_maps_names_to_cluster_urls(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(_CLI_CONFIG, encoding="utf-8")
    assert cli_contexts(config) == {"local": "http://127.0.0.1:18080", "aire": "https://nemo-helix.example.com/"}


def test_cli_contexts_without_a_config_file(tmp_path: Path) -> None:
    assert cli_contexts(tmp_path / "missing.yaml") == {}


def test_login_hint_uses_the_users_context_for_the_cluster() -> None:
    hint = login_hint(
        "https://nemo-helix.example.com", {"local": "http://127.0.0.1:18080", "aire": "https://nemo-helix.example.com/"}
    )
    assert "nemo auth login --context aire --base-url https://nemo-helix.example.com\n" in hint
    assert '--access-token "$(nemo --context aire auth token)"' in hint


def test_login_hint_names_a_new_context_after_the_host() -> None:
    hint = login_hint("https://nemo-helix.example.com/", {"local": "http://127.0.0.1:18080"})
    assert "nemo auth login --context nemo-helix.example.com --base-url https://nemo-helix.example.com\n" in hint


# --- Driver ----------------------------------------------------------------------------------------------------


def test_submit_timeout_leaves_key_and_partial_output_in_evidence(tmp_path: Path) -> None:
    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            ["nemo"], timeout or 0, output=b"Published taskset ws/ts#sha256:abc (2 of 2 tasks)\n", stderr=b"slow\n"
        )

    driver, _ = make_driver(tmp_path, respond, Clock())
    with pytest.raises(subprocess.TimeoutExpired):
        driver.submit("test_1", "--harbor-dir", "d", cap=900)

    test_1 = evidence_file(driver)["test_1"]
    assert test_1["phase"] == "timed_out"
    assert test_1["key"].startswith("e2e-") and len(test_1["key"]) == 36
    assert test_1["workspace"] == "ws"
    assert test_1["taskset_ref"] == "ws/ts#sha256:abc"
    assert test_1["stderr"] == "slow\n"
    assert driver.unresolved == [test_1["key"]]  # no run ID printed: cancel_open must look for it


def test_submit_timeout_with_a_printed_run_id_resolves_the_key(tmp_path: Path) -> None:
    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["nemo"], 1, output=f"Run {RUN} accepted: 2 tasks, queued\n".encode())

    driver, _ = make_driver(tmp_path, respond, Clock())
    with pytest.raises(subprocess.TimeoutExpired):
        driver.submit("test_1", cap=900)

    assert evidence_file(driver)["test_1"]["run_id"] == RUN
    assert list(driver.run_ids.values()) == [RUN] and driver.unresolved == []


def test_submit_interrupted_leaves_key_unresolved_for_cancel_open(tmp_path: Path) -> None:
    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        raise KeyboardInterrupt

    driver, _ = make_driver(tmp_path, respond, Clock())
    with pytest.raises(KeyboardInterrupt):
        driver.submit("test_1", cap=900)

    test_1 = evidence_file(driver)["test_1"]
    assert test_1["phase"] == "submitting"
    assert driver.unresolved == [test_1["key"]]


def test_submit_returns_the_printed_run_id(tmp_path: Path) -> None:
    driver, nemo = make_driver(
        tmp_path, lambda args, timeout: done(f"Published taskset ws/ts#d (2 of 2 tasks)\nRun {RUN} accepted\n"), Clock()
    )

    assert driver.submit("test_1", "--agent", "oracle", cap=900) == RUN
    assert nemo.calls[0][:5] == ("evals", "runs", "submit", "--agent", "oracle")
    assert nemo.calls[0][-2] == "--idempotency-key"
    assert evidence_file(driver)["test_1"]["phase"] == "submitted"


def test_cancel_open_cancels_known_run_ids_directly(tmp_path: Path) -> None:
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done({"run_id": RUN}), Clock())
    driver.run_ids["k"] = RUN

    driver.cancel_open()

    assert nemo.commands() == [f"runs cancel {RUN}"]
    assert evidence_file(driver)["cleanup"]["cancelled"] == [RUN]


def test_cancel_open_skips_runs_seen_terminal(tmp_path: Path) -> None:
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done(), Clock())
    driver.run_ids["k"] = RUN
    driver.settled.add(RUN)

    driver.cancel_open()

    assert nemo.calls == []


def test_cancel_open_records_truncated_listing(tmp_path: Path) -> None:
    driver, _ = make_driver(tmp_path, lambda args, timeout: done(page(pages=3)), Clock())
    driver.unresolved.append("e2e-key")

    driver.cancel_open()

    cleanup = evidence_file(driver)["cleanup"]
    assert cleanup["truncated"] is True
    assert cleanup["unresolved"] == ["e2e-key"]


def test_cancel_open_finds_a_run_accepted_after_the_first_listing(tmp_path: Path) -> None:
    listings = iter([page(), page({"run_id": RUN, "idempotency_key": "e2e-key"})])

    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        return done(next(listings)) if args[2] == "list" else done({"run_id": RUN})

    driver, nemo = make_driver(tmp_path, respond, Clock())
    driver.unresolved.append("e2e-key")

    driver.cancel_open()

    assert nemo.commands() == ["runs list", "runs list", f"runs cancel {RUN}"]
    assert "--page-size" in nemo.calls[0] and "1000" in nemo.calls[0] and "--mine" in nemo.calls[0]
    cleanup = evidence_file(driver)["cleanup"]
    assert cleanup["cancelled"] == [RUN] and cleanup["unresolved"] == []


def test_cancel_open_leaves_a_foreign_active_run_alone(tmp_path: Path) -> None:
    clock = Clock()
    driver, nemo = make_driver(
        tmp_path, lambda args, timeout: done(page({"run_id": "run-other", "idempotency_key": "e2e-other"})), clock
    )
    driver.unresolved.append("e2e-key")
    started = clock()

    driver.cancel_open()

    assert all(call[2] == "list" for call in nemo.calls)  # polled, never cancelled
    assert clock() - started <= 30
    assert evidence_file(driver)["cleanup"]["unresolved"] == ["e2e-key"]


def test_cancel_open_survives_connection_errors(tmp_path: Path) -> None:
    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        if args[2] == "cancel":
            return done(returncode=1, stderr="Error: connection refused")
        raise ConnectionRefusedError("connection refused")

    driver, _ = make_driver(tmp_path, respond, Clock(), failed=lambda: True)
    driver.run_ids["k"] = RUN
    driver.unresolved.append("e2e-key")

    driver.cancel_open()  # never raises

    cleanup = evidence_file(driver)["cleanup"]
    assert cleanup["cancelled"] == []
    assert any("connection refused" in error for error in cleanup["errors"])
    assert cleanup["unresolved"] == ["e2e-key"]


def test_cancel_open_stops_at_its_cleanup_deadline(tmp_path: Path) -> None:
    clock = Clock()

    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        clock.sleep(timeout or 0)  # every call hangs until its timeout
        raise subprocess.TimeoutExpired(["nemo"], timeout or 0)

    driver, nemo = make_driver(tmp_path, respond, clock, time_left=0)  # the test's own budget is spent
    driver.run_ids.update({"a": "run-a", "b": "run-b", "c": "run-c", "d": "run-d", "e": "run-e"})
    driver.unresolved.append("e2e-key")
    started = clock()

    driver.cancel_open()

    assert clock() - started <= CLEANUP
    assert nemo.calls  # it had its own deadline, not the spent budget
    assert evidence_file(driver)["cleanup"]["unresolved"] == ["e2e-key"]


def test_cli_and_sdk_timeouts_are_clamped_to_the_time_left(tmp_path: Path) -> None:
    timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(404, json={"detail": "not found"})

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done("{}"), Clock(), time_left=10, client=client)

    driver.cli("evals", "runs", "list", cap=60)
    with pytest.raises(NemoClientError):
        JobsClient.from_client(driver.sdk(cap=30)).get_job(workspace="ws", name="j")

    assert nemo.timeouts == [10]
    assert timeouts and timeouts[0]["read"] == 10


def test_calls_fail_once_the_budget_is_spent(tmp_path: Path) -> None:
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done(), Clock(), time_left=0)

    with pytest.raises(AssertionError, match=r"\[budget\]"):
        driver.cli("evals", "runs", "list")
    assert nemo.calls == []


def _rate_limited_client() -> NemoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "60"}, json={"detail": "slow down"})

    return NemoClient(
        base_url="http://platform.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry=PLATFORM_DEFAULT_RETRY_POLICY,
    )


def test_sdk_ignores_retry_after_and_stays_within_time_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr(client_module, "sleep", delays.append)  # the client's own binding, not time.sleep
    client = _rate_limited_client()
    driver, _ = make_driver(tmp_path, lambda args, timeout: done(), Clock(), client=client)

    with pytest.raises(NemoClientError):
        JobsClient.from_client(driver.sdk()).get_job(workspace="ws", name="j")
    assert delays == [0.5, 1.0]  # the same 2 retries, without the 60 s waits

    delays.clear()
    with pytest.raises(NemoClientError):  # control: the e2e client's own policy waits 60 s twice
        JobsClient.from_client(client).get_job(workspace="ws", name="j")
    assert delays == [60.0, 60.0]


def test_evidence_record_merges_fields(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path, invocation="inv")

    evidence.record("test_1", key="k", durations_s={"submit": 5})
    evidence.record("test_1", run_id=RUN, durations_s={"run": 9})
    evidence.record(workspace="ws")

    assert json.loads(evidence.path.read_text()) == {
        "invocation": "inv",
        "workspace": "ws",
        "test_1": {"key": "k", "run_id": RUN, "durations_s": {"submit": 5, "run": 9}},
    }


def test_failure_captures_diagnostics_before_cancel(tmp_path: Path) -> None:
    order: list[str] = []

    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        order.append(f"nemo {args[2]}")
        return done({"run": {"run_id": RUN}})

    def kubectl(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        order.append(f"kubectl {args[2]}")
        assert args[:2] == ["--context", "kind-e2e"]
        return done("pods\n")

    driver, _ = make_driver(
        tmp_path, respond, Clock(), failed=lambda: True, kube_context="kind-e2e", kube_namespace="ns", kubectl=kubectl
    )
    driver.run_ids["k"] = RUN

    driver.cancel_open()

    assert order == ["nemo status", "kubectl get", "kubectl describe", "kubectl get", "nemo cancel"]
    assert (tmp_path / "evidence" / f"taskset-runs-e2e-inv-{RUN}-status.json").is_file()
    assert (tmp_path / "evidence" / f"taskset-runs-e2e-inv-{RUN}-sandboxes.yaml").is_file()
    assert evidence_file(driver)["cleanup"]["diagnostics"] == "complete"


def test_passing_test_skips_diagnostics(tmp_path: Path) -> None:
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done(), Clock(), kube_context="kind-e2e")
    driver.run_ids["k"] = RUN

    driver.cancel_open()

    assert nemo.commands() == [f"runs cancel {RUN}"]
    assert "diagnostics" not in evidence_file(driver)["cleanup"]


def test_slow_diagnostics_still_leave_time_to_cancel(tmp_path: Path) -> None:
    clock = Clock()

    def respond(args: tuple[str, ...], timeout: int | None) -> subprocess.CompletedProcess[str]:
        if args[2] == "status":  # hangs until its timeout
            clock.sleep(timeout or 0)
            raise subprocess.TimeoutExpired(["nemo"], timeout or 0)
        return done({"run_id": RUN})

    def kubectl(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        clock.sleep(timeout)
        raise subprocess.TimeoutExpired(["kubectl"], timeout)

    driver, nemo = make_driver(tmp_path, respond, clock, failed=lambda: True, kube_context="kind-e2e", kubectl=kubectl)
    driver.run_ids["k"] = RUN
    started = clock()

    driver.cancel_open()

    assert nemo.commands()[-1] == f"runs cancel {RUN}"
    assert clock() - started <= CLEANUP
    cleanup = evidence_file(driver)["cleanup"]
    assert cleanup["diagnostics"] == "partial" and cleanup["cancelled"] == [RUN]


def test_wait_fails_with_the_last_state_after_its_cap(tmp_path: Path) -> None:
    status = status_body(phase="running", result_state="pending", task_phases=("registered", "registered"))
    driver, nemo = make_driver(tmp_path, lambda args, timeout: done(status), Clock())

    with pytest.raises(AssertionError, match=r"\[dispatch\] run run-0123456789abcdef still running after 60 s"):
        driver.wait("test_1", RUN, cap=60)
    assert 1 < len(nemo.calls) <= 13


# --- Checker inputs ----------------------------------------------------------------------------------------------
# Built as the CLI prints them, then parsed through the e2e's own boundary. A mutation edits the dict and parses again:
# never ``model_copy(update=...)``, which skips validation.

TASKS = ("solved", "unsolved")
STAMP = "2026-11-09T08:31:56Z"


def status_body(
    *,
    phase: str = "completed",
    result_state: str = "ready",
    task_phases: tuple[str, str] = ("succeeded", "succeeded"),
) -> dict[str, Any]:
    """``runs status --json`` of a run of the two tasks."""
    return {
        "run": {
            "run_id": RUN,
            "workspace": "ws",
            "taskset_ref": f"ws/taskset-e2e#{'c' * 64}",
            "phase": phase,
            "result_state": result_state,
            "accepted_at": STAMP,
            "timeout_seconds": 86400,
            "execution_deadline_at": "2026-11-10T08:32:10Z",
            "finalization_deadline_at": "2026-11-10T09:02:10Z",
        },
        "bundle": {"job": f"{RUN}-bundle-1", "attempt": 1, "status": "completed", "error": None},
        "tasks": [
            {
                "index": index,
                "name": f"taskset-e2e-{name}",
                "task": {
                    "task": f"task-{index}",
                    "task_index": index,
                    "phase": task_phase,
                    "job": f"{RUN}-task-{index}",
                    "job_status": "completed",
                },
                "state": task_phase,
            }
            for index, (name, task_phase) in enumerate(zip(TASKS, task_phases, strict=True))
        ],
    }


def _record(job: str, *, reward: float, format_ok: float, count: int = 1) -> dict[str, Any]:
    """A result record with the two Harbor scores."""
    return {
        "id": f"agent-eval-result-{job}",
        "name": job,
        "workspace": "ws",
        "job_id": job,
        "scores": {
            "scores": [
                {"name": REWARD, "count": count, "nan_count": 0, "mean": reward},
                {"name": FORMAT_OK, "count": count, "nan_count": 0, "mean": format_ok},
            ]
        },
        "bundle_ref": f"ws/{job}",
        "created_at": STAMP,
        "updated_at": STAMP,
    }


def results_body(*, rewards: Mapping[str, float] = EXPECTED_REWARDS) -> dict[str, Any]:
    """``runs results --json``: each task's record and the merged one."""
    status = status_body()
    tasks = [
        {**task, "result": _record(task["task"]["job"], reward=rewards[name], format_ok=1.0)}
        for task, name in zip(status["tasks"], TASKS, strict=True)
    ]
    merged = _record(RUN, reward=sum(rewards.values()) / len(rewards), format_ok=1.0, count=len(rewards))
    return {"run": status["run"], "tasks": tasks, "result": merged}


def _score(record: dict[str, Any], name: str) -> dict[str, Any]:
    return next(score for score in record["scores"]["scores"] if score["name"] == name)


def trial_rows() -> list[dict[str, Any]]:
    """``trials.jsonl``: one completed trial per task."""
    return [
        {"id": f"trial-{name}", "task_id": f"taskset-e2e/{name}", "status": "completed", "output": {"output_text": ""}}
        for name in TASKS
    ]


def score_rows() -> list[dict[str, Any]]:
    """``scores.jsonl``: each trial's Harbor reward row."""
    return [
        {
            "id": f"score-{name}",
            "run_id": RUN,
            "task_id": f"taskset-e2e/{name}",
            "trial_id": f"trial-{name}",
            "metric_type": "harbor_reward",
            "status": "completed",
            "outputs": [{"name": "reward", "value": EXPECTED_REWARDS[name]}, {"name": "format_ok", "value": 1.0}],
        }
        for name in TASKS
    ]


def write_bundle(
    directory: Path, *, trials: list[dict[str, Any]] | None = None, scores: list[dict[str, Any]] | None = None
) -> Path:
    """A downloaded ``agent-eval-results/``; its trials and score rows validated through the models first."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        (directory / name).write_text("")
    _write_rows(directory / "trials.jsonl", [AgentEvalTrial.model_validate(row) for row in trials or trial_rows()])
    _write_rows(directory / "scores.jsonl", [AgentEvalTaskScore.model_validate(row) for row in scores or score_rows()])
    return directory


def _write_rows(path: Path, rows: Sequence[BaseModel]) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))


def green_jobs() -> tuple[list[JobView], dict[str, int]]:
    """The Jobs API's view of the run's task Jobs: both completed, on their first attempt."""
    names = [f"{RUN}-task-{index}" for index in range(len(TASKS))]
    return [JobView(name=name, status="completed") for name in names], dict.fromkeys(names, 1)


def green_builds(build_id: str = "1") -> Snapshot:
    """Each task's newest image build: ready, with an image."""
    return {
        f"taskset-e2e-{name}": BuildView(
            id=build_id,
            job=f"task-image-build-{name}-{build_id}",
            status="ready",
            error=None,
            images=(f"registry.test/ws/harbor-env-{name}@sha256:{'a' * 64}",),
        )
        for name in TASKS
    }


# --- Boundary ----------------------------------------------------------------------------------------------------


def test_parse_status_reads_the_cli_output_without_state() -> None:
    status = parse_status(status_body())

    assert [task.task.job for task in status.tasks if task.task] == [f"{RUN}-task-0", f"{RUN}-task-1"]
    assert status.bundle is not None and status.bundle.status == "completed"


def test_parse_results_reads_the_cli_output_without_state() -> None:
    results = parse_results(results_body())

    assert [task.name for task in results.tasks] == ["taskset-e2e-solved", "taskset-e2e-unsolved"]
    assert results.result is not None


@pytest.mark.parametrize("where", ["top level", "run", "task", "task row", "bundle"])
def test_parse_status_rejects_an_unknown_key(where: str) -> None:
    body = status_body()
    target = {
        "top level": body,
        "run": body["run"],
        "task": body["tasks"][0],
        "task row": body["tasks"][0]["task"],
        "bundle": body["bundle"],
    }[where]
    target["surprise"] = 1
    with pytest.raises(AssertionError, match=r"\[dispatch\] `runs status --json` doesn't match StatusView"):
        parse_status(body)


def test_parse_status_rejects_a_missing_required_field() -> None:
    body = status_body()
    del body["run"]["phase"]
    with pytest.raises(AssertionError, match=r"(?s)\[dispatch\] .*run.phase.*Field required"):
        parse_status(body)


def test_parse_results_rejects_an_unknown_key_in_a_task() -> None:
    body = results_body()
    body["tasks"][0]["surprise"] = 1
    with pytest.raises(AssertionError, match=r"\[verifier\] `runs results --json` doesn't match TasksetRunResults"):
        parse_results(body)


def test_parse_results_ignores_an_unknown_key_in_a_result_record() -> None:
    """Pinned: the result record ignores extra keys, and no checker reads them. A change to that model shows here."""
    body = results_body()
    body["tasks"][0]["result"]["surprise"] = 1
    parse_results(body)


@pytest.mark.parametrize("trial_id", [None, ""], ids=["missing", "empty"])
def test_read_bundle_rejects_a_trial_without_an_id(tmp_path: Path, trial_id: str | None) -> None:
    bundle = write_bundle(tmp_path)
    rows = trial_rows()
    if trial_id is None:
        del rows[0]["id"]
    else:
        rows[0]["id"] = trial_id
    (bundle / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))  # raw: the model refuses it
    with pytest.raises(AssertionError, match=r"\[bundle\] .* doesn't match the bundle models"):
        read_bundle(bundle)


def test_read_bundle_rejects_a_score_row_without_a_trial_id(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path)
    rows = score_rows()
    del rows[0]["trial_id"]
    (bundle / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError, match=r"\[bundle\] .* doesn't match the bundle models"):
        read_bundle(bundle)


def test_evidence_writes_models_and_views_as_json(tmp_path: Path) -> None:
    evidence = Evidence(tmp_path, invocation="inv")

    evidence.record("test_1", bundle_job=parse_status(status_body()).bundle, image_builds=green_builds())

    written = json.loads(evidence.path.read_text())["test_1"]
    assert written["bundle_job"] == {"job": f"{RUN}-bundle-1", "attempt": 1, "status": "completed"}
    assert written["image_builds"]["taskset-e2e-solved"]["id"] == "1"


# --- Checkers ----------------------------------------------------------------------------------------------------


def test_check_run_accepts_a_green_run() -> None:
    check_run(parse_status(status_body()))


def test_check_run_rejects_a_failed_task() -> None:
    body = status_body()
    body["tasks"][1]["task"].update(phase="failed", error="job_failed")
    with pytest.raises(AssertionError, match=r"\[dispatch\].*failed"):
        check_run(parse_status(body))


def test_check_task_jobs_accepts_a_green_run() -> None:
    jobs, attempts = green_jobs()
    check_task_jobs(parse_status(status_body()), jobs, attempts)


def test_check_task_jobs_rejects_a_second_attempt() -> None:
    jobs, attempts = green_jobs()
    with pytest.raises(AssertionError, match=r"\[dispatch\].*2 attempts"):
        check_task_jobs(parse_status(status_body()), jobs, {**attempts, f"{RUN}-task-1": 2})


def test_check_task_jobs_rejects_disagreeing_apis() -> None:
    jobs, attempts = green_jobs()
    jobs = [jobs[0], JobView(name="run-other-task-9", status="completed")]
    with pytest.raises(AssertionError, match=r"\[dispatch\].*aren't the run's"):
        check_task_jobs(parse_status(status_body()), jobs, attempts)


def test_check_scores_accepts_a_green_run() -> None:
    assert check_scores(parse_results(results_body())) == {"solved": 1.0, "unsolved": 0.0}


def test_check_scores_rejects_flipped_reward() -> None:
    with pytest.raises(AssertionError, match=r"\[verifier\] unsolved: reward 1.0"):
        check_scores(parse_results(results_body(rewards={"solved": 1.0, "unsolved": 1.0})))


def test_check_scores_rejects_format_ok_zero() -> None:
    body = results_body()
    _score(body["tasks"][1]["result"], FORMAT_OK)["mean"] = 0.0
    with pytest.raises(AssertionError, match=r"\[verifier\] unsolved: format_ok 0.0"):
        check_scores(parse_results(body))


def test_check_scores_rejects_a_wrong_merged_mean() -> None:
    body = results_body()
    _score(body["result"], REWARD)["count"] = 1
    with pytest.raises(AssertionError, match=r"\[bundle\] merged reward"):
        check_scores(parse_results(body))


def test_check_bundle_accepts_a_green_bundle(tmp_path: Path) -> None:
    check_bundle(write_bundle(tmp_path), parse_results(results_body()))


def test_check_bundle_rejects_missing_scores_file(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path)
    (bundle / "scores.jsonl").unlink()
    with pytest.raises(AssertionError, match=r"\[bundle\].*scores.jsonl"):
        check_bundle(bundle, parse_results(results_body()))


def test_check_bundle_rejects_a_third_trial(tmp_path: Path) -> None:
    trials = [*trial_rows(), {"id": "t3", "task_id": "x", "status": "completed", "output": {}}]
    with pytest.raises(AssertionError, match=r"\[bundle\] expected 2 trials, got 3"):
        check_bundle(write_bundle(tmp_path, trials=trials), parse_results(results_body()))


def test_check_bundle_rejects_scores_that_differ_from_results(tmp_path: Path) -> None:
    body = results_body()
    _score(body["tasks"][1]["result"], REWARD)["mean"] = 0.5
    with pytest.raises(AssertionError, match=r"\[bundle\] scores.jsonl's \(reward, format_ok\) by task"):
        check_bundle(write_bundle(tmp_path), parse_results(body))


def test_check_bundle_rejects_rewards_swapped_between_tasks(tmp_path: Path) -> None:
    scores = score_rows()
    scores[0]["outputs"], scores[1]["outputs"] = scores[1]["outputs"], scores[0]["outputs"]
    with pytest.raises(AssertionError, match=r"\[bundle\] scores.jsonl's \(reward, format_ok\) by task"):
        check_bundle(write_bundle(tmp_path, scores=scores), parse_results(results_body()))


def test_check_bundle_rejects_a_score_row_of_another_trial(tmp_path: Path) -> None:
    scores = score_rows()
    scores[0]["trial_id"] = scores[1]["trial_id"]  # right task and outputs, wrong trial
    with pytest.raises(AssertionError, match=r"\[bundle\] score row .* which trials.jsonl doesn't have"):
        check_bundle(write_bundle(tmp_path, scores=scores), parse_results(results_body()))


def test_check_snapshot_accepts_green_builds() -> None:
    check_snapshot(green_builds(), fresh=True)
    check_snapshot_unchanged(green_builds(), green_builds())


def test_check_snapshot_rejects_a_reused_build_in_a_fresh_workspace() -> None:
    builds = green_builds(build_id="2")
    check_snapshot(builds, fresh=False)
    with pytest.raises(AssertionError, match=r"\[build\].*attempt 2"):
        check_snapshot(builds, fresh=True)


def test_check_snapshot_rejects_new_build_id() -> None:
    after = green_builds()
    after["taskset-e2e-unsolved"] = replace(after["taskset-e2e-unsolved"], id="2")
    with pytest.raises(AssertionError, match=r"\[reuse\] taskset-e2e-unsolved was rebuilt"):
        check_snapshot_unchanged(green_builds(), after)


def test_checkers_reject_empty_inputs(tmp_path: Path) -> None:
    status = status_body()
    status["tasks"] = []
    with pytest.raises(AssertionError, match=r"\[dispatch\]"):
        check_run(parse_status(status))
    with pytest.raises(AssertionError, match=r"\[dispatch\]"):
        check_task_jobs(parse_status(status), [], {})
    results = results_body()
    results["tasks"] = []
    with pytest.raises(AssertionError, match=r"\[verifier\]"):
        check_scores(parse_results(results))
    bundle = write_bundle(tmp_path)
    for name in ("trials.jsonl", "scores.jsonl"):
        (bundle / name).write_text("")
    with pytest.raises(AssertionError, match=r"\[bundle\] expected 2 trials, got 0"):
        check_bundle(bundle, parse_results(results))
    with pytest.raises(AssertionError, match=r"\[build\]"):
        check_snapshot({}, fresh=False)
    with pytest.raises(AssertionError, match=r"\[reuse\]"):
        check_snapshot_unchanged({}, {})
