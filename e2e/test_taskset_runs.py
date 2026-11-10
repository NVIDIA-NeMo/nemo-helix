# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor taskset run on Kubernetes, end to end through ``nemo evals runs``.

Two submissions of the dataset in ``e2e/data/taskset_runs_dataset/``, recorded as ``test_1`` and ``test_2`` in the
evidence:

- test_1 submits it with ``--harbor-dir`` (Builder builds its images), waits for the run, and checks its task Jobs
  through the Jobs API, its scores, and the downloaded merged bundle.
- test_2 resubmits the taskset test_1 published, with ``--taskset``, and checks no image was rebuilt.

Every step is written to the evidence JSON as it happens; see ``RunDriver``.

The platform needs the taskset run controller, Builder with ``system/``-qualified registry and signing secrets,
and OpenSandbox. Run it with::

    uv run pytest e2e/test_taskset_runs.py --cluster-url http://127.0.0.1:18080 --feature taskset-runs -v

With auth on, add ``--access-token "$(nemo --context <context> auth token)"`` (or set ``NHX_E2E_ACCESS_TOKEN``): a
token that can create workspaces and outlives the run. Without one the test stops at once and prints the
``nemo auth login`` to run.

- ``NHX_E2E_WORKSPACE``: run in that workspace and keep it; warm, it reuses the ready images. Default: a new one.
- ``NHX_E2E_KUBE_CONTEXT``: the cluster's kubectl context. Adds the no-sandbox-left check, and saves sandbox
  diagnostics on failure; it needs cluster-wide read on ``batchsandboxes``. ``NHX_E2E_KUBE_NAMESPACE`` adds that
  namespace's events to them.
- ``E2E_SERVICES_LOG_DIR``: where the evidence goes; default ``.pytest_cache/taskset-runs-e2e/``.
"""

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from nemo_helix_ext.config.config import Config
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import AuthenticationError
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.testing import NemoRun

from e2e.taskset_runs_driver import (
    BUDGET,
    PYTEST_TIMEOUT,
    Evidence,
    RunDriver,
    check_bundle,
    check_cancelled_while_queued,
    check_foreign_job_cancel,
    check_run,
    check_scores,
    check_snapshot,
    check_snapshot_unchanged,
    check_task_jobs,
    cli_contexts,
    evidence_dir,
    login_hint,
)

pytestmark = [
    pytest.mark.feature("taskset-runs"),
    pytest.mark.platform("kubernetes"),
    pytest.mark.timeout(PYTEST_TIMEOUT),  # see RunDriver: it must never fire
]

#: Step caps, in seconds, within BUDGET.
TEST_1_SUBMIT_CAP = 900  # publish, and build 3 images in a fresh workspace
RUN_CAP = 900
TEST_2_CAP = 600  # test_2's submit and wait together
#: How long a cancel-paths ``slow`` task's oracle sleeps (its ``solve.sh``): its Job stays active while the test
#: cancels around it.
SLOW_SECONDS = 180

#: Two Harbor tasks: ``solved`` (shared verifier, oracle right) and ``unsolved`` (separate verifier, oracle wrong).
DATASET = Path(__file__).parent / "data" / "taskset_runs_dataset"
#: Cancel paths: ``slow-1`` and ``slow-2`` hold the creator's 2 slots; ``quick`` is the run that queues behind them.
CANCEL_DATASET = Path(__file__).parent / "data" / "taskset_runs_cancel_dataset"


@pytest.fixture
def budget() -> float:
    """The test's one end time. Requested first, so creating the workspace counts against it."""
    return time.monotonic() + BUDGET


@pytest.fixture
def authed(client: NemoClient) -> None:
    """Fails at once, with the login to run, when the cluster wants a token this run doesn't have."""
    try:
        WorkspacesClient.from_client(client).list_workspaces().page()  # raises on the status
        return
    except AuthenticationError:
        pass  # failed below, outside the handler, so the report shows only the hint
    pytest.fail(login_hint(client.base_url, cli_contexts(Config.get_default_config_path())), pytrace=False)


@pytest.fixture
def evidence(request: pytest.FixtureRequest, record_property: Callable[[str, object], None]) -> Evidence:
    evidence = Evidence(evidence_dir(Path(request.config.rootpath)))
    record_property("evidence", str(evidence.path))
    print(f"evidence: {evidence.path}")
    return evidence


@pytest.fixture
def driver(
    request: pytest.FixtureRequest,
    budget: float,
    authed: None,  # before `workspace`, whose create would fail on a bare 401
    nemo_run: NemoRun,
    client: NemoClient,
    workspace: str,
    evidence: Evidence,
) -> RunDriver:
    fresh = not (request.config.getoption("--workspace") or os.environ.get("NHX_E2E_WORKSPACE"))
    evidence.record(workspace=workspace, fresh_workspace=fresh)
    failed_before = request.session.testsfailed
    driver = RunDriver(
        nemo_run,
        client,
        workspace,
        evidence,
        end=budget,
        failed=lambda: request.session.testsfailed > failed_before,
        kube_context=os.environ.get("NHX_E2E_KUBE_CONTEXT"),
        kube_namespace=os.environ.get("NHX_E2E_KUBE_NAMESPACE"),
    )
    # Registered before any submit, and torn down before `workspace` deletes the workspace.
    request.addfinalizer(driver.cancel_open)
    return driver


@pytest.mark.taskset_runs_e2e
def test_taskset_run_on_kubernetes(
    driver: RunDriver,
    subtests: pytest.Subtests,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    fresh = driver.evidence.data["fresh_workspace"]
    # The nightly's report warns when a green step used more than half its cap.
    driver.evidence.record("test_1", caps_s={"submit": TEST_1_SUBMIT_CAP, "run": RUN_CAP})
    driver.evidence.record("test_2", caps_s={"submit": TEST_2_CAP, "run": TEST_2_CAP})
    run_id = driver.submit("test_1", "--harbor-dir", str(DATASET), "--agent", "oracle", cap=TEST_1_SUBMIT_CAP)
    record_property("run_id", run_id)
    taskset_ref = driver.evidence.data["test_1"].get("taskset_ref")
    assert taskset_ref, "[submit] `runs submit` printed no published taskset"
    before = driver.snapshot(taskset_ref)
    driver.evidence.record("test_1", run_id=run_id, image_builds=before)
    check_snapshot(before, fresh=fresh)

    status = driver.wait("test_1", run_id, cap=RUN_CAP)  # outside the subtests: every step needs a finished run
    task_jobs = {task.name: task.task.job if task.task else None for task in status.tasks}
    driver.evidence.record("test_1", task_jobs=task_jobs, bundle_job=status.bundle)
    record_property("task_jobs", task_jobs)
    check_run(status)

    with subtests.test("task jobs"):
        jobs, attempts = driver.task_jobs(status)
        driver.evidence.record("test_1", task_job_attempts=attempts)
        check_task_jobs(status, jobs, attempts)

    results = driver.results(run_id)
    with subtests.test("scores"):
        driver.evidence.record("test_1", rewards=check_scores(results))

    with subtests.test("bundle"):
        out = driver.cli("evals", "runs", "download", run_id, "-o", str(tmp_path / "bundle"), cap=120)
        assert out.returncode == 0, f"[bundle] `runs download` exited {out.returncode}: {out.stderr.strip()}"
        check_bundle(tmp_path / "bundle" / "agent-eval-results", results)

    with subtests.test("no sandbox left"):
        context = os.environ.get("NHX_E2E_KUBE_CONTEXT")
        if not context:
            pytest.skip("NHX_E2E_KUBE_CONTEXT is not set; the nightly sets it")
        left = subprocess.run(
            ["kubectl", "--context", context, "get", "batchsandboxes", "-A", "-l", f"nemo-run={run_id}", "-o", "name"],
            capture_output=True,
            text=True,
            timeout=driver.sec_left(30),
            check=False,
        )
        assert left.returncode == 0, f"[dispatch] kubectl failed: {left.stderr.strip()}"
        assert not left.stdout.strip(), f"[dispatch] sandboxes left behind: {left.stdout.split()}"

    with subtests.test("resubmit by taskset"):
        test_2_end = time.monotonic() + TEST_2_CAP
        driver.evidence.record("test_2", taskset_ref=taskset_ref)  # test_2's output never prints it
        test_2_run = driver.submit("test_2", "--taskset", taskset_ref, "--agent", "oracle", cap=TEST_2_CAP)
        test_2_status = driver.wait("test_2", test_2_run, cap=max(test_2_end - time.monotonic(), 1))
        check_run(test_2_status)
        test_2_results = driver.results(test_2_run)
        driver.evidence.record("test_2", rewards=check_scores(test_2_results))
        after = driver.snapshot(taskset_ref)
        check_snapshot_unchanged(before, after)
        driver.evidence.record("test_2", builds_unchanged=True)


def _created_ref(driver: RunDriver, dataset: Path, tasks: str) -> str:
    """Publish ``tasks`` of ``dataset`` as a taskset and build its images; return its ref."""
    out = driver.cli("evals", "tasksets", "create", "--harbor-dir", str(dataset), "--tasks", tasks, cap=120)
    assert out.returncode == 0, f"[build] `tasksets create` exited {out.returncode}: {out.stderr.strip()}"
    ref = out.stdout.strip().splitlines()[-1]  # without --json the last line is the ref
    built = driver.cli("evals", "tasksets", "build-images", ref, "--wait", cap=900)
    assert built.returncode == 0, f"[build] `build-images {ref}` exited {built.returncode}: {built.stdout[-2000:]}"
    return ref


@pytest.mark.taskset_runs_cancel_e2e
@pytest.mark.feature("taskset-runs", "taskset-runs-cancel")  # needs the per-creator cap of 2 (taskset-runs.yaml)
def test_taskset_run_cancel_paths(driver: RunDriver) -> None:
    """E1: a run cancelled while queued never creates a task Job. C3: a task Job cancelled from outside fails only
    its task, and the run completes over the rest.

    Run H's two slow tasks hold both of the creator's slots, so run Q waits. Each cancel waits for its precondition
    (Q still queued, H's Job active) rather than sleeping a fixed time.
    """
    hold_ref = _created_ref(driver, CANCEL_DATASET, "slow-1,slow-2")  # images first: building after H starts would let
    queue_ref = _created_ref(driver, CANCEL_DATASET, "quick")  # H's tasks finish before Q is even submitted

    hold = driver.submit("hold", "--taskset", hold_ref, "--agent", "oracle", cap=120)
    held = driver.wait_for(
        "cancel",
        f"both task Jobs of {hold} active",
        lambda: driver.status(hold),
        lambda status: [task.task.job_status if task.task else None for task in status.tasks] == ["active"] * 2,
        cap=600,
    )
    hold_jobs = [task.task.job for task in held.tasks if task.task and task.task.job]

    queued = driver.submit("e1", "--taskset", queue_ref, "--agent", "oracle", cap=120)
    driver.wait_for("cancel", f"{queued}'s task row", lambda: driver.status(queued), lambda s: bool(s.tasks), cap=60)
    driver.pause(10)  # two controller cycles (interval_seconds 5): a slot would have been given by now
    still = driver.status(queued)
    row = still.tasks[0].task
    assert row is not None and row.phase in {"image_preparing", "image_ready"} and not row.job, (
        f"[cancel] {queued} got a slot while {hold} held both: {row}"
    )
    statuses = {job: driver.job_status(job) for job in hold_jobs}  # None: the Jobs API didn't find it
    assert all(status == "active" for status in statuses.values()), (
        f"[cancel] H's Jobs ended before E1's cancel: {statuses}"
    )
    out = driver.cli("evals", "runs", "cancel", queued, "--json", cap=30)
    assert out.returncode == 0, f"[cancel] `runs cancel {queued}` exited {out.returncode}: {out.stderr.strip()}"
    cancelled = driver.wait("e1", queued, cap=120)
    jobs, _ = driver.task_jobs(cancelled)
    check_cancelled_while_queued(cancelled, jobs)

    assert driver.job_status(hold_jobs[0]) == "active", f"[cancel] {hold_jobs[0]} ended before C3's cancel"
    out = driver.cli("jobs", "cancel", hold_jobs[0], cap=30)
    assert out.returncode == 0, f"[cancel] `jobs cancel {hold_jobs[0]}` exited {out.returncode}: {out.stderr.strip()}"
    driver.evidence.record("c3", run_id=hold, cancelled_job=hold_jobs[0])
    driver.wait("hold", hold, cap=SLOW_SECONDS + 300)
    results = driver.results(hold, stage="cancel")
    check_foreign_job_cancel(results, hold_jobs[0])
