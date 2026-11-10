# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lifecycle and the assertions of the taskset-run E2E (``e2e/test_taskset_runs.py``).

``RunDriver`` owns what the test must never get wrong after a failure: the idempotency keys, the evidence file,
the one time budget and the cancel of every run it may have started. The ``check_*`` functions are the test's
assertions, as pure functions of what it read from the cluster, so ``tests/unit`` can prove each one can fail.

Everything read from the cluster is parsed into the platform's own models where it's read (``parse_status``,
``parse_results``, ``read_bundle``), so the checkers never see a dict: a field the test and the server disagree on
fails at the boundary, and ``ty`` checks every field a checker reads.

This module must not import ``e2e.conftest``: the unit tests load it without the e2e harness.
"""

import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from nemo_evals.api.schemas import AgentEvalResult
from nemo_evals.api.taskset_run_schemas import TasksetRunStatus
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.sdk.taskset_run_resources import NamedTask, TasksetRunResults
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.jobs.client import JobsClient
from nhx.testing import NemoRun
from nhx_evals_sdk.agent_eval.persistence import read_trials
from nhx_evals_sdk.agent_eval.scores import AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from pydantic import BaseModel, ConfigDict, ValidationError

#: Seconds. The whole test, setup included; the finalizer; one SDK overrun (92 s) plus the workspace delete (300 s).
BUDGET, CLEANUP, MARGIN = 2400, 120, 420
#: The pytest timeout: it must never fire, since ``timeout_method = thread`` skips every finalizer.
PYTEST_TIMEOUT = BUDGET + CLEANUP + MARGIN

TERMINAL_PHASES = frozenset({"completed", "failed", "cancelled", "expired"})
BUNDLE_FILES = (
    "report.html",
    "summary.json",
    "tasks.jsonl",
    "trials.jsonl",
    "scores.jsonl",
    "metadata.json",
    "run.json",
)
#: The dataset's tasks and what oracle scores on each: ``solved`` writes the right answer, ``unsolved`` a wrong one.
EXPECTED_REWARDS = {"solved": 1.0, "unsolved": 0.0}
REWARD = "harbor_reward.reward"
FORMAT_OK = "harbor_reward.format_ok"

_RUN_ID = re.compile(r"\bRun (run-[0-9a-f]+)\b")
_TASKSET = re.compile(r"^Published taskset (\S+)", re.MULTILINE)
_UNRESOLVED_POLL_SECONDS = 30.0
_UNRESOLVED_POLL_INTERVAL = 5.0
_CLEANUP_RESERVE = 60.0  # of CLEANUP, kept for the cancels however slow the diagnostics are


Kubectl = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


class BudgetExhausted(AssertionError):
    """The test's time budget ran out before a call could start."""


def _kubectl(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=timeout, check=False)


def _value(value: object) -> str:
    """An enum's value, or the text of anything else."""
    return str(getattr(value, "value", value))


def _jsonable(value: object) -> object:
    """How the evidence writes a model or a view: as the JSON the CLI would print."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


# --- Boundary --------------------------------------------------------------------------------------------------
# What the test reads, parsed into the platform's models (or, for shapes the driver builds, frozen views). A
# mismatch fails as an AssertionError tagged with the step's stage, like every other failure.


class BundleJob(BaseModel):
    """``runs status --json``'s ``bundle``: the CLI's ``BundleJobView`` dataclass, validated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job: str
    attempt: int
    status: str
    error: str | None = None


class StatusView(BaseModel):
    """``runs status --json``, without each task's ``state``. Unknown top-level keys fail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: TasksetRunStatus
    tasks: list[NamedTask]
    bundle: BundleJob | None = None


@dataclass(frozen=True)
class JobView:
    """A task Job as the Jobs API lists it."""

    name: str
    status: str


@dataclass(frozen=True)
class BuildView:
    """A task's newest image build."""

    id: str
    job: str | None
    status: str
    error: str | None
    images: tuple[str, ...]


#: Each task's newest image build, by task name.
Snapshot = dict[str, BuildView]


def _validated[M: BaseModel](stage: str, model: type[M], body: object, source: str) -> M:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise AssertionError(f"[{stage}] {source} doesn't match {model.__name__}: {exc}") from exc


def _without_state(body: Mapping[str, Any]) -> dict[str, Any]:
    """The CLI adds a ``state`` cell to each task; it's the one key the models don't have."""
    tasks = body.get("tasks")
    if not isinstance(tasks, list):
        return dict(body)
    return {**body, "tasks": [{k: v for k, v in task.items() if k != "state"} for task in tasks]}


def parse_status(body: Mapping[str, Any]) -> StatusView:
    """``runs status --json``."""
    return _validated("dispatch", StatusView, _without_state(body), "`runs status --json`")


def parse_results(body: Mapping[str, Any], stage: str = "verifier") -> TasksetRunResults:
    """``runs results --json``; a schema mismatch fails with ``[stage]``."""
    return _validated(stage, TasksetRunResults, _without_state(body), "`runs results --json`")


def read_bundle(bundle_dir: Path) -> tuple[list[AgentEvalTrial], list[AgentEvalTaskScore]]:
    """A downloaded bundle's trials (through the SDK's own reader) and score rows."""
    try:
        trials = read_trials(bundle_dir)
        scores = [AgentEvalTaskScore.model_validate(row) for row in _jsonl(bundle_dir / "scores.jsonl")]
    except ValidationError as exc:
        raise AssertionError(f"[bundle] {bundle_dir} doesn't match the bundle models: {exc}") from exc
    return trials, scores


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text(value: str | bytes | None) -> str:
    """``TimeoutExpired`` carries bytes even for a ``text=True`` run."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def parse_submit(stdout: str | bytes | None, stderr: str | bytes | None) -> dict[str, str]:
    """``runs submit``'s output, and the run ID and taskset ref when it printed them."""
    out, err = _text(stdout), _text(stderr)
    parsed = {"stdout": out, "stderr": err}
    if run := _RUN_ID.search(out):
        parsed["run_id"] = run.group(1)
    if taskset := _TASKSET.search(out):
        parsed["taskset_ref"] = taskset.group(1)
    return parsed


def evidence_dir(repo_root: Path) -> Path:
    """``$E2E_SERVICES_LOG_DIR`` when set, as in CI (its finalize step uploads it); else ``.pytest_cache``."""
    configured = os.environ.get("E2E_SERVICES_LOG_DIR")
    return Path(configured) if configured else repo_root / ".pytest_cache" / "taskset-runs-e2e"


def cli_contexts(config_file: Path) -> dict[str, str]:
    """The ``nemo`` CLI's contexts in ``config_file``, as name to base URL; empty when there is no file."""
    if not config_file.is_file():
        return {}
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    clusters = {cluster.get("name"): cluster.get("base_url") for cluster in config.get("clusters") or []}
    return {
        context["name"]: str(clusters[context.get("cluster")])
        for context in config.get("contexts") or []
        if clusters.get(context.get("cluster"))
    }


def login_hint(cluster_url: str, contexts: Mapping[str, str]) -> str:
    """What to run when ``cluster_url`` wants a token: log in with the user's own context for it, else one named
    after its host, then pass that context's token."""
    url = cluster_url.rstrip("/")
    context = next((name for name, base in contexts.items() if base.rstrip("/") == url), urlsplit(url).hostname)
    return (
        f"[auth] {url} answered 401: it needs a token, and this run has none or an expired one. Log in, then rerun "
        f"with its token:\n"
        f"  nemo auth login --context {context} --base-url {url}\n"
        f'  uv run pytest ... --access-token "$(nemo --context {context} auth token)"'
    )


class Evidence:
    """The evidence JSON: one section per step, rewritten after every write so a killed test still leaves it."""

    def __init__(self, directory: Path, invocation: str | None = None) -> None:
        self.invocation = invocation or uuid.uuid4().hex
        self.directory = directory
        self.path = directory / f"taskset-runs-e2e-{self.invocation}.json"
        self.data: dict[str, Any] = {"invocation": self.invocation}

    def record(self, step: str | None = None, **fields: Any) -> None:
        """Merge ``fields`` into ``step``'s section (top level when None) and rewrite the file.

        Dict values merge one level deep, so ``durations_s`` collects every step's timing.
        """
        section = self.data if step is None else self.data.setdefault(step, {})
        for key, value in fields.items():
            if isinstance(value, Mapping) and isinstance(section.get(key), dict):
                section[key] = {**section[key], **value}
            else:
                section[key] = value
        self.directory.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_suffix(".tmp")
        scratch.write_text(json.dumps(self.data, indent=2, default=_jsonable), encoding="utf-8")
        scratch.replace(self.path)

    def save(self, name: str, text: str) -> Path:
        """A diagnostic file next to the evidence, named after this invocation."""
        path = self.directory / f"taskset-runs-e2e-{self.invocation}-{name}"
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class RunDriver:
    """Submits, waits and cleans up within one budget; every call gets the smaller of its cap and the time left."""

    def __init__(
        self,
        run: NemoRun,
        client: NemoClient,
        workspace: str,
        evidence: Evidence,
        end: float,
        *,
        failed: Callable[[], bool] = lambda: False,
        kube_context: str | None = None,
        kube_namespace: str | None = None,
        kubectl: Kubectl = _kubectl,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._run = run  # the ``nemo_run`` fixture: ``nemo -f json <args>`` with NHX_WORKSPACE=workspace
        self.client = client
        self.workspace = workspace
        self.evidence = evidence
        self.end = end
        self._failed = failed
        self._kube_context = kube_context
        self._kube_namespace = kube_namespace
        self._kubectl = kubectl
        self._clock = clock
        self._sleep = sleep
        self.run_ids: dict[str, str] = {}  # key -> run ID, once submit printed one
        self.unresolved: list[str] = []  # keys submitted without a known run ID yet
        self.settled: set[str] = set()  # run IDs seen in a terminal phase

    # --- Calls -----------------------------------------------------------------------------------------------

    def sec_left(self, cap: float, end: float | None = None) -> float:
        """Seconds for the next call: ``cap``, or less when the budget ends sooner."""
        left = min(cap, (self.end if end is None else end) - self._clock())
        if left <= 0:
            raise BudgetExhausted(f"[budget] no time left for the next call (budget {BUDGET} s)")
        return left

    def cli(self, *args: str, cap: float = 60, end: float | None = None) -> subprocess.CompletedProcess[str]:
        """``nemo <args>`` in the test's workspace; raises ``TimeoutExpired`` when it outlives its time."""
        return self._run(*args, workspace=self.workspace, timeout=max(1, int(self.sec_left(cap, end))))

    def cli_json(self, stage: str, *args: str, cap: float = 60) -> Any:
        """``nemo <args>``'s JSON; a non-zero exit fails with ``stage`` and its stderr."""
        out = self.cli(*args, cap=cap)
        assert out.returncode == 0, f"[{stage}] `nemo {' '.join(args)}` exited {out.returncode}: {out.stderr.strip()}"
        return json.loads(out.stdout)

    def sdk(self, cap: float = 30) -> NemoClient:
        """The client, timed out by the budget, retrying as usual but ignoring ``Retry-After``.

        Honouring a 60 s ``Retry-After`` twice would let one call outlive the budget by minutes; this keeps the
        same two retries with 0.5 s and 1 s backoffs, so one call overruns by at most about 92 s.
        """
        retry = replace(self.client.retry or PLATFORM_DEFAULT_RETRY_POLICY, respect_retry_after_headers=False)
        return self.client.with_options(timeout=self.sec_left(cap), retry=retry)

    # --- Steps -----------------------------------------------------------------------------------------------

    def submit(self, step: str, *args: str, cap: float) -> str:
        """``runs submit <args>`` under a new idempotency key; return the run ID it printed.

        The key is known, and written down, before the CLI starts: a timeout or an interrupt at any point
        leaves it for ``cancel_open`` to find the run by.
        """
        key = f"e2e-{uuid.uuid4().hex}"  # full uuid: keys are scoped to (workspace, caller)
        self.unresolved.append(key)
        self.evidence.record(step, workspace=self.workspace, key=key, phase="submitting")
        started = self._clock()
        try:
            out = self.cli("evals", "runs", "submit", *args, "--idempotency-key", key, cap=cap)
        except subprocess.TimeoutExpired as exc:  # the run may already be accepted
            parsed = parse_submit(exc.stdout, exc.stderr)
            self._resolve(key, parsed.get("run_id"))
            self.evidence.record(step, phase="timed_out", **parsed)
            raise
        parsed = parse_submit(out.stdout, out.stderr)
        self._resolve(key, parsed.get("run_id"))
        self.evidence.record(
            step,
            phase="submitted" if out.returncode == 0 else "submit_failed",
            durations_s={"submit": round(self._clock() - started)},
            **parsed,
        )
        assert out.returncode == 0 and parsed.get("run_id"), (
            f"[submit] `runs submit` exited {out.returncode} without a run ID: {out.stderr.strip()}"
        )
        return parsed["run_id"]

    def _resolve(self, key: str, run_id: str | None) -> None:
        if run_id is not None and key in self.unresolved:
            self.unresolved.remove(key)
            self.run_ids[key] = run_id

    def status(self, run_id: str, cap: float = 30) -> StatusView:
        """``runs status <run> --json``, parsed."""
        return parse_status(self.cli_json("dispatch", "evals", "runs", "status", run_id, "--json", cap=cap))

    def results(self, run_id: str, cap: float = 60, *, stage: str = "verifier") -> TasksetRunResults:
        """``runs results <run> --json``, parsed; a failed read fails with ``[stage]``."""
        return parse_results(self.cli_json(stage, "evals", "runs", "results", run_id, "--json", cap=cap), stage)

    def wait(self, step: str, run_id: str, *, cap: float, poll: float = 5.0) -> StatusView:
        """Poll ``runs status --json`` until the run is terminal; fail with its last state after ``cap``."""
        started = self._clock()
        deadline = min(started + cap, self.end)
        while True:
            status = self.status(run_id, cap=min(30, max(deadline - self._clock(), 1)))
            phase = _value(status.run.phase)
            if phase in TERMINAL_PHASES:
                self.settled.add(run_id)
                self.evidence.record(
                    step,
                    phase=phase,
                    result_state=_value(status.run.result_state),
                    durations_s={"run": round(self._clock() - started)},
                )
                return status
            if self._clock() + poll >= deadline:
                states = {task.name: _value(task.task.phase) if task.task else "waiting" for task in status.tasks}
                raise AssertionError(f"[dispatch] run {run_id} still {phase} after {cap:.0f} s; tasks: {states}")
            self._sleep(poll)

    def wait_for[T](
        self, stage: str, what: str, read: Callable[[], T], done: Callable[[T], bool], *, cap: float, poll: float = 2.0
    ) -> T:
        """Poll ``read()`` until ``done`` holds; fail with ``[stage]`` and the last value after ``cap``.

        For a precondition, such as a Job being active before the test cancels it, instead of a fixed sleep.
        """
        deadline = min(self._clock() + cap, self.end)
        while True:
            value = read()
            if done(value):
                return value
            if self._clock() + poll >= deadline:
                raise AssertionError(f"[{stage}] {what} didn't happen within {cap:.0f} s; last: {value}")
            self._sleep(poll)

    def pause(self, seconds: float) -> None:
        """Wait ``seconds``, to show something does *not* happen in that time; fails if the budget can't cover it."""
        if self.end - self._clock() < seconds:
            raise BudgetExhausted(f"[budget] no time left to wait {seconds:.0f} s")
        self._sleep(seconds)

    def job_status(self, name: str) -> str | None:
        """A Job's status from the Jobs API, or None when there's no such Job."""
        try:
            return _value(JobsClient.from_client(self.sdk()).get_job(workspace=self.workspace, name=name).data().status)
        except NotFoundError:
            return None

    def task_jobs(self, status: StatusView) -> tuple[list[JobView], dict[str, int]]:
        """The run's task Jobs as the Jobs API lists them, and each named Job's attempt count.

        The list filters by name on the server, so a large retained workspace isn't scanned.
        """
        run_id = status.run.run_id
        jobs_client = JobsClient.from_client(self.sdk())
        page = jobs_client.list_jobs(
            workspace=self.workspace,
            query_params={"filter": {"name": {"$like": f"{run_id}-task-%"}}, "page_size": 100},
        ).page()
        jobs = [JobView(name=job.name, status=_value(job.status)) for job in page.items]
        attempts: dict[str, int] = {}
        for task in status.tasks:
            name = task.task.job if task.task else None
            if name:
                found = JobsClient.from_client(self.sdk()).list_job_attempts(workspace=self.workspace, job=name)
                attempts[name] = len(found.page().items)
        return jobs, attempts

    def snapshot(self, taskset_ref: str) -> Snapshot:
        """Each task's newest image build, read through the public SDK in the taskset's workspace.

        ``build-images --status`` can't do this: it prints counts and skips ready builds.
        """
        evaluator = Evaluator.from_client(self.sdk())
        head, _, digest = taskset_ref.partition("#")
        workspace, _, name = head.partition("/")
        taskset = evaluator.tasksets.retrieve(name, revision=digest, workspace=workspace)
        snapshot: Snapshot = {}
        for member in taskset.tasks:
            task_head, _, revision = member.root.partition("#")
            task_workspace, _, task = task_head.partition("/")
            builds = Evaluator.from_client(self.sdk()).tasks.list_image_builds(
                task, revision=revision, page_size=1, workspace=task_workspace
            )
            for build in builds.data:  # newest first; page_size=1 keeps only the newest
                snapshot[task] = BuildView(
                    id=build.id,
                    job=build.job,
                    status=_value(build.status),
                    error=build.error,
                    images=tuple(sorted(image.image_ref or "" for image in build.images)),
                )
        return snapshot

    # --- Cleanup ---------------------------------------------------------------------------------------------

    def cancel_open(self) -> None:
        """The finalizer: save diagnostics on failure, then cancel every run this test may have started.

        It never runs past ``CLEANUP`` and never raises, so a failed cleanup can't hide the test's own failure.
        """
        cleanup_end = self._clock() + CLEANUP
        cleanup: dict[str, Any] = {"cancelled": [], "unresolved": [], "truncated": False, "errors": []}
        try:
            if self._failed():
                cleanup["diagnostics"] = self._diagnose(cleanup_end, cleanup["errors"])
            for run_id in self.run_ids.values():
                if run_id not in self.settled and self._cancel(run_id, cleanup_end, cleanup["errors"]):
                    cleanup["cancelled"].append(run_id)
            if self.unresolved:
                self._cancel_unresolved(cleanup_end, cleanup)
        except Exception as exc:  # noqa: BLE001 - cleanup must never mask the test's own failure
            cleanup["errors"].append(f"{type(exc).__name__}: {exc}")
        cleanup["unresolved"] = list(self.unresolved)
        try:
            self.evidence.record("cleanup", **cleanup)
        except OSError:
            pass

    def _cancel(self, run_id: str, cleanup_end: float, errors: list[str]) -> bool:
        try:
            out = self.cli("evals", "runs", "cancel", run_id, "--json", cap=30, end=cleanup_end)
        except (subprocess.TimeoutExpired, BudgetExhausted, OSError) as exc:
            errors.append(f"cancel {run_id}: {type(exc).__name__}: {exc}")
            return False
        if out.returncode != 0:
            errors.append(f"cancel {run_id}: exit {out.returncode}: {out.stderr.strip()}")
            return False
        return True

    def _cancel_unresolved(self, cleanup_end: float, cleanup: dict[str, Any]) -> None:
        """Find the runs of keys submitted without a known run ID, and cancel them; never resubmit.

        Polls for 30 s at most: a killed CLI can leave the submit in flight, so the run may be accepted after the
        first listing, but a submit that timed out while waiting for builds never sent the request at all.
        """
        poll_end = min(self._clock() + _UNRESOLVED_POLL_SECONDS, cleanup_end)
        while self.unresolved:
            try:
                out = self.cli(
                    "evals", "runs", "list", "--mine", "--active", "--json", "--page-size", "1000",
                    cap=15, end=poll_end,
                )  # fmt: skip
                if out.returncode != 0:
                    cleanup["errors"].append(f"list: exit {out.returncode}: {out.stderr.strip()}")
                else:
                    page = json.loads(out.stdout)
                    pagination = page.get("pagination") or {}
                    if pagination.get("total_pages", 1) > pagination.get("page", 1):
                        cleanup["truncated"] = True
                    for run in page.get("data", []):
                        key = run.get("idempotency_key")
                        # Resolved only once cancelled: a failed cancel keeps the key, so the next poll retries it.
                        if key in self.unresolved and self._cancel(run["run_id"], cleanup_end, cleanup["errors"]):
                            self._resolve(key, run["run_id"])
                            cleanup["cancelled"].append(run["run_id"])
            except (subprocess.TimeoutExpired, BudgetExhausted, OSError, ValueError) as exc:
                cleanup["errors"].append(f"list: {type(exc).__name__}: {exc}")
            if not self.unresolved or self._clock() + _UNRESOLVED_POLL_INTERVAL >= poll_end:
                return
            self._sleep(_UNRESOLVED_POLL_INTERVAL)

    def _diagnose(self, cleanup_end: float, errors: list[str]) -> str:
        """Save each run's status and, with a kube context, its sandboxes and pods, before the cancel deletes them.

        Gets ``min(40, time left - 60)`` s in all, so at least 60 s of ``CLEANUP`` stays for the cancels.
        """
        end = min(self._clock() + 40, cleanup_end - _CLEANUP_RESERVE)
        complete = True
        for run_id in self.run_ids.values():
            try:
                out = self.cli("evals", "runs", "status", run_id, "--json", cap=15, end=end)
                self.evidence.save(f"{run_id}-status.json", out.stdout or out.stderr)
            except (subprocess.TimeoutExpired, BudgetExhausted, OSError) as exc:
                errors.append(f"diagnostics {run_id}: {type(exc).__name__}: {exc}")
                complete = False
            if self._kube_context is None:
                continue
            selector = f"nemo-run={run_id}"
            commands = {
                "sandboxes.yaml": ["get", "batchsandboxes,pods", "-A", "-l", selector, "-o", "yaml"],
                "describe.txt": ["describe", "batchsandboxes,pods", "-A", "-l", selector],
            }
            if self._kube_namespace:
                commands["events.txt"] = ["get", "events", "-n", self._kube_namespace, "--sort-by=.lastTimestamp"]
            for name, args in commands.items():
                left = end - self._clock()
                if left <= 0:
                    complete = False
                    break
                try:
                    out = self._kubectl(["--context", self._kube_context, *args], left)
                    self.evidence.save(f"{run_id}-{name}", out.stdout + out.stderr)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    errors.append(f"kubectl {name}: {type(exc).__name__}: {exc}")
                    complete = False
        return "complete" if complete else "partial"


# --- Checkers ------------------------------------------------------------------------------------------------
# Pure functions of what the test read, already parsed. Each checks how many items it got before checking every
# item, so an empty list fails instead of passing.


def task_key(text: str | None) -> str | None:
    """``solved`` or ``unsolved`` for a task name or id that names one (``taskset-e2e-unsolved``), else None."""
    text = text or ""
    return "unsolved" if "unsolved" in text else "solved" if "solved" in text else None


def score_stat(result: AgentEvalResult | None, name: str) -> tuple[float | None, int | None]:
    """The mean and count of score ``name`` in a result record; ``(None, None)`` when it has none."""
    for score in result.scores.scores if result is not None else ():
        if score.name == name:
            return score.mean, score.count
    return None, None


def score_mean(result: AgentEvalResult | None, name: str) -> float | None:
    """The mean of score ``name`` in a result record."""
    return score_stat(result, name)[0]


def check_run(status: StatusView) -> None:
    """The run completed, its merged result is ready, and every one of its 2 tasks succeeded."""
    run = status.run
    assert run.phase == "completed", f"[dispatch] run phase {_value(run.phase)}: {run.error}"
    assert run.result_state == "ready", f"[bundle] run result_state {_value(run.result_state)}"
    assert len(status.tasks) == len(EXPECTED_REWARDS), f"[dispatch] expected 2 tasks, got {len(status.tasks)}"
    for task in status.tasks:
        row = task.task
        phase, error = (_value(row.phase), row.error) if row else ("not created", None)
        assert phase == "succeeded" and not error, f"[dispatch] task {task.name} is {phase}: {error}"


def check_task_jobs(status: StatusView, jobs: Sequence[JobView], attempts: Mapping[str, int]) -> None:
    """The evals API and the Jobs API agree on the run's task Jobs; each completed on its first attempt.

    ``jobs`` is the Jobs API's list of the run's task Jobs; ``attempts`` counts each Job's attempts. Task Job names
    are fixed per task index, so a re-dispatch shows as a second attempt.
    """
    named = [task.task.job if task.task else None for task in status.tasks]
    assert len(named) == len(EXPECTED_REWARDS) and all(named), f"[dispatch] expected 2 named task Jobs, got {named}"
    assert len(jobs) == len(named), f"[dispatch] the Jobs API lists {len(jobs)} task Jobs, the run names {len(named)}"
    assert {job.name for job in jobs} == set(named), (
        f"[dispatch] the Jobs API's task Jobs {sorted(job.name for job in jobs)} aren't the run's {named}"
    )
    for job in jobs:
        assert job.status == "completed", f"[dispatch] task Job {job.name} is {job.status}"
    assert set(attempts) == set(named), f"[dispatch] attempts read for {sorted(attempts)}, not {named}"
    for name, count in attempts.items():
        assert count == 1, f"[dispatch] task Job {name} ran {count} attempts; expected 1 (a re-dispatch)"
    bundle = status.bundle
    assert bundle is not None and bundle.status == "completed", (
        f"[bundle] bundle Job {bundle and bundle.job} is {bundle and bundle.status}"
    )


def check_scores(results: TasksetRunResults) -> dict[str, float]:
    """Rewards 1 and 0, ``format_ok`` 1 on both (the separate verifier got the artifact), merged mean 0.5 over 2.

    Returns each task's reward, for the evidence.
    """
    assert len(results.tasks) == len(EXPECTED_REWARDS), f"[verifier] expected 2 task results, got {len(results.tasks)}"
    rewards: dict[str, float] = {}
    for task in results.tasks:
        key = task_key(task.name)
        assert key is not None, f"[verifier] unexpected task {task.name}"
        assert task.result is not None, f"[verifier] task {key} has no result record"
        reward, format_ok = score_mean(task.result, REWARD), score_mean(task.result, FORMAT_OK)
        assert reward == EXPECTED_REWARDS[key], f"[verifier] {key}: reward {reward}, expected {EXPECTED_REWARDS[key]}"
        assert format_ok == 1.0, f"[verifier] {key}: format_ok {format_ok}; its artifact didn't reach the verifier"
        rewards[key] = EXPECTED_REWARDS[key]  # asserted equal above
    assert sorted(rewards) == sorted(EXPECTED_REWARDS), f"[verifier] tasks {sorted(rewards)}"
    assert results.result is not None, "[bundle] the run has no merged result record"
    mean, count = score_stat(results.result, REWARD)
    assert mean == 0.5 and count == 2, f"[bundle] merged reward mean {mean} over {count}; expected 0.5 over 2"
    return rewards


def check_bundle(bundle_dir: Path, results: TasksetRunResults) -> None:
    """The downloaded ``agent-eval-results/`` holds every file, 2 clean trials, and the scores ``results`` shows.

    ``read_bundle`` already rejects a trial or score row without an id: the models require one.
    """
    missing = [name for name in BUNDLE_FILES if not (bundle_dir / name).is_file()]
    assert not missing, f"[bundle] {bundle_dir} lacks {missing}"
    trials, scores = read_bundle(bundle_dir)
    assert len(trials) == len(EXPECTED_REWARDS), f"[bundle] expected 2 trials, got {len(trials)}"
    assert len({trial.task_id for trial in trials}) == len(trials), "[bundle] two trials of one task"
    for trial in trials:
        assert trial.status == AgentEvalTrialStatus.COMPLETED and trial.error is None, (
            f"[bundle] trial {trial.id} is {_value(trial.status)}: {trial.error}"
        )
    # Keyed by task, not sorted: sorting would let the two tasks' rewards swap unnoticed.
    rows = [row for row in scores if row.metric_type == "harbor_reward"]
    assert len(rows) == len(trials), f"[bundle] scores.jsonl has {len(rows)} reward rows for {len(trials)} trials"
    trial_ids = {(trial.id, trial.task_id) for trial in trials}
    for row in rows:
        assert (row.trial_id, row.task_id) in trial_ids, (
            f"[bundle] score row {row.id} names trial {row.trial_id} of {row.task_id}, "
            f"which trials.jsonl doesn't have: {sorted(trial_ids)}"
        )
    bundled: dict[str | None, tuple[Any, Any]] = {}
    for row in rows:
        outputs = {output.name: output.value for output in row.outputs}
        bundled[task_key(row.task_id)] = (outputs.get("reward"), outputs.get("format_ok"))
    assert sorted(bundled, key=str) == sorted(EXPECTED_REWARDS), (
        f"[bundle] scores.jsonl scores tasks {sorted(map(str, bundled))}, expected {sorted(EXPECTED_REWARDS)}"
    )
    shown = {
        task_key(task.name): (score_mean(task.result, REWARD), score_mean(task.result, FORMAT_OK))
        for task in results.tasks
    }
    assert bundled == shown, f"[bundle] scores.jsonl's (reward, format_ok) by task {bundled} aren't results' {shown}"


def check_snapshot(snapshot: Snapshot, *, fresh: bool) -> None:
    """Each task has one ready build with images; in a fresh workspace it's the first attempt, so test_1 built it."""
    assert len(snapshot) == len(EXPECTED_REWARDS), f"[build] expected 2 tasks' builds, got {sorted(snapshot)}"
    for name, build in snapshot.items():
        assert build.status == "ready", f"[build] {name}: build {build.id} is {build.status}: {build.error}"
        assert build.images and all(build.images), (
            f"[build] {name}: build {build.id} has no image digests: {build.images}"
        )
        if fresh:
            assert build.id == "1", f"[build] {name}: newest build is attempt {build.id}, expected 1"


def check_snapshot_unchanged(before: Snapshot, after: Snapshot) -> None:
    """test_2 reused test_1's images: each task's newest build has the same id, Job and images, and is still ready."""
    assert before and len(after) == len(before), f"[reuse] snapshots of {sorted(before)} and {sorted(after)}"
    for name, build in before.items():
        now = after.get(name)
        assert now is not None, f"[reuse] {name} has no build after test_2"
        same = ("id", "job", "images", "status")
        assert {k: getattr(now, k) for k in same} == {k: getattr(build, k) for k in same}, (
            f"[reuse] {name} was rebuilt: {build} -> {now}"
        )


# --- Cancel paths (nightly only) --------------------------------------------------------------------------------


def check_cancelled_while_queued(status: StatusView, jobs: Sequence[JobView]) -> None:
    """E1: a run cancelled while its one task waited for a slot ends cancelled, and no task Job was ever created.

    ``jobs`` is the Jobs API's list of the run's task Jobs.
    """
    assert status.run.phase == "cancelled", f"[cancel] queued run is {_value(status.run.phase)}, expected cancelled"
    assert len(status.tasks) == 1, f"[cancel] expected 1 task, got {len(status.tasks)}"
    row = status.tasks[0].task
    phase, error = (_value(row.phase), row.error) if row else ("not created", None)
    assert phase == "cancelled", f"[cancel] its task is {phase}: {error}"
    assert not jobs, f"[cancel] a task Job was created for the queued run: {[job.name for job in jobs]}"


def check_foreign_job_cancel(results: TasksetRunResults, cancelled_job: str) -> None:
    """C3: cancelling one task Job from outside fails only that task; the run completes with a result over the rest.

    The surviving task must have solved its task, not merely ended: a trial that timed out still leaves its Job
    ``completed`` and its task ``succeeded``, with reward 0 and an error.
    """
    run = results.run
    assert run.phase == "completed", f"[cancel] run is {_value(run.phase)}: {run.error}"
    assert run.result_state == "ready", f"[cancel] run result_state {_value(run.result_state)}"
    assert len(results.tasks) == 2, f"[cancel] expected 2 tasks, got {len(results.tasks)}"
    rows = {task.task.job if task.task else None: task.task for task in results.tasks}
    assert cancelled_job in rows, f"[cancel] {cancelled_job} isn't one of the run's task Jobs {sorted(map(str, rows))}"
    for job, row in rows.items():
        expected = ("failed", "job_cancelled") if job == cancelled_job else ("succeeded", None)
        actual = (_value(row.phase), row.error) if row else ("not created", None)
        assert actual == expected, (
            f"[cancel] task Job {job} is {actual[0]} ({actual[1]}), expected {expected[0]} ({expected[1]})"
        )
    survivor = next(task for task in results.tasks if task.task is None or task.task.job != cancelled_job)
    result = survivor.result
    assert result is not None, f"[cancel] surviving task {survivor.name} has no result record"
    reward, errors = score_mean(result, REWARD), result.summary.error_count if result.summary else None
    assert (reward, errors) == (1.0, 0), (
        f"[cancel] surviving task {survivor.name}: reward {reward} with {errors} errored trials, expected 1.0 "
        "with none (did its oracle time out?)"
    )
    # The merge counts every task of the run; one without an accepted bundle adds no trials.
    summary = results.result.summary if results.result else None
    counts = (summary.task_count, summary.trial_count, summary.error_count) if summary else (None, None, None)
    assert counts == (2, 1, 0), f"[cancel] the run's result has (tasks, trials, errors) {counts}, expected (2, 1, 0)"
