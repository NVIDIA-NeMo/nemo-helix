# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for JobsServiceProgressReporter's status_details handling.

The Jobs service merges ``status_details`` key-wise rather than replacing the
blob (``JobDispatcher._update_status_details_object``, verified end-to-end
against a running platform), so a report only ever has to state what it
observed. These tests pin that: each report sends its own fields and nothing
else, and the only read the reporter makes is the one-shot resume seeding.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.errors import (
    InternalServerError,
    NemoHTTPError,
    NemoTransportError,
    NotFoundError,
)
from nmp.customization_common.training.progress import JobsServiceProgressReporter

SERIES: dict[str, list[dict[str, Any]]] = {
    "train_loss": [{"step": 10, "epoch": 1, "value": 0.5}],
    "train_reward": [{"step": 10, "epoch": 1, "value": 0.62}],
}
#: A blob a mid-run job would have stored, as read back on resume.
STORED: dict[str, Any] = {
    "phase": "training",
    "step": 10,
    "epoch": 1,
    "max_steps": 30,
    "num_epochs": 3,
    "train_loss": 0.5,
    "lr": 5e-06,
    "checkpoint_path": "/ckpt/step-10",
    "metrics": SERIES,
}


class _JobCtx:
    """What the reporter reads off a job context.

    The four identifiers `update_task` sends, plus `is_configured`, which the
    real constructor gates reporting on. Only the tests that go through that
    constructor need the last one -- `_Reporter` below bypasses `__init__` -- but
    a stub that answers everything the real object does is one fewer thing to
    trip over.
    """

    normalized_task = "training"
    workspace = "default"
    job_id = "job-1"
    step = "train"
    is_configured = True


class _Reporter(JobsServiceProgressReporter):
    """Reporter with the SDK and job context stubbed out.

    Bypasses ``__init__`` rather than mocking the client factory: what is under
    test is the status_details logic, and the real constructor wants task
    credentials. Every attribute ``update_task`` touches is set here.

    The Jobs client itself is injected rather than ``fetch_current_metrics``, so
    the real fetch runs.
    """

    def __init__(self, jobs: "_Jobs") -> None:
        self._job_ctx = _JobCtx()  # type: ignore[assignment] - duck-typed stand-in
        self._jobs = jobs.client()  # type: ignore[assignment] - duck-typed stand-in
        self._is_main_rank = True
        self._enabled = True
        self._max_steps = 0
        self._num_epochs = 0


class _Task:
    def __init__(self, status_details: Any) -> None:
        self.status_details = status_details

    def data(self) -> "_Task":
        return self


class _Jobs:
    """A mini Jobs service: records writes, serves reads, counts fetches."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        #: Typed loosely on purpose: several tests store a blob no annotation
        #: describes honestly, because that is what a corrupted read returns.
        self.stored: Any = {}
        self.fetches = 0
        #: Raised by the read instead of answering, for the failure cases.
        self.fetch_error: Exception | None = None
        #: Raised by the *write* instead of accepting, likewise.
        self.send_error: Exception | None = None

    def client(self) -> Any:
        harness = self

        class _Client:
            def update_job_step_task(self, **kwargs: Any) -> None:
                if harness.send_error is not None:
                    raise harness.send_error
                harness.sent.append(kwargs)

            def get_job_step_task(self, **kwargs: Any) -> _Task:
                harness.fetches += 1
                if harness.fetch_error is not None:
                    raise harness.fetch_error
                return _Task(harness.stored)

        return _Client()


@pytest.fixture
def jobs() -> _Jobs:
    """Return a Jobs service harness so update_task and fetch both run for real."""
    harness = _Jobs()
    return harness


def _reporter(jobs: _Jobs, stored: dict[str, Any] | None = None) -> _Reporter:
    """A reporter over the harness, with the server pre-seeded if given."""
    if stored is not None:
        jobs.stored = stored
    return _Reporter(jobs)


def _details(jobs: _Jobs, index: int = -1) -> dict[str, Any]:
    assert jobs.sent, "expected at least one task update"
    return dict(jobs.sent[index]["body"].status_details or {})


# --------------------------------------------------------------------------- #
# A report states what it observed, and nothing else
# --------------------------------------------------------------------------- #


def test_a_report_sends_only_its_own_fields(jobs: _Jobs) -> None:
    """The server merges, so restating untouched fields would be pure upload."""
    _reporter(jobs, STORED).report_running("processing_checkpoint")

    assert _details(jobs) == {"phase": "processing_checkpoint"}


def test_completion_sends_only_its_own_fields(jobs: _Jobs) -> None:
    """The stored series, schedule and checkpoint path survive on the server."""
    _reporter(jobs, STORED).report_completed("Training completed")

    assert _details(jobs) == {"message": "Training completed", "phase": "completed"}
    assert jobs.sent[-1]["body"].status == "completed"


def test_percentage_done_is_derived_from_a_stated_step(jobs: _Jobs) -> None:
    reporter = _reporter(jobs)
    reporter.configure_progress_tracking(max_steps=40, num_epochs=1)
    reporter.report_running("training", step=10, metrics=SERIES)

    assert _details(jobs)["percentage_done"] == 25


def test_percentage_done_is_clamped(jobs: _Jobs) -> None:
    """A resumed or over-run job can report past max_steps."""
    reporter = _reporter(jobs)
    reporter.configure_progress_tracking(max_steps=10, num_epochs=1)
    reporter.report_running("training", step=99, metrics=SERIES)

    assert _details(jobs)["percentage_done"] == 100


def test_error_details_still_ride_along(jobs: _Jobs) -> None:
    _reporter(jobs, STORED).report_error({"message": "oom", "code": "OOM"})

    assert jobs.sent[0]["body"].error_details == {"message": "oom", "code": "OOM"}


# --------------------------------------------------------------------------- #
# Reads: exactly one, for resume seeding
# --------------------------------------------------------------------------- #


def test_no_report_reads_the_blob_back(jobs: _Jobs) -> None:
    """Writes are fire-and-forget. A read per report was the old carry-forward tax."""
    reporter = _reporter(jobs, STORED)
    for step in range(1, 11):
        reporter.report_running("training", step=step, metrics=SERIES)
    reporter.report_running("processing_checkpoint")
    reporter.report_completed("Training completed")
    reporter.report_error("boom")

    assert jobs.fetches == 0


def test_fetch_current_metrics_returns_every_series(jobs: _Jobs) -> None:
    """A resumed job that only seeded train_loss would restart the other curves."""
    assert _reporter(jobs, STORED).fetch_current_metrics() == SERIES


def test_fetch_current_metrics_drops_non_list_values(jobs: _Jobs) -> None:
    """A malformed blob must not poison the accumulator."""
    stored = {"metrics": {"train_loss": [{"step": 1, "epoch": 1, "value": 1.0}], "junk": 3}}

    seeded = _reporter(jobs, stored).fetch_current_metrics()

    assert seeded is not None
    assert set(seeded) == {"train_loss"}


def test_fetch_current_metrics_copies_the_point_lists(jobs: _Jobs) -> None:
    """The caller appends to what it gets back; it must not alias the response."""
    seeded = _reporter(jobs, STORED).fetch_current_metrics()
    assert seeded is not None
    seeded["train_loss"].append({"step": 20, "epoch": 2, "value": 0.4})

    assert len(SERIES["train_loss"]) == 1


def test_fetch_current_metrics_survives_an_unseeded_task(jobs: _Jobs) -> None:
    """First run: the task has no stored details, which is not an error."""
    assert _reporter(jobs, {}).fetch_current_metrics() == {}


# --------------------------------------------------------------------------- #
# A read that failed is not a read that found nothing
# --------------------------------------------------------------------------- #


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "http://jobs/task"))


def test_a_missing_task_seeds_from_nothing(jobs: _Jobs) -> None:
    """A 404 is the one failure that genuinely means "nothing stored"."""
    jobs.fetch_error = NotFoundError(_response(404))

    assert _reporter(jobs, STORED).fetch_current_metrics() == {}


@pytest.mark.parametrize(
    "error",
    [
        NemoTransportError(httpx.ConnectError("connection refused")),
        InternalServerError(_response(500)),
        NemoHTTPError(_response(503)),
    ],
    ids=["transport", "500", "503"],
)
def test_a_failed_read_is_not_reported_as_an_empty_one(jobs: _Jobs, error: Exception) -> None:
    """None, not {}: a caller told "nothing stored" overwrites what it could not read.

    A distributed launch racing a briefly unreachable Jobs service lands here,
    not on the 404, and the merge replaces a sent key wholesale -- so the two
    cases have to be distinguishable at the call site.
    """
    jobs.fetch_error = error

    assert _reporter(jobs, STORED).fetch_current_metrics() is None


@pytest.mark.parametrize(
    "stored",
    [
        ["not", "an", "object"],
        "junk",
        {"metrics": ["not", "an", "object"]},
        {"metrics": "junk"},
        {"metrics": 3},
    ],
    ids=["list-blob", "str-blob", "list-metrics", "str-metrics", "int-metrics"],
)
def test_a_malformed_blob_seeds_nothing_instead_of_raising(jobs: _Jobs, stored: Any) -> None:
    """This runs from the callback's constructor, which backends build outside any try.

    Raising here kills the training process over a blob that only ever cost its
    own points. ``{}`` rather than ``None``: the read succeeded and the data is
    unusable, so overwriting it is the repair, not the damage.
    """
    jobs.stored = stored

    assert _reporter(jobs).fetch_current_metrics() == {}


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def test_disabled_reporter_sends_nothing(jobs: _Jobs) -> None:
    reporter = _reporter(jobs, STORED)
    reporter._enabled = False

    reporter.report_completed("Training completed")

    assert jobs.sent == []


def test_disabled_reporter_does_not_fetch(jobs: _Jobs) -> None:
    reporter = _reporter(jobs, STORED)
    reporter._enabled = False

    assert reporter.fetch_current_metrics() == {}
    assert jobs.fetches == 0


def test_non_main_rank_sends_nothing(jobs: _Jobs) -> None:
    reporter = _reporter(jobs, STORED)
    reporter._is_main_rank = False

    reporter.report_completed("Training completed")

    assert jobs.sent == []


# --------------------------------------------------------------------------- #
# A failed write must not reach the training loop
#
# The single most load-bearing safety property here. `update_task` is called
# synchronously from a backend's logging hook, between one optimizer step and the
# next, with nothing catching underneath -- so an exception escaping it ends the
# run. The chartable-value filter in the callback exists because of what happens
# when a write fails *silently*; this is the other half of that story, and it had
# no test at all.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("jobs service unreachable"),
        TimeoutError("read timed out"),
        RuntimeError("500 Internal Server Error"),
        ValueError("Object of type Histogram is not JSON serializable"),
    ],
)
def test_a_failed_write_does_not_raise_into_the_training_loop(jobs: _Jobs, error: Exception) -> None:
    """Every failure mode a real transport produces, including the serialisation
    error that a non-scalar metric slipping past the filter would cause."""
    reporter = _reporter(jobs)
    jobs.send_error = error

    reporter.report_running(phase="training", step=1, train_loss=0.5)


def test_a_failed_write_is_logged_rather_than_swallowed_silently(jobs: _Jobs, caplog: pytest.LogCaptureFixture) -> None:
    """Swallowing is right; swallowing quietly is not.

    A run whose reporting stopped an hour ago looks exactly like a run that is
    reporting fine, unless the log says otherwise.
    """
    reporter = _reporter(jobs)
    jobs.send_error = ConnectionError("jobs service unreachable")

    with caplog.at_level(logging.WARNING):
        reporter.report_running(phase="training", step=1)

    assert "jobs service unreachable" in caplog.text


def test_reporting_survives_a_failure_and_resumes(jobs: _Jobs) -> None:
    """One failed write must not poison the reporter for the rest of the run.

    Transport failures are usually transient -- a restarting pod, a brief network
    partition -- and a reporter that gave up on the first one would lose the whole
    remaining history for a blip.
    """
    reporter = _reporter(jobs)
    jobs.send_error = ConnectionError("brief blip")
    reporter.report_running(phase="training", step=1)

    jobs.send_error = None
    reporter.report_running(phase="training", step=2)

    assert [sent["body"].status_details["step"] for sent in jobs.sent] == [2]


# --------------------------------------------------------------------------- #
# How long an update may block the training loop
# --------------------------------------------------------------------------- #


def test_progress_updates_use_a_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK default is tuned for a call someone is waiting on; this is not one.

    These updates are made synchronously between optimizer steps, so the timeout
    is how long a struggling Jobs service may hold the GPU idle. At the SDK's
    60s read timeout with two retries that is roughly three minutes per report,
    repeated for the rest of the run.

    Asserted against the SDK the reporter actually built, not against the
    constant, so that wiring the constant to nothing would fail here.
    """
    timeouts: list[httpx.Timeout] = []
    optioned_clients: list[_JobsClient] = []

    class _JobsClient:
        def __init__(self, optioned: bool = False) -> None:
            self.optioned = optioned

        def with_options(self, *, timeout: httpx.Timeout) -> "_JobsClient":
            timeouts.append(timeout)
            client = _JobsClient(optioned=True)
            optioned_clients.append(client)
            return client

    monkeypatch.setattr(
        "nmp.customization_common.training.progress.get_task_nemo_client",
        lambda service_name: object(),
    )
    monkeypatch.setattr(
        "nmp.customization_common.training.progress.JobsClient.from_client",
        lambda client: _JobsClient(),
    )
    JobsServiceProgressReporter(_JobCtx(), service_name="customizer")  # ty: ignore[invalid-argument-type] # duck-typed stub

    # A 12 MB blob -- far past anything a real run produces -- costs 46ms plus
    # 0.31ms/KB, so about 4s on the wire. 15s leaves room and still bounds the stall.
    assert timeouts == [httpx.Timeout(10.0, connect=5.0)]
    assert len(optioned_clients) == 1
    assert optioned_clients[0].optioned, "kept the un-optioned Jobs client, so the timeout never applies"


def test_the_task_client_is_not_rebuilt_with_a_custom_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_task_nemo_client` skips workload identity when handed a client.

    Setting the timeout by passing our own `http_client` would therefore change
    how a task authenticates, silently and only in the deployment that uses
    workload identity -- which is the one hardest to test. `with_options` leaves
    that path alone.
    """
    seen: dict[str, object | None] = {}

    class _JobsClient:
        def with_options(self, *, timeout: httpx.Timeout) -> "_JobsClient":
            return self

    def _factory(service_name: str, *, http_client: object | None = None, workspace: str | None = None) -> object:
        seen["http_client"] = http_client
        seen["workspace"] = workspace
        return object()

    monkeypatch.setattr("nmp.customization_common.training.progress.get_task_nemo_client", _factory)
    monkeypatch.setattr(
        "nmp.customization_common.training.progress.JobsClient.from_client",
        lambda client: _JobsClient(),
    )
    JobsServiceProgressReporter(_JobCtx(), service_name="customizer")  # ty: ignore[invalid-argument-type] # duck-typed stub

    assert seen["http_client"] is None, "no client passed, so the auth path is untouched"


def test_close_closes_the_owning_task_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived service clients share transport, so the root task client owns close."""

    class _TaskClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _JobsClient:
        def __init__(self) -> None:
            self.closed = False

        def with_options(self, *, timeout: httpx.Timeout) -> "_JobsClient":
            return self

        def close(self) -> None:
            self.closed = True

    task_client = _TaskClient()
    jobs_client = _JobsClient()
    seen_clients: list[_TaskClient] = []

    def _from_client(client: _TaskClient) -> _JobsClient:
        seen_clients.append(client)
        return jobs_client

    monkeypatch.setattr(
        "nmp.customization_common.training.progress.get_task_nemo_client",
        lambda service_name: task_client,
    )
    monkeypatch.setattr(
        "nmp.customization_common.training.progress.JobsClient.from_client",
        _from_client,
    )

    reporter = JobsServiceProgressReporter(_JobCtx(), service_name="customizer")  # ty: ignore[invalid-argument-type] # duck-typed stub
    reporter.close()

    assert seen_clients == [task_client]
    assert task_client.closed
    assert not jobs_client.closed
