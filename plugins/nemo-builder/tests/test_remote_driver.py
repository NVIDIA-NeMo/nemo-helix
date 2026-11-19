# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote-build step's driver, against a fake build service, a fake builder and a fake clock."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.run import remote as driver
from nemo_builder_plugin.run.fetch import ArchiveRefused
from nemo_builder_plugin.run.remote import (
    CANCEL_PAUSE_SECONDS,
    CANCEL_SECONDS,
    CANCEL_TRIES,
    COMPLETION_SECONDS,
    FAILURE_MARKER,
    LINE_OVERHEAD,
    MAX_BACKOFF_SECONDS,
    MAX_REASON_CHARS,
    POLL_SECONDS,
    START_PATIENCE_SECONDS,
    STOPPED,
    TAIL_CHARS,
    Clock,
    ImageFailed,
    JobAttempt,
    OverBudgetError,
    PermanentError,
    RemoteBuild,
    RemoteBuildError,
    RemoteBuildStatus,
    RemoteImage,
    RetryableError,
    idempotency_key,
    run_remote_build,
    sleep_unless,
    stage_context,
    step_main,
    take_credential,
)
from nemo_builder_plugin.settle import _FAILURE_LINE, _marked_lines
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError

DESTINATION = "registry.example.com/team/images"
#: What follows a reason when the build was to be cancelled and the service couldn't be told.
UNCANCELLED = "[build build-1; the service could not be told to stop it, it may still be running]"
DIGEST = "sha256:" + "d" * 64
ATTEMPT = JobAttempt(workspace="default", job_uid="job-uid-1", attempt_id="attempt-1")

QUEUED = RemoteBuildStatus("queued")
BUILDING = RemoteBuildStatus("building", "submitting")
VERIFYING = RemoteBuildStatus("verifying", "checking the image")
DONE = RemoteBuildStatus(
    "done", image=RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", attributes={"builder": "v1"})
)


def _row(
    name: str, digest: str | None = None, status: str = "pending", repository: str = "team/images"
) -> ContainerImage:
    return ContainerImage.model_validate(
        {
            "name": name,
            "workspace": "default",
            "status": status,
            "registry": "registry.example.com",
            "repository": repository,
            "digest": digest,
            "provenance": {
                "build_set": "demo",
                "revision": 1,
                "job": "default/builder-demo-1",
                "request_digest": "sha256:" + "a" * 64,
                "backend": "remote",
            },
        }
    )


def _config(*images: str, timeout: int = 600, dockerfile: str = "Dockerfile") -> RemoteBuildStepConfig:
    return RemoteBuildStepConfig.model_validate(
        {
            "images": [
                {
                    "image": name,
                    "source": {"fileset": "default/fs-a"},
                    "dockerfile": dockerfile,
                    "platform": "linux/amd64",
                }
                for name in images or ("demo-1",)
            ],
            "registry": "registry.example.com",
            "repository": "team/images",
            "build_timeout_seconds": timeout,
        }
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        #: Waits a stop doesn't cut short: the ones between two tries to cancel.
        self.pauses: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds

    @property
    def clock(self) -> Clock:
        return Clock(monotonic=self.monotonic, sleep=self.sleep, pause=self.pause)


class FakeService:
    """Answers ``start``, ``status`` and ``cancel`` from scripts. The last entry of a script repeats."""

    def __init__(
        self,
        statuses: list[RemoteBuildStatus | Exception],
        *,
        start: list[str | Exception] | None = None,
        cancel: Exception | list[Exception | None] | None = None,
    ) -> None:
        self._statuses = list(statuses)
        self._start: list[str | Exception] = list(start or ["build-1"])
        #: What each ``cancel`` raises; None is a cancel the service takes.
        self._cancel: list[Exception | None] = list(cancel) if isinstance(cancel, list) else [cancel]
        self.started: list[RemoteBuild] = []
        self.polled: list[str] = []
        self.cancelled: list[str] = []

    @staticmethod
    def _next[T](script: list[T]) -> T:
        return script.pop(0) if len(script) > 1 else script[0]

    def start(self, build: RemoteBuild) -> str:
        self.started.append(build)
        answer = self._next(self._start)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def status(self, build_id: str) -> RemoteBuildStatus:
        self.polled.append(build_id)
        answer = self._next(self._statuses)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def cancel(self, build_id: str) -> None:
        self.cancelled.append(build_id)
        answer = self._next(self._cancel)
        if answer is not None:
            raise answer


def _unreachable() -> NemoTransportError:
    return NemoTransportError(httpx.ConnectError("connection refused"))


def _refused(status: int, detail: str) -> NemoHTTPError:
    request = httpx.Request("POST", "http://builder.test/complete")
    return NemoHTTPError(httpx.Response(status, json={"detail": detail}, request=request))


class FakeBuilder:
    def __init__(
        self,
        statuses: dict[str, str] | None = None,
        *,
        repository: str = "team/images",
        failing: list[Exception] | None = None,
        unreadable: dict[str, Exception] | None = None,
        completes_as: tuple[str, str] | None = None,
    ) -> None:
        #: The status and digest ``complete`` answers with, when not ``ready`` at the digest it was given.
        self.completes_as = completes_as
        self.statuses = statuses or {}
        self.repository = repository
        #: Rows whose read fails.
        self.unreadable = unreadable or {}
        #: What the next calls to ``complete`` raise, before one succeeds.
        self.failing = list(failing or [])
        self.completed: list[tuple[str, str, ExternalImage | None]] = []

    def row(self, image: str) -> ContainerImage:
        if image in self.unreadable:
            raise self.unreadable[image]
        return _row(image, status=self.statuses.get(image, "pending"), repository=self.repository)

    def complete(
        self,
        image: str,
        digest: str,
        runtime_config: ImageRuntimeConfig | None,
        external: ExternalImage | None = None,
    ) -> ContainerImage:
        assert runtime_config is None, "a remote build leaves no image config to read"
        if self.failing:
            raise self.failing.pop(0)
        self.completed.append((image, digest, external))
        status, answered = self.completes_as or ("ready", digest)
        return _row(image, answered, status)


@pytest.fixture
def context(tmp_path: Path) -> Path:
    directory = tmp_path / "context"
    (directory / "services" / "feed").mkdir(parents=True)
    (directory / "Dockerfile").write_text("FROM scratch\n")
    (directory / "services" / "feed" / "Dockerfile").write_text("FROM busybox\n")
    (directory / "app.py").write_text("print('hello')\n")
    return directory


def _run(
    service: FakeService,
    builder: FakeBuilder,
    context: Path,
    tmp_path: Path,
    *,
    config: RemoteBuildStepConfig | None = None,
    clock: FakeClock | None = None,
    stop_after_polls: int | None = None,
    cancelled: Callable[[], bool] | None = None,
    fetch: Callable[[RemoteBuildImage, Path], Path] | None = None,
    redact: Callable[[str], str] | None = None,
) -> int:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return run_remote_build(
        config or _config(),
        service=service,
        builder=builder,
        fetch=fetch or (lambda image, directory: context),
        attempt=ATTEMPT,
        scratch=scratch,
        clock=(clock or FakeClock()).clock,
        cancelled=cancelled or (lambda: stop_after_polls is not None and len(service.polled) >= stop_after_polls),
        redact=redact or (lambda text: text),
    )


class TestABuildThatSucceeds:
    def test_the_row_is_completed_once_with_the_digest_and_the_services_handle(
        self, context: Path, tmp_path: Path
    ) -> None:
        service = FakeService([QUEUED, BUILDING, VERIFYING, DONE])
        builder = FakeBuilder()
        clock = FakeClock()

        assert _run(service, builder, context, tmp_path, clock=clock) == 0

        external = ExternalImage(image_id="img-1", build_id="build-1", attributes={"builder": "v1"})
        assert builder.completed == [("demo-1", DIGEST, external)]
        assert service.cancelled == []
        # One poll per state, with the poll interval between them.
        assert service.polled == ["build-1"] * 4
        assert clock.sleeps == [POLL_SECONDS] * 3

    def test_the_service_is_handed_the_context_the_platform_and_a_key(self, context: Path, tmp_path: Path) -> None:
        service = FakeService([DONE])
        _run(service, FakeBuilder(), context, tmp_path)
        (build,) = service.started
        assert build.context == context
        assert build.platform == "linux/amd64"
        assert build.key == idempotency_key(ATTEMPT, "demo-1")
        assert build.labels == {"workspace": "default", "image": "demo-1", "job": "job-uid-1"}
        # For a service that can end a build by itself: a step killed outright cancels nothing.
        assert build.timeout_seconds == 600

    @pytest.mark.parametrize("build_id", ["", "  ", None], ids=["empty", "blank", "none"])
    def test_a_build_the_client_doesnt_name_fails_and_is_said_to_maybe_be_running(
        self, build_id: str | None, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The row names its build by the ID, and no remote row is completed without one."""
        service = FakeService([DONE], start=[build_id])  # ty: ignore[invalid-argument-type]
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert (service.polled, builder.completed) == ([], [])
        assert "started a build without naming it" in caplog.text
        assert f"[a build may be running, started with key {idempotency_key(ATTEMPT, 'demo-1')}]" in caplog.text

    def test_a_service_with_no_handle_of_its_own_still_completes(self, context: Path, tmp_path: Path) -> None:
        done = RemoteBuildStatus("done", image=RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id=""))
        builder = FakeBuilder()
        assert _run(FakeService([done]), builder, context, tmp_path) == 0
        assert builder.completed == [("demo-1", DIGEST, ExternalImage(build_id="build-1"))]

    def test_each_stage_is_logged_once(self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            _run(FakeService([QUEUED, BUILDING, BUILDING, VERIFYING, DONE]), FakeBuilder(), context, tmp_path)
        stages = [record.getMessage() for record in caplog.records if record.getMessage().startswith("build build-1")]
        assert stages == [
            "build build-1: queued",
            "build build-1: building (submitting)",
            "build build-1: verifying (checking the image)",
            "build build-1: done",
        ]

    def test_what_the_service_merely_reports_is_cut_to_fit_not_failed(self, context: Path, tmp_path: Path) -> None:
        """A long attribute must not cost an image that is built and published."""
        attributes = {"log": "x" * 2000} | {f"k{i}": "v" for i in range(40)}
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", attributes=attributes)
        builder = FakeBuilder()
        assert _run(FakeService([RemoteBuildStatus("done", image=image)]), builder, context, tmp_path) == 0
        ((_, _, external),) = builder.completed
        assert external is not None
        assert (len(external.attributes), len(external.attributes["log"])) == (32, 512)

    def test_a_builder_that_cant_be_reached_is_tried_again(self, context: Path, tmp_path: Path) -> None:
        """The image is built: only telling the builder is left, and that is safe to repeat."""
        builder = FakeBuilder(failing=[_unreachable(), _refused(503, "restarting"), _refused(429, "slow down")])
        clock = FakeClock()
        assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 0
        assert len(builder.completed) == 1
        assert clock.sleeps == [2.0, 4.0, 8.0]

    def test_an_api_pod_from_before_the_handle_existed_is_tried_past(self, context: Path, tmp_path: Path) -> None:
        """Mid-upgrade, an older pod refuses the request's shape. The next try may reach a newer one."""
        builder = FakeBuilder(failing=[_refused(422, "external: extra fields not permitted")])
        clock = FakeClock()
        assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 0
        assert (len(builder.completed), clock.sleeps) == (1, [2.0])


class TestRetrying:
    def test_a_lost_response_repeats_the_same_start(self, context: Path, tmp_path: Path) -> None:
        service = FakeService([DONE], start=[RetryableError("connection reset"), "build-1"])
        builder = FakeBuilder()

        assert _run(service, builder, context, tmp_path) == 0

        first, second = service.started
        # The same key and the same context: the service starts one build, whichever call it saw.
        assert (first.key, first.context) == (second.key, second.context)
        assert len(builder.completed) == 1

    def test_a_call_that_keeps_failing_gives_the_image_up(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[RetryableError("connection reset")])
        builder = FakeBuilder()
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path, config=_config(timeout=3600), clock=clock) == 1
        assert builder.completed == []
        # Given up on well before the build's own hour, and not after a few tries: a service restarts for minutes.
        assert START_PATIENCE_SECONDS - MAX_BACKOFF_SECONDS <= clock.now <= START_PATIENCE_SECONDS
        tries = len(service.started)
        assert f"{FAILURE_MARKER} demo-1: starting the build failed {tries} times over {clock.now:.0f}s" in caplog.text
        # A start whose answer was lost may have started a build this step can't name, so can't cancel. Its key,
        # whole, is what finds it.
        assert f"[a build may be running, started with key {idempotency_key(ATTEMPT, 'demo-1')}]" in caplog.text
        assert service.cancelled == []

    @pytest.mark.parametrize(
        ("start", "timeout", "reason"),
        [
            ([RetryableError("connection reset"), PermanentError("400: bad request")], 600, "400: bad request"),
            ([RetryableError("connection reset")], 120, "connection reset; still failing at the build timeout"),
            ([RetryableError("connection reset"), KeyError("state")], 600, "unexpected KeyError"),
        ],
        ids=["then-refused", "until-the-timeout", "then-a-bug"],
    )
    def test_however_a_start_is_given_up_a_lost_one_is_said_to_maybe_be_running(
        self,
        start: list[str | Exception],
        timeout: int,
        reason: str,
        context: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        service = FakeService([DONE], start=start)
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=timeout)) == 1
        assert f"{FAILURE_MARKER} demo-1: starting the build: {reason}" in caplog.text
        assert f"[a build may be running, started with key {idempotency_key(ATTEMPT, 'demo-1')}]" in caplog.text

    def test_a_stop_after_a_lost_start_names_the_build_that_may_be_running(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[RetryableError("connection reset")])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, cancelled=lambda: bool(service.started)) == 1
        assert len(service.started) == 1
        assert f"{STOPPED} [a build may be running, started with key {idempotency_key(ATTEMPT, 'demo-1')}]" in (
            caplog.text
        )

    def test_a_wait_for_budget_starts_the_count_of_lost_responses_again(self, context: Path, tmp_path: Path) -> None:
        """Nine lost responses, a busy answer, nine more: the service answered in between, so neither run of nine
        is the long silence a start is given up for."""
        lost: list[str | Exception] = [RetryableError("connection reset")] * 9
        service = FakeService([DONE], start=[*lost, OverBudgetError("busy"), *lost, "build-1"])
        builder = FakeBuilder()
        clock = FakeClock()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=3600), clock=clock) == 0
        assert (len(service.started), len(builder.completed)) == (20, 1)
        # After the busy answer, the waits for lost responses start from the shortest again.
        assert clock.sleeps[8:12] == [MAX_BACKOFF_SECONDS, 2.0, 2.0, 4.0]

    def test_a_start_rides_out_a_service_that_is_down_for_a_few_minutes(self, context: Path, tmp_path: Path) -> None:
        """Queued builds reach their slot one after another: a short outage must not fail each in turn."""
        down: list[str | Exception] = [RetryableError("connection refused")] * 8
        service = FakeService([DONE], start=[*down, "build-1"])
        builder = FakeBuilder()
        clock = FakeClock()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=3600), clock=clock) == 0
        assert clock.now > 240
        assert len(builder.completed) == 1
        # The same key on every try, so the service started one build, whichever call it saw.
        assert len({build.key for build in service.started}) == 1

    def test_a_backoff_never_grows_past_its_cap(self) -> None:
        """A day of a full queue is over a thousand waits; the doubling must not run off the end of a float."""
        assert [driver._backoff(tries) for tries in (1, 2, 6, 7, 5000)] == [2.0, 4.0, 60.0, 60.0, MAX_BACKOFF_SECONDS]

    def test_a_used_up_budget_waits_and_then_goes_on(self, context: Path, tmp_path: Path) -> None:
        service = FakeService([DONE], start=[OverBudgetError("queue full"), OverBudgetError("queue full"), "build-1"])
        clock = FakeClock()
        assert _run(service, FakeBuilder(), context, tmp_path, clock=clock) == 0
        assert clock.sleeps == [2.0, 4.0]

    def test_a_budget_that_never_frees_fails_at_the_build_timeout(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[OverBudgetError("queue full")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=300), clock=clock) == 1
        assert clock.now < 300
        assert max(clock.sleeps) == MAX_BACKOFF_SECONDS
        assert "queue full; still failing at the build timeout" in caplog.text
        # The service answered every time, and never that it had started anything.
        assert "a build may be running" not in caplog.text

    def test_one_lost_response_after_a_long_wait_for_budget_is_still_repeated(
        self, context: Path, tmp_path: Path
    ) -> None:
        """The two are counted apart: a busy service says nothing about the connection."""
        waiting: list[str | Exception] = [OverBudgetError("queue full")] * 30
        service = FakeService([DONE], start=[*waiting, RetryableError("connection reset"), "build-1"])
        builder = FakeBuilder()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=3600)) == 0
        assert len(builder.completed) == 1

    def test_a_status_read_that_fails_for_a_while_does_not_cost_a_running_build(
        self, context: Path, tmp_path: Path
    ) -> None:
        """A build takes minutes of the service's queue; a service that restarts answers 503 for a while."""
        failing: list[RemoteBuildStatus | Exception] = [RetryableError("503")] * 40
        service = FakeService([BUILDING, *failing, DONE])
        builder = FakeBuilder()
        clock = FakeClock()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=3600), clock=clock) == 0
        # Longer than a start is given: a build that is running is worth more than one not yet started.
        assert clock.now > START_PATIENCE_SECONDS
        assert service.cancelled == []
        assert len(builder.completed) == 1

    def test_a_status_read_still_failing_at_the_timeout_cancels_the_build(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([RetryableError("503")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=120), clock=clock) == 1
        assert service.cancelled == ["build-1"]
        # Read one last time at the deadline itself, not given up when the next wait would have passed it.
        assert clock.now == 120
        assert "reading the build: 503; still failing at the build timeout [build build-1]" in caplog.text

    def test_a_service_that_answers_again_before_the_timeout_keeps_its_build(
        self, context: Path, tmp_path: Path
    ) -> None:
        """The waits double. With a minute for the build and the service silent for 40 seconds, the wait after
        the read at 30 would end at 62: cut to end at 60, it finds the build done."""
        clock = FakeClock()

        class Offline(FakeService):
            def status(self, build_id: str) -> RemoteBuildStatus:
                self.polled.append(build_id)
                if clock.now < 40:
                    raise RetryableError("timeout")
                return DONE

        service = Offline([])
        builder = FakeBuilder()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=60), clock=clock) == 0
        assert (clock.sleeps, service.cancelled) == ([2.0, 4.0, 8.0, 16.0, 30.0], [])
        assert len(builder.completed) == 1

    def test_a_start_is_not_made_when_its_build_could_only_be_cancelled(self, context: Path, tmp_path: Path) -> None:
        """A read at the deadline costs nothing. A start there would be a build to cancel."""
        clock = FakeClock()
        service = FakeService([DONE], start=[OverBudgetError("busy")])
        assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=60), clock=clock) == 1
        # 2, 4, 8 and 16 seconds waited: the next wait, 32, would pass the minute, so no fifth try is made.
        assert (len(service.started), clock.now) == (5, 30.0)

    @pytest.mark.parametrize("failure", [RemoteBuildError("no subclass"), KeyError("state")], ids=["base", "a-bug"])
    def test_a_client_that_breaks_the_contract_fails_the_image_and_leaves_no_build_running(
        self, failure: Exception, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([failure])
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert service.cancelled == ["build-1"]
        assert builder.completed == []
        assert f"reading the build: unexpected {type(failure).__name__}" in caplog.text
        assert "[build build-1]" in caplog.text

    def test_a_permanent_error_fails_at_once(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[PermanentError("403: not permitted to build")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, clock=clock) == 1
        assert (len(service.started), clock.sleeps) == (1, [])
        assert f"{FAILURE_MARKER} demo-1: starting the build: 403: not permitted to build" in caplog.text
        # Refused outright, nothing was started: there is no build to go looking for.
        assert "a build may be running" not in caplog.text


class TestABuildThatFails:
    def test_the_services_reason_and_its_build_id_are_reported(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([BUILDING, RemoteBuildStatus("failed", "base image could not be pulled")])
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert builder.completed == []
        assert "reports it failed: base image could not be pulled [build build-1]" in caplog.text

    def test_a_build_past_its_timeout_is_cancelled(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([BUILDING])
        builder = FakeBuilder()
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path, config=_config(timeout=35), clock=clock) == 1
        assert service.cancelled == ["build-1"]
        assert builder.completed == []
        assert clock.now == 35
        assert "timed out after 35s, while building (submitting) [build build-1]" in caplog.text

    def test_a_job_told_to_stop_cancels_its_build(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([BUILDING])
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path, stop_after_polls=2) == 1
        assert service.cancelled == ["build-1"]
        assert builder.completed == []
        assert "cancelled: the job was told to stop [build build-1]" in caplog.text

    def test_a_job_told_to_stop_before_it_started_starts_nothing(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE])
        fetched: list[str] = []

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            fetched.append(image.image)
            return context

        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, cancelled=lambda: True, fetch=fetch) == 1
        assert (fetched, service.started) == ([], [])
        assert f"{FAILURE_MARKER} demo-1: {STOPPED}" in caplog.text
        assert "a build may be running" not in caplog.text

    def test_a_stop_while_the_context_is_fetched_starts_nothing(self, context: Path, tmp_path: Path) -> None:
        """Jobs ends a step seconds after it starts when its time is already up: the signal lands mid-fetch."""
        service = FakeService([DONE])
        fetched: list[str] = []

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            fetched.append(image.image)
            return context

        assert _run(service, FakeBuilder(), context, tmp_path, cancelled=lambda: bool(fetched), fetch=fetch) == 1
        assert (fetched, service.started, service.cancelled) == (["demo-1"], [], [])

    def test_a_stop_during_a_backoff_cancels_the_build(self, context: Path, tmp_path: Path) -> None:
        """A pod has about 30 seconds from the signal. A backoff is up to a minute."""
        service = FakeService([BUILDING, OverBudgetError("429")])
        clock = FakeClock()
        assert _run(service, FakeBuilder(), context, tmp_path, clock=clock, cancelled=lambda: clock.now >= 12) == 1
        assert service.cancelled == ["build-1"]
        # The poll, then one backoff: the stop is seen before the next read, not after the retries run out.
        assert clock.sleeps == [POLL_SECONDS, 2.0]

    def test_after_a_stop_no_other_image_is_started(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([BUILDING], start=["build-1", "build-2"])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config("a", "b"), stop_after_polls=1) == 1
        assert [build.labels["image"] for build in service.started] == ["a"]
        assert service.cancelled == ["build-1"]
        assert f"{FAILURE_MARKER} b: {STOPPED}" in caplog.text

    def test_status_that_cannot_be_read_cancels_the_build(self, context: Path, tmp_path: Path) -> None:
        service = FakeService([PermanentError("404: no such build")])
        assert _run(service, FakeBuilder(), context, tmp_path) == 1
        assert service.cancelled == ["build-1"]

    def test_a_driver_that_breaks_while_the_build_runs_cancels_it(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Shapeless(FakeService):
            def status(self, build_id: str) -> RemoteBuildStatus:
                self.polled.append(build_id)
                return object()  # ty: ignore[invalid-return-type]

        service = Shapeless([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path) == 1
        assert service.cancelled == ["build-1"]
        assert "unexpected AttributeError" in caplog.text
        assert "[build build-1]" in caplog.text

    def test_a_driver_that_breaks_after_the_build_ended_cancels_nothing(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE])
        builder = FakeBuilder(failing=[RuntimeError("boom")])
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert service.cancelled == []
        assert f"{FAILURE_MARKER} demo-1: unexpected RuntimeError: boom [build build-1]" in caplog.text

    def test_a_cancel_that_fails_does_not_replace_the_reason(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([PermanentError("404: no such build")], cancel=ConnectionError("reset"))
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path) == 1
        # The reason stands, and what follows it says the build this step started may still be running.
        assert f"{FAILURE_MARKER} demo-1: reading the build: 404: no such build {UNCANCELLED}" in caplog.text
        # Not an error that says "try again": one try.
        assert service.cancelled == ["build-1"]

    def test_a_cancel_that_may_not_have_arrived_is_repeated(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lost: list[Exception | None] = [RetryableError("503"), RetryableError("503"), None]
        service = FakeService([BUILDING], cancel=lost)
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=35), clock=clock) == 1
        assert service.cancelled == ["build-1"] * 3
        assert clock.pauses == [CANCEL_PAUSE_SECONDS] * 2
        # The third try was taken, so the row says nothing of a build left running.
        assert "timed out after 35s, while building (submitting) [build build-1]" in caplog.text

    def test_a_cancel_that_never_arrives_is_given_up_and_said_on_the_row(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([BUILDING], cancel=RetryableError("503"))
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, stop_after_polls=1, clock=clock) == 1
        assert service.cancelled == ["build-1"] * CANCEL_TRIES
        # Waited between the tries although the job was told to stop: the step's own sleep would not have. The one
        # sleep is the poll before the stop was seen.
        assert (clock.pauses, clock.sleeps) == ([CANCEL_PAUSE_SECONDS] * (CANCEL_TRIES - 1), [POLL_SECONDS])
        assert f"{STOPPED} {UNCANCELLED}" in caplog.text

    @pytest.mark.parametrize(
        ("statuses", "timeout", "reason"),
        [
            ([BUILDING], 35, "timed out after 35s, while building (submitting)"),
            ([RemoteBuildStatus("succeeded")], 600, "a stage the driver doesn't know: 'succeeded'"),  # ty: ignore[invalid-argument-type]
            ([KeyError("state")], 600, "reading the build: unexpected KeyError: 'state'"),
        ],
        ids=["timed-out", "unknown-stage", "a-bug-in-the-client"],
    )
    def test_whyever_a_build_is_cancelled_a_cancel_that_fails_is_said_on_the_row(
        self,
        statuses: list[RemoteBuildStatus | Exception],
        timeout: int,
        reason: str,
        context: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A service that is busy is asked again too, as one that didn't answer is."""
        service = FakeService(statuses, cancel=OverBudgetError("429"))
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=timeout)) == 1
        assert service.cancelled == ["build-1"] * CANCEL_TRIES
        assert f"{reason} {UNCANCELLED}" in caplog.text

    def test_a_driver_that_breaks_while_the_build_runs_says_when_it_could_not_cancel(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class Shapeless(FakeService):
            def status(self, build_id: str) -> RemoteBuildStatus:
                self.polled.append(build_id)
                return object()  # ty: ignore[invalid-return-type]

        service = Shapeless([DONE], cancel=ConnectionError("reset"))
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path) == 1
        assert service.cancelled == ["build-1"]
        assert "unexpected AttributeError" in caplog.text
        assert UNCANCELLED in caplog.text

    def test_cancelling_does_not_outlast_the_time_a_stopped_pod_has(self, context: Path, tmp_path: Path) -> None:
        """Each call is bounded by the client's own timeouts, not by the driver. One is repeated only if another
        as slow would still end in time."""

        def cancelling(seconds: float) -> tuple[int, float]:
            clock = FakeClock()
            began: list[float] = []

            class Slow(FakeService):
                def cancel(self, build_id: str) -> None:
                    began.append(clock.now)
                    clock.now += seconds
                    super().cancel(build_id)

            service = Slow([BUILDING], cancel=RetryableError("timed out"))
            assert _run(service, FakeBuilder(), context, tmp_path, stop_after_polls=1, clock=clock) == 1
            return len(service.cancelled), clock.now - began[0]

        # How many tries, and how long they took in all.
        assert cancelling(0.0) == (3, 2 * CANCEL_PAUSE_SECONDS)
        assert cancelling(5.0) == (3, 19.0)
        assert cancelling(9.0) == (1, 9.0)
        assert cancelling(15.0) == (1, 15.0)
        assert all(cancelling(seconds)[1] < CANCEL_SECONDS for seconds in (0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 9.0, 12.0))

    def test_a_stage_the_driver_doesnt_know_fails_at_once_and_cancels_the_build(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A client that passes a service's own state through. Read as "still running", it would be polled until
        the build's timeout, even after the build had ended."""
        service = FakeService([RemoteBuildStatus("succeeded")])  # ty: ignore[invalid-argument-type]
        builder = FakeBuilder()
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path, config=_config(timeout=3600), clock=clock) == 1
        assert (service.polled, service.cancelled, clock.sleeps) == (["build-1"], ["build-1"], [])
        assert builder.completed == []
        assert "reports a stage the driver doesn't know: 'succeeded' [build build-1]" in caplog.text

    def test_a_build_reported_done_with_no_image_fails(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([RemoteBuildStatus("done")]), builder, context, tmp_path) == 1
        assert builder.completed == []
        assert "reports it done, with no image [build build-1]" in caplog.text

    @pytest.mark.parametrize(
        "completes_as", [("failed", DIGEST), ("ready", "sha256:" + "e" * 64)], ids=["failed", "another-digest"]
    )
    def test_a_row_the_builder_completed_as_something_else_fails(
        self, completes_as: tuple[str, str], context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        builder = FakeBuilder(completes_as=completes_as)
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path) == 1
        status, digest = completes_as
        assert f"the builder completed it as {status} at {digest}, not ready at {DIGEST} [build build-1]" in (
            caplog.text
        )

    def test_a_stop_while_the_builder_cant_be_reached_ends_the_tries(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Completing is tried for minutes. A pod told to stop has seconds."""
        builder = FakeBuilder(failing=[_unreachable()])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            stopped = _run(
                FakeService([DONE]), builder, context, tmp_path, clock=clock, cancelled=lambda: not builder.failing
            )
        assert (stopped, clock.sleeps, builder.completed) == (1, [], [])
        assert "it was built, but the builder could not be told" in caplog.text

    def test_a_build_the_service_cancelled_fails(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([RemoteBuildStatus("cancelled", "cancelled by an operator")])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path) == 1
        assert "reports it cancelled: cancelled by an operator [build build-1]" in caplog.text
        # It has ended: there is nothing left to cancel.
        assert service.cancelled == []

    def test_an_image_published_anywhere_but_the_rows_repository_is_never_completed(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The row, not the step's config, says where: the two differ if the deployment's repository changed
        between a submit that wrote the rows and the one that created the job."""
        builder = FakeBuilder(repository="team/older")
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path) == 1
        assert builder.completed == []
        assert "not a digest in registry.example.com/team/older [build build-1]" in caplog.text

    def test_a_handle_too_long_to_record_fails_with_the_build_named(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="i" * 600)
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([RemoteBuildStatus("done", image=image)]), builder, context, tmp_path) == 1
        assert builder.completed == []
        assert "handle for the image can't be recorded (image_id:" in caplog.text
        assert "[build build-1]" in caplog.text

    def test_a_completion_the_builder_refuses_fails_with_the_build_named(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        builder = FakeBuilder(failing=[_refused(409, "demo-1 has already failed")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 1
        assert clock.sleeps == []
        assert "it was built, but the builder refused to complete it" in caplog.text
        assert "demo-1 has already failed [build build-1]" in caplog.text

    def test_a_builder_that_stays_unreachable_gives_the_image_up(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        builder = FakeBuilder(failing=[_unreachable()] * 50)
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 1
        assert clock.now < COMPLETION_SECONDS
        assert "it was built, but the builder could not be told" in caplog.text

    @pytest.mark.parametrize(
        ("failure", "reason"),
        [
            (ArchiveRefused("it isn't a tar archive this step can read"), "it isn't a tar archive"),
            (_refused(404, "fileset fs-a not found"), "fileset fs-a not found"),
            (OSError("No space left on device"), "No space left on device"),
        ],
        ids=["archive", "files", "disk"],
    )
    def test_a_context_that_cant_be_fetched_says_why(
        self, failure: Exception, reason: str, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            raise failure

        service = FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, fetch=fetch) == 1
        assert service.started == []
        assert f"{FAILURE_MARKER} demo-1: cannot fetch its build context: " in caplog.text
        assert reason in caplog.text

    @pytest.mark.parametrize("separator", ["\n", "\r\n", "\x0c", "\u2028", "\x85"], ids=repr)
    def test_nothing_a_submitter_chose_reaches_the_log_with_a_line_break(
        self, separator: str, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Settling reads the marked line from the log, so an archive entry's name must not be able to forge one.
        A newline is not the only character that ends a line: a form feed and a line separator do too."""

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            raise ArchiveRefused(f"entry 'a{separator}{FAILURE_MARKER} demo-1: forged' is not allowed")

        def broken(image: RemoteBuildImage, directory: Path) -> Path:
            raise RuntimeError(f"a{separator}{FAILURE_MARKER} demo-1: forged")

        with caplog.at_level(logging.INFO):
            _run(FakeService([DONE]), FakeBuilder(), context, tmp_path, fetch=fetch)
            _run(FakeService([DONE]), FakeBuilder(), context, tmp_path, fetch=broken)
        # As settling reads it: every marked line in the log as it is written, a traceback's lines included.
        found = [match.groups() for line in _marked_lines(caplog.text) if (match := _FAILURE_LINE.match(line))]
        # One line for each run, each the driver's own: the forged text is inside it, not a line of its own.
        assert [image for image, _ in found] == ["demo-1", "demo-1"]
        assert not any(reason.startswith("forged") for _, reason in found)
        assert all(reason.endswith(("is not allowed", "demo-1: forged")) for _, reason in found)
        # And whatever else reads the log, by any rule for where a line ends, finds each on one line.
        marked = [r.getMessage() for r in caplog.records if r.getMessage().startswith(FAILURE_MARKER)]
        assert [len(line.splitlines()) for line in marked] == [1, 1]

    def test_a_reason_is_cut_to_what_the_end_of_a_log_holds(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([RemoteBuildStatus("failed", "x" * 5000)])
        with caplog.at_level(logging.ERROR):
            _run(service, FakeBuilder(), context, tmp_path)
        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith(FAILURE_MARKER)]
        assert len(line) <= len(f"{FAILURE_MARKER} demo-1: ") + MAX_REASON_CHARS
        # The reason is what is cut. The build's ID, which finds the rest, is kept.
        assert line.endswith("… [build build-1]")

    def test_what_names_the_build_is_never_what_is_cut(self) -> None:
        """Even with no room at all: the reason gives way, down to a mark that it was cut."""
        assert driver._line("the service said a great deal", " [build b-1]", 20) == "the ser… [build b-1]"
        assert driver._line("the service said a great deal", " [build b-1]", 5) == "… [build b-1]"
        assert driver._line("short", " [build b-1]", 400) == "short [build b-1]"

    def test_a_credential_in_a_reason_is_taken_out_before_the_reason_is_cut(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cut first, the log would hold the start of the credential, which no filter then recognises."""
        secret = "s3cr3t-" * 100
        service = FakeService([DONE], start=[PermanentError(f"401: token {secret} is not valid")])
        with caplog.at_level(logging.ERROR):
            _run(service, FakeBuilder(), context, tmp_path, redact=driver._redactor(secret))
        assert "s3cr3t" not in caplog.text
        assert "401: token [credential] is not valid" in caplog.text

    @pytest.mark.parametrize(
        "reference",
        [
            f"elsewhere.example.com/team/images@{DIGEST}",
            f"registry.example.com/another/repository@{DIGEST}",
            f"{DESTINATION}:latest",
            f"{DESTINATION}@sha256:short",
        ],
    )
    def test_an_image_that_is_not_the_rows_is_never_completed(
        self, reference: str, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([RemoteBuildStatus("done", image=RemoteImage(reference=reference))])
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert builder.completed == []
        assert f"not a digest in {DESTINATION}" in caplog.text


class TestSeveralImages:
    def test_one_images_failure_does_not_cost_the_next(self, context: Path, tmp_path: Path) -> None:
        service = FakeService(
            [RemoteBuildStatus("failed", "scan found a critical vulnerability"), DONE], start=["build-1", "build-2"]
        )
        builder = FakeBuilder()
        assert _run(service, builder, context, tmp_path, config=_config("demo-1.a", "demo-1.b")) == 1
        assert [(image, external.build_id if external else None) for image, _, external in builder.completed] == [
            ("demo-1.b", "build-2")
        ]
        # Each image has its own key, so the service tells the two builds apart.
        assert len({build.key for build in service.started}) == 2

    def test_every_failed_images_line_is_among_the_last_the_step_logs(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Jobs keeps the end of a failed step's log. What a later image logs comes after an earlier one's line."""
        failed = RemoteBuildStatus("failed", "x" * 5000)
        service = FakeService(
            [failed, QUEUED, BUILDING, VERIFYING, failed, failed], start=["build-1", "build-2", "build-3"]
        )
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config("a", "b", "c")) == 1
        messages = [record.getMessage() for record in caplog.records]
        last = messages[messages.index("3 image(s), 3 failure(s)") + 1 :]
        assert [line.split(": ")[0] for line in last] == [f"{FAILURE_MARKER} {name}" for name in "abc"]
        assert [line.rpartition(" [")[2] for line in last] == ["build build-1]", "build build-2]", "build build-3]"]
        # All three fit in the 1,800 characters Jobs is taken to keep, with the 280 a logger and an executor put
        # around each line: 1,797 as the reasons are cut, and 2,040 if they were left at their own 400.
        assert (LINE_OVERHEAD, TAIL_CHARS) == (280, 1800)
        assert sum(len(line) - len(f"{FAILURE_MARKER} a: ") + 280 for line in last) <= 1800

    def test_a_build_is_still_named_when_many_lines_share_the_end_of_the_log(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reason is what is cut to share the space. The key that finds a build that may be running never is,
        even where that means the first lines fall outside what Jobs keeps."""
        names = [f"image-{n}" for n in range(6)]
        service = FakeService([DONE], start=[RetryableError("connection reset")])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(*names, timeout=3600)) == 1
        messages = [record.getMessage() for record in caplog.records]
        last = [line for line in messages if line.startswith(FAILURE_MARKER)][-6:]
        for name, line in zip(names, last, strict=True):
            assert line.endswith(f"started with key {idempotency_key(ATTEMPT, name)}]")

    def test_a_row_that_cant_be_read_does_not_cost_the_next(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Whatever the client raises: a step image is versioned apart from the API it reads."""
        builder = FakeBuilder(unreadable={"a": NemoClientError("the row is not a shape this step knows")})
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, config=_config("a", "b")) == 1
        assert [image for image, _, _ in builder.completed] == ["b"]
        assert f"{FAILURE_MARKER} a: cannot read its row: the row is not a shape this step knows" in caplog.text

    @pytest.mark.parametrize(("status", "exit_code"), [("ready", 0), ("failed", 1)])
    def test_a_row_already_settled_is_left_alone(
        self, status: str, exit_code: int, context: Path, tmp_path: Path
    ) -> None:
        service = FakeService([DONE])
        builder = FakeBuilder({"demo-1": status})
        assert _run(service, builder, context, tmp_path) == exit_code
        assert (service.started, builder.completed) == ([], [])


class TestTheContext:
    @pytest.mark.parametrize("dockerfile", ["Dockerfile", "./Dockerfile", "services/../Dockerfile"])
    def test_a_root_dockerfile_builds_from_the_context_as_fetched(self, dockerfile: str, context: Path) -> None:
        before = sorted(path.relative_to(context) for path in context.rglob("*"))
        assert stage_context(context, dockerfile) == context
        assert sorted(path.relative_to(context) for path in context.rglob("*")) == before
        assert (context / "Dockerfile").read_text() == "FROM scratch\n"

    def test_a_dockerfile_elsewhere_is_put_at_the_root_in_place(
        self, context: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            staged = stage_context(context, "services/feed/Dockerfile")
        # The same directory, not a copy: a copy would move the context's relative links.
        assert staged == context
        assert (context / "Dockerfile").read_text() == "FROM busybox\n"
        # The rest of the context is there, where the Dockerfile's COPY lines expect it.
        assert (context / "app.py").read_text() == "print('hello')\n"
        assert (context / "services" / "feed" / "Dockerfile").exists()
        # A build that copies its own root Dockerfile now copies the named one; the log says so.
        assert "root Dockerfile is replaced by services/feed/Dockerfile" in caplog.text

    def test_a_context_with_no_root_dockerfile_gains_one(self, context: Path) -> None:
        (context / "Dockerfile").unlink()
        stage_context(context, "services/feed/Dockerfile")
        assert (context / "Dockerfile").read_text() == "FROM busybox\n"

    def test_a_link_in_the_context_resolves_where_it_did_when_it_was_unpacked(self, context: Path) -> None:
        """Unpacking only checks that a link stays inside the context where it is. Moving the tree would move it."""
        (context / "services" / "feed" / "up").symlink_to("../../app.py")
        staged = stage_context(context, "services/feed/Dockerfile")
        assert (staged / "services" / "feed" / "up").resolve() == (context / "app.py").resolve()

    def test_a_root_dockerfile_that_is_a_link_becomes_a_file(self, context: Path) -> None:
        """A service's packager may not follow links."""
        (context / "Dockerfile").unlink()
        (context / "Dockerfile").symlink_to("services/feed/Dockerfile")
        stage_context(context, "Dockerfile")
        assert not (context / "Dockerfile").is_symlink()
        assert (context / "Dockerfile").read_text() == "FROM busybox\n"
        assert (context / "services" / "feed" / "Dockerfile").read_text() == "FROM busybox\n"

    def test_the_driver_hands_the_service_the_named_dockerfile(self, context: Path, tmp_path: Path) -> None:
        seen: list[str] = []

        class Reading(FakeService):
            def start(self, build: RemoteBuild) -> str:
                seen.append((build.context / "Dockerfile").read_text())
                return super().start(build)

        config = _config(dockerfile="services/feed/Dockerfile")
        assert _run(Reading([DONE]), FakeBuilder(), context, tmp_path, config=config) == 0
        assert seen == ["FROM busybox\n"]

    @pytest.mark.parametrize("dockerfile", ["../outside/Dockerfile", "missing/Dockerfile", "services"])
    def test_a_dockerfile_that_is_not_a_file_in_the_context_is_refused(
        self, dockerfile: str, context: Path, tmp_path: Path
    ) -> None:
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "Dockerfile").write_text("FROM secret\n")
        with pytest.raises(ImageFailed, match="not a file inside the build context"):
            stage_context(context, dockerfile)

    def test_a_dockerfile_that_links_out_of_the_context_is_refused(self, context: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside-file"
        outside.write_text("FROM secret\n")
        (context / "linked").symlink_to(outside)
        with pytest.raises(ImageFailed, match="not a file inside the build context"):
            stage_context(context, "linked")

    @pytest.mark.parametrize("missing", ["the-dockerfile", "the-context"])
    def test_a_default_dockerfile_that_isnt_there_is_refused_before_the_service_is_asked(
        self, missing: str, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A context path that matches nothing in the archive, or a context with no Dockerfile."""
        (context / "Dockerfile").unlink()
        fetched = context if missing == "the-dockerfile" else tmp_path / "no-such-directory"
        service = FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), fetched, tmp_path) == 1
        assert service.started == []
        assert "dockerfile 'Dockerfile' is not a file inside the build context" in caplog.text

    def test_a_directory_where_the_dockerfile_goes_is_refused(self, context: Path) -> None:
        (context / "Dockerfile").unlink()
        (context / "Dockerfile").mkdir()
        with pytest.raises(ImageFailed, match="a directory named Dockerfile at its root"):
            stage_context(context, "services/feed/Dockerfile")

    def test_a_dockerfiles_own_ignore_file_goes_to_the_root_with_it(self, context: Path) -> None:
        """A builder reads `<Dockerfile>.dockerignore` before the context's `.dockerignore`. Left behind, a file
        the build excluded where it was written, a `private.env` say, would be sent and could be copied in."""
        (context / "private.env").write_text("sample only\n")
        (context / ".dockerignore").write_text("*.md\n")
        (context / "Dockerfile.dockerignore").write_text("services\n")
        (context / "services" / "feed" / "Dockerfile.dockerignore").write_text("private.env\n")

        staged = stage_context(context, "services/feed/Dockerfile")

        assert (staged / "Dockerfile.dockerignore").read_text() == "private.env\n"
        # The context's own, which a builder falls back to, is as it was.
        assert (staged / ".dockerignore").read_text() == "*.md\n"

    def test_the_replaced_root_dockerfiles_ignore_file_does_not_stand_in_for_one(self, context: Path) -> None:
        """The named Dockerfile has no ignore file of its own, so the context's `.dockerignore` is what applies
        to it. The one beside the root Dockerfile it replaces was never written for it."""
        (context / ".dockerignore").write_text("*.md\n")
        (context / "Dockerfile.dockerignore").write_text("services\n")

        staged = stage_context(context, "services/feed/Dockerfile")

        assert not (staged / "Dockerfile.dockerignore").exists()
        assert (staged / ".dockerignore").read_text() == "*.md\n"

    def test_a_dockerfile_already_at_the_root_keeps_its_ignore_file(self, context: Path) -> None:
        (context / "Dockerfile.dockerignore").write_text("services\n")
        assert stage_context(context, "Dockerfile") == context
        assert (context / "Dockerfile.dockerignore").read_text() == "services\n"

    def test_a_root_dockerfile_under_another_name_brings_its_ignore_file(self, context: Path) -> None:
        (context / "Dockerfile.dev").write_text("FROM alpine\n")
        (context / "Dockerfile.dev.dockerignore").write_text("secrets\n")
        staged = stage_context(context, "Dockerfile.dev")
        assert (staged / "Dockerfile").read_text() == "FROM alpine\n"
        assert (staged / "Dockerfile.dockerignore").read_text() == "secrets\n"

    @pytest.mark.parametrize("kind", ["a-file", "a-directory", "by-an-absolute-path"])
    def test_a_context_with_a_link_that_leads_out_of_it_is_refused(
        self, kind: str, context: Path, tmp_path: Path
    ) -> None:
        """A packager that follows links would send what is beside the context along with it."""
        (tmp_path / "beside").mkdir()
        (tmp_path / "beside" / "answer.txt").write_text("not this build's\n")
        target = {
            "a-file": "../../beside/answer.txt",
            "a-directory": "../../beside",
            "by-an-absolute-path": str(tmp_path / "beside" / "answer.txt"),
        }[kind]
        (context / "services" / "fixture").symlink_to(target)
        with pytest.raises(ImageFailed, match="a link that leads out of it: 'services/fixture'"):
            stage_context(context, "Dockerfile")

    def test_links_that_go_round_in_a_loop_lead_nowhere_and_are_left(self, context: Path) -> None:
        """They reach no file, inside the context or out of it."""
        (context / "one").symlink_to("two")
        (context / "two").symlink_to("one")
        assert stage_context(context, "Dockerfile") == context

    def test_links_that_stay_inside_the_context_are_left_alone(self, context: Path) -> None:
        (context / "services" / "feed" / "up").symlink_to("../../app.py")
        (context / "all-services").symlink_to("services")
        staged = stage_context(context, "services/feed/Dockerfile")
        assert (staged / "all-services" / "feed" / "up").read_text() == "print('hello')\n"


class TestTheImagesOfOneSet:
    """A set's images are built one after another, and a source can change in between."""

    @staticmethod
    def _sources(tmp_path: Path, files: dict[str, str]) -> Callable[[RemoteBuildImage, Path], Path]:
        """A fetch that writes ``files`` as they are when it is called, and counts its calls."""

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            fetch.calls.append(image.source.fileset)  # ty: ignore[unresolved-attribute]
            for name, content in files.items():
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_text(content)
            return directory

        fetch.calls = []  # ty: ignore[unresolved-attribute]
        return fetch

    def test_every_image_is_built_from_the_source_as_it_was_first_fetched(self, tmp_path: Path) -> None:
        files = {"Dockerfile": "FROM scratch\n", "version": "first"}
        fetch = self._sources(tmp_path, files)
        seen: list[str] = []

        class Changing(FakeService):
            def start(self, build: RemoteBuild) -> str:
                seen.append((build.context / "version").read_text())
                files["version"] = "second"
                return super().start(build)

        builder = FakeBuilder()
        assert _run(Changing([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=_config("a", "b")) == 0
        assert seen == ["first", "first"]
        assert fetch.calls == ["default/fs-a"]  # ty: ignore[unresolved-attribute]
        assert [image for image, _, _ in builder.completed] == ["a", "b"]

    def test_each_image_stages_its_own_copy(self, tmp_path: Path) -> None:
        """Staging moves a Dockerfile to the root of the context. One image's must not become the next one's."""
        files = {"Dockerfile": "FROM root\n", "one/Dockerfile": "FROM one\n", "two/Dockerfile": "FROM two\n"}
        config = RemoteBuildStepConfig.model_validate(
            {
                "images": [
                    {
                        "image": name,
                        "source": {"fileset": "default/fs-a"},
                        "dockerfile": path,
                        "platform": "linux/amd64",
                    }
                    for name, path in (("a", "one/Dockerfile"), ("b", "Dockerfile"), ("c", "two/Dockerfile"))
                ],
                "registry": "registry.example.com",
                "repository": "team/images",
                "build_timeout_seconds": 600,
            }
        )
        built: list[str] = []

        class Reading(FakeService):
            def start(self, build: RemoteBuild) -> str:
                built.append((build.context / "Dockerfile").read_text())
                return super().start(build)

        assert (
            _run(
                Reading([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=self._sources(tmp_path, files), config=config
            )
            == 0
        )
        assert built == ["FROM one\n", "FROM root\n", "FROM two\n"]

    def test_a_set_of_one_image_is_built_where_it_was_fetched(self, context: Path, tmp_path: Path) -> None:
        """Nothing to share, so no second copy of what may be a large context."""
        service = FakeService([DONE])
        assert _run(service, FakeBuilder(), context, tmp_path) == 0
        assert service.started[0].context == context


class TestIdentity:
    def test_the_key_is_stable_in_an_attempt_and_new_in_a_rerun_another_job_or_another_image(self) -> None:
        key = idempotency_key(ATTEMPT, "demo-1")
        assert key == idempotency_key(ATTEMPT, "demo-1")
        assert len(key) == 64
        others = [
            idempotency_key(JobAttempt("default", "job-uid-1", "attempt-2"), "demo-1"),
            idempotency_key(JobAttempt("default", "job-uid-2", "attempt-1"), "demo-1"),
            idempotency_key(ATTEMPT, "demo-2"),
        ]
        assert len({key, *others}) == 4

    def test_the_attempt_comes_from_what_jobs_sets(self) -> None:
        environ = {"NEMO_JOB_WORKSPACE": "default", "NEMO_JOB_UID": "job-uid-1", "NEMO_JOB_ATTEMPT_ID": "attempt-1"}
        assert JobAttempt.from_environment(environ) == ATTEMPT
        with pytest.raises(RuntimeError, match="NEMO_JOB_ATTEMPT_ID not set"):
            JobAttempt.from_environment({"NEMO_JOB_WORKSPACE": "default", "NEMO_JOB_UID": "job-uid-1"})

    def test_the_credential_is_taken_out_of_the_environment(self) -> None:
        environ = {REMOTE_CREDENTIAL_ENV: " token-value \n", "PATH": "/bin"}
        assert take_credential(environ) == "token-value"
        assert environ == {"PATH": "/bin"}
        with pytest.raises(RuntimeError, match="did not reach this step"):
            take_credential(environ)


@pytest.fixture
def signals() -> Iterator[None]:
    """``step_main`` installs signal handlers, and takes the credential out of every record logged from then on;
    put the test runner's handlers and its way of making records back."""
    saved = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    make_record = logging.getLogRecordFactory()
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)
    logging.setLogRecordFactory(make_record)


class TestTheStepProgram:
    """``step_main`` itself: the step's config file, what Jobs sets, the credential, and the signals."""

    @pytest.fixture
    def step(self, context: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signals: None) -> dict[str, list]:
        config = _config().model_copy(update={"credential_in_environment": True})
        path = tmp_path / "job_step_config.json"
        path.write_text(json.dumps(config.model_dump(mode="json")))
        storage = tmp_path / "task-storage"
        storage.mkdir()
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(path))
        monkeypatch.setenv("NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH", str(storage))
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "default")
        monkeypatch.setenv("NEMO_JOB_UID", "job-uid-1")
        monkeypatch.setenv("NEMO_JOB_ATTEMPT_ID", "attempt-1")
        monkeypatch.setenv(REMOTE_CREDENTIAL_ENV, "token-value")
        seen: dict[str, list] = {"platform": [], "fetched": [], "builder": []}
        builder = FakeBuilder()
        seen["builder"].append(builder)

        def fetch_context(files: object, image: RemoteBuildImage, *, workspace: str, root: Path) -> Path:
            seen["fetched"].append((workspace, root))
            return context

        monkeypatch.setattr(driver, "get_task_nemo_client", lambda name: seen["platform"].append(name) or object())
        monkeypatch.setattr(driver, "client_from_platform", lambda platform, kind: object())
        monkeypatch.setattr(driver, "Builder", lambda client, *, workspace: builder)
        monkeypatch.setattr(driver, "fetch_context", fetch_context)
        monkeypatch.setattr(driver, "POLL_SECONDS", 0.01)
        return seen

    def test_it_builds_what_its_config_names_as_the_builder_for_the_submitter(
        self, step: dict[str, list], tmp_path: Path
    ) -> None:
        made: list[tuple[RemoteBuildStepConfig, str | None]] = []
        service = FakeService([BUILDING, DONE])

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            made.append((config, credential))
            return service

        assert step_main(make_service) == 0

        ((config, credential),) = made
        assert (config.destination, credential) == (DESTINATION, "token-value")
        # Taken out of the environment, so nothing the step runs inherits it.
        assert REMOTE_CREDENTIAL_ENV not in os.environ
        assert step["platform"] == ["builder"]
        (builder,) = step["builder"]
        assert [image for image, _, _ in builder.completed] == ["demo-1"]
        assert service.started[0].key == idempotency_key(ATTEMPT, "demo-1")
        # Fetched onto the step's task storage, into a directory that is gone once the image is done.
        ((workspace, root),) = step["fetched"]
        assert workspace == "default"
        assert (tmp_path / "task-storage") in root.parents
        assert list((tmp_path / "task-storage").iterdir()) == []

    def test_a_signal_from_jobs_cancels_the_build_without_waiting_out_a_sleep(
        self, step: dict[str, list], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pod has about 30 seconds from the signal, and a signal doesn't cut a plain sleep short."""

        class Signalled(FakeService):
            def status(self, build_id: str) -> RemoteBuildStatus:
                if not self.polled:
                    os.kill(os.getpid(), signal.SIGTERM)
                return super().status(build_id)

        monkeypatch.setattr(driver, "POLL_SECONDS", 30.0)
        service = Signalled([BUILDING])
        began = time.monotonic()
        assert step_main(lambda config, credential: service) == 1
        assert time.monotonic() - began < 5
        assert service.cancelled == ["build-1"]
        (builder,) = step["builder"]
        assert builder.completed == []

    def test_the_credential_stays_out_of_what_the_step_logs(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[PermanentError("401: token-value was refused")])
        with caplog.at_level(logging.INFO):
            assert step_main(lambda config, credential: service) == 1
        assert "token-value" not in caplog.text
        assert f"{FAILURE_MARKER} demo-1: starting the build: 401: [credential] was refused" in caplog.text

    def test_the_credential_stays_out_of_every_line_not_only_the_reason(
        self, step: dict[str, list], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A call that is tried again logs why, and a build's stage is logged as the service words it. Neither
        is a reason, so only the filter on the log itself keeps the credential out of them."""
        monkeypatch.setattr(driver, "_backoff", lambda tries: 0.01)
        echoing = RemoteBuildStatus("building", "authorised as token-value")
        service = FakeService([echoing, DONE], start=[OverBudgetError("busy, said to token-value"), "build-1"])
        with caplog.at_level(logging.INFO):
            assert step_main(lambda config, credential: service) == 0
        assert "token-value" not in caplog.text
        assert "starting the build failed (busy, said to [credential])" in caplog.text
        assert "build build-1: building (authorised as [credential])" in caplog.text

    def test_a_client_that_cant_be_set_up_fails_every_image_and_says_why(
        self, step: dict[str, list], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            raise ValueError(f"no endpoint is configured for {credential}")

        with caplog.at_level(logging.ERROR):
            assert step_main(make_service) == 1
        assert "token-value" not in caplog.text
        assert (
            f"{FAILURE_MARKER} demo-1: the step could not start: ValueError: no endpoint is configured for [credential]"
        ) in caplog.text
        # The step's own directory is made last, so a step that fails before it leaves nothing on task storage.
        assert list((tmp_path / "task-storage").iterdir()) == []

    def test_a_step_that_cant_start_says_why_on_each_images_line(
        self, step: dict[str, list], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A credential that never reached the step is a likely first-deployment mistake. Without the line, the
        row would show only what Jobs says of any failed pod."""
        monkeypatch.delenv(REMOTE_CREDENTIAL_ENV)
        with caplog.at_level(logging.ERROR):
            assert step_main(lambda config, credential: FakeService([DONE])) == 1
        assert f"{FAILURE_MARKER} demo-1: the step could not start: RuntimeError: {REMOTE_CREDENTIAL_ENV} not set" in (
            caplog.text
        )

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores a directory's mode")
    def test_a_step_with_nowhere_to_put_a_context_says_so_on_each_images_line(
        self, step: dict[str, list], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Task storage the pod's user can't write to is a mistake in a hand-written profile, and is found only
        when the step starts."""
        storage = tmp_path / "task-storage"
        storage.chmod(0o500)
        try:
            with caplog.at_level(logging.ERROR):
                assert step_main(lambda config, credential: FakeService([DONE])) == 1
        finally:
            storage.chmod(0o700)
        assert f"{FAILURE_MARKER} demo-1: the step could not start: PermissionError" in caplog.text

    def test_a_credential_the_profile_mounts_is_kept_out_of_the_log_too(
        self,
        context: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        signals: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The step program reads the mounted file and hands the driver the way to; the driver then treats the
        credential as it does one from the environment."""
        path = tmp_path / "job_step_config.json"
        path.write_text(json.dumps(_config().model_dump(mode="json")))
        mounted = tmp_path / "token"
        mounted.write_text("mounted-token\n")
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(path))
        monkeypatch.setenv("NEMO_JOB_WORKSPACE", "default")
        monkeypatch.setenv("NEMO_JOB_UID", "job-uid-1")
        monkeypatch.setenv("NEMO_JOB_ATTEMPT_ID", "attempt-1")
        monkeypatch.setattr(driver, "get_task_nemo_client", lambda name: object())
        monkeypatch.setattr(driver, "client_from_platform", lambda platform, kind: object())
        monkeypatch.setattr(driver, "Builder", lambda client, *, workspace: FakeBuilder())
        monkeypatch.setattr(driver, "fetch_context", lambda files, image, *, workspace, root: context)
        given: list[str | None] = []

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            given.append(credential)
            return FakeService([DONE], start=[PermanentError("401: mounted-token was refused")])

        with caplog.at_level(logging.INFO):
            assert step_main(make_service, read_credential=mounted.read_text) == 1
        assert given == ["mounted-token"]
        assert "mounted-token" not in caplog.text
        assert f"{FAILURE_MARKER} demo-1: starting the build: 401: [credential] was refused" in caplog.text

    def test_a_sleep_is_taken_in_slices_and_ends_when_the_job_is_told_to_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slept(seconds: float, cancelled: Callable[[FakeClock], bool]) -> list[float]:
            fake = FakeClock()
            monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=fake.monotonic, sleep=fake.sleep))
            sleep_unless(lambda: cancelled(fake))(seconds)
            return fake.sleeps

        assert slept(2.5, lambda fake: False) == [1.0, 1.0, 0.5]
        # Told to stop during the first slice: the other twenty-nine seconds are not slept.
        assert slept(30.0, lambda fake: fake.now >= 1.0) == [1.0]
        assert slept(30.0, lambda fake: True) == []

    @pytest.mark.parametrize("line_break", ["\n", "\r\n"], ids=["lf", "crlf"])
    def test_a_credential_of_more_than_one_line_stays_out_of_a_reason_and_of_the_log(
        self, line_break: str, step: dict[str, list], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A key file. A reason turns its line breaks into spaces and a log line drops them, so the credential
        is in neither as it was given."""
        secret = f"-----BEGIN KEY-----{line_break}c2VjcmV0{line_break}-----END KEY-----"
        monkeypatch.setenv(REMOTE_CREDENTIAL_ENV, secret)
        monkeypatch.setattr(driver, "_backoff", lambda tries: 0.01)
        refusals: list[str | Exception] = [OverBudgetError(f"busy, said to {secret}"), PermanentError(f"401: {secret}")]
        service = FakeService([DONE], start=refusals)
        with caplog.at_level(logging.INFO):
            assert step_main(lambda config, credential: service) == 1
        assert "c2VjcmV0" not in caplog.text
        assert f"{FAILURE_MARKER} demo-1: starting the build: 401: [credential]" in caplog.text
        assert "starting the build failed (busy, said to [credential])" in caplog.text

    def test_a_credential_of_more_than_one_line_stays_out_of_why_a_step_could_not_start(
        self, step: dict[str, list], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(REMOTE_CREDENTIAL_ENV, "first-half\nsecond-half")

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            raise ValueError(f"no endpoint is configured for {credential}")

        with caplog.at_level(logging.ERROR):
            assert step_main(make_service) == 1
        assert "first-half" not in caplog.text and "second-half" not in caplog.text
        assert "no endpoint is configured for [credential]" in caplog.text

    def test_the_credential_stays_out_of_a_traceback_a_client_logs(self) -> None:
        """A formatter prints the exception after the message, with the exception's own text."""
        try:
            raise ValueError("refused: Bearer token-value")
        except ValueError:
            record = logging.LogRecord("x", logging.ERROR, __file__, 1, "the call failed", None, sys.exc_info())
        record.stack_info = "Stack (most recent call last):\n  authorise('token-value')"
        driver._redact_record(record, driver._redactor("token-value"))
        printed = logging.Formatter().format(record)
        assert "token-value" not in printed
        assert "ValueError: refused: Bearer [credential]" in printed
        assert "authorise('[credential]')" in printed

    def test_the_credential_never_reaches_the_log(self, caplog: pytest.LogCaptureFixture) -> None:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "refused: %s", ("Bearer token-value",), None)
        driver._redact_record(record, driver._redactor("token-value"))
        assert record.getMessage() == "refused: Bearer [credential]"

    def test_a_record_with_nothing_to_take_out_is_left_as_it_was_logged(self) -> None:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "build %s: %s", ("build-1", "queued"), None)
        driver._redact_record(record, driver._redactor("token-value"))
        assert (record.msg, record.args) == ("build %s: %s", ("build-1", "queued"))

    def test_a_log_call_that_doesnt_fit_its_message_neither_raises_nor_prints_the_credential(self) -> None:
        """Logging never raises into the code that logs. A handler would report such a call by printing its
        message and arguments as they are."""
        record = logging.LogRecord("x", logging.WARNING, __file__, 1, "try %d of %d", ("token-value",), None)
        driver._redact_record(record, driver._redactor("token-value"))
        assert logging.Formatter().format(record) == "try %d of %d (logged with ('[credential]',))"

    @pytest.mark.parametrize(
        "quoted",
        [
            "Invalid header value b'Bearer dXNlcjpw\\nYXNz'",
            'body was {"token": "dXNlcjpw\\nYXNz"}',
            "GET /builds?key=dXNlcjpw%0AYXNz failed",
            "line 2 of the key is malformed: YXNzd29yZA==",
        ],
        ids=["repr", "json", "url", "one-line-of-it"],
    )
    def test_a_credential_an_error_quotes_or_escapes_is_taken_out_too(self, quoted: str) -> None:
        """An HTTP client that refuses a header prints it with its line break escaped."""
        redact = driver._redactor("dXNlcjpw\nYXNz" if "YXNzd29yZA" not in quoted else "dXNlcjpw\nYXNzd29yZA==")
        assert "YXNz" not in redact(quoted)
        assert "[credential]" in redact(quoted)

    def test_taking_a_credential_out_twice_changes_nothing_more(self) -> None:
        """A reason is redacted before and after it is made one line, and a short credential may be part of the
        mark that takes its place."""
        redact = driver._redactor("cred")
        once = redact("refused: cred is not known")
        assert once == "refused: [credential] is not known"
        assert redact(once) == once

    def test_the_fields_a_log_call_attaches_are_redacted_too(self, signals: None) -> None:
        """A client that logs a failed request with its headers. `extra` fields reach a record after it is made,
        and a formatter that prints one would print the credential in it."""
        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        handler = Own()
        handler.setFormatter(logging.Formatter("%(message)s %(headers)s %(attempt)s %(sent)s"))
        client_log = logging.getLogger("some.build.client.requests")
        client_log.propagate = False
        client_log.addHandler(handler)
        driver.redact_log("token-value")
        try:
            headers = {"Authorization": "Bearer token-value", "Accept": ["application/json"]}
            client_log.error("request failed", extra={"headers": headers, "attempt": 2, "sent": b"key=token-value"})
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == [
            "request failed {'Authorization': 'Bearer [credential]', 'Accept': ['application/json']} 2 "
            "b'key=[credential]'"
        ]
        # The caller's own dictionary is not changed under it.
        assert headers["Authorization"] == "Bearer token-value"

    def test_a_logged_value_without_the_credential_is_passed_on_as_it_is(self) -> None:
        redact = driver._redactor("token-value")
        untouched = {"Accept": ["application/json"], "tries": 3}
        assert driver._redact_value(untouched, redact) is untouched

        class Request:
            def __repr__(self) -> str:
                return "<Request Authorization: Bearer token-value>"

        assert driver._redact_value(Request(), redact) == "<Request Authorization: Bearer [credential]>"
        nested: list[object] = ["token-value"]
        for _ in range(10):
            nested = [nested]
        assert "token-value" not in repr(driver._redact_value(nested, redact))

    def test_every_record_is_redacted_whichever_logger_and_handler_it_goes_through(
        self, signals: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A client may log through a logger that doesn't pass its records up, to a handler of its own."""
        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        client_log = logging.getLogger("some.build.client")
        client_log.propagate = False
        driver.redact_log("token-value")
        handler = Own()
        client_log.addHandler(handler)
        try:
            client_log.warning("sent Authorization: Bearer %s", "token-value")
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == ["sent Authorization: Bearer [credential]"]
