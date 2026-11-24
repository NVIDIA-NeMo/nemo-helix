# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote-build step's driver, against a fake build service, a fake builder and a fake clock."""

from __future__ import annotations

import collections
import errno
import filecmp
import io
import itertools
import json
import logging
import os
import pickle
import shutil
import signal
import stat
import sys
import tarfile
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import FrameType, SimpleNamespace
from typing import Any, NamedTuple, TypeVar, cast

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.run import fetch as fetch_step
from nemo_builder_plugin.run import remote as driver
from nemo_builder_plugin.run.fetch import MAX_UNPACKED_BYTES, MAX_UNPACKED_ENTRIES, ArchiveRefused, unpack
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
    RemoteContext,
    RemoteImage,
    RetryableError,
    fetch_context,
    idempotency_key,
    run_remote_build,
    sleep_unless,
    stage_context,
    step_main,
    take_credential,
)
from nemo_builder_plugin.settle import _FAILURE_LINE, _marked_lines
from nemo_builder_plugin.steps import (
    REMOTE_CREDENTIAL_ENV,
    ContextSource,
    RemoteBuildImage,
    RemoteBuildStepConfig,
    WorkLayout,
)
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError
from nemo_helix_plugin.files.client import FilesClient

DESTINATION = "registry.example.com/team/images"
#: What follows a reason when the build was to be cancelled and the service couldn't be told.
UNCANCELLED = "[build build-1; the build service could not be told to stop it, it may still be running]"
DIGEST = "sha256:" + "d" * 64
ATTEMPT = JobAttempt(workspace="default", job_uid="job-uid-1", attempt_id="attempt-1")

QUEUED = RemoteBuildStatus("queued")
BUILDING = RemoteBuildStatus("building", "step 2 of 5")
VERIFYING = RemoteBuildStatus("verifying", "step 4 of 5")
DONE = RemoteBuildStatus(
    "done", image=RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", attributes={"builder": "v1"})
)

_T = TypeVar("_T")


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


class _ListedFirst:
    """The driver's ``os``, whose walk is given each directory's subdirectories with one name put first. A file
    system lists a directory's entries in an order of its own, which a test on one disk can't choose."""

    def __init__(self, first: str) -> None:
        self._first = first

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def walk(self, top: Path) -> Iterator[tuple[str, list[str], list[str]]]:
        for directory, directories, files in os.walk(top):
            # In place: the walk goes on into them in the order this list is left in.
            directories.sort(key=lambda name: name != self._first)
            yield directory, directories, files


class LateClock(FakeClock):
    """A clock whose sleeps end later than they were asked to, as a sleep does on a node that is busy."""

    def __init__(self, late: float) -> None:
        super().__init__()
        #: How long after its time each sleep ends.
        self.late = late

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.now += self.late


class FakeService:
    """Answers ``upload_context``, ``start``, ``status`` and ``cancel`` from scripts. The last entry of a script
    repeats."""

    def __init__(
        self,
        statuses: list[RemoteBuildStatus | Exception],
        *,
        upload: list[str | Exception] | None = None,
        start: list[str | Exception] | None = None,
        cancel: Exception | list[Exception | None] | None = None,
    ) -> None:
        self._statuses = list(statuses)
        self._upload: list[str | Exception] = list(upload or ["context-1"])
        self.uploaded: list[RemoteContext] = []
        self._start: list[str | Exception] = list(start or ["build-1"])
        #: What each ``cancel`` raises; None is a cancel the service takes.
        self._cancel: list[Exception | None] = list(cancel) if isinstance(cancel, list) else [cancel]
        self.started: list[RemoteBuild] = []
        self.polled: list[str] = []
        self.cancelled: list[str] = []

    @staticmethod
    def _next(script: list[_T]) -> _T:
        return script.pop(0) if len(script) > 1 else script[0]

    def upload_context(self, context: RemoteContext) -> str:
        self.uploaded.append(context)
        answer = self._next(self._upload)
        if isinstance(answer, Exception):
            raise answer
        return answer

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
        runtime_config: ImageRuntimeConfig | None = None,
    ) -> None:
        #: The status and digest ``complete`` answers with, when not ``ready`` at the digest it was given.
        self.completes_as = completes_as
        #: How ``complete`` is to be told the image starts: nothing, unless the service's client reports it.
        self.runtime_config = runtime_config
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
        assert runtime_config == self.runtime_config, "the row records how the image starts as the client reports it"
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


def _unpacked_elsewhere(context: Path, destination: Path) -> Path:
    """``context`` as a packager that keeps its links sends it, unpacked in ``destination``, under another
    name, by an extractor that refuses a link out of where it unpacks."""
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w") as archive:
        archive.add(context, arcname=".")
    packed.seek(0)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=packed) as archive:
        archive.extractall(destination, filter="data")
    return destination


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
        (upload,) = service.uploaded
        assert (upload.key, upload.path, upload.labels) == (
            idempotency_key(ATTEMPT, "demo-1"),
            context,
            {"workspace": "default", "image": "demo-1", "job": "job-uid-1"},
        )
        (build,) = service.started
        assert build.context == "context-1"
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

    def test_the_credential_is_taken_out_of_what_the_service_reports_for_the_row(
        self, context: Path, tmp_path: Path
    ) -> None:
        """Every reader of the image reads these. A service may report a link to its console, with a key in it."""
        attributes = {"console": "https://build.example.com/b/42?key=token-value"}
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", attributes=attributes)
        builder = FakeBuilder()
        service = FakeService([RemoteBuildStatus("done", image=image)])
        assert _run(service, builder, context, tmp_path, redact=driver._redactor("token-value")) == 0
        ((_, _, external),) = builder.completed
        assert external == ExternalImage(
            image_id="img-1",
            build_id="build-1",
            attributes={"console": "https://build.example.com/b/42?key=[credential]"},
        )

    def test_how_the_image_starts_is_recorded_when_the_services_client_reports_it(
        self, context: Path, tmp_path: Path
    ) -> None:
        """A runtime that replaces a container's entrypoint starts the image's own, which only its config holds."""
        reported = ImageRuntimeConfig(entrypoint=["/entrypoint.sh"], cmd=["serve"], working_dir="/app", user="")
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", runtime_config=reported)
        builder = FakeBuilder(runtime_config=reported)
        assert _run(FakeService([RemoteBuildStatus("done", image=image)]), builder, context, tmp_path) == 0
        assert [image for image, _, _ in builder.completed] == ["demo-1"]

    @pytest.mark.parametrize(
        "reported",
        [
            ImageRuntimeConfig(entrypoint=["/entrypoint.sh", "--token", "token-value"], cmd=["serve"], user=""),
            ImageRuntimeConfig(entrypoint=["/bin/sh", "-c"], cmd=["serve --key token-value"], user="1000"),
            ImageRuntimeConfig(entrypoint=["/entrypoint.sh"], working_dir="/srv/token-value"),
            ImageRuntimeConfig(entrypoint=["/entrypoint.sh"], user="token-value"),
        ],
        ids=["in-its-entrypoint", "in-its-command", "in-its-working-directory", "in-its-user"],
    )
    def test_how_an_image_starts_is_not_recorded_when_any_part_of_it_holds_the_credential(
        self, reported: ImageRuntimeConfig, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """How an image starts is what a consumer runs. With the credential taken out of an argument, what is
        left is not how the image starts: a consumer would start it with the mark in the argument's place."""
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", runtime_config=reported)
        # A builder that is told nothing of how the image starts: `complete` checks that.
        builder = FakeBuilder()
        service = FakeService([RemoteBuildStatus("done", image=image)])
        with caplog.at_level(logging.INFO):
            assert _run(service, builder, context, tmp_path, redact=driver._redactor("token-value")) == 0
        # The image is built and published all the same, and the log says why its row holds no way to start it.
        assert len(builder.completed) == 1
        assert "not recording how demo-1 starts: it holds the build service's credential" in caplog.text
        assert "token-value" not in caplog.text

    def test_how_an_image_starts_is_recorded_as_reported_when_nothing_is_taken_out_of_it(
        self, context: Path, tmp_path: Path
    ) -> None:
        """A credential the step knows changes nothing of a configuration that doesn't hold it."""
        reported = ImageRuntimeConfig(entrypoint=["/bin/sh", "-c"], cmd=["serve --key $KEY"], working_dir="/srv")
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", runtime_config=reported)
        builder = FakeBuilder(runtime_config=reported)
        service = FakeService([RemoteBuildStatus("done", image=image)])
        assert _run(service, builder, context, tmp_path, redact=driver._redactor("token-value")) == 0
        assert len(builder.completed) == 1

    def test_how_an_image_starts_that_cant_be_recorded_does_not_cost_the_image(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """As the push step treats a config it can't read: the image is built and published all the same."""
        reported = ImageRuntimeConfig.model_construct(entrypoint=["/entrypoint.sh"], user="u" * 300)
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", runtime_config=reported)
        builder = FakeBuilder()
        with caplog.at_level(logging.WARNING):
            assert _run(FakeService([RemoteBuildStatus("done", image=image)]), builder, context, tmp_path) == 0
        assert len(builder.completed) == 1
        assert "can't record how demo-1 starts; recording none" in caplog.text

    def test_each_stage_is_logged_once(self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            _run(FakeService([QUEUED, BUILDING, BUILDING, VERIFYING, DONE]), FakeBuilder(), context, tmp_path)
        stages = [record.getMessage() for record in caplog.records if record.getMessage().startswith("build build-1")]
        assert stages == [
            "build build-1: queued",
            "build build-1: building (step 2 of 5)",
            "build build-1: verifying (step 4 of 5)",
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

    def test_a_completion_tried_again_says_the_same_of_how_the_image_starts(
        self, context: Path, tmp_path: Path
    ) -> None:
        """The builder refuses a second completion of a remote image that says it starts another way, or names
        another build. So what the step works out for an image, it sends on every try."""
        reported = ImageRuntimeConfig(entrypoint=["/entrypoint.sh"], cmd=["serve"], working_dir="/app", user="")
        image = RemoteImage(reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", runtime_config=reported)
        #: What each call to `complete` carried, the ones that failed included.
        sent: list[tuple[ImageRuntimeConfig | None, ExternalImage | None]] = []

        class Noting(FakeBuilder):
            def complete(
                self,
                image: str,
                digest: str,
                runtime_config: ImageRuntimeConfig | None,
                external: ExternalImage | None = None,
            ) -> ContainerImage:
                sent.append((runtime_config, external))
                return super().complete(image, digest, runtime_config, external)

        builder = Noting(failing=[_unreachable(), _refused(503, "restarting")], runtime_config=reported)
        assert _run(FakeService([RemoteBuildStatus("done", image=image)]), builder, context, tmp_path) == 0
        assert sent == [(reported, ExternalImage(image_id="img-1", build_id="build-1"))] * 3

    def test_an_api_pod_from_before_the_handle_existed_is_tried_past(self, context: Path, tmp_path: Path) -> None:
        """Mid-upgrade, an older pod refuses the request's shape. The next try may reach a newer one."""
        builder = FakeBuilder(failing=[_refused(422, "external: extra fields not permitted")])
        clock = FakeClock()
        assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 0
        assert (len(builder.completed), clock.sleeps) == (1, [2.0])


class TestUploadingTheContext:
    """The context goes to the service in a step of its own, before any build starts from it."""

    def test_the_context_is_uploaded_before_the_build_starts_from_it(self, context: Path, tmp_path: Path) -> None:
        calls: list[str] = []

        class Ordered(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                calls.append("upload")
                return super().upload_context(context)

            def start(self, build: RemoteBuild) -> str:
                calls.append(f"start from {build.context}")
                return super().start(build)

        service = Ordered([DONE], upload=["uploaded-context-7"])
        assert _run(service, FakeBuilder(), context, tmp_path) == 0
        assert calls == ["upload", "start from uploaded-context-7"]
        (upload,), (build,) = service.uploaded, service.started
        # One key for both steps: the service can tie a build to the upload it was started from.
        assert upload.key == build.key and upload.labels == build.labels

    def test_a_lost_response_repeats_the_same_upload(self, context: Path, tmp_path: Path) -> None:
        service = FakeService([DONE], upload=[RetryableError("connection reset"), "context-1"])
        builder = FakeBuilder()

        assert _run(service, builder, context, tmp_path) == 0

        first, second = service.uploaded
        assert first == second
        assert len(service.started) == 1 and len(builder.completed) == 1

    @pytest.mark.parametrize(
        ("upload", "reason"),
        [
            ([PermanentError("413: context too large")], "uploading the build context: 413: context too large"),
            ([RetryableError("connection reset"), KeyError("id")], "uploading the build context: unexpected KeyError"),
            (["  "], "the build service's client uploaded the build context without naming it"),
            ([""], "the build service's client uploaded the build context without naming it"),
            ([None], "the build service's client uploaded the build context without naming it"),
        ],
        ids=["refused", "then-a-bug", "blank-reference", "empty-reference", "no-reference"],
    )
    def test_an_upload_that_fails_starts_nothing_and_cancels_nothing(
        self,
        upload: list[str | Exception],
        reason: str,
        context: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        service = FakeService([DONE], upload=upload)
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path) == 1
        assert service.started == [] and service.cancelled == [] and builder.completed == []
        assert f"{FAILURE_MARKER} demo-1: {reason}" in caplog.text
        # No build was started, so there is none to look for by its key.
        assert "a build may be running" not in caplog.text

    def test_an_upload_that_keeps_failing_is_given_up_after_the_patience_for_a_start(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], upload=[RetryableError("connection reset")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=3600), clock=clock) == 1
        # Given up on well before the build's own hour, and not after a few tries.
        assert START_PATIENCE_SECONDS - MAX_BACKOFF_SECONDS <= clock.now <= START_PATIENCE_SECONDS
        tries = len(service.uploaded)
        assert (
            f"{FAILURE_MARKER} demo-1: uploading the build context failed {tries} times over {clock.now:.0f}s"
            in caplog.text
        )
        assert service.started == []

    def test_an_upload_that_is_kept_waiting_is_given_up_at_the_build_timeout(
        self, context: Path, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        service = FakeService([DONE], upload=[OverBudgetError("busy")])
        assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=60), clock=clock) == 1
        # Five tries, at 0, 2, 6, 14 and 30 seconds. The next wait, 32, would pass the minute.
        assert (len(service.uploaded), clock.now, service.started) == (5, 30.0, [])

    def test_an_upload_that_took_all_of_the_builds_time_starts_nothing(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The upload succeeds, on its first try, after the build's time has passed. A build started then
        would only be cancelled: none is started, and there is none to read or cancel."""
        clock = FakeClock()

        class SlowToUpload(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                clock.now += 65
                return super().upload_context(context)

        service = SlowToUpload([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=60), clock=clock) == 1
        assert (len(service.uploaded), service.started, service.polled, service.cancelled) == (1, [], [], [])
        assert f"{FAILURE_MARKER} demo-1: timed out after 60s, while uploading the build context" in caplog.text

    def test_the_upload_counts_against_the_build_timeout(self, context: Path, tmp_path: Path) -> None:
        """One deadline from before the upload: the time it takes is the start's to lose."""
        clock = FakeClock()
        busy = OverBudgetError("busy")
        service = FakeService([DONE], upload=[busy, busy, busy, busy, "context-1"], start=[busy])
        assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=60), clock=clock) == 1
        # Uploaded at 30 seconds. Starts at 30, 32, 36 and 44: the next wait, 16, would reach the minute. A start
        # on its own gets five tries in the minute.
        assert (len(service.uploaded), len(service.started), clock.now) == (5, 4, 44.0)


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
            ([RetryableError("connection reset")], 120, "connection reset; the build timeout leaves no time to try"),
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

    def test_a_budget_that_never_frees_is_given_up_before_the_build_timeout(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeService([DONE], start=[OverBudgetError("queue full")])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=300), clock=clock) == 1
        assert clock.now < 300
        assert max(clock.sleeps) == MAX_BACKOFF_SECONDS
        # Given up before the timeout: a build started with no time left would only be cancelled.
        assert "queue full; the build timeout leaves no time to try again" in caplog.text
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
        # Five tries, at 0, 2, 6, 14 and 30 seconds. The next wait, 32, would pass the minute: there is no sixth.
        assert (len(service.started), clock.now) == (5, 30.0)

    def test_a_start_is_not_made_after_a_wait_that_ended_past_the_deadline(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A sleep ends no sooner than it was asked to, and may end later. The wait fitted before the deadline
        when it began: the start after it would still be a build to cancel."""
        clock = LateClock(late=30.0)
        service = FakeService([DONE], start=[OverBudgetError("busy"), "build-1"])
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, tmp_path, config=_config(timeout=30), clock=clock) == 1
        # One try, at 0. Its wait of 2 seconds ended at 32, past the 30 the build had: there is no second.
        assert (len(service.started), clock.sleeps, clock.now) == (1, [2.0], 32.0)
        assert (service.polled, service.cancelled, builder.completed) == ([], [], [])
        assert f"{FAILURE_MARKER} demo-1: starting the build: busy; the build timeout passed before it could" in (
            caplog.text
        )
        # The service answered, and not that it had started anything.
        assert "a build may be running" not in caplog.text

    def test_a_lost_start_given_up_after_a_late_wait_is_still_said_to_maybe_be_running(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Given up without another try, the build a lost start may have begun is still named by its key."""
        clock = LateClock(late=30.0)
        service = FakeService([DONE], start=[RetryableError("connection reset"), "build-1"])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=30), clock=clock) == 1
        assert len(service.started) == 1
        assert "connection reset; the build timeout passed before it could be tried again" in caplog.text
        assert f"[a build may be running, started with key {idempotency_key(ATTEMPT, 'demo-1')}]" in caplog.text

    def test_a_read_is_still_made_after_a_wait_that_ended_past_the_deadline(
        self, context: Path, tmp_path: Path
    ) -> None:
        """A read costs nothing, and may find the build done: however late its wait ended, it is made."""
        clock = LateClock(late=30.0)
        service = FakeService([RetryableError("503"), DONE])
        builder = FakeBuilder()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=30), clock=clock) == 0
        assert (service.polled, service.cancelled, len(builder.completed)) == (["build-1"] * 2, [], 1)

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
        assert "timed out after 35s, while building (step 2 of 5) [build build-1]" in caplog.text

    def test_a_build_that_ends_on_its_last_poll_is_completed_not_cancelled(self, context: Path, tmp_path: Path) -> None:
        """The read at the deadline is made before the deadline is judged to have passed."""
        service = FakeService([BUILDING, BUILDING, DONE])
        builder = FakeBuilder()
        clock = FakeClock()
        assert _run(service, builder, context, tmp_path, config=_config(timeout=20), clock=clock) == 0
        assert (clock.now, service.cancelled, len(builder.completed)) == (20, [], 1)

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
        assert (fetched, service.uploaded, service.started) == ([], [], [])
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
        assert (fetched, service.uploaded, service.started, service.cancelled) == (["demo-1"], [], [], [])

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
        assert "timed out after 35s, while building (step 2 of 5) [build build-1]" in caplog.text

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
            ([BUILDING], 35, "timed out after 35s, while building (step 2 of 5)"),
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

    def test_a_stop_during_the_wait_to_complete_again_ends_the_tries_too(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The step's sleep ends early once the job is told to stop. Another try would be made in the seconds
        the pod has left, and one try may take them all."""
        builder = FakeBuilder(failing=[_unreachable()])
        clock = FakeClock()
        with caplog.at_level(logging.ERROR):
            # Told to stop during the first wait: not before it, when the failed try is judged.
            stopped = _run(
                FakeService([DONE]), builder, context, tmp_path, clock=clock, cancelled=lambda: bool(clock.sleeps)
            )
        # One try and one wait. No second try, though the builder would have taken it.
        assert (stopped, clock.sleeps, builder.completed) == (1, [2.0], [])
        assert "it was built, but the builder could not be told" in caplog.text
        assert "[build build-1]" in caplog.text

    def test_a_wait_to_complete_again_that_ends_late_ends_the_tries_too(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A wait fits the time given to completing when it begins, and a sleep on a busy node ends late. No
        try is made after that time, as no wait is begun that would end after it."""
        builder = FakeBuilder(failing=[_unreachable()])
        clock = LateClock(late=driver.COMPLETION_SECONDS)
        with caplog.at_level(logging.ERROR):
            failed = _run(FakeService([DONE]), builder, context, tmp_path, clock=clock)
        # One try and one wait. No second try, though the builder would have taken it.
        assert (failed, clock.sleeps, builder.completed) == (1, [2.0], [])
        assert "it was built, but the builder could not be told" in caplog.text
        assert "[build build-1]" in caplog.text

    def test_an_image_found_built_as_the_stop_came_is_still_completed(self, context: Path, tmp_path: Path) -> None:
        """The stop is read before each call to the service, not after one. A build that the last read found
        done is published, and its one completion is still made."""
        service = FakeService([DONE])
        builder = FakeBuilder()
        assert _run(service, builder, context, tmp_path, stop_after_polls=1) == 0
        assert (service.cancelled, len(builder.completed)) == ([], 1)

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
        # Not at the start of time: the window is counted from when the image was built.
        clock.now = began = 1000.0
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, clock=clock) == 1
        assert COMPLETION_SECONDS - MAX_BACKOFF_SECONDS <= clock.now - began < COMPLETION_SECONDS
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

    def test_a_credential_that_holds_the_marks_own_text_stays_out_of_a_reason(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "k3y-[credential]-9f2a"
        service = FakeService([DONE], start=[PermanentError(f"401: token {secret} is not valid")])
        with caplog.at_level(logging.ERROR):
            _run(service, FakeBuilder(), context, tmp_path, redact=driver._redactor(secret))
        assert "k3y-" not in caplog.text and "-9f2a" not in caplog.text
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
        service = FakeService([RemoteBuildStatus("failed", "its last step failed"), DONE], start=["build-1", "build-2"])
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
        names = ["a", "b", "c", "d", "one-with-a-longer-name"]
        service = FakeService(
            [failed, QUEUED, BUILDING, VERIFYING, failed, failed, failed, failed],
            start=[f"build-{number}" for number in range(1, 6)],
        )
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(*names)) == 1
        messages = [record.getMessage() for record in caplog.records]
        last = messages[messages.index("5 image(s), 5 failure(s)") + 1 :]
        assert [line.split(": ")[0] for line in last] == [f"{FAILURE_MARKER} {name}" for name in names]
        assert [line.rpartition(" [")[2] for line in last] == [f"build build-{number}]" for number in range(1, 6)]
        # As first logged, each reason cut to its 400, the five would not fit in the 1,800 characters Jobs is
        # taken to keep. Logged again they do, to the character: each line with its image's name and the 100 a
        # logger puts around it.
        assert (LINE_OVERHEAD, TAIL_CHARS, MAX_REASON_CHARS) == (100, 1800, 400)
        assert sum(len(line) - len(f"{FAILURE_MARKER} : ") + LINE_OVERHEAD for line in last) == TAIL_CHARS

    def test_a_line_with_no_room_left_for_its_reason_is_not_logged_again(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Settling reads the last line for an image. With many failures, a line logged again would be cut to
        nothing, and would replace the one that says why."""
        names = [f"image-{n:02}" for n in range(16)]
        service = FakeService([DONE], start=[PermanentError("401: the token was refused")])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(*names)) == 1
        lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith(FAILURE_MARKER)]
        # Once each, whole. 1,800 characters shared by sixteen lines leave a reason less than forty.
        assert lines == [f"{FAILURE_MARKER} {name}: starting the build: 401: the token was refused" for name in names]
        assert driver._room(16, "image-00") < driver.MIN_REASON_ROOM <= driver._room(8, "image-00")

    def test_an_unexpected_error_reading_a_row_does_not_cost_the_next_image_either(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Such as a token the client could not renew, which is no client error. It would otherwise end the
        step as a traceback, with no line for this image and the rest never tried."""
        builder = FakeBuilder(unreadable={"a": RuntimeError("token exchange answered 503")})
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, config=_config("a", "b")) == 1
        assert [image for image, _, _ in builder.completed] == ["b"]
        assert (
            f"{FAILURE_MARKER} a: cannot read its row: unexpected RuntimeError: token exchange answered 503"
        ) in caplog.text

    def test_a_build_is_still_named_when_many_lines_share_the_end_of_the_log(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reason is what is cut to share the space. The key that finds a build that may be running never is,
        even where that means the first lines fall outside what Jobs keeps."""
        names = [f"image-{n}" for n in range(6)]
        service = FakeService([DONE], start=[RetryableError("connection reset " * 20)])
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(*names, timeout=3600)) == 1
        messages = [record.getMessage() for record in caplog.records]
        last = messages[messages.index("6 image(s), 6 failure(s)") + 1 :]
        assert len(last) == 6
        for name, line in zip(names, last, strict=True):
            # The reason is cut, and what follows it is whole.
            assert "… [a build may be running, started with key " in line
            assert line.endswith(f"started with key {idempotency_key(ATTEMPT, name)}]")

    def test_a_line_that_names_a_build_is_logged_again_for_that_alone(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With too little room left for a reason worth reading, the build's name is still what finds it."""
        names = list("abcdefgh")
        builds = [f"{'b' * 85}{number}" for number in range(8)]
        service = FakeService([RemoteBuildStatus("failed", "no such base image")], start=list(builds))
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(*names)) == 1
        messages = [record.getMessage() for record in caplog.records]
        last = messages[messages.index("8 image(s), 8 failure(s)") + 1 :]
        assert [line.split(": ")[0] for line in last] == [f"{FAILURE_MARKER} {name}" for name in names]
        assert [line.rpartition(" [")[2] for line in last] == [f"build {build}]" for build in builds]

    def test_each_image_of_a_set_has_the_whole_build_timeout_to_itself(self, context: Path, tmp_path: Path) -> None:
        """Counted from when its own build is asked for, however long the images before it took."""
        service = FakeService([BUILDING] * 8 + [DONE], start=["build-1", "build-2"])
        builder, clock = FakeBuilder(), FakeClock()
        assert _run(service, builder, context, tmp_path, config=_config("a", "b", timeout=35), clock=clock) == 1
        assert service.cancelled == ["build-1"]
        assert [(image, external.build_id if external else None) for image, _, external in builder.completed] == [
            ("b", "build-2")
        ]
        # The first is given up at 35s. The second is read at 35, 45 and 55, and is done at 65: inside its own 35s.
        assert clock.now == 65

    def test_an_image_whose_directory_cant_be_made_does_not_cost_the_next(
        self, context: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Task storage that is full, say. It fails that image, with its line, not the step with a traceback."""
        make = tempfile.mkdtemp
        made: list[str] = []

        def mkdtemp(*, dir: Path, prefix: str) -> str:
            if prefix == "image-" and not made:
                made.append("refused")
                raise OSError(28, "No space left on device")
            return make(dir=dir, prefix=prefix)

        # The driver's own name for the module, not the module: the test suite shares that with everything else.
        monkeypatch.setattr(driver, "tempfile", SimpleNamespace(mkdtemp=mkdtemp))
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, context, tmp_path, config=_config("a", "b")) == 1
        assert [image for image, _, _ in builder.completed] == ["b"]
        assert f"{FAILURE_MARKER} a: cannot make a directory for its build context" in caplog.text

    def test_a_directory_that_cant_be_removed_does_not_cost_the_next_image_either(
        self, context: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client that still holds a file open in it, on storage that then refuses to remove the directory."""
        refused: list[str] = []

        def rmtree(path: str, ignore_errors: bool = False) -> None:
            refused.append(Path(path).name[:6])
            if not ignore_errors:
                raise OSError(39, "Directory not empty")

        kept = SimpleNamespace(rmtree=rmtree, copyfile=shutil.copyfile, copytree=shutil.copytree, copy2=shutil.copy2)
        monkeypatch.setattr(driver, "shutil", kept)
        builder = FakeBuilder()
        assert _run(FakeService([DONE]), builder, context, tmp_path, config=_config("a", "b")) == 0
        assert [image for image, _, _ in builder.completed] == ["a", "b"]
        # Asked to remove each image's directory, in a way that a refusal doesn't raise from.
        assert refused == ["image-", "image-"]

    def test_a_directory_too_deep_to_remove_does_not_cost_the_next_image_either(
        self, context: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before Python 3.13 a tree nested deeper than the interpreter's limit can't be walked, whatever
        errors the removal is told to ignore."""

        def rmtree(path: str, ignore_errors: bool = False) -> None:
            raise RecursionError("maximum recursion depth exceeded")

        kept = SimpleNamespace(rmtree=rmtree, copyfile=shutil.copyfile, copytree=shutil.copytree, copy2=shutil.copy2)
        monkeypatch.setattr(driver, "shutil", kept)
        builder = FakeBuilder()
        assert _run(FakeService([DONE]), builder, context, tmp_path, config=_config("a", "b")) == 0
        assert [image for image, _, _ in builder.completed] == ["a", "b"]

    def test_what_a_service_says_of_one_build_is_never_read_as_another_images_failure(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Settling reads a marked line as an image's reason, wherever in a line the marker is. A service's
        account of a build can hold a build's own output, and so anything."""
        forged = RemoteBuildStatus("building", f"step 3: {FAILURE_MARKER} b: an invented reason")
        service = FakeService([forged, DONE])
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config("a", "b")) == 0
        assert list(_marked_lines(caplog.text)) == []
        assert "build build-1: building (step 3: NHX-BUILD-FAILED b: an invented reason)" in caplog.text

    def test_what_a_service_says_of_a_call_that_failed_is_not_read_as_a_failure_line_either(
        self, context: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A start and a cancel that are tried again log why. Only the image's own line, last, is marked."""
        said = f"busy; {FAILURE_MARKER} b: an invented reason"
        service = FakeService([BUILDING], start=[OverBudgetError(said), "build-1"], cancel=RetryableError(said))
        with caplog.at_level(logging.INFO):
            assert _run(service, FakeBuilder(), context, tmp_path, config=_config(timeout=35)) == 1
        assert [line.split(": ", 2)[0] for line in _marked_lines(caplog.text)] == [f"{FAILURE_MARKER} demo-1"]
        assert caplog.text.count("NHX-BUILD-FAILED b: an invented reason") == 2

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
        """Unpacking only checks that a link stays inside the archive's directory. A second copy of the tree
        would change where it leads."""
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

    @pytest.mark.parametrize(
        ("leads_to", "named", "said"),
        [
            ("services/feed/Dockerfile", "Dockerfile", False),
            ("services/feed/Dockerfile", "services/feed/Dockerfile", False),
            ("docker/Dockerfile.prod", "services/feed/Dockerfile", True),
            ("no-such-file", "services/feed/Dockerfile", True),
        ],
        ids=["named-itself", "a-way-to-the-named-file", "a-way-to-another", "a-way-to-nothing"],
    )
    def test_a_root_dockerfile_that_is_a_link_is_said_to_be_replaced_when_it_was_another_file(
        self, context: Path, caplog: pytest.LogCaptureFixture, leads_to: str, named: str, said: bool
    ) -> None:
        """As a root Dockerfile that is a file is: a build that copies its own root Dockerfile copies the
        named one instead."""
        (context / "docker").mkdir()
        (context / "docker" / "Dockerfile.prod").write_text("FROM prod\n")
        (context / "Dockerfile").unlink()
        (context / "Dockerfile").symlink_to(leads_to)
        with caplog.at_level(logging.WARNING):
            stage_context(context, named)
        assert (context / "Dockerfile").read_text() == "FROM busybox\n"
        assert not (context / "Dockerfile").is_symlink()
        assert (f"root Dockerfile is replaced by {named}" in caplog.text) is said

    def test_the_driver_hands_the_service_the_named_dockerfile(self, context: Path, tmp_path: Path) -> None:
        seen: list[str] = []

        class Reading(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                seen.append((context.path / "Dockerfile").read_text())
                return super().upload_context(context)

        config = _config(dockerfile="services/feed/Dockerfile")
        assert _run(Reading([DONE]), FakeBuilder(), context, tmp_path, config=config) == 0
        assert seen == ["FROM busybox\n"]

    def test_the_client_is_handed_what_the_build_leaves_out_with_the_rules_that_leave_it_out(
        self, context: Path, tmp_path: Path
    ) -> None:
        """Nothing applies the ignore rules before the client is called. It is handed the whole context, a file
        the build leaves out included, with the ignore file a builder would read at the root: a client that
        sends the context anywhere applies that file first."""
        (context / "private.env").write_text("sample only\n")
        (context / ".dockerignore").write_text("*.md\n")
        (context / "services" / "feed" / "Dockerfile.dockerignore").write_text("private.env\n")
        handed: list[dict[str, str]] = []

        class Looking(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                names = ("private.env", "Dockerfile.dockerignore", ".dockerignore")
                handed.append({name: (context.path / name).read_text() for name in names})
                return super().upload_context(context)

        config = _config(dockerfile="services/feed/Dockerfile")
        assert _run(Looking([DONE]), FakeBuilder(), context, tmp_path, config=config) == 0
        # The Dockerfile's own ignore file, which a builder reads before the context's, and the file it names.
        assert handed == [
            {"private.env": "sample only\n", "Dockerfile.dockerignore": "private.env\n", ".dockerignore": "*.md\n"}
        ]

    def test_the_client_is_handed_a_file_the_contexts_own_rules_leave_out(self, context: Path, tmp_path: Path) -> None:
        """The Dockerfile is at the root, and nothing is moved. What `.dockerignore` leaves out of the build is
        still in the directory the client is handed, beside the rules: packed as it is, it would leave the step."""
        (context / "local.env").write_text("sample only\n")
        (context / ".dockerignore").write_text("local.env\n")
        handed: list[list[str]] = []

        class Looking(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                handed.append(sorted(path.name for path in context.path.iterdir()))
                return super().upload_context(context)

        assert _run(Looking([DONE]), FakeBuilder(), context, tmp_path) == 0
        assert handed == [[".dockerignore", "Dockerfile", "app.py", "local.env", "services"]]

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
        said = {
            "the-dockerfile": "dockerfile 'Dockerfile' is not a file inside the build context",
            "the-context": "its build context is not a directory in what was fetched",
        }
        assert said[missing] in caplog.text

    def test_a_directory_where_the_dockerfile_goes_is_refused(self, context: Path) -> None:
        (context / "Dockerfile").unlink()
        (context / "Dockerfile").mkdir()
        with pytest.raises(ImageFailed, match="a directory named Dockerfile at its root"):
            stage_context(context, "services/feed/Dockerfile")

    def test_a_directory_where_the_ignore_file_goes_is_refused(self, context: Path) -> None:
        (context / "Dockerfile.dockerignore").mkdir()
        with pytest.raises(ImageFailed, match="a directory named Dockerfile.dockerignore at its root"):
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

    def test_a_dockerfile_named_too_long_to_have_an_ignore_file_is_built_with_the_contexts_rules(
        self, context: Path
    ) -> None:
        """A Dockerfile's own ignore file is named for it, with thirteen characters more. Under a name within
        thirteen of the longest a directory holds, no file can have that name: asked for it, the system
        answers with an error, not that nothing is there, and a path's own `is_file` raises that on some
        Pythons and not on others. The image is staged as any whose Dockerfile has no ignore file of its
        own: with the context's rules, and without the replaced root Dockerfile's."""
        named = "Dockerfile." + "x" * 239
        (context / named).write_text("FROM alpine\n")
        (context / ".dockerignore").write_text("*.md\n")
        (context / "Dockerfile.dockerignore").write_text("services\n")
        with pytest.raises(OSError) as refused:
            os.stat(context / (named + ".dockerignore"))
        assert (len(named), refused.value.errno) == (250, errno.ENAMETOOLONG)

        staged = stage_context(context, named)

        assert (staged / "Dockerfile").read_text() == "FROM alpine\n"
        assert not (staged / "Dockerfile.dockerignore").exists()
        assert (staged / ".dockerignore").read_text() == "*.md\n"

    def test_a_dockerfile_at_a_path_too_long_to_look_beside_is_not_staged_without_its_ignore_file(
        self, context: Path
    ) -> None:
        """Every name here fits, and the path as a whole does not: the Dockerfile is six characters short of
        the longest path the system reads, and its ignore file, thirteen characters on, is past it. That
        file may be there all the same, for a builder that reads from the context's root to find. So the
        system's error is not taken for a missing file, on any Python: the image fails, and nothing is
        written into the context."""
        context = context.resolve()
        # Directories of a hundred characters, then a name that takes up what is left of the path.
        room = os.pathconf(context, "PC_PATH_MAX") - 6 - len(os.fsencode(context)) - len("/Dockerfile.")
        levels, more = divmod(room - 1, 101)
        dockerfile = "/".join([*["d" * 100] * levels, "Dockerfile." + "x" * (more + 1)])
        (context / dockerfile).parent.mkdir(parents=True, exist_ok=True)
        (context / dockerfile).write_text("FROM alpine\n")
        with pytest.raises(OSError) as refused:
            os.stat(context / (dockerfile + ".dockerignore"))
        assert (os.path.isfile(context / dockerfile), refused.value.errno) == (True, errno.ENAMETOOLONG)

        with pytest.raises(OSError) as failed:
            stage_context(context, dockerfile)

        assert failed.value.errno == errno.ENAMETOOLONG
        assert (context / "Dockerfile").read_text() == "FROM scratch\n"
        assert not (context / "Dockerfile.dockerignore").exists()

    @pytest.mark.parametrize("linked", ["by-a-link", "by-a-hard-link"])
    def test_the_root_dockerfile_under_another_name_brings_that_names_ignore_file(
        self, linked: str, context: Path
    ) -> None:
        """`Dockerfile.slim` is the root `Dockerfile` itself, so there is nothing to copy. Built by that name,
        though, it is that name's ignore file a builder reads."""
        if linked == "by-a-link":
            (context / "Dockerfile.slim").symlink_to("Dockerfile")
        else:
            os.link(context / "Dockerfile", context / "Dockerfile.slim")
        (context / "Dockerfile.slim.dockerignore").write_text("private.env\n")
        (context / "Dockerfile.dockerignore").write_text("services\n")

        staged = stage_context(context, "Dockerfile.slim")

        assert (staged / "Dockerfile").read_text() == "FROM scratch\n"
        assert not (staged / "Dockerfile").is_symlink()
        assert (staged / "Dockerfile.dockerignore").read_text() == "private.env\n"

    def test_the_root_dockerfile_under_another_name_does_not_gain_the_root_names_ignore_file(
        self, context: Path
    ) -> None:
        (context / "Dockerfile.slim").symlink_to("Dockerfile")
        (context / "Dockerfile.dockerignore").write_text("services\n")
        staged = stage_context(context, "Dockerfile.slim")
        assert not (staged / "Dockerfile.dockerignore").exists()

    def test_the_contexts_rules_are_kept_when_they_are_a_link_to_the_ignore_file_that_goes(self, context: Path) -> None:
        """The named Dockerfile has no ignore file of its own, so the context's `.dockerignore` is what applies
        to it: here a link to the root Dockerfile's own, which goes. Left a link, it would lead nowhere, and a
        `COPY .` would publish the `private.env` that the build left out where it was written."""
        (context / "private.env").write_text("sample only\n")
        (context / "Dockerfile.dockerignore").write_text("private.env\n")
        (context / ".dockerignore").symlink_to("Dockerfile.dockerignore")
        before = sorted(path.name for path in context.iterdir())

        staged = stage_context(context, "services/feed/Dockerfile")

        assert (staged / ".dockerignore").read_text() == "private.env\n"
        assert not (staged / ".dockerignore").is_symlink()
        # The replaced root Dockerfile's own still doesn't stand in for one, and nothing else is left behind.
        assert sorted(path.name for path in staged.iterdir()) == sorted(set(before) - {"Dockerfile.dockerignore"})

    @pytest.mark.parametrize("dockerfile", ["Dockerfile", "services/feed/Dockerfile"], ids=["at-the-root", "elsewhere"])
    def test_the_contexts_rules_that_are_a_link_to_another_file_become_a_file(
        self, dockerfile: str, context: Path
    ) -> None:
        """A builder reads the rules at the context's root, and a service's packager may not follow links."""
        (context / "rules").mkdir()
        (context / "rules" / "ignore").write_text("private.env\n")
        (context / ".dockerignore").symlink_to("rules/ignore")

        staged = stage_context(context, dockerfile)

        assert (staged / ".dockerignore").read_text() == "private.env\n"
        assert not (staged / ".dockerignore").is_symlink()
        assert (staged / "rules" / "ignore").read_text() == "private.env\n"

    def test_a_root_dockerfiles_ignore_file_that_is_a_link_becomes_a_file(self, context: Path) -> None:
        """Nothing is moved for the root Dockerfile. The rules it is built with are still made a file."""
        (context / "rules").mkdir()
        (context / "rules" / "prod.ignore").write_text("private.env\n")
        (context / "Dockerfile.dockerignore").symlink_to("rules/prod.ignore")

        staged = stage_context(context, "Dockerfile")

        assert (staged / "Dockerfile.dockerignore").read_text() == "private.env\n"
        assert not (staged / "Dockerfile.dockerignore").is_symlink()
        assert sorted(path.name for path in staged.iterdir()) == [
            "Dockerfile",
            "Dockerfile.dockerignore",
            "app.py",
            "rules",
            "services",
        ]

    def test_rules_reached_by_a_chain_of_links_are_kept_when_a_link_of_the_chain_goes(self, context: Path) -> None:
        """`.dockerignore -> Dockerfile.dockerignore -> rules/prod.ignore`. The link in the middle was the
        replaced root Dockerfile's, and goes; the context's own rules are still what they led to."""
        (context / "rules").mkdir()
        (context / "rules" / "prod.ignore").write_text("private.env\n")
        (context / "Dockerfile.dockerignore").symlink_to("rules/prod.ignore")
        (context / ".dockerignore").symlink_to("Dockerfile.dockerignore")

        staged = stage_context(context, "services/feed/Dockerfile")

        assert (staged / ".dockerignore").read_text() == "private.env\n"
        assert not (staged / ".dockerignore").is_symlink()
        assert not (staged / "Dockerfile.dockerignore").is_symlink()
        assert not (staged / "Dockerfile.dockerignore").exists()
        assert (staged / "rules" / "prod.ignore").read_text() == "private.env\n"

    def test_the_contexts_rules_are_as_they_were_when_the_file_they_link_to_is_replaced(self, context: Path) -> None:
        """The named Dockerfile brings its own ignore file to the root, in place of the one the context's
        `.dockerignore` links to. The context's own rules stay the ones they were, not the ones brought."""
        (context / "Dockerfile.dockerignore").write_text("services\n")
        (context / ".dockerignore").symlink_to("Dockerfile.dockerignore")
        (context / "services" / "feed" / "Dockerfile.dockerignore").write_text("private.env\n")

        staged = stage_context(context, "services/feed/Dockerfile")

        assert (staged / "Dockerfile.dockerignore").read_text() == "private.env\n"
        assert (staged / ".dockerignore").read_text() == "services\n"
        assert not (staged / ".dockerignore").is_symlink()

    def test_an_ignore_file_that_links_to_the_root_dockerfiles_is_the_one_staged(self, context: Path) -> None:
        """`Dockerfile.dev.dockerignore -> Dockerfile.dockerignore`: two Dockerfiles that share their rules."""
        (context / "Dockerfile.dev").write_text("FROM alpine\n")
        (context / "Dockerfile.dockerignore").write_text("private.env\n")
        (context / "Dockerfile.dev.dockerignore").symlink_to("Dockerfile.dockerignore")

        staged = stage_context(context, "Dockerfile.dev")

        assert (staged / "Dockerfile").read_text() == "FROM alpine\n"
        assert (staged / "Dockerfile.dockerignore").read_text() == "private.env\n"
        assert not (staged / "Dockerfile.dockerignore").is_symlink()
        assert (staged / "Dockerfile.dev.dockerignore").read_text() == "private.env\n"

    @pytest.mark.parametrize("leads_to", ["no-such-file", "services"], ids=["nowhere", "a-directory"])
    def test_the_contexts_rules_that_link_to_no_file_are_left_as_they_were(self, leads_to: str, context: Path) -> None:
        """There were no rules to read before the Dockerfile was moved, and there are none after."""
        (context / ".dockerignore").symlink_to(leads_to)
        staged = stage_context(context, "services/feed/Dockerfile")
        assert (staged / ".dockerignore").is_symlink()
        assert os.readlink(staged / ".dockerignore") == leads_to

    def test_a_dockerfiles_own_ignore_file_is_never_held_in_memory_whole(self, context: Path) -> None:
        """Unpacking lets one file of an archive be gigabytes. Read whole, it would end the step, and with it
        the turn of every image of the set still to be built."""
        block = (b"*.log\n" * 175_000)[: 1024 * 1024]
        size = 16 * len(block)
        ignore = context / "services" / "feed" / "Dockerfile.dockerignore"
        with ignore.open("wb") as written:
            for _ in range(16):
                written.write(block)
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            held, _ = tracemalloc.get_traced_memory()
            staged = stage_context(context, "services/feed/Dockerfile")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not tracing:
                tracemalloc.stop()

        assert peak - held < size // 8
        assert (staged / "Dockerfile.dockerignore").stat().st_size == size
        assert filecmp.cmp(staged / "Dockerfile.dockerignore", ignore, shallow=False)
        assert sorted(path.name for path in staged.iterdir()) == [
            "Dockerfile",
            "Dockerfile.dockerignore",
            "app.py",
            "services",
        ]

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
        with pytest.raises(ImageFailed, match="a link that leads out of it, or back .*: 'services/fixture'"):
            stage_context(context, "Dockerfile")

    def test_a_link_to_a_directory_whose_name_only_begins_as_the_contexts_does_is_refused(
        self, context: Path, tmp_path: Path
    ) -> None:
        """`context-data`, beside `context`. It is not in the context for sharing the start of its name."""
        (tmp_path / "context-data").mkdir()
        (tmp_path / "context-data" / "answer.txt").write_text("not this build's\n")
        (context / "fixture").symlink_to("../context-data/answer.txt")
        with pytest.raises(ImageFailed, match="a link that leads out of it, or back .*: 'fixture'"):
            stage_context(context, "Dockerfile")

    @pytest.mark.parametrize(
        ("link", "target"),
        [("round", "."), ("services/feed/up", ".."), ("services/feed/top", "../..")],
        ids=["to-the-root", "to-its-parent", "to-the-root-from-below"],
    )
    def test_a_link_back_to_a_directory_it_is_in_is_refused(self, link: str, target: str, context: Path) -> None:
        """A packager that follows links would go round it for ever."""
        (context / link).symlink_to(target)
        with pytest.raises(ImageFailed, match=f"back to a directory it is in: {link!r}"):
            stage_context(context, "Dockerfile")

    def test_a_way_round_by_two_links_is_refused(self, context: Path) -> None:
        """Neither link leads to a directory it is in. Followed one after the other, they come back."""
        (context / "a").mkdir()
        (context / "b").mkdir()
        (context / "a" / "x").symlink_to("../b")
        (context / "b" / "y").symlink_to("../a")
        with pytest.raises(ImageFailed, match="back to a directory it is in: '(a/x|b/y)'"):
            stage_context(context, "Dockerfile")

    @pytest.mark.parametrize("link_first", [True, False], ids=["the-link-listed-first", "the-directory-first"])
    def test_a_way_round_that_closes_on_a_directory_names_the_link_taken_to_get_there(self, link_first: bool) -> None:
        """`z -> a/sub` beside `a`, and `a/sub/back -> ..`. Walked by `z` first, the walk is back in `a/sub`
        by the directory itself, not by a link: the way round is still one, in whichever order a directory
        lists its entries."""
        root, a, sub = Path("/c"), Path("/c/a"), Path("/c/a/sub")
        entries: list[tuple[Path, str | None]] = [(sub, "z"), (a, None)]
        ways: dict[Path, list[tuple[Path, str | None]]] = {
            root: entries if link_first else entries[::-1],
            sub: [(a, "a/sub/back")],
            a: [(sub, None)],
        }
        assert driver._link_round(root, ways) == "a/sub/back"

    def test_many_links_to_one_directory_are_no_way_round(self, context: Path) -> None:
        (context / "shared").mkdir()
        (context / "shared" / "lib.py").write_text("")
        for name in ("one", "two", "three"):
            (context / "services" / name).symlink_to("../shared")
        (context / "services" / "feed" / "lib").symlink_to("../../shared")
        assert stage_context(context, "Dockerfile") == context

    def test_a_context_reached_by_a_link_is_handed_on_as_the_directory_itself(self, tmp_path: Path) -> None:
        """An archive's `current -> v2`, named as the context. A packager that doesn't follow links would
        otherwise be handed the link, and package nothing."""
        (tmp_path / "v2").mkdir()
        (tmp_path / "v2" / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / "current").symlink_to("v2")
        staged = stage_context(tmp_path / "current", "Dockerfile")
        assert (staged, staged.is_symlink()) == ((tmp_path / "v2").resolve(), False)

    def test_a_dockerfile_that_is_a_loop_of_links_is_refused_as_any_that_is_no_file(self, context: Path) -> None:
        """Whatever the Python: some raise for a loop where others resolve it to nothing."""
        (context / "Dockerfile.a").symlink_to("Dockerfile.b")
        (context / "Dockerfile.b").symlink_to("Dockerfile.a")
        with pytest.raises(ImageFailed, match="dockerfile 'Dockerfile.a' is not a file inside the build context"):
            stage_context(context, "Dockerfile.a")

    def test_a_link_that_comes_back_to_itself_and_then_steps_back_is_a_way_to_the_directory_it_is_in(
        self, context: Path
    ) -> None:
        """`again -> again/..`. The system gives up on it. A path is read here as `os.path.realpath` reads
        one, which leaves a link it has come back to in the path as a name and goes on: so read, this one
        leads to the directory it is in, and a packager that follows links that way would go round it for
        ever."""
        (context / "services" / "again").symlink_to("again/..")
        with pytest.raises(ImageFailed, match="back to a directory it is in: 'services/again'"):
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

    @pytest.mark.parametrize(
        ("link", "written", "rewritten"),
        [
            ("cached.txt", "../app/payload", "payload"),
            ("lib/cached.txt", "../../app/payload", "../payload"),
            ("cached.txt", "{context}/payload", "payload"),
            ("cached.txt", "../beside/alias", "payload"),
            ("cached.txt", "lib/vendored/../../app/payload", "payload"),
        ],
        ids=[
            "by-its-directorys-name",
            "from-a-directory-below",
            "by-an-absolute-path",
            "by-a-link-beside-it",
            "past-a-link-to-a-directory",
        ],
    )
    def test_a_link_that_leaves_the_context_on_its_way_back_into_it_is_rewritten_to_stay_inside(
        self, link: str, written: str, rewritten: str, tmp_path: Path
    ) -> None:
        """`cached.txt -> ../app/payload`, in a context named `app`. It leads inside the context where the step
        holds it, and out of it wherever a service unpacks the context under another name."""
        source = tmp_path / "fetched" / "app"
        (source / "lib").mkdir(parents=True)
        (source / "vendor").mkdir()
        (source / "Dockerfile").write_text("FROM scratch\nCOPY . /srv\n")
        (source / "payload").write_text("built once\n")
        # A way to a directory of the context, and a link beside the context that comes back into it.
        (source / "lib" / "vendored").symlink_to("../vendor")
        (tmp_path / "fetched" / "beside").mkdir()
        (tmp_path / "fetched" / "beside" / "alias").symlink_to("../app/payload")
        (source / link).symlink_to(written.format(context=source))

        staged = stage_context(source, "Dockerfile")

        assert os.readlink(staged / link) == rewritten
        unpacked = _unpacked_elsewhere(staged, tmp_path / "service" / "build-context")
        assert (unpacked / link).read_text() == "built once\n"
        # A link that stayed inside as it was written is as it was.
        assert os.readlink(unpacked / "lib" / "vendored") == "../vendor"

    @pytest.mark.parametrize(
        ("link", "written"),
        [
            ("services/feed/up", "../../app.py"),
            ("here", "./services"),
            ("down-and-up", "services/../app.py"),
            ("by-a-link", "all-services/feed/../../app.py"),
            ("nowhere", "no-such-file"),
        ],
        ids=["up-to-the-root", "with-a-dot", "down-and-up", "by-a-link-to-a-directory", "to-nothing"],
    )
    def test_a_link_that_stays_inside_the_context_as_it_is_written_is_left_exactly_as_it_is(
        self, link: str, written: str, context: Path
    ) -> None:
        """Not made the shortest way to what it leads to: a build that copies the link copies it as it was."""
        (context / "all-services").symlink_to("services")
        (context / link).symlink_to(written)
        staged = stage_context(context, "Dockerfile")
        assert (os.readlink(staged / link), os.readlink(staged / "all-services")) == (written, "services")

    def test_a_rewritten_link_still_names_the_link_it_named(
        self, context: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`first -> ../context/second`, and `second -> app.py`. Rewritten, the first is still a link to the
        second, not to the file the two lead to."""
        (context / "second").symlink_to("app.py")
        (context / "first").symlink_to("../context/second")
        (context / "gone").symlink_to("../context/no-such-file")
        with caplog.at_level(logging.INFO):
            staged = stage_context(context, "Dockerfile")
        assert (os.readlink(staged / "first"), os.readlink(staged / "second")) == ("second", "app.py")
        # The step's log says how many links it rewrote: a build that copies one copies it as rewritten.
        assert "2 link(s) of the build context left it on the way to what they lead to inside it" in caplog.text
        # One that leads to nothing is rewritten too: where it led is still inside the context, and still empty.
        assert os.readlink(staged / "gone") == "no-such-file"
        assert _unpacked_elsewhere(staged, context.parent / "elsewhere").joinpath("first").read_text() == (
            "print('hello')\n"
        )

    @pytest.mark.parametrize("ending", ["/", "/."], ids=["a-separator", "a-dot-after-one"])
    def test_a_rewritten_link_that_led_only_to_a_directory_still_leads_only_to_one(
        self, ending: str, tmp_path: Path
    ) -> None:
        """`cached -> ../app/payload/`, in a context named `app` whose `payload` is a file. A path that ends
        in a separator leads to a directory or to nothing, so the link led nowhere. Rewritten without the
        separator it would lead to the file: staging would have made a link that could not be read into one
        that can. A link to a directory, written the same way, leads to it as before."""
        source = tmp_path / "fetched" / "app"
        (source / "vendor").mkdir(parents=True)
        (source / "Dockerfile").write_text("FROM scratch\n")
        (source / "payload").write_text("built once\n")
        (source / "cached").symlink_to(f"../app/payload{ending}")
        (source / "libs").symlink_to(f"../app/vendor{ending}")
        assert not (source / "cached").exists()

        staged = stage_context(source, "Dockerfile")

        assert (os.readlink(staged / "cached"), os.readlink(staged / "libs")) == ("payload/", "vendor/")
        assert not (staged / "cached").exists()
        assert (staged / "libs").is_dir()

    def test_a_link_through_a_loop_that_climbs_out_of_the_context_after_it_is_refused(
        self, context: Path, tmp_path: Path
    ) -> None:
        """It leads nowhere here, for the loop. Whatever the Python, and whether or not it raises for a loop,
        a path that comes to a file beside the context is not one to hand on."""
        (tmp_path / "beside").mkdir()
        (tmp_path / "beside" / "answer.txt").write_text("not this build's\n")
        (context / "one").symlink_to("two")
        (context / "two").symlink_to("one")
        (context / "fixture").symlink_to("one/../../beside/answer.txt")
        with pytest.raises(ImageFailed, match="a link that leads out of it, or back .*: 'fixture'"):
            stage_context(context, "Dockerfile")

    @staticmethod
    def _doubling(context: Path, levels: int) -> Path:
        """``levels`` directories in ``context``, one inside the other, each with a link to it beside it: two
        ways into every level, so a walk that follows links comes to the last one ``2 ** levels`` times.
        Returns the last."""
        level = context
        for _ in range(levels):
            (level / "n").mkdir()
            (level / "a").symlink_to("n")
            level = level / "n"
        return level

    def test_links_that_multiply_a_context_past_the_entries_a_source_may_unpack_to_are_refused(
        self, context: Path
    ) -> None:
        """No link here comes back round, so a packager that follows links follows every one, and sends the
        one file at the bottom a million million times. Counted with each directory's total remembered:
        walked, forty levels would never end."""
        (self._doubling(context, 40) / "payload").write_text("x")
        with pytest.raises(ImageFailed, match=f"more than {MAX_UNPACKED_ENTRIES} entries with its links followed"):
            stage_context(context, "Dockerfile")

    def test_links_that_multiply_a_context_past_the_bytes_a_source_may_unpack_to_are_refused(
        self, context: Path
    ) -> None:
        """Thirteen levels are 8,192 copies of one mebibyte: entries enough to be allowed, and twice the bytes."""
        (self._doubling(context, 13) / "payload").write_bytes(b"x" * 1024 * 1024)
        with pytest.raises(ImageFailed, match=f"more than {MAX_UNPACKED_BYTES} bytes with its links followed"):
            stage_context(context, "Dockerfile")

    def test_a_context_is_measured_as_a_packager_that_follows_its_links_would_send_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each link counts as what it leads to, each time a walk comes to it. A context that comes to the
        limit exactly is built."""
        context = tmp_path / "measured"
        context.mkdir()
        (context / "Dockerfile").write_text("FROM scratch\n")
        (self._doubling(context, 3) / "payload").write_bytes(b"x" * 100)
        (context / "again").symlink_to("n/n/n/payload")
        # At the bottom, the payload. At each level above it, a directory and a link to it, each with all that
        # is below. At the root, the Dockerfile and a link to the payload besides.
        entries = 2 + (2 + 2 * (2 + 2 * (2 + 2 * 1)))
        size = len("FROM scratch\n") + 100 + 8 * 100
        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", entries)
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size)
        assert stage_context(context, "Dockerfile") == context
        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", entries - 1)
        with pytest.raises(ImageFailed, match=f"more than {entries - 1} entries with its links followed"):
            stage_context(context, "Dockerfile")
        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", entries)
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size - 1)
        with pytest.raises(ImageFailed, match=f"more than {size - 1} bytes with its links followed"):
            stage_context(context, "Dockerfile")

    def test_one_file_under_many_names_is_measured_once_as_unpacking_counts_it(
        self, context: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hard links. An archive that was let through for its size is not refused for it here, for holding
        a link as well."""
        files = [path for path in context.rglob("*") if path.is_file()]
        (context / "payload").write_bytes(b"x" * 1000)
        for number in range(8):
            os.link(context / "payload", context / f"payload-{number}")
        (context / "alias").symlink_to("app.py")
        size = sum(path.stat().st_size for path in files) + 1000 + (context / "app.py").stat().st_size
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size)
        assert stage_context(context, "Dockerfile") == context
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size - 1)
        with pytest.raises(ImageFailed, match="bytes with its links followed"):
            stage_context(context, "Dockerfile")

    @pytest.mark.parametrize("listed_first", ["a", "b"], ids=["a-listed-first", "b-listed-first"])
    @pytest.mark.parametrize(("aliased", "times"), [("a", 2), ("b", 1)], ids=["a-link-to-a", "a-link-to-b"])
    def test_one_file_named_in_two_directories_is_measured_the_same_whichever_a_listing_puts_first(
        self, aliased: str, times: int, listed_first: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`a/payload` and `b/payload` are one file, and `again` is a link to one of the two directories: a
        walk that follows links comes to that one twice. The file is counted in the first of its
        directories the walk comes to, and the walk takes directories in the order of their names: in `a`,
        so twice when the link leads to `a`, and once when it leads to `b`. Taken in the order a directory
        lists them, which no file system promises, the same context came to a thousand bytes more or
        less by where it was unpacked. The walk is given the two directories here in either order: a real
        disk lists them in one order of its own, whichever was made first."""
        context = tmp_path / "measured"
        context.mkdir()
        (context / "Dockerfile").write_text("FROM scratch\n")
        for name in ("a", "b"):
            (context / name).mkdir()
        (context / "a" / "payload").write_bytes(b"x" * 1000)
        os.link(context / "a" / "payload", context / "b" / "payload")
        (context / "again").symlink_to(aliased)
        size = len("FROM scratch\n") + times * 1000
        monkeypatch.setattr(driver, "os", _ListedFirst(listed_first))

        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size)
        assert stage_context(context, "Dockerfile") == context
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size - 1)
        with pytest.raises(ImageFailed, match=f"more than {size - 1} bytes with its links followed"):
            stage_context(context, "Dockerfile")

    def test_a_context_with_no_links_is_not_measured(self, context: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No link makes more of it than was fetched, and what bounds a fetch is the fetch's to say. Nor does
        staging: its Dockerfile is at the root already, and nothing is written."""
        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", 1)
        monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", 1)
        assert stage_context(context, "Dockerfile") == context

    def test_links_that_multiply_a_context_within_what_a_source_may_unpack_to_are_followed(self, context: Path) -> None:
        """Eight levels are 256 copies of a small file: no more than an archive may hold without a link."""
        (self._doubling(context, 8) / "payload").write_text("x")
        assert stage_context(context, "Dockerfile") == context


class TestTheImagesOfOneSet:
    """A set's images are built one after another, and a source can change in between."""

    @staticmethod
    def _sources(files: dict[str, str]) -> Callable[[RemoteBuildImage, Path], Path]:
        """A fetch that writes ``files`` as they are when it is called, and counts its calls."""

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            fetch.calls.append(image.source.fileset)  # ty: ignore[unresolved-attribute]
            fetch.directories.append((image.source.archive, image.source.context_path))  # ty: ignore[unresolved-attribute]
            for name, content in files.items():
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_text(content)
            return directory

        fetch.calls = []  # ty: ignore[unresolved-attribute]
        fetch.directories = []  # ty: ignore[unresolved-attribute]
        return fetch

    @staticmethod
    def _set(*sources: dict[str, str]) -> RemoteBuildStepConfig:
        """A set with one image, named a, b, c..., for each source."""
        return RemoteBuildStepConfig.model_validate(
            {
                "images": [
                    {"image": name, "source": source, "dockerfile": "Dockerfile", "platform": "linux/amd64"}
                    for name, source in zip("abcdef", sources, strict=False)
                ],
                "registry": "registry.example.com",
                "repository": "team/images",
                "build_timeout_seconds": 600,
            }
        )

    @staticmethod
    def _reading(read: str) -> tuple[type[FakeService], list[str]]:
        """A service that notes ``read``, a file of each context it is handed."""
        seen: list[str] = []

        class Reading(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                seen.append((context.path / read).read_text())
                return super().upload_context(context)

        return Reading, seen

    def test_every_image_is_built_from_the_source_as_it_was_first_fetched(self, tmp_path: Path) -> None:
        files = {"Dockerfile": "FROM scratch\n", "version": "first"}
        fetch = self._sources(files)
        seen: list[str] = []

        class Changing(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                seen.append((context.path / "version").read_text())
                files["version"] = "second"
                return super().upload_context(context)

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
            def upload_context(self, context: RemoteContext) -> str:
                built.append((context.path / "Dockerfile").read_text())
                return super().upload_context(context)

        assert _run(Reading([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=self._sources(files), config=config) == 0
        assert built == ["FROM one\n", "FROM root\n", "FROM two\n"]

    def test_an_archive_is_downloaded_once_whichever_of_its_directories_each_image_builds(self, tmp_path: Path) -> None:
        """Unpacked whole, as the in-cluster build's fetch step does. Fetched again for its second directory, a
        revision's images could come from two uploads of the archive."""
        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        files = {"environment/Dockerfile": "FROM env\n", "tests/Dockerfile": "FROM tests\n"}
        fetch = self._sources(files)
        service, built = self._reading("Dockerfile")
        config = self._set(archive | {"context_path": "environment"}, archive | {"context_path": "tests"})

        assert _run(service([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 0

        assert built == ["FROM env\n", "FROM tests\n"]
        # One download, of the archive and not of a directory of it.
        assert fetch.directories == [("task.tar.gz", None)]  # ty: ignore[unresolved-attribute]

    def test_a_filesets_whole_download_serves_the_images_that_build_a_directory_of_it(self, tmp_path: Path) -> None:
        files = {"Dockerfile": "FROM root\n", "feed/Dockerfile": "FROM feed\n"}
        fetch = self._sources(files)
        service, built = self._reading("Dockerfile")
        config = self._set({"fileset": "default/fs-a", "context_path": "feed"}, {"fileset": "default/fs-a"})

        assert _run(service([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 0

        assert built == ["FROM feed\n", "FROM root\n"]
        assert fetch.directories == [(None, None)]  # ty: ignore[unresolved-attribute]

    def test_a_copy_keeps_its_directorys_name_so_a_link_back_to_it_by_name_still_leads_inside(
        self, tmp_path: Path
    ) -> None:
        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            for name in ("environment", "tests"):
                (directory / name).mkdir()
                (directory / name / "Dockerfile").write_text(f"FROM {name}\n")
            (directory / "environment" / "again").symlink_to("../environment/Dockerfile")
            return directory

        service, built = self._reading("again")
        config = self._set(archive | {"context_path": "environment"}, archive | {"context_path": "tests"})
        builder = FakeBuilder()
        assert _run(service([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=config) == 1
        # The first image's link came back to its own directory. The second has no such file to read.
        assert (built, [image for image, _, _ in builder.completed]) == (["FROM environment\n"], ["a"])

    def test_a_download_that_fails_leaves_nothing_behind_and_is_tried_again_for_the_next_image(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[str] = []

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            calls.append(image.image)
            (directory / "half").write_text("part of a download")
            if len(calls) == 1:
                raise OSError("disk full")
            (directory / "Dockerfile").write_text("FROM scratch\n")
            return directory

        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=_config("a", "b")) == 1
        assert f"{FAILURE_MARKER} a: cannot fetch its build context: disk full" in caplog.text
        assert (calls, [image for image, _, _ in builder.completed]) == (["a", "b"], ["b"])
        # The failed download's directory is gone: only the one that worked is kept, for the rest of the set.
        assert len(list((tmp_path / "scratch").glob("source-*"))) == 1

    def test_each_source_of_a_set_is_its_own_download(self, tmp_path: Path) -> None:
        fetch = self._sources({"Dockerfile": "FROM scratch\n"})
        config = self._set({"fileset": "default/fs-a"}, {"fileset": "default/fs-b"}, {"fileset": "default/fs-a"})
        assert _run(FakeService([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 0
        assert fetch.calls == ["default/fs-a", "default/fs-b"]  # ty: ignore[unresolved-attribute]

    @pytest.mark.parametrize(
        ("first", "second"),
        [("app", "app/"), ("app", "./app"), ("app/x", "app//x"), ("./app/x/", "app/./x")],
        ids=["a-trailing-slash", "a-leading-dot", "a-doubled-slash", "all-of-them"],
    )
    def test_two_spellings_of_one_directory_are_one_download(self, first: str, second: str, tmp_path: Path) -> None:
        """`app` and `app/` name one directory of a fileset. Downloaded once under each spelling, a set's two
        images would be built from the source as it was at two different times."""
        files = {"Dockerfile": "FROM scratch\n", "version": "first"}
        fetch = self._sources(files)
        seen: list[str] = []

        class Changing(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                seen.append((context.path / "version").read_text())
                files["version"] = "second"
                return super().upload_context(context)

        fileset = {"fileset": "default/fs-a"}
        config = self._set(fileset | {"context_path": first}, fileset | {"context_path": second})
        builder = FakeBuilder()
        assert _run(Changing([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=config) == 0
        assert seen == ["first", "first"]
        assert [image for image, _, _ in builder.completed] == ["a", "b"]
        # One download, asked for under the one spelling the two come to.
        assert fetch.directories == [(None, PurePosixPath(first).as_posix())]  # ty: ignore[unresolved-attribute]

    def test_a_copy_keeps_the_names_of_one_file_as_one_file(self, tmp_path: Path) -> None:
        """An archive may hold one file under many names, as hard links, and unpacking counts its size once.
        Copied as a file for each name, an archive that is allowed would fill the step's storage."""
        payload = b"x" * 65536

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            (directory / "Dockerfile").write_text("FROM scratch\n")
            (directory / "payload").write_bytes(payload)
            (directory / "payload").chmod(0o750)
            (directory / "names").mkdir()
            for number in range(32):
                os.link(directory / "payload", directory / "names" / f"payload-{number}")
            return directory

        #: For each image's copy: its names, its files, their bytes, and their modes.
        copies: list[tuple[int, int, int, set[int]]] = []
        #: Whether a copy shares a file with the download, and how many names the download's file then has.
        downloads: list[tuple[bool, int]] = []
        read: list[bool] = []

        class Measuring(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                (download,) = (tmp_path / "scratch").glob("source-*/payload")
                names = [context.path / "payload", *(context.path / "names").iterdir()]
                files = {name.stat().st_ino: name.stat() for name in names}
                sizes = sum(status.st_size for status in files.values())
                copies.append((len(names), len(files), sizes, {status.st_mode & 0o777 for status in files.values()}))
                downloads.append((download.stat().st_ino in files, download.stat().st_nlink))
                read.append(all(name.read_bytes() == payload for name in names))
                return super().upload_context(context)

        assert _run(Measuring([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=_config("a", "b")) == 0
        # Thirty-three names and one file of 64 KiB, as it was, in each image's copy.
        assert copies == [(33, 1, len(payload), {0o750})] * 2
        assert read == [True, True]
        # A copy of its own: none of its names is a name of the download's file.
        assert downloads == [(False, 33)] * 2

    def test_a_copy_keeps_a_link_as_a_link_so_one_that_leads_out_is_still_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Copied with its links followed, the file from beside the context would be in every image's copy, as
        an ordinary file that nothing could tell from the context's own."""
        (tmp_path / "beside").mkdir()
        (tmp_path / "beside" / "answer.txt").write_text("not this build's\n")

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            (directory / "Dockerfile").write_text("FROM scratch\n")
            (directory / "fixture").symlink_to(tmp_path / "beside" / "answer.txt")
            return directory

        service = FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=_config("a", "b")) == 1
        assert service.started == []
        assert caplog.text.count("a link that leads out of it, or back to a directory it is in: 'fixture'") >= 2

    def test_a_directory_of_a_download_that_isnt_there_is_said_to_be_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        fetch = self._sources({"environment/Dockerfile": "FROM env\n"})
        config = self._set(archive | {"context_path": "environment"}, archive | {"context_path": "no-such"})
        builder = FakeBuilder()
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=config) == 1
        assert [image for image, _, _ in builder.completed] == ["a"]
        assert f"{FAILURE_MARKER} b: context path 'no-such' is not a directory in its source" in caplog.text

    @pytest.mark.parametrize("outside", ["../beside", "/etc", "environment/../.."], ids=["up", "absolute", "up-below"])
    def test_a_directory_named_out_of_a_shared_download_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, outside: str
    ) -> None:
        """A submit refuses such a path. A step config written some other way is not taken at its word."""
        # Where `../beside` leads from the download, which is made in the step's scratch directory: a directory
        # that could be built from, were the path taken at its word.
        (tmp_path / "scratch" / "beside").mkdir(parents=True)
        (tmp_path / "scratch" / "beside" / "Dockerfile").write_text("FROM beside\n")
        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        fetch = self._sources({"environment/Dockerfile": "FROM env\n"})
        config = self._set(archive | {"context_path": "environment"}, archive | {"context_path": outside})
        builder, service = FakeBuilder(), FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, tmp_path, tmp_path, fetch=fetch, config=config) == 1
        assert [image for image, _, _ in builder.completed] == ["a"]
        assert len(service.started) == 1
        assert f"{FAILURE_MARKER} b: context path {outside!r} is not a directory inside its source" in caplog.text

    @pytest.mark.parametrize(
        ("context_path", "refused"),
        [("up", True), ("up/below", True), ("current", False), (None, False)],
        ids=["a-link-out", "below-a-link-out", "a-link-that-stays-inside", "the-whole-source"],
    )
    def test_a_fetched_context_is_inside_what_was_fetched_once_its_links_are_followed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context_path: str | None, refused: bool
    ) -> None:
        """One image's own download. Where a context path leads rests on how the source was unpacked, and the
        step, which holds the service's credential, checks it for itself."""
        outside = tmp_path / "mounted"
        (outside / "below").mkdir(parents=True)

        def fetch_source(files_client: object, layout: WorkLayout, source: ContextSource, *, workspace: str) -> None:
            unpacked = Path(layout.context(source.model_copy(update={"context_path": None})))
            (unpacked / "v2").mkdir(parents=True)
            (unpacked / "current").symlink_to("v2")
            (unpacked / "up").symlink_to(outside)

        monkeypatch.setattr(driver, "fetch_source", fetch_source)
        source = {"fileset": "default/fs-a", "archive": "task.tar.gz", "context_path": context_path}
        image = RemoteBuildImage.model_validate(
            {"image": "a", "source": source, "dockerfile": "Dockerfile", "platform": "linux/amd64"}
        )
        root = tmp_path / "fetched"
        if refused:
            with pytest.raises(
                ImageFailed, match=f"context path '{context_path}' is not a directory inside its source"
            ):
                driver.fetch_context(object(), image, workspace="default", root=root)  # ty: ignore[invalid-argument-type]
        else:
            fetched = driver.fetch_context(object(), image, workspace="default", root=root)  # ty: ignore[invalid-argument-type]
            assert fetched.resolve().is_relative_to(root.resolve())
            assert fetched.name == (context_path or fetched.name)

    @pytest.mark.parametrize("leads_to", ["a-file", "nothing", "a-loop-of-links"])
    def test_a_context_path_that_leads_to_no_directory_is_refused_as_one_that_matches_none_wherever_it_leads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, leads_to: str
    ) -> None:
        """One image's own download. The system is asked whether the path is a directory before the step
        follows its links for itself, as it is of a download that is shared: only a directory is followed,
        to see that it is inside its source. What is no directory is no context, wherever its links lead:
        a file beside the source, nothing there, or round to itself. It is refused as a context path that
        matches nothing is, and nothing of where it led is handed on."""
        outside = tmp_path / "mounted"
        outside.mkdir()
        (outside / "answer.txt").write_text("not this build's\n")
        written = {"a-file": str(outside / "answer.txt"), "nothing": str(outside / "no-such"), "a-loop-of-links": "up"}

        def fetch_source(files_client: object, layout: WorkLayout, source: ContextSource, *, workspace: str) -> None:
            unpacked = Path(layout.context(source.model_copy(update={"context_path": None})))
            unpacked.mkdir(parents=True)
            (unpacked / "up").symlink_to(written[leads_to])

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            return driver.fetch_context(object(), image, workspace="default", root=directory / "fetched")  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(driver, "fetch_source", fetch_source)
        config = self._set({"fileset": "default/fs-a", "archive": "task.tar.gz", "context_path": "up"})
        builder, service = FakeBuilder(), FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, tmp_path, tmp_path, fetch=fetch, config=config) == 1
        assert (service.started, builder.completed) == ([], [])
        assert (
            f"{FAILURE_MARKER} a: its build context is not a directory in what was fetched: its context path "
            "matches none"
        ) in caplog.text

    def test_a_link_that_comes_back_by_the_directories_above_it_leads_inside_the_copy_too(self, tmp_path: Path) -> None:
        """`services/api/shared -> ../../services/api/lib`. Where it leads is decided where the download is,
        with the directories above the context still there, so such a link builds from a set as it does
        alone."""

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            for name in ("api", "web"):
                (directory / "services" / name).mkdir(parents=True)
                (directory / "services" / name / "Dockerfile").write_text(f"FROM {name}\n")
            (directory / "services" / "api" / "lib").mkdir()
            (directory / "services" / "api" / "lib" / "util.py").write_text("")
            (directory / "services" / "api" / "shared").symlink_to("../../services/api/lib")
            return directory

        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        config = self._set(archive | {"context_path": "services/api"}, archive | {"context_path": "services/web"})
        #: Each context's Dockerfile, and what its link to the library is written as, where it has one.
        seen: list[tuple[str, str | None]] = []

        class Reading(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                shared = context.path / "shared"
                written = os.readlink(shared) if shared.is_symlink() else None
                seen.append(((context.path / "Dockerfile").read_text(), written))
                return super().upload_context(context)

        builder = FakeBuilder()
        assert _run(Reading([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=config) == 0
        assert [image for image, _, _ in builder.completed] == ["a", "b"]
        # Handed on, the link no longer climbs out of the context to come back: it leads where it did, inside.
        assert seen == [("FROM api\n", "lib"), ("FROM web\n", None)]

    def test_a_shared_download_is_removed_once_its_last_image_has_had_its_turn(self, tmp_path: Path) -> None:
        """Two archives, two images each. While the second pair builds, the first archive is no longer held."""
        fetch = self._sources({"Dockerfile": "FROM scratch\n"})
        held: list[int] = []

        class Looking(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                held.append(sum(path.name.startswith("source-") for path in (tmp_path / "scratch").iterdir()))
                return super().upload_context(context)

        first, second = ({"fileset": "default/fs-a", "archive": f"{name}.tar.gz"} for name in ("one", "two"))
        config = self._set(first, first, second, second)
        assert _run(Looking([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 0
        assert (len(fetch.calls), held) == (2, [1, 1, 1, 1])  # ty: ignore[unresolved-attribute]

    def test_an_archive_that_is_refused_is_not_fetched_again_for_each_image_that_names_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One that unpacks to more than is allowed, say: it would be downloaded and unpacked as far as that
        for every image, to be refused the same way."""
        calls: list[str] = []

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            calls.append(image.image)
            raise ArchiveRefused("it unpacks to more than 4294967296 bytes")

        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        config = self._set(*(archive | {"context_path": name} for name in ("one", "two", "three")))
        with caplog.at_level(logging.ERROR):
            assert _run(FakeService([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 1
        assert calls == ["a"]
        for name in "abc":
            assert (
                f"{FAILURE_MARKER} {name}: cannot fetch its build context: it unpacks to more than 4294967296 bytes"
            ) in caplog.text
        # Nothing of the refused download is left in the step's directory.
        assert not [path for path in (tmp_path / "scratch").iterdir() if path.name.startswith("source-")]

    def test_a_context_named_by_a_link_is_built_from_a_set_as_it_is_alone(self, tmp_path: Path) -> None:
        """`current -> v2`, and in `v2` a link that comes back to it by its own name. The copy made for a set
        is of the directory, under the directory's name, so the link still leads inside it."""

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            for name in ("v2", "other"):
                (directory / name).mkdir()
                (directory / name / "Dockerfile").write_text(f"FROM {name}\n")
            (directory / "v2" / "again").symlink_to("../v2/Dockerfile")
            (directory / "current").symlink_to("v2")
            return directory

        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        config = self._set(archive | {"context_path": "current"}, archive | {"context_path": "other"})
        #: Each context's Dockerfile, and what its link back is written as, where it has one.
        seen: list[tuple[str, str | None]] = []

        class Reading(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                again = context.path / "again"
                written = os.readlink(again) if again.is_symlink() else None
                seen.append(((context.path / "Dockerfile").read_text(), written))
                return super().upload_context(context)

        builder = FakeBuilder()
        assert _run(Reading([DONE]), builder, tmp_path, tmp_path, fetch=fetch, config=config) == 0
        assert [image for image, _, _ in builder.completed] == ["a", "b"]
        # Handed on, the link no longer comes back by the directory's name: it leads where it did, inside.
        assert seen == [("FROM v2\n", "Dockerfile"), ("FROM other\n", None)]

    def test_a_directory_reached_by_a_link_out_of_a_shared_download_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The path itself is relative and climbs nowhere. A link in the download, where unpacking let one
        through, leads it out: to a directory of the step's own, which holds the service's credential."""
        outside = tmp_path / "mounted"
        outside.mkdir()
        (outside / "Dockerfile").write_text("FROM mounted\n")

        def fetch(image: RemoteBuildImage, directory: Path) -> Path:
            (directory / "v2").mkdir()
            (directory / "v2" / "Dockerfile").write_text("FROM v2\n")
            (directory / "current").symlink_to("v2")
            (directory / "up").symlink_to(outside)
            return directory

        archive = {"fileset": "default/fs-a", "archive": "task.tar.gz"}
        config = self._set(archive | {"context_path": "current"}, archive | {"context_path": "up"})
        builder, service = FakeBuilder(), FakeService([DONE])
        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, tmp_path, tmp_path, fetch=fetch, config=config) == 1
        # A link that stays inside the download is a way to a context like any other.
        assert [image for image, _, _ in builder.completed] == ["a"]
        assert len(service.started) == 1
        assert f"{FAILURE_MARKER} b: context path 'up' is not a directory inside its source" in caplog.text

    def test_a_download_only_one_image_names_is_fetched_where_it_is_built_and_removed_with_it(
        self, tmp_path: Path
    ) -> None:
        """Nothing shares it, so it is not kept for the rest of the set, nor copied: a set of large contexts
        then holds one at a time."""
        fetch = self._sources({"Dockerfile": "FROM scratch\n"})
        held: list[list[str]] = []

        class Looking(FakeService):
            def upload_context(self, context: RemoteContext) -> str:
                held.append(sorted(path.name[:6] for path in (tmp_path / "scratch").iterdir()))
                return super().upload_context(context)

        config = self._set({"fileset": "default/fs-a"}, {"fileset": "default/fs-b"})
        assert _run(Looking([DONE]), FakeBuilder(), tmp_path, tmp_path, fetch=fetch, config=config) == 0
        # One image's directory at a time, and no download kept beside it.
        assert held == [["image-"], ["image-"]]

    def test_a_set_of_one_image_is_built_where_it_was_fetched(self, context: Path, tmp_path: Path) -> None:
        """Nothing to share, so no second copy of what may be a large context."""
        service = FakeService([DONE])
        assert _run(service, FakeBuilder(), context, tmp_path) == 0
        assert service.uploaded[0].path == context


class LinkReads:
    """Every read of a link on this thread while this is entered, whoever makes it.

    Counted by a profile hook, which Python calls for each call of ``os.readlink``: the driver's own, and any
    the standard library makes for it. Nothing is patched.
    """

    def __init__(self) -> None:
        self.total = 0
        self._previous: Any = None

    def _see(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "c_call" and arg is os.readlink:
            self.total += 1

    def __enter__(self) -> LinkReads:
        self._previous = sys.getprofile()
        sys.setprofile(self._see)
        return self

    def __exit__(self, *raised: object) -> None:
        sys.setprofile(self._previous)


class TestFollowingAContextsLinks:
    """What it costs to learn where a context's links lead. An archive is small for the links it holds: a
    chain of a hundred thousand, each written as the name of the next, is under a megabyte."""

    #: The chains that are staged. Followed from scratch for each link, they took 1,056 reads of a link,
    #: 4,160, 16,512 and 65,792: their length times one more.
    CHAINS = (32, 64, 128, 256)

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        """Where the contexts are made, with no link in its own path: one there would be read on the way to
        each context, and counted."""
        return tmp_path.resolve()

    @staticmethod
    def _name(number: int) -> str:
        return f"link-{number:03d}"

    @classmethod
    def _chain(cls, directory: Path, links: int, *, ends_at: str = "payload") -> Path:
        """A build context, ``directory``, whose links are one chain: each is written as the name of the next,
        and the last as ``ends_at``."""
        directory.mkdir()
        (directory / "Dockerfile").write_text("FROM scratch\n")
        (directory / "payload").write_bytes(b"x")
        for number in range(links):
            (directory / cls._name(number)).symlink_to(cls._name(number + 1) if number + 1 < links else ends_at)
        return directory

    @staticmethod
    def _staged(context: Path) -> int:
        """How many times a link is read to stage ``context``, which is staged as it is."""
        with LinkReads() as reads:
            assert stage_context(context, "Dockerfile") == context
        return reads.total

    @pytest.mark.parametrize("links", CHAINS)
    def test_a_link_is_read_once_however_many_links_lead_through_it(self, links: int, root: Path) -> None:
        """Each link of a chain is read once to learn where it leads, though every link before it leads
        through it, once to see how it is written, and once more when the staged context is measured."""
        context = self._chain(root / "context", links)

        assert self._staged(context) <= 3 * links
        # Staged as it was: every link is still what it was written as.
        assert [os.readlink(context / self._name(number)) for number in range(links)] == [
            *(self._name(number) for number in range(1, links)),
            "payload",
        ]

    def test_reading_links_grows_no_faster_than_the_links(self, root: Path) -> None:
        """Twice the links is no more than twice the reads. It was four times, less a little: eight times
        the links took sixty-two times the reads."""
        reads = {links: self._staged(self._chain(root / f"context-{links}", links)) for links in self.CHAINS}

        for links in self.CHAINS[:-1]:
            assert reads[2 * links] <= 2 * reads[links]
        assert reads[256] <= 8 * reads[32]

    def test_a_chain_past_a_limit_is_refused_after_one_read_of_each_link(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chain is as many entries as it has links, and as many copies of the file it ends at. What says
        it comes to too much is asked once every link is followed, and that is no more than a read of each."""
        links = 128
        context = self._chain(root / "context", links)

        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", links + 1)
        with LinkReads() as reads, pytest.raises(ImageFailed, match=f"more than {links + 1} entries"):
            stage_context(context, "Dockerfile")
        assert reads.total <= links

    def test_a_build_from_a_chain_is_started_after_no_more_reads_than_that(self, root: Path) -> None:
        """Through the driver. The links are followed before the image's build timeout starts, and nothing
        looks at a stop meanwhile, so what they cost is the step's to bound."""
        links = 64
        context = self._chain(root / "context", links)
        read_by_the_start: list[int] = []

        with LinkReads() as reads:

            class Counting(FakeService):
                def start(self, build: RemoteBuild) -> str:
                    read_by_the_start.append(reads.total)
                    return super().start(build)

            builder = FakeBuilder()
            assert _run(Counting([DONE]), builder, context, root) == 0

        assert [image for image, _, _ in builder.completed] == ["demo-1"]
        assert read_by_the_start == [reads.total] and reads.total <= 3 * links

    def test_a_chain_longer_than_the_system_follows_is_staged_as_it_was(self, root: Path) -> None:
        """The system gives up on a path after a few tens of links. The driver gives up on none, as before:
        what it remembers of each link is what keeps a long chain cheap, not a limit on its length."""
        context = self._chain(root / "context", 128)
        with pytest.raises(OSError) as gave_up:
            os.stat(context / self._name(0))
        assert gave_up.value.errno == errno.ELOOP

        assert stage_context(context, "Dockerfile") == context
        assert os.readlink(context / self._name(127)) == "payload"

    @pytest.mark.parametrize("reached", ["the-dockerfile", "its-directory"])
    def test_a_dockerfile_behind_more_links_than_the_system_follows_is_refused(self, reached: str, root: Path) -> None:
        """Whether the Dockerfile's path leads to a file is asked of the system, which follows a few tens of
        links and no more: forty on Linux. A Dockerfile that is itself the first of more is no file to it,
        nor to anything else that opens a path through it. One in a directory behind more is no file to
        it either, and neither is the ignore file written for that Dockerfile, which is looked for beside
        the same path: found by a longer way, the Dockerfile would have been put at the root without it.
        One rule refuses both, and nothing is written into such a context."""
        if reached == "the-dockerfile":
            context = self._chain(root / "context", 64, ends_at="Dockerfile.real")
            (context / "Dockerfile.real").write_text("FROM real\n")
            (context / "Dockerfile").unlink()
            (context / "Dockerfile").symlink_to(self._name(0))
            dockerfile = "Dockerfile"
        else:
            context = self._chain(root / "context", 64, ends_at="real")
            (context / "real").mkdir()
            (context / "real" / "Dockerfile").write_text("FROM real\n")
            (context / "real" / "Dockerfile.dockerignore").write_text(".env\n")
            dockerfile = self._name(0) + "/Dockerfile"
        before = sorted(os.listdir(context))

        with pytest.raises(ImageFailed, match=f"dockerfile '{dockerfile}' is not a file inside the build context"):
            stage_context(context, dockerfile)
        assert sorted(os.listdir(context)) == before
        assert os.readlink(context / self._name(0)) == self._name(1)

    @pytest.mark.parametrize("reached", ["the-dockerfile", "its-directory"])
    def test_a_dockerfile_reached_by_a_few_links_is_staged_with_its_own_ignore_file(
        self, reached: str, root: Path
    ) -> None:
        """A chain the system follows: the Dockerfile it leads to goes to the root, and so does the ignore
        file beside the path the build names, which is what leaves the build's secrets out."""
        if reached == "the-dockerfile":
            context = self._chain(root / "context", 8, ends_at="Dockerfile.real")
            (context / "Dockerfile.real").write_text("FROM real\n")
            (context / "docker").mkdir()
            (context / "docker" / "Dockerfile").symlink_to("../" + self._name(0))
            (context / "docker" / "Dockerfile.dockerignore").write_text(".env\n")
            dockerfile = "docker/Dockerfile"
        else:
            context = self._chain(root / "context", 8, ends_at="real")
            (context / "real").mkdir()
            (context / "real" / "Dockerfile").write_text("FROM real\n")
            (context / "real" / "Dockerfile.dockerignore").write_text(".env\n")
            dockerfile = self._name(0) + "/Dockerfile"

        assert stage_context(context, dockerfile) == context
        assert ((context / "Dockerfile").is_symlink(), (context / "Dockerfile").read_text()) == (False, "FROM real\n")
        staged_ignore = context / "Dockerfile.dockerignore"
        assert (staged_ignore.is_symlink(), staged_ignore.read_text()) == (False, ".env\n")

    def test_a_small_archive_of_many_links_costs_no_more_to_stage_than_its_links(self, root: Path) -> None:
        """Unpacked as the step unpacks one: under four kilobytes of archive, 256 links, and no more than
        three reads of each."""
        links = 256
        archive = root / "context.tar.gz"
        with tarfile.open(archive, "w:gz") as packed:
            for name, content in (("Dockerfile", b"FROM scratch\n"), ("payload", b"x")):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                packed.addfile(member, io.BytesIO(content))
            for number in range(links):
                member = tarfile.TarInfo(self._name(number))
                member.type = tarfile.SYMTYPE
                member.linkname = self._name(number + 1) if number + 1 < links else "payload"
                packed.addfile(member)
        context = root / "unpacked"
        context.mkdir()

        assert archive.stat().st_size < 4096
        assert unpack(archive, context) == links + 2
        assert self._staged(context) <= 3 * links

    def test_a_path_leads_to_the_same_place_whichever_link_was_followed_first(self, root: Path) -> None:
        """`into -> round`, `round -> about`, `about -> round`, and `through -> into`. A link that is come
        back to while it is followed is left in the path as a name, so where a loop's links end rests on
        which of them is come to first. Nothing of that is remembered from one path to the next: a directory
        lists its entries in no promised order, and a context must not be staged one way or another by it."""
        written = {"into": "round", "round": "about", "about": "round", "through": "into"}
        for name, leads_to in written.items():
            (root / name).symlink_to(leads_to)

        led: set[tuple[str, ...]] = set()
        for order in itertools.permutations(written):
            follower = driver._Follower()
            ends = {name: follower.end(str(root), name) for name in order}
            led.add(tuple(ends[name] for name in written))

        assert led == {tuple(str(root / name) for name in ("round", "round", "about", "round"))}

    @pytest.mark.parametrize(
        ("costly", "limit"), [("many-links", 100), ("long-paths", 100), ("a-chain-that-ends-in-a-loop", 1000)]
    )
    def test_links_that_take_too_many_steps_to_follow_fail_the_image(
        self, costly: str, limit: int, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What is remembered bounds how often a link is read, not how long the paths are that links are
        written as. And nothing is remembered of a link that ends in a loop of links, whose end rests on
        which link of the loop is come to first: every link of a chain that ends in one is followed to the
        loop again. So every name the driver takes is counted, and past a limit the image fails and says
        so. At the limit as it ships, each of these is staged, as it was."""
        if costly == "many-links":
            context = self._chain(root / "context", 64)
        elif costly == "long-paths":
            context = self._chain(root / "context", 0)
            for number in range(8):
                (context / f"alias-{number}").symlink_to("./" * 100 + "payload")
        else:
            context = self._chain(root / "context", 64, ends_at="round")
            (context / "round").symlink_to("about")
            (context / "about").symlink_to("round")

        monkeypatch.setattr(driver, "MAX_LINK_STEPS", limit)
        with pytest.raises(ImageFailed, match=f"the build context's links take more than {limit} steps to follow"):
            stage_context(context, "Dockerfile")
        if costly == "a-chain-that-ends-in-a-loop":
            # As long a chain that ends at a file is within this limit: each of its links is followed once.
            assert self._staged(self._chain(root / "ends-at-a-file", 64)) <= 3 * 64
        monkeypatch.undo()

        assert driver.MAX_LINK_STEPS == 100 * MAX_UNPACKED_ENTRIES
        assert stage_context(context, "Dockerfile") == context
        assert os.readlink(context / "alias-0" if costly == "long-paths" else context / self._name(63)) == (
            {"many-links": "payload", "long-paths": "./" * 100 + "payload"}.get(costly, "round")
        )

    def test_an_image_whose_links_take_too_long_to_follow_is_failed_before_any_build_is_started(
        self, root: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the driver. The links are followed before the image's build timeout starts, and nothing
        looks at a stop meanwhile: the limit is what ends it."""
        context = self._chain(root / "context", 64)
        monkeypatch.setattr(driver, "MAX_LINK_STEPS", 100)
        service, builder = FakeService([DONE]), FakeBuilder()

        with caplog.at_level(logging.ERROR):
            assert _run(service, builder, context, root) == 1

        assert (service.started, builder.completed) == ([], [])
        assert f"{FAILURE_MARKER} demo-1: the build context's links take more than 100 steps to follow" in caplog.text

    def test_the_names_of_a_chain_are_counted_before_they_are_held(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each link is written as the next one, then a hundred names. The next link is the first name
        taken, so every link of the chain is read before any of the names after one is: counted only as
        they were taken, sixty-four links' names were all held by the sixty-fourth step. Counted as each
        link is read, the limit is reached at the tenth link, and no more are read."""
        context = self._chain(root / "context", 0)
        for number in range(64):
            (context / self._name(number)).symlink_to(self._name(number + 1) + "/." * 100)

        monkeypatch.setattr(driver, "MAX_LINK_STEPS", 1000)
        with (
            LinkReads() as reads,
            pytest.raises(ImageFailed, match="the build context's links take more than 1000 steps to follow"),
        ):
            stage_context(context, "Dockerfile")
        assert reads.total <= 10

    #: Names that are not there, as a link is written with them after the link it leads to: 800 characters.
    NOT_THERE = "/missing" * 100

    def test_links_that_each_add_names_that_are_not_there_fail_the_image_before_their_paths_grow(
        self, root: Path
    ) -> None:
        """Each link is written as the next one, then a hundred names that are not there. A name that isn't
        there stays in the path, so the first link ends at every link's names, one after another, and each
        name costs the path so far to take: sixty-four links are 6,400 steps, far inside the limit on steps,
        over a path that comes to fifty thousand characters, and a thousand links would be hours of it.
        Nothing is at a path the system can't read, so the image fails once one is longer than any one link
        leads to, and where a link ended is never remembered as more than that."""
        context = self._chain(root / "context", 0)
        for number in range(64):
            (context / self._name(number)).symlink_to(self._name(number + 1) + self.NOT_THERE)
        too_long = f"links lead to a path of more than {driver.MAX_LINK_PATH_CHARS} characters, where nothing can be"

        follower = driver._Follower()
        with pytest.raises(ImageFailed, match=too_long):
            follower.end(str(context), self._name(0))
        # Every link was read, and its names counted: the first link's own name, and 101 for each. Given up
        # while the path was short, on the way back from the last link, with only the last few links ended.
        assert follower.steps == 1 + 64 * 101
        assert 0 < len(follower._ends) < 64
        assert max(map(len, follower._ends.values())) <= driver.MAX_LINK_PATH_CHARS

        with pytest.raises(ImageFailed, match=too_long):
            stage_context(context, "Dockerfile")

    def test_a_dockerfile_that_is_the_first_of_such_links_is_refused_with_none_of_them_followed(
        self, root: Path
    ) -> None:
        """The Dockerfile's path is a submitter's, and so is every link on the way to it. Whether it leads
        to a file is asked of the system, which gives up on a chain after a few tens of links, before the
        step follows the path for itself to see that it stays inside the context: followed first, with no
        limit, the path came to twelve thousand characters to learn that no file is there, and each name
        of it cost the path so far. Each link adds few names, and the links are many, so that it is the
        links the system gives up for."""
        links, not_there = 512, "/missing" * 3
        context = self._chain(root / "context", 0)
        (context / "Dockerfile").unlink()
        (context / "Dockerfile").symlink_to(self._name(0) + not_there)
        for number in range(links):
            (context / self._name(number)).symlink_to(self._name(number + 1) + not_there)
        assert links * len(not_there) > driver.MAX_LINK_PATH_CHARS

        with LinkReads() as reads, pytest.raises(ImageFailed, match="dockerfile 'Dockerfile' is not a file inside"):
            stage_context(context, "Dockerfile")
        assert reads.total == 0

    def test_a_path_the_system_refuses_outright_is_no_file_and_no_directory(self, root: Path) -> None:
        """A name longer than the system allows: it answers with an error, not that nothing is there. The
        image fails as for any path that leads to no file, or to no directory, and not in the words of
        the system's error, which a path's own `is_file` raises on some Pythons and not on others."""
        context = self._chain(root / "context", 0)
        too_long = "a" * 300
        with pytest.raises(OSError) as refused:
            os.stat(context / too_long)
        assert refused.value.errno == errno.ENAMETOOLONG

        with pytest.raises(ImageFailed, match="is not a file inside the build context"):
            stage_context(context, f"{too_long}/Dockerfile")
        with pytest.raises(ImageFailed, match="its build context is not a directory in what was fetched"):
            stage_context(context / too_long, "Dockerfile")
        assert sorted(os.listdir(context)) == ["Dockerfile", "payload"]

    def test_a_context_path_that_is_the_first_of_such_links_is_refused_with_none_of_them_followed(
        self, root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One image's own download, unpacked as the step unpacks it, whose context path names the first
        link of the chain. Whether that is a directory is asked of the system, which gives up on a chain
        after a few tens of links, before the step follows the path for itself to see that it stays
        inside its source: followed first, with no limit, the path came to twelve thousand characters to
        learn that nothing is there. Each link adds few names, and the links are many: followed, the path
        would come to more than the longest the step follows."""
        links, not_there = 512, "/missing" * 3
        entries: Entries = {"app": Link(self._name(0) + not_there)}
        entries |= {self._name(number): Link(self._name(number + 1) + not_there) for number in range(links)}
        fetch, _ = _fetching(entries)

        with LinkReads() as reads:
            outcome = _built(root, caplog, fetch, "app")

        assert (outcome.exit, outcome.handed, outcome.completed) == (1, {}, [])
        assert outcome.reasons == {
            "a": "its build context is not a directory in what was fetched: its context path matches none"
        }
        assert reads.total == 0

    def test_a_path_within_what_one_link_can_lead_to_is_followed_though_nothing_is_there(self, root: Path) -> None:
        """What is refused is a path no one link could be written as. A link that leads nowhere, by as many
        names as its own path holds, is handed on as it was, and so is a short chain of them."""
        context = self._chain(root / "context", 0)
        (context / "alone").symlink_to("nowhere" + self.NOT_THERE)
        for number in range(4):
            (context / self._name(number)).symlink_to(self._name(number + 1) + self.NOT_THERE)
        assert len(str(context)) + 5 * len(self._name(0) + self.NOT_THERE) < driver.MAX_LINK_PATH_CHARS

        assert stage_context(context, "Dockerfile") == context
        assert os.readlink(context / "alone") == "nowhere" + self.NOT_THERE
        assert os.readlink(context / self._name(0)) == self._name(1) + self.NOT_THERE

    def test_no_one_link_leads_to_a_path_as_long_as_the_longest_followed(self) -> None:
        """The system reads no path longer than 4,096 characters, so a link's directory and the path it is
        written as are each within that: twice it is more than any one link comes to."""
        assert driver.MAX_LINK_PATH_CHARS == 2 * 4096


@dataclass(frozen=True)
class Link:
    """A symbolic link in an archive, with its path as the archive writes it."""

    to: str


@dataclass(frozen=True)
class SameFileAs:
    """A hard link in an archive: another name for the member ``to``, a path from the archive's root."""

    to: str


Entries = dict[str, bytes | Link | SameFileAs]

FILESET = "default/fs-a"
ARCHIVE = "task.tar"
#: Why staging refuses a context for one of its links, before the link's name.
LEADS_OUT = "the build context has a link that leads out of it, or back to a directory it is in: "


def _archive(entries: Entries) -> bytes:
    """A tar of ``entries``: what a submitter puts in a fileset."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            if isinstance(content, Link):
                info.type, info.linkname = tarfile.SYMTYPE, content.to
                tar.addfile(info)
            elif isinstance(content, SameFileAs):
                info.type, info.linkname = tarfile.LNKTYPE, content.to
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class FakeFiles:
    """The one call the step makes of Files for an archive, with each download noted."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.downloads: list[str] = []

    def download_file(self, *, workspace: str, name: str, path: str) -> Any:
        self.downloads.append(path)
        content = self.files[path]

        class _Body:
            @contextmanager
            def stream(self, chunk_size: int | None = None) -> Generator[Iterator[bytes], None, None]:
                size = chunk_size or 7
                yield iter([content[i : i + size] for i in range(0, len(content), size)])

        return _Body()


def _fetching(entries: Entries) -> tuple[Callable[[RemoteBuildImage, Path], Path], FakeFiles]:
    """The step's own fetch, as ``step_main`` passes it, of an archive that holds ``entries``."""
    files = FakeFiles({ARCHIVE: _archive(entries)})

    def fetch(image: RemoteBuildImage, directory: Path) -> Path:
        return fetch_context(cast(FilesClient, files), image, workspace="default", root=directory / "fetched")

    return fetch, files


def _archive_set(*contexts: str | None, dockerfile: str = "Dockerfile") -> RemoteBuildStepConfig:
    """A set with one image, named a, b, c..., for each of ``contexts``: directories of the one archive."""
    return RemoteBuildStepConfig.model_validate(
        {
            "images": [
                {
                    "image": name,
                    "source": {"fileset": FILESET, "archive": ARCHIVE, "context_path": context},
                    "dockerfile": dockerfile,
                    "platform": "linux/amd64",
                }
                for name, context in zip("abcdef", contexts, strict=False)
            ],
            "registry": "registry.example.com",
            "repository": "team/images",
            "build_timeout_seconds": 600,
        }
    )


@dataclass(frozen=True)
class Handed:
    """A build context as the build service was handed it."""

    #: Every name in it, from its root.
    names: tuple[str, ...]
    #: Each link in it, with the path it is written as.
    links: dict[str, str]
    #: What each file reads as, through a link too. None for a link that leads to nothing.
    reads: dict[str, bytes | None]

    @classmethod
    def of(cls, context: Path) -> Handed:
        paths = sorted(context.rglob("*"))
        names = tuple(path.relative_to(context).as_posix() for path in paths)
        links = {name: os.readlink(path) for name, path in zip(names, paths, strict=True) if path.is_symlink()}
        reads: dict[str, bytes | None] = {}
        for name, path in zip(names, paths, strict=True):
            if path.is_file():
                reads[name] = path.read_bytes()
            elif path.is_symlink() and not path.exists():
                reads[name] = None
        return cls(names, links, reads)


class Inspecting(FakeService):
    """A build service that looks at each context it is handed, and builds it."""

    def __init__(self) -> None:
        super().__init__([DONE])
        self.handed: dict[str, Handed] = {}

    def upload_context(self, context: RemoteContext) -> str:
        self.handed[context.labels["image"]] = Handed.of(context.path)
        return super().upload_context(context)


@dataclass
class Outcome:
    """What one run of the step came to."""

    #: 0 if every image was built.
    exit: int
    #: Each context the service was handed, by its image.
    handed: dict[str, Handed]
    #: The images whose rows were completed, in order.
    completed: list[str]
    #: Why each image that failed did, as its failure line says it.
    reasons: dict[str, str]


def _built(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fetch: Callable[[RemoteBuildImage, Path], Path],
    *contexts: str | None,
    service: Inspecting | None = None,
) -> Outcome:
    """Build a set with one image for each of ``contexts``, directories of one archive, through the driver."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    service, builder = service or Inspecting(), FakeBuilder()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        code = _run(service, builder, tmp_path, tmp_path, fetch=fetch, config=_archive_set(*contexts))
    reasons: dict[str, str] = {}
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(FAILURE_MARKER + " "):
            image, _, reason = message.removeprefix(FAILURE_MARKER + " ").partition(": ")
            reasons.setdefault(image, reason)
    return Outcome(code, service.handed, [image for image, _, _ in builder.completed], reasons)


#: `app/cached -> ../beside/alias -> ../app/payload`: a link of the context that reaches a file of the context
#: by way of a link beside it. `tools` is a second directory of the archive, for a second spec to build.
BY_A_LINK_BESIDE_IT: Entries = {
    "app/Dockerfile": b"FROM scratch\nCOPY . /app\n",
    "app/payload": b"the payload\n",
    "beside/alias": Link("../app/payload"),
    "app/cached": Link("../beside/alias"),
    "tools/Dockerfile": b"FROM scratch\n",
}
#: What a build service is handed for `app` when nothing shares the download: the link leads where it did, by
#: a path that stays inside the context wherever the context is unpacked.
BUILT_ALONE = Handed(
    names=("Dockerfile", "cached", "payload"),
    links={"cached": "payload"},
    reads={"Dockerfile": b"FROM scratch\nCOPY . /app\n", "cached": b"the payload\n", "payload": b"the payload\n"},
)

#: A second directory of each archive below, for the spec that is added.
TOOLS: Entries = {"tools/Dockerfile": b"FROM scratch\n"}
APP: Entries = {"app/Dockerfile": b"FROM scratch\n", "app/payload": b"the payload\n"} | TOOLS

#: Contexts built alone and from a set: the archive, the first spec's context path, and why the first image is
#: refused, or None if it is built.
ALONE_AND_FROM_A_SET: dict[str, tuple[Entries, str | None, str | None]] = {
    # A link that never leaves the context.
    "a-link-inside-the-context": (APP | {"app/cached": Link("payload")}, "app", None),
    # One that comes back by the context's own name.
    "back-by-the-contexts-own-name": (APP | {"app/cached": Link("../app/payload")}, "app", None),
    # One that comes back by the archive's own directory.
    "back-by-the-downloads-own-name": (APP | {"app/cached": Link(f"../../{ARCHIVE}/app/payload")}, "app", None),
    # One that climbs past the archive's own directory and comes back by the names the step unpacks under: the
    # fileset's and the archive's. Unpacking lets it through, since it ends inside the archive.
    "back-by-the-names-above-the-download": (
        APP | {"app/cached": Link(f"../../../fs-a/{ARCHIVE}/app/payload")},
        "app",
        None,
    ),
    # The same of an image that builds the whole archive.
    "back-by-the-names-above-the-download-in-the-whole-archive": (
        {
            "Dockerfile": b"FROM scratch\n",
            "payload": b"the payload\n",
            "cached": Link(f"../../fs-a/{ARCHIVE}/payload"),
        }
        | TOOLS,
        None,
        None,
    ),
    # One that passes a link to a directory, beside the context, that leads back to the context.
    "by-way-of-a-link-to-a-directory-beside-it": (
        APP | {"beside/up": Link("../app"), "app/cached": Link("../beside/up/payload")},
        "app",
        None,
    ),
    # A link to a directory of the context, by way of a link beside the context.
    "a-link-to-a-directory-by-way-of-one-beside-it": (
        APP | {"app/vendor/lib.py": b"", "beside/libs": Link("../app/vendor"), "app/lib": Link("../beside/libs")},
        "app",
        None,
    ),
    # The Dockerfile itself is such a link.
    "the-dockerfile-by-way-of-a-link-beside-it": (
        {
            "app/Dockerfile.real": b"FROM scratch\n",
            "app/payload": b"the payload\n",
            "beside/dockerfile": Link("../app/Dockerfile.real"),
            "app/Dockerfile": Link("../beside/dockerfile"),
        }
        | TOOLS,
        "app",
        None,
    ),
    # The context's own ignore file is such a link.
    "the-ignore-file-by-way-of-a-link-beside-it": (
        APP
        | {
            "app/rules": b"payload\n",
            "beside/rules": Link("../app/rules"),
            "app/.dockerignore": Link("../beside/rules"),
        },
        "app",
        None,
    ),
    # The context path names a link to the directory, and a link in it comes back by that name.
    "back-by-the-link-the-context-path-names": (
        {
            "v2/Dockerfile": b"FROM scratch\n",
            "v2/payload": b"the payload\n",
            "current": Link("v2"),
            "v2/cached": Link("../current/payload"),
        }
        | TOOLS,
        "current",
        None,
    ),
    # Two links that lead to each other, one of them beside the context. The one in the context leads nowhere,
    # and is handed on as a link to itself.
    "a-loop-by-way-of-a-link-beside-it": (
        APP | {"beside/round": Link("../app/round"), "app/round": Link("../beside/round")},
        "app",
        None,
    ),
    # One that leads nowhere, inside the context: left as it is.
    "a-link-to-nothing-inside-the-context": (APP | {"app/cached": Link("nothing-here")}, "app", None),
    # Another name for a file beside the context, not a link to it: the copy has the file's bytes.
    "one-file-under-a-name-beside-the-context": (
        APP | {"beside/big": b"shared bytes\n", "app/cached": SameFileAs("beside/big")},
        "app",
        None,
    ),
    # A link to a file beside the context: refused, as it must be.
    "a-link-to-a-file-beside-the-context": (
        APP | {"beside/answer": b"not this build's\n", "app/cached": Link("../beside/answer")},
        "app",
        LEADS_OUT + "'cached'",
    ),
    # A link to nothing beside the context: refused too.
    "a-link-to-nothing-beside-the-context": (
        APP | {"app/cached": Link("../beside/nothing")},
        "app",
        LEADS_OUT + "'cached'",
    ),
    # A Dockerfile that is a link to a file beside the context: refused before its links are looked at.
    "a-dockerfile-that-links-to-a-file-beside-the-context": (
        {"beside/Dockerfile": b"FROM beside\n", "app/Dockerfile": Link("../beside/Dockerfile")} | TOOLS,
        "app",
        "dockerfile 'Dockerfile' is not a file inside the build context",
    ),
}


class TestAContextFromASharedDownload:
    """A download only one image names is staged where it was unpacked, with the rest of an archive still
    beside the image's directory. One that several images name is unpacked once, and each image stages a copy
    of its own directory. An image is handed the same context either way."""

    def test_an_image_that_shares_its_download_is_handed_the_context_it_is_handed_alone(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`cached` reaches the context's own `payload` by way of a link beside the context, which a copy of
        the directory alone does not hold. Where each link leads is decided where the archive was unpacked,
        so adding a second spec of the same archive does not fail the first."""
        alone = _built(tmp_path / "alone", caplog, _fetching(BY_A_LINK_BESIDE_IT)[0], "app")
        fetch, files = _fetching(BY_A_LINK_BESIDE_IT)

        shared = _built(tmp_path / "set", caplog, fetch, "app", "tools")

        assert (alone.exit, alone.completed, alone.reasons) == (0, ["a"], {})
        assert alone.handed == {"a": BUILT_ALONE}
        # One download, of the archive whole, served both images.
        assert files.downloads == [ARCHIVE]
        assert (shared.exit, shared.completed, shared.reasons) == (0, ["a", "b"], {})
        assert shared.handed["a"] == BUILT_ALONE

    def test_two_images_of_one_directory_are_each_handed_it_as_one_is_alone(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two specs that build `app` itself, for two platforms say, name one download too."""
        twice = _built(tmp_path, caplog, _fetching(BY_A_LINK_BESIDE_IT)[0], "app", "app")

        assert (twice.exit, twice.completed, twice.reasons) == (0, ["a", "b"], {})
        assert twice.handed == {"a": BUILT_ALONE, "b": BUILT_ALONE}

    def test_the_download_is_as_it_was_unpacked_for_the_image_that_builds_all_of_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The first image's link is rewritten in its own copy. The second builds the whole archive, in which
        the same link stays inside the context as it is written: had the download been rewritten for the
        first, the second would be handed the first image's link."""
        entries = BY_A_LINK_BESIDE_IT | {"Dockerfile": b"FROM scratch\n"}

        shared = _built(tmp_path, caplog, _fetching(entries)[0], "app", None)

        assert (shared.exit, shared.completed, shared.reasons) == (0, ["a", "b"], {})
        assert shared.handed["a"] == BUILT_ALONE
        assert shared.handed["b"].links == {"app/cached": "../beside/alias", "beside/alias": "../app/payload"}
        assert shared.handed["b"].reads["app/cached"] == b"the payload\n"

    def test_an_images_copy_is_of_its_directory_alone_and_nothing_is_written_where_it_was_unpacked(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """What the first image's context is uploaded from, and what the download is at that moment."""
        scratch = (tmp_path / "scratch").resolve()
        #: Where the context is, what the image's directory holds, and each link of the download as written.
        seen: dict[str, Any] = {}

        class Noting(Inspecting):
            def upload_context(self, context: RemoteContext) -> str:
                if context.labels["image"] == "a":
                    (download,) = scratch.glob(f"source-*/fetched/unpacked/{FILESET}/{ARCHIVE}")
                    seen["where"] = context.path.relative_to(scratch).parts
                    seen["copied"] = sorted(
                        path.relative_to(context.path.parent).as_posix() for path in context.path.parent.rglob("*")
                    )
                    seen["download"] = Handed.of(download)
                return super().upload_context(context)

        shared = _built(tmp_path, caplog, _fetching(BY_A_LINK_BESIDE_IT)[0], "app", "tools", service=Noting())

        assert shared.handed["a"] == BUILT_ALONE
        # The copy is in the image's own directory, under the download's name and then the directory's.
        image_directory, *placed = seen["where"]
        assert image_directory.startswith("image-") and placed == [ARCHIVE, "app"]
        # It holds the directory, and nothing of the rest of the archive.
        assert seen["copied"] == ["app", "app/Dockerfile", "app/cached", "app/payload"]
        # The download is as it was unpacked: the link that was rewritten in the copy is as the archive wrote it.
        assert seen["download"].links == {"app/cached": "../beside/alias", "beside/alias": "../app/payload"}
        assert seen["download"].names == (
            "app",
            "app/Dockerfile",
            "app/cached",
            "app/payload",
            "beside",
            "beside/alias",
            "tools",
            "tools/Dockerfile",
        )

    @pytest.mark.parametrize("case", list(ALONE_AND_FROM_A_SET))
    def test_a_context_is_handed_on_or_refused_the_same_alone_and_from_a_set(
        self, case: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Built with a second spec of the same archive, each of these contexts is handed to the service
        exactly as it is alone, the same names, links and contents, or is refused for the same reason."""
        entries, context, refused = ALONE_AND_FROM_A_SET[case]

        alone = _built(tmp_path / "alone", caplog, _fetching(entries)[0], context)
        shared = _built(tmp_path / "set", caplog, _fetching(entries)[0], context, "tools")

        assert alone.reasons == shared.reasons == ({} if refused is None else {"a": refused})
        assert (alone.completed, shared.completed) == ((["a"], ["a", "b"]) if refused is None else ([], ["b"]))
        assert ("a" in alone.handed) is (refused is None)
        assert shared.handed.get("a") == alone.handed.get("a")
        # No link that is handed on climbs out of the context to come back.
        for written in shared.handed["a"].links.values() if refused is None else ():
            assert PurePosixPath(written).parts[0] != ".."

    @pytest.mark.parametrize("unpacked", ["as-the-archive-wrote-it", "as-the-step-unpacks-it"])
    def test_a_link_that_leads_beside_the_context_by_a_step_back_from_a_link_is_refused_from_a_set_too(
        self, unpacked: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`app/cached -> ../beside/far/../../app/payload`, where `beside/far` is a link to a directory three
        levels down. Followed where the archive was unpacked, the path ends at `other/app/payload`, a file
        beside the context. In a copy of `app` alone, `beside/far` is a name with nothing behind it, the two
        steps back are taken from the name, and the path ends at the context's own `payload`: checked there,
        the context was built from a set, with the link made one to a file it never led to.

        An interpreter that shortens a link's path as it unpacks leaves `cached -> ../app/payload`, which
        leads to the context's own file wherever it is. So the files are also written here as unpacking
        leaves them with each path kept as the archive wrote it. Either way, the two ways agree, and on the
        one outcome the link allows as it was unpacked: read from each download, the archive's own path is
        refused, and the shortened one is built with the context's own file."""
        files: Entries = APP | {
            "app/payload": b"the context's own\n",
            "other/app/payload": b"another directory's\n",
            "other/deep/deeper/keep": b"",
            "beside/far": Link("../other/deep/deeper"),
            "app/cached": Link("../beside/far/../../app/payload"),
        }

        def as_written(image: RemoteBuildImage, directory: Path) -> Path:
            root = Path(WorkLayout(PurePosixPath(directory / "fetched")).archive(FILESET, ARCHIVE))
            for name, content in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Link):
                    (root / name).symlink_to(content.to)
                elif isinstance(content, bytes):
                    (root / name).write_bytes(content)
            return root / image.source.context_path if image.source.context_path else root

        fetch = as_written if unpacked == "as-the-archive-wrote-it" else _fetching(files)[0]
        #: The path `app/cached` is written as in each download, read before anything is staged from it.
        spelt: list[str] = []

        def noting(image: RemoteBuildImage, directory: Path) -> Path:
            context = fetch(image, directory)
            root = Path(WorkLayout(PurePosixPath(directory / "fetched")).archive(FILESET, ARCHIVE))
            spelt.append(os.readlink(root / "app/cached"))
            return context

        alone = _built(tmp_path / "alone", caplog, noting, "app")
        shared = _built(tmp_path / "set", caplog, noting, "app", "tools")

        as_archived, shortened = "../beside/far/../../app/payload", "../app/payload"
        # One download for each run, with the link written the same way in both.
        assert spelt in ([as_archived] * 2, [shortened] * 2)
        if unpacked == "as-the-archive-wrote-it":
            assert spelt[0] == as_archived
        if spelt[0] == as_archived:
            # Followed where it was unpacked, the path ends beside the context: refused, alone and from a set.
            assert alone.reasons == shared.reasons == {"a": LEADS_OUT + "'cached'"}
            assert (alone.completed, shared.completed) == ([], ["b"])
            assert "a" not in alone.handed and "a" not in shared.handed
        else:
            # Shortened, it leads to the context's own file wherever it is: built, alone and from a set.
            assert alone.reasons == shared.reasons == {}
            assert (alone.completed, shared.completed) == (["a"], ["a", "b"])
            assert alone.handed["a"].reads["cached"] == b"the context's own\n"
            assert shared.handed["a"] == alone.handed["a"]


def _regular_bytes(root: Path) -> int:
    """The bytes of the regular files under ``root``, with no link followed: what a packager that packs a
    link as a link sends in file content."""
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            status = os.lstat(Path(directory, name))
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


def _followed_bytes(root: Path) -> int:
    """The bytes a packager that follows links sends: a link to a file counts as the file, each time the walk
    comes to it. A link that leads nowhere counts for nothing."""
    total = 0
    for directory, _, names in os.walk(root, followlinks=True):
        for name in names:
            try:
                status = os.stat(Path(directory, name))
            except OSError:
                continue
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


def _names(root: Path) -> list[str]:
    """Every name under ``root``, from it."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def _sized(first_line: bytes, size: int) -> bytes:
    """``size`` bytes that start with ``first_line``. The rest is one comment line."""
    content = first_line + b"\n" + b"#" * (size - len(first_line) - 2) + b"\n"
    assert len(content) == size
    return content


class TestWhatStagingAdds:
    """The limits a source is held to as it is unpacked hold for the context a service is handed. Staging adds
    to a context: the named Dockerfile at its root, its ignore file beside it, and what any link to either
    then leads to. The limit on bytes is scaled to 100 here, as the tests of the measure scale it."""

    LIMIT = 100
    #: The Dockerfile the spec names, which is not at the context's root, and its own ignore file.
    NAMED = "services/feed/Dockerfile"
    NAMED_IGNORE = NAMED + ".dockerignore"
    #: A context of exactly the limit: 40 bytes of Dockerfile, 20 of its ignore file, and 40 of anything else.
    WITH_IGNORE: dict[str, bytes] = {
        NAMED: _sized(b"FROM scratch", 40),
        NAMED_IGNORE: _sized(b"*.log", 20),
        "payload.bin": b"\0" * 40,
    }
    #: Five links to the root Dockerfile, which the context doesn't hold: they lead nowhere until it is staged.
    LINKS = {f"images/{name}/Dockerfile": "../../Dockerfile" for name in ("api", "web", "worker", "cron", "docs")}

    @staticmethod
    def _context(directory: Path, files: dict[str, bytes], links: dict[str, str] | None = None) -> Path:
        """A build context, ``directory``, of ``files`` and of ``links`` as symbolic links."""
        for name, content in files.items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_bytes(content)
        for name, leads_to in (links or {}).items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).symlink_to(leads_to)
        return directory

    @pytest.fixture
    def limit(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
        """Sets the limit on bytes, under both names the code reads it by: the fetch's own, and the driver's."""

        def scale(size: int) -> None:
            monkeypatch.setattr(fetch_step, "MAX_UNPACKED_BYTES", size)
            monkeypatch.setattr(driver, "MAX_UNPACKED_BYTES", size)

        return scale

    @pytest.mark.parametrize("built", ["alone", "from-a-set"])
    @pytest.mark.parametrize(
        ("files", "staged"),
        [
            pytest.param({NAMED: _sized(b"FROM scratch", 60), "payload.bin": b"\0" * 40}, 160, id="its-dockerfile"),
            pytest.param(WITH_IGNORE, 160, id="its-dockerfile-and-ignore-file"),
        ],
    )
    def test_a_context_that_its_named_dockerfile_carries_past_the_limit_is_refused_before_anything_is_written(
        self, files: dict[str, bytes], staged: int, built: str, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """A hundred bytes of files, within a limit of a hundred, and no link. Staged, the named Dockerfile is
        at the root as well, and its ignore file beside it: 160 bytes, which the sizes of the files to be
        written tell before any is. So does a context staged in a copy, for an image of a set."""
        context = self._context(tmp_path / "fetched", files)
        into = tmp_path / "copy" / "context" if built == "from-a-set" else None
        before = _names(context)
        assert _regular_bytes(context) == self.LIMIT

        limit(staged - 1)
        with pytest.raises(ImageFailed, match=f"the build context comes to more than {staged - 1} bytes"):
            stage_context(context, self.NAMED, into=into)
        # Nothing was written where a Dockerfile goes, nor left behind.
        for directory in (context, *([into] if into is not None else [])):
            assert _names(directory) == before

        limit(staged)
        handed = stage_context(context, self.NAMED, into=None if into is None else tmp_path / "again" / "context")
        assert (handed / "Dockerfile").read_bytes() == files[self.NAMED]
        assert (_regular_bytes(handed), _followed_bytes(handed)) == (staged, staged)
        # Staged where it was fetched, when it is the image's alone. Where it is not, that is as it was.
        assert (handed == context.resolve()) is (into is None)
        assert into is None or _names(context) == before

    @pytest.mark.parametrize(
        ("files", "links", "before", "after"),
        [
            pytest.param(WITH_IGNORE, LINKS, (100, 100), (160, 360), id="links-that-lead-nowhere-until-it-is-staged"),
            pytest.param(
                {
                    "Dockerfile": b"FROM tiny\n",
                    NAMED: _sized(b"FROM scratch", 40),
                    NAMED_IGNORE: _sized(b"*.log", 20),
                    "payload.bin": b"\0" * 10,
                },
                {"images/api/Dockerfile": "../../Dockerfile", "images/web/Dockerfile": "../../Dockerfile"},
                (80, 100),
                (130, 210),
                id="links-to-a-root-dockerfile-that-is-replaced",
            ),
            pytest.param(
                {
                    "docker/Dockerfile.prod": b"FROM tiny\n",
                    NAMED: _sized(b"FROM scratch", 40),
                    NAMED_IGNORE: _sized(b"*.log", 20),
                    "payload.bin": b"\0" * 10,
                },
                # The root Dockerfile is a link itself. Followed to its end, the link to it leads to another
                # file than the root one before it is staged, and to the root one after.
                {"Dockerfile": "docker/Dockerfile.prod", "images/api/Dockerfile": "../../Dockerfile"},
                (80, 100),
                (140, 180),
                id="a-link-to-a-root-dockerfile-that-was-a-link-itself",
            ),
        ],
    )
    def test_what_a_link_to_the_root_dockerfile_leads_to_once_it_is_staged_is_counted(
        self,
        files: dict[str, bytes],
        links: dict[str, str],
        before: tuple[int, int],
        after: tuple[int, int],
        tmp_path: Path,
        limit: Callable[[int], None],
    ) -> None:
        """A context that comes to a hundred bytes with its links followed, as a packager that follows them
        would send it. Its links to the root `Dockerfile` lead to nothing, or to a few bytes, until the named
        Dockerfile is written there: then each of them is another copy of it.

        The files to be written do not tell this by their sizes: with them, the context's own files come to
        the first number of ``after``, and a limit of as much lets it through to be staged. Only the measure
        of what was staged finds the second."""
        regular, followed = after
        for name, size in (("as-its-files-alone-allow", regular), ("one-byte-short", followed - 1)):
            context = self._context(tmp_path / name, files, links)
            assert (_regular_bytes(context), _followed_bytes(context)) == before
            limit(size)
            with pytest.raises(ImageFailed, match=f"more than {size} bytes with its links followed"):
                stage_context(context, self.NAMED)

        context = self._context(tmp_path / "at-the-limit", files, links)
        limit(followed)
        staged = stage_context(context, self.NAMED)
        assert (_regular_bytes(staged), _followed_bytes(staged)) == after
        # The root Dockerfile is a file, the named one, and each link elsewhere is still a link, and leads to it.
        assert not (staged / "Dockerfile").is_symlink()
        for link in links:
            assert (staged / link).read_bytes() == files[self.NAMED]
            assert (staged / link).is_symlink() or link == "Dockerfile"

    def test_every_file_staging_writes_is_counted_in_place_of_what_it_replaces(
        self, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """Three files are written here. The named Dockerfile, over the context's own root one. Its ignore
        file, over the root one's. And the context's `.dockerignore`, a link to the ignore file that goes,
        as a copy of what it led to. The context comes to 100 bytes with its link followed, and to 145 staged:
        told before any of the three is written."""
        files = {
            self.NAMED: _sized(b"FROM scratch", 40),
            self.NAMED_IGNORE: _sized(b"*.log", 20),
            "payload.bin": b"\0" * 20,
            "Dockerfile": b"FROM tiny\n",
            "Dockerfile.dockerignore": b"*.md\n",
        }
        context = self._context(tmp_path / "context", files, {".dockerignore": "Dockerfile.dockerignore"})
        before = Handed.of(context)
        assert (_regular_bytes(context), _followed_bytes(context)) == (95, self.LIMIT)

        limit(144)
        with pytest.raises(ImageFailed, match="more than 144 bytes with its links followed"):
            stage_context(context, self.NAMED)
        assert Handed.of(context) == before

        limit(145)
        staged = stage_context(context, self.NAMED)
        assert (_regular_bytes(staged), _followed_bytes(staged)) == (145, 145)
        assert Handed.of(staged).links == {}
        assert (staged / ".dockerignore").read_bytes() == b"*.md\n"

    def test_an_ignore_file_that_is_all_staging_writes_is_counted_too(
        self, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """`Dockerfile.slim` is the root `Dockerfile` itself, under another name, so no Dockerfile is copied.
        Built by that name, its own ignore file is written at the root, and that is thirty bytes more."""
        files = {"Dockerfile": b"FROM tiny\n", "Dockerfile.slim.dockerignore": _sized(b"*.md", 30), "app.py": b"#" * 60}
        for name, fits in (("over", False), ("within", True)):
            context = self._context(tmp_path / name, files)
            os.link(context / "Dockerfile", context / "Dockerfile.slim")
            before = _names(context)
            limit(130 if fits else 129)
            if fits:
                # Counted here for each name, the Dockerfile's ten bytes twice: the driver counts one file once.
                assert _followed_bytes(stage_context(context, "Dockerfile.slim")) == 130 + len(files["Dockerfile"])
            else:
                with pytest.raises(ImageFailed, match="more than 129 bytes with its links followed"):
                    stage_context(context, "Dockerfile.slim")
                assert _names(context) == before

    def test_the_ignore_file_that_goes_with_a_replaced_dockerfile_is_not_counted(
        self, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """The named Dockerfile has no ignore file of its own, so the replaced root Dockerfile's goes. A
        hundred bytes before, and a hundred staged: thirty more of Dockerfile, and thirty fewer of rules."""
        files = {
            self.NAMED: _sized(b"FROM scratch", 40),
            "payload.bin": b"\0" * 20,
            "Dockerfile": b"FROM tiny\n",
            "Dockerfile.dockerignore": _sized(b"*.md", 30),
        }
        context = self._context(tmp_path / "context", files)
        assert _regular_bytes(context) == self.LIMIT

        limit(self.LIMIT)
        staged = stage_context(context, self.NAMED)

        assert not (staged / "Dockerfile.dockerignore").exists()
        assert (_regular_bytes(staged), _followed_bytes(staged)) == (self.LIMIT, self.LIMIT)

    def test_a_link_that_what_staging_takes_away_leaves_leading_out_of_the_context_is_refused(
        self, tmp_path: Path
    ) -> None:
        """`Dockerfile.slim` is the root `Dockerfile` under another name, with no ignore file of its own, so
        the root one's goes: here a link to a directory two levels down. `fixture` is written as a path
        through it and two steps back, which led to the context's own `app.py`. With the link gone, the two
        steps are taken from a name with nothing behind it, and the path leads to a file beside the context.

        Nothing was written into this context, and no link of it was rewritten. It is still looked at again
        as it is handed on, because it holds links, and what they lead to is only known then."""
        files = {"Dockerfile": b"FROM scratch\n", "app.py": b"the context's own\n", "a/b/keep": b""}
        context = self._context(tmp_path / "fetched" / "app", files)
        (tmp_path / "fetched" / "app.py").write_bytes(b"not this build's\n")
        (context / "Dockerfile.slim").symlink_to("Dockerfile")
        (context / "Dockerfile.dockerignore").symlink_to("a/b")
        (context / "a" / "fixture").symlink_to("../Dockerfile.dockerignore/../../app.py")
        assert (context / "a" / "fixture").read_bytes() == b"the context's own\n"
        # Built by the root Dockerfile's own name, nothing is taken away, and the link is handed on as it is.
        assert os.readlink(stage_context(context, "Dockerfile") / "a" / "fixture") == (
            "../Dockerfile.dockerignore/../../app.py"
        )

        with pytest.raises(ImageFailed, match="a link that leads out of it, or back .*: 'a/fixture'"):
            stage_context(context, "Dockerfile.slim")

    def test_a_replaced_dockerfile_that_the_context_holds_under_another_name_is_still_counted(
        self, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """The root `Dockerfile` and `docker/Dockerfile.default` are one file, as an archive may hold one. The
        named Dockerfile takes the root one's place, and the file stays, under its other name. Before it is
        written, a file that is replaced is taken to go, so this context is let through: it is the measure of
        what was staged that refuses it, though it holds no link."""
        files = {"docker/Dockerfile.default": _sized(b"FROM default", 60), self.NAMED: _sized(b"FROM scratch", 40)}
        for name, fits in (("over", False), ("within", True)):
            context = self._context(tmp_path / name, files)
            os.link(context / "docker/Dockerfile.default", context / "Dockerfile")
            # A hundred bytes of files: the one of sixty has two names.
            assert _regular_bytes(context) == self.LIMIT + 60
            limit(140 if fits else 139)
            if fits:
                assert _followed_bytes(stage_context(context, self.NAMED)) == 140
            else:
                with pytest.raises(ImageFailed, match="more than 139 bytes with its links followed"):
                    stage_context(context, self.NAMED)

    def test_the_names_staging_adds_are_counted_as_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same of the limit beside it. Five entries, and the root `Dockerfile` and its ignore file are
        two more."""
        context = self._context(tmp_path / "context", self.WITH_IGNORE)
        before = _names(context)
        assert len(before) == 5

        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", 6)
        with pytest.raises(ImageFailed, match="the build context comes to more than 6 entries"):
            stage_context(context, self.NAMED)
        assert _names(context) == before

        monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", 7)
        assert _names(stage_context(context, self.NAMED)) == sorted([*before, "Dockerfile", "Dockerfile.dockerignore"])

    @pytest.mark.parametrize("built", ["alone", "from-a-set"])
    @pytest.mark.parametrize("past", ["its-bytes", "its-entries"])
    def test_a_context_with_no_links_that_was_past_a_limit_as_it_was_fetched_is_staged_all_the_same(
        self, past: str, built: str, tmp_path: Path, limit: Callable[[int], None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fileset that is not an archive has no limit as it is fetched, and holds no link. Here it is a
        hundred bytes in five names, under a limit of ninety-nine bytes, or of four entries. It was never
        within the limit, so it is not staging that carries it past, and whether it builds does not rest on
        where its Dockerfile is: built by a `Dockerfile` at its root it is not measured, and built by one
        elsewhere it is staged as any context is, with that Dockerfile and its ignore file at the root.

        A context that is within the limits as it was fetched is still held to them: at a limit of exactly
        what it holds, the same context is refused for what staging adds."""
        files = self.WITH_IGNORE
        context = self._context(tmp_path / "fetched", files)
        into = tmp_path / "copy" / "context" if built == "from-a-set" else None
        before = _names(context)
        assert (_regular_bytes(context), len(before)) == (self.LIMIT, 5)

        def scale(by: int) -> None:
            if past == "its-bytes":
                limit(self.LIMIT + by)
            else:
                monkeypatch.setattr(driver, "MAX_UNPACKED_ENTRIES", 5 + by)

        scale(0)
        with pytest.raises(ImageFailed, match="the build context comes to more than"):
            stage_context(context, self.NAMED, into=into)
        assert _names(context) == before

        scale(-1)
        handed = stage_context(context, self.NAMED, into=None if into is None else tmp_path / "again" / "context")
        assert (handed / "Dockerfile").read_bytes() == files[self.NAMED]
        assert (handed / "Dockerfile.dockerignore").read_bytes() == files[self.NAMED_IGNORE]
        assert (_regular_bytes(handed), len(_names(handed))) == (160, 7)

    def test_a_link_makes_a_context_one_that_is_measured_whatever_it_came_to_as_fetched(
        self, tmp_path: Path, limit: Callable[[int], None]
    ) -> None:
        """What is left unmeasured is a context with no link, which is as large as it was fetched. The same
        hundred bytes with one link among them are measured with the link followed, as they always were."""
        context = self._context(tmp_path / "fetched", self.WITH_IGNORE, {"alias": "payload.bin"})
        limit(self.LIMIT - 1)
        with pytest.raises(ImageFailed, match=f"more than {self.LIMIT - 1} bytes with its links followed"):
            stage_context(context, self.NAMED)

    def test_an_archive_that_lists_no_directories_is_not_refused_for_the_names_it_comes_to_on_disk(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Through the driver, with the step's own fetch. Unpacking counts the entries an archive lists, and
        an archive need not list its directories: four files, under five directories it doesn't name, are
        four entries, within a limit of four, and nine names on disk. The measure counts names, so staged
        with a Dockerfile that is not at its root this context was refused for entries it was never held
        to. It holds no link, and was past the limit before anything was written into it: it is built."""
        entries: Entries = {self.NAMED: b"FROM scratch\n", "a/one": b"1", "b/two": b"2", "c/three": b"3"}
        for module in (fetch_step, driver):
            monkeypatch.setattr(module, "MAX_UNPACKED_ENTRIES", 4)

        # One entry more is refused where it is unpacked: the limit is the one this archive was held to.
        with caplog.at_level(logging.ERROR):
            code = _run(
                Inspecting(),
                FakeBuilder(),
                tmp_path,
                tmp_path,
                fetch=_fetching(entries | {"d/four": b"4"})[0],
                config=_archive_set(None, dockerfile=self.NAMED),
            )
        assert code == 1 and "it holds more than 4 entries" in caplog.text

        service, builder = Inspecting(), FakeBuilder()
        code = _run(
            service,
            builder,
            tmp_path,
            tmp_path,
            fetch=_fetching(entries)[0],
            config=_archive_set(None, dockerfile=self.NAMED),
        )
        assert (code, [image for image, _, _ in builder.completed]) == (0, ["a"])
        assert len(service.handed["a"].names) == 10
        assert service.handed["a"].reads["Dockerfile"] == b"FROM scratch\n"

    def test_such_an_archive_with_a_link_in_it_is_measured_by_its_names_on_disk(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A context with links is measured with them followed, as it always was, and the measure counts
        names: five entries to unpack, within a limit of five, are eight names on disk and one more for
        the file the link leads to. So this one fails for entries it was unpacked within, wherever its
        Dockerfile is. The README says so."""
        entries: Entries = {"Dockerfile": b"FROM scratch\n", "a/one": b"1", "b/two": b"2", "c/three": b"3"}
        for module in (fetch_step, driver):
            monkeypatch.setattr(module, "MAX_UNPACKED_ENTRIES", 5)

        service = Inspecting()
        with caplog.at_level(logging.ERROR):
            code = _run(
                service,
                FakeBuilder(),
                tmp_path,
                tmp_path,
                fetch=_fetching(entries | {"alias": Link("a/one")})[0],
                config=_archive_set(None),
            )
        assert (code, service.started) == (1, [])
        assert f"{FAILURE_MARKER} a: the build context comes to more than 5 entries with its links followed" in (
            caplog.text
        )

    def test_the_step_starts_no_build_for_a_context_that_is_past_the_limit_once_it_is_staged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, limit: Callable[[int], None]
    ) -> None:
        """Through the driver, with the step's own fetch. The archive unpacks to exactly the limit, and one
        byte more is refused there. Staged, it would be 160 bytes of files, 360 with its links followed: the
        image fails with the limit's own reason, and the service is not started."""
        entries: Entries = {**self.WITH_IGNORE, **{name: Link(leads_to) for name, leads_to in self.LINKS.items()}}
        limit(self.LIMIT)

        over = Inspecting()
        with caplog.at_level(logging.ERROR):
            code = _run(
                over,
                FakeBuilder(),
                tmp_path,
                tmp_path,
                fetch=_fetching(entries | {"payload.bin": b"\0" * 41})[0],
                config=_archive_set(None, dockerfile=self.NAMED),
            )
        assert (code, over.started) == (1, [])
        assert f"it unpacks to more than {self.LIMIT} bytes" in caplog.text

        caplog.clear()
        service, builder = Inspecting(), FakeBuilder()
        with caplog.at_level(logging.ERROR):
            code = _run(
                service,
                builder,
                tmp_path,
                tmp_path,
                fetch=_fetching(entries)[0],
                config=_archive_set(None, dockerfile=self.NAMED),
            )
        assert (code, service.started, builder.completed) == (1, [], [])
        assert (
            f"{FAILURE_MARKER} a: the build context comes to more than {self.LIMIT} bytes with its links followed"
        ) in caplog.text


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
    secrets = driver._REDACTION._redactors[:]
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)
    logging.setLogRecordFactory(make_record)
    # The process's redaction outlives a step in a test run, as it never does in a step.
    driver._REDACTION._redactors[:] = secrets


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
        # Where it failed is said first: the lines settling reads stay last, in the end of the log Jobs keeps.
        said = [record.getMessage() for record in caplog.records if record.name == driver.logger.name]
        assert said[0].startswith("the step could not start, at:\n") and "in make_service" in said[0]
        assert [line.split(" ", 1)[0] for line in said[1:]] == [FAILURE_MARKER]
        # The step's own directory is made last, so a step that fails before it leaves nothing on task storage.
        assert list((tmp_path / "task-storage").iterdir()) == []

    def test_the_credential_is_taken_out_of_what_reaches_the_row_too(self, step: dict[str, list]) -> None:
        """What the step hands the driver to cut reasons with is what the driver takes to the service's handle."""
        reported = RemoteImage(
            reference=f"{DESTINATION}@{DIGEST}", image_id="img-1", attributes={"console": "https://x/?key=token-value"}
        )
        assert step_main(lambda config, credential: FakeService([RemoteBuildStatus("done", image=reported)])) == 0
        (builder,) = step["builder"]
        ((_, _, external),) = builder.completed
        assert external.attributes == {"console": "https://x/?key=[credential]"}

    def test_a_secret_given_after_a_library_took_over_the_log_is_still_known_to_the_driver(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tracing library installs a record factory of its own. A token the client then trades the credential
        for must still be cut out of a reason before the reason is cut to length, and out of the row."""
        token = "eyJ" + "t0k" * 300

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            under = logging.getLogRecordFactory()
            logging.setLogRecordFactory(lambda *args, **kwargs: under(*args, **kwargs))
            driver.redact_log(token)
            return FakeService([DONE], start=[PermanentError(f"401: {token} was refused")])

        with caplog.at_level(logging.INFO):
            assert step_main(make_service) == 1
        assert "t0k" not in caplog.text
        assert f"{FAILURE_MARKER} demo-1: starting the build: 401: [credential] was refused" in caplog.text
        # Back on top, so that what the library's factory makes is redacted as well.
        assert isinstance(logging.getLogRecordFactory(), driver._RedactingFactory)

    def test_a_client_that_replaces_the_record_factory_does_not_end_the_redaction(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A library that sets the factory back to the plain one, and gives nothing to `redact_log`. The step
        puts its own back once the client is made, so what the driver logs of the service is still redacted."""
        echoing = RemoteBuildStatus("building", "authorised as token-value")

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            logging.setLogRecordFactory(logging.LogRecord)
            return FakeService([echoing, DONE])

        with caplog.at_level(logging.INFO):
            assert step_main(make_service) == 0
        assert "token-value" not in caplog.text
        assert "build build-1: building (authorised as [credential])" in caplog.text

    def test_what_a_library_logs_under_the_step_is_never_read_as_a_failure_line(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Such as the fetch, which logs the path it is given: an archive may be named anything."""

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            logging.getLogger("nemo_builder_plugin.run.fetch").info(
                "fetching archive %s", f"{FAILURE_MARKER} demo-1: built and published.tar"
            )
            return FakeService([DONE])

        with caplog.at_level(logging.INFO):
            assert step_main(make_service) == 0
        assert list(_marked_lines(caplog.text)) == []
        assert "fetching archive NHX-BUILD-FAILED demo-1: built and published.tar" in caplog.text

    def test_an_exception_a_library_logs_is_never_read_as_a_failure_line_either(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A client that logs a call that failed, with what the service said: a line of a submitter's Dockerfile."""

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            try:
                raise RuntimeError(f"RUN echo {FAILURE_MARKER} demo-1: built and published")
            except RuntimeError:
                logging.getLogger("some.build.client").exception("the first call failed", stack_info=True)
            return FakeService([DONE])

        with caplog.at_level(logging.INFO):
            assert step_main(make_service) == 0
        assert list(_marked_lines(caplog.text)) == []
        assert "RuntimeError: RUN echo NHX-BUILD-FAILED demo-1: built and published" in caplog.text
        (logged,) = [record for record in caplog.records if record.name == "some.build.client"]
        # The exception itself still holds the marker, and some handlers print it themselves.
        assert logged.exc_info is None

    def test_an_exception_a_client_logs_during_a_step_reaches_no_handler_as_itself(
        self, step: dict[str, list], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The credential the step was given is a secret from its first record on. An error that says nothing
        of it is still handed on as text: the frames it passed through are the client's, which holds it."""

        class Logging(FakeService):
            def start(self, build: RemoteBuild) -> str:
                try:
                    raise ConnectionError("connection refused")
                except ConnectionError:
                    logging.getLogger("some.build.client").exception("the first call failed")
                return super().start(build)

        with caplog.at_level(logging.INFO):
            assert step_main(lambda config, credential: Logging([DONE])) == 0
        (logged,) = [record for record in caplog.records if record.name == "some.build.client"]
        assert logged.exc_info is None
        assert "ConnectionError: connection refused" in (logged.exc_text or "")
        # Nor does any line of the driver's own carry one.
        assert [record.getMessage() for record in caplog.records if record.exc_info] == []

    def test_the_marker_is_taken_out_of_a_stack_a_library_logs_too(self) -> None:
        """A stack shows lines of source. One of them may spell the marker out."""
        record = logging.LogRecord("some.build.client", logging.ERROR, __file__, 1, "the call failed", None, None)
        record.stack_info = f'Stack (most recent call last):\n    print("{FAILURE_MARKER} demo-1: built")'
        driver._unmark_record(record)
        assert record.stack_info == 'Stack (most recent call last):\n    print("NHX-BUILD-FAILED demo-1: built")'

    def test_a_reason_that_quotes_the_marker_holds_it_only_at_the_start_of_its_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Jobs keeps the end of the log, cut by length. A line cut before a marker that its reason quotes
        would be read from that marker on, as another image's reason."""
        quoting = f"dockerfile 'x {FAILURE_MARKER} b: built and published/Dockerfile' is not a file"
        with caplog.at_level(logging.ERROR):
            driver._report("a", quoting, f" [build {FAILURE_MARKER}-1]")
        (line,) = _marked_lines(caplog.text)
        assert line == (
            f"{FAILURE_MARKER} a: dockerfile 'x NHX-BUILD-FAILED b: built and published/Dockerfile' is not a file"
            " [build NHX-BUILD-FAILED-1]"
        )
        # Cut inside the line, as the end of a log may be: nothing of it is read.
        assert list(_marked_lines(line[len(FAILURE_MARKER) :])) == []

    @pytest.mark.parametrize("wrapped", [True, False], ids=["a-stream-that-can-be-told", "one-that-cant"])
    def test_standard_output_is_line_buffered_where_it_can_be(
        self, step: dict[str, list], monkeypatch: pytest.MonkeyPatch, wrapped: bool
    ) -> None:
        """What a step program prints is then in the log before the driver's last lines, not after them."""
        stream = io.TextIOWrapper(io.BytesIO()) if wrapped else io.StringIO()
        monkeypatch.setattr(driver, "sys", SimpleNamespace(stdout=stream))
        assert step_main(lambda config, credential: FakeService([DONE])) == 0
        assert getattr(stream, "line_buffering", False) is wrapped

    @pytest.mark.parametrize(
        ("raised", "said"),
        [
            (ValueError("unexpected text in the key file: t0p-s3cret-value"), "ValueError"),
            (OSError("unexpected text in the key file: t0p-s3cret-value"), "OSError"),
            (OSError(5, "read failed near: t0p-s3cret-value"), os.strerror(5)),
            (FileNotFoundError(2, "No such file or directory", "/var/run/build/token"), "/var/run/build/token"),
        ],
        ids=[
            "a-reader-that-quotes-what-it-read",
            "one-that-quotes-it-in-an-error-of-the-systems-kind",
            "one-that-quotes-it-beside-an-error-number",
            "a-file-that-is-not-there",
        ],
    )
    def test_a_credential_that_cant_be_read_is_not_reported_in_the_readers_words(
        self, step: dict[str, list], tmp_path: Path, caplog: pytest.LogCaptureFixture, raised: Exception, said: str
    ) -> None:
        """Nothing can take the credential out of a reason yet: the driver was never given it."""
        (tmp_path / "job_step_config.json").write_text(json.dumps(_config().model_dump(mode="json")))

        def read_credential() -> str:
            raise raised

        with caplog.at_level(logging.ERROR):
            assert step_main(lambda config, credential: FakeService([DONE]), read_credential=read_credential) == 1
        assert "t0p-s3cret-value" not in caplog.text
        (line,) = _marked_lines(caplog.text)
        assert "the build service's credential could not be read: " in line and said in line

    def test_a_step_with_no_credential_of_its_own_leaves_it_to_the_client(
        self, step: dict[str, list], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not in a platform secret, and no function to read it: the client finds it itself."""
        (tmp_path / "job_step_config.json").write_text(json.dumps(_config().model_dump(mode="json")))
        monkeypatch.delenv(REMOTE_CREDENTIAL_ENV)
        made: list[str | None] = []

        def make_service(config: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            made.append(credential)
            return FakeService([DONE])

        assert step_main(make_service) == 0
        assert made == [None]

    def test_a_mounted_credential_that_isnt_there_fails_every_image_and_says_so(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        signals: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A profile that doesn't mount the Secret it should. Handed on as no credential, the client would
        report whatever its service says to a request with none."""
        path = tmp_path / "job_step_config.json"
        path.write_text(json.dumps(_config("a", "b").model_dump(mode="json")))
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(path))
        made: list[object] = []
        with caplog.at_level(logging.ERROR):
            assert step_main(lambda config, credential: made.append(credential), read_credential=lambda: " \n") == 1  # ty: ignore[invalid-argument-type]
        assert made == []
        for image in ("a", "b"):
            assert (
                f"{FAILURE_MARKER} {image}: the step could not start: RuntimeError: the build service's credential "
                "is empty: the step's Jobs profile did not mount it"
            ) in caplog.text

    def test_a_set_that_cant_start_says_why_on_every_images_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        signals: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every image gets a line, and none is cut to less than says why. With this many, the first lines fall
        outside what Jobs keeps of the log, and those rows say only what Jobs says of the job."""
        names = [f"image-{n:02}" for n in range(30)]
        config = _config(*names).model_copy(update={"credential_in_environment": True})
        path = tmp_path / "job_step_config.json"
        path.write_text(json.dumps(config.model_dump(mode="json")))
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(path))
        monkeypatch.delenv(REMOTE_CREDENTIAL_ENV, raising=False)
        with caplog.at_level(logging.ERROR):
            assert step_main(lambda config, credential: FakeService([DONE])) == 1
        lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith(FAILURE_MARKER)]
        why = f"the step could not start: RuntimeError: {REMOTE_CREDENTIAL_ENV} not set"
        assert [line.split(": ", 1)[0] for line in lines] == [f"{FAILURE_MARKER} {name}" for name in names]
        assert all(why in line for line in lines)

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

    def test_a_credential_in_a_log_calls_argument_is_taken_out_of_the_message(self) -> None:
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

    @pytest.mark.parametrize(
        ("secret", "text"),
        [
            ("QUJD+REVG/R0hJ==", "GET /builds?key=QUJD%2BREVG/R0hJ%3D%3D failed"),
            ("QUJD+REVG/R0hJ==", "GET /builds?key=QUJD%2bREVG%2fR0hJ== failed"),
            ("two words", "GET /builds?key=two+words failed"),
            ("QUJD~REVG", "GET /builds?key=QUJD%7EREVG failed"),
            ("pa'ss-w0rd-123", repr(("401", 'reply: {"error": "bad password pa\'ss-w0rd-123"}'))),
            ("pa'ss-w0rd-123", repr(repr('reply: {"error": "bad password pa\'ss-w0rd-123"}'))),
            ("pa\\'ss-w0rd-123", repr(("401", 'reply: {"error": "bad password pa\\\'ss-w0rd-123"}'))),
            ("C:\\keys\\build", repr(repr("refused C:\\keys\\build"))),
            ("pässwörd-123", json.dumps({"error": "bad password pässwörd-123"})),
            ("pässwörd-123", ascii("bad password pässwörd-123")),
            ('pä"ss-wörd', json.dumps('refused pä"ss-wörd', ensure_ascii=False)),
            ('say "hi" twice', json.dumps({"error": json.dumps('refused say "hi" twice')})),
        ],
        ids=[
            "url-slash-left",
            "url-lowercase-hex",
            "url-plus",
            "url-tilde",
            "apostrophe-in-a-repr",
            "apostrophe-in-a-repr-of-a-repr",
            "backslash-before-an-apostrophe",
            "escaped-twice",
            "json-ascii-only",
            "ascii-of-a-repr",
            "json-as-it-is",
            "json-twice",
        ],
    )
    def test_a_credential_is_taken_out_however_what_holds_it_wrote_it(self, secret: str, text: str) -> None:
        redacted = driver._redactor(secret)(text)
        assert "[credential]" in redacted
        for part in ("REVG", "words", "w0rd", "keys", "hi", "rd-123", "ss-w"):
            assert part not in redacted or part not in secret

    def test_a_credential_of_several_lines_in_a_url_is_taken_out_whole_not_one_line_of_it(self) -> None:
        """Its first line is long enough to be taken out on its own. Tried before the whole, it would be, and
        the rest of the credential left behind it."""
        redact = driver._redactor("dXNlcjpw\nYXNz")
        assert redact("GET /builds?key=dXNlcjpw%0aYXNz failed") == "GET /builds?key=[credential] failed"

    def test_a_record_the_step_made_can_be_sent_to_another_process(self, signals: None) -> None:
        """A handler that hands the record itself to a queue between processes pickles it."""
        driver.redact_log("token-value")
        record = logging.getLogger("some.build.client.queue").makeRecord(
            "x", logging.ERROR, __file__, 1, "401 for %s", ("token-value",), None, extra={"attempt": 2}
        )
        sent = pickle.loads(pickle.dumps(record))
        assert (sent.getMessage(), sent.attempt) == ("401 for [credential]", 2)

    def test_a_secret_a_client_adds_is_taken_out_with_the_credential(self, signals: None) -> None:
        """A client that trades the credential for a token gives the token to `redact_log`. Both stay out of
        the message, the fields and a reason, whichever was given first."""
        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        handler = Own()
        handler.setFormatter(logging.Formatter("%(message)s %(headers)s"))
        client_log = logging.getLogger("some.build.client.tokens")
        client_log.propagate = False
        client_log.addHandler(handler)
        redact = driver.log_redaction()
        driver.redact_log("the-credential")
        driver.redact_log("the-access-token")
        try:
            client_log.error(
                "the-credential was traded for the-access-token",
                extra={"headers": ["the-credential", "the-access-token"]},
            )
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == ["[credential] was traded for [credential] ['[credential]', '[credential]']"]
        # The same function the driver cuts a reason with: it knows the token too, though it was given later.
        assert redact("401: the-access-token expired") == "401: [credential] expired"

    def test_an_exception_is_not_left_on_the_record_whether_or_not_its_text_held_the_credential(self) -> None:
        """Some handlers print the exception themselves, from the record, and more of it than its text: the
        frames it passed through, with what each one held."""
        try:
            raise ValueError("refused: Bearer token-value")
        except ValueError:
            leaking = logging.LogRecord("x", logging.ERROR, __file__, 1, "the call failed", None, sys.exc_info())
        try:
            raise ValueError("refused")
        except ValueError:
            harmless = logging.LogRecord("x", logging.ERROR, __file__, 1, "the call failed", None, sys.exc_info())
        redact = driver._redactor("token-value")
        driver._redact_record(leaking, redact)
        driver._redact_record(harmless, redact)
        assert leaking.exc_info is None and "Bearer [credential]" in (leaking.exc_text or "")
        assert harmless.exc_info is None and "ValueError: refused" in (harmless.exc_text or "")

    @pytest.mark.parametrize("made", ["by-a-log-call", "from-a-mapping"])
    def test_a_handler_that_shows_what_each_frame_held_is_handed_no_exception_to_show(
        self, made: str, signals: None
    ) -> None:
        """A client's variable holds the credential where its error says only that the connection was refused,
        and no line of source that a traceback shows spells it out. A handler that prints a traceback with the
        variables of each frame would print it."""
        credential = "token-value"
        printed: list[str] = []

        class ShowsVariables(logging.Handler):
            """Prints a record's exception as such a handler does: its frames, and what each one held."""

            def emit(self, record: logging.LogRecord) -> None:
                lines = [self.format(record)]
                at = record.exc_info[2] if record.exc_info else None
                while at is not None:
                    lines.extend(f"{name} = {value!r}" for name, value in at.tb_frame.f_locals.items())
                    at = at.tb_next
                printed.append("\n".join(lines))

        def send(authorization: str) -> None:
            raise ConnectionError("connection refused")

        handler = ShowsVariables()
        client_log = logging.getLogger("some.build.client.frames")
        client_log.propagate = False
        client_log.addHandler(handler)
        driver.redact_log(credential)
        try:
            try:
                send(f"Bearer {credential}")
            except ConnectionError:
                if made == "by-a-log-call":
                    client_log.exception("the call failed")
                else:
                    fields = {"msg": "the call failed", "levelno": logging.ERROR, "exc_info": sys.exc_info()}
                    client_log.handle(logging.makeLogRecord(fields))
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        (shown,) = printed
        assert credential not in shown
        # All that a formatter prints of the exception is there, as text.
        assert "ConnectionError: connection refused" in shown and "in send" in shown

    def test_a_record_that_cant_be_checked_is_withheld_not_printed(self, signals: None) -> None:
        """Logging never raises into the code that logs, and what could not be checked is not let through."""

        class Unprintable:
            def __repr__(self) -> str:
                raise RuntimeError("no repr")

        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        handler = Own()
        handler.setFormatter(logging.Formatter("%(message)s | %(request)s"))
        client_log = logging.getLogger("some.build.client.broken")
        client_log.propagate = False
        client_log.addHandler(handler)
        driver.redact_log("token-value")
        try:
            client_log.warning("try %d of %d", Unprintable(), extra={"request": "ok"})
            client_log.warning("sent", extra={"request": Unprintable()})
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == [f"{driver.WITHHELD} | ok", f"sent | {driver.WITHHELD}"]

    def test_taking_a_credential_out_twice_changes_nothing_more(self) -> None:
        """A reason is redacted before and after it is made one line, and a short credential may be part of the
        mark that takes its place."""
        redact = driver._redactor("cred")
        once = redact("refused: cred is not known")
        assert once == "refused: [credential] is not known"
        assert redact(once) == once

    @pytest.mark.parametrize(
        ("secret", "text", "redacted"),
        [
            ("k3y-[credential]-9f2a", "401: k3y-[credential]-9f2a was refused", "401: [credential] was refused"),
            ("[credential]-9f2a", "401: [credential]-9f2a was refused", "401: [credential] was refused"),
            ("k3y-[credential]", "401: k3y-[credential] was refused", "401: [credential] was refused"),
            (
                "k3y [credential] 9f2a",
                "GET /builds?key=k3y%20[credential]+9f2a failed",
                "GET /builds?key=[credential] failed",
            ),
            (
                "first-line-k3y\n9f2a-[credential]-end",
                "line 2 of the key is malformed: 9f2a-[credential]-end",
                "line 2 of the key is malformed: [credential]",
            ),
            ("k3y-[cred\nential]-9f2a", "refused k3y-[credential]-9f2a", "refused [credential]"),
            (
                "[credential]%25",
                "GET /builds?key=%5Bcredential%5D%2525 failed",
                "GET /builds?key=[credential] failed",
            ),
            ("k3y-[credential]-9f2a", "k3y-k3y-[credential]-9f2a-9f2a", "[credential]"),
            ("A[credential]", "AA[credential]", "[credential]"),
        ],
        ids=[
            "in-its-middle",
            "at-its-start",
            "at-its-end",
            "in-a-url-that-leaves-brackets-as-they-are",
            "in-one-line-of-several",
            "only-as-a-log-line-writes-it",
            "ending-in-a-percent-sign-in-a-url-that-encodes-it",
            "between-copies-of-its-own-two-ends",
            "after-a-copy-of-its-own-start",
        ],
    )
    def test_a_credential_that_holds_the_marks_own_text_is_taken_out_whole(
        self, secret: str, text: str, redacted: str
    ) -> None:
        """Read between the marks alone, such a credential is never found: the mark's text in it ends what is
        read, and the two halves are passed on with that text between them, which is the credential."""
        redact = driver._redactor(secret)
        assert redact(text) == redacted
        # The mark that took its place is kept, as any other is.
        assert redact(redacted) == redacted

    @pytest.mark.parametrize(
        ("secret", "text"),
        [
            ("cred", "401: [credential] was refused"),
            ("]", "401: [credential] was refused"),
            ("[credential]", "401: [credential] was refused"),
            ("k3y-[cred", "401: k3y-[credential] was refused"),
            ("ential]-9f2a", "401: [credential]-9f2a was refused"),
        ],
        ids=[
            "part-of-the-mark",
            "the-marks-last-character",
            "the-mark-itself",
            "ending-as-the-mark-begins",
            "beginning-as-the-mark-ends",
        ],
    )
    def test_a_mark_already_in_a_text_is_kept_whole_whatever_the_credential(self, secret: str, text: str) -> None:
        """A text is redacted more than once, and by more than one secret. A mark in it stands for what was
        taken out: nothing is taken out of the middle of it, nor from across its edge, which would leave a
        piece of it beside a second mark."""
        assert driver._redactor(secret)(text) == text

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

    def test_a_field_that_prints_a_mask_is_redacted_by_what_it_holds(self, signals: None) -> None:
        """A request's headers print a mask where they hold the credential, and give it whole to a format
        that asks for one of them by name."""
        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        handler = Own()
        handler.setFormatter(logging.Formatter("{message} {headers[authorization]} {headers[accept]}", style="{"))
        client_log = logging.getLogger("some.build.client.headers")
        client_log.propagate = False
        client_log.addHandler(handler)
        driver.redact_log("token-value")
        headers = httpx.Headers({"Authorization": "Bearer token-value", "Accept": "application/json"})
        try:
            assert "token-value" not in repr(headers)
            client_log.error("request failed", extra={"headers": headers})
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == ["request failed Bearer [credential] application/json"]
        # The caller's own headers are not changed under it.
        assert headers["authorization"] == "Bearer token-value"

    def test_a_container_of_any_kind_is_looked_into_not_read_as_it_prints(self) -> None:
        """Whatever is a mapping, a sequence or a set, not only the plain ones. One that holds the credential
        becomes a plain container without it; one that doesn't is passed on as it is."""

        class Masked(Mapping[str, str]):
            def __init__(self, held: dict[str, str]) -> None:
                self._held = held

            def __getitem__(self, key: str) -> str:
                return self._held[key]

            def __iter__(self) -> Iterator[str]:
                return iter(self._held)

            def __len__(self) -> int:
                return len(self._held)

            def __repr__(self) -> str:
                return f"Masked({sorted(self._held)})"

        class Call(NamedTuple):
            url: str
            tries: int

        redact = driver._redactor("token-value")
        leaking = Masked({"authorization": "Bearer token-value", "accept": "application/json"})
        without = {"authorization": "Bearer [credential]", "accept": "application/json"}

        rebuilt = driver._redact_value(leaking, redact)
        assert (type(rebuilt), rebuilt) == (dict, without)
        # Inside a plain container too, which is then rebuilt around it.
        assert driver._redact_value({"request": [leaking]}, redact) == {"request": [without]}
        assert driver._redact_value(collections.deque(["token-value", 2]), redact) == ["[credential]", 2]
        assert driver._redact_value(Call("https://build.example.com/builds?key=token-value", 2), redact) == (
            "https://build.example.com/builds?key=[credential]",
            2,
        )
        assert driver._redact_value({"token-value": 1}.keys(), redact) == {"[credential]"}
        # Text and bytes are sequences too, and are still read as text.
        assert driver._redact_value(b"key=token-value", redact) == "b'key=[credential]'"

        harmless = Masked({"accept": "application/json"})
        queue = collections.deque(["queued", "building"])
        assert driver._redact_value(harmless, redact) is harmless
        assert driver._redact_value(queue, redact) is queue
        assert driver._redact_value(range(3), redact) == range(3)

    def test_a_container_too_deep_to_look_into_is_passed_on_only_if_it_prints_all_it_holds(self) -> None:
        """Past the depth the driver looks to, a value is read as it prints. That shows everything in plain
        containers, and may not in any other: such a one is passed on as its text, not as itself."""

        class Masked(collections.UserDict):
            def __repr__(self) -> str:
                return "Masked(…)"

        def nested(value: object, levels: int) -> object:
            for _ in range(levels):
                value = [value]
            return value

        redact = driver._redactor("token-value")
        leaking = Masked({"authorization": "Bearer token-value"})
        for levels in (driver._MAX_FIELD_DEPTH, driver._MAX_FIELD_DEPTH + 3):
            passed_on = driver._redact_value(nested(leaking, levels), redact)
            assert "Masked(…)" in repr(passed_on)
            # Nothing passed on still holds the mapping, to give the credential to whoever asks it.
            while isinstance(passed_on, list):
                (passed_on,) = passed_on
            assert isinstance(passed_on, str)
        plain = nested({"tries": 3, "stages": ("queued", "building")}, driver._MAX_FIELD_DEPTH + 3)
        assert driver._redact_value(plain, redact) is plain

    @pytest.mark.parametrize("installed", ["before-the-steps", "after-it-and-the-steps-put-back-on-top"])
    def test_what_a_factory_under_the_steps_put_on_a_record_is_redacted_too(
        self, installed: str, signals: None
    ) -> None:
        """A tracing library's record factory. Its fields are on the record when the step's factory is handed
        it, set as attributes, where a log call's own fields are set afterwards and by key. The step puts its
        factory back on top of one a client installed, so such a factory is always under the step's."""
        printed: list[str] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                printed.append(self.format(record))

        traced = {
            "request_url": "https://build.example.com/builds?key=token-value",
            "baggage": httpx.Headers({"Authorization": "Bearer token-value"}),
            "span": 7,
        }

        def tracing(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = under(*args, **kwargs)
            for name, value in traced.items():
                setattr(record, name, value)
            return record

        if installed == "before-the-steps":
            under = logging.getLogRecordFactory()
            logging.setLogRecordFactory(tracing)
            driver.redact_log("token-value")
        else:
            driver.redact_log("token-value")
            under = logging.getLogRecordFactory()
            logging.setLogRecordFactory(tracing)
            driver.log_redaction()
        handler = Own()
        handler.setFormatter(logging.Formatter("%(message)s %(request_url)s %(baggage)s %(span)s"))
        client_log = logging.getLogger("some.build.client.traced")
        client_log.propagate = False
        client_log.addHandler(handler)
        try:
            client_log.error("request failed")
            record = client_log.makeRecord("x", logging.INFO, __file__, 1, "build %s: %s", ("build-1", "queued"), None)
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        assert printed == [
            "request failed https://build.example.com/builds?key=[credential] "
            "{'authorization': 'Bearer [credential]'} 7"
        ]
        # What holds no credential is as the factory under the step's made it, and as the call logged it.
        assert (vars(record)["span"], record.msg, record.args) == (7, "build %s: %s", ("build-1", "queued"))

    def test_what_a_message_doesnt_print_of_what_it_is_made_from_does_not_stay_on_the_record(
        self, signals: None
    ) -> None:
        """A message prints its arguments as they print: a request's headers as a mask, a mapping as the one
        value its format names. A handler that reads the arguments themselves, to report them apart from the
        message, would be handed the credential that the message left out. So it would from a message that
        was logged as an object, and not as text."""
        seen: list[logging.LogRecord] = []

        class Own(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                seen.append(record)

        handler = Own()
        client_log = logging.getLogger("some.build.client.arguments")
        client_log.propagate = False
        client_log.addHandler(handler)
        driver.redact_log("token-value")
        headers = httpx.Headers({"Authorization": "Bearer token-value"})
        request = {"method": "POST", "url": "https://build.example.com/builds?key=token-value"}
        stages = {"stages": ("queued", "building")}
        try:
            client_log.error("request failed, sent %s", headers)
            client_log.error("%(method)s failed", request)
            client_log.error(headers)
            client_log.error("try %d of %d", 2, 3)
            client_log.error(stages)
        finally:
            client_log.removeHandler(handler)
            client_log.propagate = True
        masked, named, an_object, harmless, a_harmless_object = seen
        # Each message is what it was, as text, and what held more than it printed is gone.
        assert "token-value" not in repr(headers)
        assert (masked.msg, masked.args) == (f"request failed, sent {headers!r}", None)
        assert (named.msg, named.args) == ("POST failed", None)
        assert (an_object.msg, an_object.args) == (str(headers), None)
        # What holds no credential is left for a handler as it was logged.
        assert (harmless.msg, harmless.args) == ("try %d of %d", (2, 3))
        assert a_harmless_object.msg is stages

    def test_a_logged_value_without_the_credential_is_passed_on_as_it_is(self) -> None:
        redact = driver._redactor("token-value")
        untouched = {"Accept": ["application/json"], "tries": 3}
        assert driver._redact_value(untouched, redact) is untouched

        class Stage(str, Enum):
            BUILDING = "building"
            LEAKING = "token-value"

        # A string of another type too: `%(stage)s` prints a member's name, which its plain text would not.
        assert driver._redact_value(Stage.BUILDING, redact) is Stage.BUILDING
        assert driver._redact_value(Stage.LEAKING, redact) == "[credential]"

        class Request:
            def __repr__(self) -> str:
                return "<Request Authorization: Bearer token-value>"

        assert driver._redact_value(Request(), redact) == "<Request Authorization: Bearer [credential]>"
        nested: list[object] = ["token-value"]
        for _ in range(10):
            nested = [nested]
        assert "token-value" not in repr(driver._redact_value(nested, redact))

    def test_a_logged_object_is_checked_as_a_formatter_would_print_it(self) -> None:
        """`%(field)s` prints what `str` gives, which can say more than `repr`: a URL that masks its key in one
        and not the other."""

        class Url:
            def __repr__(self) -> str:
                return "URL('https://build.example.com/builds?key=********')"

            def __str__(self) -> str:
                return "https://build.example.com/builds?key=token-value"

        redact = driver._redactor("token-value")
        assert driver._redact_value(Url(), redact) == "https://build.example.com/builds?key=[credential]"

    def test_a_record_made_from_a_mapping_is_redacted_too(self, signals: None) -> None:
        """As a record relayed from another process is made: its fields are set all at once, after the factory
        has returned it."""
        driver.redact_log("token-value")
        record = logging.makeLogRecord({"msg": "401 for %s", "args": ("token-value",), "levelno": logging.ERROR})
        assert record.getMessage() == "401 for [credential]"

    @pytest.mark.parametrize("chained", [False, True], ids=["in-its-own-text", "in-what-it-was-raised-from"])
    def test_an_exception_on_a_record_made_from_a_mapping_is_redacted_as_text(
        self, signals: None, chained: bool
    ) -> None:
        """A formatter prints the exception with everything it was raised from. It needs the exception whole,
        so one that holds the credential is kept as its text, redacted, and not as itself."""
        driver.redact_log("token-value")
        try:
            try:
                raise ValueError("refused: Bearer token-value")
            except ValueError as refused:
                if chained:
                    raise RuntimeError("the call failed") from refused
                raise
        except Exception:
            # In the order a record's own fields are in: the exception, then its text, unset.
            record = logging.makeLogRecord({"msg": "the call failed", "exc_info": sys.exc_info(), "exc_text": None})
        assert record.exc_info is None
        printed = logging.Formatter().format(record)
        assert "token-value" not in printed and "ValueError: refused: Bearer [credential]" in printed

    def test_a_mappings_own_text_of_its_exception_is_kept_redacted(self, signals: None) -> None:
        """A record relayed with the text another process made of its exception."""
        driver.redact_log("token-value")
        try:
            raise ValueError("refused: Bearer token-value")
        except ValueError:
            fields = {"msg": "the call failed", "exc_info": sys.exc_info(), "exc_text": "relayed: Bearer token-value"}
        record = logging.makeLogRecord(fields)
        assert record.exc_info is None
        assert logging.Formatter().format(record) == "the call failed\nrelayed: Bearer [credential]"

    def test_an_exception_that_cant_be_checked_is_withheld_from_a_record_made_from_a_mapping(
        self, signals: None
    ) -> None:
        """`True` in place of the exception, which only a log call turns into one."""
        driver.redact_log("token-value")
        record = logging.makeLogRecord({"msg": "the call failed", "exc_info": True})
        assert (record.exc_info, record.exc_text) == (None, driver.WITHHELD)

    def test_a_handlers_own_way_of_formatting_an_exception_is_not_used_once_there_is_a_secret(
        self, signals: None
    ) -> None:
        """A step program's format may put a traceback on one line. It is handed the exception as text, made
        as the standard formatter makes it, whether or not anything was taken out of it: the exception itself
        holds more than its text."""
        driver.redact_log("token-value")

        class OneLine(logging.Formatter):
            def formatException(self, ei: Any) -> str:
                return f"raised {type(ei[1]).__name__}"

        try:
            raise ValueError("no such build")
        except ValueError:
            record = logging.getLogger("some.build.client").makeRecord(
                "some.build.client", logging.ERROR, __file__, 1, "start failed", (), sys.exc_info()
            )
        assert record.exc_info is None
        printed = OneLine("%(message)s").format(record)
        assert printed.startswith("start failed\nTraceback (most recent call last):\n")
        assert printed.endswith("\nValueError: no such build")

    @pytest.mark.parametrize("made", ["by-a-log-call", "from-a-mapping"])
    def test_with_no_secret_to_take_out_an_exception_is_left_for_a_handlers_own_formatter(
        self, made: str, signals: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step whose client finds the credential itself, and has given none to `redact_log` yet. The text
        made to check the exception is not left on the record in its place."""
        monkeypatch.setattr(driver, "_REDACTION", driver.Redaction())
        driver.log_redaction()

        class OneLine(logging.Formatter):
            def formatException(self, ei: Any) -> str:
                return f"raised {type(ei[1]).__name__}"

        try:
            raise ValueError("no such build")
        except ValueError:
            raised = sys.exc_info()
        if made == "by-a-log-call":
            record = logging.getLogger("some.build.client").makeRecord(
                "some.build.client", logging.ERROR, __file__, 1, "start failed", (), raised
            )
        else:
            record = logging.makeLogRecord({"msg": "start failed", "exc_info": raised, "exc_text": None})
        assert record.exc_info is raised and record.exc_text is None
        assert OneLine("%(message)s").format(record) == "start failed\nraised ValueError"

    def test_an_exception_without_the_credential_is_text_on_a_record_made_from_a_mapping_too(
        self, signals: None
    ) -> None:
        driver.redact_log("token-value")
        try:
            raise ValueError("refused: no such build")
        except ValueError:
            raised = sys.exc_info()
        record = logging.makeLogRecord({"msg": "the call failed", "exc_info": raised, "exc_text": None})
        assert record.exc_info is None
        assert "ValueError: refused: no such build" in logging.Formatter().format(record)

    def test_every_record_is_redacted_whichever_logger_and_handler_it_goes_through(self, signals: None) -> None:
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
