# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote build step's driver, against a fake service, a fake builder and a fake clock."""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig
from nemo_builder_plugin.run import remote
from nemo_builder_plugin.run.remote import (
    Failed,
    PermanentError,
    RemoteBuild,
    RemoteBuildDriver,
    RemoteBuildService,
    RetryableError,
    Running,
    Status,
    Succeeded,
    idempotency_key,
    step_main,
)
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, ContextSource, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.constants import NEMO_JOB_UID_ENVVAR

REPOSITORY = "reg.example.com/builds"
DIGEST = "sha256:" + "a" * 64
CREDENTIAL = "s3cret-token"


class FakeStop(threading.Event):
    """A stop event whose waits pass on a fake clock at once; ``at`` sets it once the clock reaches that time."""

    def __init__(self, at: float | None = None) -> None:
        super().__init__()
        self.now = 0.0
        self.at = at

    def wait(self, timeout: float | None = None) -> bool:
        self.now += timeout or 0
        if self.at is not None and self.now >= self.at:
            self.set()
        return self.is_set()


class FakeService:
    def __init__(self, statuses: list[Status | Exception] | None = None) -> None:
        self.statuses = statuses if statuses is not None else [Running(), Succeeded(f"{REPOSITORY}@{DIGEST}")]
        self.errors: dict[str, list[Exception]] = {}
        self.calls: list[tuple[str, str]] = []
        self.builds: list[RemoteBuild] = []

    def _call(self, name: str, argument: str) -> None:
        self.calls.append((name, argument))
        if self.errors.get(name):
            raise self.errors[name].pop(0)

    def upload_context(self, build: RemoteBuild) -> str:
        self.builds.append(build)
        self._call("upload", build.key)
        return f"ctx-{build.key[:6]}"

    def start(self, build: RemoteBuild, context: str) -> str:
        self._call("start", context)
        return f"build-{build.key[:6]}"

    def status(self, build_id: str) -> Status:
        self._call("status", build_id)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, Exception):
            raise status
        return status

    def cancel(self, build_id: str) -> None:
        self._call("cancel", build_id)

    def named(self, name: str) -> list[str]:
        return [call for call, _ in self.calls if call == name]


class FakeRows:
    def __init__(self, *names: str, status: str = "pending") -> None:
        self.rows = {name: ContainerImage.model_construct(name=name, status=status) for name in names}
        self.completed: list[tuple[str, str, ImageRuntimeConfig | None]] = []

    def row(self, image: str) -> ContainerImage:
        return self.rows[image]

    def complete(self, image: str, digest: str, runtime_config: ImageRuntimeConfig | None) -> ContainerImage:
        self.completed.append((image, digest, runtime_config))
        return ContainerImage.model_construct(name=image, status="ready", digest=digest)


@pytest.fixture(autouse=True)
def _packed(monkeypatch: pytest.MonkeyPatch) -> None:
    def pack(files: FilesClient, source: ContextSource, *, workspace: str, scratch: Path) -> Path:
        (packed := scratch / "context.tar.gz").write_bytes(b"tar")
        return packed

    monkeypatch.setattr(remote, "pack_context", pack)


def _image(name: str) -> RemoteBuildImage:
    return RemoteBuildImage(
        name=name,
        source=ContextSource(fileset="default/ctx"),
        platform="linux/amd64",
        registry="reg.example.com",
        repository="builds",
        tag=f"default--{name}",
    )


def _driver(
    service: RemoteBuildService, rows: FakeRows, *names: str, stop: FakeStop | None = None
) -> RemoteBuildDriver:
    stop = stop or FakeStop()
    return RemoteBuildDriver(
        service,
        rows,
        cast(FilesClient, None),
        RemoteBuildStepConfig(images=[_image(n) for n in names or ("a",)], build_timeout_seconds=60),
        workspace="default",
        job_uid="uid",
        credential=CREDENTIAL,
        stop=stop,
        monotonic=lambda: stop.now,
        sleep=lambda seconds: None,
    )


def test_each_image_is_built_and_completed_at_its_digest(tmp_path: Path) -> None:
    config = ImageRuntimeConfig(cmd=["serve"])
    service, rows = (
        FakeService([Running(), Succeeded(f"{REPOSITORY}@{DIGEST}", config, {"id": "x"})]),
        FakeRows("a", "b"),
    )
    assert _driver(service, rows, "a", "b").run(tmp_path) == 0
    assert rows.completed == [("a", DIGEST, config), ("b", DIGEST, config)]
    assert [build.target for build in service.builds] == [
        "reg.example.com/builds:default--a",
        "reg.example.com/builds:default--b",
    ]
    assert len({build.key for build in service.builds}) == 2
    assert service.builds[0].key == idempotency_key("uid", "a") != idempotency_key("other-job", "a")
    assert service.named("cancel") == []


def test_a_row_that_is_not_pending_is_left_alone(tmp_path: Path) -> None:
    service = FakeService()
    assert _driver(service, FakeRows("a", status="ready")).run(tmp_path) == 0
    assert _driver(service, FakeRows("a", status="failed")).run(tmp_path / "2") == 1
    assert service.calls == []


def test_a_transient_error_is_retried_with_the_same_key(tmp_path: Path) -> None:
    service, rows = FakeService(), FakeRows("a")
    service.errors = {"upload": [RetryableError("503")], "start": [RetryableError("timeout")] * 2}
    assert _driver(service, rows).run(tmp_path) == 0
    assert service.named("upload") == ["upload"] * 2 and service.named("start") == ["start"] * 3
    assert len({build.key for build in service.builds}) == 1


def test_a_call_that_keeps_failing_fails_that_image_only(tmp_path: Path) -> None:
    service, rows = FakeService(), FakeRows("a", "b", "c")
    service.errors = {"start": [RetryableError("503")] * 3 + [PermanentError("quota")]}
    assert _driver(service, rows, "a", "b", "c").run(tmp_path) == 1
    assert [image for image, _, _ in rows.completed] == ["c"]
    assert service.named("cancel") == []


def test_a_failed_build_is_not_cancelled_and_a_status_error_keeps_polling(tmp_path: Path) -> None:
    service, rows = FakeService([RetryableError("503"), Running(), Failed("step 3 exited 1")]), FakeRows("a")
    assert _driver(service, rows).run(tmp_path) == 1
    assert service.named("status") == ["status"] * 3 and service.named("cancel") == []
    assert rows.completed == []


def test_a_build_past_its_deadline_is_cancelled(tmp_path: Path) -> None:
    service, rows = FakeService([Running()]), FakeRows("a")
    service.errors = {"cancel": [RetryableError("503")] * 5}
    assert _driver(service, rows).run(tmp_path) == 1
    assert len(service.named("status")) == 7  # at 0, 10, ..., 60s
    assert service.named("cancel") == ["cancel"] * 3


def test_sigterm_cancels_the_build_and_starts_no_other(tmp_path: Path) -> None:
    service, rows = FakeService([Running()]), FakeRows("a", "b")
    assert _driver(service, rows, "a", "b", stop=FakeStop(at=15)).run(tmp_path) == 1
    assert service.named("cancel") == ["cancel"] and len(service.builds) == 1


@pytest.mark.parametrize(
    "reference",
    [
        f"reg.example.com/other@{DIGEST}",
        f"elsewhere.example.com/builds@{DIGEST}",
        f"{REPOSITORY}:latest",
        f"{REPOSITORY}@sha256:abc",
        f"{REPOSITORY}@{DIGEST}\n",
    ],
)
def test_an_image_published_anywhere_else_is_not_completed(tmp_path: Path, reference: str) -> None:
    rows = FakeRows("a")
    assert _driver(FakeService([Succeeded(reference)]), rows).run(tmp_path) == 1
    assert rows.completed == []


def test_each_images_files_are_removed_once_it_is_done(tmp_path: Path) -> None:
    assert _driver(FakeService(), FakeRows("a", "b"), "a", "b").run(tmp_path) == 0
    assert list(tmp_path.iterdir()) == []


def test_no_build_is_started_once_the_upload_took_the_builds_time(tmp_path: Path) -> None:
    stop = FakeStop()

    class SlowUpload(FakeService):
        def upload_context(self, build: RemoteBuild) -> str:
            stop.now += 61
            return super().upload_context(build)

    service = SlowUpload()
    assert _driver(service, FakeRows("a"), stop=stop).run(tmp_path) == 1
    assert service.named("start") == [] and service.named("cancel") == []


def test_a_stop_during_the_start_cancels_without_reading_the_build(tmp_path: Path) -> None:
    stop = FakeStop()

    class StoppedDuringStart(FakeService):
        def start(self, build: RemoteBuild, context: str) -> str:
            stop.set()
            return super().start(build, context)

    service = StoppedDuringStart()
    assert _driver(service, FakeRows("a"), stop=stop).run(tmp_path) == 1
    assert service.named("status") == [] and service.named("cancel") == ["cancel"]


def test_a_status_that_is_none_of_the_three_fails_fast_and_cancels(tmp_path: Path) -> None:
    service = FakeService([cast(Status, "building")])
    assert _driver(service, FakeRows("a")).run(tmp_path) == 1
    assert service.named("status") == ["status"] and service.named("cancel") == ["cancel"]


def test_the_credential_never_reaches_a_log_or_a_row(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    leaky = ImageRuntimeConfig(cmd=[CREDENTIAL])
    service, rows = FakeService([Succeeded(f"{REPOSITORY}@{DIGEST}", leaky)]), FakeRows("a", "b")
    service.errors = {"upload": [PermanentError(f"denied for {CREDENTIAL}")]}
    assert _driver(service, rows, "a", "b").run(tmp_path) == 1
    assert rows.completed == [("b", DIGEST, None)]
    assert "denied for [credential]" in caplog.text and CREDENTIAL not in caplog.text


class TestStepMain:
    @pytest.fixture
    def run(
        self, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
    ) -> Callable[..., tuple[int, list[str | None]]]:
        config = RemoteBuildStepConfig(images=[_image("a")], build_timeout_seconds=60)
        monkeypatch.setattr(remote, "read_step_config", lambda: config.model_dump() | getattr(request, "param", {}))
        monkeypatch.setattr(remote, "job_identity", lambda: ("default", "job"))
        monkeypatch.setattr(remote, "get_task_nemo_client", lambda name: None)
        monkeypatch.setattr(remote, "client_from_platform", lambda client, kind: None)
        monkeypatch.setattr(remote, "Builder", lambda client, workspace: FakeRows("a"))
        # step_main installs its own SIGTERM handler: put the test process's back afterwards.
        previous = signal.getsignal(signal.SIGTERM)
        request.addfinalizer(lambda: signal.signal(signal.SIGTERM, previous))
        # And the redacting filters it adds to the root logger's handlers, pytest's own among them.
        request.addfinalizer(_remove_redacting_filters)
        monkeypatch.setenv(NEMO_JOB_UID_ENVVAR, "uid")
        seen: list[str | None] = []

        def make_service(step: RemoteBuildStepConfig, credential: str | None) -> FakeService:
            assert REMOTE_CREDENTIAL_ENV not in os.environ
            logging.getLogger("a.client").warning("using %s", credential)
            seen.append(credential)
            return FakeService([Succeeded(f"{REPOSITORY}@{DIGEST}")])

        def run(**env: str) -> tuple[int, list[str | None]]:
            for name, value in env.items():
                monkeypatch.setenv(name, value)
            return step_main(make_service), seen

        return run

    def test_the_credential_is_taken_out_of_the_environment_and_the_logs(
        self, run: Callable[..., tuple[int, list[str | None]]], caplog: pytest.LogCaptureFixture
    ) -> None:
        code, seen = run(**{REMOTE_CREDENTIAL_ENV: CREDENTIAL})
        assert (code, seen) == (0, [CREDENTIAL])
        assert "using [credential]" in caplog.text and CREDENTIAL not in caplog.text

    def test_a_step_without_a_credential_gets_none(self, run: Callable[..., Any]) -> None:
        assert run() == (0, [None])

    @pytest.mark.parametrize("run", [{"uses_credential": True}], indirect=True)
    def test_a_configured_credential_that_did_not_arrive_stops_the_step(self, run: Callable[..., Any]) -> None:
        assert run() == (1, [])
        assert run(**{REMOTE_CREDENTIAL_ENV: ""}) == (1, [])


def _remove_redacting_filters() -> None:
    for handler in logging.getLogger().handlers:
        for found in [f for f in handler.filters if isinstance(f, remote._Redacting)]:
            handler.removeFilter(found)
