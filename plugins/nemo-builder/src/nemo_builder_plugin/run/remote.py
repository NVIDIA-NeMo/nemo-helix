# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote build step: what a build service's client implements, and the driver that runs it.

A step image for a service holds a client that implements :class:`RemoteBuildService`, and a console script that
calls :func:`step_main` with a factory for it. The driver packs each context, has the service build and publish it,
and completes the image's row through the builder's routes, as the in-cluster push step does.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_builder_plugin.run.fetch import ArchiveRefused
from nemo_builder_plugin.run.push import Builder
from nemo_builder_plugin.run.remote_context import ContextRefused, pack_context
from nemo_builder_plugin.run.utils import job_identity, read_step_config
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_UID_ENVVAR,
)
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class RemoteBuildError(Exception):
    """A service call failed. The message is a one-line reason, with no secret in it."""


class RetryableError(RemoteBuildError):
    """Transient: a network error, a 429 or 5xx, or a call past its own time bound. Repeating it is safe."""


class PermanentError(RemoteBuildError):
    """Repeating the call would fail the same way."""


@dataclass(frozen=True, slots=True)
class RemoteBuild:
    #: The same for every call of one image in one job attempt, and new on a rerun.
    key: str
    #: A gzipped tar of regular files and directories only, with ``Dockerfile`` at its root.
    context: Path
    platform: str
    #: ``<registry>/<repository>:<tag>``, where the service publishes the image.
    target: str
    #: When the driver cancels the build. A service that takes a timeout of its own is given this one too.
    timeout_seconds: int
    #: ``builder.remote.settings``, which the builder doesn't read.
    settings: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Running:
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Failed:
    reason: str


@dataclass(frozen=True, slots=True)
class Succeeded:
    #: ``<registry>/<repository>@sha256:<hex>``, where the service published the image.
    reference: str
    runtime_config: ImageRuntimeConfig | None = None
    #: The service's own names for what it built. Logged, not stored.
    attributes: Mapping[str, str] = field(default_factory=dict)


Status = Running | Failed | Succeeded


class RemoteBuildService(Protocol):
    """A build service's client. Each call returns, or raises a :class:`RemoteBuildError`, within a bounded time."""

    def upload_context(self, build: RemoteBuild) -> str:
        """Store ``build.context`` with the service and return its handle. Safe to repeat with one key."""
        ...

    def start(self, build: RemoteBuild, context: str) -> str:
        """Start the build and return its ID. Repeating it with one key never starts a second build."""
        ...

    def status(self, build_id: str) -> Status:
        """Read the build's status once."""
        ...

    def cancel(self, build_id: str) -> None:
        """Ask the service to stop the build. Safe to repeat; a build that already ended counts as cancelled."""
        ...


class Rows(Protocol):
    """What the driver needs from the builder: :class:`~nemo_builder_plugin.run.push.Builder`."""

    def row(self, image: str) -> ContainerImage: ...

    def complete(self, image: str, digest: str, runtime_config: ImageRuntimeConfig | None) -> ContainerImage: ...


class ImageFailed(Exception):
    """One image was not built. The message is the reason, which may hold a service's words: redact it to log it."""


def idempotency_key(job_uid: str, attempt_id: str, image: str) -> str:
    return hashlib.sha256("\0".join((job_uid, attempt_id, image)).encode()).hexdigest()


class RemoteBuildDriver:
    POLL_SECONDS = 10.0
    #: How often upload, start and cancel are tried on a :class:`RetryableError`, and how far apart.
    CALL_TRIES = 3
    RETRY_SECONDS = 2.0

    def __init__(
        self,
        service: RemoteBuildService,
        builder: Rows,
        files: FilesClient,
        config: RemoteBuildStepConfig,
        *,
        workspace: str,
        job_uid: str,
        attempt_id: str,
        credential: str | None,
        stop: threading.Event,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._service = service
        self._builder = builder
        self._files = files
        self._config = config
        self._workspace = workspace
        self._job_uid = job_uid
        self._attempt_id = attempt_id
        self._credential = credential
        self._stop = stop
        self._monotonic = monotonic
        self._sleep = sleep

    def redact(self, text: str) -> str:
        return text.replace(self._credential, "[credential]") if self._credential else text

    def run(self, scratch: Path) -> int:
        """Build each image whose row waits for it. 0 if every image is ready, else 1: the builder fails the rest
        when the job ends."""
        failures = 0
        for index, image in enumerate(self._config.images):
            name = sanitize_for_log(image.name)
            if self._stop.is_set():
                logger.error("stopping: %s not built", name)
                failures += 1
                continue
            try:
                row = self._builder.row(image.name)
                # `queued` too: the queue controller moves the rows to `pending` only after it creates this job.
                if row.status not in ("queued", "pending"):
                    logger.info("%s is already %s; not building it", name, sanitize_for_log(row.status))
                    if row.status != "ready":
                        failures += 1
                    continue
                try:
                    self.build_one(image, scratch / str(index))
                finally:
                    # One image's files at a time: a set of many large contexts must not fill the pod's disk.
                    shutil.rmtree(scratch / str(index), ignore_errors=True)
            except Exception as exc:
                # Any error: one image must not cost the rest of the set.
                reason = str(exc) if isinstance(exc, ImageFailed) else f"{type(exc).__name__}: {exc}"
                logger.error("%s not built: %s", name, sanitize_for_log(self.redact(reason)))
                failures += 1
        logger.info("built %d image(s), %d failure(s)", len(self._config.images) - failures, failures)
        return 1 if failures else 0

    def build_one(self, image: RemoteBuildImage, scratch: Path) -> None:
        scratch.mkdir(parents=True)
        try:
            context = pack_context(self._files, image.source, workspace=self._workspace, scratch=scratch)
        except (ContextRefused, ArchiveRefused) as exc:
            raise ImageFailed(f"refusing the context: {exc}") from None
        build = RemoteBuild(
            key=idempotency_key(self._job_uid, self._attempt_id, image.name),
            context=context,
            platform=image.platform,
            target=f"{image.registry}/{image.repository}:{image.tag}",
            timeout_seconds=self._config.build_timeout_seconds,
            settings=self._config.settings,
        )
        # The upload counts against the build's time: no build is started once that time has passed.
        deadline = self._monotonic() + build.timeout_seconds
        handle = self._call("uploading the context", lambda: self._service.upload_context(build))
        if self._monotonic() >= deadline:
            raise ImageFailed(f"the upload took the whole {build.timeout_seconds}s; no build was started")
        build_id = self._call("starting the build", lambda: self._service.start(build, handle))
        logger.info("%s is building as %s", sanitize_for_log(image.name), sanitize_for_log(build_id))
        try:
            status = self._wait(build_id, deadline)
        except BaseException:
            self._cancel(build_id)
            raise
        if isinstance(status, Failed):
            raise ImageFailed(f"the service failed the build: {status.reason}")
        digest = self._check(image, status)
        runtime_config = status.runtime_config
        if runtime_config is not None and self._credential and self._credential in runtime_config.model_dump_json():
            logger.warning("dropping %s's runtime config: it holds the credential", sanitize_for_log(image.name))
            runtime_config = None
        done = self._builder.complete(image.name, digest, runtime_config)
        if done.status != "ready" or done.digest != digest:
            raise ImageFailed(f"the builder completed it as {done.status} at {done.digest}, not ready at {digest}")
        logger.info(
            "completed %s at %s %s",
            sanitize_for_log(image.name),
            sanitize_for_log(status.reference),
            sanitize_for_log(self.redact(str(dict(status.attributes)))),
        )

    def _call(self, what: str, call: Callable[[], _T]) -> _T:
        for attempt in range(1, self.CALL_TRIES + 1):
            if self._stop.is_set():
                raise ImageFailed(f"stopped before {what}")
            try:
                return call()
            except RetryableError as exc:
                if attempt == self.CALL_TRIES:
                    raise ImageFailed(f"{what} failed {attempt} times: {exc}") from None
                logger.warning("%s failed (%s); retrying", what, sanitize_for_log(self.redact(str(exc))))
                self._stop.wait(self.RETRY_SECONDS)
            except RemoteBuildError as exc:
                raise ImageFailed(f"{what} failed: {exc}") from None
        raise AssertionError("unreachable")

    def _wait(self, build_id: str, deadline: float) -> Succeeded | Failed:
        """Poll until the build ends. Raises :class:`ImageFailed`, for the caller to cancel, at the deadline, on
        SIGTERM or on a status call that fails for good."""
        while True:
            if self._stop.is_set():
                raise ImageFailed("stopped while the build ran")
            try:
                status = self._service.status(build_id)
                if isinstance(status, Succeeded | Failed):
                    return status
                if not isinstance(status, Running):
                    raise ImageFailed(f"the service's client reported {type(status).__name__}, not a status")
            except RetryableError as exc:
                logger.warning(
                    "reading %s failed (%s)", sanitize_for_log(build_id), sanitize_for_log(self.redact(str(exc)))
                )
            except RemoteBuildError as exc:
                raise ImageFailed(f"reading the build failed: {exc}") from None
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ImageFailed(f"not built within {self._config.build_timeout_seconds}s")
            if self._stop.wait(min(self.POLL_SECONDS, remaining)):
                raise ImageFailed("stopped while the build ran")

    def _cancel(self, build_id: str) -> None:
        """Tried a few times, a couple of seconds apart: few enough to end within the pod's grace period."""
        for attempt in range(1, self.CALL_TRIES + 1):
            try:
                self._service.cancel(build_id)
                logger.info("cancelled %s", sanitize_for_log(build_id))
                return
            except RetryableError as exc:
                failure: Exception = exc
                if attempt < self.CALL_TRIES:
                    self._sleep(self.RETRY_SECONDS)
            except Exception as exc:
                failure = exc
                break
        logger.error("could not cancel %s: %s", sanitize_for_log(build_id), sanitize_for_log(self.redact(str(failure))))

    def _check(self, image: RemoteBuildImage, result: Succeeded) -> str:
        """The digest, only if the service published to the image's own repository."""
        repository, _, digest = result.reference.rpartition("@")
        expected = f"{image.registry}/{image.repository}"
        if repository != expected or not re.fullmatch(DIGEST_PATTERN, digest):
            raise ImageFailed(f"the service reported {result.reference!r}, not an image in {expected} by digest")
        return digest


class _Redacting(logging.Filter):
    def __init__(self, credential: str) -> None:
        super().__init__()
        self._credential = credential

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = record.getMessage().replace(self._credential, "[credential]")
        record.args = None
        if record.exc_text:
            record.exc_text = record.exc_text.replace(self._credential, "[credential]")
        if record.exc_info:
            text = logging.Formatter().formatException(record.exc_info)
            record.exc_text, record.exc_info = text.replace(self._credential, "[credential]"), None
        return True


def step_main(make_service: Callable[[RemoteBuildStepConfig, str | None], RemoteBuildService]) -> int:
    """The remote build step: what a service's step image runs, with a factory for its client.

    The credential is taken out of the environment before the client exists, so no process it starts inherits it,
    and replaced in every message the root logger's handlers write.
    """
    credential = os.environ.pop(REMOTE_CREDENTIAL_ENV, None) or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if credential is not None:
        for handler in logging.getLogger().handlers:
            handler.addFilter(_Redacting(credential))
    config = RemoteBuildStepConfig.model_validate(read_step_config())
    if config.uses_credential and credential is None:
        # Fail closed: a client that falls back to the pod's own identity must not do so by accident.
        logger.error("a credential secret is configured, but %s is unset or empty", REMOTE_CREDENTIAL_ENV)
        return 1
    workspace, _ = job_identity()
    job_uid, attempt_id = os.environ.get(NEMO_JOB_UID_ENVVAR), os.environ.get(NEMO_JOB_ATTEMPT_ID_ENVVAR)
    if not job_uid or not attempt_id:
        logger.error("%s and %s must both be set", NEMO_JOB_UID_ENVVAR, NEMO_JOB_ATTEMPT_ID_ENVVAR)
        return 1
    # As `service:builder`, acting for the submitter, as the in-cluster push step does.
    client = get_task_nemo_client("builder")
    builder = Builder(client_from_platform(client, BuilderClient), workspace=workspace)
    files_client = client_from_platform(client, FilesClient)
    try:
        service = make_service(config, credential)
    except Exception as exc:
        logger.error("cannot create the build service's client: %s: %s", type(exc).__name__, exc)
        return 1
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    driver = RemoteBuildDriver(
        service,
        builder,
        files_client,
        config,
        workspace=workspace,
        job_uid=job_uid,
        attempt_id=attempt_id,
        credential=credential,
        stop=stop,
    )
    # On the task storage Jobs mounts on every job pod, which a read-only root filesystem leaves writable.
    with tempfile.TemporaryDirectory(
        prefix="nhx-remote-build-", dir=os.environ.get(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR)
    ) as scratch:
        return driver.run(Path(scratch))
