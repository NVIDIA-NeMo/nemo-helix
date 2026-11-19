# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote-build step: the interface a build service implements, and the driver that runs it.

A build service's step image ships a small program around :func:`step_main`::

    raise SystemExit(step_main(lambda config, credential: MyBuildService(credential)))

Everything that is the same for every service is here: fetching the context, the deadline, polling, retrying,
cancelling when the job is told to stop, and completing each row. A service implements three calls.

Each image that isn't built gets one line in the step's log, marked :data:`FAILURE_MARKER`. Jobs keeps the end of a
failed step's log on the job, and settling reads the line from there (see ``settle``), so it is the last thing the
driver logs for an image, and nothing a submitter chooses is logged with a line break in it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, TypeVar, get_args
from urllib.parse import quote, quote_plus

from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.run.fetch import ArchiveRefused, fetch_source
from nemo_builder_plugin.run.push import Builder
from nemo_builder_plugin.run.utils import read_step_config
from nemo_builder_plugin.steps import (
    REMOTE_CREDENTIAL_ENV,
    ROOT_DOCKERFILE,
    RemoteBuildImage,
    RemoteBuildStepConfig,
    WorkLayout,
)
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_UID_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from nemo_helix_plugin.log_utils import sanitize_for_log
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

#: How often the driver reads a build's status.
POLL_SECONDS = 10.0
#: The longest it waits between two tries of one call.
MAX_BACKOFF_SECONDS = 60.0
#: How long it goes on repeating a start that fails for no stated reason, before giving the image up: long enough
#: to ride out a service's restart, short enough that a deployment pointed at the wrong place says so.
START_PATIENCE_SECONDS = 600.0
#: How long it keeps trying to complete a row whose image is built, when the builder can't be reached.
COMPLETION_SECONDS = 300.0
#: How many times it asks a service to cancel one build, how long it waits between two tries, and how long it
#: means to spend on it in all: a try is repeated only if one as slow would still end within that. A call the
#: driver has made can't be cut short. Short: a pod has about 30 seconds from the stop signal.
CANCEL_TRIES = 3
CANCEL_PAUSE_SECONDS = 2.0
CANCEL_SECONDS = 20.0
#: What takes a credential's place in a text.
REDACTED = "[credential]"
#: The shortest part of a credential that is taken out on its own: one line of a credential of several.
_MIN_PART_CHARS = 8

#: Marks the one line that says why an image wasn't built: ``NHX_BUILD_FAILED <row name>: <reason>``.
FAILURE_MARKER = "NHX_BUILD_FAILED"
#: The longest reason it reports. Jobs keeps about 2,000 characters of a failed step's log.
MAX_REASON_CHARS = 400
#: What of that end is left for a set's failure lines when they are logged again, last.
TAIL_CHARS = 1800
#: What surrounds a reason on its line: the logger's prefix, and what an executor wraps a log line in (the
#: subprocess executor keeps each as a JSON object of its own).
LINE_OVERHEAD = 280
#: The largest power of two a backoff counts up to; past it the wait is :data:`MAX_BACKOFF_SECONDS` anyway.
_MAX_BACKOFF_EXPONENT = 6
#: Why an image the job was told to stop building wasn't built.
STOPPED = "cancelled: the job was told to stop"

Stage = Literal["queued", "building", "verifying", "done", "failed", "cancelled"]
_STAGES: tuple[str, ...] = get_args(Stage)
_TERMINAL: tuple[Stage, ...] = ("done", "failed", "cancelled")

_T = TypeVar("_T")


class RemoteBuildError(Exception):
    """A build service's failure. Which subclass it raises tells the driver what to do."""


class RetryableError(RemoteBuildError):
    """The call may not have happened: repeat the same call.

    Raised by ``start``, it may also mean a build was started whose ID never arrived. The driver can't cancel a
    build it can't name, so when it gives such a start up it reports the build's key.
    """


class OverBudgetError(RemoteBuildError):
    """A budget that frees with time is used up, or the service says it is busy: back off, until the build's
    deadline."""


class PermanentError(RemoteBuildError):
    """Repeating the call would fail the same way: fail the image with this message."""


@dataclass(frozen=True, slots=True)
class RemoteBuild:
    """One image for a service to build."""

    #: Stable for this job attempt and image. ``start`` twice with one key starts one build.
    key: str
    #: The build context, with the Dockerfile at its root.
    context: Path
    platform: str
    #: Where the image came from, for the service's own records: ``workspace``, ``image`` and ``job``.
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RemoteImage:
    """What a service built."""

    #: ``<registry>/<repository>@sha256:<digest>``.
    reference: str
    #: The service's own handle for the image, if it has one.
    image_id: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RemoteBuildStatus:
    stage: Stage
    #: One line: the service's own name for where the build is, or why it failed.
    detail: str | None = None
    #: Set when ``stage`` is ``done``.
    image: RemoteImage | None = None


class RemoteBuildService(Protocol):
    """What a build service implements. Each call raises a :class:`RemoteBuildError` subclass when it fails.

    Each call returns or raises within a bounded time, by its own request timeouts, and raises
    :class:`RetryableError` when one passes. The driver can't interrupt a call: it reads the build's deadline and
    the job's stop signal only between calls, and a pod has about 30 seconds from that signal to cancel its build.
    """

    def start(self, build: RemoteBuild) -> str:
        """Start building, and return the service's ID for the build.

        Safe to repeat with the same ``build.key``: the driver calls again when a response is lost. A repeated
        call must send the service the same request, so an implementation that uploads the context keeps the first
        upload for that key and reuses it.
        """
        ...

    def status(self, build_id: str) -> RemoteBuildStatus: ...

    def cancel(self, build_id: str) -> None:
        """Ask the service to stop. It may finish anyway.

        Safe to repeat: the driver asks again, a few times, when this raises :class:`RetryableError` or
        :class:`OverBudgetError`. It returns normally for a build that has already ended or been cancelled: any
        error is taken to mean the build may still be running, and the image's row then says so.
        """
        ...


class _Builder(Protocol):
    """What the driver needs from the builder's routes."""

    def row(self, image: str) -> ContainerImage: ...

    def complete(
        self,
        image: str,
        digest: str,
        runtime_config: ImageRuntimeConfig | None,
        external: ExternalImage | None = None,
    ) -> ContainerImage: ...


class ImageFailed(Exception):
    """One image wasn't built. The message is the one-line reason."""

    def __init__(self, reason: str, *, ending: str = "", maybe_called: bool = False, uncancelled: bool = False) -> None:
        super().__init__(reason)
        #: What follows the reason and is kept when the reason is cut: the service's name for the build.
        self.ending = ending
        #: Whether a call that was given up may have reached the service all the same.
        self.maybe_called = maybe_called
        #: Whether the build was to be cancelled and the service could not be told: it may still be running.
        self.uncancelled = uncancelled


@dataclass(frozen=True, slots=True)
class Clock:
    """Time, as the driver reads it. Tests pass their own."""

    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    #: A wait that the job being told to stop does not cut short, as ``sleep`` may be: between two tries to cancel.
    pause: Callable[[float], None] = time.sleep


@dataclass(frozen=True, slots=True)
class JobAttempt:
    """This run of the job, from what Jobs sets in a step's environment."""

    workspace: str
    #: The job's entity ID. A job's name isn't unique enough: a later job can reuse it.
    job_uid: str
    #: A rerun of the job is a new attempt, under the same job ID.
    attempt_id: str

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> JobAttempt:
        names = (NEMO_JOB_WORKSPACE_ENVVAR, NEMO_JOB_UID_ENVVAR, NEMO_JOB_ATTEMPT_ID_ENVVAR)
        values = [environ.get(name, "") for name in names]
        missing = [name for name, value in zip(names, values, strict=True) if not value]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} not set; this program only runs as a platform job step")
        return cls(*values)


def idempotency_key(attempt: JobAttempt, image: str) -> str:
    """The key for one image in one attempt: the same on a repeated call, different in a rerun or another job."""
    material = "\0".join((attempt.job_uid, attempt.attempt_id, image))
    return hashlib.sha256(material.encode()).hexdigest()


def stage_context(context: Path, dockerfile: str) -> Path:
    """``context``, once ``dockerfile`` is a regular file named ``Dockerfile`` at its root, where a service builds it.

    Done in place, in the step's own copy of the context. A copy of the tree made elsewhere would move its relative
    links, which were only checked to stay inside the context where they were unpacked.

    Docker resolves ``COPY`` against the context, not the Dockerfile's directory, so moving the file changes what a
    build reads in two cases. A context with a root ``Dockerfile`` of its own loses it to the one the spec names.
    And an ignore file named for a Dockerfile (``<its path>.dockerignore``) is not moved with it, so a builder that
    reads one reads the root ``Dockerfile.dockerignore`` instead, if the context has one.
    """
    root = context.resolve()
    source = (context / dockerfile).resolve()
    if not context.is_dir() or root not in source.parents or not source.is_file():
        raise ImageFailed(f"dockerfile {dockerfile!r} is not a file inside the build context")
    target = context / ROOT_DOCKERFILE
    if not target.is_symlink():
        if target.is_dir():
            raise ImageFailed(
                f"the build context has a directory named {ROOT_DOCKERFILE} at its root, where the build service "
                "reads the Dockerfile"
            )
        if target.is_file() and target.samefile(source):
            return context
        if target.exists():
            logger.warning(
                "the context's own root %s is replaced by %s for this build",
                ROOT_DOCKERFILE,
                sanitize_for_log(dockerfile),
            )
    # A link too: a service's packager may not follow one.
    target.unlink(missing_ok=True)
    shutil.copyfile(source, target)
    return context


def _backoff(tries: int) -> float:
    """How long to wait after ``tries`` failed tries: doubling, up to :data:`MAX_BACKOFF_SECONDS`."""
    return min(MAX_BACKOFF_SECONDS, 2.0 ** min(tries, _MAX_BACKOFF_EXPONENT))


def _call(
    what: str,
    call: Callable[[], _T],
    *,
    clock: Clock,
    deadline: float,
    cancelled: Callable[[], bool],
    patience: float | None = START_PATIENCE_SECONDS,
) -> _T:
    """``call``, repeated as its failure says, until the build's deadline or the job is told to stop.

    A used-up budget is waited out for as long as the deadline allows. A failure with no stated reason is repeated
    for ``patience`` seconds on end; with None, until the deadline too.
    """
    failures = waits = 0
    # Once a try has failed for no stated reason, the call may have happened: whoever gives it up is told so.
    lost = False
    #: When the failures for no stated reason, now running, began.
    since: float | None = None
    while True:
        if cancelled():
            raise ImageFailed(STOPPED, maybe_called=lost)
        try:
            return call()
        except PermanentError as exc:
            raise ImageFailed(f"{what}: {exc}", maybe_called=lost) from exc
        except OverBudgetError as exc:
            failure: Exception = exc
            failures, since = 0, None
            waits += 1
            wait = _backoff(waits)
        except RetryableError as exc:
            failure = exc
            lost = True
            failures += 1
            now = clock.monotonic()
            since = now if since is None else since
            wait = _backoff(failures)
            if patience is not None and now + wait - since > patience:
                raise ImageFailed(
                    f"{what} failed {failures} times over {now - since:.0f}s: {failure}", maybe_called=True
                ) from failure
        except Exception as exc:
            # Not one of the three: the service's client broke its contract, and a repeat would tell nothing.
            raise ImageFailed(f"{what}: unexpected {type(exc).__name__}: {exc}", maybe_called=True) from exc
        if clock.monotonic() + wait >= deadline:
            raise ImageFailed(f"{what}: {failure}; still failing at the build timeout", maybe_called=lost) from failure
        logger.warning("%s failed (%s); trying again in %.0fs", what, sanitize_for_log(str(failure)), wait)
        clock.sleep(wait)


def _cancel(service: RemoteBuildService, build_id: str, *, clock: Clock) -> bool:
    """Ask the service to stop ``build_id``. False if it could not be told: the build may still be running.

    A try that may not have reached the service is repeated, a few times and briefly. The row fails either way, and
    whatever goes wrong here must not replace the reason the build is being cancelled.
    """
    began = clock.monotonic()
    tries = 0
    while True:
        tries += 1
        asked = clock.monotonic()
        try:
            service.cancel(build_id)
        except (RetryableError, OverBudgetError) as exc:
            failure: Exception = exc
            now = clock.monotonic()
            # Another try may take as long as this one did, and must still end within the time there is.
            ends = (now - began) + CANCEL_PAUSE_SECONDS + (now - asked)
            if tries < CANCEL_TRIES and ends < CANCEL_SECONDS:
                clock.pause(CANCEL_PAUSE_SECONDS)
                continue
        except Exception as exc:
            failure = exc
        else:
            logger.info("asked the build service to cancel %s", sanitize_for_log(build_id))
            return True
        logger.warning(
            "could not cancel %s after %d tries: %s",
            sanitize_for_log(build_id),
            tries,
            sanitize_for_log(str(failure)),
        )
        return False


def _ending(build_id: str, *, uncancelled: bool = False) -> str:
    """What follows an image's reason: the service's name for its build, and whether that build may live on."""
    if uncancelled:
        return f" [build {build_id}; the service could not be told to stop it, it may still be running]"
    return f" [build {build_id}]"


def _wait(
    service: RemoteBuildService,
    build_id: str,
    *,
    clock: Clock,
    deadline: float,
    timeout_seconds: int,
    cancelled: Callable[[], bool],
) -> RemoteBuildStatus:
    """Poll until the build ends.

    Cancels it, and raises, if the deadline passes, the job is told to stop, its status can't be read, or the
    service's client reports a stage the driver doesn't know. The :class:`ImageFailed` says whether the service
    could be told (``uncancelled``).
    """
    last: str | None = None
    while True:
        try:
            # A read that fails is repeated until the deadline: the build is running, and worth more than a read.
            status = _call(
                "reading the build",
                lambda: service.status(build_id),
                clock=clock,
                deadline=deadline,
                cancelled=cancelled,
                patience=None,
            )
        except ImageFailed as exc:
            exc.uncancelled = not _cancel(service, build_id, clock=clock)
            raise
        seen = f"{status.stage} ({status.detail})" if status.detail else status.stage
        if seen != last:
            logger.info("build %s: %s", sanitize_for_log(build_id), sanitize_for_log(seen))
            last = seen
        if status.stage in _TERMINAL:
            return status
        if status.stage not in _STAGES:
            # The service's client broke its contract. Read as "still running", the build would be polled until
            # its timeout, and one that had ended would never be seen to end.
            confirmed = _cancel(service, build_id, clock=clock)
            raise ImageFailed(
                f"the build service's client reports a stage the driver doesn't know: {status.stage!r}",
                uncancelled=not confirmed,
            )
        remaining = deadline - clock.monotonic()
        if remaining <= 0:
            confirmed = _cancel(service, build_id, clock=clock)
            raise ImageFailed(f"timed out after {timeout_seconds}s, while {seen}", uncancelled=not confirmed)
        clock.sleep(min(POLL_SECONDS, remaining))


def _handle(built: RemoteImage, build_id: str) -> ExternalImage:
    """What the row records of the service's own names for the image.

    The handle is taken whole or not at all. What the service merely reports about the build is cut to fit: a
    long attribute must not cost an image that is built and published.
    """
    attributes = {str(key)[:128]: str(value)[:512] for key, value in list(built.attributes.items())[:32]}
    try:
        return ExternalImage(image_id=built.image_id or None, build_id=build_id or None, attributes=attributes)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors())
        raise ImageFailed(f"the build service's handle for the image can't be recorded ({problems})") from exc


def _complete(
    builder: _Builder,
    image: str,
    digest: str,
    external: ExternalImage,
    *,
    clock: Clock,
    cancelled: Callable[[], bool],
) -> ContainerImage:
    """Complete the row, trying for :data:`COMPLETION_SECONDS` while the builder can't be reached.

    The image is built and published, and only this call is left; completing is safe to repeat.
    """
    deadline = clock.monotonic() + COMPLETION_SECONDS
    waits = 0
    while True:
        try:
            return builder.complete(image, digest, None, external)
        except NemoTransportError as exc:
            failure: Exception = exc
        except NemoHTTPError as exc:
            # 422 is tried again too. The request is made with the builder's own model, so the pod that refuses
            # its shape is one from before the handle existed, mid-upgrade, and the next try may reach a newer one.
            if exc.status_code not in (422, 429) and exc.status_code < 500:
                raise ImageFailed(f"it was built, but the builder refused to complete it: {exc}") from exc
            failure = exc
        waits += 1
        wait = _backoff(waits)
        if cancelled() or clock.monotonic() + wait >= deadline:
            raise ImageFailed(f"it was built, but the builder could not be told: {failure}") from failure
        logger.warning("completing %s failed (%s); trying again in %.0fs", image, sanitize_for_log(failure), wait)
        clock.sleep(wait)


def _build_one(
    image: RemoteBuildImage,
    *,
    row: ContainerImage,
    config: RemoteBuildStepConfig,
    service: RemoteBuildService,
    builder: _Builder,
    fetch: Callable[[RemoteBuildImage, Path], Path],
    attempt: JobAttempt,
    clock: Clock,
    cancelled: Callable[[], bool],
    scratch: Path,
) -> None:
    try:
        context = stage_context(fetch(image, scratch), image.dockerfile)
    except ImageFailed:
        raise
    except (ArchiveRefused, NemoHTTPError, NemoTransportError, OSError, ValueError) as exc:
        raise ImageFailed(f"cannot fetch its build context: {exc}") from exc
    build = RemoteBuild(
        key=idempotency_key(attempt, image.image),
        context=context,
        platform=image.platform,
        labels={"workspace": attempt.workspace, "image": image.image, "job": attempt.job_uid},
    )
    # The row was written with the service's repository at submit. An image anywhere else isn't the row's.
    destination = f"{row.registry}/{row.repository}"
    deadline = clock.monotonic() + config.build_timeout_seconds
    # The same `build` on every try, so a response lost on the way back starts nothing twice. Nothing is started
    # once the job has been told to stop.
    try:
        build_id = _call(
            "starting the build", lambda: service.start(build), clock=clock, deadline=deadline, cancelled=cancelled
        )
    except ImageFailed as exc:
        if exc.maybe_called:
            # A build may be running whose ID never arrived, so this step can't cancel it. Its key finds it.
            exc.ending = f" [a build may be running, started with key {build.key}]"
        raise
    logger.info("%s is build %s", sanitize_for_log(image.image), sanitize_for_log(build_id))
    ended = False
    try:
        status = _wait(
            service,
            build_id,
            clock=clock,
            deadline=deadline,
            timeout_seconds=config.build_timeout_seconds,
            cancelled=cancelled,
        )
        ended = True
        if status.stage != "done":
            raise ImageFailed(f"the build service reports it {status.stage}: {status.detail or 'no reason given'}")
        built = status.image
        if built is None:
            raise ImageFailed("the build service reports it done, with no image")
        repository, _, digest = built.reference.rpartition("@")
        if repository != destination or not _DIGEST.fullmatch(digest):
            raise ImageFailed(f"the build service published {built.reference!r}, not a digest in {destination}")
        done = _complete(builder, image.image, digest, _handle(built, build_id), clock=clock, cancelled=cancelled)
        if done.status != "ready" or done.digest != digest:
            raise ImageFailed(f"the builder completed it as {done.status} at {done.digest}, not ready at {digest}")
    except ImageFailed as exc:
        # The service's record of the build is what an operator needs to look further.
        exc.ending = _ending(build_id, uncancelled=exc.uncancelled)
        raise
    except Exception as exc:
        # Whatever broke, the build this step started is not left running with nothing watching it.
        uncancelled = not ended and not _cancel(service, build_id, clock=clock)
        raise ImageFailed(
            f"unexpected {type(exc).__name__}: {exc}", ending=_ending(build_id, uncancelled=uncancelled)
        ) from exc
    logger.info("completed %s@%s", sanitize_for_log(destination), sanitize_for_log(digest))


def _one_line(text: str) -> str:
    """``text`` with every character that ends a line taken for a space, not only the two a terminal shows."""
    return " ".join(text.splitlines())


def _line(reason: str, ending: str, room: int) -> str:
    """``reason`` then ``ending``. ``reason`` is cut to keep the two within ``room``; ``ending`` never is: it
    names the build, and is what finds the rest."""
    space = max(room - len(ending), 1)
    if len(reason) > space:
        reason = reason[: space - 1] + "…"
    return reason + ending


def _report(image: str, reason: str, ending: str = "", *, room: int = MAX_REASON_CHARS) -> None:
    """Log the one line that says why ``image`` wasn't built."""
    logger.error("%s %s: %s", FAILURE_MARKER, sanitize_for_log(image), _line(reason, ending, room))


def _redactor(secret: str | None) -> Callable[[str], str]:
    """Takes ``secret`` out of a text, in each form a text may hold it.

    As it is. As a log line holds it, without its line breaks, and as a one-line reason does, with a space for
    each: the two differ for a secret of more than one line, such as a key file. As an error that quotes it
    holds it: escaped, as ``repr`` or JSON write it, or as a URL does. And for a secret of several lines, each
    line long enough to be worth taking out on its own.
    """
    if not secret:
        return lambda text: text
    forms = {
        secret,
        sanitize_for_log(secret),
        _one_line(secret),
        repr(secret)[1:-1],
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
        secret.encode("unicode_escape").decode("ascii"),
        quote(secret, safe=""),
        quote_plus(secret),
    }
    lines = [line.strip() for line in secret.splitlines()]
    if len(lines) > 1:
        forms.update(line for line in lines if len(line) >= _MIN_PART_CHARS)
    # The longest first, so that one form is never left as the remains of another.
    found = re.compile("|".join(re.escape(form) for form in sorted(forms - {""}, key=len, reverse=True)))

    def redact(text: str) -> str:
        # What is already taken out is not read again: a short credential may be part of the mark itself.
        return REDACTED.join(found.sub(REDACTED, part) for part in text.split(REDACTED))

    return redact


def run_remote_build(
    config: RemoteBuildStepConfig,
    *,
    service: RemoteBuildService,
    builder: _Builder,
    fetch: Callable[[RemoteBuildImage, Path], Path],
    attempt: JobAttempt,
    scratch: Path,
    clock: Clock | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    redact: Callable[[str], str] = lambda text: text,
) -> int:
    """Build each image the step was given. 0 if every one is ``ready``, 1 otherwise.

    One image's failure doesn't cost the others. ``fetch`` downloads an image's build context into the directory it
    is given, which is removed once the image is done, and returns the context's directory; the driver may change
    what is in it. ``redact`` takes whatever must not be reported out of a reason, before the reason is cut.
    """
    clock = clock or Clock()
    failures = 0
    #: Each image this run failed, with its reason and what follows it, as they were logged.
    failed: list[tuple[str, str, str]] = []

    def report(image: str, reason: str, ending: str = "") -> None:
        reason, ending = redact(_one_line(redact(reason))), redact(_one_line(redact(ending)))
        _report(image, reason, ending)
        failed.append((image, reason, ending))

    for image in config.images:
        if cancelled():
            report(image.image, STOPPED)
            failures += 1
            continue
        try:
            row = builder.row(image.image)
        except NemoClientError as exc:
            report(image.image, f"cannot read its row: {exc}")
            failures += 1
            continue
        if row.status not in ("queued", "pending"):
            logger.info("%s is already %s; nothing to build", sanitize_for_log(image.image), row.status)
            failures += row.status != "ready"
            continue
        try:
            with tempfile.TemporaryDirectory(dir=scratch, prefix="image-") as directory:
                _build_one(
                    image,
                    row=row,
                    config=config,
                    service=service,
                    builder=builder,
                    fetch=fetch,
                    attempt=attempt,
                    clock=clock,
                    cancelled=cancelled,
                    scratch=Path(directory),
                )
        except ImageFailed as exc:
            report(image.image, str(exc), exc.ending)
            failures += 1
        except Exception as exc:
            # Any error, anticipated or not: one image must not cost the rest of the set. Where it was raised, but
            # not its message as a traceback prints it: that may hold a line break a submitter chose.
            logger.error(
                "building %s failed unexpectedly, at:\n%s",
                sanitize_for_log(image.image),
                "".join(traceback.format_tb(exc.__traceback__)),
            )
            report(image.image, f"unexpected {type(exc).__name__}: {exc}")
            failures += 1
    logger.info("%d image(s), %d failure(s)", len(config.images), failures)
    if failed and len(config.images) > 1:
        # Jobs keeps only the end of a failed step's log, and whatever a later image logged came after an earlier
        # image's line. So each line again, last, its reason cut to share what is kept. A build's ID or key is
        # never cut, so past about four failures in one set the first of these lines fall outside it.
        share = TAIL_CHARS // len(failed)
        for image, reason, ending in failed:
            room = max(share - len(image) - LINE_OVERHEAD, len(ending) + 1)
            _report(image, reason, ending, room=min(MAX_REASON_CHARS, room))
    return 1 if failures else 0


def take_credential(environ: MutableMapping[str, str]) -> str:
    """The service's credential, removed from ``environ`` so nothing the step runs inherits it."""
    value = environ.pop(REMOTE_CREDENTIAL_ENV, "").strip()
    if not value:
        raise RuntimeError(f"{REMOTE_CREDENTIAL_ENV} not set: the build service's credential did not reach this step")
    return value


def _redact_record(record: logging.LogRecord, redact: Callable[[str], str]) -> logging.LogRecord:
    """``record``, with the credential out of everything a handler prints of it."""
    try:
        message, formatted = record.getMessage(), True
    except Exception:
        # A log call whose arguments don't fit its message. A handler would report that by printing both as
        # they are, so they are put together here, where the credential can be taken out of them.
        message, formatted = f"{record.msg} (logged with {record.args!r})", False
    if (redacted := redact(message)) != message or not formatted:
        record.msg, record.args = redacted, None
    # A formatter prints the exception and the stack after the message, and an exception's own text with it.
    # It prints `exc_text` as it finds it, so the redacted text is put there first.
    if record.exc_info and not record.exc_text:
        record.exc_text = logging.Formatter().formatException(record.exc_info)
    if record.exc_text:
        record.exc_text = redact(record.exc_text)
    if record.stack_info:
        record.stack_info = redact(record.stack_info)
    return record


def redact_log(secret: str) -> None:
    """From here on, every record this process logs has ``secret`` taken out of it.

    Where records are made, not on the handlers there are now: a service's client may log through a logger that
    doesn't pass its records up, or add a handler of its own later.
    """
    redact = _redactor(secret)
    make = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        return _redact_record(make(*args, **kwargs), redact)

    logging.setLogRecordFactory(factory)


def stop_requested() -> Callable[[], bool]:
    """True once Jobs tells the step to stop, with SIGTERM or SIGINT. Call it from the main thread."""
    stopped: list[int] = []
    for number in (signal.SIGTERM, signal.SIGINT):
        # Only notes the signal: nothing a handler may not do while the driver is in the middle of something.
        signal.signal(number, lambda received, _frame: stopped.append(received))
    return lambda: bool(stopped)


def sleep_unless(cancelled: Callable[[], bool]) -> Callable[[float], None]:
    """A sleep that ends early once ``cancelled``.

    A signal doesn't cut ``time.sleep`` short, and a pod has about 30 seconds from the signal to cancel its build.
    """

    def sleep(seconds: float) -> None:
        end = time.monotonic() + seconds
        while not cancelled() and (left := end - time.monotonic()) > 0:
            time.sleep(min(1.0, left))

    return sleep


def fetch_context(files_client: FilesClient, image: RemoteBuildImage, *, workspace: str, root: Path) -> Path:
    """Download ``image``'s build context from Files into ``root``, acting for the submitter, and return its
    directory."""
    layout = WorkLayout(PurePosixPath(root))
    fetch_source(files_client, layout, image.source, workspace=workspace)
    return Path(layout.context(image.source))


def step_main(
    make_service: Callable[[RemoteBuildStepConfig, str | None], RemoteBuildService],
    *,
    read_credential: Callable[[], str] | None = None,
) -> int:
    """The whole step. ``make_service`` gets the step's config and the service's credential.

    The credential is the one in the environment, when the deployment keeps it in a platform secret. Otherwise it
    is what ``read_credential`` returns, for a step program whose Jobs profile mounts it as a file. Either way the
    driver keeps it out of the log and out of every reason. With neither, ``make_service`` gets None and finds
    the credential itself, and keeping it out of its own errors is then the service's.
    """
    # The level first, in brackets: the Jobs launcher reads it from there, and records any other line as an error.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    config = RemoteBuildStepConfig.model_validate(read_step_config())
    redact = _redactor(None)
    try:
        if config.credential_in_environment:
            credential: str | None = take_credential(os.environ)
        else:
            credential = (read_credential() if read_credential is not None else "").strip() or None
        if credential is not None:
            redact = _redactor(credential)
            redact_log(credential)
        attempt = JobAttempt.from_environment(os.environ)
        # As `service:builder`, acting for the submitter through the job's delegation.
        client = get_task_nemo_client("builder")
        builder = Builder(client_from_platform(client, BuilderClient), workspace=attempt.workspace)
        files_client = client_from_platform(client, FilesClient)
        service = make_service(config, credential)
        cancelled = stop_requested()
        # On the step's task storage when Jobs gives it one, not the container's own disk: a context can be large.
        # Last, so that nothing after it can fail and leave the directory behind.
        storage = os.environ.get(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR)
        scratch = Path(
            tempfile.mkdtemp(prefix="nhx-remote-", dir=storage if storage and Path(storage).is_dir() else None)
        )
    except Exception as exc:
        # Before there is anything to drive. Said for each image on the line settling reads, and through the log,
        # which keeps the credential out, not as the interpreter would print it.
        for image in config.images:
            reason = f"the step could not start: {type(exc).__name__}: {exc}"
            _report(image.image, redact(_one_line(redact(reason))))
        return 1
    try:
        return run_remote_build(
            config,
            service=service,
            builder=builder,
            fetch=lambda image, directory: fetch_context(
                files_client, image, workspace=attempt.workspace, root=directory / "fetched"
            ),
            attempt=attempt,
            scratch=scratch,
            clock=Clock(sleep=sleep_unless(cancelled)),
            cancelled=cancelled,
            redact=redact,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
