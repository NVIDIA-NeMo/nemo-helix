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
#: What an ignore file that belongs to one Dockerfile is named: the Dockerfile's name, then this.
_IGNORE_SUFFIX = ".dockerignore"
#: How deep into a logged value the driver looks for a credential before it treats what is left as text.
_MAX_FIELD_DEPTH = 6
#: The shortest part of a credential that is taken out on its own: one line of a credential of several.
_MIN_PART_CHARS = 8

#: Marks the one line that says why an image wasn't built: ``NHX_BUILD_FAILED <row name>: <reason>``.
FAILURE_MARKER = "NHX_BUILD_FAILED"
#: The longest reason it reports. Jobs keeps about 2,000 characters of a failed step's log.
MAX_REASON_CHARS = 400
#: What of that end is left for a set's failure lines when they are logged again, last.
TAIL_CHARS = 1800
#: What surrounds a reason on its line: the logger's prefix (level, time and logger name, about 65 characters),
#: the marker, and a little to spare. The executors that run the step keep its log lines as they are printed.
LINE_OVERHEAD = 100
#: The least room for a reason that makes a failure line worth logging again: a line cut to less says nothing,
#: and settling reads the last line for an image.
MIN_REASON_ROOM = 40
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
    #: How long the driver waits for the build before it cancels it. A service that can end a build by itself
    #: should be told: a step that is killed outright, out of memory or with its node, cancels nothing.
    timeout_seconds: int | None = None


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
        upload for that key and reuses it. The ID is not empty: it is what the image's row names the build by.
        """
        ...

    def status(self, build_id: str) -> RemoteBuildStatus: ...

    def cancel(self, build_id: str) -> None:
        """Ask the service to stop. It may finish anyway.

        Safe to repeat: the driver asks again, a few times, when this raises :class:`RetryableError` or
        :class:`OverBudgetError`. It returns normally for a build that has already ended or been cancelled: any
        error is taken to mean the build may still be running, and the image's failure line then says so.
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
    build reads in one case: a context with a root ``Dockerfile`` of its own loses it to the one the spec names.

    What the build leaves out must not change with the move. A builder reads the ignore file named for a
    Dockerfile (``<its path>.dockerignore``) in preference to the context's own ``.dockerignore``, so that file
    goes to the root with its Dockerfile, and one that was the replaced root Dockerfile's doesn't stay to stand in
    for it.

    A context with a link that leads out of it is refused. Its links were checked, where it was unpacked, to
    stay inside what was unpacked, which may be more than this context: a service's packager that follows links
    would send a file from beside the context with it.
    """
    if not context.is_dir():
        raise ImageFailed("its build context is not a directory in what was fetched: its context path matches none")
    root = context.resolve()
    named = context / dockerfile
    source = named.resolve()
    if root not in source.parents or not source.is_file():
        raise ImageFailed(f"dockerfile {dockerfile!r} is not a file inside the build context")
    if (link := _link_out(context)) is not None:
        raise ImageFailed(f"the build context has a link that leads out of it: {link!r}")
    target = context / ROOT_DOCKERFILE
    # Whether the spec names the root Dockerfile by its own path, however that path is spelt.
    at_root = named.name == ROOT_DOCKERFILE and named.parent.resolve() == root
    #: Whether the root Dockerfile is already the named file: nothing to copy, though under another name its
    #: ignore file is another one.
    same = False
    if not target.is_symlink():
        if target.is_dir():
            raise ImageFailed(
                f"the build context has a directory named {ROOT_DOCKERFILE} at its root, where the build service "
                "reads the Dockerfile"
            )
        same = target.is_file() and target.samefile(source)
        if same and at_root:
            return context
        if target.exists() and not same:
            logger.warning(
                "the context's own root %s is replaced by %s for this build",
                ROOT_DOCKERFILE,
                sanitize_for_log(dockerfile),
            )
    ignore, staged_ignore = named.with_name(named.name + _IGNORE_SUFFIX), context / (ROOT_DOCKERFILE + _IGNORE_SUFFIX)
    if staged_ignore.is_dir() and not staged_ignore.is_symlink():
        raise ImageFailed(f"the build context has a directory named {staged_ignore.name} at its root")
    kept = ignore.read_bytes() if ignore.is_file() else None
    if not same:
        # A link too: a service's packager may not follow one.
        target.unlink(missing_ok=True)
        shutil.copyfile(source, target)
    staged_ignore.unlink(missing_ok=True)
    if kept is not None:
        staged_ignore.write_bytes(kept)
    return context


def _link_out(context: Path) -> str | None:
    """The first link under ``context`` that leads out of it, as its path in the context. None if none does."""
    root = context.resolve()
    for directory, directories, files in os.walk(context):
        for name in (*directories, *files):
            path = Path(directory, name)
            if not path.is_symlink():
                continue
            try:
                end = path.resolve()
            except (OSError, RuntimeError):
                # A loop of links, on a Python that raises for one: it reaches no file, in the context or out.
                continue
            if end != root and root not in end.parents:
                return path.relative_to(context).as_posix()
    return None


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
    at_the_deadline: bool = False,
) -> _T:
    """``call``, repeated as its failure says, until the build's deadline or the job is told to stop.

    A used-up budget is waited out for as long as the deadline allows. A failure with no stated reason is repeated
    for ``patience`` seconds on end; with None, until the deadline too.

    With ``at_the_deadline``, a wait that would pass the deadline is cut to end on it, and the call is made once
    more there: right for a read of a build that is running, which costs nothing and may find it done. Without,
    the call is given up as soon as its next try would come too late: right for a start, which would only be
    cancelled.
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
        remaining = deadline - clock.monotonic()
        if remaining <= 0:
            raise ImageFailed(f"{what}: {failure}; still failing at the build timeout", maybe_called=lost) from failure
        if wait >= remaining and not at_the_deadline:
            raise ImageFailed(
                f"{what}: {failure}; the build timeout leaves no time to try again", maybe_called=lost
            ) from failure
        wait = min(wait, remaining)
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
                at_the_deadline=True,
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


def _handle(built: RemoteImage, build_id: str, redact: Callable[[str], str]) -> ExternalImage:
    """What the row records of the service's own names for the image.

    The handle is taken whole or not at all. What the service merely reports about the build is cut to fit: a
    long attribute must not cost an image that is built and published. Every reader of the image reads these,
    so the credential is taken out of them as it is out of a reason: a service may report a link that carries it.
    """
    attributes = {
        redact(str(key))[:128]: redact(str(value))[:512] for key, value in list(built.attributes.items())[:32]
    }
    try:
        return ExternalImage(
            image_id=redact(built.image_id) if built.image_id else None,
            build_id=redact(build_id) if build_id else None,
            attributes=attributes,
        )
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
    redact: Callable[[str], str],
) -> None:
    """Build ``image``. ``fetch`` puts its build context in ``scratch``, which is this image's alone."""
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
        timeout_seconds=config.build_timeout_seconds,
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
        if not isinstance(build_id, str) or not build_id.strip():
            # The row names its build by this, and the builder completes no remote row without it.
            raise ImageFailed("the build service's client started a build without naming it", maybe_called=True)
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
        external = _handle(built, build_id, redact)
        done = _complete(builder, image.image, digest, external, clock=clock, cancelled=cancelled)
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


def _in_a_url(secret: str) -> str:
    """A pattern for ``secret`` in a URL, whichever of its characters the encoder percent-encoded."""
    pieces: list[str] = []
    for character in secret:
        literal = re.escape(character)
        if character.isascii() and (character.isalnum() or character in "-._~"):
            pieces.append(literal)
            continue
        encoded = "".join(f"%{byte:02X}" for byte in character.encode())
        choices = [literal, f"(?i:{re.escape(encoded)})"]
        if character == " ":
            choices.append(re.escape("+"))
        pieces.append(f"(?:{'|'.join(choices)})")
    return "".join(pieces)


def _redactor(secret: str | None) -> Callable[[str], str]:
    """Takes ``secret`` out of a text, in each form a text may hold it.

    As it is. As a log line holds it, without its line breaks, and as a one-line reason does, with a space for
    each: the two differ for a secret of more than one line, such as a key file. As an error that quotes it
    holds it: escaped, as ``repr`` or JSON write it, or as a URL does. And for a secret of several lines, each
    line long enough to be worth taking out on its own.
    """
    if not secret:
        return lambda text: text
    shown = repr(secret)[1:-1]
    escaped = {
        shown,
        # `repr` leaves an apostrophe as it is unless the text around it also holds a double quote.
        re.sub(r"(?<!\\)'", r"\\'", shown),
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
        secret.encode("unicode_escape").decode("ascii"),
    }
    # Quoted once more, as an error that holds another error's text is: `repr` of a tuple of messages, say.
    twice = {repr(form)[1:-1] for form in escaped} | {json.dumps(form)[1:-1] for form in escaped}
    forms = {secret, sanitize_for_log(secret), _one_line(secret), quote(secret, safe=""), quote_plus(secret)}
    forms |= escaped | twice
    lines = [line.strip() for line in secret.splitlines()]
    if len(lines) > 1:
        forms.update(line for line in lines if len(line) >= _MIN_PART_CHARS)
    # The longest first, so that one form is never left as the remains of another. Last, the secret as any URL
    # holds it: encoders differ in which characters they leave alone.
    patterns = [re.escape(form) for form in sorted(forms - {""}, key=len, reverse=True)]
    found = re.compile("|".join([*patterns, _in_a_url(secret)]))

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
    is given and returns the context's directory; the driver may change what is in it. Everything it writes is
    under ``scratch``, which the caller removes. ``redact`` takes whatever must not be reported out of a reason,
    before the reason is cut.
    """
    clock = clock or Clock()
    failures = 0
    #: Each image this run failed, with its reason and what follows it, as they were logged.
    failed: list[tuple[str, str, str]] = []
    #: Each download this run made, by what names it, and where it is: read by `context_for`.
    fetched: dict[str, Path] = {}
    #: The filesets some image of the set builds from whole: that download covers their directories too.
    whole_filesets = {
        image.source.fileset
        for image in config.images
        if image.source.archive is None and image.source.context_path is None
    }

    def context_for(image: RemoteBuildImage, directory: Path) -> Path:
        """``image``'s build context, in ``directory``.

        The images of one set are built one after another, and a source may change in between. So what a set
        downloads it downloads once, as the in-cluster build's fetch step does: an archive whole, whichever
        directory of it each image builds from, and a fileset whole if any image builds from all of it. Each
        image gets its own copy of its directory, so every image built from one download sees the same files. A
        set of one image has nothing to share, and is spared the copy.
        """
        if len(config.images) == 1:
            return fetch(image, directory)
        source = image.source
        whole = source.archive is not None or source.fileset in whole_filesets
        download = source.model_copy(update={"context_path": None}) if whole else source
        key = download.model_dump_json()
        if key not in fetched:
            into = Path(tempfile.mkdtemp(dir=scratch, prefix="source-"))
            try:
                fetched[key] = fetch(image.model_copy(update={"source": download}), into)
            except BaseException:
                # Not kept: the next image that names it tries again, in a directory of its own.
                shutil.rmtree(into, ignore_errors=True)
                raise
        context = fetched[key]
        if whole and source.context_path:
            inside = PurePosixPath(source.context_path)
            if inside.is_absolute() or ".." in inside.parts:
                raise ImageFailed(f"context path {source.context_path!r} is not a directory inside its source")
            context = context.joinpath(*inside.parts)
            if not context.is_dir():
                raise ImageFailed(f"context path {source.context_path!r} is not a directory in its source")
        # Under its own name, so that a link which comes back to it by name still does. Links as they are,
        # never followed: staging checks where each one leads.
        return Path(shutil.copytree(context, directory / (context.name or "context"), symlinks=True))

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
        except Exception as exc:
            # Such as a token that couldn't be renewed. One image must not cost the rest of the set.
            report(image.image, f"cannot read its row: unexpected {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if row.status not in ("queued", "pending"):
            logger.info("%s is already %s; nothing to build", sanitize_for_log(image.image), row.status)
            failures += row.status != "ready"
            continue
        # The failure is reported inside, before the image's context is removed: a large one takes a while to
        # delete, and a step that was told to stop has seconds left.
        with tempfile.TemporaryDirectory(dir=scratch, prefix="image-") as directory:
            try:
                _build_one(
                    image,
                    row=row,
                    config=config,
                    service=service,
                    builder=builder,
                    fetch=context_for,
                    attempt=attempt,
                    clock=clock,
                    cancelled=cancelled,
                    scratch=Path(directory),
                    redact=redact,
                )
            except ImageFailed as exc:
                report(image.image, str(exc), exc.ending)
                failures += 1
            except Exception as exc:
                # Any error, anticipated or not: one image must not cost the rest of the set. Where it was raised,
                # but not its message as a traceback prints it: that may hold a line break a submitter chose.
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
        # image's line. So each line again, last, its reason cut to share what is kept. A line with too little
        # room left for its reason is not logged again: settling reads the last line for an image, and the first
        # one, though it may have fallen outside what is kept, says more than one cut to nothing.
        for image, reason, ending in failed:
            room = _room(len(failed), image)
            if room - len(ending) >= MIN_REASON_ROOM:
                _report(image, reason, ending, room=room)
    return 1 if failures else 0


def _room(lines: int, image: str) -> int:
    """How long a reason and what follows it may be on ``image``'s failure line, for ``lines`` such lines to
    fit in what Jobs keeps of a failed step's log."""
    return min(MAX_REASON_CHARS, TAIL_CHARS // lines - len(image) - LINE_OVERHEAD)


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
    if record.exc_text and (redacted := redact(record.exc_text)) != record.exc_text:
        # The exception itself still holds what was taken out of its text, and some handlers print it themselves.
        record.exc_text, record.exc_info = redacted, None
    if record.stack_info:
        record.stack_info = redact(record.stack_info)
    return record


def _redact_value(value: Any, redact: Callable[[str], str], depth: int = 0) -> Any:
    """``value`` as a log call attached it, with the credential out of it.

    A value that doesn't hold the credential is returned as it is, whatever it is. One that does is rebuilt if
    it is a plain container, and otherwise replaced by what it prints as, redacted.
    """
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if depth < _MAX_FIELD_DEPTH:
        # Each string in it on its own, not the container as it prints: a string is quoted differently there.
        rebuilt: Any = None
        if type(value) is dict:
            rebuilt = {
                _redact_value(key, redact, depth + 1): _redact_value(item, redact, depth + 1)
                for key, item in value.items()
            }
        elif type(value) in (list, tuple, set, frozenset):
            rebuilt = type(value)(_redact_value(item, redact, depth + 1) for item in value)
        if rebuilt is not None:
            return value if rebuilt == value else rebuilt
    shown = repr(value)
    redacted = redact(shown)
    return value if redacted == shown else redacted


class _RedactedFields(dict[str, Any]):
    """A log record's fields, with the credential taken out of each one set by key.

    ``Logger.makeRecord`` attaches a log call's ``extra`` fields to the record after the record factory has
    returned it, one at a time, by key in the record's ``__dict__``. A formatter that prints such a field, a
    request's headers say, would print the credential in it. So a record's fields are kept in one of these.
    """

    def __init__(self, fields: Mapping[str, Any], redact: Callable[[str], str]) -> None:
        super().__init__(fields)
        self.redact = redact

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            value = _redact_value(value, self.redact)
        except Exception:
            # Never into the code that logged. What can't be checked is withheld, not kept as it is.
            value = WITHHELD
        super().__setitem__(key, value)


#: What stands in for a logged value, or a whole record, that could not be checked for a secret.
WITHHELD = "<withheld: it could not be checked for the build service's credential>"


class Redaction:
    """Takes every secret it has been given out of a text."""

    def __init__(self) -> None:
        self._redactors: list[Callable[[str], str]] = []

    def add(self, secret: str) -> None:
        if secret:
            self._redactors.append(_redactor(secret))

    def __call__(self, text: str) -> str:
        for redact in self._redactors:
            text = redact(text)
        return text


class _RedactingFactory:
    """A log record factory that makes records as ``make`` does, then takes the secrets out of them."""

    def __init__(self, make: Callable[..., logging.LogRecord]) -> None:
        self._make = make
        self.redaction = Redaction()

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._make(*args, **kwargs)
        try:
            _redact_record(record, self.redaction)
        except Exception:
            # Never into the code that logged. A record that can't be checked is withheld, not printed as it is.
            record.msg, record.args = WITHHELD, None
            record.exc_info = record.exc_text = record.stack_info = None
        fields = record.__dict__
        # A factory under this one may be redacting already, for secrets of its own: both, then.
        under = fields.redact if isinstance(fields, _RedactedFields) else None
        record.__dict__ = _RedactedFields(
            fields, self.redaction if under is None else lambda text: self.redaction(under(text))
        )
        return record


def log_redaction() -> Redaction:
    """What takes secrets out of everything this process logs, installed on first use.

    Where records are made, not on the handlers there are now: a service's client may log through a logger that
    doesn't pass its records up, or add a handler of its own later. That covers a record's message, its
    exception and stack, and the ``extra`` fields of the call that logged it. It is also a function, which takes
    the same secrets out of any text.
    """
    factory = logging.getLogRecordFactory()
    if not isinstance(factory, _RedactingFactory):
        factory = _RedactingFactory(factory)
        logging.setLogRecordFactory(factory)
    return factory.redaction


def redact_log(secret: str) -> None:
    """From here on, every record this process logs has ``secret`` taken out of it too, and so has every reason.

    The driver calls it with the credential it is given. A service's client calls it with whatever else must
    stay out of the log: a part of that credential it uses alone, or a token it trades the credential for. Only
    a secret as it was given here is recognised, whole, or line by line for one of several lines.
    """
    log_redaction().add(secret)


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
    driver keeps it out of the log and out of every reason, as it was given: whole, or line by line. A part of
    it that a client uses alone, or a token a client trades it for, the driver doesn't know: the client gives
    those to :func:`redact_log`. With neither, ``make_service`` gets None and finds the credential itself, and
    does the same with it.
    """
    # The level first, in brackets: the Jobs launcher reads it from there, and records any other line as an error.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    config = RemoteBuildStepConfig.model_validate(read_step_config())
    # Every secret given to `redact_log`, the client's own included, whenever it is given.
    redact = log_redaction()
    try:
        if config.credential_in_environment:
            credential: str | None = take_credential(os.environ)
        else:
            credential = (read_credential() if read_credential is not None else "").strip() or None
        if credential is not None:
            redact.add(credential)
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
        reason = redact(_one_line(redact(f"the step could not start: {type(exc).__name__}: {exc}")))
        for image in config.images:
            # Cut to share what Jobs keeps of the log, so that every image's line is in it.
            _report(image.image, reason, room=max(_room(len(config.images), image.image), MIN_REASON_ROOM))
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
