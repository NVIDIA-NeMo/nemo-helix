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

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, TypeVar, get_args
from urllib.parse import quote, quote_plus

from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.entities import (
    MAX_EXTERNAL_ATTRIBUTES,
    MAX_EXTERNAL_KEY,
    MAX_EXTERNAL_VALUE,
    ContainerImage,
    ExternalImage,
    ImageRuntimeConfig,
)
from nemo_builder_plugin.run.fetch import MAX_UNPACKED_BYTES, MAX_UNPACKED_ENTRIES, ArchiveRefused, fetch_source
from nemo_builder_plugin.run.push import Builder
from nemo_builder_plugin.run.utils import read_step_config
from nemo_builder_plugin.steps import (
    REMOTE_CREDENTIAL_ENV,
    ROOT_DOCKERFILE,
    ContextSource,
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
#: How many names the driver takes, one after another, to learn where the links of one build context lead: each
#: name of each path it follows is one. A context of as many links as a source may unpack to entries, each
#: written as a path of thirty names, is followed in fewer. One that takes more was made to be costly to
#: follow, by paths of thousands of names or by thousands of links that end in a loop, and its image fails,
#: where its links would otherwise hold the step for hours. It bounds the work, not the time. Where each
#: step reads the disk, for links that end in a loop or names that are not there, a context made to reach it
#: has held the step for about half a minute on a local disk; where none does, for names such as ``.``, for
#: a second or two. A step costs no more than the longest path it is taken at: see
#: :data:`MAX_LINK_PATH_CHARS`. The limit is on each following of a context's links, and there are two:
#: where it was fetched, and again once it is staged.
MAX_LINK_STEPS = 100 * MAX_UNPACKED_ENTRIES
#: The longest path the driver follows a context's links to. The system reads no path longer than 4,096
#: characters, so nothing is there to find at a longer one, and no one link leads to more than twice that:
#: the path of its directory, and the path it is written as. What comes to more is made of links that each
#: add names that are not there to where another led. No build can follow those. Taking a name costs the
#: length of the path it is added to, and so does remembering where a link ended, so without this the steps
#: above would bound neither the time nor the memory that following them takes.
MAX_LINK_PATH_CHARS = 2 * 4096
#: What takes a credential's place in a text.
REDACTED = "[credential]"
#: Fails where a mark that is already in a text has begun and not yet ended, at each place in it. What is taken
#: out of a text never ends there: the rest of the mark would be left beside the new one, as text.
_NOT_IN_A_MARK = "".join(
    f"(?!(?<={re.escape(REDACTED[:at])}){re.escape(REDACTED[at:])})" for at in range(1, len(REDACTED))
)
#: What an ignore file that belongs to one Dockerfile is named: the Dockerfile's name, then this. By itself, it
#: is the name of the context's own ignore file, which a builder reads for a Dockerfile that has none.
_IGNORE_SUFFIX = ".dockerignore"
#: How deep into a logged value the driver looks for a credential before it treats what is left as text.
_MAX_FIELD_DEPTH = 6
#: The containers that print everything they hold. Any other may print a mask where it holds a credential.
_PLAIN_CONTAINERS = (dict, list, tuple, set, frozenset)
#: What is a sequence and is still read as text, or holds only numbers.
_NOT_CONTAINERS = (str, bytes, bytearray, memoryview, range)
#: The shortest part of a credential that is taken out on its own: one line of a credential of several.
_MIN_PART_CHARS = 8

#: Marks the one line that says why an image wasn't built: ``NHX_BUILD_FAILED <row name>: <reason>``.
FAILURE_MARKER = "NHX_BUILD_FAILED"
#: What the marker is written as where it is not the driver's own: in what a service or a library says.
UNMARKED = "NHX-BUILD-FAILED"
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
    #: The build context, with the Dockerfile at its root. It is the whole context, as the step holds it on
    #: disk: nothing has applied its ignore rules, so what a build leaves out is still in it. The ignore file
    #: a builder reads is at its root, as a regular file: ``Dockerfile.dockerignore`` if there is one, and
    #: otherwise ``.dockerignore``. What a client owes a context it sends anywhere is said on
    #: :meth:`RemoteBuildService.start`.
    context: Path
    platform: str
    #: Where the image came from, for the service's own records: ``workspace``, ``image`` and ``job``.
    labels: Mapping[str, str] = field(default_factory=dict)
    #: How long the driver waits for the build before it cancels it. A client hands this to its service as the
    #: build's own deadline, wherever the service takes one. The driver's cancel is not enough: a step that is
    #: killed outright, out of memory or with its node, cancels nothing, and nothing else can, since the
    #: build's ID is only in the step's memory and its log. The row is then failed once its job has ended, and
    #: the build runs until the service ends it.
    timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteImage:
    """What a service built."""

    #: ``<registry>/<repository>@sha256:<digest>``.
    reference: str
    #: The service's own handle for the image, if it has one.
    image_id: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    #: How the published image starts by default: its entrypoint, command, working directory and user, from
    #: the image's config, with what it inherits from its base image. A client that can read that config
    #: reports it, with ``user`` an empty string for an image that sets none, and the row records it as it
    #: does for an image built in the cluster. Left None, the row records none, and so it does when any part
    #: of what is reported holds the build service's credential: a consumer runs what the row records, and
    #: with the credential taken out that is no longer how the image starts. A consumer whose runtime
    #: replaces a container's entrypoint then can't start the image's own: it has to refuse a workload that
    #: sets a command of its own and leaves the entrypoint to the image, and building the image again changes
    #: nothing.
    runtime_config: ImageRuntimeConfig | None = None


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

        Wherever the service takes a deadline for a build, give it ``build.timeout_seconds`` as this build's.
        The driver cancels a build it gives up, or is told to stop. A step that is killed outright cancels
        nothing, and nothing else holds the build's ID to cancel it by: only the service's own deadline then
        ends the build.

        ``build.context`` is the whole context, as it is on disk: the driver applies no ignore rules to it.
        An implementation that sends the context anywhere first applies the ignore file at the context's
        root, as a container build does, and sends nothing those rules leave out: the Dockerfile's own,
        ``Dockerfile.dockerignore``, if there is one, and otherwise ``.dockerignore``. The Dockerfile and
        that ignore file are sent whatever the rules say, as a container build still reads them: rules
        that leave out everything and then name what to keep seldom name those two. What a build leaves
        out, a file of local settings say, then never leaves the step. Rules that the service applies to
        what it was sent come too late for that.

        It follows no link: it packs each link as a link, or refuses a context that holds one. And it
        packs the names of one file, hard links of one another, as one file. The driver has checked the
        links as they are written: each leads inside the context wherever the context is unpacked, and
        none leads back to a directory it is in.
        Followed, a link is sent as a copy of what it leads to, which the ignore rules may have left out
        under its own name, and which a context's links can repeat many times over. The driver refuses a
        context that its links, followed, would make more of than a source may unpack to, but it measures
        one file once, however many names the file has: sent as a file for each name, it is as many times
        its size.
        """
        ...

    def status(self, build_id: str) -> RemoteBuildStatus:
        """Where the build is. Once it is ``done``, with the image the service published.

        A client that can read the published image's config reports how the image starts
        (``RemoteImage.runtime_config``): the row is the only place a consumer of the image finds it.
        """
        ...

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


def stage_context(context: Path, dockerfile: str, *, into: Path | None = None) -> Path:
    """``context``, once ``dockerfile`` is a regular file named ``Dockerfile`` at its root, where a service builds it.

    Done in place, where the context was fetched, when the context is this image's alone. With ``into``, it is
    not: another image of the set builds from the same download. The context is then left exactly as it is,
    and what is staged, and returned, is a copy of it at ``into``.

    Either way its links are checked where it was fetched, before anything is written. A link may lead back
    into the context by way of what is beside it there, the rest of an archive say, and a copy has none of
    that beside it: checked in the copy, such a link would lead nowhere, and a context that builds alone
    would be refused for being built with another. What is decided for each link where the context was fetched
    is then written into the context that is staged, so an image is handed the context it is handed alone,
    or is refused for the same reason.

    Docker resolves ``COPY`` against the context, not the Dockerfile's directory, so moving the file changes what a
    build reads only where it copies these files themselves: the root gains the named Dockerfile, in place of
    any ``Dockerfile`` there, and the root ``Dockerfile.dockerignore`` becomes the named Dockerfile's own, or goes.

    What the build leaves out must not change with the move. A builder reads the ignore file named for a
    Dockerfile (``<its path>.dockerignore``) in preference to the context's own ``.dockerignore``, so that file
    goes to the root with its Dockerfile, and one that was the replaced root Dockerfile's doesn't stay to stand in
    for it. Whichever of the two the build reads is left at the root as a file, never as a link: a service's
    packager may not follow one, and a link between them leads nowhere once the file it led to goes, as the
    context's ``.dockerignore`` does when it is a link to the replaced Dockerfile's own. Each becomes a copy of
    what it led to before anything at the root was replaced. The copies are made file to file, under another
    name in the context and then renamed, and no ignore file is read into memory: unpacking lets one file of
    an archive be gigabytes.

    A context with a link that leads out of it is refused. Its links were checked, where it was unpacked, to
    stay inside what was unpacked, which may be more than this context: a service's packager that follows links
    would send a file from beside the context with it. So is one with a link that leads back to a directory it
    is in, which such a packager would follow without end. And so is one that its links make too much of:
    links to directories that never come back round still repeat what they lead to, each time one is
    followed, so a context with links is measured as a packager that follows them would send it, and refused
    if that comes to more entries or bytes than a source may unpack to.

    Those limits hold for the context as the service is handed it, and staging adds to a context: the
    Dockerfile and the ignore files it writes at the root, and whatever a link to either of them then leads
    to. A context that the files to be written would carry past a limit is refused before they are written,
    where their sizes alone tell it. And one that staging wrote a file into, or that holds a link, is measured
    again, the same way, once it is staged: what a link leads to after the root's files are replaced is only
    known then.

    That is of a context that was within the limits as it was fetched, which staging must not carry past
    them. One with no link that was past a limit already is handed on unmeasured, whatever staging writes
    into it: it is as large as its fetch let it be. A fileset that is not an archive has no limit as it is
    fetched, and where its Dockerfile is doesn't give it one. And an archive is counted, as it is unpacked,
    by the entries it lists, which need not name its directories: on disk it may be more names than that.

    A link that is kept leads inside the context wherever the context is unpacked, and not only here. One
    whose path leaves the context on the way to what it leads to inside it, by the context's own name
    (``../app/payload``, in a context named ``app``) or as an absolute path, would lead out of the context
    under any other name, so it is rewritten in the context that is staged, as the way from its own directory
    to what it leads to. A build that copies the link itself then copies it as rewritten. A link whose path
    stays inside the context is left exactly as it is written.
    """
    # Asked as `os.path` asks: a path the system refuses outright, for a name longer than it allows say, is no
    # directory. A path's own `is_dir` raises for that before Python 3.14, and the image would fail in the
    # system's words on one Python and in these on another.
    if not os.path.isdir(context):
        raise ImageFailed("its build context is not a directory in what was fetched: its context path matches none")
    # The directory itself, should the context be reached by a link: a packager that doesn't follow links would
    # otherwise be handed one, and nothing else.
    context = context.resolve()
    _dockerfile(context, dockerfile)
    links = _check_links(context)
    if into is not None:
        # Nothing has been written so far, and nothing is where the context was fetched: the rest is the copy's.
        context = _copy_directory(context, into).resolve()
    _rewrite(context, links.rewritten)
    root = context
    named = context / dockerfile
    # Where it is staged, and now that its links stay inside: in a copy, the way to it is by the copy's links.
    source = _dockerfile(context, dockerfile)
    target = context / ROOT_DOCKERFILE
    # Whether the spec names the root Dockerfile by its own path, however that path is spelt. Its directory is
    # one the system read, on the way to the Dockerfile.
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
        replaced = target.exists() and not same
    else:
        # A link is always replaced by a copy. That is another Dockerfile than the context's own only when the
        # link is not a way to the named file.
        replaced = not at_root and not (target.exists() and target.samefile(source))
    if replaced:
        logger.warning(
            "the context's own root %s is replaced by %s for this build",
            ROOT_DOCKERFILE,
            _quoted(dockerfile),
        )
    #: Whether the named Dockerfile is put at the root, or brings another ignore file there. Not when it is
    #: the root Dockerfile, there as a file under its own name: then only links among the ignore files change.
    moved = not (same and at_root)
    ignore, staged_ignore = named.with_name(named.name + _IGNORE_SUFFIX), context / (ROOT_DOCKERFILE + _IGNORE_SUFFIX)
    if moved and staged_ignore.is_dir() and not staged_ignore.is_symlink():
        raise ImageFailed(f"the build context has a directory named {staged_ignore.name} at its root")
    # Not the path's own `is_file`: the name is the Dockerfile's with more added, and may be one no file can have.
    has_ignore = _is_file(ignore)
    #: Each ignore file to write at the root, with the file whose content it gets.
    brought: dict[Path, Path] = {}
    if has_ignore and not (
        staged_ignore.is_file() and not staged_ignore.is_symlink() and ignore.samefile(staged_ignore)
    ):
        # The named Dockerfile's own. Unless that is the root one already, there as a file.
        brought[staged_ignore] = ignore
    if (context_ignore := context / _IGNORE_SUFFIX).is_symlink() and context_ignore.is_file():
        # The context's own, which is what applies where the named Dockerfile has none.
        brought[context_ignore] = context_ignore
    #: Whether staging writes a file into the context: the Dockerfile at its root, or an ignore file there.
    adds = bool(brought) or not same
    #: Whether the context is within the limits before anything is written into it, with no link followed.
    #: One with links is: it was measured with them followed, which comes to no less. One with none is as
    #: large as it was fetched, and if that is past a limit, its fetch put none on it: a fileset that is not
    #: an archive has no limit there, and an archive is counted by what it lists, which need not name its
    #: directories. Staging carries no context past a limit it was within, and holds none to a limit that
    #: its fetch did not.
    within = links.entries <= MAX_UNPACKED_ENTRIES and links.size <= MAX_UNPACKED_BYTES
    if adds and within:
        # Before any of it is written, as far as sizes alone tell. Staged, the context holds its own names
        # and files with each copy in place of what was there, and comes to no less than that, whatever its
        # links then lead to. A file that is replaced is taken to go, though it stays if the context holds it
        # under another name too: this may let through what the measure below refuses, never the reverse.
        entries, size = links.entries, links.size
        for place, file in (*brought.items(), *(() if same else ((target, source),))):
            entry, held = _held(place)
            entries, size = entries + 1 - entry, size + file.stat().st_size - held
        if moved and not has_ignore:
            entry, held = _held(staged_ignore)
            entries, size = entries - entry, size - held
        _hold_to_limits(entries, size)
    aside: Path | None = None
    try:
        copies: list[tuple[Path, Path]] = []
        if brought:
            # Set aside before anything at the root is replaced: a link there may lead to what is. As a file
            # is copied, in pieces, and inside the context, so that putting a copy in its place is a rename.
            aside = Path(tempfile.mkdtemp(dir=context, prefix=".staging-"))
            copies = [(Path(shutil.copyfile(file, aside / place.name)), place) for place, file in brought.items()]
        if not same:
            # A link too: a service's packager may not follow one.
            target.unlink(missing_ok=True)
            shutil.copyfile(source, target)
        if moved and not has_ignore:
            staged_ignore.unlink(missing_ok=True)
        for copy, place in copies:
            # In place of whatever is there, a link itself and not what it leads to.
            os.replace(copy, place)
    finally:
        if aside is not None:
            _remove(aside)
    if links.found or (adds and within):
        # As the service is handed it: with what was written, and with what each link leads to now. By the
        # same walk, so a link that the files now at the root leave leading out is refused here as well.
        _measured(context, _Follower(), always=True)
    return context


def _dockerfile(context: Path, dockerfile: str) -> Path:
    """The file that ``dockerfile`` names in ``context``, with the links on the way to it followed. Refused if
    that is not a file inside the context.

    The path is a submitter's, and so is every link on the way to it. Whether it leads to a file is asked of
    the system first, which follows no more links than its own limit and reads no path past its longest. The
    standard library follows a path with no limit, and one made of links that each add names that are not
    there costs it the square of their number, so it is asked only of a path the system read to a file:
    every name of one is there, and following it again here costs no more than the system let it. A path
    the system reads to no file is refused with none of it followed, and so is one it refuses outright, for
    a name longer than it allows say: asked as ``os.path`` asks, that is no file on any Python.

    That is also a Dockerfile behind more links than the system follows, which the standard library would
    still find. Nothing that reads a path through the system could open it, a builder among them. And where
    the links are on the way to its directory, the ignore file written for it, which is looked for beside
    the same path, could not be read either: the Dockerfile would be staged without it.
    """
    named = context / dockerfile
    source: Path | None = None
    if os.path.isfile(named):
        try:
            source = named.resolve()
        except (OSError, RuntimeError):
            # A loop of links, on a Python that raises for one: not expected of a path the system read.
            source = None
    if source is None or context not in source.parents:
        raise ImageFailed(f"dockerfile {dockerfile!r} is not a file inside the build context")
    return source


def _is_file(path: Path) -> bool:
    """Whether ``path`` leads to a file, where a name too long for its directory is the name of no file.

    For a path made by adding to a name a submitter chose, as a Dockerfile's own ignore file is looked for.
    Asked of the system, so the answer is the same on any Python: a path's own ``is_file`` raises for what
    the system refuses outright before Python 3.14, and takes all of it for no file from then on. Neither
    is right here. A name longer than its directory holds is the name of no file: a Dockerfile named that
    close to the limit has no ignore file of its own, and is built with the context's. A path that is too
    long as a whole, though each of its names fits, may still lead to a file, which a builder that reads
    from the context's root would find: taken for missing, the Dockerfile would be staged without its
    ignore file. So that is raised, with whatever else the system refuses, and fails the image.
    """
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            # Nothing is there, or a link that leads nowhere: as a path's own `is_file` has it on any Python.
            return False
        if exc.errno == errno.ENAMETOOLONG:
            # The longest name the directory holds. Less than none where the system sets no limit.
            longest = os.pathconf(path.parent, "PC_NAME_MAX")
            if 0 <= longest < len(os.fsencode(path.name)):
                return False
        raise


def _rewrite(context: Path, rewritten: Mapping[str, str]) -> None:
    """Write each link of ``context`` that :func:`_check_links` said to rewrite, as it said to."""
    for link, written in rewritten.items():
        place = context / link
        place.unlink()
        place.symlink_to(written)
    if rewritten:
        logger.info(
            "%d link(s) of the build context left it on the way to what they lead to inside it, and are rewritten "
            "to stay inside wherever it is unpacked: the first is %s",
            len(rewritten),
            _quoted(next(iter(rewritten))),
        )


def _held(path: Path) -> tuple[int, int]:
    """What the name ``path`` is of a context's own entries and bytes: an entry if anything is there, and its
    size if that is a file. A link holds no bytes of its own."""
    try:
        status = os.lstat(path)
    except OSError:
        return 0, 0
    return 1, status.st_size if stat.S_ISREG(status.st_mode) else 0


def _hold_to_limits(entries: int, size: int) -> None:
    """Refuse a context that comes to ``entries`` entries and ``size`` bytes, if either is more than a source
    may unpack to."""
    if entries > MAX_UNPACKED_ENTRIES:
        raise ImageFailed(
            f"the build context comes to more than {MAX_UNPACKED_ENTRIES} entries with its links followed"
        )
    if size > MAX_UNPACKED_BYTES:
        raise ImageFailed(f"the build context comes to more than {MAX_UNPACKED_BYTES} bytes with its links followed")


@dataclass(slots=True)
class _Links:
    """What one walk of a build context finds of its links."""

    #: The first link that leads out of the context, as its path in the context. The walk ends at it.
    out: str | None = None
    #: From each directory, the ways on: the directories in it, and where each link in it leads, with the link.
    ways: dict[Path, list[tuple[Path, str | None]]] = field(default_factory=dict)
    #: What each directory holds itself, in entries and in bytes, with a link to a file counted as the file.
    holds: dict[Path, tuple[int, int]] = field(default_factory=dict)
    #: Every link, where it is.
    found: list[Path] = field(default_factory=list)
    #: What the context holds with no link followed: its names, and the bytes of its files, the names of one
    #: file counted once.
    entries: int = 0
    size: int = 0
    #: The links to rewrite, each by its path in the context, with what it is to be written as. Empty until
    #: :func:`_check_links` has asked it of each link.
    rewritten: dict[str, str] = field(default_factory=dict)


class _Follower:
    """Where the paths of one build context lead, with every link in them followed.

    One of these serves a whole walk of a context. It remembers where each link it has followed ends, so a
    link is read once, however many paths lead through it. Followed from scratch for every link that leads
    to it, a chain of links costs the square of its length to read, and an archive of a megabyte holds a
    chain of a hundred thousand. It also counts the names it takes, and fails the image past
    :data:`MAX_LINK_STEPS`: what it remembers bounds how often a link is read, not how long the paths are
    that the links are written as. The names are counted as it is given them, before it takes one, so it
    never holds more of them than the limit. And it follows no path past :data:`MAX_LINK_PATH_CHARS`,
    which fails the image too: a name costs the length of the path it is added to, and names that are not
    there, added by one link after another, would make each step cost more than the last.

    A path is read as the system reads one, a name at a time: a step back is from where the path has come
    to, which after a link to a directory is where the link led. Unlike the system, it gives up on no path. A
    name that isn't there is kept in the path as a name. So is a link that is come back to while it is being
    followed, a loop of links: the path goes on from the link's name. That is how ``os.path.realpath`` reads
    a path on Python 3.13 and later, and a path leads here where it leads for that, whatever the Python: what
    is refused for a link is what was refused when each link was followed from scratch.

    Where a link ends by way of such a loop rests on which link of the loop was come to first. So that is
    remembered for the one path only, and worked out again for the next: a path then leads to the same place
    whatever was followed before it, and in whatever order a directory lists its entries. Only the limit on
    steps bounds what links like that cost. No build can follow them.

    What it remembers holds for the context as it is. Nothing is written to a context while one of these is
    asked about it: the links to rewrite are all decided first.
    """

    def __init__(self) -> None:
        #: Where each link that has been followed ends, by the link's own path: those whose end does not rest
        #: on a loop of links. One end for a link, and none longer than :data:`MAX_LINK_PATH_CHARS`.
        self._ends: dict[str, str] = {}
        #: How many names it has been given to take: those of each path it was asked about, and of each
        #: link it read on the way.
        self.steps = 0

    def _names(self, path: str) -> list[str]:
        """The names of ``path``, the first one last, counted before any of them is taken.

        Counted as they are taken, the names of a chain of links would all be held before the limit was
        reached: each link of one is read at the first name of the link before it.
        """
        names = path.split(os.sep)[::-1]
        self.steps += len(names)
        if self.steps > MAX_LINK_STEPS:
            raise ImageFailed(
                f"the build context's links take more than {MAX_LINK_STEPS} steps to follow: there are too "
                "many of them, or the paths they are written as are too long"
            )
        return names

    def end(self, directory: str, path: str) -> str:
        """Where ``path`` leads, read from ``directory``: a directory with no link in its own path."""
        sep = os.sep
        at = sep if path.startswith(sep) else directory
        #: The names still to take, the next one last.
        names = self._names(path)
        #: The links being followed, each with how many of ``names`` are left once its own are taken.
        following: list[tuple[str, int]] = []
        #: For this path alone: None for each link being followed, and where each link ended whose end rests
        #: on a loop of links.
        here: dict[str, str | None] = {}
        #: How many of ``following``, from the first, end where they do by way of a loop.
        unsure = 0
        while True:
            while following and len(names) == following[-1][1]:
                # Every name of the link's own path is taken: where that came to is where the link ends.
                link = following.pop()[0]
                if len(following) >= unsure:
                    del here[link]
                    self._ends[link] = at
                else:
                    here[link] = at
                    unsure = len(following)
            if not names:
                return at
            name = names.pop()
            if not name or name == os.curdir:
                continue
            if name == os.pardir:
                at = at[: at.rindex(sep)] or sep
                continue
            there = at + name if at.endswith(sep) else at + sep + name
            if len(there) > MAX_LINK_PATH_CHARS:
                # Nothing is there to find, and a name that isn't there stays in the path: each one after it
                # would cost the length of the path so far to take, and to remember where a link ended.
                raise ImageFailed(
                    f"the build context's links lead to a path of more than {MAX_LINK_PATH_CHARS} characters, "
                    "where nothing can be: the paths they are written as are too long"
                )
            if there in self._ends:
                at = self._ends[there]
                continue
            if there in here:
                # A link that is being followed, come back to, or one that ended by way of such a loop. Every
                # link that is being followed now ends where it does by way of it too.
                at = here[there] or there
                unsure = len(following)
                continue
            try:
                written = os.readlink(there) if stat.S_ISLNK(os.lstat(there).st_mode) else None
            except OSError:
                # Not there, or not to be read: kept as a name.
                written = None
            if written is None:
                at = there
                continue
            here[there] = None
            following.append((there, len(names)))
            if written.startswith(sep):
                at = sep
            names.extend(self._names(written))


def _check_links(context: Path) -> _Links:
    """Refuse ``context`` for links a packager must not follow, and say which of its links only lead inside it
    here, with what each is to be rewritten as.

    Nothing is written: the context may be another image's too, and the links are rewritten where the context
    is staged (see :func:`stage_context`). ``context`` is a directory with no link in its own path.
    """
    follower = _Follower()
    links = _measured(context, follower)
    for link in links.found:
        if (written := _kept_inside(context, link, follower)) is not None:
            links.rewritten[link.relative_to(context).as_posix()] = written
    return links


def _measured(context: Path, follower: _Follower, *, always: bool = False) -> _Links:
    """What a walk of ``context`` finds of its links, once the context is one a packager may be handed.

    A link that leads out of the context would bring in what is beside it. One by which a walk that follows
    links comes back to a directory it is in, straight away or by way of other links, would be followed round
    and round. Links to directories that never come back round are each followed, and what they lead to sent
    again each time: a context that comes to more that way than a source may unpack to is refused. A context
    with no links is as large as it was fetched, which is the fetch's to bound: it is measured only with
    ``always``, once staging has written into one that was within the limits.
    """
    links = _links(context, follower)
    if (refused := links.out or _link_round(context, links.ways)) is not None:
        raise _leads_out(refused)
    if links.found or always:
        _hold_to_limits(*_followed(context, links))
    return links


def _leads_out(link: str) -> ImageFailed:
    """Why a context with ``link``, a link that a packager must not follow, isn't built."""
    return ImageFailed(f"the build context has a link that leads out of it, or back to a directory it is in: {link!r}")


def _links(context: Path, follower: _Follower) -> _Links:
    """What a walk of ``context`` finds of its links. One pass over the context, which follows no link itself:
    where each link leads is asked of ``follower``. ``context`` is a directory with no link in its own path.

    It ends at the first link that leads out of the context, which is all that is then asked of it.
    """
    root = str(context)
    links = _Links()
    #: The files with more than one name whose bytes are counted: such names are one file, and unpacking
    #: counts it once. Counted for each name, a source that was let through could be refused here.
    counted: set[tuple[int, int]] = set()
    for directory, directories, files in os.walk(context):
        # In the order of their names, which is then the order the walk takes them in. A file with more
        # than one name is counted in the first directory the walk comes to that holds one, and a walk
        # that follows links may come to one such directory more often than to another. A directory lists
        # its entries in no promised order: by that, one context came to more on one disk than on another.
        directories.sort()
        here = Path(directory)
        on = links.ways.setdefault(here, [])
        #: What the directory holds with each link to a file counted as the file, and of that, its own files.
        entries = size = own = 0
        for name, listed_as_directory in (*((name, True) for name in directories), *((name, False) for name in files)):
            path = here / name
            entries += 1
            try:
                status = os.lstat(path)
            except OSError:
                # Gone, or not to be read: no link that can be followed.
                status = None
            if status is None or not stat.S_ISLNK(status.st_mode):
                if listed_as_directory:
                    on.append((path, None))
                elif status is not None and stat.S_ISREG(status.st_mode):
                    number = (status.st_dev, status.st_ino)
                    if number not in counted:
                        own += status.st_size
                    if status.st_nlink > 1:
                        counted.add(number)
                continue
            links.found.append(path)
            end = follower.end(directory, name)
            where = path.relative_to(context).as_posix()
            if not _inside(root, end):
                links.out = where
                return links
            try:
                led_to = os.stat(end)
            except OSError:
                # It leads to nothing, or round to itself.
                continue
            if stat.S_ISDIR(led_to.st_mode):
                on.append((Path(end), where))
            elif stat.S_ISREG(led_to.st_mode):
                # Followed, the link is another copy of the file.
                size += led_to.st_size
        links.holds[here] = (entries, size + own)
        links.entries, links.size = links.entries + entries, links.size + own
    return links


def _followed(root: Path, links: _Links) -> tuple[int, int]:
    """How many entries, and how many bytes, a walk from ``root`` that follows links comes to.

    As a packager that follows links would send the context: what a link leads to is counted where the link
    is, each time the walk comes to it. Each directory's total is worked out once and remembered, so this
    costs no more than the context's own size, however many times over its links lead to one directory. Its
    place is kept in a list and not in calls: a context may be nested deeper than a function may call itself.
    Neither total is counted past one more than its limit, since each level of a context may double them.

    No way in ``links`` comes back round: :func:`_link_round` is asked first.
    """
    most = (MAX_UNPACKED_ENTRIES + 1, MAX_UNPACKED_BYTES + 1)
    totals: dict[Path, tuple[int, int]] = {}
    trail: list[tuple[Path, Iterator[tuple[Path, str | None]]]] = [(root, iter(links.ways.get(root, ())))]
    while trail:
        here, on = trail[-1]
        # The next directory on from here that is not yet measured, before this one can be.
        there = next((there for there, _ in on if there not in totals), None)
        if there is not None:
            trail.append((there, iter(links.ways.get(there, ()))))
            continue
        trail.pop()
        entries, size = links.holds.get(here, (0, 0))
        for there, _ in links.ways.get(here, ()):
            entries, size = min(entries + totals[there][0], most[0]), min(size + totals[there][1], most[1])
        totals[here] = (entries, size)
    return totals[root]


def _kept_inside(root: Path, link: Path, follower: _Follower) -> str | None:
    """What ``link`` is to be rewritten as, if its path leaves ``root`` on the way to what it leads to inside
    it. None if its path stays inside, and it is left as it is written.

    Such a link leads inside the context only while the context is where it is, under the name it has. It
    becomes the way from its own directory to what its path names, with the links on the way there followed
    and that one not, should it be a link too: it still names the link it named. Where what it names is
    beside the context, a link there that leads back in, it becomes the way to where that leads. A path
    that ends in a separator leads only to a directory, and is rewritten to end in one.

    One that leads nowhere is rewritten too, to where its path comes to. If that is out of the context, a
    path through a loop of links that climbs out after it, the context is refused: nothing says where such
    a link leads once the context is unpacked.
    """
    context, directory, written = str(root), str(link.parent), os.readlink(link)
    if _stays_inside(context, directory, written, follower):
        return None
    named = PurePosixPath(written)
    if named.name in ("", ".."):
        to = follower.end(directory, written)
    else:
        to = os.path.join(follower.end(directory, str(named.parent)), named.name)
    if not _inside(context, to):
        to = follower.end(directory, link.name)
    if not _inside(context, to):
        raise _leads_out(link.relative_to(root).as_posix())
    rewritten = os.path.relpath(to, directory)
    # A path that ends in a separator, or in `.` after one, leads to a directory or to nothing. Written
    # without it, a link that led nowhere, for ending at a file, would lead to that file.
    return rewritten + os.sep if written.rpartition(os.sep)[2] in ("", os.curdir) else rewritten


def _stays_inside(root: str, directory: str, written: str, follower: _Follower) -> bool:
    """Whether a link in ``directory`` with the path ``written`` is followed to its end without leaving ``root``.

    Read as the system reads it, a name at a time and with each link on the way followed, not as it is
    spelt: after a link to a directory, a step back is from where the link led, and may be out of the context
    though the path never seems to climb that far.
    """
    if os.path.isabs(written):
        return False
    here = directory
    for name in PurePosixPath(written).parts:
        if name == "..":
            if here == root:
                return False
            here = os.path.dirname(here)
            continue
        here = follower.end(here, name)
        if not _inside(root, here):
            return False
    return True


def _inside(root: str, path: str) -> bool:
    """Whether ``path`` is ``root`` or in it. Both as they are with their links followed."""
    return path == root or path.startswith(root if root.endswith(os.sep) else root + os.sep)


def _link_round(root: Path, ways: Mapping[Path, list[tuple[Path, str | None]]]) -> str | None:
    """The link by which a walk from ``root`` that follows links comes back to a directory it is in. None if
    it never does.

    Each directory is walked once, so a context with many links to one directory costs no more than its size.
    """
    walked: set[Path] = set()
    walking = {root}
    trail: list[tuple[Path, str | None, Iterator[tuple[Path, str | None]]]] = [(root, None, iter(ways.get(root, ())))]
    while trail:
        here, _, on = trail[-1]
        way = next(on, None)
        if way is None:
            trail.pop()
            walking.discard(here)
            walked.add(here)
            continue
        there, link = way
        if there in walking:
            # Back where it is. By this link, or by the last one taken to get here: a way round always has one.
            return link or next((taken for _, taken, _ in reversed(trail) if taken), None)
        if there not in walked:
            walking.add(there)
            trail.append((there, link, iter(ways.get(there, ()))))
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
    cancelled. It is given up after a wait as well, if the wait ended at the deadline or past it: a sleep ends
    no sooner than it was asked to, and on a busy node it ends later.
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
        logger.warning("%s failed (%s); trying again in %.0fs", what, _quoted(str(failure)), wait)
        clock.sleep(wait)
        # The wait fitted when it began, and may have ended late. A stop is said first, as before any try.
        if not at_the_deadline and clock.monotonic() >= deadline and not cancelled():
            raise ImageFailed(
                f"{what}: {failure}; the build timeout passed before it could be tried again", maybe_called=lost
            ) from failure


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
            logger.info("asked the build service to cancel %s", _quoted(build_id))
            return True
        logger.warning(
            "could not cancel %s after %d tries: %s",
            _quoted(build_id),
            tries,
            _quoted(str(failure)),
        )
        return False


def _ending(build_id: str, *, uncancelled: bool = False) -> str:
    """What follows an image's reason: the service's name for its build, and whether that build may live on."""
    if uncancelled:
        return f" [build {build_id}; the build service could not be told to stop it, it may still be running]"
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
            logger.info("build %s: %s", _quoted(build_id), _quoted(seen))
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
        redact(str(key))[:MAX_EXTERNAL_KEY]: redact(str(value))[:MAX_EXTERNAL_VALUE]
        for key, value in list(built.attributes.items())[:MAX_EXTERNAL_ATTRIBUTES]
    }
    try:
        return ExternalImage(
            image_id=redact(built.image_id) if built.image_id else None,
            build_id=redact(build_id) if build_id else None,
            attributes=attributes,
        )
    except ValidationError as exc:
        raise ImageFailed(f"the build service's handle for the image can't be recorded ({_problems(exc)})") from exc


def _problems(exc: ValidationError) -> str:
    """What is wrong with each field, on one line, without the values that were refused."""
    return "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors())


def _starts_as(built: RemoteImage, image: str, redact: Callable[[str], str]) -> ImageRuntimeConfig | None:
    """How ``image`` starts by default, as the service's client reports it, for the row. None if it reports none.

    Every reader of the image reads this too, so it must not hold the credential. But it is not a text to take
    the credential out of, as the handle is: it is what a consumer runs, and with a mark where an argument
    was it is no longer how the image starts. So one that holds the credential in any part is left out whole,
    and the log says so. One that can't be recorded is logged and left out too, as the push step leaves out
    an image config it can't read. Either way the image is built and published all the same.
    """
    if built.runtime_config is None:
        return None
    try:
        reported = ImageRuntimeConfig.model_validate(built.runtime_config).model_dump()
        recorded = ImageRuntimeConfig.model_validate(reported)
    except ValidationError as exc:
        logger.warning("can't record how %s starts; recording none: %s", _quoted(image), _quoted(_problems(exc)))
        return None
    if _redact_value(reported, redact) is not reported:
        # Which part, and what it held, is not said: the row and the log are read by more than the step.
        logger.warning(
            "not recording how %s starts: it holds the build service's credential, and with that taken out it "
            "would not be how the image starts",
            _quoted(image),
        )
        return None
    return recorded


def _complete(
    builder: _Builder,
    image: str,
    digest: str,
    runtime_config: ImageRuntimeConfig | None,
    external: ExternalImage,
    *,
    clock: Clock,
    cancelled: Callable[[], bool],
) -> ContainerImage:
    """Complete the row, trying for :data:`COMPLETION_SECONDS` while the builder can't be reached.

    The image is built and published, and only this call is left; completing is safe to repeat. The first try
    is made even when the job has been told to stop. No other is made once it has: a pod told to stop has
    seconds, and one try may take them all.
    """
    deadline = clock.monotonic() + COMPLETION_SECONDS
    waits = 0
    while True:
        try:
            return builder.complete(image, digest, runtime_config, external)
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
        logger.warning("completing %s failed (%s); trying again in %.0fs", image, _quoted(failure), wait)
        clock.sleep(wait)
        if cancelled():
            # The stop cut the wait short. As before the wait, no other try is made.
            raise ImageFailed(f"it was built, but the builder could not be told: {failure}") from failure


def _build_one(
    image: RemoteBuildImage,
    *,
    row: ContainerImage,
    config: RemoteBuildStepConfig,
    service: RemoteBuildService,
    builder: _Builder,
    fetch: Callable[[RemoteBuildImage, Path], tuple[Path, Path | None]],
    attempt: JobAttempt,
    clock: Clock,
    cancelled: Callable[[], bool],
    scratch: Path,
    redact: Callable[[str], str],
) -> None:
    """Build ``image``. ``fetch`` says where its build context is, and where in ``scratch``, which is this
    image's alone, a copy of the context is to be staged: nowhere, if the context is in ``scratch`` already."""
    try:
        fetched, copy = fetch(image, scratch)
        context = stage_context(fetched, image.dockerfile, into=copy)
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
    logger.info("%s is build %s", _quoted(image.image), _quoted(build_id))
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
        runtime_config = _starts_as(built, image.image, redact)
        done = _complete(builder, image.image, digest, runtime_config, external, clock=clock, cancelled=cancelled)
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
    logger.info("completed %s@%s", _quoted(destination), _quoted(digest))


def _quoted(text: object) -> str:
    """``text`` for a log line that is not an image's failure line: on one line, and without the marker.

    Settling reads a marked line as an image's reason, wherever in a line the marker is. What a service says of
    one build, a stage or an error, may hold anything, the marker with another image's name included.
    """
    return sanitize_for_log(text).replace(FAILURE_MARKER, UNMARKED)


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
    """Log the one line that says why ``image`` wasn't built.

    The marker is at its start and nowhere else in it. Jobs keeps the end of the log, cut by length: a line cut
    before a marker that its reason quotes, a path a submitter chose say, would be read from that marker on.
    """
    said = _line(reason, ending, room).replace(FAILURE_MARKER, UNMARKED)
    logger.error("%s %s: %s", FAILURE_MARKER, _quoted(image), said)


def _in_a_url(secret: str) -> str:
    """A pattern for ``secret`` in a URL, whichever of its characters the encoder percent-encoded.

    Letters, digits and ``-._`` are matched as written: no encoder writes those any other way.
    """
    pieces: list[str] = []
    for character in secret:
        literal = re.escape(character)
        if character.isascii() and (character.isalnum() or character in "-._"):
            pieces.append(literal)
            continue
        encoded = "".join(f"%{byte:02X}" for byte in character.encode())
        choices = [literal, f"(?i:{re.escape(encoded)})"]
        if character == " ":
            choices.append(re.escape("+"))
        pieces.append(f"(?:{'|'.join(choices)})")
    return "".join(pieces)


def _in_quotes(text: str) -> str:
    """``text`` as ``repr`` writes it inside a string that also holds a double quote: with its apostrophes
    escaped, which ``repr`` of the text alone leaves as they are."""
    return repr(text + '"')[1:-2]


def _redactor(secret: str | None) -> Callable[[str], str]:
    """Takes ``secret`` out of a text, in each form the driver knows a text to hold it.

    As it is. As a log line holds it, without its line breaks, and as a one-line reason does, with a space for
    each: the two differ for a secret of more than one line, such as a key file. As an error that quotes it
    holds it: escaped, as Python's ``repr`` and ``json.dumps`` write it, or as any URL encoder does. And for a
    secret of several lines, each line long enough to be worth taking out on its own. Another JSON encoder's
    escapes are not known: a client parses a service's reply before it puts any of it in an error.
    """
    if not secret:
        return lambda text: text
    shown = repr(secret)[1:-1]
    escaped = {
        shown,
        # `repr` leaves an apostrophe as it is unless the text around it also holds a double quote.
        _in_quotes(secret),
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
        secret.encode("unicode_escape").decode("ascii"),
    }
    # Quoted once more, as an error that holds another error's text is: `repr` of a tuple of messages, say.
    twice = {repr(form)[1:-1] for form in escaped} | {json.dumps(form)[1:-1] for form in escaped}
    twice |= {_in_quotes(form) for form in escaped}
    forms = {secret, sanitize_for_log(secret), _one_line(secret), quote(secret, safe=""), quote_plus(secret)}
    forms |= escaped | twice
    lines = [line.strip() for line in secret.splitlines()]
    parts = {line for line in lines if len(line) >= _MIN_PART_CHARS} - forms if len(lines) > 1 else set()

    def longest_first(texts: set[str]) -> list[str]:
        # So that one form is never left as the remains of another.
        return sorted(texts - {""}, key=len, reverse=True)

    def ways(*, with_the_mark: bool) -> list[str]:
        # The whole secret before any one line of it: as it may be written, then as any URL holds it, since
        # encoders differ in which characters they leave alone. Tried the other way round, a line would be
        # taken out of the middle of the whole, and the rest left.
        written = [re.escape(form) for form in longest_first(forms) if (REDACTED in form) is with_the_mark]
        in_a_url = [_in_a_url(secret)] if (REDACTED in secret) is with_the_mark else []
        one_line = [re.escape(part) for part in longest_first(parts) if (REDACTED in part) is with_the_mark]
        return [*written, *in_a_url, *one_line]

    # A mark that is already in the text is found too, and its own text takes its place. So it is read past
    # whole, and what it stands for is not read again: a short credential may be part of the mark itself.
    # Before it come the ways of writing a secret that themselves hold the mark's text. Those are found only
    # with the marks in them: read between the marks alone, such a secret was passed on whole.
    tried = [*ways(with_the_mark=True), re.escape(REDACTED), *ways(with_the_mark=False)]
    found = re.compile(f"(?:{'|'.join(tried)}){_NOT_IN_A_MARK}")

    def redact(text: str) -> str:
        return found.sub(REDACTED, text)

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
    #: Each download that more than one image of the set names, by what names it, and where it is kept: read by
    #: `context_for`.
    fetched: dict[str, Path] = {}
    #: The set's images, each with its context path spelt one way: what names a download is then the same for
    #: every image that builds from one directory, however its spec wrote the path.
    images = [image.model_copy(update={"source": _spelt_one_way(image.source)}) for image in config.images]
    #: The filesets some image of the set builds from whole: that download covers their directories too.
    whole_filesets = {
        image.source.fileset for image in images if image.source.archive is None and image.source.context_path is None
    }

    def downloaded(source: ContextSource) -> tuple[ContextSource, bool]:
        """What is downloaded for ``source``, and whether that is more than the source's own directory."""
        whole = source.archive is not None or source.fileset in whole_filesets
        return (source.model_copy(update={"context_path": None}) if whole else source), whole

    #: Each image's download, by what names it, in the set's order.
    keys = [downloaded(image.source)[0].model_dump_json() for image in images]
    #: How many of the set's images each download serves, and the last image that needs it.
    shared = Counter(keys)
    last = {key: index for index, key in enumerate(keys)}
    #: The directory each shared download was fetched into, removed once its last image has had its turn.
    sources: dict[str, Path] = {}
    #: A shared archive that was refused, with why: the same archive is refused the same way.
    refused: dict[str, ArchiveRefused] = {}

    def context_for(image: RemoteBuildImage, directory: Path) -> tuple[Path, Path | None]:
        """Where ``image``'s build context is, and where in ``directory`` its own copy of it goes, if it needs one.

        The images of one set are built one after another, and a source may change in between. So what a set
        downloads it downloads once, as the in-cluster build's fetch step does: an archive whole, whichever
        directory of it each image builds from, and a fileset whole if any image builds from all of it. Each
        image gets its own copy of its directory, so every image built from one download sees the same files. A
        download only one image names has nothing to share, and is spared the copy.

        The copy is staging's to make, and not made here: an image's links are checked where the download
        is, with the rest of it beside them, as they are for an image built alone. The download itself is
        never written to.
        """
        source = image.source
        download, whole = downloaded(source)
        key = download.model_dump_json()
        if shared[key] == 1:
            # Nothing to share it with: fetched where it is built, and removed with the image's directory.
            return fetch(image, directory), None
        if key in refused:
            # Not fetched and unpacked again for each image that names it.
            raise refused[key]
        if key not in fetched:
            into = Path(tempfile.mkdtemp(dir=scratch, prefix="source-"))
            try:
                fetched[key] = fetch(image.model_copy(update={"source": download}), into)
            except BaseException as exc:
                # Not kept: the next image that names it tries again, in a directory of its own. Unless it is
                # the archive that was refused, which a second try would be too.
                if isinstance(exc, ArchiveRefused):
                    refused[key] = exc
                _remove(into)
                raise
            sources[key] = into
        context = fetched[key]
        root = context.resolve()
        if whole and source.context_path:
            inside = PurePosixPath(source.context_path)
            if inside.is_absolute() or ".." in inside.parts:
                raise ImageFailed(f"context path {source.context_path!r} is not a directory inside its source")
            context = context.joinpath(*inside.parts)
            if not os.path.isdir(context):
                raise ImageFailed(f"context path {source.context_path!r} is not a directory in its source")
            if not _within(context, fetched[key]):
                raise ImageFailed(f"context path {source.context_path!r} is not a directory inside its source")
            # The directory that was checked, not a link to it: as an image built alone is staged.
            context = context.resolve()
        # The copy goes where the directory was in the download, under the download's own name. No link rests
        # on those names: one that came back to the context by way of them is rewritten to stay inside, as is
        # decided where the download is.
        inside = context.relative_to(root) if root in context.parents else Path()
        return context, directory / (root.name or "context") / inside

    def report(image: str, reason: str, ending: str = "") -> None:
        reason, ending = redact(_one_line(redact(reason))), redact(_one_line(redact(ending)))
        _report(image, reason, ending)
        failed.append((image, reason, ending))

    for index, image in enumerate(images):
        # A download the set shared, once the last image that names it has had its turn: a set of large
        # contexts then holds no more of them than it still needs.
        for key in [key for key in sources if last[key] < index]:
            fetched.pop(key, None)
            _remove(sources.pop(key))
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
            logger.info("%s is already %s; nothing to build", _quoted(image.image), row.status)
            failures += row.status != "ready"
            continue
        try:
            directory = tempfile.mkdtemp(dir=scratch, prefix="image-")
        except OSError as exc:
            report(image.image, f"cannot make a directory for its build context: {exc}")
            failures += 1
            continue
        try:
            # The failure is reported here, before the image's context is removed: a large one takes a while
            # to delete, and a step that was told to stop has seconds left.
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
                    _quoted(image.image),
                    "".join(traceback.format_tb(exc.__traceback__)),
                )
                report(image.image, f"unexpected {type(exc).__name__}: {exc}")
                failures += 1
        finally:
            # What can't be removed now, a file a client still holds open say, goes with `scratch` when the step
            # ends. It must not cost the next image its turn.
            _remove(directory)
    logger.info("%d image(s), %d failure(s)", len(config.images), failures)
    if failed and len(config.images) > 1:
        # Jobs keeps only the end of a failed step's log, and whatever a later image logged came after an earlier
        # image's line. So each line again, last, its reason cut to share what is kept. A line with too little
        # room left for its reason is not logged again: settling reads the last line for an image, and the first
        # one, though it may have fallen outside what is kept, says more than one cut to nothing.
        for image, reason, ending in failed:
            room = _room(len(failed), image)
            if room - len(ending) >= MIN_REASON_ROOM or (ending and room > len(ending)):
                _report(image, reason, ending, room=room)
    return 1 if failures else 0


def _spelt_one_way(source: ContextSource) -> ContextSource:
    """``source``, with its context path written the one way that has nothing to spare.

    ``app``, ``app/`` and ``./app`` name one directory, and so do ``app/x`` and ``app//x``. A set downloads a
    directory once for each way of naming it, so two spellings of one would be two downloads, of a source that
    may change between them. A path that names no directory, or climbs with ``..``, is not made a better one:
    whatever fetches it takes or refuses it as it would have.
    """
    if not source.context_path:
        return source
    path = PurePosixPath(source.context_path)
    if not path.parts or path.as_posix() == source.context_path:
        return source
    return source.model_copy(update={"context_path": path.as_posix()})


def _copy_directory(source: Path, destination: Path) -> Path:
    """Copy the directory ``source`` to ``destination``, and return ``destination``.

    A link is copied as a link, never followed. Names that are one file in ``source``, hard links of one
    another, are one file in the copy too. Unpacking counts such a file once, however many names an archive
    gives it, so a copy with a file for each name could be many times the size an archive is allowed to unpack
    to. The copy shares no file with ``source``: staging changes what is in it.
    """
    #: The first copy of each file of `source` that has more than one name, by the file's device and number.
    copied: dict[tuple[int, int], str] = {}

    def copy(file: str, to: str) -> None:
        status = os.stat(file)
        number = (status.st_dev, status.st_ino)
        if number in copied:
            os.link(copied[number], to)
            return
        shutil.copy2(file, to)
        if status.st_nlink > 1:
            copied[number] = to

    return Path(shutil.copytree(source, destination, symlinks=True, copy_function=copy))


def _within(context: Path, source: Path) -> bool:
    """Whether ``context``, with every link in its path followed, is ``source`` or a directory inside it.

    A context path is checked as it is written, relative and with no ``..``, when a build is submitted. Where
    it leads once links are followed rests on how the source was unpacked. This is the driver's own check of
    that: the step holds the build service's credential, and hands the service what it takes for a context.

    It is asked only of a ``context`` that the system reads to a directory, and of a ``source`` the step made.
    The standard library follows a path with no limit, and one made of links that each add names that are
    not there costs it the square of their number. The system reads no such path to a directory: every name
    of one it does is there, and it has no more links than the system follows, so following it again here
    costs no more than the system let it.
    """
    try:
        real, root = context.resolve(), source.resolve()
    except (OSError, RuntimeError):
        # A loop of links, on a Python that raises for one: it leads to no directory, and is not asked about.
        return False
    return real == root or root in real.parents


def _os_error(exc: OSError) -> str:
    """What the operating system says of ``exc``, and the file it was about: none of the error's own text."""
    said = os.strerror(exc.errno) if exc.errno is not None else type(exc).__name__
    return f"{said}: {exc.filename}" if exc.filename else said


def _remove(directory: str | Path) -> None:
    """Remove ``directory`` and what is in it, as far as that can be done. It never raises.

    ``ignore_errors`` covers what the file system refuses. Before Python 3.13 a tree nested deeper than the
    interpreter's recursion limit can't be walked at all, and that is not one of those errors.
    """
    try:
        shutil.rmtree(directory, ignore_errors=True)
    except RecursionError:
        # What is left goes with the step's storage.
        pass


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


def _redact_record(
    record: logging.LogRecord, redact: Callable[[str], str], *, keep_exception: bool = False
) -> logging.LogRecord:
    """``record``, with the credential out of everything a handler prints of it.

    And out of what its message is made from, which stays on the record for a handler that reads it apart from
    the message: where that holds the credential, the record keeps the message as it is formatted, as text,
    and not the arguments.

    Its exception is passed on as text, formatted as the standard formatter writes one and redacted, and not
    as itself, whether or not the text held the credential. A handler may print more of an exception than
    its text: one that shows the variables of each frame prints a client's variable that holds the
    credential, though the error says nothing of it. ``keep_exception`` is for a redaction that has no
    secret to take out: the exception is then left on the record.
    """
    try:
        message, formatted = record.getMessage(), True
    except Exception:
        # A log call whose arguments don't fit its message. A handler would report that by printing both as
        # they are, so they are put together here, where the credential can be taken out of them.
        message, formatted = f"{record.msg} (logged with {record.args!r})", False
    if (redacted := redact(message)) != message or not formatted or _holds_unprinted(record, redact):
        record.msg, record.args = redacted, None
    # A formatter prints the exception and the stack after the message, and an exception's own text with it.
    # It prints `exc_text` as it finds it, so the redacted text is put there first.
    if record.exc_info and not record.exc_text:
        record.exc_text = logging.Formatter().formatException(record.exc_info)
    text = record.exc_text
    if text:
        record.exc_text = redact(text)
    if not keep_exception or record.exc_text != text:
        # The exception itself holds more than its text, what was taken out of the text included: the frames
        # it passed through, with their variables. Some handlers print those themselves.
        record.exc_info = None
    if record.stack_info:
        record.stack_info = redact(record.stack_info)
    return record


def _unmark_record(record: logging.LogRecord) -> logging.LogRecord:
    """``record``, which the driver did not log, without the failure marker in its message, exception and stack.

    Its ``extra`` fields are left as they are: the step's own log format prints none of them.

    A failure line is the driver's to write. Anything else that logs the marker, a library given a path a
    submitter chose, or an exception that quotes a line of a Dockerfile, would have settling read its line as
    an image's reason. Called after :func:`_redact_record`, which has put the exception's text on the record.
    """
    if FAILURE_MARKER in (message := record.getMessage()):
        record.msg, record.args = message.replace(FAILURE_MARKER, UNMARKED), None
    if record.exc_text and FAILURE_MARKER in record.exc_text:
        # As for a secret: the exception itself still holds it, and some handlers print it themselves.
        record.exc_text, record.exc_info = record.exc_text.replace(FAILURE_MARKER, UNMARKED), None
    if record.stack_info and FAILURE_MARKER in record.stack_info:
        record.stack_info = record.stack_info.replace(FAILURE_MARKER, UNMARKED)
    return record


def _is_container(value: Any) -> bool:
    """Whether ``value`` is a mapping, a sequence or a set of any kind, other than text."""
    return isinstance(value, Mapping) or (isinstance(value, Sequence | Set) and not isinstance(value, _NOT_CONTAINERS))


def _prints_all(value: Any) -> bool:
    """Whether ``value`` prints everything it holds: every container in it, however deep, is a plain one.

    Without recursion: it is asked of a value that is already too deep to look into.
    """
    left = [value]
    seen: set[int] = set()
    while left:
        item = left.pop()
        if type(item) in _PLAIN_CONTAINERS:
            if id(item) not in seen:
                seen.add(id(item))
                left.extend(item)
                if type(item) is dict:
                    left.extend(item.values())
        elif _is_container(item):
            return False
    return True


def _redact_container(value: Any, redact: Callable[[str], str], depth: int) -> Any:
    """A mapping, a sequence or a set, with the credential out of everything in it.

    Itself, if nothing in it holds the credential. Otherwise a plain container of what it held, each thing
    redacted on its own: a mapping becomes a ``dict``, a tuple a ``tuple``, a set a ``set`` or a ``frozenset``,
    and any other sequence a ``list``.
    """
    mapping = isinstance(value, Mapping)
    # A mapping's keys and values, turn about.
    items = [part for pair in value.items() for part in pair] if mapping else list(value)
    rebuilt = [_redact_value(item, redact, depth + 1) for item in items]
    if all(new is old for new, old in zip(rebuilt, items, strict=True)):
        return value
    if mapping:
        return dict(zip(rebuilt[::2], rebuilt[1::2], strict=True))
    if isinstance(value, tuple):
        return tuple(rebuilt)
    if isinstance(value, Set):
        try:
            return frozenset(rebuilt) if isinstance(value, frozenset) else set(rebuilt)
        except TypeError:
            # One of its members, rebuilt, is no longer something a set can hold.
            return rebuilt
    return rebuilt


def _redact_value(value: Any, redact: Callable[[str], str], depth: int = 0) -> Any:
    """``value`` as a log call attached it, with the credential out of it.

    A value that doesn't hold the credential is returned as it is, whatever it is. One that does is rebuilt, as
    a plain container, if it is a mapping, a sequence or a set, and otherwise replaced by what it prints as,
    redacted. A container is looked into, not read as it prints, whatever kind it is: a request's headers
    print a mask where they hold the credential, and give it whole to a format that asks for one of them.
    """
    if isinstance(value, str):
        # As `str` compares, not as the value's own type may: a string of another type, a member of an
        # enumeration say, is kept as it is unless the credential was in it.
        if str.__ne__(redacted := redact(value), value):
            return redacted
        if type(value) is str:
            return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if _is_container(value):
        if depth < _MAX_FIELD_DEPTH:
            # Each string in it on its own, not the container as it prints: a string is quoted differently there.
            return _redact_container(value, redact, depth)
        if not _prints_all(value):
            # Too deep to look into, and it may hold what it doesn't print: passed on as its text, not itself.
            return redact(repr(value))
    # As a formatter may print it either way: `%(field)s` prints what `str` gives, which can say more than `repr`.
    for shown in (repr(value), str(value)):
        if (redacted := redact(shown)) != shown:
            return redacted
    return value


def _holds_unprinted(record: logging.LogRecord, redact: Callable[[str], str]) -> bool:
    """Whether what ``record``'s message is made from holds the credential, though the message doesn't print it.

    A message prints its arguments as they print: a request's headers as a mask, a mapping as the one value its
    format names. So it does a message that was logged as an object, and not as text. What can't be looked into
    is taken to hold the credential: the message, which was checked, says all that it printed.
    """
    made_from = (record.msg, record.args)
    try:
        return _redact_value(made_from, redact) is not made_from
    except Exception:
        return True


#: The fields of a log record that :func:`_redact_record` takes the credential out of: its message, as it is
#: formatted from the two, its exception and its stack.
_RECORD_TEXT = frozenset({"msg", "args", "exc_info", "exc_text", "stack_info"})


class _RedactedFields(dict[str, Any]):
    """A log record's fields, with the credential taken out of each one set by key.

    ``Logger.makeRecord`` attaches a log call's ``extra`` fields to the record after the record factory has
    returned it, one at a time, by key in the record's ``__dict__``. A formatter that prints such a field, a
    request's headers say, would print the credential in it. So a record's fields are kept in one of these.
    """

    def __init__(self, fields: Mapping[str, Any], redact: Redaction) -> None:
        super().__init__(fields)
        self.redact = redact

    def check(self) -> None:
        """Take the credential out of the fields that are here already, however they came to be.

        A record factory under the step's, a tracing library's, puts its own fields on a record before the
        step's factory is handed it. It sets them as attributes, which reach these fields by none of the ways
        below, whether or not the record's fields were one of these by then.
        """
        for key, value in list(self.items()):
            if key not in _RECORD_TEXT:
                self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "exc_info" and value:
            self._set_exception(value)
            return
        try:
            value = _redact_value(value, self.redact)
        except Exception:
            # Never into the code that logged. What can't be checked is withheld, not kept as it is.
            value = WITHHELD
        super().__setitem__(key, value)

    def _set_exception(self, exc_info: Any) -> None:
        """An exception set on a record after it was made, as ``logging.makeLogRecord`` sets one.

        As :func:`_redact_record` treats one a log call attached. Once there is a secret to take out, its
        text is kept, the exceptions it was raised from included, without the secret, and the exception is
        not, whether or not the text held the secret: a formatter prints the text it finds, and a handler
        that is handed the exception may print what its frames held. With no secret, it is kept as it is.
        """
        try:
            text = logging.Formatter().formatException(exc_info)
            redacted = self.redact(text)
        except Exception:
            text, redacted = "", WITHHELD
        if redacted == text and not self.redact.holds_a_secret:
            super().__setitem__("exc_info", exc_info)
            return
        super().__setitem__("exc_info", None)
        if not self.get("exc_text"):
            super().__setitem__("exc_text", redacted)

    def update(self, *args: Any, **kwargs: Any) -> None:
        # `logging.makeLogRecord` sets a record's fields this way, its message among them. The exception
        # last: its text is kept only where the mapping brought none of its own.
        fields = dict(*args, **kwargs)
        for key, value in fields.items():
            if key != "exc_info":
                self[key] = value
        if "exc_info" in fields:
            self["exc_info"] = fields["exc_info"]

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __reduce__(self) -> tuple[type[dict], tuple[dict[str, Any]]]:
        # A handler may send a record to another process as it is. What takes secrets out stays in this one.
        return dict, (dict(self),)


#: What stands in for a logged value, or a whole record, that could not be checked for a secret.
WITHHELD = "<withheld: it could not be checked for the build service's credential>"


class Redaction:
    """Takes every secret it has been given out of a text."""

    def __init__(self) -> None:
        self._redactors: list[Callable[[str], str]] = []

    def add(self, secret: str) -> None:
        if secret:
            self._redactors.append(_redactor(secret))

    @property
    def holds_a_secret(self) -> bool:
        """Whether it has been given a secret to take out. Until then, it changes nothing."""
        return bool(self._redactors)

    def __call__(self, text: str) -> str:
        for redact in self._redactors:
            text = redact(text)
        return text


#: The one redaction of this process: every secret given to :func:`redact_log`, whenever it was given, and
#: whatever log record factory was installed in between.
_REDACTION = Redaction()


class _RedactingFactory:
    """A log record factory that makes records as ``make`` does, then takes the process's secrets out of them."""

    def __init__(self, make: Callable[..., logging.LogRecord]) -> None:
        self._make = make

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._make(*args, **kwargs)
        try:
            # Once there is a secret, no record is passed on with its exception: only with the exception's text.
            _redact_record(record, _REDACTION, keep_exception=not _REDACTION.holds_a_secret)
            if record.name != logger.name:
                _unmark_record(record)
            if record.exc_info:
                # There is no secret to take out, and the exception's text holds no marker. That text, made
                # here to be checked, is not kept: a handler's own formatter writes it as it would have.
                record.exc_text = None
        except Exception:
            # Never into the code that logged. A record that can't be checked is withheld, not printed as it is.
            record.msg, record.args = WITHHELD, None
            record.exc_info = record.exc_text = record.stack_info = None
        fields = record.__dict__
        if not isinstance(fields, _RedactedFields):
            record.__dict__ = fields = _RedactedFields(fields, _REDACTION)
        # What the factories under this one put on the record. The call's own fields come after, by key.
        fields.check()
        return record


def log_redaction() -> Redaction:
    """What takes secrets out of everything this process logs through a log call.

    Where records are made, not on the handlers there are now: a service's client may log through a logger that
    doesn't pass its records up, or add a handler of its own later. That covers a record's message and the
    arguments it was made from, its exception and stack, the ``extra`` fields of the call that logged it, and
    the fields a record factory under this one put on it. A field or an argument that is a mapping, a sequence
    or a set is looked into, whatever it prints as; what any other object holds and doesn't print is not known.
    A field that a filter sets on a record once it is made is not checked at all. A record made some other way,
    built by hand or read back from a queue, is covered only if it is made with ``logging.makeLogRecord``.

    Once it has been given a secret, a handler receives a record's exception as text and never as the
    exception: ``exc_text`` holds it as the standard formatter writes one, with the secrets out, and
    ``exc_info`` is None, whether or not the text held a secret. An exception holds more than its text. A
    handler that prints a traceback with the variables of each frame would print a client's variable that
    holds the credential, whatever the error says. So a handler's own way of formatting an exception is not
    used: a formatter's ``formatException`` is not called, and a handler that draws tracebacks itself is
    handed none to draw. Until a secret is given, records are passed on with their exceptions as they were
    logged.

    It is also a function, which takes the same secrets out of any text. Call it again after anything that may
    have installed a record factory of its own, as a tracing library does: that puts this one back on top.
    """
    if not isinstance(logging.getLogRecordFactory(), _RedactingFactory):
        logging.setLogRecordFactory(_RedactingFactory(logging.getLogRecordFactory()))
    return _REDACTION


def redact_log(secret: str) -> None:
    """From here on, every record this process logs has ``secret`` taken out of it too, and so has every reason.

    The driver calls it with the credential it is given. A service's client calls it with whatever else must
    stay out of the log: a part of that credential it uses alone, or a token it trades the credential for. Only
    a secret as it was given here is recognised: whole, and for one of several lines, each line of eight
    characters or more.
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
    directory.

    A context path that the system reads to a directory is followed here, to see that the directory is inside
    its source. One that it reads to no directory is returned as it is, with no link in it followed, and
    staging refuses it: what an archive's links make of such a path is not the step's to work out.
    """
    layout = WorkLayout(PurePosixPath(root))
    source = image.source
    fetch_source(files_client, layout, source, workspace=workspace)
    context = Path(layout.context(source))
    # Asked of the system first, as it is of a download that is shared: see `_within`.
    if (
        source.context_path
        and os.path.isdir(context)
        and not _within(context, Path(layout.context(source.model_copy(update={"context_path": None}))))
    ):
        raise ImageFailed(f"context path {source.context_path!r} is not a directory inside its source")
    return context


def step_main(
    make_service: Callable[[RemoteBuildStepConfig, str | None], RemoteBuildService],
    *,
    read_credential: Callable[[], str] | None = None,
) -> int:
    """The whole step. ``make_service`` gets the step's config and the service's credential.

    The credential is the one in the environment, when the deployment keeps it in a platform secret. Otherwise it
    is what ``read_credential`` returns, for a step program whose Jobs profile mounts it as a file. Either way the
    driver keeps it out of the log and out of every reason, as it was given: whole, and for one of several
    lines, each line of eight characters or more. A part of it that a client uses alone, or a token a client
    trades it for, the driver doesn't know: the client gives those to :func:`redact_log`. A ``read_credential``
    that finds nothing fails every image, and says so. With neither, ``make_service`` gets None and finds the
    credential itself, and gives it to :func:`redact_log`.

    The step logs with the level first, in brackets, which Jobs' launcher reads. A step program that sets up
    logging of its own before it calls this keeps that format, which must end a line with the message: what
    follows an image's name on its failure line is taken for the reason. Its handlers receive a logged
    exception as text, already formatted and redacted, and not the exception, from when the driver or the
    client first gives a secret to keep out (see :func:`log_redaction`): a handler's own way of formatting
    an exception, a traceback on one line or one drawn with each frame's variables, is not used. What a step
    program prints, instead of logging, is not redacted, and what it prints after the driver's last line
    pushes the failure lines out of the end of the log that Jobs keeps: standard output is made line-buffered
    here so that nothing is held back until then.
    """
    # The level first, in brackets: the Jobs launcher reads it from there, and records any other line as an error.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    try:
        # Not every stream that stands in for standard output can be told to.
        sys.stdout.reconfigure(line_buffering=True)  # ty: ignore[unresolved-attribute]
    except (AttributeError, ValueError, OSError):
        pass
    config = RemoteBuildStepConfig.model_validate(read_step_config())
    # Every secret given to `redact_log`, the client's own included, whenever it is given.
    redact = log_redaction()
    try:
        if config.credential_in_environment:
            credential: str | None = take_credential(os.environ)
        elif read_credential is not None:
            try:
                credential = read_credential().strip()
            except Exception as exc:
                # Not what it says, unless the file system says it: a reader that quotes what it read would
                # put the credential, which nothing can take out yet, on every row.
                why = _os_error(exc) if isinstance(exc, OSError) and exc.errno is not None else type(exc).__name__
                raise RuntimeError(f"the build service's credential could not be read: {why}") from None
            if not credential:
                raise RuntimeError("the build service's credential is empty: the step's Jobs profile did not mount it")
        else:
            credential = None
        if credential is not None:
            redact.add(credential)
        attempt = JobAttempt.from_environment(os.environ)
        # As `service:builder`, acting for the submitter through the job's delegation.
        client = get_task_nemo_client("builder")
        builder = Builder(client_from_platform(client, BuilderClient), workspace=attempt.workspace)
        files_client = client_from_platform(client, FilesClient)
        service = make_service(config, credential)
        # A client may have installed a log record factory of its own, as a tracing library does.
        log_redaction()
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
        # Where it was raised, before the lines settling reads, which stay last. Not its message as a traceback
        # prints it: the lines below carry that, on one line each.
        logger.error("the step could not start, at:\n%s", "".join(traceback.format_tb(exc.__traceback__)))
        reason = redact(_one_line(redact(f"the step could not start: {type(exc).__name__}: {exc}")))
        for image in config.images:
            # Cut to share what Jobs keeps of the log, but never to less than says why: in a large set the first
            # images' lines then fall outside what is kept, and their rows say only what Jobs says of the job.
            _report(image.image, reason, room=max(_room(len(config.images), image.image), MAX_REASON_CHARS // 2))
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
        _remove(scratch)
