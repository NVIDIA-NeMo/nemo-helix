# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process supervision for the benchmark harness.

The harness fans out several long-lived child processes per run (two mock LLM
``uvicorn`` servers, ``nemo services run``, and the AIPerf shim) and must clean
them all up — including any workers they fork — even when the parent dies
mid-run. This module wraps ``subprocess.Popen`` in a context-managed
``SupervisedProcess`` that:

* runs each child in a new session (``start_new_session=True``) so we can send
  signals to the whole process group via ``os.killpg`` and reap workers that
  ``uvicorn --workers N`` and ``nemo services`` fork off,
* merges stdout/stderr into a per-process log file under ``run_dir/logs/``,
* escalates SIGTERM → SIGKILL on shutdown if the child doesn't exit in time.

``supervised_processes`` starts the children in order, polls each spec's
``health_url`` before moving on, and guarantees every already-started child is
stopped if a later one fails to come up. ``wait_http`` implements the readiness
probe used by ``supervised_processes`` and for externally managed dependencies
(ex. ``--reuse-services``).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

import httpx
from nemo_guardrails_plugin.benchmarks.bootstrap import build_env

log = logging.getLogger(__name__)

# How long to wait for a child (and its process group) to exit after SIGTERM
# before we escalate to SIGKILL.
_TERMINATE_TIMEOUT_SECONDS = 20


@dataclass
class SupervisedProcess:
    """A long-lived child managed as a context manager.

    Stdout and stderr are merged into a single log file. The child is placed in a
    new session via ``start_new_session=True`` so ``os.killpg`` reaps any workers
    it forks.
    """

    # Human-readable identifier used in log messages (e.g. ``"mock-llama"``,
    # ``"nhx-services"``). Also drives the log filename via ``log_path``.
    name: str
    # The argv list passed to ``subprocess.Popen``. Constructed by the caller;
    # never assembled from user input here (hence the ``noqa: S603`` in ``start``).
    cmd: list[str]
    # File to merge stdout + stderr into. Parent dirs are created on ``start``.
    # The harness uploads these as CI artifacts on failure.
    log_path: Path
    # Working directory for the child process. Set per-process so e.g. the mock
    # LLM servers run from inside the NeMo-Guardrails checkout where their
    # configs live.
    cwd: Path
    # Extra env vars to overlay on top of the parent's ``os.environ`` (e.g.
    # ``PYTHONPATH``, ``NHX_DATA_DIR``). ``None`` means "inherit unchanged".
    env: dict[str, str] | None = None
    # Readiness probe polled after ``start()`` when this process is entered via
    # ``supervised_processes``. ``None`` skips the probe (e.g. when reusing an
    # externally managed dependency).
    health_url: str | None = None
    health_timeout_seconds: float = 60.0
    # The live ``Popen`` handle, populated by ``start()`` and consulted by
    # ``stop()``. Excluded from ``__init__`` and ``repr`` since it's pure state.
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    # Open file handle for ``log_path``, held so ``stop()`` can close it in a
    # ``finally`` block regardless of how termination unwinds.
    _log_fh: IO[bytes] | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Spawn the child, redirecting stdout+stderr to ``log_path``.

        Raises ``RuntimeError`` if called twice on the same instance. A
        ``SupervisedProcess`` wraps exactly one child process over its lifetime;
        to restart, build a new ``SupervisedProcess`` rather than calling
        ``start()`` again on a stopped one.
        """
        if self._proc is not None:
            raise RuntimeError(f"Process {self.name!r} already started")

        # Ensure the log file's parent directory exists.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open the log file for writing.
        self._log_fh = self.log_path.open("wb")

        log.info("Starting %s; log=%s", self.name, self.log_path)
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd),
                env=build_env(extra_env=self.env),
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            # Popen failed; close the log handle so we don't leak it.
            self._log_fh.close()
            self._log_fh = None
            raise

    def stop(self) -> None:
        """Terminate the child's process group and close the log file.

        Sends SIGTERM to the whole group, waits up to ``_TERMINATE_TIMEOUT_SECONDS``,
        then escalates to SIGKILL. Safe to call when the child was never started,
        has already exited on its own, or races with us between signals — every
        such case becomes a no-op that still closes the log handle.
        """
        proc = self._proc
        if proc is None:
            return

        try:
            # Child already exited on its own (crashed, finished early, etc.) —
            # nothing to signal, just fall through to the log-close in `finally`.
            if proc.poll() is not None:
                return

            # Look up the process group id so we can signal the child *and*
            # every worker it forked (uvicorn --workers, nemo services).
            try:
                pgid = os.getpgid(proc.pid)
                log.info("Stopping %s pid=%d (pgid=%d)", self.name, proc.pid, pgid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return

            # Give the group a chance to shut down gracefully. If SIGTERM
            # doesn't take effect in time, escalate to SIGKILL on the whole
            # group and then wait unconditionally.
            try:
                proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                log.warning(
                    "%s did not exit after %ds; sending SIGKILL",
                    self.name,
                    _TERMINATE_TIMEOUT_SECONDS,
                )
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        finally:
            # Always close the log handle, even if we returned early above or
            # the wait/signal calls raised something unexpected. Idempotent:
            # subsequent stop() calls (ex. via __exit__ after manual stop) are
            # no-ops once the handle is None.
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None

    def __enter__(self) -> "SupervisedProcess":
        """Context-manager entry: start the child and return ``self``."""
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit: stop the child regardless of exception state."""
        self.stop()


@contextmanager
def supervised_processes(specs: list[SupervisedProcess]) -> Iterator[list[SupervisedProcess]]:
    """Start every spec in order; stop them in reverse on exit.

    Backed by ``ExitStack`` so that if any spec's ``start()`` raises, every
    already-started child gets ``stop()``-ed before the exception propagates.
    On clean exit, children are torn down in LIFO order (i.e. NHX services
    stop before the mock LLMs they depend on).
    """
    with ExitStack() as stack:
        for spec in specs:
            stack.enter_context(spec)
            if spec.health_url is not None:
                wait_http(
                    spec.health_url,
                    timeout_seconds=spec.health_timeout_seconds,
                    label=spec.name,
                )
        yield specs


def wait_http(url: str, *, timeout_seconds: float, label: str, poll_interval: float = 1.0) -> None:
    """Poll ``url`` until it returns < 400 or ``timeout_seconds`` elapses.

    Used as a readiness gate between starting one supervised process and the
    next. ``label`` is included in the ``TimeoutError`` message so it's clear
    which dependency failed to come up. Raises ``TimeoutError`` with the last
    HTTP error or transport exception attached.
    """
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code < 400:
                return
            last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for {label} at {url}: {last_error}")
