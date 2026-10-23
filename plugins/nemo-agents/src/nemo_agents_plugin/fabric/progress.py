# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort, payload-free observation of local Relay ATOF file sinks.

This deliberately does not use Fabric's streaming invocation: its 0.5 SDK
shields the invocation task and can wait indefinitely in stream cleanup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import threading
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

from nemo_fabric import FabricConfig

logger = logging.getLogger(__name__)
_POLL_SECONDS = 0.5
_REPORT_SECONDS = 5.0
_WORKER_SHUTDOWN_SECONDS = 0.1
_MAX_FILES = 4
_MAX_ENTRIES = 64
_MAX_READ = 64 * 1024
_MAX_LINE = 16 * 1024


@contextmanager
def _directory(root: Path, parts: tuple[str, ...]) -> Generator[int, None, None]:
    """Open each path component without following symlinks (including root)."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _sinks(config: FabricConfig, base: Path, artifacts: Path) -> tuple[list[tuple[tuple[str, ...], str]], bool]:
    relay = config.relay
    observability = relay.observability if relay else None
    atof = observability.atof if observability else None
    payload = atof if isinstance(atof, dict) else atof.model_dump() if atof else {}
    if not payload.get("enabled"):
        return [], True
    sinks = []
    for sink in (payload.get("sinks") or [])[:_MAX_FILES]:
        if sink.get("type") != "file":
            continue
        directory = Path(sink.get("output_directory") or "artifacts/relay")
        if not directory.is_absolute():
            directory = base / directory
        # Lexical containment first; actual traversal below rejects symlinks.
        if ".." in directory.parts or not directory.is_relative_to(artifacts):
            continue
        filename = str(sink.get("filename") or "events.atof.jsonl")
        if filename in {".", ".."} or Path(filename).name != filename:
            continue
        sinks.append((directory.relative_to(base).parts, filename))
    return sinks, False


class LocalProgressObserver:
    """Read bounded batches; emit only fixed lifecycle labels and counters."""

    def __init__(self, config: FabricConfig, base: Path, artifacts: Path) -> None:
        self.base = base.absolute()
        self.sinks, self.disabled = _sinks(config, self.base, artifacts.absolute())
        self.started = monotonic()
        self.last_event: float | None = None
        self.last_report = self.started - _REPORT_SECONDS
        self.counts: Counter[str] = Counter()
        self.offsets: dict[tuple[int, int], int] = {}
        self.buffers: dict[tuple[int, int], bytes] = {}
        self.dropping: set[tuple[int, int]] = set()
        self.warned = False
        self._stop = threading.Event()

    def _read(self, fd: int, seen: set[tuple[int, int]]) -> None:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular telemetry file")
        key = (info.st_dev, info.st_ino)
        if key in seen or (key not in self.offsets and len(self.offsets) >= _MAX_FILES):
            return
        seen.add(key)
        offset = self.offsets.get(key, 0)
        if info.st_size < offset:
            offset = 0
            self.buffers.pop(key, None)
            self.dropping.discard(key)
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, _MAX_READ)
        self.offsets[key] = offset + len(data)
        data = self.buffers.pop(key, b"") + data
        lines = data.split(b"\n")
        for line in lines[:-1]:
            if key in self.dropping:
                self.dropping.discard(key)
                continue
            if len(line) > _MAX_LINE:
                self.counts["discarded"] += 1
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError, RecursionError):
                self.counts["discarded"] += 1
                continue
            if not isinstance(record, dict) or record.get("kind") != "scope":
                continue
            category, phase = record.get("category"), record.get("scope_category")
            # Never echo arbitrary values, even names, IDs, status or timestamps.
            if category in ("agent", "llm", "tool") and phase in ("start", "end"):
                self.counts[f"{category}_{phase}"] += 1
                self.last_event = monotonic()
        tail = lines[-1]
        if len(tail) > _MAX_LINE:
            self.dropping.add(key)
            self.counts["discarded"] += 1
        elif key not in self.dropping:
            self.buffers[key] = tail

    def poll(self) -> None:
        seen: set[tuple[int, int]] = set()
        for parts, filename in self.sinks:
            if self._stop.is_set():
                return
            try:
                with _directory(self.base, parts) as directory_fd:
                    with os.scandir(directory_fd) as entries:
                        for index, entry in enumerate(entries):
                            if self._stop.is_set():
                                return
                            if index >= _MAX_ENTRIES:
                                break
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            try:
                                with _directory(self.base, (*parts, entry.name)) as runtime_fd:
                                    fd = os.open(
                                        filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=runtime_fd
                                    )
                                    try:
                                        self._read(fd, seen)
                                    finally:
                                        os.close(fd)
                            except FileNotFoundError:
                                continue
            except FileNotFoundError:
                continue
        # A cancelled observer may finish an in-flight syscall later. Do not
        # publish a late batch after the invocation has already ended.
        if self._stop.is_set():
            return
        now = monotonic()
        if self.last_event is not None and now - self.last_report >= _REPORT_SECONDS:
            logger.info(
                "Agent telemetry progress: elapsed=%.1fs last_event_age=%.1fs "
                "agent_start=%d agent_end=%d llm_start=%d llm_end=%d tool_start=%d tool_end=%d discarded=%d.",
                now - self.started,
                now - self.last_event,
                *(
                    self.counts[key]
                    for key in (
                        "agent_start",
                        "agent_end",
                        "llm_start",
                        "llm_end",
                        "tool_start",
                        "tool_end",
                        "discarded",
                    )
                ),
            )
            self.last_report = now
        elif self.last_event is None and not self.warned and now - self.started >= 30:
            logger.warning("No live agent telemetry observed; files may be absent or buffered. Invocation continues.")
            self.warned = True

    async def run(self) -> None:
        if not self.sinks:
            if self.disabled:
                logger.info("Live agent telemetry unavailable: Relay ATOF telemetry is disabled or unconfigured.")
            else:
                logger.info("Live agent telemetry unavailable: requires a local ATOF file sink beneath artifacts.")
            return
        logger.info("Observing local agent telemetry; waiting for lifecycle records (payloads omitted).")
        loop = asyncio.get_running_loop()
        done = asyncio.Event()

        def observe() -> None:
            """Keep blocking filesystem work outside the invocation event loop."""
            try:
                while not self._stop.is_set():
                    self.poll()
                    if self._stop.wait(_POLL_SECONDS):
                        break
            except Exception:
                # Neither exception text nor paths are safe telemetry output.
                if not self._stop.is_set():
                    logger.warning("Live agent telemetry observer failed; invocation continues without observation.")
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:
                    # A stalled syscall can return after asyncio.run has exited.
                    pass

        # Do not use asyncio.to_thread/default executors: asyncio.run waits for
        # those workers on exit. A syscall cannot be forcibly cancelled; one
        # daemon worker may remain until it returns or the job process exits.
        worker = threading.Thread(target=observe, name="agent-job-telemetry", daemon=True)
        try:
            worker.start()
            await done.wait()
        except Exception:
            logger.warning("Live agent telemetry observer failed; invocation continues without observation.")
        finally:
            self._stop.set()
            if worker.ident is not None:
                try:
                    await asyncio.wait_for(done.wait(), timeout=_WORKER_SHUTDOWN_SECONDS)
                except TimeoutError:
                    logger.warning("Live agent telemetry observer shutdown timed out; abandoning daemon worker.")
