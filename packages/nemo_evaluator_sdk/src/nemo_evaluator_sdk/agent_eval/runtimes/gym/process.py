# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running the ``gym`` CLI: locating it, watching it, and tearing it down.

Gym spawns a Ray cluster and several uvicorn servers, so a subprocess here is a process *tree*.
Everything about that lifecycle — resolution, streamed logs, readiness parsing, group teardown —
lives here so the runtime module can read as orchestration rather than process management.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import GymRuntimeConfig

logger = logging.getLogger(__name__)


_GYM_CLI = "gym"
#: Bound on `gym env validate`. It merges config without starting anything and returns in about a
#: second; this only exists so a wedged invocation cannot stall the run before it begins.
VALIDATE_TIMEOUT_S = 120.0
#: Where Gym's Hydra run directories are redirected, relative to the run's work dir.
#: Token-count keys read to decide whether a model was called at all. Both vocabularies, because
#: Gym's model servers report in either depending on the adapter: the Responses API spells them
#: `input_tokens`/`output_tokens`, Chat Completions `prompt_tokens`/`completion_tokens`.
LOG_TAIL_LINES = 40


#: Substrings that mark an override as carrying a credential. Matched case-insensitively against the
#: full dotted path, so nesting cannot hide one behind an innocuous leaf name.
def gym_executable() -> str:
    """Locate the ``gym`` CLI on PATH, or fail saying what to do about it.

    Deliberately PATH-only, with no config field pointing at a checkout or a particular venv: these
    runner configs become serialized job specs, and a local filesystem path cannot cross that
    boundary. Resolving here rather than at spawn time turns a missing Gym into one legible error
    instead of an ``ENOENT`` out of ``create_subprocess_exec`` after the run has already started.

    Note that Gym generally cannot live in this SDK's own environment: it imports Ray at module load,
    and nemo-helix excludes Ray by constraint over an unfixed CVE. Install Gym separately and put
    its ``bin`` on PATH; in a job image, the image owns PATH and this resolves normally.
    """
    resolved = shutil.which(_GYM_CLI)
    if resolved is None:
        raise RuntimeError(
            f"The {_GYM_CLI!r} CLI was not found on PATH. Install NeMo Gym in its own environment "
            "(it needs Ray, which nemo-helix excludes over an unfixed CVE, so it cannot share this "
            "one) and put that environment's `bin` directory on PATH. Each resources-server also "
            "ships its own requirements.txt, installed from that server's directory."
        )
    return resolved


_PENDING_SERVERS_RE = re.compile(r"(\d+)\s*/\s*(\d+) servers ready\. Waiting for servers to spin up: \[([^\]]*)\]")


def pending_servers(text: str) -> tuple[int, int, tuple[str, ...]] | None:
    """Recover ``(ready, total, still-pending)`` from the *last* readiness line Gym logged.

    The last line is the one that matters: earlier polls list servers that have since come up.
    Returns ``None`` when Gym never got as far as printing one (it died during composition, say),
    which the caller must treat as "no detail available" rather than "nothing pending".

    Never raises — this only ever decorates an error that is already being reported, and a parsing
    failure here must not replace the real one. Gym's log format is not contractual.
    """
    matches = _PENDING_SERVERS_RE.findall(text)
    if not matches:
        return None
    ready, total, names = matches[-1]
    try:
        return int(ready), int(total), tuple(re.findall(r"['\"]([^'\"]+)['\"]", names))
    except ValueError:  # pragma: no cover - the regex only matches digits
        return None


async def pump_stream(
    stream: asyncio.StreamReader | None,
    path: Path,
    *,
    label: str,
    tails: dict[str, deque[str]] | None = None,
    key: str = "stdout",
) -> None:
    """Stream a subprocess pipe to ``path`` while mirroring it to the module logger at ``DEBUG``.

    Reads in chunks rather than by line so a pathologically long line can't overrun asyncio's stream
    limit, and retains only the last :data:`LOG_TAIL_LINES` lines in memory (via ``tails[key]``) for
    inclusion in a failure message — the file on disk is the complete record.
    """
    tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
    if tails is not None:
        tails[key] = tail
    if stream is None:
        return

    def emit(raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        logger.debug("[%s %s] %s", label, key, text)
        tail.append(text)

    with path.open("wb") as handle:
        buffer = b""
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            handle.write(chunk)
            handle.flush()
            buffer += chunk
            *complete, buffer = buffer.split(b"\n")
            for line in complete:
                emit(line)
        if buffer:
            emit(buffer)


async def drain_pumps(pumps: Sequence[asyncio.Task[None]], *, grace_s: float, what: str) -> None:
    """Await log pumps after teardown, but never block on them indefinitely.

    A pump ends at pipe EOF, which requires *every* inheritor of the write end to close it. Gym's
    descendants inherit it, and Ray daemonizes ``gcs_server`` into its own session where our
    process-group signals can't reach it (see :func:`terminate`) — so a leaked grandchild can hold
    the pipe open forever. Waiting unconditionally would hang the whole run inside a ``finally``.

    Bounded instead: give the pumps ``grace_s`` to flush, then cancel and move on. Everything read
    before the stall is already written and flushed to the log file, so cancelling costs at most the
    tail of a transcript belonging to a process we just killed — and it is reported, not silent.
    """
    if not pumps:
        return
    _done, pending = await asyncio.wait(pumps, timeout=grace_s)
    if not pending:
        return
    logger.warning(
        "%d `%s` log pump(s) did not finish within %.1fs; the pipe is still held open (likely a Gym "
        "grandchild that outlived teardown, e.g. Ray's detached gcs_server). Abandoning them — the log "
        "file holds everything read up to this point, but its tail may be truncated.",
        len(pending),
        what,
        grace_s,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def _signal_group(pgid: int, sig: int) -> None:
    """Send ``sig`` to a process group, tolerating an already-dead group."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


async def terminate(proc: asyncio.subprocess.Process, *, grace_s: float = 30.0) -> None:
    """Tear down a backgrounded Gym subprocess *and its whole process group*.

    ``gym env start`` fans out into a Ray cluster + uvicorn servers, and Gym stops them from a
    ``KeyboardInterrupt`` handler (``finally: self.shutdown()``) — so we send **SIGINT** (not SIGTERM,
    which would bypass that handler and orphan Ray's detached ``gcs_server``). The child is spawned with
    ``start_new_session=True`` and leads its own group, so SIGINT to the group mimics Ctrl-C exactly;
    SIGKILL to the group is the escalation if Gym's graceful shutdown overstays ``grace_s``.

    Known limitation: Ray daemonizes ``gcs_server`` into its *own* session, outside this group, so the
    escalation SIGKILL cannot reach it — clean Ray teardown depends on Gym's SIGINT ``shutdown()``
    completing within ``grace_s``. POSIX-only (``killpg`` / ``getpgid`` / ``start_new_session`` / SIGINT).
    """
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    _signal_group(pgid, signal.SIGINT)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except asyncio.TimeoutError:
        _signal_group(pgid, signal.SIGKILL)
        await proc.wait()


def gym_invocation_env(config: GymRuntimeConfig) -> dict[str, str]:
    """The environment the ``gym`` CLI is invoked with.

    Three layers, lowest precedence first:

    1. **This process's environment.** Gym reads credentials from its own gitignored ``env.yaml``,
       but honours plenty of ordinary variables (``HF_TOKEN``, proxies) a caller expects to carry
       over.
    2. **``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0``.** Gym launches each server from its own subdir with its
       own ``.venv``; Ray (>=2.56) otherwise detects a ``uv run`` ancestor and tries to replicate that
       uv project onto its workers, asserting the project's ``pyproject.toml`` lives in the driver's
       cwd — which aborts startup. That hook is wrong for Gym, whose servers manage their own deps.
    3. **``config.env_vars``.** Named explicitly in the run config, so it wins over both. That
       includes the Ray setting: it is a default that makes Gym work, not an invariant, and someone
       debugging that hook needs a way to put it back.

    Lives here rather than in the runtime so the precedence is assertable without starting Ray.
    """
    return {**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0", **config.env_vars}
