# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract the exceptions a failed training subprocess printed.

Training runs in a subprocess (torchrun for Automodel, a Ray driver for NeMo-RL), so the only record of a
failure is its output. The classifier matches the exceptions found here against the error rules.
"""

import re
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

# Number of recent output lines to keep for error parsing. Every rank prints its traceback and a
# native backtrace, so the window must hold several of them.
MAX_OUTPUT_LINES = 2000

# Lines that suggest an error when no exception line was printed.
ERROR_INDICATORS = [
    "error:",
    "exception:",
    "traceback",
    "instantiation failed",
    "model compilation failed",
    "actor died",
    "worker crashed",
    "out of memory",
    "cuda error",
    "nccl warn",  # not "nccl", which matches config lines such as "backend: nccl"
    "segmentation fault",
    "signal",
    "killed",
    "failed",
    "abort",
]

# A Python exception line: "TypeName: message". The type may be module qualified
# ("torch.OutOfMemoryError") and need not end in Error ("TimeoutExpired", pyarrow's "ArrowInvalid").
# It must not follow a ``::`` or be followed by one, which leaves out C++ names in native backtraces,
# such as "c10::DistNetworkError::DistNetworkError(c10::SourceLocation, ...)".
_EXCEPTION_RE = re.compile(
    r"(?<![\w:.])((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Expired|Interrupt|Exit|Failed|Aborted|Rejected|Invalid)):(?!:)\s*(.*)",
)

# Wrapper exceptions that only report that another process failed.
WRAPPER_EXCEPTION_PATTERNS = [
    "childfailederror",
    "torch.distributed.elastic",
    "multiprocessing.errors",
]

# Exceptions printed while a failed process shuts down, never the cause of the failure.
_SHUTDOWN_EXCEPTION_TYPES = {"CancelledError", "KeyboardInterrupt", "SystemExit"}
_SHUTDOWN_MESSAGES = ("sys.meta_path is None", "Python is likely shutting down")

# Failures that some processes report without an exception line, or on a detail line under one, such
# as Ray's memory monitor, a raylet that shut down because its agent died, a vLLM worker, NCCL, or
# torchrun's report of a worker killed by a signal. A line with one of these becomes an exception
# named ``LogMessage``.
_FAILURE_LINE_RE = re.compile(
    r"CUDA out of memory|running low on memory|OOM killer|No space left on device|Disk quota exceeded"
    r"|Error while creating shared memory segment|Signal \d+ \(SIG[A-Z]+\) received by PID"
    r"|runtime_env_agent failed and raylet fate-shares with it"
    r"|dashboard_agent failed and raylet fate-shares with it"
)
LOG_MESSAGE_TYPE = "LogMessage"

# Ray appends this to a line it has deduplicated across workers.
_RAY_REPEATED_RE = re.compile(r"\s*\[repeated \d+x across cluster\]\s*$")


@dataclass(frozen=True)
class ParsedError:
    """An exception printed in subprocess output: its type name and message."""

    exception_type: str
    message: str

    def to_exception(self) -> Exception:
        """Rebuild an exception whose class name matches the printed type, for ``type_name`` rules."""
        exc_class = type(self.exception_type, (RuntimeError,), {})
        return exc_class(self.message)


def _clean_line(line: str) -> str:
    """Remove prefixes like ``[rank0]:`` from distributed output."""
    return re.sub(r"^\[rank\d+\]:\s*", "", line.strip()).strip()


def _extract_exception(line: str) -> ParsedError | None:
    """Return the failure on this line, or None for other lines, wrappers, and shutdown noise."""
    line = _RAY_REPEATED_RE.sub("", line)
    if any(pattern in line.lower() for pattern in WRAPPER_EXCEPTION_PATTERNS):
        return None
    match = _EXCEPTION_RE.search(line)
    if match is None:
        if _FAILURE_LINE_RE.search(line):
            return ParsedError(LOG_MESSAGE_TYPE, _clean_line(line))
        return None
    # "subprocess.TimeoutExpired" -> "TimeoutExpired", the name type_name rules use.
    exc_type = match.group(1).strip().rsplit(".", 1)[-1]
    message = match.group(2).strip()
    if exc_type in _SHUTDOWN_EXCEPTION_TYPES or any(text in message for text in _SHUTDOWN_MESSAGES):
        return None
    return ParsedError(exception_type=exc_type, message=message or exc_type)


def find_exceptions(lines: Iterable[str]) -> list[ParsedError]:
    """Return each distinct exception printed, ordered by its last appearance.

    Every rank and every retry prints the same exception, so each is kept once, at the place it last
    appeared.
    """
    seen: dict[ParsedError, None] = {}
    for line in lines:
        parsed = _extract_exception(line)
        if parsed is not None:
            seen.pop(parsed, None)
            seen[parsed] = None
    return list(seen)


def fallback_error(lines: Iterable[str], returncode: int) -> ParsedError:
    """Describe a failure that printed no exception line, from the lines that look like errors."""
    all_lines = list(lines)
    if not all_lines:
        return ParsedError("RuntimeError", f"Training failed with exit code: {returncode}")
    error_lines: list[str] = []
    for line in reversed(all_lines):
        if any(indicator in line.lower() for indicator in ERROR_INDICATORS):
            cleaned = _clean_line(line)
            if cleaned and cleaned not in error_lines:
                error_lines.insert(0, cleaned)
            if len(error_lines) >= 10:
                break
    if error_lines:
        return ParsedError("RuntimeError", "\n".join(error_lines))
    last_lines = [_clean_line(line) for line in all_lines[-10:]]
    return ParsedError(
        "RuntimeError", f"Training failed with exit code {returncode}. Last output:\n" + "\n".join(last_lines)
    )


def read_subprocess_output(proc: subprocess.Popen, buffer: deque) -> None:
    """Stream a subprocess's output to the console and keep the recent lines in ``buffer``.

    Meant to run in a daemon thread next to the subprocess.
    """
    if proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            buffer.append(line.rstrip("\n"))
    except (ValueError, OSError):
        # The process closed its output.
        pass
