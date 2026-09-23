# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Error parser for subprocess output.

This module provides utilities to parse and extract meaningful error messages from
training subprocess output (stdout/stderr). It should be used by all training backends
(Automodel, NeMo-RL, Megatron Bridge) to capture errors for classification.

The extracted error messages are then matched against YAML rules by the
error converter to produce user-friendly error messages.
"""

import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass

# Number of recent output lines to keep for error parsing
MAX_OUTPUT_LINES = 500

# Patterns that indicate an error line (case-insensitive search)
# These match Python exception types and common error patterns from training libraries
ERROR_INDICATORS = [
    # Python exception type names (appear as "ExceptionType: message")
    "runtimeerror",
    "valueerror",
    "assertionerror",
    "importerror",
    "attributeerror",
    "keyerror",
    "typeerror",
    "filenotfounderror",
    "permissionerror",
    "oserror",
    "ioerror",
    # Generic error patterns
    "error:",
    "exception:",
    "traceback",
    # Automodel-specific patterns
    "instantiation failed",  # From ConfigNode.instantiate()
    "model compilation failed",  # From compile_utils.py
    # NeMo-RL patterns
    "ray error",
    "actor died",
    "worker crashed",
    # Megatron Bridge patterns
    "nemo error",
    "lightning error",
    # CUDA/GPU patterns
    "cuda out of memory",
    "out of memory",
    "oom",
    "cuda error",
    "cublas error",
    "cudnn error",
    # Distributed training patterns
    "nccl",
    "gloo",
    "distributed",
    "mpi error",
    # General failure patterns
    "failed",
    "failure",
    "abort",
    "killed",
    "segmentation fault",
    "signal",
]

# Regex to detect Python exception lines ("SomeError: message") and extract
# the type name (group 1) and message (group 2) as separate captures.
_EXCEPTION_RE = re.compile(
    r"\b(\w*(?:Error|Exception)):\s*(.*)",
    re.IGNORECASE,
)

# Wrapper exceptions from distributed training - skip these to find root cause
WRAPPER_EXCEPTION_PATTERNS = [
    "childfailederror",  # torch.distributed wrapper
    "torch.distributed.elastic",  # torch elastic wrapper
    "multiprocessing.errors",  # multiprocessing wrapper
]


@dataclass(frozen=True)
class ParsedError:
    """Error extracted from subprocess output.

    Preserves both the original exception type name (as printed in the
    traceback) and the message, so callers can reconstruct a typed
    exception for the converter's type-based matchers.
    """

    exception_type: str
    message: str

    def to_exception(self) -> Exception:
        """Reconstruct an exception that preserves the original type name.

        Dynamically creates an exception class whose ``__name__`` matches
        the original type (e.g. ``ValueError``, ``ResourceInsufficientError``)
        so that ``type_name`` YAML matchers can match it.  The class inherits
        from ``RuntimeError`` so that standard ``except Exception`` handling
        works without needing the real library class to be importable.
        """
        exc_class = type(self.exception_type, (RuntimeError,), {})
        return exc_class(self.message)


def _clean_line(line: str) -> str:
    """Remove common prefixes like [rank0]: from distributed output."""
    line = re.sub(r"^\[rank\d+\]:\s*", "", line.strip())
    return line.strip()


def _is_wrapper_exception(line: str) -> bool:
    """Check if this is a wrapper exception that should be skipped."""
    line_lower = line.lower()
    return any(pattern in line_lower for pattern in WRAPPER_EXCEPTION_PATTERNS)


def _extract_exception(line: str) -> ParsedError | None:
    """
    Extract the exception type and message from a subprocess output line.

    Examples:
        >>> _extract_exception("[rank0]: ValueError: invalid input")
        ParsedError(exception_type='ValueError', message='invalid input')
        >>> _extract_exception("torch.cuda.OutOfMemoryError: CUDA OOM")
        ParsedError(exception_type='OutOfMemoryError', message='CUDA OOM')
        >>> _extract_exception("  File 'train.py', line 42")
        None
        >>> _extract_exception("ChildFailedError: worker 0 failed")
        None  # Wrapper exception, skipped

    Returns None for non-exception lines and wrapper exceptions.
    """
    if _is_wrapper_exception(line):
        return None

    match = _EXCEPTION_RE.search(line)
    if match:
        exc_type = match.group(1).strip()
        message = match.group(2).strip() if match.group(2) else ""
        return ParsedError(
            exception_type=exc_type,
            message=message or exc_type,
        )

    return None


def parse_error_from_output(output_lines: deque, returncode: int) -> ParsedError:
    """
    Parse subprocess output and extract a structured error.

    Searches the captured output for Python exception lines and returns a
    ``ParsedError`` preserving both the exception type name and message.
    Callers use ``result.to_exception()`` to reconstruct a typed exception
    that works with both message-based *and* type-based YAML matchers.

    Strategy:
    1. Find the LAST Python exception line (e.g., "ValueError: message")
    2. Extract the type name and message separately
    3. Deduplicate across distributed ranks

    Args:
        output_lines: Rolling buffer of recent output lines.
        returncode: Process exit code.

    Returns:
        ParsedError with exception_type and message.
    """
    if not output_lines:
        return ParsedError("RuntimeError", f"Training failed with exit code: {returncode}")

    lines = list(output_lines)

    # Search backwards for exception lines and collect unique ones
    # (distributed training often prints the same error multiple times)
    found: list[ParsedError] = []
    seen_messages: set[str] = set()

    for i in range(len(lines) - 1, -1, -1):
        parsed = _extract_exception(lines[i])
        if parsed and parsed.message not in seen_messages:
            seen_messages.add(parsed.message)
            found.append(parsed)
            if len(found) >= 3:
                break

    if found:
        return found[0]

    # Fallback: search for any error-related lines
    error_lines: list[str] = []
    for line in reversed(lines):
        line_lower = line.lower()
        is_error_line = any(indicator in line_lower for indicator in ERROR_INDICATORS)
        if is_error_line:
            cleaned = _clean_line(line)
            if cleaned and cleaned not in error_lines:
                error_lines.insert(0, cleaned)
            if len(error_lines) > 10:
                break

    if error_lines:
        return ParsedError("RuntimeError", "\n".join(error_lines[-10:]))

    # Last resort: return last N lines of output
    last_lines = [_clean_line(line) for line in lines[-10:]]
    message = f"Training failed with exit code {returncode}. Last output:\n" + "\n".join(last_lines)
    return ParsedError("RuntimeError", message)


def read_subprocess_output(proc: subprocess.Popen, buffer: deque) -> None:
    """
    Read subprocess output, stream to console, and capture in buffer.

    This function is designed to run in a daemon thread alongside a subprocess,
    reading its stdout line-by-line, printing to console in real-time, and
    storing lines in a rolling buffer for later error extraction.

    Args:
        proc: The subprocess.Popen object with stdout=PIPE.
        buffer: A deque with maxlen to store recent output lines.
    """
    if proc.stdout is None:
        return

    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            # Stream to console
            sys.stdout.write(line)
            sys.stdout.flush()
            # Capture in rolling buffer
            buffer.append(line.rstrip("\n"))
    except (ValueError, OSError):
        # Process closed or pipe broken
        pass


__all__ = [
    "ERROR_INDICATORS",
    "MAX_OUTPUT_LINES",
    "ParsedError",
    "parse_error_from_output",
    "read_subprocess_output",
]
