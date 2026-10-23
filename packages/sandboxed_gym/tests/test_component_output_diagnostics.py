# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise inherited component fds in a separate host, without Gym installs or GPUs."""

import codecs
import collections
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandboxed_gym.host.models import render_host_error
from sandboxed_gym.runtime import gym_host_runtime as runtime

_HOST = r'''
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from sandboxed_gym.runtime import gym_host_runtime as runtime

result_path, mode, fd, cause = sys.argv[1:]
fd = int(fd)

def component():
    # A real child inherits fd1/2, exactly where sys.stdout/sys.stderr wrappers lose it.
    # More than a pipe buffer proves the reader drains throughout execution, not only on error.
    script = r"""
import os, sys
fd = int(sys.argv[1])
os.write(fd, b'old-line\n' * 20000)
os.write(fd, b'x' * 5000 + b'\n')
os.write(fd, b'credential=' + os.environ['COMPONENT_API_KEY'][:10].encode())
os.write(fd, os.environ['COMPONENT_API_KEY'][10:].encode() + b'\n')
os.write(fd, sys.argv[2].encode())  # deliberately no final newline
"""
    subprocess.run([sys.executable, '-c', script, str(fd), cause], check=True, timeout=10)

if mode in ('rollout', 'capture-failure', 'sink-failure'):
    runtime._install_output_tail()
    if mode == 'capture-failure':
        def broken_capture(text):
            raise ValueError('capture unavailable')
        runtime._OUTPUT_CAPTURES[fd - 1]._tail.capture = broken_capture
    elif mode == 'sink-failure':
        runtime._OUTPUT_CAPTURES[fd - 1]._destination.close()
    component()
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError('component request failed'))
    handler = object.__new__(runtime.Handler)
    handler._chunked = False
    body = json.loads(handler._await_results(future, time.monotonic()))
else:
    def bootstrap():
        component()
        if mode == 'bootstrap':
            raise RuntimeError('Process custom_reward finished unexpectedly')
        return None, None, None
    class Server:
        def __init__(self, *args):
            pass
        def serve_forever(self):
            pass
    runtime.bootstrap_gym_host = bootstrap
    runtime._ensure_event_loop = lambda: None
    runtime._Server = Server
    runtime.main()
    body = runtime._BOOTSTRAP_ERROR or {'ready': runtime._READY}
    for capture in runtime._OUTPUT_CAPTURES:
        capture.drain()
Path(result_path).write_text(json.dumps(body))
'''


def _host(tmp_path: Path, mode: str, fd: int, cause: str) -> tuple[dict, subprocess.CompletedProcess]:
    result_path = tmp_path / "result.json"
    env = dict(os.environ)
    # Ensure the child tests this checkout, not a shared environment's editable installation.
    env["PYTHONPATH"] = str(Path(runtime.__file__).parents[2])
    env["COMPONENT_API_KEY"] = "component-secret-123456789"
    result = subprocess.run(
        [sys.executable, "-c", _HOST, str(result_path), mode, str(fd), cause],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=True,
    )
    return json.loads(result_path.read_text()), result


@pytest.mark.parametrize("fd", [1, 2])
@pytest.mark.parametrize(
    ("mode", "cause"),
    [
        (
            "bootstrap",
            "Missing pyproject.toml or requirements.txt for uv venv setup in server dir: "
            "/job/environment/resources_servers/custom_reward",
        ),
        ("bootstrap", "No solution found: intentionally-missing-bootstrap-package==0.0.0 was not found"),
        ("rollout", "RuntimeError: custom reward rollout failed: missing scoring dependency"),
    ],
)
def test_child_cause_survives_public_error_rendering(tmp_path, mode, fd, cause):
    body, result = _host(tmp_path, mode, fd, cause)
    error = body["error"]
    public_diagnostic = render_host_error(error)

    assert error["code"] == ("bootstrap_failed" if mode == "bootstrap" else "internal")
    assert cause in "\n".join(error["host_output_tail"]), "generic process exits are not actionable causes"
    assert cause in public_diagnostic
    assert "credential=***" in public_diagnostic
    assert "component-secret-123456789" not in json.dumps(body)
    assert len(error["host_output_tail"]) <= runtime._OUTPUT_TAIL_LINES
    assert all(len(line) <= runtime._MAX_CAPTURED_LINE_CHARS for line in error["host_output_tail"])
    assert "x" * runtime._MAX_CAPTURED_LINE_CHARS in public_diagnostic
    assert "x" * (runtime._MAX_CAPTURED_LINE_CHARS + 1) not in public_diagnostic
    assert cause in (result.stdout if fd == 1 else result.stderr), "container logs still receive child output"


def test_component_output_does_not_turn_success_into_a_bootstrap_failure(tmp_path):
    body, result = _host(tmp_path, "success", 2, "custom reward server ready")
    assert body == {"ready": True}
    assert "custom reward server ready" in result.stderr


def test_uncaught_host_failure_flushes_final_traceback_at_exit():
    """A parked daemon makes losing the final bind-error traceback deterministic."""
    script = """
import os
from sandboxed_gym.runtime import gym_host_runtime as runtime
runtime.threading.Thread.start = lambda self: None
runtime._ensure_event_loop = lambda: None

def bootstrap():
    os.write(2, b'last component setup detail\\n')
    return None, None, None

class BindFailure:
    def __init__(self, *args):
        raise OSError('intentional host bind failure')

runtime.bootstrap_gym_host = bootstrap
runtime._Server = BindFailure
runtime.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(runtime.__file__).parents[2])},
        timeout=10,
    )
    assert result.returncode != 0
    assert b"last component setup detail" in result.stderr
    assert b"Traceback (most recent call last):" in result.stderr
    assert b"OSError: intentional host bind failure" in result.stderr


@pytest.mark.parametrize("fd", [1, 2])
def test_exit_tee_preserves_raw_bytes(fd):
    """Container logs remain byte-exact, even when text diagnostics need replacement."""
    raw = b"invalid-utf8=\xff\xfe\r\nunterminated=\xe2\x82"
    script = f"""
import os
from sandboxed_gym.runtime import gym_host_runtime as runtime
runtime.threading.Thread.start = lambda self: None
runtime._install_output_tail()
os.write({fd}, {raw!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(runtime.__file__).parents[2])},
        timeout=10,
        check=True,
    )
    assert (result.stdout if fd == 1 else result.stderr) == raw


def _capture_without_thread(monkeypatch, chunks):
    """Model queued reads larger than one chunk without platform-specific pipe sizing."""
    capture = object.__new__(runtime._FdOutputTail)
    capture._lock = threading.Lock()
    capture._read_fd = -1
    monkeypatch.setattr(capture, "_destination", io.BytesIO(), raising=False)
    capture._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = collections.deque(maxlen=runtime._OUTPUT_TAIL_LINES)
    capture._tail = runtime._OutputTail(None, buffer)
    monkeypatch.setattr(runtime, "_OUTPUT_TAIL", buffer)
    monkeypatch.setattr(runtime, "_OUTPUT_CAPTURES", (capture,))
    calls = []

    def read(fd, size):
        calls.append(size)
        return next(chunks, b"")

    monkeypatch.setattr(runtime.os, "read", read)
    return capture, calls


def test_error_drain_retains_cause_after_multiple_queued_chunks(monkeypatch):
    cause = b"missing-install-marker in custom_reward"
    noise = b"component startup log\n" * 3000
    capture, calls = _capture_without_thread(monkeypatch, iter([noise, noise, cause]))

    body = runtime._runtime_error("bootstrap_failed", "component finished unexpectedly")

    assert cause.decode() in render_host_error(body["error"])
    assert capture._destination.getvalue() == noise * 2 + cause
    assert len(calls) == 4, "drain until the queue empties, not just one read"


def test_error_drain_is_bounded_for_a_continuous_writer(monkeypatch):
    def forever():
        while True:
            yield b"chatty component\n"

    _, calls = _capture_without_thread(monkeypatch, forever())

    runtime._runtime_error("internal", "failure")

    assert len(calls) == 16, "continuous output must not keep the error path draining forever"


@pytest.mark.parametrize("fd", [1, 2])
@pytest.mark.parametrize("mode", ["capture-failure", "sink-failure"])
def test_capture_or_sink_failure_does_not_block_component_writes(tmp_path, mode, fd):
    """A broken capture or log sink must still drain more than a pipe's capacity."""
    cause = "RuntimeError: component original cause"
    body, result = _host(tmp_path, mode, fd, cause)
    assert body["error"]["code"] == "internal"
    if mode == "sink-failure":
        diagnostic = render_host_error(body["error"])
        assert cause in diagnostic
        assert "component-secret-123456789" not in diagnostic
    else:
        assert cause in (result.stdout if fd == 1 else result.stderr)


def test_failed_reader_restores_host_fd_and_closes_pipe(monkeypatch):
    """If the pipe itself breaks, do not leave existing child writers blocked forever."""
    capture, _ = _capture_without_thread(monkeypatch, iter(()))
    capture._fd = 2
    capture._destination = SimpleNamespace(fileno=lambda: 99)
    restored = []
    closed = []

    def broken_select(*args):
        raise OSError("reader unavailable")

    monkeypatch.setattr(runtime.select, "select", broken_select)
    monkeypatch.setattr(runtime.os, "dup2", lambda source, target: restored.append((source, target)))
    monkeypatch.setattr(runtime.os, "close", closed.append)

    capture._run()

    assert restored == [(99, 2)]
    assert closed == [-1]
