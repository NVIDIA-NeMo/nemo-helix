# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise inherited component fds in a separate host, without Gym installs or GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

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

if mode == 'rollout':
    runtime._install_output_tail()
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
