# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the agent-hardener ``serve`` synth service.

The war-game job spawns ``agent-hardener serve`` (its own venv) and drives the interview + review over these
endpoints. Thin wrapper over httpx: each call returns the service's JSON dict
(``{thread_id, status, questions|suite, ...}``).
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_SYNTH_SERVICE, AgentHardenerRunError


class SynthClient:
    """Sync client for one synth run against a local ``agent-hardener serve`` instance."""

    def __init__(self, base_url: str, *, timeout: float = 900.0, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def healthz(self) -> bool:
        """True once the service is serving."""
        try:
            resp = self._client.get("/healthz")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    def start(self, config: str, *, validator: str | None = None) -> dict[str, Any]:
        """Begin a synth run for the manifest at *config*; returns the first interview or review step."""
        return self._post("/synth", {"config": config, "validator": validator})

    def answers(self, thread_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one interview round's answers; returns the next interview or review step."""
        return self._post(f"/synth/{thread_id}/answers", {"answers": answers})

    def write_suite(self, thread_id: str, suite: list[dict[str, Any]]) -> dict[str, Any]:
        """Persist the reviewed benign suite to ``requests.csv``; returns the final ``done`` step."""
        return self._post(f"/synth/{thread_id}/suite", {"suite": suite})

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        # The synth service is a local agent-hardener subprocess; a transport error or non-2xx from it is a
        # benign-suite generation failure, not a victim/network issue — classify it as such.
        try:
            resp = self._client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AgentHardenerRunError(
                CATEGORY_SYNTH_SERVICE, f"benign-suite service request to {path} failed: {exc}"
            ) from exc


def _free_port() -> int:
    """Pick a free localhost port (bind-and-release) to hand to ``agent-hardener serve``."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def launch_synth_service(
    agent_hardener_bin: Path, env: dict[str, str], *, log_path: Path | None = None, ready_timeout: float = 90.0
) -> Iterator[SynthClient]:
    """Spawn ``agent-hardener serve`` on a free localhost port, yield a connected client, tear it down.

    Raises ``AgentHardenerRunError`` if the server exits early or isn't healthy within *ready_timeout*.
    """
    port = _free_port()
    cmd = [str(agent_hardener_bin), "serve", "--host", "127.0.0.1", "--port", str(port)]
    with contextlib.ExitStack() as stack:
        sink = stack.enter_context(log_path.open("w", encoding="utf-8")) if log_path else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, env=env, stdout=sink, stderr=subprocess.STDOUT)
        stack.callback(_terminate, proc)
        client = stack.enter_context(SynthClient(f"http://127.0.0.1:{port}"))
        _await_ready(client, proc, ready_timeout)
        yield client


def _await_ready(client: SynthClient, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AgentHardenerRunError(
                CATEGORY_SYNTH_SERVICE, f"agent-hardener serve exited early (code {proc.returncode})"
            )
        if client.healthz():
            return
        time.sleep(0.5)
    raise AgentHardenerRunError(CATEGORY_SYNTH_SERVICE, f"agent-hardener serve not healthy within {timeout:.0f}s")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
