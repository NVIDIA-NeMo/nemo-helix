# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve entrypoints: orchestrator proxy and host-urls modes."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Literal

from uvicorn import run as run_uvicorn

from sandboxed_gym.netutil import get_node_ip
from sandboxed_gym.orchestrator import SandboxedGymOrchestrator, SandboxedGymSession
from sandboxed_gym.proxy_app import build_proxy_app
from sandboxed_gym.serve_config import SandboxedGymServeConfig

LOGGER = logging.getLogger(__name__)


def write_session_file(path: Path, session: SandboxedGymSession, *, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = session.descriptor(mode=mode, orchestrator_url=session.orchestrator_url)
    path.write_text(desc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote session descriptor to %s", path)


def _wait_forever() -> None:
    event = threading.Event()

    def _stop(signum: int, _frame: object) -> None:
        LOGGER.info("Received signal %s; shutting down", signum)
        event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    event.wait()


def _advertise_host(bind_host: str) -> str:
    """Address to publish for a proxy bound to ``bind_host``.

    The descriptor this lands in is the cross-job handoff, so a wildcard bind has to resolve to an
    address another pod can route to. Resolution runs after the broker and Gym host are up, so a
    node with no routable address degrades to loopback rather than taking the run down.
    """
    if bind_host not in ("", "0.0.0.0", "::"):
        return bind_host
    try:
        return get_node_ip()
    except OSError:
        LOGGER.warning("Could not resolve a node address for bind %r; advertising loopback", bind_host)
        return "127.0.0.1"


def serve(
    cfg: SandboxedGymServeConfig,
    *,
    mode: Literal["orchestrator", "host-urls"] | None = None,
    bind: str = "0.0.0.0:8090",
    advertise_url: str | None = None,
    session_file: Path | None = None,
) -> None:
    """Start broker + Gym host, then either proxy or print host URLs.

    ``advertise_url`` overrides the orchestrator URL published to cross-job clients, for the cases
    the bind address cannot describe -- a published container port, or an ingress in front of it.
    """
    mode = mode or cfg.serve_mode
    if advertise_url and mode != "orchestrator":
        # Rejected before provisioning: host-urls publishes the Gym host directly, so there is no
        # proxy for this URL to name and the descriptor would come out unchanged.
        raise ValueError(f"advertise_url applies to mode='orchestrator', not {mode!r}")
    orch = SandboxedGymOrchestrator()
    session = orch.start(cfg)

    try:
        if mode == "orchestrator":
            host, _, port_s = bind.partition(":")
            if not port_s:
                raise ValueError(f"bind must be host:port, got {bind!r}")
            port = int(port_s)
            session.orchestrator_url = (
                advertise_url.rstrip("/") if advertise_url else f"http://{_advertise_host(host)}:{port}"
            )
            if session_file is not None:
                write_session_file(session_file, session, mode=mode)
            print(
                session.descriptor(mode=mode).model_dump_json(indent=2),
                flush=True,
            )
            app = build_proxy_app(session)
            LOGGER.info("Orchestrator proxy listening on %s", bind)
            run_uvicorn(app, host=host, port=port, log_level="info")
        elif mode == "host-urls":
            if session_file is not None:
                write_session_file(session_file, session, mode=mode)
            print(
                session.descriptor(mode=mode).model_dump_json(indent=2),
                flush=True,
            )
            LOGGER.info("Host URLs published; waiting for SIGTERM/SIGINT")
            _wait_forever()
        else:
            raise ValueError(f"unknown mode: {mode!r}")
    finally:
        session.shutdown()
