# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator URL that `serve` publishes is the cross-job handoff, so it has to be routable."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sandboxed_gym import serve as serve_module


@pytest.fixture
def started_session(monkeypatch):
    session = MagicMock()
    session.descriptor.return_value.model_dump_json.return_value = "{}"
    monkeypatch.setattr(serve_module.SandboxedGymOrchestrator, "start", lambda self, cfg: session)
    monkeypatch.setattr(serve_module, "build_proxy_app", lambda _session: MagicMock())
    monkeypatch.setattr(serve_module, "run_uvicorn", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_module, "get_node_ip", lambda: "10.0.0.9")
    return session


@pytest.mark.parametrize("bind", ["0.0.0.0:8090", ":8090"])
def test_a_wildcard_bind_advertises_the_node_address(started_session, bind: str) -> None:
    # Loopback here would hand Job B a URL that resolves inside its own pod.
    serve_module.serve(MagicMock(), mode="orchestrator", bind=bind)

    assert started_session.orchestrator_url == "http://10.0.0.9:8090"


def test_an_explicit_bind_host_is_advertised_verbatim(started_session) -> None:
    serve_module.serve(MagicMock(), mode="orchestrator", bind="127.0.0.1:8090")

    assert started_session.orchestrator_url == "http://127.0.0.1:8090"


def test_advertise_url_overrides_whatever_the_bind_implies(started_session) -> None:
    # A published container port or an ingress is not derivable from the bind address.
    serve_module.serve(MagicMock(), mode="orchestrator", bind="0.0.0.0:8090", advertise_url="http://localhost:9000/")

    assert started_session.orchestrator_url == "http://localhost:9000"


def test_a_node_without_a_routable_address_falls_back_to_loopback(started_session, monkeypatch) -> None:
    """Resolution runs after the broker and Gym host are already up, so it must not be the thing
    that takes the run down."""

    def _no_route() -> str:
        raise OSError("no route to host")

    monkeypatch.setattr(serve_module, "get_node_ip", _no_route)

    serve_module.serve(MagicMock(), mode="orchestrator", bind="0.0.0.0:8090")

    assert started_session.orchestrator_url == "http://127.0.0.1:8090"


def test_advertise_url_is_refused_in_host_urls_mode(started_session) -> None:
    with pytest.raises(ValueError, match="orchestrator"):
        serve_module.serve(MagicMock(), mode="host-urls", advertise_url="https://ingress.example/gym")

    started_session.shutdown.assert_not_called()
