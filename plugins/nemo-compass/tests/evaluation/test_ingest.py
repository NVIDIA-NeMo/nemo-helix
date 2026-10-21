# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from datetime import datetime, timezone
from typing import Any

import pytest
from evaluation import ingest

mint_agent_id = ingest.mint_agent_id


def test_mint_agent_id_format():
    assert re.fullmatch(r"tau2-airline-\d{8}-\d{6}-[0-9a-f]{4}", mint_agent_id("tau2-airline"))


def test_mint_agent_id_unique():
    assert mint_agent_id("x") != mint_agent_id("x")


def test_mint_agent_id_uses_utc(monkeypatch):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return cls(2026, 7, 14, 20, 42, 1, tzinfo=tz)

    monkeypatch.setattr(ingest, "datetime", _Clock)
    monkeypatch.setattr(ingest.os, "urandom", lambda _size: b"\xab\xcd")

    assert mint_agent_id("agent") == "agent-20260714-204201-abcd"


class _StubResponse:
    def __init__(self, *, status_code: int, text: str, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.body = body or {}

    def json(self) -> dict[str, Any]:
        return self.body


class _StubClient:
    def __init__(
        self, status: int = 201, body: dict | None = None, get_status: int = 200, get_body: dict | None = None
    ) -> None:
        self.status = status
        self.body = body or {}
        self.get_status = get_status
        self.get_body = get_body or {}
        self.calls: list[tuple[str, str, dict | None]] = []  # (method, url, json)

    def post(self, url: str, *, json: dict[str, Any]) -> _StubResponse:
        self.calls.append(("POST", url, json))
        return _StubResponse(status_code=self.status, text="body", body=self.body)

    def get(self, url: str) -> _StubResponse:
        self.calls.append(("GET", url, None))
        return _StubResponse(status_code=self.get_status, text="body", body=self.get_body)

    def close(self) -> None:
        pass


class _PollResp:
    status_code = 200
    text = ""

    def __init__(self, spans: list[dict]) -> None:
        self._spans = spans

    def json(self) -> dict[str, Any]:
        return {"data": self._spans}


class _PollClient:
    def __init__(self, spans: list[dict]) -> None:
        self._spans = spans
        self.closed = False

    def get(self, url: str, *, params: dict[str, Any]) -> _PollResp:
        return _PollResp(self._spans)

    def close(self) -> None:
        self.closed = True


def test_poll_visible_returns_when_all_visible():
    from evaluation.ingest import poll_visible

    ids = {"s1", "s2"}
    client = _PollClient([{"session_id": "s1"}, {"session_id": "s2"}])
    seen = poll_visible("http://x", "w", ids, client=client, sleep=lambda _s: None)
    assert seen == ids


def test_poll_visible_times_out_partial():
    from evaluation.ingest import poll_visible

    client = _PollClient([])  # nothing ever visible
    seen = poll_visible("http://x", "w", {"s1"}, client=client, timeout_s=0.0, sleep=lambda _s: None)
    assert seen == set()


def test_poll_visible_closes_client_it_creates(monkeypatch: pytest.MonkeyPatch):
    """When no client is injected, poll_visible must close the one it owns."""
    import evaluation.ingest as ingest

    created = _PollClient([{"session_id": "s1"}])
    monkeypatch.setattr(ingest.httpx, "Client", lambda **_kwargs: created)
    seen = ingest.poll_visible("http://x", "w", {"s1"}, sleep=lambda _s: None)
    assert seen == {"s1"}
    assert created.closed is True


def test_ensure_workspace_posts_to_entities_route():
    from evaluation.ingest import ensure_workspace

    stub = _StubClient(status=201)
    ensure_workspace("http://localhost:8080/", "tau2-airline-1", client=stub)
    assert stub.calls == [("POST", "http://localhost:8080/apis/entities/v2/workspaces", {"name": "tau2-airline-1"})]


def test_ensure_workspace_treats_409_as_ok():
    from evaluation.ingest import ensure_workspace

    ensure_workspace("http://x", "w", client=_StubClient(status=409))  # must not raise


def test_ensure_workspace_accepts_other_2xx():
    from evaluation.ingest import ensure_workspace

    ensure_workspace("http://x", "w", client=_StubClient(status=200))  # must not raise


def test_ensure_workspace_raises_on_other_error():
    from evaluation.ingest import ensure_workspace

    with pytest.raises(RuntimeError):
        ensure_workspace("http://x", "w", client=_StubClient(status=500))


def test_ensure_experiment_group_returns_id_on_create():
    from evaluation.ingest import ensure_experiment_group

    stub = _StubClient(status=201, body={"id": "grp-123", "name": "tau2-airline"})
    gid = ensure_experiment_group("http://x/", "tau2-airline-oracle", "tau2-airline", client=stub)
    assert gid == "grp-123"
    assert stub.calls == [
        ("POST", "http://x/apis/intake/v2/workspaces/tau2-airline-oracle/experiment-groups", {"name": "tau2-airline"})
    ]


def test_ensure_experiment_group_gets_id_on_409():
    from evaluation.ingest import ensure_experiment_group

    stub = _StubClient(status=409, get_status=200, get_body={"id": "grp-existing"})
    gid = ensure_experiment_group("http://x", "ws", "tau2-airline", client=stub)
    assert gid == "grp-existing"
    assert stub.calls == [
        ("POST", "http://x/apis/intake/v2/workspaces/ws/experiment-groups", {"name": "tau2-airline"}),
        ("GET", "http://x/apis/intake/v2/workspaces/ws/experiment-groups/tau2-airline", None),
    ]


def test_ensure_experiment_group_raises_on_error():
    from evaluation.ingest import ensure_experiment_group

    with pytest.raises(RuntimeError):
        ensure_experiment_group("http://x", "ws", "g", client=_StubClient(status=500))


def test_create_experiment_posts_full_body():
    from evaluation.ingest import create_experiment

    stub = _StubClient(status=201)
    create_experiment(
        "http://x",
        "ws",
        name="tau2-airline-20260626-000000-abcd",
        experiment_group_id="grp-1",
        dataset_name="tau2:airline",
        dataset_version="v1",
        metadata={"seed": 300},
        client=stub,
    )
    assert stub.calls == [
        (
            "POST",
            "http://x/apis/intake/v2/workspaces/ws/evaluations",
            {
                "name": "tau2-airline-20260626-000000-abcd",
                "experiment_group_id": "grp-1",
                "dataset_name": "tau2:airline",
                "dataset_version": "v1",
                "metadata": {"seed": "300"},
            },
        )
    ]


def test_create_experiment_raises_on_error():
    from evaluation.ingest import create_experiment

    with pytest.raises(RuntimeError):
        create_experiment(
            "http://x",
            "ws",
            name="n",
            experiment_group_id="g",
            dataset_name="d",
            dataset_version="v",
            metadata={},
            client=_StubClient(status=409),
        )
