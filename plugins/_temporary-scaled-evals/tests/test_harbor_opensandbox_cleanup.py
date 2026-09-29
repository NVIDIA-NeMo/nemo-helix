# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import pytest
from scaled_evals import harbor_opensandbox_cleanup as cleanup


def test_cleanup_selector_requires_deployment_and_evaluation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match=cleanup.EVALUATION_METADATA_KEY):
        cleanup.validate_selector({cleanup.DEPLOYMENT_METADATA_KEY: "d"})

    assert cleanup.main(["--protocol", "http", "--selector", f"{cleanup.EVALUATION_METADATA_KEY}=e"]) == 2
    assert cleanup.DEPLOYMENT_METADATA_KEY in json.loads(capsys.readouterr().out)["error"]


class _FakeManager:
    def __init__(self, live: list[list[str]]) -> None:
        self._live = live
        self.killed: list[str] = []
        self.closed = False

    async def kill_sandbox(self, sandbox_id: str) -> None:
        self.killed.append(sandbox_id)

    async def close(self) -> None:
        self.closed = True


def _patch_manager(monkeypatch: pytest.MonkeyPatch, manager: _FakeManager) -> None:
    async def fake_manager(_env: Mapping[str, str], _protocol: str) -> _FakeManager:
        return manager

    async def fake_live(m: _FakeManager, _selector: Mapping[str, str]) -> list[str]:
        return m._live.pop(0) if len(m._live) > 1 else m._live[0]

    monkeypatch.setattr(cleanup, "_manager", fake_manager)
    monkeypatch.setattr(cleanup, "_live", fake_live)


SELECTOR = {cleanup.DEPLOYMENT_METADATA_KEY: "d", cleanup.EVALUATION_METADATA_KEY: "e"}


def test_cleanup_kills_until_nothing_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager([["sb-1", "sb-2"], []])
    _patch_manager(monkeypatch, manager)

    report = asyncio.run(cleanup.destroy_owned_sandboxes({}, "http", SELECTOR, timeout_s=5, poll_s=0))

    assert report == {"killed": ["sb-1", "sb-2"], "failed": [], "remaining": []}
    assert manager.closed


def test_cleanup_reports_survivors_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager([["sb-1"]])
    _patch_manager(monkeypatch, manager)

    report = asyncio.run(cleanup.destroy_owned_sandboxes({}, "http", SELECTOR, timeout_s=0, poll_s=0))

    assert report["remaining"] == ["sb-1"]
    assert manager.killed == ["sb-1"]
