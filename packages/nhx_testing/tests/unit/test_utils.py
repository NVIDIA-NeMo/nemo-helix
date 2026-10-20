# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for testing utils (e.g. short_unique_name, add_mock_provider validation)."""

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nhx.common.entities.constants import NAME_PATTERN
from nhx.testing import utils
from nhx.testing.utils import short_unique_name

_ENTITY_NAME_PATTERN = re.compile(NAME_PATTERN)


class TestEnsurePassthroughVirtualModel:
    @pytest.fixture
    def clock(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        now = [0.0]

        def advance(seconds: float) -> None:
            now[0] += seconds

        # Replace the helper's module reference, leaving other threads' clocks alone.
        monkeypatch.setattr(utils, "time", SimpleNamespace(time=lambda: now[0], sleep=advance))
        return now

    @pytest.mark.parametrize(
        ("statuses", "methods"),
        [
            ([201, 200], ["POST", "GET"]),
            # exist_ok=True follows a create conflict with a GET that can race deletion.
            ([409, 404, 201, 200], ["POST", "GET", "POST", "GET"]),
            # The explicit verification GET can also race deletion.
            ([201, 404, 201, 200], ["POST", "GET", "POST", "GET"]),
        ],
    )
    def test_retries_create_and_read(self, clock: list[float], statuses: list[int], methods: list[str]) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            status = statuses[len(requests)]
            requests.append(request)
            body = {"name": "model", "workspace": "test"} if status < 400 else {"detail": "model unavailable"}
            return httpx.Response(status, json=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = NemoClient(base_url="http://test", http_client=http_client)
            utils.ensure_passthrough_virtual_model(client, "test", "model")

        assert [request.method for request in requests] == methods
        assert clock[0] == (0.5 if 404 in statuses else 0.0)

    def test_persistent_create_read_race_times_out(self, clock: list[float]) -> None:
        methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return httpx.Response(409 if request.method == "POST" else 404, json={"detail": "model unavailable"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = NemoClient(base_url="http://test", http_client=http_client)
            with pytest.raises(TimeoutError, match="VirtualModel test/model not available after 1s.*HTTP 404"):
                utils.ensure_passthrough_virtual_model(client, "test", "model", timeout=1, poll_interval=0.5)

        assert methods == ["POST", "GET", "POST", "GET"]
        assert clock[0] == 1

    def test_create_permission_error_is_not_retried(self, clock: list[float]) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(403, json={"detail": "Forbidden"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = NemoClient(base_url="http://test", http_client=http_client)
            with pytest.raises(PermissionDeniedError):
                utils.ensure_passthrough_virtual_model(client, "test", "model")

        assert len(requests) == 1
        assert clock[0] == 0


class TestShortUniqueName:
    """Tests that short_unique_name produces valid entity names (NAME_PATTERN)."""

    def test_matches_entity_name_pattern(self):
        """Output matches NAME_PATTERN (lowercase, starts with letter, 2-63 chars, etc.)."""
        name = short_unique_name("provider")
        assert _ENTITY_NAME_PATTERN.match(name), f"{name!r} should match NAME_PATTERN"

    def test_lowercase_prefix(self):
        """Prefix is lowercased so result has no uppercase."""
        name = short_unique_name("Provider")
        assert name == name.lower()
        assert _ENTITY_NAME_PATTERN.match(name)

    def test_digit_prefix_becomes_letter(self):
        """Prefix that would start with a digit is fixed to start with 'a'."""
        name = short_unique_name("9invalid")
        assert name[0].isalpha() and name[0].islower()
        assert _ENTITY_NAME_PATTERN.match(name)

    def test_no_trailing_hyphen(self):
        """Result does not end with a hyphen."""
        name = short_unique_name("x")
        assert not name.endswith("-"), f"{name!r} must not end with hyphen"

    def test_consecutive_hyphens_collapsed(self):
        """Consecutive hyphens in prefix are collapsed."""
        name = short_unique_name("a--b")
        assert "--" not in name
        assert _ENTITY_NAME_PATTERN.match(name)


class TestRunNemoLocal:
    def test_uses_repo_virtualenv_nemo_when_available(self, tmp_path: Path, monkeypatch):
        repo_root = tmp_path / "repo"
        nemo = repo_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "nemo"
        nemo.parent.mkdir(parents=True)
        nemo.write_text("#!/bin/sh\n")
        monkeypatch.setenv("VIRTUAL_ENV", str(repo_root / ".venv"))
        monkeypatch.setattr(utils, "get_repo_root", lambda: repo_root)

        calls: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"cmd": cmd, **kwargs})
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(utils.subprocess, "run", fake_run)

        result = utils.run_nemo_local("config", "current-context")

        assert result.returncode == 0
        assert calls[0]["cmd"] == [str(nemo), "config", "current-context"]
        assert calls[0]["env"]["NEMO_TELEMETRY_ENABLED"] == "false"

    def test_falls_back_to_uv_when_not_running_from_repo_virtualenv(self, tmp_path: Path, monkeypatch):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(utils, "get_repo_root", lambda: repo_root)

        calls: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"cmd": cmd, **kwargs})
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(utils.subprocess, "run", fake_run)

        utils.run_nemo_local("config", "view")

        assert calls[0]["cmd"] == ["uv", "run", "--project", str(repo_root), "--frozen", "nemo", "config", "view"]

    def test_env_extra_can_enable_telemetry(self, tmp_path: Path, monkeypatch):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(utils, "get_repo_root", lambda: repo_root)

        calls: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"cmd": cmd, **kwargs})
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(utils.subprocess, "run", fake_run)

        utils.run_nemo_local("config", "view", env_extra={"NEMO_TELEMETRY_ENABLED": "true"})

        assert calls[0]["env"]["NEMO_TELEMETRY_ENABLED"] == "true"
