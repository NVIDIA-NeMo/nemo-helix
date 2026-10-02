# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for testing utils (e.g. short_unique_name, add_mock_provider validation)."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from nhx.common.entities.constants import NAME_PATTERN
from nhx.testing import utils
from nhx.testing.utils import short_unique_name

_ENTITY_NAME_PATTERN = re.compile(NAME_PATTERN)


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
