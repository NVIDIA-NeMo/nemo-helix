# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gateway-status parsing and the preflight result cache."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import typer
from nemo_agent_hardener_plugin.cli import checks
from nemo_agent_hardener_plugin.config import AgentHardenerConfig

# Real `openshell status --gateway auto-defender` output, ANSI codes included.
CONNECTED = (
    "\x1b[1m\x1b[36mServer Status\x1b[39m\x1b[0m\n\n"
    "  \x1b[2mGateway:\x1b[0m auto-defender\n"
    "  \x1b[2mServer:\x1b[0m https://127.0.0.1:17670\n"
    "  \x1b[2mStatus:\x1b[0m \x1b[32mConnected\x1b[39m\n"
    "  \x1b[2mVersion:\x1b[0m 0.0.44\n"
)


def _config(tmp_path: Path, *, require_sandbox: bool = True) -> AgentHardenerConfig:
    return AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        require_sandbox=require_sandbox,
    )


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (CONNECTED, "Connected"),
        (CONNECTED.replace("Connected", "Not Connected"), "Not Connected"),
        ("  Status: Disconnected\n  Last Connected: 2026-01-01\n", "Disconnected"),
        ("no status row here", ""),
    ],
)
def test_gateway_status_parses_the_status_field(stdout: str, expected: str) -> None:
    assert checks.gateway_status(stdout) == expected


def test_gateway_status_does_not_match_substrings_of_other_rows() -> None:
    """The bug this replaced: `"Connected" in stdout` read these as healthy."""
    for stdout in (CONNECTED.replace("Connected", "Not Connected"), "  Status: Down\n  Last Connected: never\n"):
        assert checks.gateway_status(stdout).casefold() != "connected"


def test_probes_bound_a_wedged_daemon_to_a_few_seconds() -> None:
    """A hung `docker info`/`openshell status` used to stall every command for 20s/30s."""
    assert checks.PROBE_TIMEOUT_SECONDS <= 5


@pytest.mark.parametrize(
    ("probe", "binary", "expected"),
    [(checks.docker_ok, "docker", "docker info"), (checks.openshell_gateway_ok, "openshell", "openshell status")],
)
def test_a_timed_out_probe_fails_with_a_wedged_message(
    probe: Callable[[], tuple[bool, str]], binary: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checks.shutil, "which", lambda _name: f"/usr/bin/{binary}")

    def _timeout(*_args: object, **kwargs: object) -> object:
        assert kwargs["timeout"] == checks.PROBE_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd=binary, timeout=checks.PROBE_TIMEOUT_SECONDS)

    monkeypatch.setattr(checks.subprocess, "run", _timeout)
    ok, detail = probe()
    assert ok is False
    assert expected in detail and "timed out" in detail


def test_require_preflight_is_a_noop_without_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, require_sandbox=False)
    monkeypatch.setattr(checks, "run_checks", lambda _c: pytest.fail("checks must not run"))
    checks.require_preflight(config)


def test_require_preflight_exits_when_a_check_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "run_checks", lambda _c: [checks.CheckResult("docker", False, "daemon down")])
    with pytest.raises(typer.Exit):
        checks.require_preflight(_config(tmp_path))
