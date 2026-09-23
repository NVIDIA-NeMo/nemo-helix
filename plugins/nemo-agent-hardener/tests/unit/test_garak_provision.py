# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the plugin's garak-venv preflight check and provisioning delegation."""

import subprocess
from pathlib import Path

import pytest
import typer
from nemo_agent_hardener_plugin.cli import checks, provisioning
from nemo_agent_hardener_plugin.config import GARAK_PYTHON_ENVVAR, AgentHardenerConfig


def _config(tmp_path: Path) -> AgentHardenerConfig:
    return AgentHardenerConfig(venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak-venv")


def test_run_checks_includes_garak_venv(tmp_path: Path) -> None:
    labels = [c.label for c in checks.run_checks(_config(tmp_path))]
    assert "agent-hardener venv" in labels
    assert "garak venv" in labels


def test_garak_venv_ok_reports_missing(tmp_path: Path) -> None:
    ok, detail = checks.garak_venv_ok(_config(tmp_path))
    assert ok is False
    assert "garak venv missing" in detail


def test_garak_venv_ok_reports_present(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.garak_python.parent.mkdir(parents=True)
    cfg.garak_python.touch()
    ok, _ = checks.garak_venv_ok(cfg)
    assert ok is True


def test_run_agent_hardener_setup_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    recorded_cmd: list[str] = []
    recorded_env: dict[str, str] = {}

    def fake_run(cmd: list[str], action: str, env: dict[str, str] | None = None) -> None:
        recorded_cmd[:] = cmd
        recorded_env.update(env or {})
        cfg.garak_python.parent.mkdir(parents=True, exist_ok=True)
        cfg.garak_python.touch()  # simulate agent-hardener setup creating the venv

    monkeypatch.setattr(provisioning, "run_subprocess", fake_run)
    provisioning.run_agent_hardener_setup(cfg, force=False)

    assert recorded_cmd == [str(cfg.agent_hardener_bin), "setup"]
    assert recorded_env[GARAK_PYTHON_ENVVAR] == str(cfg.garak_python)


def test_run_agent_hardener_setup_force_passes_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    cfg.garak_python.parent.mkdir(parents=True)
    cfg.garak_python.touch()
    recorded: dict[str, object] = {}

    def fake_run(cmd: list[str], action: str, env: dict[str, str] | None = None) -> None:
        recorded["cmd"] = cmd

    monkeypatch.setattr(provisioning, "run_subprocess", fake_run)
    provisioning.run_agent_hardener_setup(cfg, force=True)
    assert recorded["cmd"] == [str(cfg.agent_hardener_bin), "setup", "--force"]


def test_run_agent_hardener_setup_runs_even_when_garak_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    cfg.garak_python.parent.mkdir(parents=True)
    cfg.garak_python.touch()  # garak already present, yet setup still runs to re-ensure the gateway
    called = False

    def fake_run(cmd: list[str], action: str, env: dict[str, str] | None = None) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(provisioning, "run_subprocess", fake_run)
    provisioning.run_agent_hardener_setup(cfg, force=False)
    assert called is True  # not gated on the garak venv — gateway is re-ensured every setup


# ── provision_venv: package index ────────────────────────────────────────


def _capture_provision(monkeypatch: pytest.MonkeyPatch, cfg: AgentHardenerConfig) -> list[list[str]]:
    """Record the uv commands provision_venv issues, faking a successful install."""
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], action: str, env: dict[str, str] | None = None, **_k: object) -> None:
        commands.append(cmd)
        cfg.agent_hardener_bin.parent.mkdir(parents=True, exist_ok=True)
        cfg.agent_hardener_bin.touch()  # simulate uv producing the binary

    monkeypatch.setattr(provisioning.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(provisioning, "run_subprocess", fake_run)
    return commands


def test_default_install_is_a_plain_pypi_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The private index is a temporary workaround; the *default* must stay the public-PyPI path.

    Asserted exactly rather than loosely so nobody can bake a registry into the shipped defaults —
    when agent-hardener is published to PyPI this command must already be the whole story.
    """
    cfg = _config(tmp_path)
    commands = _capture_provision(monkeypatch, cfg)

    provisioning.provision_venv(cfg, force=True)

    install = next(c for c in commands if "install" in c)
    # A version floor is fine (BYO needs `init --dockerfile`); a baked-in registry is not.
    assert install == [
        "uv",
        "pip",
        "install",
        "--python",
        str(cfg.venv_path / "bin" / "python"),
        "nvidia-agent-hardener>=0.0.11",
    ]


def test_provision_venv_passes_configured_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--index` is additive: agent-hardener comes from the registry, its deps still from PyPI."""
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        index_url="https://registry.example/simple",
        spec="agent-hardener==0.0.2",
    )
    commands = _capture_provision(monkeypatch, cfg)

    provisioning.provision_venv(cfg, force=True)

    install = next(c for c in commands if "install" in c)
    assert install[install.index("--index") + 1] == "https://registry.example/simple"
    assert "--default-index" not in install  # must not replace PyPI
    assert install[-1] == "agent-hardener==0.0.2"  # spec stays the final argument


def test_provision_venv_passes_index_strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Needed when the index carries packages shadowing PyPI — first-index would fail to resolve them."""
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        index_url="private=https://registry.example/simple",
        index_strategy="unsafe-best-match",
    )
    commands = _capture_provision(monkeypatch, cfg)

    provisioning.provision_venv(cfg, force=True)

    install = next(c for c in commands if "install" in c)
    assert install[install.index("--index") + 1] == "private=https://registry.example/simple"
    assert install[install.index("--index-strategy") + 1] == "unsafe-best-match"


def test_index_strategy_is_omitted_without_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An index alone must not silently relax uv's dependency-confusion protection."""
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        index_url="https://registry.example/simple",
    )
    commands = _capture_provision(monkeypatch, cfg)

    provisioning.provision_venv(cfg, force=True)

    assert "--index-strategy" not in next(c for c in commands if "install" in c)


@pytest.mark.parametrize(
    ("index_url", "expected"),
    [
        # Artifactory's "Set Me Up" embeds the access token directly in the URL.
        (
            "https://user:AKCp8token@artifactory.example/api/pypi/repo/simple",
            "https://***:***@artifactory.example/api/pypi/repo/simple",
        ),
        # uv's named form keeps its prefix.
        (
            "nv-shared-pypi=https://user:AKCp8token@artifactory.example/simple",
            "nv-shared-pypi=https://***:***@artifactory.example/simple",
        ),
        # Nothing to redact — left byte-for-byte alone.
        ("https://artifactory.example/simple", "https://artifactory.example/simple"),
        ("nv-shared-pypi=https://artifactory.example/simple", "nv-shared-pypi=https://artifactory.example/simple"),
    ],
)
def test_redact_index_url_masks_embedded_credentials(index_url: str, expected: str) -> None:
    assert checks.redact_index_url(index_url) == expected


def test_doctor_never_prints_an_embedded_token(tmp_path: Path) -> None:
    """A token in the index URL must not reach terminal scrollback or pasted doctor output."""
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        index_url="https://user:AKCp8SUPERSECRET@artifactory.example/api/pypi/repo/simple",
    )
    _ok, detail = checks.venv_ok(cfg)
    assert "AKCp8SUPERSECRET" not in detail
    assert "artifactory.example" in detail  # host still visible for debugging


def test_doctor_reports_the_configured_index(tmp_path: Path) -> None:
    cfg = AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        index_url="https://registry.example/simple",
    )
    _ok, detail = checks.venv_ok(cfg)
    assert "registry.example" in detail


# ── run_subprocess: streaming + timeout ──────────────────────────────────


def test_run_subprocess_streams_instead_of_capturing() -> None:
    """A captured multi-minute `uv pip install` shows no progress and reads as hung."""
    recorded: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        recorded.update(kwargs)
        return subprocess.CompletedProcess(cmd, returncode=0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(provisioning.subprocess, "run", fake_run)
        provisioning.run_subprocess(["uv", "pip", "install", "agent-hardener"], "install agent-hardener")

    assert "capture_output" not in recorded and "stdout" not in recorded  # inherits the terminal
    assert recorded["timeout"] == provisioning.SUBPROCESS_TIMEOUT_SECONDS


def test_run_subprocess_exits_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=provisioning.SUBPROCESS_TIMEOUT_SECONDS)

    monkeypatch.setattr(provisioning.subprocess, "run", fake_run)
    with pytest.raises(typer.Exit) as excinfo:
        provisioning.run_subprocess(["uv", "pip", "install", "agent-hardener"], "install agent-hardener", timeout=1)
    assert excinfo.value.exit_code == 1


def test_run_subprocess_exits_on_nonzero_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provisioning.subprocess, "run", lambda cmd, **_: subprocess.CompletedProcess(cmd, returncode=2))
    with pytest.raises(typer.Exit):
        provisioning.run_subprocess(["uv", "venv"], "create venv")
