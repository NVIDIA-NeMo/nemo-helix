# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent-hardener's own operator inference credential (dotenv read/write/resolve)."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest
import yaml
from _doubles import make_job_context
from nemo_agent_hardener_plugin.cli import credentials
from nemo_agent_hardener_plugin.config import (
    INFERENCE_API_KEY_ENVVAR,
    AgentHardenerConfig,
    missing_secrets,
    read_env_file,
)
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob


def _config(tmp_path: Path) -> AgentHardenerConfig:
    return AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        operator_env_file=tmp_path / "operator.env",
    )


# ── read_env_file ────────────────────────────────────────────────────────


def test_read_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope.env") == {}


def test_read_env_file_parses_comments_blank_export_quotes(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "export FOO=bar",
                'QUOTED="hello world"',
                "SINGLE='it works'",
                "PLAIN=value",
            ]
        ),
        encoding="utf-8",
    )
    assert read_env_file(path) == {
        "FOO": "bar",
        "QUOTED": "hello world",
        "SINGLE": "it works",
        "PLAIN": "value",
    }


# ── _resolve_inference_key ───────────────────────────────────────────────


def test_resolve_inference_key_prefers_secrets_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(INFERENCE_API_KEY_ENVVAR, "env-value")  # store must still win
    secret = types.SimpleNamespace(value="secret-value")
    fake_secrets = types.SimpleNamespace(
        access_secret=lambda name, workspace: types.SimpleNamespace(data=lambda: secret)
    )
    monkeypatch.setattr(credentials, "make_sdk", lambda base: types.SimpleNamespace())
    monkeypatch.setattr(credentials, "client_from_platform", lambda sdk, cls: fake_secrets)

    value, source = credentials.resolve_inference_key(_config(tmp_path))
    assert value == "secret-value"
    assert "agent-hardener-inference-key" in source


def test_resolve_inference_key_falls_back_to_env_when_store_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INFERENCE_API_KEY_ENVVAR, "env-value")

    def _raise(base: str) -> None:
        raise RuntimeError("platform unreachable")

    monkeypatch.setattr(credentials, "make_sdk", _raise)
    monkeypatch.setattr(credentials.sys.stdin, "isatty", lambda: False)

    value, source = credentials.resolve_inference_key(_config(tmp_path))
    assert (value, source) == ("env-value", "environment")


def test_resolve_inference_key_returns_none_when_platform_down_and_non_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(INFERENCE_API_KEY_ENVVAR, raising=False)

    def _raise(base: str) -> None:
        raise RuntimeError("platform unreachable")

    monkeypatch.setattr(credentials, "make_sdk", _raise)
    monkeypatch.setattr(credentials.sys.stdin, "isatty", lambda: False)

    value, source = credentials.resolve_inference_key(_config(tmp_path))
    assert (value, source) == (None, "unresolved")


# ── _write_operator_env ──────────────────────────────────────────────────


def test_write_operator_env_sets_mode_and_preserves_existing_keys(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.operator_env_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.operator_env_file.write_text("OTHER=keep-me\n", encoding="utf-8")

    credentials.write_operator_env(cfg, "new-key")

    values = read_env_file(cfg.operator_env_file)
    assert values == {"OTHER": "keep-me", INFERENCE_API_KEY_ENVVAR: "new-key"}
    assert (cfg.operator_env_file.stat().st_mode & 0o777) == 0o600


def test_materialized_victim_env_is_never_world_readable(tmp_path: Path) -> None:
    """The victim dotenv holds provider creds too — same 0600-at-creation rule as the operator one."""
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text(
        yaml.safe_dump({"agent": {"name": "v", "secrets": ["GITHUB_TOKEN"]}}),
        encoding="utf-8",
    )
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    previous = os.umask(0o000)
    try:
        written = _common.materialize_victim_env_file(str(manifest), {"GITHUB_TOKEN": "ghp_x"}, dest_dir)
    finally:
        os.umask(previous)

    assert written is not None
    path = Path(written)
    assert (path.stat().st_mode & 0o777) == 0o600
    assert read_env_file(path) == {"GITHUB_TOKEN": "ghp_x"}


def test_write_operator_env_never_creates_a_world_readable_file(tmp_path: Path) -> None:
    """A fresh dotenv must be 0600 at creation — not chmod'd after a default-umask write."""
    cfg = _config(tmp_path)
    previous = os.umask(0o000)  # the permissive umask that made the old chmod-after-write racy
    try:
        credentials.write_operator_env(cfg, "brand-new-key")
    finally:
        os.umask(previous)

    assert (cfg.operator_env_file.stat().st_mode & 0o777) == 0o600
    assert (cfg.operator_env_file.parent.stat().st_mode & 0o077) == 0
    assert read_env_file(cfg.operator_env_file)[INFERENCE_API_KEY_ENVVAR] == "brand-new-key"


# ── run job env injection ────────────────────────────────────────────────


def test_run_job_injects_operator_env_when_shell_env_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin_config = _config(tmp_path)
    plugin_config.venv_path.mkdir(parents=True)
    plugin_config.agent_hardener_bin.parent.mkdir(parents=True, exist_ok=True)
    plugin_config.agent_hardener_bin.touch()
    plugin_config.garak_python.parent.mkdir(parents=True, exist_ok=True)
    plugin_config.garak_python.touch()
    plugin_config.operator_env_file.write_text(f"{INFERENCE_API_KEY_ENVVAR}=from-dotenv\n", encoding="utf-8")

    monkeypatch.delenv(INFERENCE_API_KEY_ENVVAR, raising=False)
    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs.run.AgentHardenerConfig.get", lambda: plugin_config)
    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs._common.sys.stdin.isatty", lambda: False)

    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs._common.subprocess.run", fake_run)

    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: calc\n  port: 1\n", encoding="utf-8")
    ctx = make_job_context(tmp_path, job_id="", on_save=lambda *_a, **_k: types.SimpleNamespace(model_dump=lambda: {}))

    job = AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)
    job.run({"config": str(manifest)}, ctx=ctx, sdk=None)

    assert captured["env"][INFERENCE_API_KEY_ENVVAR] == "from-dotenv"


def test_run_job_does_not_override_explicit_shell_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin_config = _config(tmp_path)
    plugin_config.venv_path.mkdir(parents=True)
    plugin_config.agent_hardener_bin.parent.mkdir(parents=True, exist_ok=True)
    plugin_config.agent_hardener_bin.touch()
    plugin_config.garak_python.parent.mkdir(parents=True, exist_ok=True)
    plugin_config.garak_python.touch()
    plugin_config.operator_env_file.write_text(f"{INFERENCE_API_KEY_ENVVAR}=from-dotenv\n", encoding="utf-8")

    shell_value = "from-shell-sentinel"
    monkeypatch.setenv(INFERENCE_API_KEY_ENVVAR, shell_value)
    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs.run.AgentHardenerConfig.get", lambda: plugin_config)
    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs._common.sys.stdin.isatty", lambda: False)

    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("nemo_agent_hardener_plugin.jobs._common.subprocess.run", fake_run)

    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent:\n  name: calc\n  port: 1\n", encoding="utf-8")
    ctx = make_job_context(tmp_path, job_id="", on_save=lambda *_a, **_k: types.SimpleNamespace(model_dump=lambda: {}))

    job = AgentHardenerRunJob()
    monkeypatch.setattr(job, "report_progress", lambda *a, **k: None)
    job.run({"config": str(manifest)}, ctx=ctx, sdk=None)

    assert captured["env"][INFERENCE_API_KEY_ENVVAR] == shell_value
    assert os.environ[INFERENCE_API_KEY_ENVVAR] == shell_value


# ── missing_secrets ──────────────────────────────────────────────────────


def _manifest(tmp_path: Path, secrets: list[str], secrets_file: str = ".env") -> Path:
    path = tmp_path / "agent-hardener.yaml"
    path.write_text(
        yaml.safe_dump({"agent": {"name": "victim", "secrets": secrets, "secrets_file": secrets_file}}),
        encoding="utf-8",
    )
    return path


def test_missing_secrets_empty_when_none_declared(tmp_path: Path) -> None:
    assert missing_secrets(_manifest(tmp_path, []), environ={}) == []


def test_missing_secrets_reports_unresolvable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["GITHUB_TOKEN", "OTHER_KEY"])
    assert missing_secrets(manifest, environ={}) == ["GITHUB_TOKEN", "OTHER_KEY"]


def test_missing_secrets_resolved_from_environ_env_file_and_secrets_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["FROM_ENV", "FROM_FILE", "FROM_SECRETS_FILE"])
    (tmp_path / ".env").write_text("FROM_SECRETS_FILE=x\n", encoding="utf-8")
    extra = tmp_path / "creds.env"
    extra.write_text("FROM_FILE=y\n", encoding="utf-8")
    missing = missing_secrets(manifest, env_files=[extra], environ={"FROM_ENV": "z"})
    assert missing == []


def test_missing_secrets_returns_empty_on_unreadable_manifest(tmp_path: Path) -> None:
    assert missing_secrets(tmp_path / "nope.yaml", environ={}) == []


def test_missing_secrets_treats_a_blank_value_as_missing(tmp_path: Path) -> None:
    """`export KEY=""` used to satisfy the gate, then fail deep in the run as a provider auth error."""
    manifest = _manifest(tmp_path, ["FROM_ENV", "FROM_FILE", "FROM_SECRETS_FILE"])
    (tmp_path / ".env").write_text("FROM_SECRETS_FILE=\n", encoding="utf-8")
    extra = tmp_path / "creds.env"
    extra.write_text('FROM_FILE=""\n', encoding="utf-8")

    missing = missing_secrets(manifest, env_files=[extra], environ={"FROM_ENV": "   "})

    assert missing == ["FROM_ENV", "FROM_FILE", "FROM_SECRETS_FILE"]


def test_missing_secrets_still_accepts_real_values(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["REAL"])
    assert missing_secrets(manifest, environ={"REAL": "sk-abc"}) == []
