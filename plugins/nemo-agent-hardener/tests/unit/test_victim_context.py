# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``victim-context`` job: staging a manifest's victim build context for an outside build."""

from __future__ import annotations

import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest
from _doubles import make_job_context
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import victim_context

# Stands in for `agent-hardener stage`: writes a context under its cwd and prints the directory, the way the
# real CLI does. FAKE_STAGE switches it to a structured failure or to printing a directory with no Dockerfile.
_FAKE_AGENT_HARDENER = """#!/bin/sh
set -e
case "$FAKE_STAGE" in
  fail)
    printf '{"category": "manifest", "message": "the config declares no victim build to stage"}' \
      > "$AGENT_HARDENER_ERROR_FILE"
    echo "boom" >&2
    exit 1 ;;
  empty)
    mkdir -p empty && echo "$PWD/empty"
    exit 0 ;;
esac
ctx="$PWD/.agent-hardener/builds/victim/context"
mkdir -p "$ctx/openshell-shims"
echo "FROM scratch" > "$ctx/Dockerfile"
printf '#!/bin/sh\\n' > "$ctx/openshell-shims/start.sh"
chmod 755 "$ctx/openshell-shims/start.sh"
echo "staging victim" >&2
echo "$ctx"
"""


@pytest.fixture
def plugin_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHardenerConfig:
    config = AgentHardenerConfig(
        venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak", operator_env_file=tmp_path / "absent.env"
    )
    config.agent_hardener_bin.parent.mkdir(parents=True)
    config.agent_hardener_bin.write_text(_FAKE_AGENT_HARDENER)
    config.agent_hardener_bin.chmod(0o755)
    config.garak_python.parent.mkdir(parents=True)
    config.garak_python.touch()
    monkeypatch.setattr(victim_context.AgentHardenerConfig, "get", lambda: config)
    monkeypatch.setattr(victim_context, "_materialize_manifest", lambda _sdk, _id, _ctx: str(tmp_path / "m.yaml"))
    return config


@pytest.mark.usefixtures("plugin_config")
def test_stages_a_docker_build_ready_archive(tmp_path: Path) -> None:
    saved: dict[str, Path] = {}
    ctx = make_job_context(tmp_path / "job", on_save=lambda name, path: saved.setdefault(name, Path(path)))

    result = victim_context.AgentHardenerVictimContextJob().run({"manifest_id": "finance"}, ctx=ctx)

    assert result["status"] == "completed", result
    archive = Path(result["context_archive"])
    assert saved[victim_context.CONTEXT_RESULT] == archive
    with tarfile.open(archive) as tar:
        members = {member.name: member for member in tar.getmembers()}
    # Rooted at the context, so `docker build - < archive` finds the Dockerfile; modes survive for the image.
    assert {"Dockerfile", "openshell-shims/start.sh"} <= members.keys()
    assert stat.S_IMODE(members["openshell-shims/start.sh"].mode) == 0o755
    assert "staging victim" in saved["stage-log"].read_text()


@pytest.mark.usefixtures("plugin_config")
def test_a_failed_stage_reports_agent_hardeners_own_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STAGE", "fail")

    result = victim_context.AgentHardenerVictimContextJob().run({"manifest_id": "finance"}, ctx=_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result["error"]["category"] == "manifest"
    assert result["error"]["message"] == "the config declares no victim build to stage"


@pytest.mark.usefixtures("plugin_config")
def test_a_context_without_a_dockerfile_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_STAGE", "empty")

    result = victim_context.AgentHardenerVictimContextJob().run({"manifest_id": "finance"}, ctx=_ctx(tmp_path))

    assert result["status"] == "failed"
    assert "did not print a build context with a Dockerfile" in result["error"]["message"]


def test_an_unprovisioned_host_is_told_to_run_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AgentHardenerConfig(venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak")
    monkeypatch.setattr(victim_context.AgentHardenerConfig, "get", lambda: config)

    result = victim_context.AgentHardenerVictimContextJob().run({"manifest_id": "finance"}, ctx=_ctx(tmp_path))

    assert result["error"]["category"] == "provisioning"


def _ctx(tmp_path: Path) -> Any:
    return make_job_context(tmp_path / "job", on_save=lambda *_a: None)
