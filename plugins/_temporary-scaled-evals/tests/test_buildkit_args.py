# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for buildctl argv assembly — notably the target-platform pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build import buildkit
from scaled_evals.api.settings import settings


def test_buildctl_args_pins_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "image_build_platform", "linux/amd64")
    args = buildkit._buildctl_args(Path("/ctx"), "reg/img:rev1", Path("/tmp/meta.json"))
    assert "--opt" in args
    assert args[args.index("--opt") + 1] == "platform=linux/amd64"


def test_buildctl_args_omits_platform_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "image_build_platform", "")
    args = buildkit._buildctl_args(Path("/ctx"), "reg/img:rev1", Path("/tmp/meta.json"))
    assert "--opt" not in args
    assert not any(a.startswith("platform=") for a in args)


def test_validate_tarball_size_rejects_oversized_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    monkeypatch.setattr(buildkit.artifacts, "object_size", lambda _key: 101)

    with pytest.raises(buildkit.BuildError, match="exceeds configured size limit"):
        buildkit._validate_tarball_size("task/rev/1/tarball.tar.gz")


def test_buildctl_env_uses_mounted_registry_auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auth_file = tmp_path / ".dockerconfigjson"
    document = {
        "auths": {
            "us-central1-docker.pkg.dev": {
                "username": "oauth2accesstoken",
                "password": "token",
                "auth": "encoded",
            }
        }
    }
    auth_file.write_text(json.dumps(document))

    monkeypatch.setattr(settings, "buildkit_addr", "tcp://buildkit:1234")
    monkeypatch.setattr(settings, "registry_username", "")
    monkeypatch.setattr(settings, "registry_password", "")
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    env = buildkit._buildctl_env(tmp_path / "work")

    assert env["BUILDKIT_HOST"] == "tcp://buildkit:1234"
    assert json.loads((Path(env["DOCKER_CONFIG"]) / "config.json").read_text()) == document


def test_buildctl_env_rejects_invalid_registry_auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auth_file = tmp_path / ".dockerconfigjson"
    auth_file.write_text("{}")

    monkeypatch.setattr(settings, "registry_username", "")
    monkeypatch.setattr(settings, "registry_password", "")
    monkeypatch.setattr(settings, "task_image_registry_auth_file", str(auth_file))

    with pytest.raises(buildkit.BuildError, match="contains no auth entries"):
        buildkit._buildctl_env(tmp_path / "work")
