# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registry checks ``nemo agents package --publish`` runs before building."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from nemo_agents_plugin.container import registry
from nemo_agents_plugin.container.registry import (
    registry_has_credentials,
    registry_host,
    registry_login_hint,
    registry_scheme_error,
)

REGISTRY = "nvcr.io/my-org/my-team"


def test_scheme_error_names_the_bare_form() -> None:
    assert registry_scheme_error(f"https://{REGISTRY}/") == (
        f"--registry must not include a URL scheme; Docker image references have none. Use: --registry {REGISTRY}"
    )
    assert registry_scheme_error(REGISTRY) is None


@pytest.mark.parametrize(
    ("value", "host"),
    [
        (REGISTRY, "nvcr.io"),
        ("nvcr.io", "nvcr.io"),
        ("localhost:5000/agents", "localhost:5000"),
        ("myuser", "index.docker.io"),
        ("myuser/agents", "index.docker.io"),
    ],
)
def test_registry_host(value: str, host: str) -> None:
    assert registry_host(value) == host


@pytest.mark.parametrize(
    ("docker_config", "expected"),
    [
        ({"auths": {"nvcr.io": {}}}, True),
        ({"auths": {"https://nvcr.io": {"auth": "x"}}}, True),
        ({"auths": {"ghcr.io": {}}}, False),
        ({}, False),
        ({"credHelpers": {"nvcr.io": "gcloud"}}, True),
    ],
)
def test_credentials_from_docker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docker_config: dict[str, Any], expected: bool
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(docker_config))
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    assert registry_has_credentials(REGISTRY) is expected


@pytest.mark.parametrize(
    ("listed", "expected"), [({"https://nvcr.io": "$oauthtoken"}, True), ({}, False), (None, None)]
)
def test_credentials_from_credential_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listed: dict[str, str] | None, expected: bool | None
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"credsStore": "desktop", "auths": {"nvcr.io": {}}}))
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    hosts = None if listed is None else {registry._normalize_auth_key(key) for key in listed}
    with patch.object(registry, "_credential_store_hosts", return_value=hosts):
        assert registry_has_credentials(REGISTRY) is expected


def test_local_registry_needs_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    assert registry_has_credentials("localhost:5000/agents") is True


def test_login_hint_for_nvcr_uses_oauthtoken() -> None:
    assert "-u '$oauthtoken'" in registry_login_hint(REGISTRY)
    assert registry_login_hint("ghcr.io/me") == "Log in first: docker login ghcr.io"
