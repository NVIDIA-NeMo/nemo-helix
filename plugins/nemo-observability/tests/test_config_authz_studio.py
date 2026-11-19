# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_observability_plugin.config import ObservabilityConfig, ObservabilityConnection
from nemo_observability_plugin.service import ObservabilityService
from nemo_observability_plugin.studio import get_studio_spec
from pydantic import ValidationError


def test_every_route_is_ruled():
    contrib, problems, _warnings = _derive_service_contribution(ObservabilityService())
    assert problems == []
    base = "/apis/observability/v2/workspaces/{workspace}/connections"
    assert contrib.endpoints[base]["get"].permissions == ["observability.connections.list"]
    assert contrib.endpoints[f"{base}/{{name}}/session"]["post"].permissions == ["observability.connections.read"]
    for path in ("/apis/observability/ui/{name}", "/apis/observability/ui/{name}/{path:path}"):
        assert set(contrib.endpoints[path]) == {"get", "head", "options", "post", "put", "patch", "delete"}
        assert all(b.permissions == ["observability.ui.read"] for b in contrib.endpoints[path].values())
    for methods in contrib.endpoints.values():
        for binding in methods.values():
            assert binding.deny is False


def test_studio_spec_frames_self_only():
    spec = get_studio_spec()
    assert spec.name == "observability"
    assert spec.frame_src == ("'self'",)
    assert spec.bundle_path is not None and spec.bundle_path.name == "index.js"


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Has-Caps"},
        {"name": "../x"},
        {"base_url": "http://host:5000/path"},
        {"base_url": "ftp://host"},
        {"provider": "langsmith"},
        {"secret_env": "lower"},
    ],
)
def test_connection_validation(overrides):
    fields = {"name": "mlflow", "provider": "mlflow", "base_url": "http://localhost:5000", **overrides}
    with pytest.raises(ValidationError):
        ObservabilityConnection(**fields)


def test_duplicate_connection_names_rejected():
    connection = ObservabilityConnection(name="a", provider="mlflow", base_url="http://localhost:5000")
    with pytest.raises(ValidationError, match="duplicate"):
        ObservabilityConfig(connections=[connection, connection])


@pytest.mark.parametrize("secret", ["", "too-short"])
def test_weak_session_secret_rejected(secret):
    with pytest.raises(ValidationError, match="session_secret"):
        ObservabilityConfig(session_secret=secret)


def test_strong_session_secret_accepted():
    assert ObservabilityConfig(session_secret="x" * 32).session_secret is not None
