# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The chart's default Envoy gateway must not expose the ext-authz callout to clients."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
EXT_AUTHZ_PREFIX = "/apis/auth/ext-authz"


def _http_connection_manager() -> dict:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "nemo-helix",
            str(HELM_DIR),
            "--set",
            "platformConfig.auth.enabled=true",
            "--show-only",
            "templates/proxy/envoy-configmap.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    config_map = yaml.safe_load(rendered.stdout)
    envoy = yaml.safe_load(config_map["data"]["envoy.yaml"])
    listener = envoy["static_resources"]["listeners"][0]
    return listener["filter_chains"][0]["filters"][0]["typed_config"]


def _envoy_routes() -> list[dict]:
    return _http_connection_manager()["route_config"]["virtual_hosts"][0]["routes"]


def test_default_envoy_normalizes_paths_before_route_matching() -> None:
    # Without this, /apis/auth/ext%2Dauthz bypasses the prefix fence but uvicorn decodes it.
    manager = _http_connection_manager()

    assert manager["normalize_path"] is True
    assert manager["merge_slashes"] is True
    assert manager["path_with_escaped_slashes_action"] == "UNESCAPE_AND_REDIRECT"


def test_default_envoy_answers_ext_authz_at_the_edge() -> None:
    routes = _envoy_routes()

    prefixes = [route["match"].get("prefix") for route in routes]
    assert EXT_AUTHZ_PREFIX in prefixes
    # Routes match in order, so the fence has to come before the catch-all.
    assert prefixes.index(EXT_AUTHZ_PREFIX) < prefixes.index("/")

    fence = routes[prefixes.index(EXT_AUTHZ_PREFIX)]
    assert fence["direct_response"] == {"status": 404}
    assert "route" not in fence


def test_default_envoy_still_forwards_other_paths_to_the_api() -> None:
    routes = _envoy_routes()

    catch_all = next(route for route in routes if route["match"].get("prefix") == "/")
    assert catch_all["route"]["cluster"] == "backend_cluster"
