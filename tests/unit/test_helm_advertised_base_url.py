# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The API pod and the controllers must derive one client-facing origin.

Under the embedded PDP the API pod's NHX_BASE_URL is loopback so embedded auth
can call back into the local process. Platform-minted workload tokens take
their default issuer, and discovery/token/JWKS URLs, from the advertised origin
(platform.advertised_base_url, falling back to NHX_BASE_URL). Without an
explicit advertised origin the API pod would advertise http://localhost:<port>
while the controllers advertise platform.base_url, so tokens minted by
core-controller are rejected by the API (NVBug 7005096).
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
EDGE = "https://nemo-helix-envoy.nhx.svc:8080"


def _render_envs(tmp_path: Path, values: dict) -> dict[str, dict[str, str]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        ["helm", "template", "customer", str(HELM_DIR), "--no-hooks", "-f", str(values_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    envs: dict[str, dict[str, str]] = {}
    for doc in yaml.safe_load_all(result.stdout):
        if not doc or doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        for component in ("api", "core-controller"):
            if name == f"customer-nemo-helix-{component}":
                container = doc["spec"]["template"]["spec"]["containers"][0]
                envs[component] = {e["name"]: e.get("value") for e in container.get("env", []) if "value" in e}
    assert set(envs) == {"api", "core-controller"}, sorted(envs)
    return envs


def _workload_auth_values(platform: dict, pdp: str = "embedded") -> dict:
    return {
        "platformConfig": {
            "platform": platform,
            "auth": {
                "enabled": True,
                "policy_decision_point_provider": pdp,
                "oidc": {
                    "issuer": "https://idp.example.com/realms/nhx",
                    "public_client": {"client_id": "nhx-cli"},
                    "workload": {"client_id": "nhx-workload"},
                },
            },
        }
    }


def _advertised(env: dict[str, str]) -> str:
    """Mirror PlatformConfig.effective_advertised_base_url for rendered env."""
    return (env.get("NHX_ADVERTISED_BASE_URL") or env["NHX_BASE_URL"]).rstrip("/")


def test_embedded_pdp_api_advertises_platform_base_url(tmp_path: Path) -> None:
    envs = _render_envs(tmp_path, _workload_auth_values({"base_url": EDGE}))

    assert envs["api"]["NHX_BASE_URL"].startswith("http://localhost:")
    assert envs["api"]["NHX_ADVERTISED_BASE_URL"] == EDGE
    assert _advertised(envs["api"]) == _advertised(envs["core-controller"]) == EDGE


def test_embedded_pdp_api_prefers_explicit_advertised_base_url(tmp_path: Path) -> None:
    envs = _render_envs(
        tmp_path,
        _workload_auth_values({"base_url": EDGE, "advertised_base_url": "https://nhx.example.com/"}),
    )

    assert envs["api"]["NHX_ADVERTISED_BASE_URL"] == "https://nhx.example.com"


def test_non_embedded_pdp_api_does_not_override_advertised_base_url(tmp_path: Path) -> None:
    envs = _render_envs(tmp_path, _workload_auth_values({"base_url": EDGE}, pdp="external"))

    assert "NHX_ADVERTISED_BASE_URL" not in envs["api"]
    assert not envs["api"]["NHX_BASE_URL"].startswith("http://localhost:")


def test_default_chart_api_env_is_unchanged(tmp_path: Path) -> None:
    envs = _render_envs(tmp_path, {})

    assert "NHX_ADVERTISED_BASE_URL" not in envs["api"]
