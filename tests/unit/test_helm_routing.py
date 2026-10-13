# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
EXPECTED_PATHS = {"/", "/apis", "/studio", "/cluster-info", "/status", "/openapi.json", "/health/ready"}


def _render_route(values_file: Path, kind: str) -> dict:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    result = subprocess.run(
        ["helm", "template", "routing-test", str(HELM_DIR), "--no-hooks", "-f", str(values_file)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return next(document for document in yaml.safe_load_all(result.stdout) if document and document["kind"] == kind)


@pytest.mark.parametrize(
    ("fixture", "kind"),
    [
        ("16-enable-ingress.yaml", "Ingress"),
        ("17-ingress-default-host.yaml", "Ingress"),
        ("18-enable-httproute.yaml", "HTTPRoute"),
    ],
)
@pytest.mark.parametrize(
    ("auth_enabled", "envoy_enabled", "backend", "port"),
    [(False, True, "api", 8081), (True, True, "envoy", 8082), (True, False, "api", 8081)],
)
def test_default_routes_include_platform_endpoints(
    tmp_path: Path, fixture: str, kind: str, auth_enabled: bool, envoy_enabled: bool, backend: str, port: int
) -> None:
    values = yaml.safe_load((HELM_DIR / "ci" / fixture).read_text())
    values.update(
        {
            "fullnameOverride": "custom-platform",
            "api": {"service": {"port": 8081}},
            "envoyProxy": {"enabled": envoy_enabled, "service": {"port": 8082}},
            "platformConfig": {"auth": {"enabled": auth_enabled}},
        }
    )
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values))
    route = _render_route(values_file, kind)
    rules = route["spec"]["rules"]
    assert len(rules) == 1
    expected_backend = {"name": f"custom-platform-{backend}", "port": {"number": port}}
    if kind == "Ingress":
        if fixture == "17-ingress-default-host.yaml":
            assert rules[0]["host"] == "nhx.example.com"
        else:
            assert "host" not in rules[0]
        paths = rules[0]["http"]["paths"]
        assert {path["path"] for path in paths} == EXPECTED_PATHS
        for path in paths:
            assert path["backend"]["service"] == expected_backend
            assert path["pathType"] == ("Prefix" if path["path"] in {"/apis", "/studio"} else "Exact")
    else:
        paths = rules[0]["matches"]
        assert {match["path"]["value"] for match in paths} == EXPECTED_PATHS
        assert rules[0]["backendRefs"] == [{"name": f"custom-platform-{backend}", "port": port}]
        for match in paths:
            path = match["path"]
            assert path["type"] == ("PathPrefix" if path["value"] in {"/apis", "/studio"} else "Exact")


@pytest.mark.parametrize("kind", ["Ingress", "HTTPRoute"])
def test_custom_routes_are_not_extended(tmp_path: Path, kind: str) -> None:
    backend = {"service": "custom-service", "port": "9000"}
    if kind == "Ingress":
        values = {
            "ingress": {
                "enabled": True,
                "defaultHost": "custom.example.com",
                "hosts": [{"paths": [{"path": "/custom", "pathType": "Exact", **backend}]}],
            }
        }
    else:
        values = {
            "httpRoute": {
                "enabled": True,
                "parentRefs": [{"name": "custom-gateway"}],
                "pathRules": [{"matches": [{"path": "/custom", "type": "Exact"}], "backends": [backend]}],
            }
        }
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values))
    route = _render_route(values_file, kind)
    rules = route["spec"]["rules"]
    assert len(rules) == 1
    if kind == "Ingress":
        assert rules[0]["http"]["paths"] == [
            {
                "path": "/custom",
                "pathType": "Exact",
                "backend": {"service": {"name": "custom-service", "port": {"number": 9000}}},
            }
        ]
    else:
        assert rules[0]["matches"] == [{"path": {"type": "Exact", "value": "/custom"}}]
        assert rules[0]["backendRefs"] == [{"name": "custom-service", "port": 9000}]
