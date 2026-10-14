# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"


@pytest.mark.parametrize("fullname", [None, "custom-platform"])
def test_openshell_overlay_preserves_kubernetes_and_uses_actual_api_service(fullname: str | None) -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    argv = [
        "helm",
        "template",
        "sandbox-test",
        str(HELM_DIR),
        "--namespace",
        "platform-test",
        "--values",
        str(HELM_DIR / "examples" / "openshell" / "values.yaml"),
    ]
    if fullname:
        argv.extend(["--set", f"fullnameOverride={fullname}"])
    result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60)
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    config_map = next(doc for doc in documents if doc["kind"] == "ConfigMap" and "config.yaml" in doc.get("data", {}))
    config = yaml.safe_load(config_map["data"]["config.yaml"])
    executors = {item["name"]: item for item in config["deployments"]["executors"]}
    assert config["deployments"]["default_executor"] == "local-k8s"
    assert executors["local-k8s"]["backend"] == "k8s"
    assert executors["local-k8s"]["config"]["default_namespace"] == "platform-test"
    assert config["agents"]["deployments"]["openshell_executor"] == "openshell"
    assert executors["openshell"]["backend"] == "openshell"
    api_service = next(
        doc for doc in documents if doc["kind"] == "Service" and doc["metadata"]["name"].endswith("-api")
    )
    expected_host = f"{api_service['metadata']['name']}.platform-test.svc.cluster.local"
    assert executors["openshell"]["config"]["platform_egress"] == {"host": expected_host, "port": 8080}
    assert not any(item.get("backend") == "openshell" for item in config["jobs"].get("executors", []))
