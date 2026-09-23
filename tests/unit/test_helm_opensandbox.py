# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60


def _helm_template(*args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _platform_config(documents: list[dict]) -> dict:
    config_map = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap" and "config.yaml" in document.get("data", {})
    )
    return yaml.safe_load(config_map["data"]["config.yaml"])


def _has_opensandbox_deployment(documents: list[dict]) -> bool:
    return any(
        document.get("kind") in {"Deployment", "StatefulSet"}
        and "opensandbox" in document.get("metadata", {}).get("name", "")
        for document in documents
    )


def test_default_sandbox_cluster_capable_is_false_and_does_not_install_opensandbox() -> None:
    documents = _helm_template()
    config = _platform_config(documents)

    assert config["platform"]["sandbox_cluster_capable"] is False
    assert _has_opensandbox_deployment(documents) is False


def test_sandbox_cluster_capable_wires_domain_protocol_and_secret() -> None:
    documents = _helm_template(
        "--set",
        "sandboxClusterCapable=true",
        "--set",
        "opensandbox.domain=opensandbox-server-kata.opensandbox-system.svc.cluster.local",
        "--set",
        "opensandbox.protocol=http",
        "--set",
        "opensandbox.apiKeySecret=opensandbox-server-api-key",
        "--set",
        "opensandbox.apiKeySecretKey=api-key",
    )
    config = _platform_config(documents)
    platform = config["platform"]

    assert platform["sandbox_cluster_capable"] is True
    assert platform["sandbox_server_domain"] == "opensandbox-server-kata.opensandbox-system.svc.cluster.local"
    assert platform["sandbox_server_protocol"] == "http"
    assert platform["sandbox_api_key_secret"] == "opensandbox-server-api-key"
    assert platform["sandbox_api_key_secret_key"] == "api-key"
    assert "sandbox_cluster_capable" not in config.get("rl", {})
    assert _has_opensandbox_deployment(documents) is False
