# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell service routing requires the gateway TLS material in both Helix pods."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"


@pytest.mark.parametrize("fullname", [None, "custom-helix"])
def test_openshell_transport_mounts_and_executor_config(fullname: str | None) -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    args = ["--set-string", f"fullnameOverride={fullname}"] if fullname else []
    result = subprocess.run(
        [
            "helm",
            "template",
            "transport-test",
            str(HELM_DIR),
            "--namespace",
            "agents",
            "-f",
            str(HELM_DIR / "ci/25-controller-extra-volumes-openshell.yaml"),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    config_map = next(doc for doc in documents if doc["kind"] == "ConfigMap" and "config.yaml" in doc.get("data", {}))
    config = yaml.safe_load(config_map["data"]["config.yaml"])
    assert config["agents"]["deployments"]["openshell_executor"] == "openshell"
    assert config["deployments"]["default_executor"] == "local-k8s"
    executors = {entry["name"]: entry for entry in config["deployments"]["executors"]}
    assert executors["local-k8s"]["config"]["default_namespace"] == "agents"
    api = next(doc for doc in documents if doc["kind"] == "Service" and doc["metadata"]["name"].endswith("-api"))
    assert executors["openshell"]["config"]["platform_egress"]["host"] == f"{api['metadata']['name']}.agents.svc"
    for component in ("nhx-api", "nhx-core-controller"):
        deployment = next(
            doc
            for doc in documents
            if doc["kind"] == "Deployment" and doc["metadata"]["labels"]["app.kubernetes.io/component"] == component
        )
        pod = deployment["spec"]["template"]["spec"]
        volume = next(volume for volume in pod["volumes"] if volume["name"] == "openshell-client-tls")
        assert volume["secret"]["secretName"] == "openshell-client-tls"
        mounts = pod["containers"][0]["volumeMounts"]
        assert {"name": "openshell-client-tls", "mountPath": "/etc/openshell-tls/client", "readOnly": True} in mounts
