# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.auth_idp]

ROOT = Path(__file__).parent.parent.parent.parent
AUTHENTIK_HELM_DIR = ROOT / "contrib" / "auth" / "authentik" / "helm"
AUTHENTIK_GATEWAY_ENVOY_CONFIG = ROOT / "contrib" / "auth" / "authentik" / "gateway" / "envoy.yaml"
AUTHENTIK_UMBRELLA_ENVOY_IMAGE = "docker.io/envoyproxy/envoy:v1.37.0"
AUTHENTIK_STATIC_GATEWAY_ENVOY_IMAGE = "envoyproxy/envoy:v1.36.2"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60
ENVOY_VALIDATE_TIMEOUT_SECONDS = 60
TLS_CERT_TIMEOUT_SECONDS = 30


def _require_envoy_validation_image(image: str) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is required to run envoy --mode validate")

    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
        text=True,
        timeout=ENVOY_VALIDATE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        pytest.skip(f"{image} is required locally to run envoy --mode validate")


def _helm_template(release_name: str, chart_dir: Path, *args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render Envoy configs")

    completed = subprocess.run(
        ["helm", "template", release_name, str(chart_dir), *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stderr
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _envoy_config_from_config_map(documents: list[dict]) -> dict:
    config_map = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap" and document["metadata"]["name"] == "nemo-helix-envoy"
    )
    return yaml.safe_load(config_map["data"]["envoy.yaml"])


def _write_dummy_tls_certificate(tmp_path: Path) -> Path:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required to create TLS material for envoy --mode validate")

    tls_dir = tmp_path / "tls"
    tls_dir.mkdir()

    completed = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tls_dir / "tls.key"),
            "-out",
            str(tls_dir / "tls.crt"),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=TLS_CERT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    (tls_dir / "tls.crt").chmod(0o644)
    (tls_dir / "tls.key").chmod(0o644)
    return tls_dir


def _validate_envoy_config(config: dict, tmp_path: Path, image: str) -> None:
    _require_envoy_validation_image(image)

    config_path = tmp_path / "envoy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    tls_dir = _write_dummy_tls_certificate(tmp_path)

    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{config_path}:/etc/envoy/envoy.yaml:ro",
            "-v",
            f"{tls_dir}:/etc/envoy/tls:ro",
            "-v",
            f"{tls_dir}:/etc/nhx/workload-token-tls:ro",
            image,
            "--mode",
            "validate",
            "-c",
            "/etc/envoy/envoy.yaml",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=ENVOY_VALIDATE_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_authentik_umbrella_envoy_config_validates_with_envoy(tmp_path: Path) -> None:
    documents = _helm_template(
        "authentik-demo",
        AUTHENTIK_HELM_DIR,
        "-n",
        "nemo-authentik",
        "--show-only",
        "charts/nemo-helix/templates/proxy/envoy-configmap.yaml",
    )

    _validate_envoy_config(_envoy_config_from_config_map(documents), tmp_path, AUTHENTIK_UMBRELLA_ENVOY_IMAGE)


def test_authentik_static_gateway_envoy_config_validates_with_envoy(tmp_path: Path) -> None:
    config = yaml.safe_load(AUTHENTIK_GATEWAY_ENVOY_CONFIG.read_text(encoding="utf-8"))

    _validate_envoy_config(config, tmp_path, AUTHENTIK_STATIC_GATEWAY_ENVOY_IMAGE)
