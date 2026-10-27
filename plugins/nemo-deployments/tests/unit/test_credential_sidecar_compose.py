# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the opt-in sidecar's startup ordering and credential mount boundaries."""

import tomllib
from pathlib import Path

import yaml

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "openshell"


def test_credential_gateway_config_preserves_base_settings() -> None:
    base = tomllib.loads((EXAMPLE / "gateway.toml").read_text())
    credentials = tomllib.loads((EXAMPLE / "gateway-credentials.toml").read_text())
    assert credentials["openshell"]["gateway"].pop("credential_drivers") == ["nemo"]
    assert credentials["openshell"].pop("credential_drivers") == {
        "nemo": {"transport": "uds", "socket_path": "/run/nemo-credential-driver/private/driver.sock"}
    }
    assert credentials == base


def test_sidecar_startup_and_mount_boundaries() -> None:
    overlay = yaml.safe_load((EXAMPLE / "docker-compose.credentials.yml").read_text())
    services = overlay["services"]
    gateway, initialize, driver = (services[name] for name in ("gateway", "credential-init", "credential-driver"))
    assert gateway["depends_on"]["credential-driver"]["condition"] == "service_healthy"
    assert driver["depends_on"]["credential-init"]["condition"] == "service_completed_successfully"
    assert initialize["image"] == driver["image"]
    assert initialize["healthcheck"] == {"disable": True}  # Override the API image's HTTP probe.
    assert "healthcheck" in driver["healthcheck"]["test"]
    assert "nemo_deployments_plugin.backends.openshell.credential_driver.server" in driver["entrypoint"]
    assert driver["cap_drop"] == ["ALL"]
    assert driver["init"] is True
    assert "cap_add" not in driver
    assert driver["read_only"]
    assert not driver.get("ports")
    assert not any("docker.sock" in mount for mount in driver["volumes"])
    assert "credential-driver-credentials:/run/nemo-driver-credentials:ro" in driver["volumes"]
    assert gateway["volumes"][-1] == "credential-driver-socket:/run/nemo-credential-driver:ro"
    assert len(gateway["volumes"]) == 2  # Config override and socket; no driver credentials/state.
    assert initialize["network_mode"] == "none"
    assert initialize["volumes"][0]["read_only"]
    assert not initialize["volumes"][0]["bind"]["create_host_path"]
