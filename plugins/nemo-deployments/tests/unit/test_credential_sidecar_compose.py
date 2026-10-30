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
    assert "credential-driver-socket:/run/nemo-credential-driver:ro" in gateway["volumes"]
    gateway_mount_sources = {
        mount.split(":", 1)[0] if isinstance(mount, str) else mount["source"] for mount in gateway["volumes"]
    }
    assert gateway_mount_sources.isdisjoint({"credential-driver-credentials", "credential-driver-state"})
    assert initialize["network_mode"] == "none"
    inputs = [mount for mount in initialize["volumes"] if isinstance(mount, dict) and mount["type"] == "bind"]
    assert inputs
    assert all(mount["read_only"] and not mount["bind"]["create_host_path"] for mount in inputs)
