# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Fabric adapter ID registry."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric import adapters
from nemo_agents_plugin.fabric.adapters import (
    HARNESS_ADAPTER_IDS,
    UnknownAdapterError,
    accepted_adapter_ids,
    installed_adapter_ids,
    resolve_adapter_id,
    validate_harness_kinds,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_installed_ids_come_from_manifests_not_directory_names(installed_adapter_manifests: Path) -> None:
    assert installed_adapter_ids() == {
        "nvidia.helix.compass.analyst",
        "nvidia.fabric.nooa",
        "nvidia.fabric.nooa.bench-agent",
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "adapters/acme/acme.fabric-adapter.json",
        "adapters/acme.fabric-adapter.json",
        "acme.fabric-adapter.json",
        "vendor/acme/adapters/acme.fabric-adapter.json",
    ],
)
def test_manifests_are_found_at_any_depth(tmp_path: Path, relative_path: str) -> None:
    """Fabric discovers manifests anywhere below share/nemo-fabric, so the platform must too."""
    _write(tmp_path / relative_path, json.dumps({"adapter_id": "acme.custom"}))

    assert installed_adapter_ids(tmp_path) == {"acme.custom"}


def test_target_descriptors_are_not_adapters(tmp_path: Path) -> None:
    _write(tmp_path / "nooa/nooa.fabric-adapter.json", json.dumps({"adapter_id": "nvidia.fabric.nooa"}))
    _write(
        tmp_path / "nooa/targets/coding-agent.fabric-target.json",
        json.dumps({"adapter_id": "nvidia.fabric.nooa", "target_id": "nvidia.nooa.coding-agent"}),
    )
    _write(tmp_path / "nooa/coding-agent.fabric-target.json", json.dumps({"adapter_id": "nvidia.fabric.target"}))

    assert installed_adapter_ids(tmp_path) == {"nvidia.fabric.nooa"}


@pytest.mark.parametrize("content", ["{not json", "[]", "{}", json.dumps({"adapter_id": ""})])
def test_unreadable_manifest_is_skipped(tmp_path: Path, content: str) -> None:
    _write(tmp_path / "broken/broken.fabric-adapter.json", content)
    _write(tmp_path / "good/good.fabric-adapter.json", json.dumps({"adapter_id": "acme.good"}))

    assert installed_adapter_ids(tmp_path) == {"acme.good"}


def test_missing_root_has_no_installed_adapters(tmp_path: Path) -> None:
    assert installed_adapter_ids(tmp_path / "missing") == frozenset()
    assert accepted_adapter_ids(tmp_path / "missing") == frozenset(HARNESS_ADAPTER_IDS.values())


@pytest.mark.parametrize(("kind", "adapter_id"), sorted(HARNESS_ADAPTER_IDS.items()))
def test_stock_short_names_resolve_without_manifests(tmp_path: Path, kind: str, adapter_id: str) -> None:
    assert resolve_adapter_id(kind, tmp_path) == adapter_id
    assert resolve_adapter_id(adapter_id, tmp_path) == adapter_id


def test_adapter_ids_need_not_start_with_nvidia_fabric(tmp_path: Path) -> None:
    _write(tmp_path / "acme/acme.fabric-adapter.json", json.dumps({"adapter_id": "acme.custom.native"}))

    assert resolve_adapter_id("acme.custom.native", tmp_path) == "acme.custom.native"


@pytest.mark.usefixtures("installed_adapter_manifests")
def test_unknown_kind_lists_what_is_accepted() -> None:
    with pytest.raises(UnknownAdapterError) as excinfo:
        resolve_adapter_id("nvidia.fabric.claud")

    message = str(excinfo.value)
    assert "Unsupported harness kind 'nvidia.fabric.claud'" in message
    assert "claude" in message
    assert "nvidia.helix.compass.analyst" in message


@pytest.mark.usefixtures("installed_adapter_manifests")
def test_validate_harness_kinds_names_the_bad_harness() -> None:
    config = AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "agent",
            "default_harness": "main",
            "harnesses": {"main": {"kind": "codex"}, "spare": {"kind": "cladue"}},
        }
    )

    with pytest.raises(UnknownAdapterError, match="Harness 'spare': Unsupported harness kind 'cladue'"):
        validate_harness_kinds(config)


def test_adapter_python_environment_is_scanned_too(
    installed_adapter_manifests: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fabric discovers adapters from ADAPTER_PYTHON's environment, so the registry must as well."""
    data_root = tmp_path / "adapter-env"
    _write(
        data_root / "share/nemo-fabric/adapters/acme/acme.fabric-adapter.json",
        json.dumps({"adapter_id": "acme.isolated"}),
    )
    # Stands in for an interpreter: prints its environment's data path, as the real query does.
    interpreter = tmp_path / "python"
    interpreter.write_text(f"#!/bin/sh\necho {data_root}\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setenv(adapters.ADAPTER_PYTHON_ENV, str(interpreter))

    assert installed_adapter_ids() == {
        "acme.isolated",
        "nvidia.helix.compass.analyst",
        "nvidia.fabric.nooa",
        "nvidia.fabric.nooa.bench-agent",
    }


def test_unusable_adapter_python_is_skipped(
    installed_adapter_manifests: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(adapters.ADAPTER_PYTHON_ENV, str(tmp_path / "missing-python"))

    with caplog.at_level(logging.WARNING):
        assert "nvidia.helix.compass.analyst" in installed_adapter_ids()

    assert "Can't find Fabric adapters for ADAPTER_PYTHON" in caplog.text


def test_unknown_kind_message_does_not_suggest_a_custom_image(tmp_path: Path) -> None:
    """Naming an image only helps at deploy and job create, which add that hint themselves."""
    with pytest.raises(UnknownAdapterError) as error:
        resolve_adapter_id("acme.custom", tmp_path)

    assert adapters.CUSTOM_IMAGE_HINT not in str(error.value)
