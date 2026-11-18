# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from nemo_helix_plugin.community_plugins.registry import ExternalPlugin, RegistryConflictError, RegistryStore
from pydantic import ValidationError


def plugin(name: str = "sample") -> ExternalPlugin:
    return ExternalPlugin.model_validate(
        {
            "name": name,
            "title": name,
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/status",
                    "access": "read",
                    "upstream_url": "https://example.org/status",
                }
            ],
        }
    )


def test_atomic_writers_preserve_independent_installations(tmp_path: Path) -> None:
    path = tmp_path / "plugins" / "installed.json"

    def install(index: int) -> None:
        RegistryStore(path).install(plugin(f"sample-{index}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(install, range(24)))
    assert {entry.name for entry in RegistryStore(path).load().plugins} == {f"sample-{index}" for index in range(24)}
    assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(RegistryConflictError):
        RegistryStore(path).install(plugin("sample-0"))
    assert len(RegistryStore(path).load().plugins) == 24


@pytest.mark.parametrize("path", ["/status", "/v2/workspaces/default/status", "/v2/workspaces/{other}/status"])
def test_secret_binding_requires_explicit_workspace_template(path: str) -> None:
    value = plugin().model_dump()
    value["credential"] = {"secret_name": "tavily-api-key"}
    value["endpoints"][0]["path"] = path
    with pytest.raises(ValidationError, match="Workspace-secret"):
        ExternalPlugin.model_validate(value)


@pytest.mark.parametrize(
    "path",
    [
        "/v2/workspaces/{workspace}",
        "/other/{workspace}/traces",
        "/v2/workspaces/alpha",
        "/v2/workspaces/-/traces",
    ],
)
def test_workspace_paths_cannot_silently_fall_back_to_system_authorization(path: str) -> None:
    value = plugin().model_dump()
    value["endpoints"][0]["path"] = path
    with pytest.raises(ValidationError):
        ExternalPlugin.model_validate(value)


def test_failed_state_change_does_not_truncate_registry(tmp_path: Path) -> None:
    store = RegistryStore(tmp_path / "installed.json")
    store.install(plugin())
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.set_enabled("missing", False)
    assert store.path.read_bytes() == before
    store.set_enabled("sample", False)
    assert not store.load().plugins[0].enabled
    store.uninstall("sample")
    assert store.load().plugins == ()
