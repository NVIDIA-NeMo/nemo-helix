# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for ScanConfig CRUD, filtering, and sorting.

These tests exercise the typed garak_plugin client against the real platform without
mocking the entity store. Bracketed filter parameters are not part of the typed
list() query params, so those tests use the underlying httpx client directly.
"""

import pytest
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.garak_plugin.client import GarakPluginClient
from nemo_helix_plugin.garak_plugin.types import CreateScanConfigRequest, UpdateScanConfigRequest

from e2e.garak_plugin.utils import minimal_scan_config, unique_name


def _list_raw(garak_plugin: GarakPluginClient, workspace: str, garak_plugin_url: str, **params) -> dict:
    """GET /configs with arbitrary query params (filter, sort) via raw httpx."""
    resp = garak_plugin._client.get(
        f"{garak_plugin_url}/v2/workspaces/{workspace}/configs",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_config_create_and_get(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("cfg-cg")
    body = minimal_scan_config(description="create-and-get test")

    created = garak_plugin.create_scan_config(
        workspace=workspace, body=CreateScanConfigRequest(name=name, **body)
    ).data()

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get test"
    assert created.plugins["probe_spec"] == "test.Test"

    retrieved = garak_plugin.get_scan_config(workspace=workspace, name=name).data()
    assert retrieved.name == name
    assert retrieved.plugins["probe_spec"] == "test.Test"


def test_config_list_contains_created(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("cfg-list")
    body = minimal_scan_config(description="list test")

    garak_plugin.create_scan_config(workspace=workspace, body=CreateScanConfigRequest(name=name, **body)).data()

    page = garak_plugin.list_scan_configs(workspace=workspace, query_params={"page_size": 100})
    names = [item.name for item in page.items()]
    assert name in names


def test_config_update(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("cfg-upd")
    body = minimal_scan_config(description="original description")

    garak_plugin.create_scan_config(workspace=workspace, body=CreateScanConfigRequest(name=name, **body)).data()

    updated_body = minimal_scan_config(description="updated description")
    updated_body["plugins"]["probe_spec"] = "dan.Dan"
    updated = garak_plugin.update_scan_config(
        workspace=workspace, name=name, body=UpdateScanConfigRequest(**updated_body)
    ).data()

    assert updated.name == name
    assert updated.description == "updated description"
    assert updated.plugins["probe_spec"] == "dan.Dan"

    retrieved = garak_plugin.get_scan_config(workspace=workspace, name=name).data()
    assert retrieved.plugins["probe_spec"] == "dan.Dan"


def test_config_delete(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("cfg-del")
    garak_plugin.create_scan_config(
        workspace=workspace, body=CreateScanConfigRequest(name=name, **minimal_scan_config())
    ).data()

    names_before = [
        item.name
        for item in garak_plugin.list_scan_configs(workspace=workspace, query_params={"page_size": 100}).items()
    ]
    assert name in names_before

    garak_plugin.delete_scan_config(workspace=workspace, name=name)

    with pytest.raises(NotFoundError):
        garak_plugin.get_scan_config(workspace=workspace, name=name)


def test_config_duplicate_name_returns_conflict(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("cfg-dup")
    body = minimal_scan_config()

    garak_plugin.create_scan_config(workspace=workspace, body=CreateScanConfigRequest(name=name, **body)).data()

    with pytest.raises(ConflictError):
        garak_plugin.create_scan_config(workspace=workspace, body=CreateScanConfigRequest(name=name, **body))


def test_config_get_nonexistent_returns_404(garak_plugin: GarakPluginClient, workspace: str) -> None:
    with pytest.raises(NotFoundError):
        garak_plugin.get_scan_config(workspace=workspace, name="does-not-exist-xyzzy")


def test_config_filter_by_description(garak_plugin: GarakPluginClient, workspace: str, garak_plugin_url: str) -> None:
    needle = unique_name("cfg-filter-needle")
    other = unique_name("cfg-filter-other")

    garak_plugin.create_scan_config(
        workspace=workspace,
        body=CreateScanConfigRequest(name=needle, **minimal_scan_config(description="needle-desc")),
    ).data()
    garak_plugin.create_scan_config(
        workspace=workspace, body=CreateScanConfigRequest(name=other, **minimal_scan_config(description="other-desc"))
    ).data()

    result = _list_raw(garak_plugin, workspace, garak_plugin_url, **{"filter[description]": "needle-desc"})
    names = [item["name"] for item in result["data"]]

    assert needle in names
    assert other not in names


def test_config_sort_descending(garak_plugin: GarakPluginClient, workspace: str, garak_plugin_url: str) -> None:
    first = unique_name("cfg-sort-a")
    second = unique_name("cfg-sort-b")

    garak_plugin.create_scan_config(
        workspace=workspace, body=CreateScanConfigRequest(name=first, **minimal_scan_config())
    ).data()
    garak_plugin.create_scan_config(
        workspace=workspace, body=CreateScanConfigRequest(name=second, **minimal_scan_config())
    ).data()

    result = _list_raw(garak_plugin, workspace, garak_plugin_url, sort="-created_at", page_size=10)
    names = [item["name"] for item in result["data"]]

    assert names.index(second) < names.index(first), (
        f"Expected {second!r} (newer) before {first!r} (older) in sort=-created_at result; got order: {names}"
    )
