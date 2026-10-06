# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for ScanTarget CRUD and filtering."""

import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.garak_plugin.client import GarakPluginClient
from nemo_helix_plugin.garak_plugin.types import CreateScanTargetRequest, UpdateScanTargetRequest

from e2e.garak_plugin.utils import minimal_scan_target, unique_name


def _list_raw(garak_plugin: GarakPluginClient, workspace: str, garak_plugin_url: str, **params) -> dict:
    resp = garak_plugin._client.get(
        f"{garak_plugin_url}/v2/workspaces/{workspace}/targets",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_target_create_and_get(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("tgt-cg")
    body = minimal_scan_target(description="create-and-get target", type="nim", model="meta/llama-3.1-8b-instruct")

    created = garak_plugin.create_scan_target(
        workspace=workspace, body=CreateScanTargetRequest(name=name, **body)
    ).data()

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get target"
    assert created.type == "nim"
    assert created.model == "meta/llama-3.1-8b-instruct"

    retrieved = garak_plugin.get_scan_target(workspace=workspace, name=name).data()
    assert retrieved.name == name
    assert retrieved.type == "nim"
    assert retrieved.model == "meta/llama-3.1-8b-instruct"


def test_target_list_contains_created(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("tgt-list")

    garak_plugin.create_scan_target(
        workspace=workspace, body=CreateScanTargetRequest(name=name, **minimal_scan_target())
    ).data()

    page = garak_plugin.list_scan_targets(workspace=workspace, query_params={"page_size": 100})
    names = [item.name for item in page.items()]
    assert name in names


def test_target_update(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("tgt-upd")

    garak_plugin.create_scan_target(
        workspace=workspace,
        body=CreateScanTargetRequest(name=name, **minimal_scan_target(description="original", model="gpt-4o-mini")),
    ).data()

    updated = garak_plugin.update_scan_target(
        workspace=workspace,
        name=name,
        body=UpdateScanTargetRequest(**minimal_scan_target(description="updated", model="gpt-4o")),
    ).data()

    assert updated.name == name
    assert updated.description == "updated"
    assert updated.model == "gpt-4o"


def test_target_delete(garak_plugin: GarakPluginClient, workspace: str) -> None:
    name = unique_name("tgt-del")
    garak_plugin.create_scan_target(
        workspace=workspace, body=CreateScanTargetRequest(name=name, **minimal_scan_target())
    ).data()

    garak_plugin.delete_scan_target(workspace=workspace, name=name)

    with pytest.raises(NotFoundError):
        garak_plugin.get_scan_target(workspace=workspace, name=name)


def test_target_filter_by_type(garak_plugin: GarakPluginClient, workspace: str, garak_plugin_url: str) -> None:
    nim_name = unique_name("tgt-nim")
    openai_name = unique_name("tgt-oai")

    garak_plugin.create_scan_target(
        workspace=workspace,
        body=CreateScanTargetRequest(
            name=nim_name, **minimal_scan_target(type="nim", model="meta/llama-3.1-8b-instruct")
        ),
    ).data()
    garak_plugin.create_scan_target(
        workspace=workspace,
        body=CreateScanTargetRequest(name=openai_name, **minimal_scan_target(type="openai", model="gpt-4o-mini")),
    ).data()

    result = _list_raw(garak_plugin, workspace, garak_plugin_url, **{"filter[type]": "nim"})
    names = [item["name"] for item in result["data"]]

    assert nim_name in names
    assert openai_name not in names
