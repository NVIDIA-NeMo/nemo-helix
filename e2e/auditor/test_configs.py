# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for AuditConfig CRUD, filtering, and sorting.

These tests exercise the SDK against the real platform without mocking the
entity store. Filter and sort parameters are not exposed by the SDK list()
method, so those tests use the underlying httpx client directly.
"""

import httpx
import pytest
from nemo_helix import NeMoHelix

from e2e.auditor.utils import minimal_audit_config, unique_name


def _list_raw(sdk: NeMoHelix, workspace: str, auditor_url: str, **params) -> dict:
    """GET /configs with arbitrary query params (filter, sort) via raw httpx."""
    resp = sdk.auditor._http_client.get(
        f"{auditor_url}/v2/workspaces/{workspace}/configs",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_config_create_and_get(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("cfg-cg")
    body = minimal_audit_config(description="create-and-get test")

    created = sdk.auditor.configs.create(workspace=workspace, name=name, **body)

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get test"
    assert created.plugins.probe_spec == "test.Test"

    # Note: id/created_at are not populated by the auditor SDK (raw httpx + model_validate
    # cannot set the private _id PrivateAttr). Use name as the stable cross-call identifier.
    retrieved = sdk.auditor.configs.get(workspace=workspace, name=name)
    assert retrieved.name == name
    assert retrieved.plugins.probe_spec == "test.Test"


def test_config_list_contains_created(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("cfg-list")
    body = minimal_audit_config(description="list test")

    sdk.auditor.configs.create(workspace=workspace, name=name, **body)

    page = sdk.auditor.configs.list(workspace=workspace, page_size=100)
    names = [item["name"] for item in page["data"]]
    assert name in names


def test_config_update(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("cfg-upd")
    body = minimal_audit_config(description="original description")

    sdk.auditor.configs.create(workspace=workspace, name=name, **body)

    updated_body = minimal_audit_config(description="updated description")
    updated_body["plugins"]["probe_spec"] = "dan.Dan"
    updated = sdk.auditor.configs.update(workspace=workspace, name=name, **updated_body)

    assert updated.name == name
    assert updated.description == "updated description"
    assert updated.plugins.probe_spec == "dan.Dan"

    retrieved = sdk.auditor.configs.get(workspace=workspace, name=name)
    assert retrieved.plugins.probe_spec == "dan.Dan"


def test_config_delete(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("cfg-del")
    sdk.auditor.configs.create(workspace=workspace, name=name, **minimal_audit_config())

    names_before = [item["name"] for item in sdk.auditor.configs.list(workspace=workspace, page_size=100)["data"]]
    assert name in names_before

    sdk.auditor.configs.delete(workspace=workspace, name=name)

    # Auditor SDK uses raw httpx, so errors surface as httpx.HTTPStatusError (not nemo_helix exceptions).
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        sdk.auditor.configs.get(workspace=workspace, name=name)
    assert exc_info.value.response.status_code == 404


def test_config_duplicate_name_returns_conflict(sdk: NeMoHelix, workspace: str) -> None:
    name = unique_name("cfg-dup")
    body = minimal_audit_config()

    sdk.auditor.configs.create(workspace=workspace, name=name, **body)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        sdk.auditor.configs.create(workspace=workspace, name=name, **body)
    assert exc_info.value.response.status_code == 409


def test_config_get_nonexistent_returns_404(sdk: NeMoHelix, workspace: str) -> None:
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        sdk.auditor.configs.get(workspace=workspace, name="does-not-exist-xyzzy")
    assert exc_info.value.response.status_code == 404


def test_config_filter_by_description(sdk: NeMoHelix, workspace: str, auditor_url: str) -> None:
    needle = unique_name("cfg-filter-needle")
    other = unique_name("cfg-filter-other")

    sdk.auditor.configs.create(workspace=workspace, name=needle, **minimal_audit_config(description="needle-desc"))
    sdk.auditor.configs.create(workspace=workspace, name=other, **minimal_audit_config(description="other-desc"))

    result = _list_raw(sdk, workspace, auditor_url, **{"filter[description]": "needle-desc"})
    names = [item["name"] for item in result["data"]]

    assert needle in names
    assert other not in names


def test_config_sort_descending(sdk: NeMoHelix, workspace: str, auditor_url: str) -> None:
    first = unique_name("cfg-sort-a")
    second = unique_name("cfg-sort-b")

    sdk.auditor.configs.create(workspace=workspace, name=first, **minimal_audit_config())
    sdk.auditor.configs.create(workspace=workspace, name=second, **minimal_audit_config())

    result = _list_raw(sdk, workspace, auditor_url, sort="-created_at", page_size=10)
    names = [item["name"] for item in result["data"]]

    assert names.index(second) < names.index(first), (
        f"Expected {second!r} (newer) before {first!r} (older) in sort=-created_at result; got order: {names}"
    )
