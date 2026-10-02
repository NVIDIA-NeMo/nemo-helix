# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for AuditConfig CRUD, filtering, and sorting.

These tests exercise the typed auditor client against the real platform without
mocking the entity store. Bracketed filter parameters are not part of the typed
list() query params, so those tests use the underlying httpx client directly.
"""

import pytest
from nemo_helix_plugin.auditor.client import AuditorClient
from nemo_helix_plugin.auditor.types import CreateAuditConfigRequest, UpdateAuditConfigRequest
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError

from e2e.auditor.utils import minimal_audit_config, unique_name


def _list_raw(auditor: AuditorClient, workspace: str, auditor_url: str, **params) -> dict:
    """GET /configs with arbitrary query params (filter, sort) via raw httpx."""
    resp = auditor._client.get(
        f"{auditor_url}/v2/workspaces/{workspace}/configs",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def test_config_create_and_get(auditor: AuditorClient, workspace: str) -> None:
    name = unique_name("cfg-cg")
    body = minimal_audit_config(description="create-and-get test")

    created = auditor.create_audit_config(workspace=workspace, body=CreateAuditConfigRequest(name=name, **body)).data()

    assert created.name == name
    assert created.workspace == workspace
    assert created.description == "create-and-get test"
    assert created.plugins["probe_spec"] == "test.Test"

    retrieved = auditor.get_audit_config(workspace=workspace, name=name).data()
    assert retrieved.name == name
    assert retrieved.plugins["probe_spec"] == "test.Test"


def test_config_list_contains_created(auditor: AuditorClient, workspace: str) -> None:
    name = unique_name("cfg-list")
    body = minimal_audit_config(description="list test")

    auditor.create_audit_config(workspace=workspace, body=CreateAuditConfigRequest(name=name, **body)).data()

    page = auditor.list_audit_configs(workspace=workspace, query_params={"page_size": 100})
    names = [item.name for item in page.items()]
    assert name in names


def test_config_update(auditor: AuditorClient, workspace: str) -> None:
    name = unique_name("cfg-upd")
    body = minimal_audit_config(description="original description")

    auditor.create_audit_config(workspace=workspace, body=CreateAuditConfigRequest(name=name, **body)).data()

    updated_body = minimal_audit_config(description="updated description")
    updated_body["plugins"]["probe_spec"] = "dan.Dan"
    updated = auditor.update_audit_config(
        workspace=workspace, name=name, body=UpdateAuditConfigRequest(**updated_body)
    ).data()

    assert updated.name == name
    assert updated.description == "updated description"
    assert updated.plugins["probe_spec"] == "dan.Dan"

    retrieved = auditor.get_audit_config(workspace=workspace, name=name).data()
    assert retrieved.plugins["probe_spec"] == "dan.Dan"


def test_config_delete(auditor: AuditorClient, workspace: str) -> None:
    name = unique_name("cfg-del")
    auditor.create_audit_config(
        workspace=workspace, body=CreateAuditConfigRequest(name=name, **minimal_audit_config())
    ).data()

    names_before = [
        item.name for item in auditor.list_audit_configs(workspace=workspace, query_params={"page_size": 100}).items()
    ]
    assert name in names_before

    auditor.delete_audit_config(workspace=workspace, name=name)

    with pytest.raises(NotFoundError):
        auditor.get_audit_config(workspace=workspace, name=name)


def test_config_duplicate_name_returns_conflict(auditor: AuditorClient, workspace: str) -> None:
    name = unique_name("cfg-dup")
    body = minimal_audit_config()

    auditor.create_audit_config(workspace=workspace, body=CreateAuditConfigRequest(name=name, **body)).data()

    with pytest.raises(ConflictError):
        auditor.create_audit_config(workspace=workspace, body=CreateAuditConfigRequest(name=name, **body))


def test_config_get_nonexistent_returns_404(auditor: AuditorClient, workspace: str) -> None:
    with pytest.raises(NotFoundError):
        auditor.get_audit_config(workspace=workspace, name="does-not-exist-xyzzy")


def test_config_filter_by_description(auditor: AuditorClient, workspace: str, auditor_url: str) -> None:
    needle = unique_name("cfg-filter-needle")
    other = unique_name("cfg-filter-other")

    auditor.create_audit_config(
        workspace=workspace,
        body=CreateAuditConfigRequest(name=needle, **minimal_audit_config(description="needle-desc")),
    ).data()
    auditor.create_audit_config(
        workspace=workspace, body=CreateAuditConfigRequest(name=other, **minimal_audit_config(description="other-desc"))
    ).data()

    result = _list_raw(auditor, workspace, auditor_url, **{"filter[description]": "needle-desc"})
    names = [item["name"] for item in result["data"]]

    assert needle in names
    assert other not in names


def test_config_sort_descending(auditor: AuditorClient, workspace: str, auditor_url: str) -> None:
    first = unique_name("cfg-sort-a")
    second = unique_name("cfg-sort-b")

    auditor.create_audit_config(
        workspace=workspace, body=CreateAuditConfigRequest(name=first, **minimal_audit_config())
    ).data()
    auditor.create_audit_config(
        workspace=workspace, body=CreateAuditConfigRequest(name=second, **minimal_audit_config())
    ).data()

    result = _list_raw(auditor, workspace, auditor_url, sort="-created_at", page_size=10)
    names = [item["name"] for item in result["data"]]

    assert names.index(second) < names.index(first), (
        f"Expected {second!r} (newer) before {first!r} (older) in sort=-created_at result; got order: {names}"
    )
