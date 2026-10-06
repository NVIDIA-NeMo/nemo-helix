# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-release compatibility shims in ``garak_plugin._legacy``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from garak_plugin import _legacy
from garak_plugin.jobs.scan import GARAK_PYTHON_ENVVAR, _resolve_garak_python
from nemo_helix_plugin.client.errors import ConflictError


def _entity(workspace: str, name: str, data: dict[str, Any] | None = None, db_version: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=workspace, name=name, project=None, data=data or {"k": name}, db_version=db_version
    )


class _FakeEntities:
    """In-memory stand-in for AsyncEntitiesClient keyed by (entity_type, workspace, name)."""

    def __init__(self, rows: dict[str, list[SimpleNamespace]], conflicts: set[tuple[str, str, str]] | None = None):
        self.rows = rows
        self.conflicts = conflicts or set()
        self.created: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str, str]] = []
        self.updated: list[tuple[str, str, dict[str, Any]]] = []
        self.list_filters: list[str | None] = []

    async def list_entities(self, *, workspace: str, entity_type: str, query_params: dict[str, Any]):
        self.list_filters.append(query_params.get("filter"))
        items = list(self.rows.get(entity_type, []))
        if entity_type == _legacy.JOB_ENTITY_TYPE:
            items = [e for e in items if e.data.get("source") == _legacy.LEGACY_JOB_SOURCE]
        return SimpleNamespace(page=lambda: SimpleNamespace(items=items))

    async def create_entity(self, *, workspace: str, entity_type: str, body: Any):
        key = (entity_type, workspace, body.name)
        if key in self.conflicts:
            raise ConflictError(httpx.Response(409, json={"detail": "exists"}))
        self.created.append(key)

    async def delete_entity_by_name(self, *, workspace: str, entity_type: str, name: str):
        self.deleted.append((entity_type, workspace, name))
        self.rows[entity_type] = [e for e in self.rows[entity_type] if (e.workspace, e.name) != (workspace, name)]

    async def update_entity_by_name(self, *, workspace: str, entity_type: str, name: str, body: Any):
        self.updated.append((workspace, name, body.data))
        for e in self.rows[entity_type]:
            if (e.workspace, e.name) == (workspace, name):
                e.data = body.data


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(rows: dict[str, list[SimpleNamespace]], conflicts: set[tuple[str, str, str]] | None = None):
        client = _FakeEntities(rows, conflicts)
        monkeypatch.setattr(_legacy, "client_from_platform", lambda sdk, cls: client)
        return client

    return install


@pytest.mark.asyncio
async def test_entities_are_recreated_under_new_types_then_deleted(fake):
    client = fake(
        {
            "auditor_audit_config": [_entity("default", "cfg-a"), _entity("team", "cfg-b")],
            "auditor_audit_target": [_entity("default", "tgt-a")],
        }
    )

    moved = await _legacy.migrate_legacy_entities(object())  # type: ignore[arg-type]

    assert moved == 3
    assert sorted(client.created) == [
        ("garak_plugin_scan_config", "default", "cfg-a"),
        ("garak_plugin_scan_config", "team", "cfg-b"),
        ("garak_plugin_scan_target", "default", "tgt-a"),
    ]
    assert client.rows["auditor_audit_config"] == []
    assert client.rows["auditor_audit_target"] == []


@pytest.mark.asyncio
async def test_entity_migration_is_idempotent_and_tolerates_conflicts(fake):
    client = fake(
        {"auditor_audit_config": [_entity("default", "default")], "auditor_audit_target": []},
        conflicts={("garak_plugin_scan_config", "default", "default")},
    )

    assert await _legacy.migrate_legacy_entities(object()) == 1  # type: ignore[arg-type]
    assert client.deleted == [("auditor_audit_config", "default", "default")]
    assert await _legacy.migrate_legacy_entities(object()) == 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_job_source_is_rewritten_without_touching_other_fields(fake):
    job = _entity("default", "job-1", {"source": "auditor", "spec": {"x": 1}})
    other = _entity("default", "job-2", {"source": "evaluator"})
    client = fake({"platform_job": [job, other]})

    assert await _legacy.migrate_legacy_job_sources(object()) == 1  # type: ignore[arg-type]

    assert client.updated == [("default", "job-1", {"source": "garak-plugin", "spec": {"x": 1}})]
    assert other.data == {"source": "evaluator"}
    assert client.list_filters[0] == '{"data.source": "auditor"}'


def test_new_env_var_wins_over_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(GARAK_PYTHON_ENVVAR, "/new/python")
    monkeypatch.setenv(_legacy.LEGACY_GARAK_PYTHON_ENVVAR, "/old/python")
    assert _resolve_garak_python() == ["/new/python"]


def test_legacy_env_var_is_still_honoured_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.delenv(GARAK_PYTHON_ENVVAR, raising=False)
    monkeypatch.setenv(_legacy.LEGACY_GARAK_PYTHON_ENVVAR, "/old/python")
    with caplog.at_level("WARNING"):
        assert _resolve_garak_python() == ["/old/python"]
    assert "deprecated" in caplog.text


def test_legacy_default_venv_is_used_only_when_the_new_one_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv(GARAK_PYTHON_ENVVAR, raising=False)
    monkeypatch.delenv(_legacy.LEGACY_GARAK_PYTHON_ENVVAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    old = tmp_path / ".auditor" / ".venv" / "bin" / "python"
    old.parent.mkdir(parents=True)
    old.write_text("")

    assert _resolve_garak_python()[0] == str(old)

    new = tmp_path / ".garak-plugin" / ".venv" / "bin" / "python"
    new.parent.mkdir(parents=True)
    new.write_text("")
    assert _resolve_garak_python()[0] == str(new)
