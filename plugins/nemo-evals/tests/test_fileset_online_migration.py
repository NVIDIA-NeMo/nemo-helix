# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy references pin one allocation before their owners change or disappear."""

import pytest
from nemo_evals.api.schemas import AgentRef
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_spec import FabricRunnerTarget, RegisteredAgentSource, registered_agent_source
from nemo_evals.jobs.registered_agent_cleanup import discard_snapshot
from nemo_evals.schema_migrations import migrate_schema_agent_target
from nemo_evals.taskset_runs.plan_storage import delete_fileset_ref
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.types import CreateFilesetRequest


async def test_legacy_snapshot_upgrade_survives_name_reuse(files_service, files_client) -> None:
    """Migration returns a copy and keeps the captured allocation after name reuse."""
    original = (
        await files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name="snapshot"))
    ).data()
    target = FabricRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="default/agent"), files=FilesetRef(root="default/snapshot"))
    )
    files_service.calls.clear()
    original_target = target
    target = await migrate_schema_agent_target(target, files_client=files_client)
    original_source = registered_agent_source(original_target)
    assert original_source is not None and original_source.fileset_id is None
    source = registered_agent_source(target)
    assert source is not None and source.fileset_id == original.id
    assert files_service.calls == [("get", "default", "snapshot")]

    files_service.allocations[("default", "snapshot")] = "replacement-id"
    files_service.calls.clear()
    target = await migrate_schema_agent_target(target, files_client=files_client)
    migrated = registered_agent_source(target)
    assert migrated is not None and migrated.fileset_id == original.id
    assert files_service.calls == []


async def test_absent_legacy_snapshot_is_already_cleaned_up(files_service, files_client) -> None:
    """Missing snapshot and plan filesets require no delete request."""
    target = FabricRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="default/agent"), files=FilesetRef(root="default/missing"))
    )
    target = await migrate_schema_agent_target(target, files_client=files_client)
    await delete_fileset_ref(files_client, "default/missing#plan.json", fileset_id=None)
    assert all(operation == "get" for operation, _, _ in files_service.calls)


async def test_lookup_denial_is_not_treated_as_an_absent_snapshot(files_service, files_client) -> None:
    """A denied lookup fails migration instead of silently dropping ownership."""
    files_service.failures["get"] = 403
    target = FabricRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="default/agent"), files=FilesetRef(root="default/snapshot"))
    )
    with pytest.raises(PermissionDeniedError):
        target = await migrate_schema_agent_target(target, files_client=files_client)


async def test_legacy_plan_cleanup_uses_one_lookup_then_id_delete(files_service, files_client) -> None:
    """Plan cleanup resolves a missing ID once before deleting the captured allocation."""
    await files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name="legacy-plan"))
    files_service.calls.clear()
    await delete_fileset_ref(files_client, "default/legacy-plan#plan.json", fileset_id=None)
    assert files_service.calls == [("get", "default", "legacy-plan"), ("delete", "default", "legacy-plan")]


async def test_upgraded_snapshot_cleanup_never_resolves_a_replacement(files_service, files_client) -> None:
    """Cleanup of a recorded ID leaves an allocation reusing its name intact."""
    original = (
        await files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name="snapshot"))
    ).data()
    target = FabricRunnerTarget(
        source=RegisteredAgentSource(
            agent=AgentRef(root="default/agent"), files=FilesetRef(root="default/snapshot"), fileset_id=original.id
        )
    )
    files_service.allocations[("default", "snapshot")] = "replacement-id"
    files_service.calls.clear()
    await discard_snapshot(target, files_client)
    assert files_service.allocations[("default", "snapshot")] == "replacement-id"
    assert not any(operation == "get" for operation, _, _ in files_service.calls)


async def test_cleanup_lookup_failure_remains_best_effort(files_service, files_client, caplog) -> None:
    """A failed cleanup lookup is logged without failing the completed operation."""
    files_service.failures["get"] = 403
    target = FabricRunnerTarget(
        source=RegisteredAgentSource(agent=AgentRef(root="default/agent"), files=FilesetRef(root="default/snapshot"))
    )
    await discard_snapshot(target, files_client)
    assert "Failed to resolve or clean up" in caplog.text
