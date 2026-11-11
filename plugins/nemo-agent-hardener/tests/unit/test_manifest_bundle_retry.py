# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version conflicts reuse one uploaded agent fileset without overwriting another writer."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from _doubles import make_job_context
from nemo_agent_hardener_plugin.agent_resolver import ResolvedManifest
from nemo_agent_hardener_plugin.entities import AGENT_HARDENER_MANIFEST_TYPE
from nemo_agent_hardener_plugin.filesets import UploadedFileset
from nemo_agent_hardener_plugin.jobs import manifest as manifest_module
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.entities.types import Entity, EntityUpdate


@pytest.mark.parametrize(
    ("scenario", "expected_writes", "expected_reads", "discard"),
    [
        ("concurrent_edit", 2, 1, False),
        ("concurrent_freeze", 1, 1, True),
        ("deleted", 1, 1, True),
        ("missing_on_write", 1, 0, True),
        ("deleted_before_retry", 2, 1, True),
        ("lost_response_before_missing", 1, 0, False),
        ("changed_source", 1, 1, True),
        ("conflicts_exhausted", 2, 2, True),
        ("lost_write_response", 1, 0, False),
        ("failed_conflict_read", 1, 1, False),
        ("committed_before_conflict", 2, 2, False),
    ],
)
def test_freeze_retries_same_manifest_and_reconciles_upload_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_writes: int,
    expected_reads: int,
    discard: bool,
) -> None:
    """Retry fresh versions; retain uncertain commits and discard only confirmed unused allocations."""
    now = datetime.now(timezone.utc)
    record = Entity(
        entity_type=AGENT_HARDENER_MANIFEST_TYPE,
        id="original-manifest-id",
        workspace="default",
        name="clockbot-hardening",
        data={"agent": "default/clockbot", "rounds": 1},
        created_at=now,
        updated_at=now,
        db_version=4,
    )
    allocated = UploadedFileset("default/agent-files", "uploaded-fileset-id")
    uploaded = MagicMock(return_value=allocated)
    cleanup = MagicMock()
    monkeypatch.setattr(manifest_module, "upload_project_dir", uploaded)
    monkeypatch.setattr(manifest_module, "delete_fileset_by_id", cleanup)
    monkeypatch.setattr(manifest_module, "client_from_platform", client_from_platform)
    resolved = ResolvedManifest(
        manifest={"agent": {"name": "clockbot", "port": 8000}},
        agent_config_path=tmp_path / "agent.yaml",
        project_dir=tmp_path,
        workspace="default",
        agent_name="clockbot",
        port=8000,
        secrets=[],
    )
    writes: list[EntityUpdate] = []
    reads = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        if request.method == "PUT":
            assert request.url.path == (
                f"/apis/entities/v2/workspaces/default/entities/{AGENT_HARDENER_MANIFEST_TYPE}"
                "/by-id/original-manifest-id"
            )
            body = EntityUpdate.model_validate_json(request.content)
            writes.append(body)
            if scenario in {"lost_write_response", "lost_response_before_missing"} and len(writes) == 1:
                raise httpx.ReadTimeout("Write outcome unknown", request=request)
            if scenario in {"missing_on_write", "lost_response_before_missing"} or (
                scenario == "deleted_before_retry" and len(writes) == 2
            ):
                return httpx.Response(404, json={"detail": "Original manifest deleted"})
            if scenario == "concurrent_edit" and len(writes) == 2:
                saved = record.model_copy(update={"data": body.data, "db_version": 6})
                return httpx.Response(200, json=saved.model_dump(mode="json"))
            return httpx.Response(409, json={"detail": "Version conflict"})
        assert request.method == "GET"
        assert request.url.path == "/apis/entities/v2/entities/original-manifest-id"
        reads += 1
        if scenario == "deleted":
            return httpx.Response(404, json={"detail": "Original manifest deleted"})
        if scenario == "failed_conflict_read":
            return httpx.Response(503, json={"detail": "Unavailable"})
        data = {**record.data, "rounds": 9}
        if scenario == "changed_source":
            data["agent"] = "default/another-agent"
        elif scenario == "concurrent_freeze":
            data.update(agent_fileset="default/winning-bundle", agent_fileset_id="winning-fileset-id")
        elif scenario == "committed_before_conflict" and reads == 2:
            data.update(agent_fileset=allocated.ref, agent_fileset_id=allocated.fileset_id)
        fresh = record.model_copy(update={"data": data, "db_version": 4 + reads})
        return httpx.Response(200, json=fresh.model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = EntitiesClient(
            base_url="http://testserver",
            http_client=http,
            retry=RetryPolicy(max_retries=2 if scenario == "lost_response_before_missing" else 0, backoff_base=0),
        )
        manifest_module._persist_upgraded_bundle(client, record.name, make_job_context(tmp_path), record, resolved)

    uploaded.assert_called_once()
    assert len(writes) == expected_writes
    assert reads == expected_reads
    assert writes[0].expected_db_version == 4
    assert writes[0].data["agent_fileset_id"] == allocated.fileset_id
    if expected_writes == 2:
        assert writes[1].expected_db_version == 5
        assert writes[1].data["rounds"] == 9
        assert writes[1].data["agent_fileset_id"] == allocated.fileset_id
    if discard:
        cleanup.assert_called_once()
        assert cleanup.call_args.kwargs == {"workspace": "default", "fileset_id": allocated.fileset_id}
    else:
        cleanup.assert_not_called()
    assert record.data == {"agent": "default/clockbot", "rounds": 1}
