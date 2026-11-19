# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema migration behavior for mutation boundaries."""

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.schema_migrations import migrate_schema_fileset_id


@pytest.mark.parametrize("status", [200, 403, 404])
def test_schema_migration_lookup_captures_id_and_preserves_errors(status: int) -> None:
    """Resolve a missing ID, distinguish absence, and propagate lookup denial."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status != 200:
            return httpx.Response(status, json={"detail": "unavailable"})
        return httpx.Response(
            200,
            json={
                "id": "captured-agent-files-id",
                "name": "agent-files",
                "workspace": "dev",
                "purpose": "generic",
                "storage": {"type": "local", "path": "/filesets/dev/agent-files"},
                "description": "",
                "metadata": {},
                "custom_fields": {},
                "project": "",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = FilesClient.from_client(NemoClient(base_url="http://testserver", http_client=http))
        if status == 403:
            with pytest.raises(PermissionDeniedError):
                migrate_schema_fileset_id(client, workspace="default", ref="dev/agent-files", fileset_id=None)
        else:
            assert migrate_schema_fileset_id(client, workspace="default", ref="dev/agent-files", fileset_id=None) == (
                "captured-agent-files-id" if status == 200 else None
            )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/apis/files/v2/workspaces/dev/filesets/agent-files")
    ]


def test_schema_migration_preserves_recorded_id_without_lookup() -> None:
    """A saved allocation ID remains authoritative even without a usable reference."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = FilesClient.from_client(NemoClient(base_url="http://testserver", http_client=http))
        assert migrate_schema_fileset_id(client, ref="", fileset_id="recorded-id") == "recorded-id"
    assert requests == []
