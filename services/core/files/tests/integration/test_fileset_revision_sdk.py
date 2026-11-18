# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-only coverage for async revision fields and OTLP response headers."""

import json

from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput, UpdateFilesetRequest


def log_payload() -> bytes:
    return json.dumps(
        {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": key, "value": {"stringValue": value}}
                            for key, value in (
                                ("job", "demo"),
                                ("job_attempt", "attempt"),
                                ("job_step", "step"),
                                ("job_task", "task"),
                            )
                        ]
                    },
                    "scopeLogs": [
                        {"logRecords": [{"timeUnixNano": "1704110400000000000", "body": {"stringValue": "SDK demo"}}]}
                    ],
                }
            ]
        }
    ).encode()


def test_sync_sdk_exposes_otlp_revision_header(files_client: FilesClient, fileset: FilesetOutput):
    response = files_client.upload_otlp_logs(workspace="default", name=fileset.name, content=log_payload())
    assert response.data().partial_success is None
    revision = response.http_response.headers["X-Fileset-Revision"]
    assert revision != fileset.revision
    assert files_client.get_fileset(workspace="default", name=fileset.name).data().revision == revision


async def test_async_sdk_revision_fields_and_otlp_header(async_files_client: AsyncFilesClient):
    files = async_files_client
    created = (
        await files.create_fileset(workspace="default", body=CreateFilesetRequest(name="async-sdk-revisions"))
    ).data()
    try:
        initial = created.revision
        assert initial
        uploaded = (
            await files.upload_file(workspace="default", name=created.name, path="file.txt", content=b"first")
        ).data()
        assert uploaded.fileset_revision and uploaded.fileset_revision != initial
        observed = (await files.get_fileset(workspace="default", name=created.name)).data()
        assert observed.revision == uploaded.fileset_revision
        download = await files.download_file(workspace="default", name=created.name, path="file.txt")
        assert await download.read() == b"first"

        updated = (
            await files.update_fileset(
                workspace="default", name=created.name, body=UpdateFilesetRequest(description="async update")
            )
        ).data()
        assert updated.revision and updated.revision != uploaded.fileset_revision
        observed = (await files.get_fileset_by_id(workspace="default", fileset_id=created.id)).data()
        assert observed.revision == updated.revision
        assert observed.description == "async update"

        deleted = (await files.delete_file(workspace="default", name=created.name, path="file.txt")).data()
        assert deleted.fileset_revision and deleted.fileset_revision != updated.revision
        observed = (await files.get_fileset(workspace="default", name=created.name)).data()
        assert observed.revision == deleted.fileset_revision

        response = await files.upload_otlp_logs(workspace="default", name=created.name, content=log_payload())
        assert response.data().partial_success is None
        revision = response.http_response.headers["X-Fileset-Revision"]
        assert revision != deleted.fileset_revision
        observed = (await files.get_fileset(workspace="default", name=created.name)).data()
        assert observed.revision == revision
    finally:
        await files.delete_fileset_by_id(workspace="default", fileset_id=created.id)
