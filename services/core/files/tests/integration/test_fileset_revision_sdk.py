# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async SDK commit markers across file, metadata and OTLP mutations."""

import json

from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, UpdateFilesetRequest


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


async def test_async_sdk_commit_marker_survives_ordinary_mutations(async_files_client: AsyncFilesClient):
    files = async_files_client
    created = (
        await files.create_fileset(workspace="default", body=CreateFilesetRequest(name="async-sdk-commits"))
    ).data()
    try:
        committed = (await files.commit_fileset(workspace="default", name=created.name)).data()
        assert committed.commit_revision and committed.commit_revision != created.commit_revision
        await files.upload_file(workspace="default", name=created.name, path="file.txt", content=b"first")
        observed = (await files.get_fileset(workspace="default", name=created.name)).data()
        assert observed.commit_revision == committed.commit_revision
        download = await files.download_file(workspace="default", name=created.name, path="file.txt")
        assert await download.read() == b"first"

        updated = (
            await files.update_fileset(
                workspace="default", name=created.name, body=UpdateFilesetRequest(description="async update")
            )
        ).data()
        assert updated.commit_revision == committed.commit_revision
        assert updated.description == "async update"
        await files.delete_file(workspace="default", name=created.name, path="file.txt")
        observed = (await files.get_fileset_by_id(workspace="default", fileset_id=created.id)).data()
        assert observed.commit_revision == committed.commit_revision

        response = await files.upload_otlp_logs(workspace="default", name=created.name, content=log_payload())
        assert response.data().partial_success is None
        observed = (await files.get_fileset(workspace="default", name=created.name)).data()
        assert observed.commit_revision == committed.commit_revision
        committed_again = (await files.commit_fileset(workspace="default", name=created.name)).data()
        assert committed_again.commit_revision != committed.commit_revision
    finally:
        await files.delete_fileset_by_id(workspace="default", fileset_id=created.id)
