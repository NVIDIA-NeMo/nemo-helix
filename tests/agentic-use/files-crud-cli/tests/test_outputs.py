# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that fileset CRUD operations were performed correctly.

Checks:
- harbor-test-fileset was deleted (should not exist)
- harbor-final-fileset exists with correct description
- verify.txt was uploaded to harbor-final-fileset with correct content
- Agent trajectory shows all intermediate operations were executed
"""

import os

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from trace_reader import get_session

WORKSPACE = "default"


@pytest.fixture
def client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url, workspace=WORKSPACE)


@pytest.fixture
def files_client(client: NemoClient) -> FilesClient:
    return FilesClient.from_client(client)


def test_harbor_test_fileset_deleted(files_client: FilesClient) -> None:
    """Test that harbor-test-fileset was deleted after CRUD operations."""
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert "harbor-test-fileset" not in fileset_names, (
        f"Fileset 'harbor-test-fileset' should have been deleted but still exists! Found: {fileset_names}"
    )


def test_harbor_final_fileset_exists(files_client: FilesClient) -> None:
    """Test that harbor-final-fileset was created and has correct metadata."""
    response = files_client.get_fileset(name="harbor-final-fileset").data()
    assert response.name == "harbor-final-fileset", (
        f"Expected fileset name 'harbor-final-fileset', got '{response.name}'"
    )
    assert response.description == "Final fileset for verification", (
        f"Expected description 'Final fileset for verification', got '{response.description}'"
    )


def test_verify_file_uploaded(client: NemoClient) -> None:
    """Test that verify.txt was uploaded to harbor-final-fileset with correct content."""
    files = FilesClient.from_client(client).list_files(name="harbor-final-fileset").data()
    file_paths = [f.path for f in files.data]
    assert any("verify.txt" in p for p in file_paths), (
        f"File 'verify.txt' not found in harbor-final-fileset! Found files: {file_paths}"
    )

    # Download and check file content
    content = FilesClient.from_client(client).download_file(name="harbor-final-fileset", path="verify.txt").read()
    content_str = content.decode("utf-8").strip()
    assert content_str == "harbor-verification-content", (
        f"Expected file content 'harbor-verification-content', got '{content_str}'"
    )


def test_agent_performed_all_crud_operations() -> None:
    """
    Verify the agent executed all intermediate CRUD operations via CLI.

    Why this test exists:
    The tests above only verify final state (harbor-test-fileset deleted,
    harbor-final-fileset exists with verify.txt). But the task requires the agent
    to perform a full CRUD lifecycle on harbor-test-fileset: create it, upload a
    file, list files, download the file, delete the file, then delete the fileset.

    Since the filesets service does hard deletes with no audit trail, we cannot
    verify these intermediate operations via the API. Instead, we read the Claude
    Code session transcript to confirm the agent actually executed all the required
    CLI commands.
    """
    session = get_session()
    commands = session.get_bash_commands()

    # Helper to check if any single command contains all specified patterns
    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # 1. Created harbor-test-fileset (both patterns must be in same command)
    assert has_command("files filesets create", "harbor-test-fileset"), (
        f"Agent did not create 'harbor-test-fileset'. Commands: {commands}"
    )

    # 2. Retrieved/verified harbor-test-fileset
    assert has_command("files filesets get", "harbor-test-fileset"), (
        f"Agent did not verify 'harbor-test-fileset' with get command. Commands: {commands}"
    )

    # 3. Uploaded a file to harbor-test-fileset
    assert has_command("files upload"), f"Agent did not upload files. Commands: {commands}"

    # 4. Listed files in the fileset
    assert has_command("files list"), f"Agent did not list files. Commands: {commands}"

    # 5. Downloaded file from the fileset
    assert has_command("files download"), f"Agent did not download files. Commands: {commands}"

    # 6. Deleted file from harbor-test-fileset
    assert has_command("files delete"), f"Agent did not delete file from fileset. Commands: {commands}"

    # 7. Deleted harbor-test-fileset
    assert has_command("files filesets delete", "harbor-test-fileset"), (
        f"Agent did not delete 'harbor-test-fileset'. Commands: {commands}"
    )

    # 8. Created harbor-final-fileset
    assert has_command("files filesets create", "harbor-final-fileset"), (
        f"Agent did not create 'harbor-final-fileset'. Commands: {commands}"
    )

    print(f"Test passed: Agent performed all CRUD operations. Total commands: {len(commands)}")
