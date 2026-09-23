# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that dataset files were uploaded to the fileset correctly.

Checks:
- harbor-dataset-fileset exists with correct description
- training.jsonl, validation.jsonl, testing.jsonl are present in the fileset
- Each file contains valid JSONL with prompt/completion format
- Row counts meet minimums (training >= 3, validation >= 2, testing >= 2)
"""

import json
import os

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from trace_reader import get_session

WORKSPACE = "default"
FILESET_NAME = "harbor-dataset-fileset"
EXPECTED_FILES = {
    "training.jsonl": 3,
    "validation.jsonl": 2,
    "testing.jsonl": 2,
}


@pytest.fixture
def client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE)


@pytest.fixture
def files_client(client: NeMoHelix) -> FilesClient:
    return client_from_platform(client, FilesClient)


def test_fileset_exists(files_client: FilesClient) -> None:
    """Test that harbor-dataset-fileset was created with correct metadata."""
    response = files_client.get_fileset(name=FILESET_NAME).data()
    assert response.name == FILESET_NAME, f"Expected fileset name '{FILESET_NAME}', got '{response.name}'"
    assert response.description == "Dataset fileset for harbor eval", (
        f"Expected description 'Dataset fileset for harbor eval', got '{response.description}'"
    )


def test_all_files_uploaded(client: NeMoHelix) -> None:
    """Test that all three JSONL files are present in the fileset."""
    files = client.files.list(fileset=FILESET_NAME)
    file_paths = [f.path for f in files.data]

    for filename in EXPECTED_FILES:
        assert any(filename in p for p in file_paths), (
            f"File '{filename}' not found in {FILESET_NAME}! Found files: {file_paths}"
        )


@pytest.mark.parametrize("filename,min_rows", list(EXPECTED_FILES.items()))
def test_file_content_valid_jsonl(client: NeMoHelix, filename: str, min_rows: int) -> None:
    """Test that each uploaded file contains valid JSONL with prompt/completion format."""
    content_bytes = client.files.download_content(remote_path=filename, fileset=FILESET_NAME)
    content = content_bytes.decode("utf-8").strip()

    assert content, f"File '{filename}' is empty!"

    lines = content.split("\n")
    valid_rows = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Line {i + 1} in '{filename}' is not valid JSON: {e}")

        assert "prompt" in row, f"Line {i + 1} in '{filename}' missing 'prompt' key. Got keys: {list(row.keys())}"
        assert "completion" in row, (
            f"Line {i + 1} in '{filename}' missing 'completion' key. Got keys: {list(row.keys())}"
        )
        valid_rows += 1

    assert valid_rows >= min_rows, f"'{filename}' has {valid_rows} valid rows, expected at least {min_rows}"


def test_agent_performed_list_operations() -> None:
    """
    Verify the agent executed the expected operations via trajectory analysis.

    This is otherwise impossible to verify via the CLI or API.
    """
    session = get_session()
    commands = session.get_bash_commands()

    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # Agent should have listed files to verify
    assert has_command("files list"), f"Agent did not list files to verify uploads. Commands: {commands}"

    print(f"Test passed: Agent performed all list operations. Total commands: {len(commands)}")
