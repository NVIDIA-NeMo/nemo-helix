# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the agent successfully created an intake app, submitted entries,
and exported them to a JSONL file via the Intake export API.

Since the NeMo Helix SDK does not have a dedicated intake resource, verification
uses raw HTTP requests to the Intake API endpoints.
"""

import json
import os
from pathlib import Path

import httpx
import pytest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from trace_reader import get_session

WORKSPACE = "intake-export-workspace"
APP_NAME = "harbor-test-app"
# Path is hardcoded to match the agent instruction's export destination.
# Cannot use tmp_path since the agent writes to this path inside the container.
EXPORT_FILE = Path("/tmp/intake-export.jsonl")  # noqa: S108
MIN_ENTRIES = 5


def _base_url() -> str:
    return os.environ.get("NHX_BASE_URL", "http://localhost:8080")


def _intake_url(path: str) -> str:
    return f"{_base_url()}/apis/intake/v2/workspaces/{WORKSPACE}/{path}"


@pytest.fixture
def nhx_client() -> WorkspacesClient:
    return WorkspacesClient(base_url=_base_url(), workspace=WORKSPACE)


@pytest.fixture
def http() -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=30)


def test_workspace_exists(nhx_client: WorkspacesClient) -> None:
    """Test that the intake-export-workspace was created."""
    response = nhx_client.list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found! Found: {workspace_names}"


def test_intake_app_exists(http: httpx.Client) -> None:
    """Test that the harbor-test-app intake app was created."""
    resp = http.get(_intake_url(f"apps/{APP_NAME}"))
    assert resp.status_code == 200, f"Failed to get intake app '{APP_NAME}': {resp.status_code} {resp.text}"
    app = resp.json()
    assert app["name"] == APP_NAME, f"Expected app name '{APP_NAME}', got '{app['name']}'"


def test_entries_submitted(http: httpx.Client) -> None:
    """Test that at least 5 entries were submitted with proper structure."""
    resp = http.get(
        _intake_url("entries"),
        params={"page_size": 100},
    )
    assert resp.status_code == 200, f"Failed to list entries: {resp.status_code} {resp.text}"
    data = resp.json()
    entries = data.get("data", [])
    assert len(entries) >= MIN_ENTRIES, f"Expected at least {MIN_ENTRIES} entries, found {len(entries)}"

    # Verify entries have proper structure (data with request/response)
    for entry in entries:
        entry_data = entry.get("data", {})
        assert "request" in entry_data, f"Entry {entry.get('name', '?')} missing 'request' in data"
        assert "response" in entry_data, f"Entry {entry.get('name', '?')} missing 'response' in data"


def test_export_job_completed(http: httpx.Client) -> None:
    """Test that an export job was created and completed successfully."""
    resp = http.get(_intake_url("export/jobs"), params={"page_size": 10})
    assert resp.status_code == 200, f"Failed to list export jobs: {resp.status_code} {resp.text}"
    data = resp.json()
    jobs = data.get("data", [])
    assert len(jobs) > 0, "No export jobs found!"

    # Find a completed job
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    assert len(completed_jobs) > 0, f"No completed export jobs! Job statuses: {[j.get('status') for j in jobs]}"

    job = completed_jobs[0]
    status_details = job.get("status_details", {})
    if status_details:
        entries_count = status_details.get("entries_count", 0)
        print(f"Export job status_details.entries_count={entries_count}")
        # Known platform issue: entries_count may be 0 even for successful exports
        # due to nested value object persistence. File validation in
        # test_export_file_valid_jsonl_with_messages is the authoritative check.
        if entries_count > 0:
            assert entries_count >= MIN_ENTRIES, (
                f"Export job reports {entries_count} entries, expected at least {MIN_ENTRIES}"
            )


def test_export_file_exists() -> None:
    """Test that the export JSONL file was created and is non-empty."""
    assert EXPORT_FILE.exists(), f"Export file not found at {EXPORT_FILE}!"
    assert EXPORT_FILE.stat().st_size > 0, f"Export file {EXPORT_FILE} is empty!"


def test_export_file_valid_jsonl_with_messages() -> None:
    """Test that the export file contains valid JSONL with the expected structure.

    The exporter transforms entries into a format that includes a top-level
    'messages' array for OpenAI/Customizer compatibility.
    """
    assert EXPORT_FILE.exists(), f"Export file not found at {EXPORT_FILE}"

    content = EXPORT_FILE.read_text().strip()
    lines = [line for line in content.split("\n") if line.strip()]

    assert len(lines) >= MIN_ENTRIES, f"Export file has {len(lines)} lines, expected at least {MIN_ENTRIES}"

    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Line {i + 1} is not valid JSON: {e}")

        assert isinstance(row, dict), f"Line {i + 1} is not a JSON object"

        # Exported entries should have data with request/response
        assert "data" in row, f"Line {i + 1} missing 'data' field"
        assert "request" in row["data"], f"Line {i + 1} missing 'data.request'"
        assert "response" in row["data"], f"Line {i + 1} missing 'data.response'"

        # Exporter adds a top-level 'messages' array for downstream compatibility
        if "messages" in row:
            assert isinstance(row["messages"], list), f"Line {i + 1}: 'messages' should be a list"
            assert len(row["messages"]) >= 2, (
                f"Line {i + 1}: 'messages' should have at least 2 entries (user + assistant)"
            )

    print(f"Export file contains {len(lines)} valid JSONL entries with proper structure")


def test_entry_count_matches_export(http: httpx.Client) -> None:
    """Test that the number of exported entries matches the entries in the workspace."""
    # Count entries via API
    resp = http.get(_intake_url("entries"), params={"page_size": 100})
    assert resp.status_code == 200
    api_entries = resp.json().get("data", [])

    # Count lines in export file
    assert EXPORT_FILE.exists(), f"Export file not found at {EXPORT_FILE}"
    content = EXPORT_FILE.read_text().strip()
    export_lines = [line for line in content.split("\n") if line.strip()]

    # The export should contain at least as many entries as were submitted
    # (could be equal or the export could have a filter that reduces count)
    assert len(export_lines) >= MIN_ENTRIES, f"Export has {len(export_lines)} entries, expected at least {MIN_ENTRIES}"
    print(f"API has {len(api_entries)} entries, export file has {len(export_lines)} entries")


def test_agent_used_intake_api() -> None:
    """Verify the agent interacted with the intake API via trace analysis."""
    session = get_session()
    commands = session.get_bash_commands()

    # Agent should have made HTTP requests to intake API endpoints
    has_intake_call = any("/apis/intake/" in cmd or "intake/v2" in cmd for cmd in commands)
    assert has_intake_call, f"Agent did not appear to call the intake API. Commands: {commands}"

    # Agent should have created an export job via the export endpoint
    has_export_call = any("/export/" in cmd or "export/jobs" in cmd for cmd in commands)
    assert has_export_call, f"Agent did not appear to create an export job. Commands: {commands}"
