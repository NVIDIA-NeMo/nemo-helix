# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for job artifact helpers: events fileset upload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_save_events_fileset_uploads_file(tmp_path: Path) -> None:
    from nemo_agent_hardener_plugin.jobs.artifacts import _save_events_fileset

    events_file = tmp_path / "events.jsonl"
    events_file.write_text('{"event":"test","payload":{}}\n')

    sdk = MagicMock()
    with (
        patch(
            "nemo_agent_hardener_plugin.jobs.artifacts._events_path",
            return_value=events_file,
        ),
        patch(
            "nemo_agent_hardener_plugin.jobs.artifacts.upload_file_to_fileset",
            return_value="default/events-abc123",
        ) as mock_upload,
    ):
        result = _save_events_fileset(sdk, workspace="default", run_name="my-run")

    mock_upload.assert_called_once_with(sdk, events_file, workspace="default")
    assert result == "default/events-abc123"


def test_save_events_fileset_returns_empty_when_file_missing(tmp_path: Path) -> None:
    from nemo_agent_hardener_plugin.jobs.artifacts import _save_events_fileset

    sdk = MagicMock()
    with patch(
        "nemo_agent_hardener_plugin.jobs.artifacts._events_path",
        return_value=tmp_path / "nonexistent.jsonl",
    ):
        result = _save_events_fileset(sdk, workspace="default", run_name="my-run")

    assert result == ""


def test_save_events_fileset_returns_empty_on_upload_error(tmp_path: Path) -> None:
    from nemo_agent_hardener_plugin.jobs.artifacts import _save_events_fileset

    events_file = tmp_path / "events.jsonl"
    events_file.write_text('{"event":"test","payload":{}}\n')

    sdk = MagicMock()
    with (
        patch(
            "nemo_agent_hardener_plugin.jobs.artifacts._events_path",
            return_value=events_file,
        ),
        patch(
            "nemo_agent_hardener_plugin.jobs.artifacts.upload_file_to_fileset",
            side_effect=Exception("network error"),
        ),
    ):
        result = _save_events_fileset(sdk, workspace="default", run_name="my-run")

    assert result == ""
