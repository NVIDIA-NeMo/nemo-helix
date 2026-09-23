# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for usage leaderboard normalization."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from nemo_agents_plugin.leaderboard.normalize import normalize_report, normalize_reports


def _single_usage_report(
    *, timestamp: str = "20260429T220000Z", compute_units: int | None = 16000
) -> dict[str, object]:
    return {
        "schema_version": "v0",
        "task": {
            "task": "workspace-basic-mcp",
            "timestamp": timestamp,
            "image": "nhx-nat-workspace-basic-mcp:latest",
            "reward": 1,
            "build_status": "ok",
            "agent_status": "ok",
            "verify_status": "ok",
            "prompt_tokens": 1500,
            "completion_tokens": 500,
            "total_tokens": 2000,
            "compute_units": compute_units,
            "source_dir": "/tmp/workspace-basic-mcp",
        },
    }


def test_normalize_single_usage_report(tmp_path: Path):
    source_path = tmp_path / "result.json"
    entry = normalize_report(_single_usage_report(), source_path=source_path)

    assert entry.entry_id == "result.json"
    assert entry.task_name == "workspace-basic-mcp"
    assert entry.compute_units == 16000.0
    assert entry.compute_units_formula_version == "usage_report_v0_compute_units"
    assert entry.token_count == 2000
    assert entry.runtime_image == "nhx-nat-workspace-basic-mcp:latest"
    assert entry.created_at == datetime(2026, 4, 29, 22, 0, 0, tzinfo=timezone.utc)
    assert entry.source_path == str(source_path.resolve())
    assert entry.source_dir == "/tmp/workspace-basic-mcp"
    assert entry.run_count == 1


def test_normalize_batch_usage_report():
    entry = normalize_report(
        {
            "schema_version": "v0",
            "runs": [
                {
                    "task": "workspace-basic-mcp",
                    "timestamp": "20260429T220000Z",
                    "image": "nhx-nat-workspace-basic-mcp:latest",
                    "reward": 1,
                    "build_status": "ok",
                    "agent_status": "ok",
                    "verify_status": "ok",
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "compute_units": 12000,
                    "source_dir": "/tmp/workspace-basic-mcp",
                },
                {
                    "task": "secrets-crud-cli",
                    "timestamp": "20260429T230000Z",
                    "image": "nhx-nat-workspace-basic-mcp:latest",
                    "reward": 1,
                    "build_status": "ok",
                    "agent_status": "ok",
                    "verify_status": "ok",
                    "prompt_tokens": 1200,
                    "completion_tokens": 600,
                    "total_tokens": 1800,
                    "compute_units": 14400,
                    "source_dir": "/tmp/secrets-crud-cli",
                },
            ],
            "prompt_tokens_total": 2200,
            "completion_tokens_total": 1100,
            "total_tokens_total": 3300,
            "compute_units_total": 26400,
            "null_token_runs": 0,
            "skipped_runs": 0,
            "unparseable_runs": 0,
        }
    )

    assert entry.task_name == "secrets-crud-cli +1 more (2 runs)"
    assert entry.compute_units == 26400.0
    assert entry.compute_units_formula_version == "usage_report_v0_compute_units"
    assert entry.token_count == 3300
    assert entry.runtime_image == "nhx-nat-workspace-basic-mcp:latest"
    assert entry.created_at == datetime(2026, 4, 29, 23, 0, 0, tzinfo=timezone.utc)
    assert entry.run_count == 2


def test_normalize_batch_usage_report_without_shared_runtime_image():
    report = {
        "schema_version": "v0",
        "runs": [
            {
                "task": "workspace-basic-mcp",
                "timestamp": "20260429T220000Z",
                "image": "nhx-nat-workspace-basic-mcp:latest",
                "reward": 1,
                "build_status": "ok",
                "agent_status": "ok",
                "verify_status": "ok",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "compute_units": 12000,
                "source_dir": "/tmp/workspace-basic-mcp",
            },
            {
                "task": "secrets-crud-cli",
                "timestamp": "20260429T230000Z",
                "image": "nhx-nat-secrets-crud-cli:latest",
                "reward": 1,
                "build_status": "ok",
                "agent_status": "ok",
                "verify_status": "ok",
                "prompt_tokens": 1200,
                "completion_tokens": 600,
                "total_tokens": 1800,
                "compute_units": 14400,
                "source_dir": "/tmp/secrets-crud-cli",
            },
        ],
        "prompt_tokens_total": 2200,
        "completion_tokens_total": 1100,
        "total_tokens_total": 3300,
        "compute_units_total": 26400,
        "null_token_runs": 0,
        "skipped_runs": 0,
        "unparseable_runs": 0,
    }

    entry = normalize_report(report)

    assert entry.runtime_image is None


def test_normalize_report_rejects_invalid_schema():
    with pytest.raises(ValueError, match="Report does not satisfy leaderboard schema"):
        normalize_report({"schema_version": "v0"})


def test_normalize_report_rejects_report_without_compute_units():
    report = deepcopy(_single_usage_report())
    report["task"]["compute_units"] = None

    with pytest.raises(ValueError, match="Field 'task\\.compute_units' must be set for leaderboard ranking"):
        normalize_report(report)


def test_normalize_report_rejects_batch_report_without_compute_units_total():
    with pytest.raises(ValueError, match="Field 'compute_units_total' must be set for leaderboard ranking"):
        normalize_report(
            {
                "schema_version": "v0",
                "runs": [],
                "prompt_tokens_total": None,
                "completion_tokens_total": None,
                "total_tokens_total": None,
                "compute_units_total": None,
                "null_token_runs": 1,
                "skipped_runs": 0,
                "unparseable_runs": 0,
            }
        )


def test_normalize_report_rejects_invalid_usage_report_timestamp():
    report = _single_usage_report(timestamp="not-a-timestamp")

    with pytest.raises(ValueError, match="Report contains an invalid usage timestamp: not-a-timestamp"):
        normalize_report(report)


def test_normalize_reports_preserves_order_and_source_paths(tmp_path: Path):
    source_path_1 = tmp_path / "one.json"
    source_path_2 = tmp_path / "two.json"
    report_1 = _single_usage_report(timestamp="20260429T220000Z", compute_units=16000)
    report_2 = _single_usage_report(timestamp="20260429T230000Z", compute_units=19200)

    entries = normalize_reports((report_1, report_2), source_paths=(source_path_1, source_path_2))

    assert [entry.entry_id for entry in entries] == ["one.json", "two.json"]
    assert [entry.source_path for entry in entries] == [
        str(source_path_1.resolve()),
        str(source_path_2.resolve()),
    ]


def test_normalize_reports_requires_matching_source_path_count():
    with pytest.raises(ValueError, match="source_paths length must match reports length"):
        normalize_reports((_single_usage_report(),), source_paths=("one.json", "two.json"))
