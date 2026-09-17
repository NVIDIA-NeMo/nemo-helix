# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for run-record helpers (manifest fact extraction + entity flattening).

These import plugin modules that depend on nemo-platform packages, so they run under the
workspace test environment (``make test-package PACKAGE=nemo_agent_hardener_plugin``).
"""

from __future__ import annotations

import types

import yaml
from nemo_agent_hardener_plugin.jobs.run import _manifest_facts, _run_data
from nemo_agent_hardener_plugin.sdk import _run_to_dict


def test_manifest_facts_reads_agent_name_and_port(tmp_path):
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text(yaml.safe_dump({"agent": {"name": "calc", "port": 9123}, "backends": []}), encoding="utf-8")
    assert _manifest_facts(str(manifest)) == ("calc", 9123)


def test_manifest_facts_tolerates_missing_file():
    assert _manifest_facts("/no/such/manifest.yaml") == ("", 0)


def test_run_data_carries_source_run_for_sanity_check():
    # A validate-only sanity check records the harden run it came from, so the Harden tab re-attaches its
    # scorecard on reload; a normal run leaves it empty.
    linked = _run_data("default/scout", 8000, "m.yaml", "failed", 1, source_run="agent-hardener-run-abc")
    assert linked["source_run"] == "agent-hardener-run-abc"
    assert _run_data("default/scout", 8000, "m.yaml", "running", -1)["source_run"] == ""


def test_run_data_includes_events_fileset():
    from nemo_agent_hardener_plugin.jobs.records import _run_data

    data = _run_data(
        agent="test-agent",
        port=0,
        manifest="manifest-1",
        manifest_id="mid-1",
        status="completed",
        returncode=0,
        events_fileset="default/my-events-fileset",
    )
    assert data["events_fileset"] == "default/my-events-fileset"


def test_run_to_dict_flattens_entity_data_name_and_created_at():
    entity = types.SimpleNamespace(
        data={"agent": "default/calc", "status": "completed", "returncode": 0},
        name="agent-hardener-run-abc",
        created_at="2026-06-28T10:00:00",
    )
    flat = _run_to_dict(entity)
    assert flat["agent"] == "default/calc"
    assert flat["status"] == "completed"
    assert flat["name"] == "agent-hardener-run-abc"
    assert flat["created_at"] == "2026-06-28T10:00:00"
