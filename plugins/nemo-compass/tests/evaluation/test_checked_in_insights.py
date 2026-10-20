# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import yaml
from evaluation import cli, release
from evaluation.registry import load_registry


def test_checked_in_insights_match_current_compass_and_state_pins() -> None:
    insights_dir = getattr(cli, "INSIGHTS_DIR", cli.HERE / "insights")
    if not insights_dir.exists():
        pytest.skip("Task 5 has not generated the checked-in Insights directory")

    manifest_path = insights_dir / "manifest.yaml"
    assert manifest_path.is_file(), "checked-in Insights directory exists without manifest.yaml"

    subjects = load_registry(cli.REGISTRY_PATH)
    names = sorted(subjects)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    snapshots = manifest.get("snapshots")
    assert isinstance(snapshots, dict)

    assert {path.name for path in insights_dir.glob("*.yaml")} == {
        "manifest.yaml",
        *(f"{name}.yaml" for name in names),
    }
    assert set(snapshots) == set(names)
    for name in names:
        insights_path = insights_dir / f"{name}.yaml"
        assert snapshots[name] == {
            "compass_sha256": cli._compass_sha256(),
            "insights_sha256": hashlib.sha256(insights_path.read_bytes()).hexdigest(),
            "state": release.pinned_ref(subjects[name]),
        }
