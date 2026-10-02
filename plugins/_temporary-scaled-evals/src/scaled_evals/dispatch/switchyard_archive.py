# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard-owned evidence validation, injected into generic archive assembly."""

from pathlib import Path

from scaled_evals.archive_validation import BenchmarkArchiveError, read_archive_json
from scaled_evals.models.benchmark_archives import (
    BenchmarkArchiveArtifact,
    BenchmarkArchiveExportMember,
    UnavailableBenchmarkEvidence,
)


def check_campaign_evidence(
    output: Path,
    members: list[BenchmarkArchiveExportMember],
    artifacts: list[BenchmarkArchiveArtifact],
) -> list[UnavailableBenchmarkEvidence]:
    """Require ready campaign references to match the captured shared artifact."""
    captured = {item.object_key: item for item in artifacts}
    unavailable = []
    for member in members:
        path = output / "_scaled_evals" / "evaluations" / member.id / "switchyard" / "campaign_evidence.json"
        if not path.is_file():
            continue
        reference = read_archive_json(path)
        if reference.get("status") == "unavailable":
            unavailable.append(UnavailableBenchmarkEvidence(evaluation_id=member.id, kind="switchyard_campaign"))
            continue
        artifact = captured.get(reference.get("routing_stats_object_key"))
        expected_sha256 = str(reference.get("routing_stats_sha256") or "").removeprefix("sha256:")
        if reference.get("status") != "ready" or artifact is None or artifact.sha256 != expected_sha256:
            raise BenchmarkArchiveError("referenced Switchyard campaign evidence is missing or changed")
    return unavailable
