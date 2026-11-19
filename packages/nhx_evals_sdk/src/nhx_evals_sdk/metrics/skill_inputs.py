# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from nhx_evals_sdk.metrics.protocol import MetricInput
from nhx_evals_sdk.metrics.skills import _snapshot
from nhx_evals_sdk.values.skill_evidence import SkillContentBundle, SkillScanReport, SkillToolTrace


def artifact_contents(input: MetricInput, *, package: bool) -> dict[str, str] | None:
    snapshot = _snapshot(input)
    if snapshot is None:
        return None
    if not package:
        return {"SKILL.md": snapshot.document} if snapshot.document is not None else None
    raw = input.row.data.get("reference", {}).get("skill_contents")
    if raw is None:
        return None
    contents = SkillContentBundle.model_validate(raw)
    if contents.artifact_digest != snapshot.digest:
        raise ValueError("skill contents belong to another artifact revision")
    if set(contents.files) - set(snapshot.files):
        raise ValueError("skill contents include files outside the snapshot inventory")
    if "SKILL.md" in contents.files and contents.files["SKILL.md"] != snapshot.document:
        raise ValueError("skill contents disagree with the original entry document")
    return contents.files


def artifact_scan(input: MetricInput, key: str) -> SkillScanReport | None:
    snapshot = _snapshot(input)
    if snapshot is None:
        return None
    raw = input.row.data.get("reference", {}).get("skill_scans", {}).get(key)
    if raw is None:
        return None
    report = SkillScanReport.model_validate(raw)
    if report.subject_id != snapshot.digest:
        raise ValueError("scan report belongs to another artifact revision")
    return report


def trial_trace(input: MetricInput) -> SkillToolTrace | None:
    trial = input.row.data.get("trial", {})
    raw = trial.get("metadata", {}).get("skill_tool_trace")
    if raw is None:
        return None
    trace = SkillToolTrace.model_validate(raw)
    if trace.trial_id != trial.get("id"):
        raise ValueError("tool trace belongs to another trial")
    return trace


def trial_scan(input: MetricInput, key: str) -> SkillScanReport | None:
    trial = input.row.data.get("trial", {})
    raw = trial.get("metadata", {}).get("skill_scans", {}).get(key)
    if raw is None:
        return None
    report = SkillScanReport.model_validate(raw)
    if report.subject_id != trial.get("id"):
        raise ValueError("scan report belongs to another trial")
    return report


def mapping_at_path(value: Mapping, path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
