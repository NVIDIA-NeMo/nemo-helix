# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.skill_scans import SkillBundledTestMetric, SkillSecurityScanMetric
from nhx_evals_sdk.metrics.skill_traces import ToolExecutionOutcomeMetric
from nhx_evals_sdk.values.skill_evidence import (
    SkillFinding,
    SkillScanReport,
    SkillTestReport,
    SkillToolCall,
    SkillToolTrace,
)
from nhx_evals_sdk.values.skills import SkillSnapshot


async def main() -> None:
    document = "---\nname: demo\ndescription: Use this skill to summarize reports.\n---\nSummarize the report.\n"
    digest = hashlib.sha256(document.encode()).hexdigest()
    snapshot = SkillSnapshot(directory_name="demo", digest=digest, document=document, files=["SKILL.md"])
    scan = SkillScanReport(
        subject_id=digest,
        scanner="example-fixture",
        scanner_version="1",
        status="completed",
        files_scanned=1,
        files_expected=1,
        findings=[SkillFinding(rule="demo-policy", category="policy", severity="high", file="SKILL.md", line=5)],
    )
    tests = SkillTestReport(
        artifact_digest=digest, status="completed", passed=3, failed=1, errors=0, skipped=1, coverage=0.75
    )
    trace = SkillToolTrace(
        trial_id="example-trial",
        complete=False,
        calls=[
            SkillToolCall(id=f"exec-{index}", step=0, tool="exec_command", succeeded=outcome)
            for index, outcome in enumerate((True, False, None), start=1)
        ],
    )
    evidence = {
        "reference": {
            "skill_snapshot": snapshot.model_dump(mode="json"),
            "skill_scans": {"security": scan.model_dump(mode="json")},
            "skill_test_report": tests.model_dump(mode="json"),
        },
        "trial": {"id": trace.trial_id, "metadata": {"skill_tool_trace": trace.model_dump(mode="json")}},
    }
    inputs = MetricInput(row=DatasetRow(data=evidence), candidate=CandidateOutput())
    metrics = (
        SkillSecurityScanMetric(),
        SkillBundledTestMetric(),
        ToolExecutionOutcomeMetric(tool_pattern="exec_command"),
    )
    scores = {metric.type.value: (await metric.compute_scores(inputs)).model_dump(mode="json") for metric in metrics}
    print(
        json.dumps({"source": "synthetic fixture", "evidence": evidence, "scores": scores}, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    asyncio.run(main())
