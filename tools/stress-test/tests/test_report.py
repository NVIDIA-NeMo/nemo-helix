# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from config import PodSize, Step, Sweep
from records import write_json
from report import ReportError, write_report

PODS = {
    "exit": PodSize("100m", "256Mi", "500m", "512Mi"),
    "idle": PodSize("100m", "256Mi", "500m", "512Mi"),
    "hammer": PodSize("500m", "512Mi", "1", "1Gi"),
}


def test_partial_step_and_redaction(tmp_path: Path) -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    sweep = Sweep("stress", 8, PODS, (step,))
    run_dir = tmp_path / "runs"
    step_dir = run_dir / "steps" / step.id
    write_json(
        step_dir / "stress-exit-01-1.record.json",
        {
            "job_name": "stress-exit-01-1",
            "submit": "accepted",
            "status_history": ["completed"],
            "download_succeeded": False,
        },
    )
    (step_dir / "stress-exit-01-1.jsonl").write_text(
        '{"error":"secret-fake-access-key","latency":1,"operation":"noop"}\n'
    )
    (run_dir / "steps" / "retired").mkdir(parents=True)
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    document = (output / "report.html").read_text()
    assert "partial | yes" in markdown.replace(" ", "") or "| yes |" in markdown
    assert "secret-fake-access-key" not in markdown
    assert "secret-fake-access-key" not in document
    assert "[redacted]" in markdown
    assert "retired" not in markdown
    assert "<script" not in document.lower()


def test_missing_bundle_names_the_directory(tmp_path: Path) -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    sweep = Sweep("stress", 8, PODS, (step,))
    missing = tmp_path / "runs" / "stress"
    with pytest.raises(ReportError, match=str(missing)):
        write_report(sweep, missing, tmp_path / "output")
