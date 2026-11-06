# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from config import PodSize, Step, Sweep
from records import write_json
from report import ReportError, write_report

PODS = {
    "exit": PodSize("100m", "256Mi", "500m", "512Mi"),
    "idle": PodSize("100m", "256Mi", "500m", "512Mi"),
    "hammer": PodSize("500m", "512Mi", "1", "1Gi"),
    "barrier": PodSize("100m", "256Mi", "500m", "512Mi"),
}


def test_partial_step_and_redaction(tmp_path: Path) -> None:
    step = Step("files-01", "hammer", 1, 60, "files", 5.0)
    sweep = Sweep("stress", 8, PODS, (step,))
    run_dir = tmp_path / "runs"
    step_dir = run_dir / "steps" / step.id
    write_json(
        step_dir / "stress-files-01-1.record.json",
        {
            "job_name": "stress-files-01-1",
            "submit": "accepted",
            "status_history": ["completed"],
            "download_succeeded": False,
            "platform_errors": ["secret-fake-access-key"],
        },
    )
    (step_dir / "stress-files-01-1.jsonl").write_text(
        '{"error":"secret-fake-access-key","latency":1,"operation":"noop"}\n'
    )
    (run_dir / "steps" / "retired").mkdir(parents=True)
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    document = (output / "report.html").read_text()
    assert "partial | yes" in markdown.replace(" ", "") or "| yes |" in markdown
    assert "| CPU request | 500m |" in markdown
    assert "| CPU limit | 1 |" in markdown
    assert "Pod sizes" not in markdown
    assert "secret-fake-access-key" not in markdown
    assert "secret-fake-access-key" not in document
    assert "[redacted]" in markdown
    assert "status_history" not in markdown
    assert '"record"' not in markdown
    assert "retired" not in markdown
    assert "<script" not in document.lower()


def test_report_records_backend_and_auth(tmp_path: Path) -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    sweep = Sweep("stress", 8, PODS, (step,))
    run_dir = tmp_path / "runs"
    write_json(
        run_dir / "meta.json",
        {
            "context": "local",
            "base_url": "http://127.0.0.1:8080",
            "profile": "gpu",
            "backend": "docker",
            "auth": "disabled",
        },
    )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert "`docker`" in markdown
    assert "`profile default`" in markdown
    assert "`disabled`" in markdown


def test_ran_steps_stay_in_one_table(tmp_path: Path) -> None:
    first = Step("exit-01", "exit", 1, None, None, None)
    second = Step("exit-10", "exit", 10, None, None, None)
    sweep = Sweep("stress", 8, PODS, (first, second))
    run_dir = tmp_path / "runs"
    for step, name in ((first, "stress-exit-01-1"), (second, "stress-exit-10-01")):
        write_json(
            run_dir / "steps" / step.id / f"{name}.record.json",
            {
                "job_name": name,
                "submit": "accepted",
                "status_history": ["completed"],
                "submitted_at": 0,
                "finished_at": 2,
            },
        )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    table, _, rest = markdown.partition("## exit")
    del table
    body, _, after = rest.partition("## idle")
    assert "Pod sizes" not in markdown
    assert "| CPU request | 100m |" in body
    assert "| Step | Count | Accepted | Terminal | Scheduling | Throughput |" in body
    rows = [line for line in body.splitlines() if line.startswith("| exit-")]
    assert rows == [
        "| exit-01 | 1 | 1 | completed=1 | 0 | 0.500 jobs/s |",
        "| exit-10 | 10 | 1 | completed=1 | 0 | 0.500 jobs/s |",
    ]
    assert "history" not in markdown
    assert '"record"' not in markdown
    assert "No steps in the loaded sweep." in after
    assert "Idle runs `/bin/bash -c`" in after


def test_idle_throughput_uses_submit_to_finish(tmp_path: Path) -> None:
    step = Step("idle-10", "idle", 1000, 10, None, None)
    sweep = Sweep("stress", 8, PODS, (step,))
    run_dir = tmp_path / "runs"
    write_json(
        run_dir / "steps" / step.id / "stress-idle-10-0001.record.json",
        {
            "job_name": "stress-idle-10-0001",
            "submit": "accepted",
            "status_history": ["completed"],
            "submitted_at": 0,
            "finished_at": 10,
        },
    )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert "| Count | 1000 |" in markdown
    assert "| Duration | 10 |" in markdown
    assert "| idle-10 | 1 | completed=1 | 0 | 0.100 jobs/s |" in markdown
    assert "count divided by the duration" in markdown


def test_barrier_report_distinguishes_release_from_timeout(tmp_path: Path) -> None:
    step = Step("barrier-10", "barrier", 2, None, None, None, timeout_seconds=30, workspaces=2)
    sweep = Sweep("stress", 8, PODS, (step,))
    run_dir = tmp_path / "runs"
    for name, reason in (("stress-barrier-10-1", "barrier"), ("stress-barrier-10-2", "timeout")):
        write_json(
            run_dir / "steps" / step.id / f"{name}.record.json",
            {"job_name": name, "submit": "accepted", "status_history": ["completed"], "exit_reason": reason},
        )
    write_json(
        run_dir / "steps" / step.id / "cohort.json",
        {
            "workspaces": ["stress-stress", "stress-stress-2"],
            "ready_count": 2,
            "ready_seconds": 3,
            "submit_seconds": 1.5,
            "released": True,
            "peak_active": 2,
            "release_to_finalized_seconds": 4,
            "by_workspace": {
                "stress-stress": {
                    "accepted": 2,
                    "ready": 2,
                    "peak_active": 2,
                    "barrier": 1,
                    "timeout": 1,
                    "unknown": 0,
                },
                "stress-stress-2": {
                    "accepted": 0,
                    "ready": 0,
                    "peak_active": 0,
                    "barrier": 0,
                    "timeout": 0,
                    "unknown": 0,
                },
            },
        },
    )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert "| Jobs per workspace | 2 |" in markdown
    assert "| Workspaces | 2 |" in markdown
    assert "| Timeout (s) | 30 |" in markdown
    assert "| barrier-10 | 2 | 2 | 2 | 1.500s | 3.000s | 4.000s | 1 | 1 | 0 |" in markdown
    assert "Barrier means the release file was seen and the job exited 0." in markdown
    assert "Timeout means the job reached its timeout first and exited 1." in markdown
    assert "`stress-stress`: accepted=2 ready=2 peak=2 barrier=1 timeout=1 unknown=0" in markdown
    assert "`stress-stress-2`: accepted=0 ready=0 peak=0 barrier=0 timeout=0 unknown=0" in markdown


def test_varying_workspaces_stay_in_the_comparison(tmp_path: Path) -> None:
    one = Step("barrier-10", "barrier", 10, None, None, None, timeout_seconds=180, workspaces=1)
    two = Step("barrier-100-w2", "barrier", 100, None, None, None, timeout_seconds=600, workspaces=2)
    sweep = Sweep("stress", 8, PODS, (one, two))
    run_dir = tmp_path / "runs"
    for step in (one, two):
        write_json(
            run_dir / "steps" / step.id / f"stress-{step.id}-1.record.json",
            {
                "job_name": f"stress-{step.id}-1",
                "submit": "accepted",
                "status_history": ["completed"],
                "exit_reason": "barrier",
            },
        )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert "| CPU request | 100m |" in markdown
    assert "| Jobs per workspace |" not in markdown.split("## files", 1)[0].split("| Step |", 1)[0]
    header = next(line for line in markdown.splitlines() if line.startswith("| Step |"))
    assert (
        header
        == "| Step | Jobs per workspace | Workspaces | Timeout (s) | Accepted | Ready | Peak running | Submit | Ready time | Release to finalized | Barrier | Timeout | Unknown |"
    )


def test_operation_rates_share_one_table(tmp_path: Path) -> None:
    first = Step("files-01", "hammer", 1, 60, "files", 5.0)
    second = Step("files-10", "hammer", 10, 60, "files", 5.0)
    sweep = Sweep("stress", 8, PODS, (first, second))
    run_dir = tmp_path / "runs"
    _write_hammer(run_dir, first, ["file_upload", "file_upload", "file_download"], 10)
    _write_hammer(run_dir, second, ["file_upload", "file_upload", "file_upload", "file_upload"], 10)
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert (
        "| Step | Count | Accepted | Terminal | Scheduling | Missing | Partial | Zip fetch (s) | Throughput |"
        in markdown
    )
    assert "### files-01 latency" not in markdown
    assert "| Step | file_upload | file_download |" in markdown
    assert "| files-01 | 0.200 | 0.100 |" in markdown
    assert "| files-10 | 0.400 |  |" in markdown


def _write_hammer(run_dir: Path, step: Step, operations: list[str], elapsed: int) -> None:
    directory = run_dir / "steps" / step.id
    name = f"stress-{step.id}-1"
    write_json(
        directory / f"{name}.record.json",
        {"job_name": name, "submit": "accepted", "status_history": ["completed"], "download_succeeded": True},
    )
    lines = [json.dumps({"record": "operation", "operation": operation, "latency": 0.01}) for operation in operations]
    lines.append(
        json.dumps(
            {"record": "header", "started_at": 0, "ended_at": elapsed, "achieved_rate": 0.3, "code_fetch_seconds": 0.5}
        )
    )
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n")


def test_unrun_service_steps_are_omitted(tmp_path: Path) -> None:
    ran = Step("exit-01", "exit", 1, None, None, None)
    waiting = Step("files-01", "hammer", 1, 60, "files", 5.0)
    sweep = Sweep("stress", 8, PODS, (ran, waiting))
    run_dir = tmp_path / "runs"
    write_json(
        run_dir / "steps" / ran.id / "stress-exit-01-1.record.json",
        {"job_name": "stress-exit-01-1", "submit": "accepted", "status_history": ["error"]},
    )
    output = tmp_path / "output"
    write_report(sweep, run_dir, output)
    markdown = (output / "report.md").read_text()
    assert "exit-01" in markdown
    assert "files-01" not in markdown
    assert "No steps in this group have run." in markdown


def test_missing_bundle_names_the_directory(tmp_path: Path) -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    sweep = Sweep("stress", 8, PODS, (step,))
    missing = tmp_path / "runs" / "stress"
    with pytest.raises(ReportError, match=str(missing)):
        write_report(sweep, missing, tmp_path / "output")
