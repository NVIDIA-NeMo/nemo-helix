# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from nemo_customizer.skills import get_skills_path

SCRIPTS = get_skills_path() / "nemo-customizer" / "scripts"


def _load_job_health() -> ModuleType:
    spec = importlib.util.spec_from_file_location("job_health", SCRIPTS / "job_health.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job_health = _load_job_health()


def _points(*values: float) -> list[dict[str, float]]:
    return [{"step": step, "epoch": 0, "value": value} for step, value in enumerate(values, start=1)]


def _status(details: dict[str, Any], state: str = "active") -> dict[str, Any]:
    return {
        "status": state,
        "status_details": {"phase": "training"},
        "steps": [
            {"name": "download", "tasks": [{"status_details": {"phase": "done"}}]},
            {"name": "training", "tasks": [{"status_details": {"phase": "training", **details}}]},
        ],
    }


def _report(details: dict[str, Any]) -> tuple[str, list[str]]:
    status = _status(details)
    found = job_health.training_details(status)
    return job_health.summary_line(status, found), job_health.findings(found)


def test_sft_summary_shows_progress_and_losses() -> None:
    summary, findings = _report(
        {"step": 8, "max_steps": 94, "epoch": 1, "num_epochs": 2, "train_loss": 1.25, "val_loss": 1.5}
    )
    assert summary == "status=active phase=training step=8/94 epoch=1/2 train_loss=1.25 val_loss=1.5"
    assert findings == []


def test_latest_value_falls_back_to_the_history() -> None:
    summary, _ = _report({"metrics": {"train_loss": _points(2.0, 1.5)}})
    assert "train_loss=1.5" in summary


def test_null_top_level_value_falls_back_to_the_history() -> None:
    _, findings = _report({"train_loss": None, "metrics": {"train_loss": _points(2.0, float("nan"))}})
    assert len(findings) == 1 and findings[0].startswith("ALERT: train_loss is nan")


def test_no_training_task_yet_reports_status_only() -> None:
    status = {"status": "active", "status_details": {"phase": "downloading"}, "steps": []}
    assert job_health.summary_line(status, job_health.training_details(status)) == "status=active phase=downloading"


@pytest.mark.parametrize("name", ["train_loss", "val_loss"])
def test_non_finite_loss_raises_an_alert(name: str) -> None:
    _, findings = _report({name: float("nan")})
    assert len(findings) == 1 and findings[0].startswith(f"ALERT: {name} is nan")


def test_grpo_summary_shows_reward_and_health() -> None:
    summary, findings = _report(
        {"train_reward": 0.4, "val_accuracy": 0.35, job_health.GRPO_MIXED_GROUPS: 0.6, "train_truncation_rate": 0.02}
    )
    assert "train_reward=0.4" in summary and f"{job_health.GRPO_MIXED_GROUPS}=0.6" in summary
    assert "train_loss" not in summary
    assert findings == []


def test_grpo_zero_reward_variance_raises_an_alert() -> None:
    _, findings = _report({"train_reward": 1.0, "metrics": {job_health.GRPO_MIXED_GROUPS: _points(0.4, 0.0, 0.0, 0.0)}})
    assert [line.split(":")[0] for line in findings] == ["ALERT"]


def test_one_zero_variance_step_is_not_enough() -> None:
    _, findings = _report({"train_reward": 1.0, "metrics": {job_health.GRPO_MIXED_GROUPS: _points(0.4, 0.3, 0.0)}})
    assert findings == []


def test_zero_variance_without_history_only_warns() -> None:
    """One reading cannot show a sustained run, so it must not stop the poll."""
    early = {"train_reward": 1.0, job_health.GRPO_MIXED_GROUPS: 0.0, "step": 1}
    assert _report(early)[1] == []
    findings = _report({**early, "step": job_health.RUN_LENGTH})[1]
    assert [line.split(":")[0] for line in findings] == ["WARN"]


def test_low_training_loss_warns_of_memorizing() -> None:
    _, findings = _report({"train_loss": 0.05})
    assert len(findings) == 1 and findings[0].startswith("WARN: train_loss is 0.05")


def test_rising_validation_loss_warns_of_overfitting() -> None:
    _, findings = _report({"train_loss": 0.9, "metrics": {"val_loss": _points(1.0, 1.1, 1.2, 1.3)}})
    assert len(findings) == 1 and findings[0].startswith("WARN: val_loss rose")


def _run_poll(
    tmp_path: Path, *statuses: dict[str, Any], args: tuple[str, ...] = ("0",)
) -> subprocess.CompletedProcess[str]:
    """Run the poll script against a fake ``nemo`` that returns ``statuses`` in turn, then the last one."""
    for index, status in enumerate(statuses):
        (tmp_path / f"status{index}.json").write_text(json.dumps(status))
    fake_nemo = tmp_path / "nemo"
    fake_nemo.write_text(
        "#!/usr/bin/env bash\n"
        f'n=$(cat "{tmp_path}/calls" 2>/dev/null || echo 0); echo $((n + 1)) > "{tmp_path}/calls"\n'
        f'cat "{tmp_path}/status$(( n < {len(statuses) - 1} ? n : {len(statuses) - 1} )).json"\n'
    )
    fake_nemo.chmod(fake_nemo.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["bash", str(SCRIPTS / "poll_customization_job.sh"), "automodel-abc", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_poll_stops_on_an_alert_while_the_job_runs(tmp_path: Path) -> None:
    result = _run_poll(tmp_path, _status({"train_loss": float("nan")}))
    assert result.returncode == 3
    assert "ALERT: train_loss is nan" in result.stdout


def test_poll_exits_cleanly_when_the_job_completes(tmp_path: Path) -> None:
    result = _run_poll(tmp_path, _status({"train_loss": 0.8}, state="completed"))
    assert result.returncode == 0
    assert "status=completed" in result.stdout and "train_loss=0.8" in result.stdout


def test_poll_continues_past_an_alert_when_asked(tmp_path: Path) -> None:
    """The flag works in either position, including with no interval before it."""
    alert, done = _status({"train_loss": float("nan")}), _status({"train_loss": 0.8}, state="completed")
    result = _run_poll(tmp_path, alert, done, args=("--continue-on-alert", "0"))
    assert result.returncode == 0
    assert "ALERT: train_loss is nan" in result.stdout and "status=completed" in result.stdout
