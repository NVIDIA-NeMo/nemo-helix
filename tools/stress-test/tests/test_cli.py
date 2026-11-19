# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from config import PodSize, Step, Sweep
from main import _finish, app, configure_logging
from records import marker_path, write_json
from typer.testing import CliRunner

runner = CliRunner()


def test_list_prints_sweeps_and_steps() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "sweep.yaml" in result.stdout
    assert "exit-01" in result.stdout
    assert "auth-25" in result.stdout


def test_naked_run_prints_help() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "Usage" in result.output
    assert "--context" in result.output


def test_short_help_matches_long_help() -> None:
    invocations = [[], ["run"], ["list"], ["publish"], ["report"], ["clean-jobs"], ["clean-results"]]
    for args in invocations:
        long = runner.invoke(app, [*args, "--help"])
        short = runner.invoke(app, [*args, "-h"])
        assert long.exit_code == 0
        assert short.exit_code == 0
        assert short.output == long.output


class _Files:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def write_file(self, path: str, data: bytes) -> None:
        self.files[path] = data


def test_debug_flag_controls_http_logging() -> None:
    names = ("stress", "httpx", "httpcore", "urllib3")
    previous = {name: (logging.getLogger(name).level, list(logging.getLogger(name).handlers)) for name in names}
    try:
        configure_logging(False)
        assert logging.getLogger("stress").level == logging.INFO
        assert logging.getLogger("httpx").level == logging.WARNING
        configure_logging(True)
        assert logging.getLogger("stress").level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert logging.getLogger("httpcore").handlers
    finally:
        for name, (level, handlers) in previous.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            for handler in list(logger.handlers):
                if handler not in handlers:
                    logger.removeHandler(handler)


def test_run_help_offers_debug() -> None:
    result = runner.invoke(app, ["run", "-h"])
    assert "--debug" in result.output
    assert "--image" in result.output
    assert "--force" in result.output
    assert "--workspaces" in result.output


def test_run_rejects_an_empty_image() -> None:
    result = runner.invoke(app, ["run", "--context", "local", "--image", " "])
    assert result.exit_code == 1
    assert "--image is empty" in result.output


def test_finish_writes_the_report_and_says_what_each_step_did(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    pods = {
        "exit": PodSize("100m", "256Mi", "500m", "512Mi"),
        "idle": PodSize("100m", "256Mi", "500m", "512Mi"),
        "hammer": PodSize("500m", "512Mi", "1", "1Gi"),
    }
    ran = Step("exit-01", "exit", 1, None, None, None)
    done = Step("exit-10", "exit", 1, None, None, None)
    pending = Step("idle-01", "idle", 1, None, None, None)
    skipped = Step("auth-01", "hammer", 1, 60, "auth", 1.0)
    sweep = Sweep("stress", 8, pods, (ran, done, pending, skipped))
    run_dir = tmp_path / "runs"
    write_json(
        run_dir / "steps" / ran.id / "stress-exit-01-1.record.json",
        {"job_name": "stress-exit-01-1", "submit": "accepted", "status_history": ["error"]},
    )
    write_json(marker_path(run_dir, done.id), dict(done.measurement()))
    files = _Files()
    output = tmp_path / "output"
    with caplog.at_level(logging.DEBUG, logger="stress"):
        _finish(sweep, run_dir, output, ran=[ran.id], skipped=[skipped.id], only=None, cluster=files)
    err = capsys.readouterr().err
    assert "exit-01: 1 submitted, 1 error, 0 with results; no marker" in caplog.text
    assert "exit-10: already complete" in caplog.text
    assert "idle-01: not run" in caplog.text
    assert "auth-01: skipped because auth is disabled" in caplog.text
    assert "already complete" not in err
    assert f"report: {output / 'report.md'}" in err
    assert "report copied to the stress-stress fileset" in err
    assert (output / "report.md").is_file()
    assert "reports/report.md" in files.files


def test_finish_marks_steps_outside_one_step_rerun_as_not_selected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    pods = {"exit": PodSize("100m", "256Mi", "500m", "512Mi")}
    chosen = Step("exit-01", "exit", 1, None, None, None)
    other = Step("exit-10", "exit", 1, None, None, None)
    sweep = Sweep("stress", 8, pods, (chosen, other))
    run_dir = tmp_path / "runs"
    write_json(run_dir / "meta.json", {"backend": "docker"})
    with caplog.at_level(logging.DEBUG, logger="stress"):
        _finish(sweep, run_dir, tmp_path / "output", ran=[chosen.id], skipped=[], only=chosen.id, cluster=None)
    assert "exit-01: not run" in caplog.text
    assert "exit-10: not selected" in caplog.text
    assert "report copied" not in capsys.readouterr().err


def test_run_rejects_a_blank_context() -> None:
    result = runner.invoke(app, ["run", "--context", " "])
    assert result.exit_code == 1
    assert "--context is required" in result.output
