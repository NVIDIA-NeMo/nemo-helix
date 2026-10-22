# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.customization_common.schemas.file_io import UploadStats
from nhx.customization_common.tasks.file_io import callbacks
from nhx.customization_common.tasks.file_io.callbacks import FileUploadProgressCallback


class RecordingReporter:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    def update_progress(
        self,
        status: HelixJobStatus,
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
        error_stack: str | None = None,
    ) -> None:
        self.reports.append(status_details or {})


def test_a_long_upload_reports_bytes_in_flight_at_a_bounded_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A file that takes hours must keep the task fresh, or the step's staleness check reads it as stalled."""
    (tmp_path / "config.json").write_bytes(b"x" * 100)
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
    now = [0.0]
    monkeypatch.setattr(callbacks.time, "monotonic", lambda: now[0])
    reporter = RecordingReporter()
    parent = FileUploadProgressCallback(reporter, tmp_path, "default/out", UploadStats())

    with parent.branched(str(tmp_path / "config.json"), "default/out/config.json") as small:
        small.set_size(100)
        small.relative_update(100)
    with parent.branched(str(tmp_path / "model.safetensors"), "default/out/model.safetensors") as large:
        large.set_size(1000)
        now[0] = 30.0
        large.relative_update(300)
        now[0] = 61.0
        with caplog.at_level("INFO", logger=callbacks.__name__):
            large.relative_update(300)
        now[0] = 90.0
        large.relative_update(400)

    in_flight = reporter.reports[2]
    assert len(reporter.reports) == 4
    assert in_flight["uploaded_files"] == 1
    assert in_flight["uploaded_bytes"] == 700
    assert in_flight["current_file"] == "model.safetensors"
    assert in_flight["current_file_uploaded_bytes"] == 600
    assert in_flight["current_file_size"] == 1000
    assert "Uploading model.safetensors: 600/1000 bytes (60%)" in caplog.text
    assert reporter.reports[3]["uploaded_files"] == 2
    assert reporter.reports[3]["uploaded_bytes"] == 1100
