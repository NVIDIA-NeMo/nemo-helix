# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo data-designer create`` output rendering.

The job submit path is non-streaming: the framework's single-value driver
calls ``on_frame`` exactly once with the decoded ``PlatformJobResponse``.
These tests drive ``CreateRenderer`` through that lifecycle directly.
"""

from __future__ import annotations

import io
import json
from typing import Any

import nemo_data_designer_plugin.cli.renderers as renderers_mod
import pytest
from nemo_data_designer_plugin.cli.renderers import CreateRenderer
from nemo_platform_plugin.cli_renderer import RendererContext
from rich.console import Console


def _job_response(**overrides: Any) -> dict[str, Any]:
    """A representative submit response, trimmed to the fields we assert on."""
    frame: dict[str, Any] = {
        "id": "platform-job-T8XuYFhEz1GS384LjELhMT",
        "attempt_id": "platform-job-attempt-abc123",
        "name": "nemo-data-designer-p-cdrb2nfx",
        "workspace": "default",
        "source": "data-designer",
        "status": "created",
        "fileset": "fileset-xyz",
        "spec": {
            "job_config": {"num_records": 30, "config": {"columns": ["a", "b"]}},
            "model_configs": [{"alias": "big-model"}],
        },
        "platform_spec": {"steps": [{"name": "data-designer-job", "config": {"deeply": "nested"}}]},
        "custom_fields": {"_nemo_telemetry": {"session_id": "cc676ea7d381433dabf59e515c0ed5ca"}},
    }
    frame.update(overrides)
    return frame


def _render(frame: Any, *, cli_kwargs: dict[str, Any] | None = None) -> str:
    """Drive the full renderer lifecycle and return everything it printed.

    The renderer writes through ``data_designer.cli.ui``'s module-level
    console (imported into ``renderers``), not ``ctx.console``, so the
    ``_capture`` fixture swaps that module attribute for a recording one.
    """
    ctx = RendererContext(
        console=_recorder,
        cli_kwargs=cli_kwargs if cli_kwargs is not None else {"workspace": "default"},
        verb="submit",
        is_local=False,
    )
    renderer = CreateRenderer()
    renderer.on_start(ctx=ctx)
    renderer.on_frame(frame, ctx=ctx)
    renderer.on_complete(ctx=ctx)
    return _buffer.getvalue()


_buffer = io.StringIO()
_recorder = Console(file=_buffer, width=200, force_terminal=False, no_color=True, soft_wrap=True)


@pytest.fixture(autouse=True)
def _capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the renderer's console and ui helpers into a recording buffer."""
    _buffer.seek(0)
    _buffer.truncate(0)

    monkeypatch.setattr(renderers_mod, "console", _recorder)
    monkeypatch.setattr(renderers_mod, "print_header", lambda text: _recorder.print(text))
    monkeypatch.setattr(renderers_mod, "print_success", lambda msg: _recorder.print(msg))
    monkeypatch.setattr(renderers_mod, "print_error", lambda msg: _recorder.print(msg))


def test_prints_job_name_and_follow_up_commands() -> None:
    out = _render(_job_response())

    assert "nemo-data-designer-p-cdrb2nfx" in out
    assert "nemo jobs get nemo-data-designer-p-cdrb2nfx" in out
    assert "nemo jobs watch nemo-data-designer-p-cdrb2nfx" in out
    assert "nemo jobs results list nemo-data-designer-p-cdrb2nfx" in out


def test_omits_the_bulky_fields() -> None:
    out = _render(_job_response())

    assert "platform_spec" not in out
    assert "custom_fields" not in out
    assert "_nemo_telemetry" not in out
    assert "cc676ea7d381433dabf59e515c0ed5ca" not in out
    assert "attempt_id" not in out
    assert "platform-job-T8XuYFhEz1GS384LjELhMT" not in out
    assert "deeply" not in out


def test_shows_record_count_from_spec() -> None:
    out = _render(_job_response())

    assert "Records" in out
    assert "30" in out


def test_falls_back_to_cli_spec_for_record_count() -> None:
    """When the response omits the spec, the submitted --spec JSON still has it."""
    frame = _job_response()
    del frame["spec"]

    out = _render(
        frame,
        cli_kwargs={"workspace": "default", "spec": json.dumps({"num_records": 7, "config": {}})},
    )

    assert "Records" in out
    assert "7" in out


def test_omits_record_count_when_unavailable() -> None:
    frame = _job_response()
    del frame["spec"]

    out = _render(frame, cli_kwargs={"workspace": "default"})

    assert "Records" not in out
    # The job is still identifiable, which is the point of the ticket.
    assert "nemo-data-designer-p-cdrb2nfx" in out


def test_default_workspace_adds_no_flag() -> None:
    out = _render(_job_response())

    assert "--workspace" not in out


def test_non_default_workspace_is_appended_to_commands() -> None:
    out = _render(_job_response(workspace="research"), cli_kwargs={"workspace": "research"})

    assert "nemo jobs get nemo-data-designer-p-cdrb2nfx --workspace research" in out
    assert "nemo jobs watch nemo-data-designer-p-cdrb2nfx --workspace research" in out
    assert "nemo jobs results list nemo-data-designer-p-cdrb2nfx --workspace research" in out


def test_workspace_falls_back_to_cli_kwargs() -> None:
    frame = _job_response()
    del frame["workspace"]

    out = _render(frame, cli_kwargs={"workspace": "research"})

    assert "--workspace research" in out


def test_surfaces_error_details_when_present() -> None:
    out = _render(_job_response(error_details={"reason": "quota exceeded"}))

    assert "quota exceeded" in out


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param({"id": "platform-job-abc", "status": "created"}, id="missing-name"),
        pytest.param({"name": "", "id": "platform-job-abc"}, id="empty-name"),
        pytest.param("unexpected string payload", id="not-a-dict"),
        pytest.param(None, id="none"),
    ],
)
def test_unexpected_shape_falls_back_to_raw_dump(frame: Any) -> None:
    """Never hide the payload when we can't find the job's identity in it."""
    out = _render(frame)

    assert "nemo jobs list" in out
