# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the command_invoked telemetry hook at the root command choke point."""

from __future__ import annotations

from unittest.mock import patch

import click
import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.core.help_formatter import NhxErrorHandlingMixin
from nemo_helix_ext.cli.telemetry import runtime
from nemo_helix_ext.cli.telemetry.events import TaskStatusEnum
from typer.testing import CliRunner

runner = CliRunner()


class TestCommandInvokedHook:
    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_successful_command_emits_completed(self, emit):
        result = runner.invoke(app, ["docs", "--list"])
        assert result.exit_code == 0
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.command.startswith("docs")
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.duration_sec >= 0

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_agent_mode_flag_captured(self, emit):
        runner.invoke(app, ["--agent-mode", "docs", "--list"])
        assert emit.call_args[0][0].agent_mode is True

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_failing_command_emits_error(self, emit):
        with patch("nemo_helix_ext.cli.commands.docs._list_docs", side_effect=RuntimeError):
            runner.invoke(app, ["docs", "--list"])
        assert emit.call_count == 1
        assert emit.call_args[0][0].task_status == TaskStatusEnum.ERROR

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_click_exception_emits_error_without_traceback(self, emit):
        with patch("nemo_helix_ext.cli.commands.docs._list_docs", side_effect=click.ClickException("broken")):
            result = runner.invoke(app, ["docs", "--list"])
        assert result.exit_code == 1
        assert "Error: broken" in result.stderr
        assert "Traceback" not in result.output
        assert emit.call_count == 1
        assert emit.call_args[0][0].task_status == TaskStatusEnum.ERROR

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_keyboard_interrupt_emits_canceled(self, emit):
        class InterruptingBase:
            def main(self, **_kwargs):
                runtime.state.started = True
                runtime.state.command_parts = ["docs"]
                raise KeyboardInterrupt

        class InterruptingCommand(NhxErrorHandlingMixin, InterruptingBase):
            _nhx_emit_command_invoked = True

        with pytest.raises(KeyboardInterrupt):
            InterruptingCommand().main(args=[], standalone_mode=True)
        assert emit.call_count == 1
        assert emit.call_args[0][0].task_status == TaskStatusEnum.CANCELED

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_no_telemetry_flag_suppresses(self, emit):
        runner.invoke(app, ["--no-telemetry", "docs", "--list"])
        emit.assert_not_called()

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_bare_help_does_not_emit(self, emit):
        runner.invoke(app, ["--help"])
        emit.assert_not_called()

    @patch("nemo_helix_ext.cli.telemetry.emit.emit_event")
    def test_group_help_does_not_emit(self, emit):
        """``nemo <group> --help`` reads help; it is not usage and must not emit."""
        result = runner.invoke(app, ["docs", "--help"])
        assert result.exit_code == 0
        emit.assert_not_called()
