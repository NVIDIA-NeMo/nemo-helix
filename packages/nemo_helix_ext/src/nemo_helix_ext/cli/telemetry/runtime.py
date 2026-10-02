# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-invocation telemetry state bridged between the Typer callback and the Click main wrapper.

The Typer ``@app.callback`` sees the parsed context (command name, agent mode, opt-out
flag) but not the command outcome. The ``NhxErrorHandlingMixin.main`` wrapper sees the
outcome and duration but not the parsed flags. This module is the single small piece of
state both sides read, so exactly one ``command_invoked`` event is emitted per invocation
from the root command path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import click

from nemo_helix_ext.cli.telemetry.events import TaskStatusEnum

logger = logging.getLogger(__name__)


@dataclass
class _InvocationState:
    command_parts: list[str] = field(default_factory=list)
    agent_mode: bool = False
    started: bool = False
    opted_out: bool = False


state = _InvocationState()


def reset() -> None:
    """Clear invocation state. Called at the top of the root ``main()`` so CliRunner
    invocations sharing a process never leak command names or flags between calls."""
    state.command_parts = []
    state.agent_mode = False
    state.started = False
    state.opted_out = False


def on_callback(ctx: click.Context, *, no_telemetry: bool) -> None:
    """Capture command name + agent mode from the root Typer callback.

    Best effort: a telemetry bug must never break ``nemo <anything>``.
    """
    try:
        from nemo_helix_ext.cli.telemetry.emit import maybe_print_first_run_notice, set_invocation_opt_out

        set_invocation_opt_out(no_telemetry)
        state.opted_out = no_telemetry
        state.started = True
        state.agent_mode = bool(getattr(ctx.obj, "agent_mode", False))
        invoked = getattr(ctx, "invoked_subcommand", None)
        if invoked:
            state.command_parts = [invoked]
        maybe_print_first_run_notice()
    except Exception:  # noqa: BLE001
        # Fail closed: if telemetry setup breaks, suppress this invocation's event.
        state.opted_out = True
        logger.debug("telemetry on_callback failed", exc_info=True)


def note_subcommand(name: str | None) -> None:
    """Append a resolved sub-subcommand name (e.g. the ``list`` in ``workspaces list``)."""
    if state.started and name and name not in state.command_parts:
        state.command_parts.append(name)


def emit_command_invoked(task_status: TaskStatusEnum, duration_sec: float) -> None:
    """Emit exactly one ``command_invoked`` event. Best effort; never raises."""
    try:
        if state.opted_out:
            return
        if not (state.started and state.command_parts):
            # Bare ``--help`` and usage errors before a command resolves never emit.
            return
        from nemo_helix_ext.cli.telemetry.emit import emit_event
        from nemo_helix_ext.cli.telemetry.events import CommandInvokedEvent

        event = CommandInvokedEvent(
            command=" ".join(state.command_parts),
            duration_sec=max(0.0, duration_sec),
            agent_mode=state.agent_mode,
            task_status=task_status,
        )
        emit_event(event)
    except Exception:  # noqa: BLE001
        logger.debug("telemetry emit_command_invoked failed", exc_info=True)
