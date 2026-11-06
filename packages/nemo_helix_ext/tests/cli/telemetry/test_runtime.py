# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import UUID

import click
import pytest
from nemo_helix_ext.cli.telemetry import emit, runtime
from nemo_helix_plugin.telemetry.events import (
    CommandInvokedEvent,
    OnboardingStepEvent,
    TaskStatusEnum,
    command_tool_category,
    job_tool_category,
)


@pytest.mark.parametrize(
    "value,expected",
    [(None, "unknown"), ("agent", "agent"), ("human", "human"), ("private-agent", "unknown"), (" AGENT ", "agent")],
)
def test_command_origin_is_explicit_and_bounded(monkeypatch, value, expected):
    monkeypatch.delenv("NEMO_INVOCATION_ORIGIN", raising=False)
    if value is not None:
        monkeypatch.setenv("NEMO_INVOCATION_ORIGIN", value)
    runtime.reset()
    runtime.state.started = True
    runtime.state.command_parts = ["guardrail", "check"]
    try:
        with patch.object(emit, "emit_event") as send:
            runtime.emit_command_invoked(TaskStatusEnum.COMPLETED, 0.912)
        event = send.call_args.args[0]
        assert event.invocation_origin == expected
        assert event.agent_mode == (expected == "agent")
        assert event.tool_category == "guardrails"
    finally:
        runtime.reset()


@pytest.mark.parametrize(
    "command,expected",
    [
        ("chat", "inference"),
        ("guardrail check", "guardrails"),
        ("guardrail configs list", "unknown"),
        ("models list", "unknown"),
        ("agents optimize list-strategies", "unknown"),
        ("agents optimize run-strategy", "agent-optimization"),
        ("optimize run-strategy", "unknown"),
        ("data-designer create", "data-designer"),
        ("data-designer retrieval-generate", "data-designer"),
        ("data-designer retrieval generate", "unknown"),
        ("private-customer run", "unknown"),
    ],
)
def test_command_categories_are_exact_operations(command, expected):
    assert command_tool_category(command) == expected
    runtime.reset()
    parts = command.split()
    runtime.state.started = True
    runtime.state.command_parts = [parts[0]]
    try:
        for part in parts[1:]:
            runtime.note_subcommand(part)
        with patch.object(emit, "emit_event") as send:
            runtime.emit_command_invoked(TaskStatusEnum.COMPLETED, 0.1)
        event = send.call_args.args[0]
        assert event.command == command
        assert event.tool_category == expected
    finally:
        runtime.reset()


@pytest.mark.parametrize(
    "source,plugins,expected",
    [
        ("evaluation", [], "evals"),
        ("evaluator", [], "evals"),
        ("evals", [], "evals"),
        ("audit", [], "garak"),
        ("garak", [], "garak"),
        ("customization", [], "post-training"),
        ("post-training", [], "post-training"),
        ("private-customer", ["evals"], "evals"),
        ("private-customer", [], "unknown"),
        ("private-customer", ["agent-optimization-plugin-run-strategy"], "agent-optimization"),
        ("data-designer", ["data-designer"], "data-designer"),
        ("evaluation", ["anonymizer"], "unknown"),
    ],
)
def test_job_categories_do_not_leak_or_arbitrarily_attribute(source, plugins, expected):
    assert job_tool_category(source, plugins) == expected


@pytest.mark.parametrize(
    "failure,status",
    [
        (None, "completed"),
        (RuntimeError("failure"), "error"),
        (click.exceptions.Exit(0), "completed"),
        (click.exceptions.Exit(1), "error"),
        (KeyboardInterrupt(), "canceled"),
        (click.Abort(), "canceled"),
    ],
)
def test_onboarding_scopes_attempts_and_preserves_outcomes(failure, status):
    @emit.track_onboarding
    def setup():
        emit.emit_event(OnboardingStepEvent(step="models_discovered", task_status=TaskStatusEnum.COMPLETED))
        if failure is not None:
            raise failure
        return "result"

    with (
        patch.object(emit, "telemetry_opted_in", return_value=True),
        patch.object(emit, "TelemetryHandler") as handler,
        patch.object(emit, "get_session_id", return_value="test-session"),
    ):
        if failure is None:
            assert setup() == "result"
        else:
            with pytest.raises(type(failure)) as error:
                setup()
            assert error.value is failure
        events = [call.args[0] for call in handler.return_value.enqueue.call_args_list]
        assert [e.step for e in events] == ["setup_started", "models_discovered", "setup_finished"]
        assert events[0].task_status == "undefined"
        assert events[-1].task_status == status
        assert len({e.onboarding_attempt_id for e in events}) == 1
        assert events[0].onboarding_attempt_id.version == 4
        assert len({e.event_id for e in events}) == 3

        handler.reset_mock()
        if failure is None:
            setup()
            next_attempt = handler.return_value.enqueue.call_args_list[0].args[0].onboarding_attempt_id
            assert next_attempt != events[0].onboarding_attempt_id
        handler.reset_mock()
        emit.emit_event(OnboardingStepEvent(step="outside_setup", task_status=TaskStatusEnum.COMPLETED))
        assert handler.return_value.enqueue.call_args.args[0].onboarding_attempt_id is None


def test_onboarding_opt_out_runs_command_without_telemetry():
    @emit.track_onboarding
    def setup():
        return "result"

    with patch.object(emit, "telemetry_opted_in", return_value=False), patch.object(emit, "emit_event") as send:
        assert setup() == "result"
    send.assert_not_called()


def test_event_id_is_random_and_stable_for_repeated_serialization():
    event = CommandInvokedEvent(command="chat", task_status=TaskStatusEnum.COMPLETED, duration_sec=1)
    first = event.model_dump(mode="json", by_alias=True)
    assert UUID(first["eventId"]).version == 4
    assert event.model_dump(mode="json", by_alias=True)["eventId"] == first["eventId"]
    other = CommandInvokedEvent(command="chat", task_status=TaskStatusEnum.COMPLETED, duration_sec=1)
    assert event.event_id != other.event_id
