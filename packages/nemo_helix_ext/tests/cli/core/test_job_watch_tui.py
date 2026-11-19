# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from nemo_helix_ext.cli.core.job_watch_model import DagWatchModel, Pane
from nemo_helix_ext.cli.core.job_watch_renderer import JobWatchRenderResult
from nemo_helix_ext.cli.core.job_watch_tui import DagWatchApp, _fit
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.watch_types import JobLogEvent, JobStatusEvent, JobWatchEvent, JobWorkflowEvent
from nemo_helix_plugin.jobs.workflows import (
    WorkflowFilesetCommit,
    WorkflowNodeView,
    WorkflowScheduling,
    WorkflowStepSuccess,
    WorkflowView,
)
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

NOW = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


def node(
    name: str,
    parents: list[str] | None = None,
    *,
    state: WorkflowScheduling = "waiting_dependencies",
    execution: str | None = None,
) -> WorkflowNodeView:
    return WorkflowNodeView(
        name=name,
        depends_on=parents or [],
        freshness="dirty",
        scheduling_state=state,
        reason=None,
        blocked_by=parents or [],
        requested_run_revision="rev-1",
        active_execution_id=execution,
        latest_execution_id=execution,
        successful_execution_id=None,
        last_outcome=None,
        has_pending_run=False,
        last_inputs_checked_at=NOW,
        last_poll_attempt_at=NOW,
        last_poll_error=None,
    )


def event(nodes: list[WorkflowNodeView], *, attempt: str = "attempt-1") -> JobWorkflowEvent:
    return JobWorkflowEvent(
        kind="workflow",
        job_name="pipeline",
        job_id="job-1",
        attempt_id=attempt,
        workflow=WorkflowView(
            mode="continuous", pass_number=1, control="running", waiting_for_changes=False, degraded=False, nodes=nodes
        ),
    )


def graph() -> list[WorkflowNodeView]:
    roots = [node(f"input-{index:03}") for index in range(100)]
    center = node("train", [item.name for item in roots], state="running", execution="execution-1")
    outputs = [node(f"output-{index:03}", ["train"]) for index in range(100)]
    return roots + [center] + outputs + [node("disconnected")]


def model() -> DagWatchModel:
    result = DagWatchModel()
    result.apply_workflow(event(graph()), NOW)
    return result


def log(execution: str, message: str = "hello") -> JobLogEvent:
    return JobLogEvent(
        kind="log", job_name="pipeline", timestamp=NOW, step_id=execution, task_id="task-1", message=message
    )


def test_large_graph_navigation_and_history() -> None:
    state = model()
    assert state.focus == "train"
    assert state.running == 1
    state.choose_pane(-1)
    state.move(99)
    assert state.selected[Pane.INPUTS] == "input-099"
    state.enter_neighbor()
    assert state.focus == "input-099"
    assert state.orders[Pane.OUTPUTS] == ["train"]
    state.back()
    assert state.focus == "train"
    state.choose_pane(1)
    state.move(99)
    state.enter_neighbor()
    assert state.focus == "output-099"
    assert state.search("DISCONNECTED") == ["disconnected"]


def test_timestamp_only_snapshots_are_not_activity() -> None:
    state = model()
    update = event(graph())
    for item in update.workflow.nodes:
        item.last_inputs_checked_at = NOW + timedelta(seconds=10)
        item.last_poll_attempt_at = NOW + timedelta(seconds=10)
    state.apply_workflow(update, NOW + timedelta(seconds=10))
    assert not state.transitions
    assert not state.activity
    assert state.last_update == NOW + timedelta(seconds=10)


def test_activity_sort_freezes_active_pane_and_preserves_selection() -> None:
    state = model()
    state.choose_pane(-1)
    state.move(5)
    update = event(graph())
    update.workflow.nodes[99].scheduling_state = "running"
    state.apply_workflow(update, NOW + timedelta(seconds=1))
    assert state.orders[Pane.INPUTS][0] == "input-000"
    assert state.selected[Pane.INPUTS] == "input-005"
    assert state.running == 2
    state.reorder()
    assert state.orders[Pane.INPUTS][0] == "input-099"
    assert state.selected[Pane.INPUTS] == "input-005"
    assert "waiting_dependencies → running" in state.latest["input-099"].description


def test_pending_run_and_old_execution_do_not_override_scheduling() -> None:
    state = model()
    state.cycle_execution()
    update = event(graph())
    center = next(item for item in update.workflow.nodes if item.name == "train")
    center.active_execution_id = "execution-2"
    center.latest_execution_id = "execution-2"
    center.has_pending_run = True
    state.apply_workflow(update, NOW + timedelta(seconds=1))
    assert state.execution() == "execution-1"
    state.add_log(log("execution-1", "old"))
    state.add_log(log("execution-2", "new"))
    assert [line.event.message for line in state.visible_logs()] == ["old"]
    state.cycle_execution()
    assert state.execution() == "execution-2"
    assert [line.event.message for line in state.visible_logs()] == ["new"]
    assert "rerun pending" in state.latest["train"].description


def test_log_buffer_bounds_and_scroll_anchor() -> None:
    state = model()
    state.add_log(log("execution-1", "x" * 5000))
    assert state.logs[0].event.message.endswith("[truncated]")
    assert len(state.logs[0].event.message) < 4200
    for index in range(5005):
        state.add_log(log("execution-1", str(index)))
    assert len(state.logs) == 5000
    assert state.dropped_logs == 6
    state.scroll_logs(-10)
    cursor = state.log_cursor
    state.add_log(log("execution-1", "newest"))
    assert state.log_cursor == cursor
    state.all_logs = True
    state.add_log(log("other", "other execution"))
    assert state.visible_logs()[-1].event.step_id == "other"


def test_attempt_switch_is_rejected() -> None:
    state = model()
    with pytest.raises(ValueError, match="different Attempt"):
        state.apply_workflow(event(graph(), attempt="attempt-2"), NOW)


@pytest.mark.parametrize("emoji", [True, False])
@pytest.mark.parametrize("width,height", [(140, 36), (70, 24), (35, 10), (15, 4)])
def test_screen_fits_terminal_and_keeps_parallel_count(emoji: bool, width: int, height: int) -> None:
    state = model()
    state.emoji = emoji
    app = DagWatchApp(state, output=DummyOutput())
    lines = app.screen(Size(rows=height, columns=width))
    assert len(lines) <= height
    assert all(get_cwidth(_fit(line, width)) <= width for line in lines)
    assert "1 running" in "\n".join(lines)
    if not emoji:
        assert "🏃" not in "\n".join(lines)


def test_full_viewport_can_reach_all_neighbors() -> None:
    state = model()
    app = DagWatchApp(state, output=DummyOutput())
    state.choose_pane(-1)
    state.move(99)
    screen = "\n".join(app.screen(Size(rows=28, columns=150)))
    assert "input-099" in screen
    assert "of 100" in screen
    assert "↑ more" in screen
    state.choose_pane(1)
    state.choose_pane(1)
    state.move(99)
    assert "output-099" in "\n".join(app.screen(Size(rows=28, columns=150)))


async def wait_for(predicate: asyncio.Event) -> None:
    await asyncio.wait_for(predicate.wait(), 3)


async def key(app: DagWatchApp, pipe: PipeInput, text: str) -> None:
    processed = asyncio.Event()
    original = app.application.after_render

    def rendered(application: object) -> None:
        processed.set()

    original += rendered
    try:
        pipe.send_text(text)
        app.application.invalidate()
        await wait_for(processed)
        await asyncio.sleep(0.05)
    finally:
        original -= rendered


async def test_real_keyboard_navigation_search_tabs_and_back() -> None:
    state = model()
    with create_pipe_input() as pipe:
        app = DagWatchApp(state, input=pipe, output=DummyOutput())
        ready = asyncio.Event()
        task = asyncio.create_task(app.application.run_async(pre_run=ready.set))
        await wait_for(ready)
        try:
            await key(app, pipe, "\x1b[D")
            assert state.pane == Pane.INPUTS
            await key(app, pipe, "\x1b[B\r")
            assert state.focus == "input-001"
            await key(app, pipe, "\x7f")
            assert state.focus == "train"
            await key(app, pipe, "/disconnected\r")
            assert state.focus == "disconnected"
            await key(app, pipe, "\t")
            assert app.mode == "logs"
            await key(app, pipe, "a")
            assert state.all_logs
            await key(app, pipe, "\t")
            assert app.mode == "status"
            assert state.focus == "disconnected"
            await key(app, pipe, "q")
            assert await asyncio.wait_for(task, 3) == JobWatchRenderResult.INTERRUPTED
        finally:
            if not task.done():
                app.application.exit()
                await task


async def test_slow_logs_do_not_block_status_and_quit_cancels_both() -> None:
    state = model()
    status_updated, logs_started, logs_closed, status_closed = (asyncio.Event() for _ in range(4))

    async def statuses() -> AsyncIterator[JobWatchEvent]:
        try:
            await logs_started.wait()
            update = event(graph())
            update.workflow.nodes[0].scheduling_state = "running"
            yield update
            status_updated.set()
            await asyncio.Event().wait()
        finally:
            status_closed.set()

    async def logs() -> AsyncIterator[JobWatchEvent]:
        try:
            logs_started.set()
            await asyncio.Event().wait()
            yield log("execution-1")
        finally:
            logs_closed.set()

    with create_pipe_input() as pipe:
        app = DagWatchApp(state, input=pipe, output=DummyOutput())
        task = asyncio.create_task(app.run(statuses(), logs()))
        await wait_for(status_updated)
        assert state.running == 2
        await key(app, pipe, "q")
        assert await asyncio.wait_for(task, 3) == JobWatchRenderResult.INTERRUPTED
        assert logs_closed.is_set()
        assert status_closed.is_set()


async def test_terminal_status_remains_inspectable_until_quit() -> None:
    async def statuses() -> AsyncIterator[JobWatchEvent]:
        yield JobStatusEvent(
            kind="status", job_name="pipeline", status="completed", status_details={}, terminal=True, successful=True
        )

    async def logs() -> AsyncIterator[JobWatchEvent]:
        yield log("execution-1", "final output")

    with create_pipe_input() as pipe:
        app = DagWatchApp(model(), input=pipe, output=DummyOutput())
        task = asyncio.create_task(app.run(statuses(), logs()))
        async with asyncio.timeout(3):
            while not app.status_finished or not app.logs_finished:
                await asyncio.sleep(0.01)
        assert not task.done()
        assert "WATCH FINISHED" in "\n".join(app.screen(Size(rows=28, columns=150)))
        await key(app, pipe, "q")
        assert await asyncio.wait_for(task, 3) == JobWatchRenderResult.SUCCEEDED


async def test_typed_client_status_updates_with_real_http_boundary() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "name": "pipeline",
                "status": "completed",
                "status_details": {},
                "error_details": None,
                "steps": [],
                "created_at": NOW.isoformat(),
                "updated_at": NOW.isoformat(),
                "attempt_id": "attempt-1",
                "workflow": event(graph()).workflow.model_dump(mode="json"),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = AsyncJobsClient(base_url="http://test", http_client=transport)
        app = DagWatchApp(DagWatchModel(), output=DummyOutput())
        await app._consume_status(client.watch_job("job-1", workspace="ws", attempt_id="attempt-1", include_logs=False))
        assert app.model.running == 1
        assert app.result == JobWatchRenderResult.SUCCEEDED
    assert requests[0].url.params["attempt_id"] == "attempt-1"
    assert len(requests) == 1


async def test_timeout_and_log_failure_remain_visible() -> None:
    async def timeout() -> AsyncIterator[JobWatchEvent]:
        raise TimeoutError("watch deadline reached")
        yield log("execution-1")

    async def denied_logs() -> AsyncIterator[JobWatchEvent]:
        raise PermissionError("logs denied")
        yield log("execution-1")

    app = DagWatchApp(model(), output=DummyOutput())
    await app._consume_status(timeout())
    assert app.result == JobWatchRenderResult.FAILED
    assert "watch deadline reached" in app.model.notice
    await app._consume_logs(denied_logs())
    assert "logs denied" in app.model.notice
    assert app.model.running == 1
    assert app.logs_finished


def test_server_reuse_reason_and_pending_work_are_distinct() -> None:
    from nemo_helix_ext.cli.core.job_watch_model import phase

    item = node("reused", state="idle")
    item.freshness = "clean"
    item.last_outcome = "completed"
    assert phase(item) == "completed"
    item.reason = "unchanged_inputs"
    assert phase(item) == "reused"
    item.has_pending_run = True
    assert phase(item) == "idle"
    item.reason = "dependencies"
    item.scheduling_state = "waiting_dependencies"
    item.has_pending_run = True
    assert phase(item) == "waiting_dependencies"


def test_log_text_is_literal_and_long_lines_can_be_panned() -> None:
    state = model()
    state.add_log(log("execution-1", "<b>literal</b>\x1b[2J" + "long" * 100))
    app = DagWatchApp(state, output=DummyOutput())
    app.mode = "logs"
    screen = "\n".join(app.screen(Size(rows=24, columns=120)))
    assert "<b>literal</b>" in screen
    assert "\x1b" not in screen
    app.log_horizontal = 30
    assert "long" in "\n".join(app.screen(Size(rows=24, columns=120)))


async def test_drawn_overview_keyboard_pan_select_and_focus() -> None:
    state = model()
    with create_pipe_input() as pipe:
        app = DagWatchApp(state, input=pipe, output=DummyOutput())
        ready = asyncio.Event()
        task = asyncio.create_task(app.application.run_async(pre_run=ready.set))
        await wait_for(ready)
        try:
            await key(app, pipe, "o")
            assert app.mode == "overview"
            assert app.graph_selected == "train"
            await key(app, pipe, "j")
            assert app.graph_y > 0
            assert not app.graph_follow
            await key(app, pipe, "\x1b[H")
            assert app.graph_follow
            await key(app, pipe, "\x1b[C")
            assert app.graph_selected.startswith("output-")
            selected = app.graph_selected
            await key(app, pipe, "\r")
            assert state.focus == selected
            assert app.mode == "status"
            await key(app, pipe, "\x7f")
            assert state.focus == "train"
            await key(app, pipe, "/output-0")
            await key(app, pipe, "\x1b[B\r")
            assert state.focus == "output-001"
            await key(app, pipe, "q")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                app.application.exit()
                await task


def test_right_pane_selection_and_current_phase_have_state_styles() -> None:
    state = model()
    state.choose_pane(1)
    state.nodes["output-000"].scheduling_state = "idle"
    state.nodes["output-000"].last_outcome = "error"
    app = DagWatchApp(state, output=DummyOutput())
    lines = app.screen(Size(rows=30, columns=150))
    fragments = [fragment for row in app.body_fragments.values() for fragment in row]
    assert any(
        "class:failed" in fragment[0] and "reverse" in fragment[0] and "output-000" in fragment[1]
        for fragment in fragments
    )
    assert any("class:failed" in fragment[0] and "Failed" in fragment[1] for fragment in fragments)
    assert "No changes observed this session" in "\n".join(lines)


def test_overview_layout_does_not_move_on_status_updates() -> None:
    state = model()
    app = DagWatchApp(state, output=DummyOutput())
    app.mode = "overview"
    app.screen(Size(rows=30, columns=150))
    layout = app.graph_layout
    assert layout is not None
    boxes = dict(layout.boxes)
    update = event(graph())
    update.workflow.nodes[0].scheduling_state = "running"
    state.apply_workflow(update, NOW + timedelta(seconds=1))
    app.screen(Size(rows=30, columns=150))
    assert app.graph_layout is layout
    assert app.graph_layout.boxes == boxes


def test_neighbor_card_shows_current_phase_and_concise_transition() -> None:
    state = model()
    update = event(graph())
    changed = update.workflow.nodes[99]
    changed.scheduling_state = "running"
    changed.active_execution_id = "long-execution-identifier-for-input-099"
    state.apply_workflow(update, NOW + timedelta(seconds=1))
    app = DagWatchApp(state, output=DummyOutput())
    screen = "\n".join(app.screen(Size(rows=30, columns=150)))
    assert "input-099" in screen
    assert "Waiting on dependencies → Running" in screen
    assert "Observed" in screen


def test_typed_conditions_and_trigger_edges_render_without_fileset_nodes() -> None:
    prepare, refresh, train = node("prepare"), node("refresh"), node("train", state="admitted", execution="train-1")
    train.depends_on = [
        "prepare",
        WorkflowStepSuccess(step_success="prepare"),
        WorkflowFilesetCommit(fileset_commit="config"),
    ]
    train.trigger = [WorkflowStepSuccess(step_success="refresh"), WorkflowFilesetCommit(fileset_commit="input")]
    state = DagWatchModel()
    state.apply_workflow(event([prepare, refresh, train]), NOW)
    state.focus_node("train")
    assert state.orders[Pane.INPUTS] == ["prepare", "refresh"]
    assert state.outputs["prepare"] == ["train"]
    assert state.outputs["refresh"] == ["train"]
    assert state.running == 0
    app = DagWatchApp(state, output=DummyOutput())
    screen = "\n".join(app.screen(Size(rows=40, columns=210)))
    assert "Admitted" in screen
    assert "Ready when (all):" in screen
    assert "Triggered by (any):" in screen
    assert "Fileset config committed" in screen
    assert "Fileset input committed" in screen
    app.mode = "overview"
    app.screen(Size(rows=40, columns=140))
    assert app.graph_layout is not None
    assert app.graph_layout.dependencies["train"] == ("prepare", "refresh")
    assert set(app.graph_layout.boxes) == {"prepare", "refresh", "train"}


async def test_typed_conditions_survive_wire_status_watch() -> None:
    train = node("train", state="admitted", execution="train-1")
    train.depends_on = [WorkflowStepSuccess(step_success="prepare"), WorkflowFilesetCommit(fileset_commit="config")]
    train.trigger = [WorkflowStepSuccess(step_success="refresh")]
    workflow = event([node("prepare"), node("refresh"), train]).workflow

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "name": "pipeline",
                "status": "completed",
                "status_details": {},
                "error_details": None,
                "steps": [],
                "created_at": NOW.isoformat(),
                "updated_at": NOW.isoformat(),
                "attempt_id": "attempt-1",
                "workflow": workflow.model_dump(mode="json"),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = AsyncJobsClient(base_url="http://test", http_client=transport)
        app = DagWatchApp(DagWatchModel(), output=DummyOutput())
        await app._consume_status(client.watch_job("job-1", workspace="ws", attempt_id="attempt-1", include_logs=False))
    assert app.result == JobWatchRenderResult.SUCCEEDED
    assert app.model.outputs["refresh"] == ["train"]
    assert app.model.running == 0
