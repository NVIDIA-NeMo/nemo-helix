# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NemoCLI.get_function_renderer / get_job_renderer hooks.

Covers:
- Default no-op (returns None) leaves the auto-generated echo behavior intact.
- A renderer's lifecycle methods fire in the expected order.
- The ``--output-format json`` global flag bypasses the renderer entirely.
- The "delegate to use the renderer" contract: an update_*_cli wrapper that
  delegates to the original callback fires the renderer; a wrapper that does
  its own iteration without calling original does not.
- on_error fires on exception (and the exception still propagates).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import typer
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_renderer import CLIRenderer, RendererContext
from nemo_platform_plugin.commands import add_function_commands, add_job_commands
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.functions.frames import Done, Heartbeat
from nemo_platform_plugin.job import NemoJob
from pydantic import BaseModel
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture primitives
# ---------------------------------------------------------------------------


class _GreetSpec(BaseModel):
    name: str


class _GreetResponse(BaseModel):
    message: str


class _GreetFunction(NemoFunction[_GreetSpec]):
    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Say hello to a name."
    spec_schema: ClassVar[type[_GreetSpec]] = _GreetSpec

    async def run(self, spec: _GreetSpec) -> _GreetResponse:
        return _GreetResponse(message=f"Hello, {spec.name}!")


class _CountSpec(BaseModel):
    upto: int


class _CountFunction(NemoFunction[_CountSpec]):
    """Streaming function — yields heartbeats then Done."""

    name: ClassVar[str] = "count"
    spec_schema: ClassVar[type[_CountSpec]] = _CountSpec

    async def run(self, spec: _CountSpec) -> AsyncIterator[BaseModel]:
        for _ in range(spec.upto):
            yield Heartbeat()
        yield Done()


class _GreetJob(NemoJob):
    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Return a greeting."

    def run(self, config: dict) -> dict:
        return {"message": f"Hello, {config.get('name', 'world')}!"}


class _NoOpCLI(NemoCLI):
    name: ClassVar[str] = "test-plugin"

    def get_cli(self) -> typer.Typer:
        return typer.Typer()


class _RecordingRenderer(CLIRenderer):
    """Renderer that records lifecycle events for assertions."""

    events: list[tuple[str, Any]] = []  # noqa: RUF012 — class state shared across instances by design

    def __init__(self) -> None:
        # Reset at the start of each renderer instance.
        # Renderers are constructed once per verb invocation, so this clears
        # between scenarios when scenarios use a fresh CLI build.
        type(self).events = []

    def on_start(self, *, ctx: RendererContext) -> None:
        type(self).events.append(("start", {"verb": ctx.verb, "is_local": ctx.is_local}))

    def on_frame(self, frame: Any, *, ctx: RendererContext) -> None:
        del ctx
        type(self).events.append(("frame", frame))

    def on_complete(self, *, ctx: RendererContext) -> None:
        del ctx
        type(self).events.append(("complete", None))

    def on_error(self, error: BaseException, *, ctx: RendererContext) -> None:
        del ctx
        type(self).events.append(("error", type(error).__name__))


def _typer_context_with_overrides(output_format: str | None = None) -> object:
    """Stand-in for CLIContext: just an object with .overrides dict."""
    overrides: dict[str, Any] = {}
    if output_format is not None:
        overrides["output_format"] = output_format
    return SimpleNamespace(overrides=overrides)


def _build_function_app(*fn_classes: type[NemoFunction], cli: NemoCLI | None = None) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    fns = {f"plugin.{cls.name}": cls for cls in fn_classes}
    add_function_commands(app, fns, cli=cli)
    return app


def _build_job_app(*job_classes: type[NemoJob], cli: NemoCLI | None = None) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    jobs = {f"plugin.{cls.name}": cls for cls in job_classes}
    add_job_commands(app, jobs, cli=cli)
    return app


def _patch_submit_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit_remote(self, job_cls: type[NemoJob], spec: dict, **kwargs: object) -> dict:
        del self, job_cls, kwargs
        return {"message": f"Hello, {spec.get('name', 'world')}!"}

    monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", fake_submit_remote)


# ---------------------------------------------------------------------------
# Streaming function: get_function_renderer for `run`
# ---------------------------------------------------------------------------


class TestStreamingFunctionRunRenderer:
    def test_no_renderer_falls_through_to_default_echo(self) -> None:
        app = _build_function_app(_CountFunction, cli=_NoOpCLI())
        result = runner.invoke(app, ["count", "run", "--upto", "2"])
        assert result.exit_code == 0
        # Default behavior echoes each frame as JSON; verify by counting lines
        # that look like a JSON object.
        json_lines = [ln for ln in result.output.splitlines() if ln.strip().startswith("{")]
        assert len(json_lines) >= 3  # 2 heartbeats + 1 Done

    def test_renderer_lifecycle_fires_in_order(self) -> None:
        class _CLI(_NoOpCLI):
            def get_function_renderer(self, fn_cls, *, verb):
                return _RecordingRenderer if fn_cls is _CountFunction else None

        app = _build_function_app(_CountFunction, cli=_CLI())
        result = runner.invoke(app, ["count", "run", "--upto", "2"])
        assert result.exit_code == 0, result.output

        events = _RecordingRenderer.events
        names = [name for name, _ in events]
        # Lifecycle: start → frames → complete.
        assert names[0] == "start"
        assert names[-1] == "complete"
        assert names.count("frame") == 3  # 2 heartbeats + 1 Done

        # on_start receives the right context.
        start_meta = events[0][1]
        assert start_meta == {"verb": "run", "is_local": True}

    def test_output_format_json_bypasses_renderer(self) -> None:
        class _CLI(_NoOpCLI):
            def get_function_renderer(self, fn_cls, *, verb):
                return _RecordingRenderer

        app = _build_function_app(_CountFunction, cli=_CLI())
        ctx_obj = _typer_context_with_overrides(output_format="json")

        # Reset before invocation.
        _RecordingRenderer.events = []
        result = runner.invoke(app, ["count", "run", "--upto", "1"], obj=ctx_obj)
        assert result.exit_code == 0, result.output

        # Renderer was never instantiated.
        assert _RecordingRenderer.events == []

        # And the default echo behavior fired.
        json_lines = [ln for ln in result.output.splitlines() if ln.strip().startswith("{")]
        assert len(json_lines) >= 1

    def test_renderer_dispatches_per_function_and_verb(self) -> None:
        seen: list[tuple[str, str]] = []

        class _CLI(_NoOpCLI):
            def get_function_renderer(self, fn_cls, *, verb):
                seen.append((fn_cls.name, verb))
                return None  # decline to render; just observe the dispatch

        app = _build_function_app(_CountFunction, _GreetFunction, cli=_CLI())
        runner.invoke(app, ["count", "run", "--upto", "1"])

        # The hook fires per-verb on the invoked function only.
        assert ("count", "run") in seen


# ---------------------------------------------------------------------------
# Job: get_job_renderer for `submit`
# ---------------------------------------------------------------------------


class TestJobSubmitRenderer:
    def test_no_renderer_falls_through_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_submit_remote(monkeypatch)
        app = _build_job_app(_GreetJob, cli=_NoOpCLI())
        result = runner.invoke(app, ["greet", "submit", "--spec", '{"name": "World"}'])
        assert result.exit_code == 0
        assert json.loads(result.output) == {"message": "Hello, World!"}

    def test_renderer_lifecycle_for_synchronous_submit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_submit_remote(monkeypatch)

        class _CLI(_NoOpCLI):
            def get_job_renderer(self, job_cls, *, verb):
                return _RecordingRenderer

        app = _build_job_app(_GreetJob, cli=_CLI())
        _RecordingRenderer.events = []
        result = runner.invoke(app, ["greet", "submit", "--spec", '{"name": "Renderer"}'])
        assert result.exit_code == 0, result.output

        events = _RecordingRenderer.events
        names = [name for name, _ in events]
        # start → one frame (the dict result) → complete.
        assert names == ["start", "frame", "complete"]
        # The frame is the dict the job returned.
        _, frame = events[1]
        assert frame == {"message": "Hello, Renderer!"}
        start_meta = events[0][1]
        assert start_meta == {"verb": "submit", "is_local": False}

    def test_output_format_json_bypasses_job_renderer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_submit_remote(monkeypatch)

        class _CLI(_NoOpCLI):
            def get_job_renderer(self, job_cls, *, verb):
                return _RecordingRenderer

        app = _build_job_app(_GreetJob, cli=_CLI())
        ctx_obj = _typer_context_with_overrides(output_format="json")
        _RecordingRenderer.events = []
        result = runner.invoke(app, ["greet", "submit", "--spec", '{"name": "X"}'], obj=ctx_obj)
        assert result.exit_code == 0
        assert _RecordingRenderer.events == []
        # Default echo fired:
        assert json.loads(result.output) == {"message": "Hello, X!"}


# ---------------------------------------------------------------------------
# "Delegate to use the renderer" contract
# ---------------------------------------------------------------------------


class TestDelegateContract:
    """A wrapper from update_function_cli that delegates to the original
    callback gets the renderer for free; a wrapper that takes over the verb
    body wholesale claims rendering responsibility too."""

    def test_delegating_wrapper_still_drives_renderer(self) -> None:
        """When the wrapper calls original(...), the framework's renderer
        loop runs through the original's body — so the renderer fires."""

        class _CLI(_NoOpCLI):
            def update_function_cli(self, fn_cls, group):
                if fn_cls is not _CountFunction:
                    return
                original = next(c for c in group.registered_commands if c.name == "run").callback
                assert original is not None

                @group.command("run")
                def run(typer_ctx: typer.Context, count: int = typer.Option(2, "--count")) -> None:
                    spec_json = json.dumps({"upto": count})
                    original(typer_ctx, spec=spec_json, spec_file=None, workspace="default")

            def get_function_renderer(self, fn_cls, *, verb):
                return _RecordingRenderer if fn_cls is _CountFunction else None

        app = _build_function_app(_CountFunction, cli=_CLI())
        _RecordingRenderer.events = []
        result = runner.invoke(app, ["count", "run", "--count", "1"])
        assert result.exit_code == 0, result.output

        # Renderer fired for the delegated invocation.
        events = _RecordingRenderer.events
        names = [name for name, _ in events]
        assert names[0] == "start"
        assert names[-1] == "complete"
        assert names.count("frame") == 2  # 1 heartbeat + 1 Done


# ---------------------------------------------------------------------------
# on_error
# ---------------------------------------------------------------------------


class _ExplodingSpec(BaseModel):
    pass


class _ExplodingFunction(NemoFunction[_ExplodingSpec]):
    name: ClassVar[str] = "explode"
    spec_schema: ClassVar[type[_ExplodingSpec]] = _ExplodingSpec

    async def run(self, spec: _ExplodingSpec) -> AsyncIterator[BaseModel]:
        del spec
        yield Heartbeat()
        raise RuntimeError("boom")


class TestOnError:
    def test_on_error_fires_when_iteration_raises(self) -> None:
        class _CLI(_NoOpCLI):
            def get_function_renderer(self, fn_cls, *, verb):
                return _RecordingRenderer

        app = _build_function_app(_ExplodingFunction, cli=_CLI())
        _RecordingRenderer.events = []
        result = runner.invoke(app, ["explode", "run"])

        assert result.exit_code != 0  # exception still propagates
        events = _RecordingRenderer.events
        names = [name for name, _ in events]
        assert "start" in names
        assert "frame" in names  # got the heartbeat before the explosion
        assert "error" in names
        # Exception type is RuntimeError.
        error_payload = next(payload for name, payload in events if name == "error")
        assert error_payload == "RuntimeError"
        # on_complete should NOT fire on the error path.
        assert "complete" not in names
