# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand-written data-designer commands honor the active CLI context's workspace.

Every ``--workspace`` option here used to default to the literal string
``"default"``, which made an omitted flag indistinguishable from an explicit
``--workspace default``. The active context could therefore never win, and a
user whose context pointed at another workspace silently acted on ``default``.

These tests pin the resolution order for a representative command in each
module: explicit flag > active CLI context > ``"default"`` when no CLI state
is installed (direct/unit invocation).
"""

from __future__ import annotations

import contextlib
import json
import shlex
from collections.abc import Iterator
from typing import Any

import pytest
import typer
from nemo_data_designer_plugin.cli import inputs
from nemo_data_designer_plugin.cli import renderers as renderers_mod
from nemo_data_designer_plugin.cli.retrieval import retrieval_app
from nemo_helix_plugin.cli_renderer import RendererContext
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()


class _State:
    """Minimal stand-in for the CLI state object the top-level ``nemo`` installs."""

    def __init__(self, workspace: str | None) -> None:
        self._workspace = workspace

    def get_workspace(self) -> str | None:
        return self._workspace


@pytest.fixture(autouse=True)
def _no_env_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$NHX_WORKSPACE`` sits between the context and ``"default"``; keep it out."""
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)


# --------------------------------------------------------------------------
# retrieval.py — the printed submit command must name the resolved workspace
# --------------------------------------------------------------------------

_GENERATE_ARGS = [
    "generate",
    "--corpus",
    "hf://corpus",
    "--provider",
    "default/nvidia",
    "--chat-model",
    "chat",
    "--embed-model",
    "embed",
]


def _submitted_workspace(output: str) -> str:
    """Pull the ``--workspace`` value out of the printed submit command."""
    tokens = output.split()
    return tokens[tokens.index("--workspace") + 1]


def test_retrieval_generate_uses_context_workspace_when_flag_omitted() -> None:
    result = runner.invoke(retrieval_app, _GENERATE_ARGS, obj=_State("research"))

    assert result.exit_code == 0, result.output
    assert _submitted_workspace(result.output) == "research"


def test_retrieval_generate_explicit_flag_beats_context() -> None:
    result = runner.invoke(
        retrieval_app,
        [*_GENERATE_ARGS, "--workspace", "team-a"],
        obj=_State("research"),
    )

    assert result.exit_code == 0, result.output
    assert _submitted_workspace(result.output) == "team-a"


def test_retrieval_generate_falls_back_to_default_without_cli_state() -> None:
    result = runner.invoke(retrieval_app, _GENERATE_ARGS)

    assert result.exit_code == 0, result.output
    assert _submitted_workspace(result.output) == "default"


def test_retrieval_preview_uses_context_workspace() -> None:
    result = runner.invoke(
        retrieval_app,
        [
            "preview",
            "--corpus",
            "hf://corpus",
            "--provider",
            "default/nvidia",
            "--chat-model",
            "chat",
            "--embed-model",
            "embed",
        ],
        obj=_State("research"),
    )

    assert result.exit_code == 0, result.output
    assert _submitted_workspace(result.output) == "research"


def test_retrieval_prepare_uses_context_workspace() -> None:
    result = runner.invoke(
        retrieval_app,
        ["prepare", "--sdg-input", "fs://stage0"],
        obj=_State("research"),
    )

    assert result.exit_code == 0, result.output
    assert _submitted_workspace(result.output) == "research"


# --------------------------------------------------------------------------
# inputs.py — the create/preview wrappers delegate to the generated verbs
# --------------------------------------------------------------------------


def _create_app(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    """Build a ``create`` group whose generated callback records its kwargs."""
    group = typer.Typer(no_args_is_help=False)

    def original(typer_ctx: typer.Context, **kwargs: object) -> None:
        del typer_ctx
        captured.update(kwargs)

    group.callback(invoke_without_command=True)(original)

    @contextlib.contextmanager
    def fake_spec_from_builder(config_source: str, num_records: int) -> Iterator[str]:
        del config_source, num_records
        yield '{"spec": true}'

    monkeypatch.setattr(inputs, "_spec_from_builder", fake_spec_from_builder)
    inputs.apply_create_cli_overrides(group)

    app = typer.Typer()
    app.add_typer(group, name="create")
    return app


def test_create_wrapper_passes_context_workspace_to_generated_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    app = _create_app(captured, monkeypatch)

    result = runner.invoke(app, ["create", "config.yaml"], obj=_State("research"))

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == "research"


def test_create_wrapper_explicit_flag_beats_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    app = _create_app(captured, monkeypatch)

    result = runner.invoke(
        app,
        ["create", "config.yaml", "--workspace", "team-a"],
        obj=_State("research"),
    )

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == "team-a"


def test_create_wrapper_falls_back_to_default_without_cli_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    app = _create_app(captured, monkeypatch)

    result = runner.invoke(app, ["create", "config.yaml"])

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == "default"


def _preview_app(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    group = typer.Typer()

    # Unlike ``create``, the preview override leaves the original command
    # registered, so Typer still introspects this signature -- it has to be
    # spelled out rather than swallowed by ``**kwargs``.
    @group.command("preview")
    def original(
        typer_ctx: typer.Context,
        spec: str = typer.Option("", "--spec"),
        spec_file: str | None = typer.Option(None, "--spec-file"),
        cluster: str | None = typer.Option(None, "--cluster"),
        base_url: str | None = typer.Option(None, "--base-url"),
        workspace: str | None = typer.Option(None, "--workspace"),
        request_id: str | None = typer.Option(None, "--request-id"),
        non_interactive: bool = typer.Option(False, "--non-interactive"),
        save_results: bool = typer.Option(False, "--save-results"),
        artifact_path: str | None = typer.Option(None, "--artifact-path"),
    ) -> None:
        del typer_ctx, spec, spec_file, cluster, base_url
        del request_id, non_interactive, save_results, artifact_path
        captured["workspace"] = workspace

    @contextlib.contextmanager
    def fake_spec_from_builder(config_source: str, num_records: int) -> Iterator[str]:
        del config_source, num_records
        yield '{"spec": true}'

    monkeypatch.setattr(inputs, "_spec_from_builder", fake_spec_from_builder)
    inputs.apply_preview_cli_overrides(group)
    return group


def test_preview_wrapper_passes_context_workspace_to_generated_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    app = _preview_app(captured, monkeypatch)

    result = runner.invoke(app, ["preview", "config.yaml"], obj=_State("research"))

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == "research"


def test_preview_wrapper_falls_back_to_default_without_cli_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    app = _preview_app(captured, monkeypatch)

    result = runner.invoke(app, ["preview", "config.yaml"])

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == "default"


# --------------------------------------------------------------------------
# renderers.py — the copy-paste hint must still name a real workspace
# --------------------------------------------------------------------------


def test_create_wrapper_never_hands_the_renderer_a_none_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper resolves before delegating, so ``cli_kwargs["workspace"]``
    the renderer reads is always a concrete string -- never the ``None`` that
    the new Typer default introduces."""
    captured: dict[str, object] = {}
    app = _create_app(captured, monkeypatch)

    result = runner.invoke(app, ["create", "config.yaml"], obj=_State("research"))

    assert result.exit_code == 0, result.output
    assert isinstance(captured["workspace"], str)
    assert captured["workspace"]


@pytest.mark.parametrize(
    ("frame_workspace", "cli_workspace", "expected"),
    [
        ("research", "research", " --workspace research"),
        (None, "research", " --workspace research"),
        ("research", None, " --workspace research"),
        (None, None, ""),
    ],
)
def test_workspace_hint_survives_a_none_cli_kwarg(
    frame_workspace: str | None,
    cli_workspace: str | None,
    expected: str,
) -> None:
    """``cli_kwargs["workspace"]`` can now be ``None`` for a plugin command that
    skips resolution; the hint must degrade to omission, not crash or print
    ``--workspace None``."""
    frame: dict[str, Any] = {"name": "job-1"}
    if frame_workspace is not None:
        frame["workspace"] = frame_workspace
    ctx = RendererContext(
        console=Console(),
        cli_kwargs={"workspace": cli_workspace},
        verb="submit",
        is_local=False,
    )

    assert renderers_mod._workspace_flag(frame, ctx) == expected


def test_retrieval_submit_command_is_valid_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` workspace would blow up ``shlex.join``; the resolved value
    keeps the printed command copy-pasteable."""
    result = runner.invoke(retrieval_app, _GENERATE_ARGS, obj=_State("research"))

    assert result.exit_code == 0, result.output
    tokens = shlex.split(result.output.split("Submit with:", 1)[1])
    assert tokens[tokens.index("--workspace") + 1] == "research"
    assert json.loads(tokens[tokens.index("--spec") + 1])["corpus"] == "hf://corpus"
