# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for the Docker Compose sandbox provider."""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxCreateError
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_cli as compose_cli

from packages.nhx_evals_sdk.tests.agent_eval._compose_testkit import (
    _compose_suffix,
    _create,
    _files,
    _provider,
    _Runner,
)


async def test_default_is_image_first_and_build_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    files = _files(tmp_path, count=2)
    project_dir = tmp_path / "workspace"
    project_dir.mkdir()
    provider = _provider(
        tmp_path,
        files=files,
        project_directory=project_dir,
        profiles=("gpu", "tools"),
    )

    handle = await _create(monkeypatch, provider, runner)
    up_argv, _environment, cwd = next(call for call in runner.calls if _compose_suffix(call[0])[:1] == ("up",))
    assert cwd == project_dir
    assert up_argv.count("--file") == 2
    assert [str(path) for path in files] == [up_argv[index + 1] for index, arg in enumerate(up_argv) if arg == "--file"]
    assert ["gpu", "tools"] == [up_argv[index + 1] for index, arg in enumerate(up_argv) if arg == "--profile"]
    assert "--project-directory" in up_argv
    assert "--no-build" in up_argv
    assert "--build" not in up_argv
    assert _compose_suffix(up_argv)[-2:] == ("--pull", "missing")
    assert handle.sandbox_id.startswith("nemo-eval-")

    await provider.close(handle)
    await provider.aclose()


@pytest.mark.parametrize(
    ("build", "pull_policy", "expected"),
    [
        (True, "always", "--build"),
        (True, "never", "--build"),
        (False, "always", "--no-build"),
        (False, "never", "--no-build"),
    ],
)
async def test_build_and_pull_policy_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build: bool,
    pull_policy: str,
    expected: str,
) -> None:
    runner = _Runner()
    provider = _provider(
        tmp_path,
        project_name=f"matrix-{str(build).lower()}-{pull_policy}",
        build=build,
        pull_policy=pull_policy,
    )
    handle = await _create(monkeypatch, provider, runner)
    up = next(_compose_suffix(argv) for argv, _, _ in runner.calls if _compose_suffix(argv)[:1] == ("up",))
    assert expected in up
    assert up[-2:] == ("--pull", pull_policy)
    await provider.close(handle)


async def test_missing_image_does_not_fall_back_to_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.up_failure = True
    provider = _provider(tmp_path)

    with pytest.raises(SandboxCreateError, match="missing image"):
        await _create(monkeypatch, provider, runner)

    suffixes = [_compose_suffix(argv) for argv, _, _ in runner.calls if argv[:2] == ("docker", "compose")]
    up = next(args for args in suffixes if args[:1] == ("up",))
    assert "--no-build" in up
    assert not any(args[:1] == ("build",) for args in suffixes)
    assert any(args[:1] == ("down",) for args in suffixes)


def test_redaction_covers_environment_and_inline_secrets() -> None:
    redacted = compose_cli._redact(
        "TOKEN=secret-value\nAuthorization: Bearer bearer-value\npassword=hunter2",
        {**os.environ, "TOKEN": "secret-value"},
    )
    assert "secret-value" not in redacted
    assert "bearer-value" not in redacted
    assert "hunter2" not in redacted
    assert redacted.count("<redacted>") == 3


async def test_streaming_command_redacts_output_and_retains_raw_result(
    tmp_path: Path,
) -> None:
    environment = {**os.environ, "TEST_TOKEN": "stream-secret"}
    progress = io.StringIO()
    script = "\n".join(
        [
            "import sys",
            'print("token=stream-secret")',
            'print("password=hunter2", file=sys.stderr)',
        ]
    )

    result = await compose_cli._run_command(
        (sys.executable, "-c", script),
        cwd=tmp_path,
        environment=environment,
        timeout=5,
        stdin=None,
        stream_output=progress,
    )

    assert result.ok
    assert result.stdout == "token=stream-secret\n"
    assert result.stderr == "password=hunter2\n"
    assert "stream-secret" not in progress.getvalue()
    assert "hunter2" not in progress.getvalue()
    assert progress.getvalue().count("<redacted>") == 2


async def test_streaming_command_supports_stdout_lines_larger_than_64_kib(
    tmp_path: Path,
) -> None:
    secret = "large-stream-secret"
    line = "x" * (64 * 1024) + f" token={secret}"
    progress = io.StringIO()
    script = f'import sys; sys.stdout.write("x" * {64 * 1024} + " token={secret}\\n")'

    result = await compose_cli._run_command(
        (sys.executable, "-c", script),
        cwd=tmp_path,
        environment={**os.environ, "TEST_TOKEN": secret},
        timeout=5,
        stdin=None,
        stream_output=progress,
    )

    assert result.ok
    assert result.stdout == f"{line}\n"
    assert secret not in progress.getvalue()
    assert progress.getvalue() == f"{'x' * (64 * 1024)} token=<redacted>\n"


async def test_streaming_timeout_retains_partial_output(tmp_path: Path) -> None:
    progress = io.StringIO()
    script = 'import time; print("started", flush=True); time.sleep(30)'

    # Long enough for interpreter start-up on a loaded machine, far shorter than the child's sleep.
    result = await compose_cli._run_command(
        (sys.executable, "-c", script),
        cwd=tmp_path,
        environment=os.environ,
        timeout=3,
        stdin=None,
        stream_output=progress,
    )

    assert result.timed_out
    assert result.stdout == "started\n"
    assert "timed out after" in result.stderr.lower()
    assert progress.getvalue() == "started\n"


async def test_non_streaming_timeout_retains_partial_output(tmp_path: Path) -> None:
    script = (
        'import sys, time; print("started", flush=True); print("warning", file=sys.stderr, flush=True); time.sleep(30)'
    )

    result = await compose_cli._run_command(
        (sys.executable, "-c", script),
        cwd=tmp_path,
        environment=os.environ,
        timeout=1,
        stdin=None,
    )

    assert result.timed_out
    assert result.stdout == "started\n"
    assert result.stderr.startswith("warning\nCommand timed out after 1.0s")


async def test_streaming_cancellation_terminates_process(tmp_path: Path) -> None:
    progress = io.StringIO()
    script = 'import time; print("started", flush=True); time.sleep(30)'
    task = asyncio.create_task(
        compose_cli._run_command(
            (sys.executable, "-c", script),
            cwd=tmp_path,
            environment=os.environ,
            timeout=60,
            stdin=None,
            stream_output=progress,
        )
    )
    async with asyncio.timeout(5):
        while progress.getvalue() != "started\n":
            if task.done():
                await task
            await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert progress.getvalue() == "started\n"
