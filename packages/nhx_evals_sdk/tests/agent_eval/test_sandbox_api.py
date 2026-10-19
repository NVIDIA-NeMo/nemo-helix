# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AsyncSandbox facade over a fake in-memory provider."""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.api import AsyncSandbox
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


class _FakeProvider:
    """Records lifecycle operations; enough of SandboxProvider for the facade tests."""

    name = "fake"

    def __init__(self, *, fail_uploads: bool = False) -> None:
        self.created: list[SandboxSpec] = []
        self.uploaded: list[tuple[str, str]] = []
        self.execs: list[tuple[str, str | None, bytes | None]] = []
        self.closed = 0
        self.aclosed = 0
        self._fail_uploads = fail_uploads

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw=None)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        self.execs.append((command, cwd, stdin))
        return SandboxExecResult(stdout="ok", stderr="", return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        if self._fail_uploads:
            raise RuntimeError("upload boom")
        self.uploaded.append((target_path, source_path.read_text(encoding="utf-8")))

    async def upload_dir(self, handle: SandboxHandle, source_dir: Path, target_dir: str) -> None:
        self.uploaded.append((target_dir, str(source_dir)))

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target_path.write_text("downloaded", encoding="utf-8")

    async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.aclosed += 1


async def test_start_seeds_files_then_stop_closes() -> None:
    provider = _FakeProvider()
    spec = SandboxSpec(image="img", workdir="/workspace", files={"/workspace/a.py": "x = 1"})
    async with AsyncSandbox(provider, spec) as sandbox:
        await sandbox.start()
        assert provider.created and provider.uploaded == [("/workspace/a.py", "x = 1")]
    # Stopping the sandbox closes only its own handle; the shared provider's lifetime (aclose) is the
    # owner's, so the facade must not aclose it.
    assert provider.closed == 1 and provider.aclosed == 0


async def test_exec_defaults_cwd_to_spec_workdir() -> None:
    provider = _FakeProvider()
    sandbox = AsyncSandbox(provider, SandboxSpec(image="img", workdir="/workspace"))
    await sandbox.start()
    await sandbox.exec("pytest")  # no explicit cwd
    await sandbox.exec("ls", cwd="/other")  # explicit cwd wins
    assert provider.execs[0][1] == "/workspace"
    assert provider.execs[1][1] == "/other"


async def test_exec_before_start_raises() -> None:
    sandbox = AsyncSandbox(_FakeProvider(), SandboxSpec(image="img"))
    with pytest.raises(RuntimeError, match="not been started"):
        await sandbox.exec("true")


async def test_seed_failure_tears_down_sandbox() -> None:
    provider = _FakeProvider(fail_uploads=True)
    spec = SandboxSpec(image="img", files={"/x": "y"})
    sandbox = AsyncSandbox(provider, spec)
    with pytest.raises(RuntimeError, match="upload boom"):
        await sandbox.start()
    # A half-created sandbox must be closed, not leaked — but the shared provider is not aclosed here.
    assert provider.closed == 1 and provider.aclosed == 0


async def test_double_start_raises() -> None:
    sandbox = AsyncSandbox(_FakeProvider(), SandboxSpec(image="img"))
    await sandbox.start()
    with pytest.raises(RuntimeError, match="already started"):
        await sandbox.start()


async def test_stop_is_idempotent() -> None:
    provider = _FakeProvider()
    sandbox = AsyncSandbox(provider, SandboxSpec(image="img"))
    await sandbox.start()
    await sandbox.stop()
    await sandbox.stop()
    assert provider.closed == 1  # second stop is a no-op
