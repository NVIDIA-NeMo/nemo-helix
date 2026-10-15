# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral sandbox facade.

:class:`AsyncSandbox` is the object agent-eval runtimes use: it drives one sandbox's
``create → seed → exec → transfer → close`` lifecycle over a
:class:`~nhx_evals_sdk.agent_eval.runtimes.sandbox.base.SandboxProvider`, and writes
``spec.files`` in on ``start()``. It does **not** own the provider's lifetime: the provider is
typically shared across a batch of concurrent sandboxes, so ``stop()`` tears down only this
sandbox (``provider.close(handle)``). Disposing the provider's process-wide resources
(``provider.aclose()``) is the batch owner's job — the runtime that created the provider.
"""

from __future__ import annotations

from pathlib import Path

from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
)


class AsyncSandbox:
    """Async sandbox backed by a :class:`SandboxProvider`."""

    def __init__(self, provider: SandboxProvider, spec: SandboxSpec | None = None) -> None:
        self._provider = provider
        self._spec = spec
        self._handle: SandboxHandle | None = None
        self._started = False
        self._closed = False

    def _require_handle(self) -> SandboxHandle:
        if self._handle is None or not self._started:
            raise RuntimeError("Sandbox has not been started")
        return self._handle

    async def start(self, spec: SandboxSpec | None = None) -> AsyncSandbox:
        if self._closed:
            raise RuntimeError("Sandbox has been stopped")
        if self._started:
            raise RuntimeError("Sandbox is already started")
        resolved_spec = spec if spec is not None else self._spec
        if resolved_spec is None:
            raise ValueError("Sandbox.start() requires a SandboxSpec")

        handle = await self._provider.create(resolved_spec)
        # Seed startup files after the sandbox is up; tear the sandbox down on any seed failure so a
        # half-created sandbox never leaks.
        try:
            for target_path, contents in resolved_spec.files.items():
                await _write_file(self._provider, handle, target_path, contents)
        except BaseException:
            # Close just this half-created sandbox; the shared provider's lifetime is the owner's.
            await self._provider.close(handle)
            self._closed = True
            raise

        self._spec = resolved_spec
        self._handle = handle
        self._started = True
        return self

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        resolved_cwd = cwd if cwd is not None else (self._spec.workdir if self._spec is not None else None)
        return await self._provider.exec(
            self._require_handle(), command, cwd=resolved_cwd, env=env, timeout_s=timeout_s, stdin=stdin
        )

    async def upload_file(self, local_path: Path | str, remote_path: str) -> None:
        await self._provider.upload_file(self._require_handle(), Path(local_path), remote_path)

    async def upload_dir(self, local_dir: Path | str, remote_dir: str) -> None:
        await self._provider.upload_dir(self._require_handle(), Path(local_dir), remote_dir)

    async def download_file(self, remote_path: str, local_path: Path | str) -> None:
        await self._provider.download_file(self._require_handle(), remote_path, Path(local_path))

    async def download_dir(self, remote_dir: str, local_dir: Path | str) -> None:
        await self._provider.download_dir(self._require_handle(), remote_dir, Path(local_dir))

    async def status(self) -> SandboxStatus:
        if self._handle is None:
            return SandboxStatus.UNKNOWN
        if self._closed:
            return SandboxStatus.STOPPED
        return await self._provider.status(self._handle)

    async def stop(self) -> None:
        # Tears down only *this* sandbox. The provider is shared across sibling sandboxes, so its
        # process-wide resources (``aclose``) are the owner's to dispose — closing them here would tear
        # the provider down under still-running siblings when the first one exits.
        if self._closed:
            return
        self._closed = True
        if self._handle is not None and self._started:
            self._started = False
            await self._provider.close(self._handle)

    async def __aenter__(self) -> AsyncSandbox:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.stop()


async def _write_file(provider: SandboxProvider, handle: SandboxHandle, target_path: str, contents: str) -> None:
    """Write one text seed file into the sandbox via a host temp file + upload."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="nemo-eval-sandbox-seed-") as tmp_dir:
        source = Path(tmp_dir) / "seed"
        source.write_text(contents, encoding="utf-8")
        await provider.upload_file(handle, source, target_path)
