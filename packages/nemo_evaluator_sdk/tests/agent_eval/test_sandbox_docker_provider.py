# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for the Docker sandbox provider.

Mock at the single ``_run`` subprocess chokepoint (no real ``docker`` binary), then assert
the exact ``docker`` argv the provider builds for each operation — mirroring how NeMo Gym's
Apptainer provider tests assert their CLI command lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import (
    SANDBOX_RUNTIME_RETURN_CODE,
    SandboxCreateError,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers.docker import DockerSandboxProvider


class _Recorder:
    """Replacement for ``DockerSandboxProvider._run`` capturing argv + returning canned results."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.stdin: list[bytes | None] = []
        self._responses: dict[str, tuple[int, str, str]] = {}
        self.default = (0, "", "")
        self.raise_timeout_on: str | None = None

    def respond(self, contains: str, response: tuple[int, str, str]) -> None:
        self._responses[contains] = response

    async def __call__(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        self.calls.append(argv)
        self.stdin.append(stdin)
        joined = " ".join(argv)
        if self.raise_timeout_on is not None and self.raise_timeout_on in joined:
            raise TimeoutError(f"timed out: {argv}")
        for needle, response in self._responses.items():
            if needle in joined:
                return response
        return self.default


def _provider(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder, **kwargs: object) -> DockerSandboxProvider:
    provider = DockerSandboxProvider(**kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(provider, "_run", recorder)
    return provider


async def _created(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder, spec: SandboxSpec, **kwargs: object
) -> tuple[DockerSandboxProvider, SandboxHandle]:
    provider = _provider(monkeypatch, recorder, **kwargs)
    handle = await provider.create(spec)
    return provider, handle


async def test_create_builds_run_argv_with_env_resources_network(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(
        monkeypatch,
        recorder,
        SandboxSpec(
            image="fabric:latest",
            workdir="/workspace",
            env={"OPENAI_API_KEY": "sk-x"},
            resources=SandboxResources(cpu=2.0, memory_mib=4096, gpu=1),
        ),
        network="bridge",
    )
    argv = recorder.calls[0]
    assert argv[:5] == ["docker", "run", "-d", "--name", handle.sandbox_id]
    assert handle.sandbox_id.startswith("nemo-eval-sbx-")
    assert ["--network", "bridge"] == argv[5:7]
    assert "-w" in argv and "/workspace" in argv
    assert "-e" in argv and "OPENAI_API_KEY=sk-x" in argv
    assert ["--cpus", "2.0"] == [argv[argv.index("--cpus")], argv[argv.index("--cpus") + 1]]
    assert "--memory" in argv and "4096m" in argv
    assert "--gpus" in argv and "all" in argv
    # Keep-alive entrypoint so exec/cp can target the container across its lifetime.
    assert argv[-4:] == ["fabric:latest", "sh", "-c", "exec sleep infinity"]


async def test_create_raises_on_missing_image(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, _Recorder())
    with pytest.raises(SandboxCreateError, match="image is required"):
        await provider.create(SandboxSpec())


async def test_create_raises_on_nonzero_run(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    recorder.default = (1, "", "no such image")
    provider = _provider(monkeypatch, recorder)
    with pytest.raises(SandboxCreateError, match="docker run failed"):
        await provider.create(SandboxSpec(image="missing:latest"))
    # A failed `docker run -d` may leave a partially-created container — best-effort remove it by name.
    assert any(call[:3] == ["docker", "rm", "-f"] for call in recorder.calls)


async def test_create_timeout_removes_orphan_container(monkeypatch: pytest.MonkeyPatch) -> None:
    # A slow/killed `docker run -d` can still create the container; the timeout path must clean it up.
    recorder = _Recorder()
    recorder.raise_timeout_on = "run -d"
    provider = _provider(monkeypatch, recorder)
    with pytest.raises(SandboxCreateError, match="timed out"):
        await provider.create(SandboxSpec(image="slow:latest"))
    (rm,) = [call for call in recorder.calls if call[:3] == ["docker", "rm", "-f"]]
    assert rm[3].startswith("nemo-eval-sbx-")  # removed the same container it tried to create


async def test_exec_builds_argv_and_pipes_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    recorder.stdin.clear()

    result = await provider.exec(handle, "cat > /out.txt", cwd="/work", env={"A": "b"}, stdin=b"payload")
    assert result.ok and result.return_code == 0
    argv = recorder.calls[0]
    assert argv[:2] == ["docker", "exec"]
    assert "-i" in argv  # stdin present
    assert ["-w", "/work"] == [argv[argv.index("-w")], argv[argv.index("-w") + 1]]
    assert "-e" in argv and "A=b" in argv
    assert argv[-4:] == [handle.sandbox_id, "sh", "-c", "cat > /out.txt"]
    assert recorder.stdin[0] == b"payload"


async def test_exec_no_stdin_omits_dash_i(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    await provider.exec(handle, "true")
    assert "-i" not in recorder.calls[0]


async def test_exec_timeout_maps_to_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.raise_timeout_on = "sh -c"
    result = await provider.exec(handle, "sleep 999", timeout_s=1)
    assert not result.ok
    assert result.return_code == SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "timeout"


async def test_upload_file_mkdirs_parent_then_cp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    src = tmp_path / "f.py"
    src.write_text("x = 1")

    await provider.upload_file(handle, src, "/workspace/pkg/f.py")
    mkdir_argv, cp_argv = recorder.calls
    assert mkdir_argv[:2] == ["docker", "exec"] and "mkdir -p /workspace/pkg" in mkdir_argv[-1]
    assert cp_argv == ["docker", "cp", str(src), f"{handle.sandbox_id}:/workspace/pkg/f.py"]


async def test_upload_dir_copies_contents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()

    await provider.upload_dir(handle, tmp_path, "/workspace")
    mkdir_argv, cp_argv = recorder.calls
    assert "mkdir -p /workspace" in mkdir_argv[-1]
    # Trailing "/." copies the directory's contents into the target, not nested under it.
    assert cp_argv == ["docker", "cp", f"{tmp_path}/.", f"{handle.sandbox_id}:/workspace"]


async def test_download_dir_creates_local_target_and_cps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    dest = tmp_path / "out"

    await provider.download_dir(handle, "/out", dest)
    assert dest.is_dir()
    assert recorder.calls[0] == ["docker", "cp", f"{handle.sandbox_id}:/out/.", str(dest)]


async def test_download_file_creates_parent_and_cps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    dest = tmp_path / "nested" / "result.json"

    await provider.download_file(handle, "/out/fabric_result.json", dest)
    assert dest.parent.is_dir()
    assert recorder.calls[0] == ["docker", "cp", f"{handle.sandbox_id}:/out/fabric_result.json", str(dest)]


async def test_cp_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.respond("cp", (1, "", "no such file"))
    with pytest.raises(RuntimeError, match="docker cp"):
        await provider.download_file(handle, "/nope", tmp_path / "x")


async def test_status_maps_states(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.respond("inspect", (0, "running\n", ""))
    assert await provider.status(handle) == SandboxStatus.RUNNING
    recorder.respond("inspect", (0, "exited\n", ""))
    assert await provider.status(handle) == SandboxStatus.STOPPED
    recorder.respond("inspect", (1, "", "No such object"))
    assert await provider.status(handle) == SandboxStatus.STOPPED


async def test_close_force_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    provider, handle = await _created(monkeypatch, recorder, SandboxSpec(image="img"))
    recorder.calls.clear()
    await provider.close(handle)
    assert recorder.calls[0] == ["docker", "rm", "-f", handle.sandbox_id]


async def test_run_timeout_redacts_secrets_in_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real _run interpolates argv into its TimeoutError, and create()'s argv carries `-e KEY=<secret>`.
    # FabricAgentRuntime (sandbox mode) catches that error and persists it into error.json, so the message must
    # redact secrets rather than leak the API key into evaluation artifacts.
    from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers import docker as docker_mod

    class _HangingProc:
        pid = 4242
        returncode = None

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            await docker_mod.asyncio.sleep(3600)  # never returns before the timeout fires
            return b"", b""

        async def wait(self) -> None:
            return None

    async def _fake_exec(*argv: str, **kwargs: object) -> _HangingProc:
        return _HangingProc()

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(docker_mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(docker_mod.os, "killpg", lambda pgid, sig: None)

    provider = DockerSandboxProvider()
    argv = ["docker", "run", "-e", "NVIDIA_API_KEY=supersecret-xyz", "img:latest"]
    with pytest.raises(TimeoutError) as excinfo:
        await provider._run(argv, timeout_s=0.01)

    message = str(excinfo.value)
    assert "supersecret-xyz" not in message  # the resolved secret value must never appear
    assert "***REDACTED***" in message  # redaction was applied to the argv
