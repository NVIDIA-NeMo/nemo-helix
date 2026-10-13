# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fakes and helpers for Docker Compose sandbox provider tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxSpec
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_cli as compose_cli
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers.compose import (
    ComposeCommandResult,
    ComposeServiceTopology,
    DockerComposeSandboxProvider,
)

_TOPOLOGY = ComposeServiceTopology(
    target_service="agent",
    long_running_services=frozenset({"agent", "redis"}),
    one_shot_services=frozenset({"init"}),
)

_FILE_TARGET_OPERATION = "nemo-compose-file-target"
_FILE_REPAIR_OPERATION = "nemo-compose-file-repair"
_DIRECTORY_TARGET_OPERATION = "nemo-compose-directory-target"


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str], Path]] = []
        self.config: dict[str, Any] = {
            "services": {
                "agent": {},
                "redis": {},
                "init": {},
            }
        }
        self.config_stdout: str | None = None
        self.rows: list[dict[str, Any]] = [
            {"Service": "agent", "State": "running", "Health": "healthy", "ExitCode": 0},
            {"Service": "redis", "State": "running", "Health": "", "ExitCode": 0},
            {"Service": "init", "State": "exited", "Health": "", "ExitCode": 0},
        ]
        self.existing = False
        self.up_failure = False
        self.down_failures = 0
        self.failures: set[str] = set()
        self.directories: set[str] = set()
        self.file_prepare_result: tuple[int, str, str, bool] | None = None
        self.file_target_kinds: dict[str, str] = {}
        self.file_target_kinds_after_copy: dict[str, str] = {}
        self.file_repairs: list[tuple[str, str]] = []
        self.retained_file_targets: set[str] = set()
        self.prepared_file_targets: list[tuple[str, str]] = []
        self.directory_target_kinds: dict[str, str] = {}
        self.prepared_directory_targets: list[str] = []
        self.repair_failures_remaining = 0
        self._down_attempts = 0

    async def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout: float,
        stdin: bytes | None,
        stream_output: IO[str] | None = None,
    ) -> ComposeCommandResult:
        del timeout, stdin
        self.calls.append((argv, dict(environment), cwd))
        if argv[:2] != ("docker", "compose"):
            return ComposeCommandResult(argv, 0, "", "")

        args = _compose_suffix(argv)
        if args[:3] == ("config", "--format", "json"):
            stdout = self.config_stdout if self.config_stdout is not None else json.dumps(self.config)
            return ComposeCommandResult(argv, 0, stdout, "")
        if args[:3] == ("ps", "--all", "--quiet"):
            return ComposeCommandResult(argv, 0, "existing\n" if self.existing else "", "")
        if args[:4] == ("ps", "--all", "--format", "json"):
            return ComposeCommandResult(argv, 0, json.dumps(self.rows), "")
        if args[:1] == ("up",):
            if stream_output is not None:
                stream_output.write("compose progress\n")
            if self.up_failure:
                return ComposeCommandResult(argv, 1, "", "missing image")
        if args[:1] == ("down",):
            self._down_attempts += 1
            if self._down_attempts <= self.down_failures:
                return ComposeCommandResult(argv, 1, "", "temporary failure")
        if args[:5] == ("exec", "--no-TTY", "agent", "sh", "-c") and args[-3] == _FILE_TARGET_OPERATION:
            parent, target = args[-2:]
            if self.file_prepare_result is not None:
                return_code, stdout, stderr, timed_out = self.file_prepare_result
                return ComposeCommandResult(
                    argv,
                    return_code,
                    stdout,
                    stderr,
                    timed_out,
                )
            if "mkdir" in self.failures:
                return ComposeCommandResult(
                    argv,
                    1,
                    "",
                    f"token={environment.get('TEST_TOKEN', 'parent preparation failed')}",
                )
            self.directories.add(parent)
            self.prepared_file_targets.append((parent, target))
            if self.file_target_kinds.get(target, "absent") not in {"absent", "regular"}:
                return ComposeCommandResult(argv, 1, "", "unsafe exact file target")
            return ComposeCommandResult(argv, 0, "", "")
        if args[:1] == ("cp",) and "copy" in self.failures:
            return ComposeCommandResult(argv, 1, "", f"token={environment.get('TEST_TOKEN', 'copy failed')}")
        if args[:1] == ("cp",) and ":" in args[-1]:
            target = args[-1].split(":", 1)[1]
            self.retained_file_targets.add(target)
            self.file_target_kinds[target] = self.file_target_kinds_after_copy.get(target, "regular")
        if args[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c") and args[-3] == _FILE_REPAIR_OPERATION:
            target, identity = args[-2:]
            if self.file_target_kinds.get(target) != "regular":
                return ComposeCommandResult(argv, 1, "", "unsafe post-copy file target")
            if self.repair_failures_remaining:
                self.repair_failures_remaining -= 1
                return ComposeCommandResult(argv, 1, "", "temporary file repair failure")
            if "chown" in self.failures:
                return ComposeCommandResult(
                    argv,
                    1,
                    "",
                    f"token={environment.get('TEST_TOKEN', 'chown failed')}",
                )
            if "chmod" in self.failures:
                return ComposeCommandResult(
                    argv,
                    1,
                    "",
                    f"token={environment.get('TEST_TOKEN', 'chmod failed')}",
                )
            self.file_repairs.append((target, identity))
            return ComposeCommandResult(argv, 0, "", "")
        if (
            args[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c")
            and args[-2] == _DIRECTORY_TARGET_OPERATION
        ):
            target = args[-1]
            if "directory_prepare" in self.failures:
                return ComposeCommandResult(argv, 1, "", "directory preparation failed")
            if self.directory_target_kinds.get(target, "absent") not in {"absent", "directory"}:
                return ComposeCommandResult(argv, 1, "", "unsafe exact directory target")
            self.directories.add(target)
            self.prepared_directory_targets.append(target)
            return ComposeCommandResult(argv, 0, "", "")
        if args[:1] == ("exec",) and "printf" in args[-1]:
            if "identity" in self.failures:
                return ComposeCommandResult(argv, 1, "", f"token={environment.get('TEST_TOKEN', 'identity failed')}")
            return ComposeCommandResult(argv, 0, "1001:1002", "")
        if args[:6] == ("exec", "--no-TTY", "--user", "0", "agent", "mkdir") and "mkdir" in self.failures:
            return ComposeCommandResult(argv, 1, "", f"token={environment.get('TEST_TOKEN', 'mkdir failed')}")
        if args[:6] == ("exec", "--no-TTY", "--user", "0", "agent", "chown") and "chown" in self.failures:
            return ComposeCommandResult(argv, 1, "", f"token={environment.get('TEST_TOKEN', 'chown failed')}")
        return ComposeCommandResult(argv, 0, "", "")


def _files(tmp_path: Path, count: int = 1) -> tuple[Path, ...]:
    paths = tuple(tmp_path / f"compose-{index}.yaml" for index in range(count))
    for path in paths:
        path.write_text("services: {}\n", encoding="utf-8")
    return paths


def _provider(
    tmp_path: Path,
    *,
    files: tuple[Path, ...] | None = None,
    **kwargs: Any,
) -> DockerComposeSandboxProvider:
    lock_path = kwargs.pop("lock_path", tmp_path / "compose.lock")
    return DockerComposeSandboxProvider(
        compose_files=files or _files(tmp_path),
        service_topology=_TOPOLOGY,
        lock_path=lock_path,
        **kwargs,
    )


def _compose_suffix(argv: tuple[str, ...]) -> tuple[str, ...]:
    commands = {"config", "ps", "up", "down", "exec", "cp", "logs", "stop", "kill"}
    return argv[next(index for index, arg in enumerate(argv) if index > 1 and arg in commands) :]


async def _create(
    monkeypatch: pytest.MonkeyPatch,
    provider: DockerComposeSandboxProvider,
    runner: _Runner,
    spec: SandboxSpec | None = None,
):
    monkeypatch.setattr(compose_cli, "_run_command", runner)
    return await provider.create(spec or SandboxSpec())
