# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Literal, TextIO

import httpx
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR

ComposeLifecycle = Literal["fresh", "reuse"]
_DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS = 60


def _compose_env(env: dict[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return merged


def _compose_base_args(compose_file: Path, project_name: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), "-p", project_name]


def _parse_compose_ps_json(output: str) -> list[dict[str, object]]:
    text = output.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(entry, dict) for entry in parsed):
        return parsed
    raise ValueError("docker compose ps did not return JSON objects")


def _compose_exit_code(entry: dict[str, object]) -> int | None:
    exit_code = entry.get("ExitCode")
    if isinstance(exit_code, int):
        return exit_code
    if isinstance(exit_code, str) and exit_code:
        try:
            return int(exit_code)
        except ValueError:
            return None
    return None


def _compose_stack_readiness(entries: list[dict[str, object]], expected_services: set[str]) -> tuple[bool, list[str]]:
    entries_by_service = {
        str(entry.get("Service") or entry.get("Name")): entry
        for entry in entries
        if entry.get("Service") or entry.get("Name")
    }
    not_ready = []
    for service in sorted(expected_services):
        entry = entries_by_service.get(service)
        if entry is None:
            not_ready.append(f"{service} (missing)")
            continue
        state = str(entry.get("State") or "").lower()
        health = str(entry.get("Health") or "").lower()
        if health:
            if health != "healthy":
                not_ready.append(f"{service} (state={state or 'unknown'}, health={health})")
        elif service.endswith("-init") and state == "exited" and _compose_exit_code(entry) == 0:
            continue
        elif state != "running":
            not_ready.append(f"{service} (state={state or 'unknown'})")
    return not not_ready, not_ready


class DockerComposeE2EBackend:
    def __init__(
        self,
        *,
        compose_file: Path,
        config_path: Path,
        project_name: str,
        service_url: str,
        wait_url: str | None = None,
        wait_urls: list[str] | None = None,
        env: dict[str, str] | None = None,
        lifecycle: ComposeLifecycle = "fresh",
        wait_timeout_seconds: int = 180,
    ) -> None:
        self.compose_file = compose_file
        self.config_path = config_path
        self.project_name = project_name
        self.service_url = service_url
        self.wait_urls = wait_urls or [wait_url or service_url]
        self.wait_url = self.wait_urls[0]
        self.env = {
            "NEMO_COMPOSE_CONFIG_PATH": str(config_path.resolve()),
            **(env or {}),
        }
        self.lifecycle = lifecycle
        self.wait_timeout_seconds = wait_timeout_seconds

    def _run(self, *extra_args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
        args = _compose_base_args(self.compose_file, self.project_name)
        args.extend(extra_args)
        return subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=capture_output,
            env=_compose_env(self.env),
        )

    def exec(self, service: str, *command: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
        return self._run("exec", "-T", service, *command, capture_output=capture_output)

    def _services(self, *extra_args: str) -> set[str]:
        result = self._run(*extra_args, capture_output=True)
        return {line for line in result.stdout.splitlines() if line}

    def _ps_entries(self) -> list[dict[str, object]]:
        result = self._run("ps", "--all", "--format", "json", capture_output=True)
        return _parse_compose_ps_json(result.stdout)

    def _stack_readiness(self, expected_services: set[str]) -> tuple[bool, list[str]]:
        return _compose_stack_readiness(self._ps_entries(), expected_services)

    def start(self) -> None:
        expected_services = self._services("config", "--services")
        if not expected_services:
            raise RuntimeError(
                f"no services were discovered by docker compose config --services for {self.project_name}; "
                "compose startup cannot proceed"
            )
        if self.lifecycle == "reuse":
            ready, _not_ready = self._stack_readiness(expected_services)
            if ready:
                self._wait_ready()
                return
        else:
            try:
                self.stop()
            except subprocess.CalledProcessError:
                pass

        self._run("up", "-d")

        deadline = time.monotonic() + self.wait_timeout_seconds
        while time.monotonic() < deadline:
            ready, _not_ready = self._stack_readiness(expected_services)
            if ready:
                break
            time.sleep(2)
        else:
            _ready, not_ready = self._stack_readiness(expected_services)
            raise TimeoutError(f"compose services did not become ready for {self.project_name}: {not_ready}")

        self._wait_ready()

    def _wait_ready(self) -> None:
        verify = (
            self.env.get(NHX_CLIENT_SSL_CERT_FILE_ENVVAR)
            or self.env.get("REQUESTS_CA_BUNDLE")
            or self.env.get("SSL_CERT_FILE")
            or True
        )
        deadline = time.monotonic() + self.wait_timeout_seconds
        pending = list(dict.fromkeys(self.wait_urls))
        last_results: dict[str, str] = {}
        while time.monotonic() < deadline and pending:
            for wait_url in list(pending):
                try:
                    response = httpx.get(wait_url, timeout=5, verify=verify)
                    if response.status_code == 200:
                        pending.remove(wait_url)
                    else:
                        last_results[wait_url] = f"HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_results[wait_url] = str(exc)
            if not pending:
                return
            time.sleep(2)
        raise TimeoutError(f"compose backend did not become ready: {pending}; last_results={last_results}")

    def stop(self) -> None:
        if self.lifecycle == "reuse":
            return
        self._run("down", "-v")

    def write_logs(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            self._write_diagnostic_command(log_file, "docker compose config --services", ["config", "--services"])
            self._write_diagnostic_command(
                log_file,
                "docker compose ps --all --format json",
                ["ps", "--all", "--format", "json"],
            )
            self._write_diagnostic_command(log_file, "docker compose ps --all", ["ps", "--all"])
            self._write_diagnostic_command(
                log_file,
                "docker compose logs --no-color --timestamps",
                ["logs", "--no-color", "--timestamps"],
            )

    def _write_diagnostic_command(self, log_file: TextIO, title: str, extra_args: list[str]) -> None:
        args = _compose_base_args(self.compose_file, self.project_name)
        args.extend(extra_args)
        log_file.write(f"\n===== {title} =====\n")
        try:
            result = subprocess.run(
                args,
                check=False,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=_compose_env(self.env),
                timeout=_DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            log_file.write(f"\n[command timed out after {_DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS}s]\n")
            return
        log_file.write(f"\n[command exited with status {result.returncode}]\n")
