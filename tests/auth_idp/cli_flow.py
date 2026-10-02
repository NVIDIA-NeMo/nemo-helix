# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_helix_plugin.client.oidc import OidcClientName

from tests.auth_idp.common import runtime_tls_config
from tests.auth_idp.runtime_contract import AuthIdpCase, AuthIdpRuntime

CLI_CONTEXT_NAME = "auth-idp-cli"
CLI_TIMEOUT_SECONDS = 300.0
_URL_PATTERN = re.compile(r"(https?://[^\s│]+)")
_VISIT_URL_PATTERN = re.compile(r"Visit:\s*(https?://[^\s│]+)")
_CONFIDENTIAL_PROMPT_PATTERN = re.compile(r"(Open this URL to log in:)")
_USER_CODE_PATTERN = re.compile(r"Enter code:\s*([^\s]+)")
_BROKER_CALLBACK_REDIRECT_STATUS_CODES = frozenset({302, 303, 307, 308})


@dataclass(frozen=True)
class CliLoginSession:
    config_path: Path
    environment: dict[str, str]

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return _run_cli(self.environment, *arguments)


class _RunningCli:
    def __init__(self, environment: dict[str, str], *arguments: str):
        self._lines: queue.Queue[str | None] = queue.Queue()
        self.output: list[str] = []
        self.process = subprocess.Popen(
            _cli_command(*arguments),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output.append(line)
            self._lines.put(line)
        self._lines.put(None)

    def wait_for(self, *patterns: re.Pattern[str]) -> tuple[str, ...]:
        deadline = time.monotonic() + CLI_TIMEOUT_SECONDS
        matches: list[str | None] = [None] * len(patterns)
        while any(match is None for match in matches):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise AssertionError(f"timed out waiting for CLI login output:\n{''.join(self.output)}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                self.stop()
                raise AssertionError(f"timed out waiting for CLI login output:\n{''.join(self.output)}") from exc
            if line is None:
                raise AssertionError(
                    f"CLI login exited before emitting its authorization prompt ({self.process.returncode}):\n"
                    f"{''.join(self.output)}"
                )
            for index, pattern in enumerate(patterns):
                if matches[index] is None and (match := pattern.search(line)):
                    matches[index] = match.group(1)
        return tuple(match for match in matches if match is not None)

    def finish(self) -> str:
        try:
            return_code = self.process.wait(timeout=CLI_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            self.stop()
            raise AssertionError(f"CLI login did not finish:\n{''.join(self.output)}") from exc
        self._reader.join(timeout=5)
        output = "".join(self.output)
        if return_code != 0:
            raise AssertionError(f"CLI login failed with exit code {return_code}:\n{output}")
        return output

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def login_with_real_cli(
    *,
    case: AuthIdpCase,
    runtime: AuthIdpRuntime,
    config_path: Path,
    client_name: OidcClientName,
) -> CliLoginSession:
    tls_config = runtime_tls_config(runtime)
    ca_bundle = tls_config.get("verify")
    if ca_bundle is None:
        raise AssertionError("real CLI auth tests require a CA bundle")
    environment = os.environ.copy()
    environment.update(
        {
            "NHX_CONFIG_FILE": str(config_path),
            NHX_CLIENT_SSL_CERT_FILE_ENVVAR: ca_bundle,
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "240",
            "XDG_STATE_HOME": str(config_path.parent / "state"),
        }
    )
    environment.pop("NHX_ACCESS_TOKEN", None)
    environment.pop("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", None)
    _run_cli(
        environment,
        "config",
        "set",
        "--context",
        CLI_CONTEXT_NAME,
        "--base-url",
        runtime.gateway_base_url,
        "--certificate-authority",
        ca_bundle,
        "--workspace",
        "default",
        "--activate",
    )

    login = _RunningCli(
        environment,
        "auth",
        "login",
        "--context",
        CLI_CONTEXT_NAME,
        "--oidc-client",
        client_name,
        "--no-browser",
    )
    try:
        if client_name == "public":
            verification_url, user_code = login.wait_for(_VISIT_URL_PATTERN, _USER_CODE_PATTERN)
            runtime.approve_device_authorization(
                verification_uri_complete=verification_url,
                user_code=user_code,
                username=case.provider.interactive_user_username,
                password=case.provider.interactive_user_password,
                tls_config=tls_config,
            )
        elif client_name == "confidential":
            login.wait_for(_CONFIDENTIAL_PROMPT_PATTERN)
            (authorization_url,) = login.wait_for(_URL_PATTERN)
            provider_callback_url = runtime.complete_confidential_authorization(
                authorization_url=authorization_url,
                username=case.provider.interactive_user_username,
                password=case.provider.interactive_user_password,
                tls_config=tls_config,
            )
            with httpx.Client(follow_redirects=False, **tls_config) as client:
                broker_callback = client.get(provider_callback_url, timeout=30.0)
                loopback_url = _broker_callback_location(broker_callback)
            parsed_loopback = urlparse(loopback_url)
            if parsed_loopback.hostname != "127.0.0.1":
                raise AssertionError(f"confidential CLI callback did not target loopback: {loopback_url}")
            httpx.get(loopback_url, timeout=30.0).raise_for_status()
        output = login.finish()
    finally:
        login.stop()
    if "Authentication successful!" not in output:
        raise AssertionError(f"CLI did not report successful authentication:\n{output}")
    return CliLoginSession(config_path=config_path, environment=environment)


def _broker_callback_location(response: httpx.Response) -> str:
    if response.status_code not in _BROKER_CALLBACK_REDIRECT_STATUS_CODES:
        response.raise_for_status()
        raise AssertionError(f"confidential broker callback did not redirect: {response.status_code}")
    location = response.headers.get("location")
    if not location:
        raise AssertionError("confidential broker callback redirect omitted its Location header")
    return location


def _cli_command(*arguments: str) -> list[str]:
    return [sys.executable, "-u", "-m", "nemo_helix_ext.cli.app", *arguments]


def _run_cli(environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        _cli_command(*arguments),
        env=environment,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"CLI command failed with exit code {completed.returncode}: {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed
