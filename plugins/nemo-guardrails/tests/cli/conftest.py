# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the ``nemo guardrail`` plugin CLI tests.

The plugin Typer app is driven directly with a ``CLIContext`` whose client is
a ``NemoClient`` over ``httpx.MockTransport``, so every test pins the exact
HTTP method, path, query string, and JSON body a command sends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import httpx
import pytest
import typer
from click.testing import Result
from nemo_guardrails_plugin.cli import GuardrailCLI
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.config.config import ConfigParams
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner


@dataclass
class Recorder:
    """Records requests and answers each with the next scripted response."""

    responses: list[httpx.Response]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


@dataclass
class CliHarness:
    """Runs the plugin Typer app against a scripted typed client."""

    app: typer.Typer

    def run(
        self,
        args: list[str],
        responses: list[httpx.Response] | None = None,
        *,
        workspace: str | None = "default",
        input: str | None = None,
        output_format: Literal["table", "json", "yaml", "markdown", "csv", "raw"] = "json",
    ) -> tuple[Result, Recorder]:
        recorder = Recorder(list(responses or []))
        client = NemoClient(
            base_url="http://test",
            workspace=workspace,
            http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
        )
        overrides: ConfigParams = {"base_url": "http://test", "output_format": output_format}
        state = CLIContext(overrides=overrides, _client=client)
        result = CliRunner().invoke(self.app, args, obj=state, input=input)
        return result, recorder

    @staticmethod
    def page(items: list[dict], page: int, total_pages: int, *, page_size: int = 1) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": items,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "current_page_size": len(items),
                    "total_pages": total_pages,
                    "total_results": total_pages * page_size,
                },
            },
        )


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    for var in ("NMP_ACCESS_TOKEN", "NMP_BASE_URL", "NMP_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def guardrail_cli() -> CliHarness:
    return CliHarness(GuardrailCLI().get_cli())
