# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level harness for ``nemo insights`` CLI tests (fixtures live in ``conftest.py``).

The CLI takes its typed client from the ``nemo`` CLI state, so tests install a
stand-in state whose clients talk to :class:`FakeInsightsAPI` over an
``httpx.MockTransport``. Assertions are made on the HTTP requests the command
actually sent and on what it printed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import typer
from nemo_helix_plugin.cli_options import ListOutputFormat, TimestampFormat
from nemo_helix_plugin.cli_output import is_tty
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_insights_plugin import cli

BASE_URL = "http://nhx.test"
RUN_NAME = "insights-run-0123456789abcdef0123456789abcdef"
CONFIGURED_DEFAULT = "default/configured-big"
CONFIGURED_FAST = "default/configured-small"

_PATH = re.compile(r"^/apis/insights/v2/workspaces/(?P<workspace>[^/]+)(?P<rest>/.*)$")

ClientT = TypeVar("ClientT", bound=NemoClient)
AsyncClientT = TypeVar("AsyncClientT", bound=AsyncNemoClient)


def run_json(*, name: str = RUN_NAME, workspace: str = "default", agent: str = "demo-agent") -> dict[str, Any]:
    return {
        "name": name,
        "workspace": workspace,
        "agent": agent,
        "id": "run-id-1",
        "created_at": "2026-09-01T12:00:00Z",
        "updated_at": "2026-09-01T12:00:00Z",
        "created_by": "alice",
    }


def run_response_json(status: str | None = "created", **run: Any) -> dict[str, Any]:
    job = None if status is None else {"name": RUN_NAME, "status": status}
    return {"run": run_json(**run), "job": job}


def config_json(*, workspace: str = "default", agent: str = "demo-agent", enabled: bool = True) -> dict[str, Any]:
    return {
        "name": agent,
        "workspace": workspace,
        "agent": agent,
        "enabled": enabled,
        "default_model": CONFIGURED_DEFAULT,
        "fast_model": CONFIGURED_FAST,
        "id": "config-id-1",
        "created_at": "2026-09-01T12:00:00Z",
    }


def page_json(
    items: list[dict[str, Any]], *, page: int = 1, total_pages: int = 1, page_size: int = 20
) -> dict[str, Any]:
    return {
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "current_page_size": len(items),
            "total_pages": total_pages,
            "total_results": total_pages * len(items),
        },
        "sort": "-created_at",
    }


@dataclass
class RecordedRequest:
    method: str
    workspace: str
    path: str
    params: dict[str, str]
    body: Any


@dataclass
class FakeInsightsAPI:
    """Serves queued responses per ``(method, path after the workspace)`` and records requests.

    The last queued response for a route repeats, so a single response serves
    any number of polls.
    """

    routes: dict[tuple[str, str], list[httpx.Response]] = field(default_factory=dict)
    requests: list[RecordedRequest] = field(default_factory=list)

    def on(self, method: str, path: str, *bodies: Any, status: int = 200) -> None:
        self.routes[(method, path)] = [httpx.Response(status, json=body) for body in bodies]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        match = _PATH.match(request.url.path)
        assert match, f"unexpected path {request.url.path}"
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            RecordedRequest(
                method=request.method,
                workspace=match["workspace"],
                path=match["rest"],
                params=dict(request.url.params),
                body=body,
            )
        )
        queue = self.routes.get((request.method, match["rest"]))
        assert queue, f"no response queued for {request.method} {match['rest']}"
        return queue.pop(0) if len(queue) > 1 else queue[0]


@dataclass
class WireState:
    """Stand-in for the ``nemo`` CLI state (``CLIState``), backed by :class:`FakeInsightsAPI`."""

    api: FakeInsightsAPI
    workspace: str | None = "default"
    output_format: ListOutputFormat = "table"

    def get_client(self, timeout: float = 60.0) -> NemoClient:
        return NemoClient(base_url=BASE_URL, http_client=httpx.Client(transport=httpx.MockTransport(self.api)))

    def get_async_client(self, timeout: float = 60.0) -> AsyncNemoClient:
        raise NotImplementedError

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        return client_cls.from_client(self.get_client(timeout))

    def async_typed_client(self, client_cls: type[AsyncClientT], timeout: float = 60.0) -> AsyncClientT:
        raise NotImplementedError

    def get_workspace(self) -> str | None:
        return self.workspace

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(
        self, override: ListOutputFormat | None = None, *, apply_non_tty_default: bool = True
    ) -> ListOutputFormat:
        if override is not None:
            return override
        if apply_non_tty_default and self.output_format == "table" and not is_tty():
            return "json"
        return self.output_format

    def get_timestamp_format(self, override: TimestampFormat | None = None) -> TimestampFormat:
        return override or "iso8601"

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return bool(override)


def app_with_state(state: object | None) -> typer.Typer:
    """Mount the insights app under a parent that seeds ``ctx.obj`` the way ``nemo`` does."""
    parent = typer.Typer()

    @parent.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = state

    parent.add_typer(cli.InsightsCLI().get_cli(), name="insights")
    return parent
