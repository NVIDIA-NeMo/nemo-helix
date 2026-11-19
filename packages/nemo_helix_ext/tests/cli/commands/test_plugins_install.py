# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for the external plugin commands in ``nemo plugins``.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, and JSON
body the command sends, and that a refused install sends no install request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_plugin.client.client import NemoClient
from typer.testing import CliRunner

MANIFEST_URL = "http://127.0.0.1:6007/plugin.yaml"
DIGEST = "7da9cfb5f13a7e5b94f5d96540321cdec8f191e764a2614bbf23a95231280832"
BUNDLE_URL = "http://127.0.0.1:6007/index.js"
BUNDLE_SHA = "4122e86b6b1ec5a1e5c7b3040672e23cd6cd20484282a0763d4bc19465ef1bb9"

PLUGIN = {
    "name": "phoenix",
    "title": "Phoenix traces (Intake)",
    "enabled": True,
    "endpoints": [
        {
            "method": "GET",
            "path": "/v2/workspaces/{workspace}/spans",
            "access": "read",
            "upstream_url": "http://127.0.0.1:6007/v1/workspaces/{workspace}/spans",
        },
        {
            "method": "POST",
            "path": "/v2/workspaces/{workspace}/spans",
            "access": "write",
            "upstream_url": "http://127.0.0.1:6007/v1/workspaces/{workspace}/spans",
        },
    ],
    "bundle": {
        "url": BUNDLE_URL,
        "sha256": BUNDLE_SHA,
        "sidecar": {
            "contractVersion": "1.0.0",
            "specifiers": ["react"],
            "builtAgainst": {"react": "19.3.1"},
        },
    },
    "navigation": [{"label": "Phoenix", "path": "/workspaces/{workspace}/plugin/phoenix", "section": "Community"}],
    "credential": {"secret_name": "phoenix-traces-adapter-key", "scheme": "bearer"},
}
PREVIEW = {
    "manifest": {"schema_version": 1, "plugin": PLUGIN},
    "manifest_sha256": DIGEST,
    "warning": "A UI bundle runs with Studio-origin privileges inside Studio, not in a sandbox.",
}


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
    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


def make_runner(recorder: Recorder) -> tuple[CliRunner, CLIContext]:
    client = NemoClient(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(recorder)))
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=client)
    return CliRunner(), state


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)


def _installed_response() -> httpx.Response:
    return httpx.Response(201, json={"name": "phoenix", "manifest_sha256": DIGEST, "bundleUrl": BUNDLE_URL})


def test_preview_posts_manifest_url_and_shows_what_is_approved() -> None:
    recorder = Recorder([httpx.Response(200, json=PREVIEW)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "preview", MANIFEST_URL, "-f", "raw"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls == [("POST", "/apis/plugins/-/preview")]
    assert json.loads(recorder.requests[0].content) == {"url": MANIFEST_URL}
    for expected in ("phoenix", "phoenix-traces-adapter-key", BUNDLE_SHA, DIGEST, "not in a sandbox"):
        assert expected in result.output


def test_preview_json_output_keeps_the_digest() -> None:
    runner, state = make_runner(Recorder([httpx.Response(200, json=PREVIEW)]))

    result = runner.invoke(app, ["plugins", "preview", MANIFEST_URL], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["manifest_sha256"] == DIGEST


def test_install_with_yes_and_matching_digest_pins_the_previewed_manifest() -> None:
    recorder = Recorder([httpx.Response(200, json=PREVIEW), _installed_response()])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["plugins", "install", MANIFEST_URL, "--manifest-sha256", DIGEST, "--yes", "-f", "raw"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.calls == [("POST", "/apis/plugins/-/preview"), ("POST", "/apis/plugins/-/installed")]
    assert json.loads(recorder.requests[1].content) == {"url": MANIFEST_URL, "manifest_sha256": DIGEST, "consent": True}
    assert "Installed 'phoenix'" in result.output


def test_install_yes_without_digest_is_refused_before_any_request() -> None:
    recorder = Recorder([])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "install", MANIFEST_URL, "--yes"], obj=state)

    assert result.exit_code == 2
    assert "--manifest-sha256" in result.output
    assert recorder.requests == []


def test_install_digest_mismatch_never_sends_the_install_request() -> None:
    recorder = Recorder([httpx.Response(200, json=PREVIEW)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "install", MANIFEST_URL, "--manifest-sha256", "0" * 64, "--yes"], obj=state)

    assert result.exit_code != 0
    assert "digest mismatch" in result.output.lower()
    assert recorder.calls == [("POST", "/apis/plugins/-/preview")]


def test_install_without_confirmation_does_not_install() -> None:
    recorder = Recorder([httpx.Response(200, json=PREVIEW)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "install", MANIFEST_URL], obj=state, input="n\n")

    assert result.exit_code != 0
    assert ("POST", "/apis/plugins/-/installed") not in recorder.calls


def test_installed_lists_registered_plugins() -> None:
    body = {"schema_version": 1, "plugins": [PLUGIN]}
    recorder = Recorder([httpx.Response(200, json=body)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "installed"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls == [("GET", "/apis/plugins/-/installed")]
    [row] = json.loads(result.stdout)
    assert row["name"] == "phoenix"
    assert row["endpoints"] == 2
    assert row["enabled"] is True


def test_uninstall_with_yes_deletes_the_registration() -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "uninstall", "phoenix", "--yes"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls == [("DELETE", "/apis/plugins/-/installed/phoenix")]


def test_uninstall_unknown_plugin_reports_the_404() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Plugin not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", "uninstall", "nope", "--yes"], obj=state)

    assert result.exit_code != 0


@pytest.mark.parametrize(("command", "enabled"), [("enable", True), ("disable", False)])
def test_enable_and_disable_patch_the_flag(command: str, enabled: bool) -> None:
    recorder = Recorder([httpx.Response(204)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["plugins", command, "phoenix"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls == [("PATCH", "/apis/plugins/-/installed/phoenix")]
    assert json.loads(recorder.requests[0].content) == {"enabled": enabled}
