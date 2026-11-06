# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Studio plugin load-error report endpoint."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.config import Configuration
from nhx.studio.config import StudioConfig
from nhx.studio.service import StudioService

URL = "/studio/api/plugin-load-errors"
CODES = [
    "PLUGIN_MANIFEST_MISSING",
    "PLUGIN_SPECIFIER_UNMAPPED",
    "PLUGIN_SINGLETON_BUNDLED",
    "PLUGIN_CONTRACT_UNSUPPORTED",
    "PLUGIN_VERSION_INCOMPATIBLE",
    "PLUGIN_LOAD_FAILED",
]


def _issue(**overrides: Any) -> dict[str, Any]:
    issue = {"code": "PLUGIN_LOAD_FAILED", "subject": "pkg", "message": "boom", "steps": ["do a", "do b"]}
    issue.update(overrides)
    return issue


def _report(plugin: str = "my-plugin", **issue_overrides: Any) -> dict[str, Any]:
    return {"plugin": plugin, "issues": [_issue(**issue_overrides)]}


def _set_auth(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Toggle platform auth through the real AuthConfig env var (NHX_AUTH_ENABLED)."""
    monkeypatch.setenv("NHX_AUTH_ENABLED", "true" if enabled else "false")
    Configuration.clear_cache()


@pytest.fixture(autouse=True)
def _reset_config_cache():
    Configuration.clear_cache()
    yield
    Configuration.clear_cache()


def _client(flag: bool | None = None) -> TestClient:
    """Fresh service (and therefore fresh limiter state) per call."""
    return TestClient(StudioService().with_config(StudioConfig(plugin_error_reports_enabled=flag)).app)


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("nhx.studio") and r.levelno == logging.WARNING]


@pytest.fixture
def auth_off(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _set_auth(monkeypatch, False)
    caplog.set_level(logging.WARNING, logger="nhx.studio")


class TestPluginLoadErrorLogging:
    """Accepted reports produce exactly one well-formed log record."""

    def test_valid_report_logs_one_record(self, auth_off, caplog: pytest.LogCaptureFixture):
        """A valid report is acknowledged and logged as a single multi-line record."""
        body = {
            "plugin": "my-plugin",
            "issues": [
                _issue(code="PLUGIN_MANIFEST_MISSING", message="first msg", steps=["alpha", "beta"]),
                _issue(code="PLUGIN_LOAD_FAILED", subject="other", message="second msg"),
            ],
        }
        response = _client().post(URL, json=body)

        assert response.status_code == 204
        records = _records(caplog)
        assert len(records) == 1
        text = records[0].getMessage()
        for expected in ("my-plugin", "PLUGIN_MANIFEST_MISSING", "PLUGIN_LOAD_FAILED", "first msg", "second msg"):
            assert expected in text
        assert text.index("1.") < text.index("alpha") < text.index("2.") < text.index("beta")

    def test_control_characters_cannot_forge_log_lines(self, auth_off, caplog: pytest.LogCaptureFixture):
        """Newlines and control characters in fields are neutralised."""
        response = _client().post(
            URL, json=_report(message="x\nFAKE LOG LINE", steps=["go\r\x1b[31mred", "ok"], subject="s\nFAKE2")
        )

        assert response.status_code == 204
        records = _records(caplog)
        assert len(records) == 1
        text = records[0].getMessage()
        assert "FAKE LOG LINE" in text.replace("\n", " ")  # content kept, just defanged
        for line in text.splitlines():
            assert not line.lstrip().startswith("FAKE")
        assert not [c for c in text if ord(c) < 32 and c != "\n"]
        assert "\x7f" not in text


class TestPluginLoadErrorValidation:
    """Invalid payloads are rejected with 422 and never logged."""

    @pytest.mark.parametrize(
        "body",
        [
            _report(code="NOT_A_CODE"),
            _report(plugin="MyPlugin"),
            _report(plugin="a/b"),
            _report(plugin="a" * 65),
            {"plugin": "p", "issues": []},
            {"plugin": "p", "issues": [_issue()] * 11},
            _report(steps=["s"] * 9),
            _report(message="m" * 501),
            _report(steps=["s" * 301]),
            _report(subject="s" * 201),
            {"issues": [_issue()]},
            {"plugin": "p"},
            {"plugin": "p", "issues": [{"code": "PLUGIN_LOAD_FAILED"}]},
        ],
        ids=[
            "bad-code",
            "uppercase-plugin",
            "slash-plugin",
            "long-plugin",
            "no-issues",
            "11-issues",
            "9-steps",
            "long-message",
            "long-step",
            "long-subject",
            "missing-plugin",
            "missing-issues",
            "missing-issue-fields",
        ],
    )
    def test_invalid_payload_rejected(self, auth_off, caplog: pytest.LogCaptureFixture, body: dict[str, Any]):
        """Each violation yields 422, and a boundary-valid control request is accepted."""
        client = _client()
        assert client.post(URL, json=body).status_code == 422
        assert _records(caplog) == []
        assert client.post(URL, json=_report()).status_code == 204

    def test_maximum_lengths_accepted(self, auth_off, caplog: pytest.LogCaptureFixture):
        """Payloads exactly at every limit are valid."""
        issue = _issue(subject="s" * 200, message="m" * 500, steps=["t" * 300] * 8)
        body = {"plugin": "a" * 64, "issues": [issue] * 10}

        assert _client().post(URL, json=body).status_code == 204
        assert len(_records(caplog)) == 1


class TestPluginLoadErrorThrottling:
    """Global rate limit and duplicate suppression."""

    def test_rate_limit_21st_report_rejected(self, auth_off, caplog: pytest.LogCaptureFixture):
        """20 distinct reports pass; the 21st gets 429 and is not logged."""
        client = _client()
        for i in range(20):
            assert client.post(URL, json=_report(subject=f"s{i}")).status_code == 204
        assert len(_records(caplog)) == 20

        assert client.post(URL, json=_report(subject="s20")).status_code == 429
        assert len(_records(caplog)) == 20

    def test_duplicate_report_logged_once(self, auth_off, caplog: pytest.LogCaptureFixture):
        """An identical report is acknowledged twice but logged once; a new code set is logged."""
        client = _client()
        assert client.post(URL, json=_report()).status_code == 204
        assert client.post(URL, json=_report(message="different text, same codes")).status_code == 204
        assert len(_records(caplog)) == 1

        assert client.post(URL, json=_report(code="PLUGIN_MANIFEST_MISSING")).status_code == 204
        assert len(_records(caplog)) == 2


class TestPluginLoadErrorGating:
    """The endpoint is only available when enabled."""

    def test_gating_matrix(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        """False => 404; None follows auth; True forces on."""
        caplog.set_level(logging.WARNING, logger="nhx.studio")

        _set_auth(monkeypatch, False)
        assert _client(False).post(URL, json=_report("p-one")).status_code == 404
        assert _records(caplog) == []

        _set_auth(monkeypatch, True)
        assert _client(None).post(URL, json=_report("p-two")).status_code == 404
        assert _records(caplog) == []

        assert _client(True).post(URL, json=_report("p-three")).status_code == 204
        assert len(_records(caplog)) == 1

        _set_auth(monkeypatch, False)
        assert _client(None).post(URL, json=_report("p-four")).status_code == 204
        assert len(_records(caplog)) == 2

    def test_route_hidden_from_openapi(self, auth_off):
        """The endpoint is not advertised in the OpenAPI schema, yet is routable."""
        client = _client()
        assert client.post(URL, json=_report()).status_code == 204
        assert "plugin-load-errors" not in client.get("/openapi.json").text
