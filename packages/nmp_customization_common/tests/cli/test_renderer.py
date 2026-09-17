# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared customization submit renderer.

Covers the URL builder (encoding, no-base-url), the Studio ``/status``
availability probe (ready / not-ready / unreachable), and the renderer's
completion output (link shown only when Studio is ready + a job name resolved;
CLI tracking instructions always shown).
"""

from __future__ import annotations

import httpx
import pytest
from nemo_platform_plugin.cli_renderer import RendererContext
from nmp.customization_common.cli import renderer as renderer_mod
from nmp.customization_common.cli.renderer import (
    CustomizationSubmitRenderer,
    build_studio_customization_url,
    studio_is_available,
)
from rich.console import Console

# --------------------------------------------------------------------------- #
# URL builder                                                                 #
# --------------------------------------------------------------------------- #


def test_build_url_basic() -> None:
    url = build_studio_customization_url("https://nmp.test", "default", "job-1")
    assert url == "https://nmp.test/studio/workspaces/default/customizations/job-1"


def test_build_url_strips_trailing_slash() -> None:
    url = build_studio_customization_url("https://nmp.test/", "default", "job-1")
    assert url == "https://nmp.test/studio/workspaces/default/customizations/job-1"


def test_build_url_encodes_custom_workspace_and_job() -> None:
    # A workspace/job with spaces + slash must be percent-encoded (safe="").
    url = build_studio_customization_url("https://nmp.test", "acme corp/eu", "sft job/1")
    assert url == "https://nmp.test/studio/workspaces/acme%20corp%2Feu/customizations/sft%20job%2F1"


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_build_url_no_base_returns_none(base_url: str | None) -> None:
    assert build_studio_customization_url(base_url, "default", "job-1") is None


# --------------------------------------------------------------------------- #
# Studio availability probe                                                   #
# --------------------------------------------------------------------------- #


_PROBE_REQUEST = httpx.Request("GET", "https://nmp.test/status")


def _status_response(ready: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"status": "healthy", "services": {"ready": ready, "not_ready": []}},
        request=_PROBE_REQUEST,
    )


def test_studio_available_when_in_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        return _status_response(["entities", "studio", "jobs"])

    monkeypatch.setattr(renderer_mod.httpx, "get", fake_get)
    assert studio_is_available("https://nmp.test") is True
    assert captured["url"] == "https://nmp.test/status"


def test_studio_not_available_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer_mod.httpx, "get", lambda url, **kw: _status_response(["entities", "jobs"]))
    assert studio_is_available("https://nmp.test") is False


def test_studio_not_available_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(renderer_mod.httpx, "get", boom)
    assert studio_is_available("https://nmp.test") is False


def test_studio_not_available_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        renderer_mod.httpx, "get", lambda url, **kw: httpx.Response(503, json={}, request=_PROBE_REQUEST)
    )
    assert studio_is_available("https://nmp.test") is False


def test_studio_not_available_on_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        renderer_mod.httpx,
        "get",
        lambda url, **kw: httpx.Response(200, json={"unexpected": 1}, request=_PROBE_REQUEST),
    )
    assert studio_is_available("https://nmp.test") is False


@pytest.mark.parametrize("base_url", [None, "", "  "])
def test_studio_not_available_without_base_url(base_url: str | None) -> None:
    # No probe should be attempted; a missing base URL is simply "not available".
    assert studio_is_available(base_url) is False


# --------------------------------------------------------------------------- #
# Renderer completion output                                                  #
# --------------------------------------------------------------------------- #


def _ctx(base_url: str | None, workspace: str = "default") -> RendererContext:
    return RendererContext(
        console=Console(force_terminal=False, no_color=True, width=200),
        cli_kwargs={"workspace": workspace},
        verb="submit",
        is_local=False,
        base_url=base_url,
    )


def _run_complete(ctx: RendererContext, result: object, capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    """Run on_frame+on_complete, returning (stdout, stderr) via capsys.

    stdout carries the pure-JSON echo (typer.echo); stderr carries the human
    guidance + link (the renderer's Console(stderr=True)). capsys captures the
    real fds, so this also proves the stdout/stderr split holds end to end.
    """
    renderer = CustomizationSubmitRenderer()
    renderer.on_frame(result, ctx=ctx)
    renderer.on_complete(ctx=ctx)
    captured = capsys.readouterr()
    return captured.out, captured.err


def test_renderer_prints_link_when_studio_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(renderer_mod, "studio_is_available", lambda base_url: True)
    out, err = _run_complete(_ctx("https://nmp.test", workspace="acme"), {"name": "job-1"}, capsys)
    # Link + guidance go to STDERR, not stdout.
    assert "View in Studio: https://nmp.test/studio/workspaces/acme/customizations/job-1" in err
    assert "nemo jobs list --workspace acme" in err
    assert "nemo jobs get-status job-1 --workspace acme" in err
    # stdout is PURE JSON (the automation contract) — no guidance leaks into it.
    assert '"name": "job-1"' in out
    assert "View in Studio" not in out
    assert "nemo jobs" not in out


def test_renderer_omits_link_when_studio_not_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(renderer_mod, "studio_is_available", lambda base_url: False)
    out, err = _run_complete(_ctx("https://nmp.test"), {"name": "job-1"}, capsys)
    assert "View in Studio" not in err
    # Fallback: CLI tracking instructions still shown (on stderr).
    assert "nemo jobs list --workspace default" in err
    assert '"name": "job-1"' in out


def test_renderer_omits_link_when_no_base_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Even if the probe were somehow truthy, a missing base URL yields no link.
    monkeypatch.setattr(renderer_mod, "studio_is_available", lambda base_url: bool(base_url))
    out, err = _run_complete(_ctx(None), {"name": "job-1"}, capsys)
    assert "View in Studio" not in err
    assert "nemo jobs list --workspace default" in err
    assert '"name": "job-1"' in out


def test_renderer_omits_link_when_no_job_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No name in the response -> no link and no `get-status` hint, but list hint stays.
    called = {"probed": False}

    def fake_probe(base_url: str | None) -> bool:
        called["probed"] = True
        return True

    monkeypatch.setattr(renderer_mod, "studio_is_available", fake_probe)
    out, err = _run_complete(_ctx("https://nmp.test"), {"id": "internal-uuid"}, capsys)
    assert "View in Studio" not in err
    assert "nemo jobs get-status" not in err
    assert "nemo jobs list --workspace default" in err
    # The probe is skipped entirely when there's no job name to link to.
    assert called["probed"] is False
    # The raw response is still echoed to stdout.
    assert '"id": "internal-uuid"' in out


def test_renderer_defaults_workspace_when_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(renderer_mod, "studio_is_available", lambda base_url: True)
    ctx = RendererContext(
        console=Console(force_terminal=False, no_color=True, width=200),
        cli_kwargs={},  # no workspace
        verb="submit",
        is_local=False,
        base_url="https://nmp.test",
    )
    _out, err = _run_complete(ctx, {"name": "job-1"}, capsys)
    assert "workspaces/default/customizations/job-1" in err


def test_renderer_shell_quotes_workspace_with_spaces(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A workspace with a space must be shell-quoted in the copy-pasteable command
    # hints (so the command parses as one --workspace arg), while the URL path
    # segment is percent-encoded.
    monkeypatch.setattr(renderer_mod, "studio_is_available", lambda base_url: True)
    _out, err = _run_complete(_ctx("https://nmp.test", workspace="acme corp"), {"name": "job-1"}, capsys)
    assert "nemo jobs list --workspace 'acme corp'" in err
    assert "nemo jobs get-status job-1 --workspace 'acme corp'" in err
    # URL segment is percent-encoded, not shell-quoted.
    assert "workspaces/acme%20corp/customizations/job-1" in err
