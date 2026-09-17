# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for ``nemo agents redeploy``.

``redeploy`` wraps the forced four-step "iterate on a deployed agent" dance
(``undeploy`` -> ``delete`` -> ``create`` -> ``deploy``) into one command,
because the agent entity is immutable by name (``create`` 409s on a duplicate)
and there is no update route. These tests pin the user-visible contract:

- **Sequence order:** undeploy live deployments, delete the entity, recreate
  from the new config, then deploy.
- **Fail-fast:** an invalid config aborts *before* any teardown (zero side
  effects).
- **Confirmation:** ``--yes`` skips the prompt; declining aborts with no calls.
- **Idempotent skips:** no live deployments -> no deployment DELETE; agent
  already absent -> no agent DELETE, still proceeds to create.
- **``--wait`` exit codes:** running -> exit 0; failed -> exit 1.
- **Hybrid arg detection:** omitted runtime flags reuse the existing
  deployment; explicit flags override; disagreeing deployments error.
"""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from nemo_agents_plugin.cli import AgentsCLI
from typer.testing import CliRunner

runner = CliRunner()

_PATCH_PREFIX = "nemo_agents_plugin.cli"


def _install_mock_transport(handler) -> AbstractContextManager[Any]:
    """Patch ``httpx.Client`` in the CLI module to use a ``MockTransport``."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    return patch(f"{_PATCH_PREFIX}.httpx.Client", _Client)


def _page(data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": data,
        "pagination": {
            "page": 1,
            "page_size": len(data),
            "current_page_size": len(data),
            "total_pages": 1,
            "total_results": len(data),
        },
    }


def _write_config(tmp_path: Path, *, config_format: str | None = None) -> Path:
    """Write a minimal NAT-workflow agent config (no ethos fileset upload)."""
    cfg = tmp_path / "agent.yaml"
    lines = ["workflow:", "  _type: react_agent"]
    if config_format is not None:
        lines.insert(0, f"config_format: {config_format}")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


class _Recorder:
    """Records (method, path) of every request and drives status polling."""

    def __init__(
        self,
        *,
        deployments: list[dict[str, Any]],
        poll_statuses: list[str] | None = None,
        create_agent_status: int = 201,
    ) -> None:
        self.deployments = deployments
        self.calls: list[tuple[str, str]] = []
        self._poll = iter(poll_statuses or ["running"])
        self._create_agent_status = create_agent_status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        self.calls.append((req.method, path))
        # GET /deployments (list, for auto-detect + live filter)
        if req.method == "GET" and path.endswith("/deployments"):
            return httpx.Response(200, json=_page(self.deployments))
        # GET /deployments/{name} (wait polling)
        if req.method == "GET" and "/deployments/" in path:
            status = next(self._poll, "running")
            return httpx.Response(200, json={"name": "agent-new", "status": status, "endpoint": "http://x"})
        # DELETE /deployments/{name} (undeploy)
        if req.method == "DELETE" and "/deployments/" in path:
            return httpx.Response(204)
        # DELETE /agents/{name} (delete entity)
        if req.method == "DELETE" and "/agents/" in path:
            return httpx.Response(204)
        # GET /agents/{name} (idempotent existence check) — match the entity
        # route by its trailing ``/agents/{name}``, not the ``/apis/agents/v2/``
        # service prefix that every route carries.
        if req.method == "GET" and re.search(r"/agents/[^/]+$", path):
            return httpx.Response(200, json={"name": "my-agent"})
        # POST /agents (recreate)
        if req.method == "POST" and path.endswith("/agents"):
            return httpx.Response(self._create_agent_status, json={"name": "my-agent"})
        # POST /deployments (deploy)
        if req.method == "POST" and path.endswith("/deployments"):
            return httpx.Response(201, json={"name": "agent-new", "status": "pending", "agent": "my-agent"})
        return httpx.Response(404)

    def methods_for(self, needle: str) -> list[str]:
        return [m for (m, p) in self.calls if needle in p]

    def ordered_ops(self) -> list[str]:
        """A compact op sequence for order assertions."""
        ops: list[str] = []
        for method, path in self.calls:
            if method == "DELETE" and "/deployments/" in path:
                ops.append("undeploy")
            elif method == "DELETE" and "/agents/" in path:
                ops.append("delete")
            elif method == "POST" and path.endswith("/agents"):
                ops.append("create")
            elif method == "POST" and path.endswith("/deployments"):
                ops.append("deploy")
        return ops


# ---------------------------------------------------------------------------
# Sequence order
# ---------------------------------------------------------------------------


def test_redeploy_runs_steps_in_order(tmp_path: Path) -> None:
    """undeploy -> delete -> create -> deploy, in that order."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    assert rec.ordered_ops() == ["undeploy", "delete", "create", "deploy"]


# ---------------------------------------------------------------------------
# Fail-fast: bad config aborts before any teardown
# ---------------------------------------------------------------------------


def test_redeploy_bad_config_aborts_before_teardown(tmp_path: Path) -> None:
    """An unsupported config_format aborts with zero destructive calls."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}])
    bad = _write_config(tmp_path, config_format="totally_bogus")
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec):
        result = runner.invoke(
            app,
            ["redeploy", "--agent", "my-agent", "--agent-config", str(bad), "--yes", "--base-url", "http://test"],
        )
    assert result.exit_code != 0
    # No teardown or rebuild calls at all — validation is a pre-flight gate.
    assert rec.ordered_ops() == []
    assert "unsupported config_format" in result.output


# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------


def test_redeploy_prompts_and_proceeds_on_yes(tmp_path: Path) -> None:
    """Without --yes, the user is prompted; 'y' proceeds through all steps."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
            input="y\n",
        )
    assert result.exit_code == 0, result.output
    assert rec.ordered_ops() == ["undeploy", "delete", "create", "deploy"]


def test_redeploy_aborts_on_decline(tmp_path: Path) -> None:
    """Declining the prompt performs no destructive calls."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--base-url",
                "http://test",
            ],
            input="n\n",
        )
    assert result.exit_code != 0
    # The GET /deployments list happens (auto-detect) before the prompt, but no
    # DELETE/POST teardown-or-rebuild should occur.
    assert rec.ordered_ops() == []


# ---------------------------------------------------------------------------
# Idempotent skips
# ---------------------------------------------------------------------------


def test_redeploy_skips_undeploy_when_no_live_deployments(tmp_path: Path) -> None:
    """No live deployments -> no deployment DELETE, still deletes + recreates."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "failed"}])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    assert rec.ordered_ops() == ["delete", "create", "deploy"]
    assert "No live deployments" in result.output


def test_redeploy_skips_delete_when_agent_absent(tmp_path: Path) -> None:
    """Agent entity already gone (no deployments, 404 on lookup) -> no DELETE."""

    class _AbsentRecorder(_Recorder):
        def __call__(self, req: httpx.Request) -> httpx.Response:
            # Only the agent-entity lookup (``.../agents/{name}``) 404s; the
            # deployments list is a normal empty-200 (an absent agent simply has
            # no deployments). Match the entity route by its trailing
            # ``/agents/{name}`` so the ``/apis/agents/v2/`` service prefix that
            # every route carries is not caught.
            path = req.url.path
            if req.method == "GET" and re.search(r"/agents/[^/]+$", path):
                self.calls.append((req.method, path))
                return httpx.Response(404)
            return super().__call__(req)

    rec = _AbsentRecorder(deployments=[])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    # No deployment DELETE and no agent DELETE; proceeds straight to create+deploy.
    assert rec.ordered_ops() == ["create", "deploy"]


# ---------------------------------------------------------------------------
# --wait exit codes
# ---------------------------------------------------------------------------


def test_redeploy_wait_exits_1_on_failed(tmp_path: Path) -> None:
    """When the new deployment fails readiness, redeploy exits 1."""
    rec = _Recorder(
        deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}],
        poll_statuses=["pending", "failed"],
    )
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 1, result.output
    assert rec.ordered_ops() == ["undeploy", "delete", "create", "deploy"]


def test_redeploy_no_wait_returns_pending_json(tmp_path: Path) -> None:
    """--no-wait returns the pending deployment without polling."""
    rec = _Recorder(deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}])
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--no-wait",
                "--base-url",
                "http://test",
            ],
        )
    assert result.exit_code == 0, result.output
    # No GET /deployments/{name} poll happened.
    assert not any(m == "GET" and "/deployments/" in p for (m, p) in rec.calls)


# ---------------------------------------------------------------------------
# Hybrid arg detection
# ---------------------------------------------------------------------------


def test_redeploy_autodetects_mode_from_existing_deployment(tmp_path: Path) -> None:
    """Omitted --mode is reused from the existing deployment (docker)."""
    posted: dict[str, Any] = {}

    class _CapturingRecorder(_Recorder):
        def __call__(self, req: httpx.Request) -> httpx.Response:
            if req.method == "POST" and req.url.path.endswith("/deployments"):
                import json as _json

                posted.update(_json.loads(req.content))
            return super().__call__(req)

    rec = _CapturingRecorder(
        deployments=[
            {"name": "dep-1", "agent": "my-agent", "status": "running", "deployment_mode": "docker", "image": "img:1"}
        ]
    )
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    assert posted.get("deployment_mode") == "docker"
    assert posted.get("image") == "img:1"


def test_redeploy_explicit_flag_overrides_detection(tmp_path: Path) -> None:
    """An explicit --mode wins over the existing deployment's mode."""
    posted: dict[str, Any] = {}

    class _CapturingRecorder(_Recorder):
        def __call__(self, req: httpx.Request) -> httpx.Response:
            if req.method == "POST" and req.url.path.endswith("/deployments"):
                import json as _json

                posted.update(_json.loads(req.content))
            return super().__call__(req)

    rec = _CapturingRecorder(
        deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running", "deployment_mode": "docker"}]
    )
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--mode",
                "subprocess",
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    assert posted.get("deployment_mode") == "subprocess"


def test_redeploy_errors_when_deployments_disagree(tmp_path: Path) -> None:
    """Multiple deployments disagreeing on an omitted field errors."""
    rec = _Recorder(
        deployments=[
            {"name": "dep-1", "agent": "my-agent", "status": "running", "deployment_mode": "docker"},
            {"name": "dep-2", "agent": "my-agent", "status": "running", "deployment_mode": "subprocess"},
        ]
    )
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
            ],
        )
    assert result.exit_code == 2, result.output
    assert "disagree" in result.output
    # Errored during arg resolution, before any teardown.
    assert rec.ordered_ops() == []


# ---------------------------------------------------------------------------
# Recovery hint on post-delete failure
# ---------------------------------------------------------------------------


def test_redeploy_recovery_hint_on_recreate_failure(tmp_path: Path) -> None:
    """If recreate fails after delete, print a re-run recovery hint."""
    rec = _Recorder(
        deployments=[{"name": "dep-1", "agent": "my-agent", "status": "running"}],
        create_agent_status=500,
    )
    app = AgentsCLI().get_cli()
    with _install_mock_transport(rec), patch(f"{_PATCH_PREFIX}.time.sleep"):
        result = runner.invoke(
            app,
            [
                "redeploy",
                "--agent",
                "my-agent",
                "--agent-config",
                str(_write_config(tmp_path)),
                "--yes",
                "--base-url",
                "http://test",
                "--timeout",
                "10",
            ],
        )
    assert result.exit_code != 0
    assert "re-run to finish" in result.output
    assert "nemo agents redeploy --agent my-agent" in result.output
    # It got as far as delete but recreate failed (no deploy).
    assert rec.ordered_ops() == ["undeploy", "delete", "create"]
