# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for onboarding_step telemetry emitted from the setup flow.

Each instrumented choke point in ``setup.py`` emits exactly one
``OnboardingStepEvent``: COMPLETED on success, ERROR on failure with the
original exception still propagating. ``emit_event`` is patched at its
definition module so the module-attribute call in ``setup.py`` is intercepted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from nemo_platform_ext.cli.commands.setup import (
    SetupClients,
    _bucket_model_count,
    _create_provider,
    _deploy_demo_agent,
    _run_skill_install,
    _update_provider,
    _wait_for_models,
    setup_command,
)
from nemo_platform_ext.cli.commands.skills.base import Scope, Skill
from nemo_platform_ext.cli.telemetry.events import TaskStatusEnum
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.workspaces.client import WorkspacesClient

SETUP_MOD = "nemo_platform_ext.cli.commands.setup"
EMIT_TARGET = "nemo_platform_ext.cli.telemetry.emit.emit_event"


class _StubWorkspaces:
    """Workspaces typed-client stub whose calls always succeed."""

    def get_workspace(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(data=lambda: object())

    def create_workspace(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(data=lambda: object())


def _provider_response(provider: object) -> MagicMock:
    """Mimic the ``NemoResponse`` ``get_provider`` returns, whose ``.data()`` is *provider*."""
    response = MagicMock()
    response.data.return_value = provider
    return response


@pytest.fixture
def spinner_console():
    """Patch setup.console with a mock status spinner context manager."""
    mock_status = MagicMock()
    with patch(f"{SETUP_MOD}.console") as mock_console:
        mock_console.status.return_value.__enter__ = MagicMock(return_value=mock_status)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_console, mock_status


def _providers_client() -> Any:
    """Typed-client bundle with spec'd MagicMock service clients (typed ``Any`` to reach mock attributes)."""
    return SetupClients(
        models=MagicMock(spec=ModelsClient),
        secrets=MagicMock(spec=SecretsClient),
        gateway=MagicMock(spec=InferenceGatewayClient),
    )


def _skill(name: str) -> Skill:
    return Skill(
        name=name,
        description=f"{name} desc",
        version="0.1",
        content=f"# {name}",
        raw=f"---\nname: {name}\n---\n# {name}",
        source_plugin=None,
    )


class TestCreateProviderTelemetry:
    @patch(EMIT_TARGET)
    def test_success_emits_completed(self, emit):
        client = _providers_client()
        _create_provider(
            client,
            name="openai",
            host_url="https://api.openai.com/v1",
            secret_name="openai-api-key",
            workspace="default",
        )
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "provider_connected"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.provider_type == "openai"

    @patch(EMIT_TARGET)
    def test_failure_emits_error_and_reraises(self, emit):
        client = _providers_client()
        client.models.create_provider.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            _create_provider(
                client,
                name="anthropic",
                host_url="https://api.anthropic.com",
                secret_name="anthropic-api-key",
                workspace="default",
            )
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "provider_connected"
        assert event.task_status == TaskStatusEnum.ERROR
        assert event.provider_type == "anthropic"

    @patch(EMIT_TARGET)
    def test_custom_provider_emits_fixed_custom_provider_type(self, emit):
        client = _providers_client()
        _create_provider(
            client,
            name="team-provider",
            host_url="https://llm.example.test/v1",
            secret_name="team-provider-api-key",
            workspace="default",
        )
        event = emit.call_args[0][0]
        assert event.provider_type == "custom"


class TestUpdateProviderTelemetry:
    @patch(EMIT_TARGET)
    def test_success_emits_completed(self, emit):
        client = _providers_client()
        _update_provider(
            client,
            name="openai",
            host_url="https://api.openai.com/v1",
            secret_name="openai-api-key",
            workspace="default",
        )
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "provider_connected"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.provider_type == "openai"

    @patch(EMIT_TARGET)
    def test_failure_emits_error_and_reraises(self, emit):
        client = _providers_client()
        client.models.upsert_provider.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            _update_provider(
                client,
                name="team-provider",
                host_url="https://llm.example.test/v1",
                secret_name="team-provider-api-key",
                workspace="default",
            )
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "provider_connected"
        assert event.task_status == TaskStatusEnum.ERROR
        assert event.provider_type == "custom"


class TestWaitForModelsTelemetry:
    @pytest.mark.usefixtures("spinner_console")
    @patch(EMIT_TARGET)
    def test_emits_completed_with_bucket(self, emit):
        client = _providers_client()
        models = [MagicMock(model_entity_id=f"default/model-{i}") for i in range(3)]
        client.models.get_provider.return_value = _provider_response(MagicMock(served_models=models))

        with (
            patch(f"{SETUP_MOD}._pause"),
            patch(f"{SETUP_MOD}.time.monotonic", side_effect=[0, 0, 0, 1]),
        ):
            result = _wait_for_models(client, "nvidia-build", "default", round_seconds=30, max_rounds=1)

        assert result == [f"default/model-{i}" for i in range(3)]
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "models_discovered"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.models_discovered_bucket == "1-5"

    @pytest.mark.usefixtures("spinner_console")
    @patch(EMIT_TARGET)
    def test_no_models_emits_bucket_zero(self, emit):
        client = _providers_client()
        client.models.get_provider.return_value = _provider_response(MagicMock(served_models=[]))

        with (
            patch(f"{SETUP_MOD}._pause"),
            patch(f"{SETUP_MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 100]),
        ):
            result = _wait_for_models(client, "nvidia-build", "default", round_seconds=5, max_rounds=1)

        assert result == []
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "models_discovered"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.models_discovered_bucket == "0"


class TestRunSkillInstallTelemetry:
    def _run(self, *, install_raises: bool):
        installer = MagicMock()
        if install_raises:
            installer.install.side_effect = RuntimeError("install failed")
        all_skills = {"alpha": _skill("alpha")}
        with patch(f"{SETUP_MOD}.get_installer", return_value=installer):
            _run_skill_install(
                agents=["codex"],
                scope=Scope.PROJECT,
                skill_names=["alpha"],
                all_skills=all_skills,
                project_root=MagicMock(),
            )

    @patch(EMIT_TARGET)
    def test_success_emits_completed(self, emit):
        self._run(install_raises=False)
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "skills_installed"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert "codex" in event.skills_target

    @patch(EMIT_TARGET)
    def test_failure_emits_error_and_reraises(self, emit):
        with pytest.raises(typer.Exit):
            self._run(install_raises=True)
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "skills_installed"
        assert event.task_status == TaskStatusEnum.ERROR
        assert "codex" in event.skills_target


class TestDeployDemoAgentTelemetry:
    def _responses(self, *, status_sequence):
        create_resp = MagicMock(status_code=200)
        create_resp.raise_for_status = MagicMock()
        deploy_resp = MagicMock(status_code=200)
        deploy_resp.raise_for_status = MagicMock()
        deploy_resp.json.return_value = {"name": "calculator-agent-abc12345"}
        status_resps = []
        for s in status_sequence:
            r = MagicMock(status_code=200)
            r.json.return_value = {"name": "calculator-agent-abc12345", "status": s}
            status_resps.append(r)
        return [create_resp, deploy_resp] + status_resps

    @pytest.mark.usefixtures("spinner_console")
    @patch(EMIT_TARGET)
    def test_success_emits_completed_true(self, emit, tmp_path):
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")
        responses = self._responses(status_sequence=["running"])
        with (
            patch(f"{SETUP_MOD}.httpx.get", side_effect=responses[2:]),
            patch(f"{SETUP_MOD}.httpx.post", side_effect=responses[:2]),
            patch(f"{SETUP_MOD}._agent_exists", return_value=False),
            patch(f"{SETUP_MOD}._pause"),
            patch(f"{SETUP_MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 3]),
        ):
            result = _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")
        assert result is True
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "agent_deployed"
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.agent_deployed is True

    @pytest.mark.usefixtures("spinner_console")
    @patch(EMIT_TARGET)
    def test_failure_emits_error_and_reraises(self, emit, tmp_path):
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")
        create_resp = MagicMock(status_code=500)
        create_resp.raise_for_status = MagicMock(side_effect=RuntimeError("http 500"))
        with (
            patch(f"{SETUP_MOD}.httpx.post", return_value=create_resp),
            patch(f"{SETUP_MOD}._agent_exists", return_value=False),
        ):
            with pytest.raises(RuntimeError):
                _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "agent_deployed"
        assert event.task_status == TaskStatusEnum.ERROR
        assert event.agent_deployed is False

    @patch(EMIT_TARGET)
    def test_false_deploy_outcome_emits_error_false(self, emit, tmp_path):
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")
        with patch(f"{SETUP_MOD}._deploy_demo_agent_impl", return_value=False):
            result = _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")

        assert result is False
        assert emit.call_count == 1
        event = emit.call_args[0][0]
        assert event.step == "agent_deployed"
        assert event.task_status == TaskStatusEnum.ERROR
        assert event.agent_deployed is False


class TestBucketModelCount:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0"),
            (1, "1-5"),
            (5, "1-5"),
            (6, "6-20"),
            (20, "6-20"),
            (21, "21-50"),
            (50, "21-50"),
            (51, "51-100"),
            (100, "51-100"),
            (101, "101-250"),
            (250, "101-250"),
            (251, "251+"),
            (5000, "251+"),
        ],
    )
    def test_buckets(self, count, expected):
        assert _bucket_model_count(count) == expected


class TestSetupFinishedTelemetry:
    """setup_finished must reflect the real outcome of the dispatch.

    A clean user-cancel raises typer.Exit(0); that is a normal end, not a
    failure, so it must emit COMPLETED. Any non-zero Exit (or real exception)
    emits ERROR. In every case the Exit propagates unchanged.
    """

    def _run(self, dispatch_exc):
        """Drive setup_command to the auto/interactive dispatch, patching all
        preamble so only the dispatch outcome matters. `dispatch_exc` is raised
        from _run_interactive_mode; None means a clean finish."""
        ctx = MagicMock(spec=typer.Context)
        ctx.obj = MagicMock()
        ctx.obj.get_base_url.return_value = "http://localhost:3000"
        ctx.obj.typed_client.side_effect = lambda client_cls, timeout=60.0: (
            _StubWorkspaces() if client_cls is WorkspacesClient else MagicMock()
        )

        interactive = MagicMock()
        if dispatch_exc is not None:
            interactive.side_effect = dispatch_exc

        with (
            patch(f"{SETUP_MOD}.console"),
            patch(f"{SETUP_MOD}.is_interactive", return_value=True),
            patch(f"{SETUP_MOD}._maybe_start_services"),
            patch(f"{SETUP_MOD}._check_platform_reachable_with_retries", return_value=True),
            patch(f"{SETUP_MOD}._bootstrap_config_if_missing"),
            patch(f"{SETUP_MOD}._run_interactive_mode", interactive),
        ):
            setup_command(ctx)

    @patch(EMIT_TARGET)
    def test_clean_exit_zero_emits_completed(self, emit):
        with pytest.raises(typer.Exit) as exc_info:
            self._run(typer.Exit(0))
        assert exc_info.value.exit_code == 0
        finished = [c.args[0] for c in emit.call_args_list if c.args[0].step == "setup_finished"]
        assert len(finished) == 1
        assert finished[0].task_status == TaskStatusEnum.COMPLETED

    @patch(EMIT_TARGET)
    def test_default_exit_emits_completed(self, emit):
        # typer.Exit() defaults to exit_code 0, so a bare Exit is a clean end too.
        with pytest.raises(typer.Exit):
            self._run(typer.Exit())
        finished = [c.args[0] for c in emit.call_args_list if c.args[0].step == "setup_finished"]
        assert len(finished) == 1
        assert finished[0].task_status == TaskStatusEnum.COMPLETED

    @patch(EMIT_TARGET)
    def test_nonzero_exit_emits_error(self, emit):
        with pytest.raises(typer.Exit) as exc_info:
            self._run(typer.Exit(1))
        assert exc_info.value.exit_code == 1
        finished = [c.args[0] for c in emit.call_args_list if c.args[0].step == "setup_finished"]
        assert len(finished) == 1
        assert finished[0].task_status == TaskStatusEnum.ERROR

    @patch(EMIT_TARGET)
    def test_real_exception_emits_error(self, emit):
        with pytest.raises(RuntimeError):
            self._run(RuntimeError("boom"))
        finished = [c.args[0] for c in emit.call_args_list if c.args[0].step == "setup_finished"]
        assert len(finished) == 1
        assert finished[0].task_status == TaskStatusEnum.ERROR
