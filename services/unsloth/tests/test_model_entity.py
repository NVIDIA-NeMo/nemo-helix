# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unsloth model_entity runner.

Covers:
- Adapter (LoRA) creation
- Full / merged model entity creation
- Update-on-conflict semantics (matches automodel behavior)
- Deployment launch with string-ref and inline DeploymentParameters
- Skipping deployment when there's already an active one for a LoRA base
- sanitize_name utility
"""

from __future__ import annotations

import json
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    Engine,
    ModelDeploymentStatus,
    ModelEntity,
    UpdateAdapterRequest,
    UpdateModelDeploymentConfigRequest,
    UpdateModelEntityRequest,
)


def _make_job_ctx(workspace: str = "default"):
    from nmp.customization_common.service.context import NMPJobContext

    return NMPJobContext(
        workspace=workspace,
        job_id="job-1",
        attempt_id="attempt-0",
        step="model-entity-creation",
        task="task-1",
        jobs_url=None,
        files_url=None,
        storage_path=Path("/tmp"),
        config_path=Path("/tmp/cfg.json"),
    )


def _make_runner(models: ModelsClient, files: FilesClient):
    from nmp.customization_common.tasks.model_entity.run import ModelEntityRunner

    return ModelEntityRunner(models=models, files=files, job_ctx=_make_job_ctx())


def _make_clients() -> tuple[MagicMock, MagicMock]:
    return MagicMock(), MagicMock()


def _response(data: object) -> MagicMock:
    response = MagicMock()
    response.data.return_value = data
    return response


def _page(items: list[object]) -> MagicMock:
    response = MagicMock()
    response.items.return_value = items
    return response


def _raise_runner_conflict() -> None:
    """Raise the ``ConflictError`` class the runner is bound against.

    See test_file_io.py for the rationale; same trick applies here because
    ``tasks/model_entity/__init__.py`` re-exports ``run`` as a function and
    shadows the submodule for plain attribute access.
    """
    import sys

    run_mod = sys.modules["nmp.customization_common.tasks.model_entity.run"]
    raise run_mod.ConflictError.__new__(run_mod.ConflictError, "already exists")


def _model_entity(*, workspace: str = "default", name: str = "base", spec: object | None = None) -> MagicMock:
    me = MagicMock()
    me.workspace = workspace
    me.name = name
    me.trust_remote_code = False
    me.spec = spec
    return me


def _compiler_model_entity(*, workspace: str = "default", name: str = "base") -> ModelEntity:
    return ModelEntity(
        id=f"model-{name}",
        workspace=workspace,
        name=name,
        fileset=f"{workspace}/{name}-fileset",
        trust_remote_code=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_lowercases_and_replaces_invalid_chars(self) -> None:
        from nmp.customization_common.tasks.model_entity.run import sanitize_name

        assert sanitize_name("sft-cfg", "Qwen/Qwen3-0.6B") == "sft-cfg-qwen-qwen3-0.6b"

    def test_collapses_consecutive_hyphens(self) -> None:
        from nmp.customization_common.tasks.model_entity.run import sanitize_name

        # "/" is not in the allowed set, so each "/" becomes "-", then
        # the consecutive-hyphen collapse fires.
        assert sanitize_name("p", "a//b") == "p-a-b"

    def test_caps_length_below_60_and_strips_trailing_hyphen(self) -> None:
        from nmp.customization_common.tasks.model_entity.run import sanitize_name

        # 59-char limit accounts for the "-v1" the backend appends.
        long_name = "a" * 80
        result = sanitize_name("sft-deploy", long_name)
        assert len(result) <= 59
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# ModelEntityRunner.create_model_entity — full / merged path
# ---------------------------------------------------------------------------


class TestCreateFullEntity:
    def test_creates_model_entity_for_full_sft(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(name="base-model"))
        new_me = _model_entity(name="trained-model")
        models.create_model.return_value = _response(new_me)

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="trained-model",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="trained-model"),
            model_entity="default/base-model",
            peft=None,
        )

        result, deploy_target = runner.create_model_entity(config)

        files.get_fileset.assert_called_once_with(workspace="default", name="trained-model")
        models.get_model.assert_called_once_with(name="base-model", workspace="default")
        models.create_model.assert_called_once()
        create_call = models.create_model.call_args
        assert create_call.kwargs["workspace"] == "default"
        body = create_call.kwargs["body"]
        assert isinstance(body, CreateModelEntityRequest)
        assert body.name == "trained-model"
        assert body.fileset == "default/trained-model"
        assert body.base_model is None
        assert body.trust_remote_code is False
        assert deploy_target is new_me
        assert result is not None

    def test_conflict_falls_back_to_update(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(name="base-model"))
        models.create_model.side_effect = lambda **_: _raise_runner_conflict()
        updated_me = _model_entity(name="trained-model")
        models.update_model.return_value = _response(updated_me)

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="trained-model",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="trained-model"),
            model_entity="default/base-model",
            peft=None,
        )

        _, _ = runner.create_model_entity(config)

        models.update_model.assert_called_once()
        update_call = models.update_model.call_args
        assert update_call.kwargs["name"] == "trained-model"
        assert update_call.kwargs["workspace"] == "default"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateModelEntityRequest)
        assert body.fileset == "default/trained-model"
        assert body.base_model is None
        assert body.trust_remote_code is False

    def test_missing_fileset_raises_creation_error(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityCreationError, ModelEntityTaskConfig

        models, files = _make_clients()
        files.get_fileset.side_effect = RuntimeError("fileset missing")
        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="missing"),
            model_entity="default/base-model",
        )

        with pytest.raises(ModelEntityCreationError, match="does not exist or is not accessible"):
            runner.create_model_entity(config)

        models.get_model.assert_not_called()
        models.create_model.assert_not_called()


# ---------------------------------------------------------------------------
# ModelEntityRunner.create_model_entity — LoRA adapter path
# ---------------------------------------------------------------------------


class TestCreateAdapter:
    def test_creates_adapter_for_lora(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        base_me = _model_entity(name="base-model")
        models.get_model.return_value = _response(base_me)
        models.create_model_adapter.return_value = _response(_model_entity(name="adapter-x"))

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="adapter-x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter-x"),
            model_entity="default/base-model",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        )

        _result, deploy_target = runner.create_model_entity(config)

        models.create_model_adapter.assert_called_once()
        create_call = models.create_model_adapter.call_args
        assert create_call.kwargs["model_name"] == "base-model"
        assert create_call.kwargs["workspace"] == "default"
        body = create_call.kwargs["body"]
        assert isinstance(body, CreateModelAdapterRequest)
        assert body.name == "adapter-x"
        assert body.fileset == "default/adapter-x"
        assert body.lora_config is not None
        assert body.lora_config.rank == 8
        assert body.lora_config.alpha == 16
        assert body.enabled is True
        assert deploy_target is base_me

    def test_adapter_conflict_falls_back_to_update(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(name="base-model"))
        models.create_model_adapter.side_effect = lambda **_: _raise_runner_conflict()
        models.update_model_adapter.return_value = _response(_model_entity(name="adapter-x"))

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="adapter-x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter-x"),
            model_entity="default/base-model",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        )

        runner.create_model_entity(config)

        models.update_model_adapter.assert_called_once()
        update_call = models.update_model_adapter.call_args
        assert update_call.kwargs["adapter"] == "adapter-x"
        assert update_call.kwargs["model_name"] == "base-model"
        assert update_call.kwargs["workspace"] == "default"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateAdapterRequest)
        assert body.fileset == "default/adapter-x"
        assert body.enabled is True


# ---------------------------------------------------------------------------
# ModelEntityRunner.launch_model
# ---------------------------------------------------------------------------


class TestLaunchModel:
    def test_no_deployment_config_returns_early(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        runner = _make_runner(models, files)
        me = _model_entity(name="x")
        config = ModelEntityTaskConfig(
            name="x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="x"),
            model_entity="default/base",
            deployment_config=None,
        )

        runner.launch_model(config, me)

        models.create_deployment.assert_not_called()
        models.create_deployment_config.assert_not_called()

    def test_inline_params_creates_config_then_deployment(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import DeploymentParameters, ModelEntityTaskConfig

        models, files = _make_clients()
        deployment_config = types.SimpleNamespace(workspace="other", name="sft-cfg-x")
        deployment = types.SimpleNamespace(workspace="other", name="sft-deploy-x")
        deployment_status = types.SimpleNamespace(
            workspace="other",
            name="sft-deploy-x",
            status=ModelDeploymentStatus.PENDING,
        )
        models.create_deployment_config.return_value = _response(deployment_config)
        models.create_deployment.return_value = _response(deployment)
        models.get_deployment.return_value = _response(deployment_status)

        runner = _make_runner(models, files)
        me = _model_entity(
            workspace="other",
            name="x",
            spec=types.SimpleNamespace(family="llama", base_num_parameters=1_000_000_000),
        )
        config = ModelEntityTaskConfig(
            name="x",
            workspace="other",
            fileset=FileSetRef(workspace="other", name="x"),
            model_entity="other/base",
            deployment_config=DeploymentParameters(gpu=1, image_name="img", image_tag="1.0"),
        )

        runner.launch_model(config, me)

        config_call = models.create_deployment_config.call_args
        assert config_call.kwargs["workspace"] == "other"
        config_body = config_call.kwargs["body"]
        assert isinstance(config_body, CreateModelDeploymentConfigRequest)
        assert config_body.name == "sft-cfg-x"
        assert config_body.engine is Engine.NIM
        assert config_body.model_spec.model_name == "x"
        assert config_body.model_spec.model_namespace == "other"
        assert config_body.executor_config.gpu == 1
        assert config_body.executor_config.image_name == "img"
        assert config_body.executor_config.image_tag == "1.0"

        deployment_call = models.create_deployment.call_args
        assert deployment_call.kwargs["workspace"] == "other"
        deployment_body = deployment_call.kwargs["body"]
        assert isinstance(deployment_body, CreateModelDeploymentRequest)
        assert deployment_body.name == "sft-deploy-x"
        assert deployment_body.config == "sft-cfg-x"
        models.get_deployment.assert_called_once_with(workspace="other", name="sft-deploy-x")

    def test_inline_config_conflict_updates_before_deployment(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import DeploymentParameters, ModelEntityTaskConfig

        models, files = _make_clients()
        models.create_deployment_config.side_effect = lambda **_: _raise_runner_conflict()
        updated_config = types.SimpleNamespace(workspace="default", name="sft-cfg-x")
        deployment = types.SimpleNamespace(workspace="default", name="sft-deploy-x")
        models.update_deployment_config.return_value = _response(updated_config)
        models.create_deployment.return_value = _response(deployment)
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(
                workspace="default",
                name="sft-deploy-x",
                status=ModelDeploymentStatus.PENDING,
            )
        )

        runner = _make_runner(models, files)
        me = _model_entity(
            name="x",
            spec=types.SimpleNamespace(family="llama", base_num_parameters=1_000_000_000),
        )
        config = ModelEntityTaskConfig(
            name="x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="x"),
            model_entity="default/base",
            deployment_config=DeploymentParameters(gpu=2),
        )

        runner.launch_model(config, me)

        update_call = models.update_deployment_config.call_args
        assert update_call.kwargs["workspace"] == "default"
        assert update_call.kwargs["name"] == "sft-cfg-x"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateModelDeploymentConfigRequest)
        assert body.engine is Engine.NIM
        assert body.executor_config.gpu == 2
        models.create_deployment.assert_called_once()

    def test_string_ref_resolves_existing_config(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        deployment_config = types.SimpleNamespace(workspace="shared", name="existing-cfg")
        deployment = types.SimpleNamespace(workspace="shared", name="sft-deploy-x")
        models.get_deployment_config.return_value = _response(deployment_config)
        models.create_deployment.return_value = _response(deployment)
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(
                workspace="shared",
                name="sft-deploy-x",
                status=ModelDeploymentStatus.PENDING,
            )
        )

        runner = _make_runner(models, files)
        me = _model_entity(name="x", spec=types.SimpleNamespace(family="llama", base_num_parameters=1))
        config = ModelEntityTaskConfig(
            name="x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="x"),
            model_entity="default/base",
            deployment_config="shared/existing-cfg",
        )

        runner.launch_model(config, me)

        models.get_deployment_config.assert_called_once_with(workspace="shared", name="existing-cfg")
        models.create_deployment_config.assert_not_called()
        deployment_call = models.create_deployment.call_args
        assert deployment_call.kwargs["workspace"] == "shared"
        assert deployment_call.kwargs["body"].config == "existing-cfg"

    def test_lora_with_active_deployment_skips(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import (
            DeploymentParameters,
            ModelEntityTaskConfig,
            PEFTConfig,
        )
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        existing_config = types.SimpleNamespace(workspace="other", name="cfg-1")
        active_deployment = types.SimpleNamespace(status=ModelDeploymentStatus.READY)
        models.list_deployment_configs.return_value = _page([existing_config])
        models.list_deployments.return_value = _page([active_deployment])

        runner = _make_runner(models, files)
        me = _model_entity(workspace="other", name="base")
        config = ModelEntityTaskConfig(
            name="adapter",
            workspace="other",
            fileset=FileSetRef(workspace="other", name="adapter"),
            model_entity="other/base",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
            deployment_config=DeploymentParameters(),
        )

        runner.launch_model(config, me)

        config_call = models.list_deployment_configs.call_args
        assert config_call.kwargs["workspace"] == "other"
        assert json.loads(config_call.kwargs["query_params"]["filter"]) == {"model_entity_id": "other/base"}
        deployment_call = models.list_deployments.call_args
        assert deployment_call.kwargs["workspace"] == "other"
        assert json.loads(deployment_call.kwargs["query_params"]["filter"]) == {
            "config": "cfg-1",
            "workspace": "other",
        }
        models.create_deployment_config.assert_not_called()
        models.create_deployment.assert_not_called()

    def test_lora_with_lora_enabled_false_warns_and_skips(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import (
            DeploymentParameters,
            ModelEntityTaskConfig,
            PEFTConfig,
        )
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([])

        runner = _make_runner(models, files)
        me = _model_entity(name="base")
        config = ModelEntityTaskConfig(
            name="adapter",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter"),
            model_entity="default/base",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
            deployment_config=DeploymentParameters(lora_enabled=False),
        )

        with caplog.at_level("WARNING"):
            runner.launch_model(config, me)

        assert any("lora_enabled is false" in r.getMessage() for r in caplog.records)
        models.create_deployment.assert_not_called()


# ---------------------------------------------------------------------------
# Compiler → deployment_config plumbing
# ---------------------------------------------------------------------------


class TestCompilerDeploymentConfigPlumbing:
    @pytest.mark.asyncio
    async def test_inline_params_pass_through_to_model_entity_step(self) -> None:
        from unittest.mock import AsyncMock

        from nmp.unsloth.app.jobs.compiler import platform_job_config_compiler
        from nmp.unsloth.schemas import (
            DatasetSpec,
            DeploymentParams,
            LoRAParams,
            ModelLoadSpec,
            OutputResponse,
            ScheduleSpec,
            TrainingSpec,
            UnslothJobOutput,
        )

        spec = UnslothJobOutput(
            model=ModelLoadSpec(name="default/base"),
            dataset=DatasetSpec(path="default/training"),
            training=TrainingSpec(lora=LoRAParams()),
            schedule=ScheduleSpec(max_steps=1),
            output=OutputResponse(name="r", type="adapter", save_method="lora", fileset="r"),
            deployment_config=DeploymentParams(gpu=2, image_name="img", lora_enabled=True),
        )

        # Patch fetch_model_entity to avoid hitting the platform.
        from nmp.unsloth.app.jobs import compiler as compiler_mod

        original_fetch = compiler_mod.fetch_model_entity
        compiler_mod.fetch_model_entity = AsyncMock(return_value=_compiler_model_entity())
        try:
            job_spec = await platform_job_config_compiler(
                workspace="default",
                job_spec=spec,
                sdk=MagicMock(),
            )
        finally:
            compiler_mod.fetch_model_entity = original_fetch

        # PlatformJobSpec is a TypedDict, so we index it instead of using attributes.
        me_step = next(s for s in job_spec["steps"] if s["name"] == "model-entity-creation")
        dc = me_step["config"]["deployment_config"]
        # Inline params come through as a serialized dict, not the user-facing class.
        assert dc["gpu"] == 2
        assert dc["image_name"] == "img"
        assert dc["lora_enabled"] is True

    @pytest.mark.asyncio
    async def test_string_ref_passes_through_unchanged(self) -> None:
        from unittest.mock import AsyncMock

        from nmp.unsloth.app.jobs.compiler import platform_job_config_compiler
        from nmp.unsloth.schemas import (
            DatasetSpec,
            LoRAParams,
            ModelLoadSpec,
            OutputResponse,
            ScheduleSpec,
            TrainingSpec,
            UnslothJobOutput,
        )

        spec = UnslothJobOutput(
            model=ModelLoadSpec(name="default/base"),
            dataset=DatasetSpec(path="default/training"),
            training=TrainingSpec(lora=LoRAParams()),
            schedule=ScheduleSpec(max_steps=1),
            output=OutputResponse(name="r", type="adapter", save_method="lora", fileset="r"),
            deployment_config="my-config",
        )

        from nmp.unsloth.app.jobs import compiler as compiler_mod

        original_fetch = compiler_mod.fetch_model_entity
        compiler_mod.fetch_model_entity = AsyncMock(return_value=_compiler_model_entity())
        try:
            job_spec = await platform_job_config_compiler(
                workspace="default",
                job_spec=spec,
                sdk=MagicMock(),
            )
        finally:
            compiler_mod.fetch_model_entity = original_fetch

        me_step = next(s for s in job_spec["steps"] if s["name"] == "model-entity-creation")
        assert me_step["config"]["deployment_config"] == "my-config"
