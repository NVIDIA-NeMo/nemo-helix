# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unsloth model_entity runner.

Covers:
- Adapter (LoRA) creation
- Full / merged model entity creation
- Update-on-conflict semantics (matches automodel behavior)
- Deployment launch with string-ref and inline DeploymentParams
- Skipping deployment when there's already an active one for a LoRA base
- LoRA on a base shared from another workspace: adapter and deployment stay in the job's workspace
- sanitize_name utility
"""

from __future__ import annotations

import json
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_platform_plugin.deployment import DeploymentParams
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    CreateAdapterRequest,
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


def _expected(prefix: str, model: str, *, template_ws: str = "shared", template: str) -> str:
    """The name the runner should build for a template-derived resource."""
    from nmp.customization_common.tasks.model_entity.run import sanitize_name, template_discriminator

    return sanitize_name(prefix, model, template_discriminator(template_ws, template))


def _resolved_config(
    *,
    workspace: str = "shared",
    name: str = "existing-cfg",
    model_entity_id: str | None = "shared/x",
    model_name: str | None = "x",
    model_namespace: str | None = "shared",
    engine: Engine = Engine.VLLM,
    gpu: int = 4,
) -> types.SimpleNamespace:
    """A ``ModelDeploymentConfig`` as ``get_deployment_config`` returns it.

    Defaults are bound (they name a model); pass ``None`` for both links to get the
    unbound template shape that full-weight training has to use.
    """
    from nemo_platform_plugin.models.types import (
        ContainerExecutorConfig,
        ModelDeploymentConfigModelSpec,
    )

    return types.SimpleNamespace(
        workspace=workspace,
        name=name,
        engine=engine,
        model_entity_id=model_entity_id,
        model_spec=ModelDeploymentConfigModelSpec(
            model_name=model_name,
            model_namespace=model_namespace,
            lora_enabled=True,
        ),
        executor_config=ContainerExecutorConfig(gpu=gpu, image_name="tmpl-img", image_tag="2.0"),
    )


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

    def test_discriminator_scopes_the_name(self) -> None:
        from nmp.customization_common.tasks.model_entity.run import sanitize_name

        assert sanitize_name("sft-cfg", "my-model", "tmpl-a") == "sft-cfg-my-model-tmpl-a"
        assert sanitize_name("sft-cfg", "my-model") == "sft-cfg-my-model"

    def test_template_discriminator_separates_same_name_in_different_workspaces(self) -> None:
        """Configs are keyed by workspace and name, so the name alone is not an identity."""
        from nmp.customization_common.tasks.model_entity.run import sanitize_name, template_discriminator

        a = sanitize_name("sft-cfg", "m", template_discriminator("team-a", "prod"))
        b = sanitize_name("sft-cfg", "m", template_discriminator("team-b", "prod"))
        assert a != b

    def test_template_discriminator_separates_names_sharing_a_prefix(self) -> None:
        """The readable label is capped, so two long names can truncate together.

        The digest is what keeps them apart; without it these collapse to one config.
        """
        from nmp.customization_common.tasks.model_entity.run import (
            MAX_RESOURCE_NAME_LEN,
            sanitize_name,
            template_discriminator,
        )

        a = sanitize_name("sft-cfg", "m", template_discriminator("ws", "production-config-alpha"))
        b = sanitize_name("sft-cfg", "m", template_discriminator("ws", "production-config-beta"))
        assert a != b
        assert len(a) <= MAX_RESOURCE_NAME_LEN
        assert len(b) <= MAX_RESOURCE_NAME_LEN

    def test_template_discriminator_digest_is_never_truncated_away(self) -> None:
        """A long model name must give up room to the discriminator, not the reverse."""
        from nmp.customization_common.tasks.model_entity.run import (
            MAX_RESOURCE_NAME_LEN,
            sanitize_name,
            template_discriminator,
        )

        long_model = "x" * 200
        d_a = template_discriminator("ws", "production-config-alpha")
        d_b = template_discriminator("ws", "production-config-beta")
        a = sanitize_name("sft-cfg", long_model, d_a)
        b = sanitize_name("sft-cfg", long_model, d_b)

        assert a != b
        assert a.endswith(d_a[-8:]) and b.endswith(d_b[-8:])
        assert len(a) <= MAX_RESOURCE_NAME_LEN and len(b) <= MAX_RESOURCE_NAME_LEN

    def test_discriminator_survives_truncation(self) -> None:
        """The separation is only real if the discriminator cannot be truncated away.

        A long model name must give up room to the discriminator, not the reverse --
        otherwise two templates collapse back to one name and silently share a config.
        """
        from nmp.customization_common.tasks.model_entity.run import (
            MAX_RESOURCE_NAME_LEN,
            sanitize_name,
        )

        long_model = "x" * 200
        a = sanitize_name("sft-cfg", long_model, "template-alpha")
        b = sanitize_name("sft-cfg", long_model, "template-beta")

        assert a != b
        assert a.endswith("template-alpha")
        assert b.endswith("template-beta")
        assert len(a) <= MAX_RESOURCE_NAME_LEN
        assert len(b) <= MAX_RESOURCE_NAME_LEN

    def test_overlong_discriminator_is_capped_but_still_distinguishes(self) -> None:
        from nmp.customization_common.tasks.model_entity.run import (
            MAX_RESOURCE_NAME_LEN,
            sanitize_name,
        )

        name = sanitize_name("sft-cfg", "m", "d" * 200)
        assert len(name) <= MAX_RESOURCE_NAME_LEN
        assert name.startswith("sft-cfg-m-d")

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
        models.create_adapter.return_value = _response(_model_entity(name="adapter-x"))

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="adapter-x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter-x"),
            model_entity="default/base-model",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        )

        _result, deploy_target = runner.create_model_entity(config)

        models.create_adapter.assert_called_once()
        create_call = models.create_adapter.call_args
        assert create_call.kwargs["workspace"] == "default"
        body = create_call.kwargs["body"]
        assert isinstance(body, CreateAdapterRequest)
        assert body.model == "default/base-model"
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
        models.create_adapter.side_effect = lambda **_: _raise_runner_conflict()
        models.get_adapter.return_value = _response(types.SimpleNamespace(model="default/base-model"))
        models.update_adapter.return_value = _response(_model_entity(name="adapter-x"))

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="adapter-x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter-x"),
            model_entity="default/base-model",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        )

        runner.create_model_entity(config)

        models.update_adapter.assert_called_once()
        update_call = models.update_adapter.call_args
        assert update_call.kwargs["name"] == "adapter-x"
        assert update_call.kwargs["workspace"] == "default"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateAdapterRequest)
        assert body.fileset == "default/adapter-x"
        assert body.enabled is True

    def test_adapter_conflict_on_another_base_model_is_not_overwritten(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
        from nmp.customization_common.tasks.model_entity.run import ModelEntityCreationError
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(name="base-model"))
        models.create_adapter.side_effect = lambda **_: _raise_runner_conflict()
        models.get_adapter.return_value = _response(types.SimpleNamespace(model="default/other-model"))

        runner = _make_runner(models, files)
        config = ModelEntityTaskConfig(
            name="adapter-x",
            workspace="default",
            fileset=FileSetRef(workspace="default", name="adapter-x"),
            model_entity="default/base-model",
            peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        )

        with pytest.raises(ModelEntityCreationError, match="already exists on base model 'default/other-model'"):
            runner.create_model_entity(config)

        models.update_adapter.assert_not_called()


def _shared_base_lora_config(
    *,
    model_entity: str = "base-model",
    deployment_config: str | DeploymentParams | None = None,
):
    """A LoRA job in ``marcus`` whose base reference only resolves through the global fallback."""
    from nmp.customization_common.schemas.file_io import FileSetRef
    from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
    from nmp.unsloth.entities.values import FinetuningType

    return ModelEntityTaskConfig(
        name="adapter-x",
        workspace="marcus",
        fileset=FileSetRef(workspace=None, name="adapter-x"),
        model_entity=model_entity,
        peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
        deployment_config=deployment_config,
    )


def _marcus_runner(models: ModelsClient, files: FilesClient):
    from nmp.customization_common.tasks.model_entity.run import ModelEntityRunner

    return ModelEntityRunner(models=models, files=files, job_ctx=_make_job_ctx(workspace="marcus"))


class TestSharedBaseAdapter:
    """A base shared in from ``default`` puts the adapter, and any deployment, in the job's workspace."""

    def test_adapter_is_created_in_the_jobs_workspace(self) -> None:
        models, files = _make_clients()
        base_me = _model_entity(workspace="default", name="base-model")
        models.get_model.return_value = _response(base_me)
        models.create_adapter.return_value = _response(_model_entity(workspace="marcus", name="adapter-x"))

        _result, deploy_target = _marcus_runner(models, files).create_model_entity(_shared_base_lora_config())

        # The bare reference was looked up where the job runs and answered from `default`.
        models.get_model.assert_called_once_with(name="base-model", workspace="marcus")
        create_call = models.create_adapter.call_args
        assert create_call.kwargs["workspace"] == "marcus"
        body = create_call.kwargs["body"]
        assert isinstance(body, CreateAdapterRequest)
        assert body.model == "default/base-model"
        assert body.name == "adapter-x"
        assert body.fileset == "marcus/adapter-x"
        assert body.lora_config is not None
        assert (body.lora_config.rank, body.lora_config.alpha) == (8, 16)
        assert body.enabled is True
        assert deploy_target is base_me

    def test_adapter_conflict_updates_the_adapter_in_the_jobs_workspace(self) -> None:
        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(workspace="default", name="base-model"))
        models.create_adapter.side_effect = lambda **_: _raise_runner_conflict()
        models.get_adapter.return_value = _response(types.SimpleNamespace(model="default/base-model"))
        models.update_adapter.return_value = _response(_model_entity(workspace="marcus", name="adapter-x"))

        _marcus_runner(models, files).create_model_entity(_shared_base_lora_config())

        update_call = models.update_adapter.call_args
        assert update_call.kwargs["workspace"] == "marcus"
        assert update_call.kwargs["name"] == "adapter-x"
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateAdapterRequest)
        assert body.fileset == "marcus/adapter-x"

    def test_explicit_reference_to_another_workspace_keeps_the_original_behaviour(self) -> None:
        """Naming ``default/base`` resolved exactly where it pointed, so nothing changes for it."""
        models, files = _make_clients()
        models.get_model.return_value = _response(_model_entity(workspace="default", name="base-model"))
        models.create_adapter.return_value = _response(_model_entity(name="adapter-x"))

        _marcus_runner(models, files).create_model_entity(_shared_base_lora_config(model_entity="default/base-model"))

        create_call = models.create_adapter.call_args
        assert create_call.kwargs["workspace"] == "default"
        assert create_call.kwargs["body"].model == "default/base-model"

    def test_inline_deployment_goes_into_the_jobs_workspace(self) -> None:
        from nemo_platform_plugin.deployment import DeploymentParams

        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([])
        models.create_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="marcus", name="sft-cfg-base-model")
        )
        models.create_deployment.return_value = _response(
            types.SimpleNamespace(workspace="marcus", name="sft-deploy-base-model")
        )
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(
                workspace="marcus", name="sft-deploy-base-model", status=ModelDeploymentStatus.PENDING
            )
        )
        base_me = _model_entity(
            workspace="default",
            name="base-model",
            spec=types.SimpleNamespace(family="llama", base_num_parameters=1_000_000_000),
        )

        _marcus_runner(models, files).launch_model(
            _shared_base_lora_config(deployment_config=DeploymentParams(lora_enabled=True)), base_me
        )

        config_call = models.create_deployment_config.call_args
        assert config_call.kwargs["workspace"] == "marcus"
        # The config lives with the job but still serves the shared base where it is.
        assert config_call.kwargs["body"].model_spec.model_namespace == "default"
        assert config_call.kwargs["body"].model_spec.model_name == "base-model"
        assert models.create_deployment.call_args.kwargs["workspace"] == "marcus"

    def test_active_deployment_is_looked_for_in_the_jobs_workspace(self) -> None:
        from nemo_platform_plugin.deployment import DeploymentParams

        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([types.SimpleNamespace(name="cfg-1")])
        models.list_deployments.return_value = _page([types.SimpleNamespace(status=ModelDeploymentStatus.READY)])

        _marcus_runner(models, files).launch_model(
            _shared_base_lora_config(deployment_config=DeploymentParams(lora_enabled=True)),
            _model_entity(workspace="default", name="base-model"),
        )

        config_call = models.list_deployment_configs.call_args
        assert config_call.kwargs["workspace"] == "marcus"
        assert json.loads(config_call.kwargs["query_params"]["filter"]) == {"model_entity_id": "default/base-model"}
        assert models.list_deployments.call_args.kwargs["workspace"] == "marcus"
        models.create_deployment.assert_not_called()

    def test_bare_deployment_config_reference_resolves_in_the_jobs_workspace(self) -> None:
        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([])
        models.get_deployment_config.return_value = _response(
            _resolved_config(
                workspace="marcus",
                name="my-cfg",
                model_entity_id="default/base-model",
                model_name="base-model",
                model_namespace="default",
            )
        )
        models.create_deployment.return_value = _response(types.SimpleNamespace(workspace="marcus", name="d"))
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(workspace="marcus", name="d", status=ModelDeploymentStatus.PENDING)
        )

        _marcus_runner(models, files).launch_model(
            _shared_base_lora_config(deployment_config="my-cfg"),
            _model_entity(
                workspace="default",
                name="base-model",
                spec=types.SimpleNamespace(family="llama", base_num_parameters=1),
            ),
        )

        models.get_deployment_config.assert_called_once_with(workspace="marcus", name="my-cfg")
        assert models.create_deployment.call_args.kwargs["workspace"] == "marcus"

    def test_unbound_template_is_bound_in_the_jobs_workspace(self) -> None:
        """A template from the job's workspace is bound there, still serving the shared base."""
        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([])
        models.get_deployment_config.return_value = _response(
            _resolved_config(
                workspace="marcus", name="tmpl", model_entity_id=None, model_name=None, model_namespace=None
            )
        )
        models.create_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="marcus", name="sft-cfg-base-model")
        )
        models.create_deployment.return_value = _response(types.SimpleNamespace(workspace="marcus", name="d"))
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(workspace="marcus", name="d", status=ModelDeploymentStatus.PENDING)
        )

        _marcus_runner(models, files).launch_model(
            _shared_base_lora_config(deployment_config="tmpl"),
            _model_entity(
                workspace="default",
                name="base-model",
                spec=types.SimpleNamespace(family="llama", base_num_parameters=1),
            ),
        )

        models.get_deployment_config.assert_called_once_with(workspace="marcus", name="tmpl")
        config_call = models.create_deployment_config.call_args
        assert config_call.kwargs["workspace"] == "marcus"
        assert config_call.kwargs["body"].model_spec.model_namespace == "default"
        assert config_call.kwargs["body"].model_spec.model_name == "base-model"


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
        from nemo_platform_plugin.deployment import DeploymentParams
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

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
            deployment_config=DeploymentParams(gpu=1, image_name="img", image_tag="1.0"),
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
        from nemo_platform_plugin.deployment import DeploymentParams
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

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
            deployment_config=DeploymentParams(gpu=2),
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
        deployment_config = _resolved_config()
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

    def test_unbound_string_ref_binds_the_template_to_the_trained_model(self) -> None:
        """A referenced config naming no model is a template, not something to deploy as-is.

        Deploying it verbatim would leave the deployment with no weights to resolve,
        so the template's engine and executor are copied onto a config for this model.
        """
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        template = _resolved_config(name="template-cfg", model_entity_id=None, model_name=None, model_namespace=None)
        models.get_deployment_config.return_value = _response(template)
        expected_cfg = _expected("sft-cfg", "x", template="template-cfg")
        models.create_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="default", name=expected_cfg)
        )
        models.create_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="sft-deploy-x")
        )
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(
                workspace="default",
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
            deployment_config="shared/template-cfg",
        )

        runner.launch_model(config, me)

        body = models.create_deployment_config.call_args.kwargs["body"]
        assert isinstance(body, CreateModelDeploymentConfigRequest)
        # Scoped to the template as well as the model, so a second template cannot
        # overwrite this config.
        assert body.name == expected_cfg
        assert body.name.startswith("sft-cfg-x-template-cfg-")
        assert body.model_spec.model_name == "x"
        assert body.model_spec.model_namespace == "default"
        # Engine, executor and serving options come from the template, not NIM defaults.
        assert body.engine is Engine.VLLM
        assert body.executor_config.gpu == 4
        assert body.executor_config.image_name == "tmpl-img"
        assert body.model_spec.lora_enabled is True

        # The derived config is deployed; the template is left untouched for reuse.
        models.update_deployment_config.assert_not_called()
        deployment_body = models.create_deployment.call_args.kwargs["body"]
        assert deployment_body.config == expected_cfg
        # The deployment carries the same scope, or the second template would collide
        # here and silently reuse this deployment and its pinned config version.
        assert deployment_body.name == _expected("sft-deploy", "x", template="template-cfg")

    def test_unbound_string_ref_updates_a_derived_config_on_conflict(self) -> None:
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        models.get_deployment_config.return_value = _response(
            _resolved_config(name="template-cfg", model_entity_id=None, model_name=None, model_namespace=None)
        )
        models.create_deployment_config.side_effect = lambda **_: _raise_runner_conflict()
        models.update_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="default", name=_expected("sft-cfg", "x", template="template-cfg"))
        )
        models.create_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="sft-deploy-x")
        )
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(
                workspace="default",
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
            deployment_config="shared/template-cfg",
        )

        runner.launch_model(config, me)

        update_call = models.update_deployment_config.call_args
        assert update_call.kwargs["name"] == _expected("sft-cfg", "x", template="template-cfg")
        body = update_call.kwargs["body"]
        assert isinstance(body, UpdateModelDeploymentConfigRequest)
        assert body.engine is Engine.VLLM
        assert body.model_spec.model_name == "x"
        models.create_deployment.assert_called_once()

    def test_two_templates_for_one_model_do_not_overwrite_each_other(self) -> None:
        """The collision @anubhutivyas caught: reuse is the whole point of a template.

        Two jobs producing the same model entity from different templates used to
        resolve to one ``sft-cfg-<model>``. The second create hit ConflictError and
        updated it to a new version, while the deployment -- also named by model alone
        -- stayed pinned to the version it was created with. The first job's settings
        silently became the second's, and the second's deployment never happened.
        """
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        created: list[str] = []
        deployed: list[tuple[str, str]] = []

        def _create_cfg(**kwargs: Any) -> MagicMock:
            body = kwargs["body"]
            if body.name in created:
                _raise_runner_conflict()
            created.append(body.name)
            return _response(types.SimpleNamespace(workspace="default", name=body.name))

        def _create_dep(**kwargs: Any) -> MagicMock:
            body = kwargs["body"]
            if any(name == body.name for name, _ in deployed):
                _raise_runner_conflict()
            deployed.append((body.name, body.config))
            return _response(types.SimpleNamespace(workspace="default", name=body.name))

        models.create_deployment_config.side_effect = _create_cfg
        models.create_deployment.side_effect = _create_dep
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="d", status=ModelDeploymentStatus.PENDING)
        )

        runner = _make_runner(models, files)
        me = _model_entity(name="mymodel", spec=types.SimpleNamespace(family="llama", base_num_parameters=1))
        for template, gpu, engine in (("tmpl-nim", 1, Engine.NIM), ("tmpl-vllm", 8, Engine.VLLM)):
            models.get_deployment_config.return_value = _response(
                _resolved_config(
                    name=template,
                    model_entity_id=None,
                    model_name=None,
                    model_namespace=None,
                    engine=engine,
                    gpu=gpu,
                )
            )
            runner.launch_model(
                ModelEntityTaskConfig(
                    name="mymodel",
                    workspace="default",
                    fileset=FileSetRef(workspace="default", name="fs"),
                    model_entity="default/base",
                    deployment_config=f"shared/{template}",
                ),
                me,
            )

        assert created == [
            _expected("sft-cfg", "mymodel", template="tmpl-nim"),
            _expected("sft-cfg", "mymodel", template="tmpl-vllm"),
        ]
        assert deployed == [
            (_expected("sft-deploy", "mymodel", template="tmpl-nim"), created[0]),
            (_expected("sft-deploy", "mymodel", template="tmpl-vllm"), created[1]),
        ]
        # Neither job overwrote the other's config.
        models.update_deployment_config.assert_not_called()

    def test_rerun_with_an_edited_template_repoints_the_deployment(self) -> None:
        """Editing a template and re-running must actually change what is served.

        The derived config is updated, which creates a new version, but a deployment
        pins the version it was created with. Reusing the existing deployment on
        conflict would keep serving the old engine and executor settings.
        """
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        models.get_deployment_config.return_value = _response(
            _resolved_config(name="tmpl", model_entity_id=None, model_name=None, model_namespace=None)
        )
        expected_cfg = _expected("sft-cfg", "x", template="tmpl")
        models.create_deployment_config.side_effect = lambda **_: _raise_runner_conflict()
        models.update_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="default", name=expected_cfg)
        )
        models.create_deployment.side_effect = lambda **_: _raise_runner_conflict()
        models.update_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name=_expected("sft-deploy", "x", template="tmpl"))
        )
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="d", status=ModelDeploymentStatus.PENDING)
        )

        runner = _make_runner(models, files)
        me = _model_entity(name="x", spec=types.SimpleNamespace(family="llama", base_num_parameters=1))
        runner.launch_model(
            ModelEntityTaskConfig(
                name="x",
                workspace="default",
                fileset=FileSetRef(workspace="default", name="fs"),
                model_entity="default/base",
                deployment_config="shared/tmpl",
            ),
            me,
        )

        # The deployment is repointed at the freshly updated config, not just fetched.
        update_call = models.update_deployment.call_args
        assert update_call.kwargs["name"] == _expected("sft-deploy", "x", template="tmpl")
        assert update_call.kwargs["body"].config == expected_cfg
        # No explicit version -> the latest, which is the one just written.
        assert update_call.kwargs["body"].config_version is None

    def test_lora_template_does_not_fork_the_base_models_deployment(self) -> None:
        """A LoRA adapter must never stand up a second copy of its base model.

        The adapter is served from the base model's deployment, which is shared by
        every adapter and every template. Scoping the deployment per template would
        put a second copy of the same base model on another GPU -- the waste the
        active-deployment guard exists to prevent, reintroduced by the back door.
        """
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([])  # no active base deployment
        models.get_deployment_config.return_value = _response(
            _resolved_config(name="prod", model_entity_id=None, model_name=None, model_namespace=None)
        )
        models.create_deployment_config.return_value = _response(
            types.SimpleNamespace(workspace="default", name="sft-cfg-llama-base")
        )
        models.create_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="sft-deploy-llama-base")
        )
        models.get_deployment.return_value = _response(
            types.SimpleNamespace(workspace="default", name="d", status=ModelDeploymentStatus.PENDING)
        )

        runner = _make_runner(models, files)
        base = _model_entity(name="llama-base", spec=types.SimpleNamespace(family="llama", base_num_parameters=1))
        runner.launch_model(
            ModelEntityTaskConfig(
                name="my-adapter",
                workspace="default",
                fileset=FileSetRef(workspace="default", name="fs"),
                model_entity="default/llama-base",
                peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
                deployment_config="shared/prod",
            ),
            base,  # the LoRA path hands launch_model the BASE model entity
        )

        # Named for the base model alone -- no template scope. A second LoRA job with a
        # different template lands on this same name and reuses the deployment.
        assert models.create_deployment_config.call_args.kwargs["body"].name == "sft-cfg-llama-base"
        assert models.create_deployment.call_args.kwargs["body"].name == "sft-deploy-llama-base"

    def test_lora_with_an_active_base_deployment_still_skips(self) -> None:
        """The pre-existing deconfliction guard is untouched by template binding."""
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig, PEFTConfig
        from nmp.unsloth.entities.values import FinetuningType

        models, files = _make_clients()
        models.list_deployment_configs.return_value = _page([types.SimpleNamespace(name="base-cfg")])
        models.list_deployments.return_value = _page(
            [types.SimpleNamespace(name="base-deploy", status=ModelDeploymentStatus.READY)]
        )

        runner = _make_runner(models, files)
        base = _model_entity(name="llama-base", spec=types.SimpleNamespace(family="llama", base_num_parameters=1))
        runner.launch_model(
            ModelEntityTaskConfig(
                name="my-adapter",
                workspace="default",
                fileset=FileSetRef(workspace="default", name="fs"),
                model_entity="default/llama-base",
                peft=PEFTConfig(type=FinetuningType.LORA, rank=8, alpha=16),
                deployment_config="shared/prod",
            ),
            base,
        )

        # The live base deployment hot-loads the adapter; nothing new is stood up.
        models.get_deployment_config.assert_not_called()
        models.create_deployment_config.assert_not_called()
        models.create_deployment.assert_not_called()

    def test_bound_string_ref_is_deployed_as_is(self) -> None:
        """A config that already names a model is used verbatim -- nothing to bind."""
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import ModelEntityTaskConfig

        models, files = _make_clients()
        models.get_deployment_config.return_value = _response(
            _resolved_config(model_entity_id=None, model_name="x", model_namespace="shared")
        )
        models.create_deployment.return_value = _response(
            types.SimpleNamespace(workspace="shared", name="sft-deploy-x")
        )
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

        models.create_deployment_config.assert_not_called()
        assert models.create_deployment.call_args.kwargs["body"].config == "existing-cfg"

    def test_lora_with_active_deployment_skips(self) -> None:
        from nemo_platform_plugin.deployment import DeploymentParams
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import (
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
            deployment_config=DeploymentParams(),
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
        from nemo_platform_plugin.deployment import DeploymentParams
        from nmp.customization_common.schemas.file_io import FileSetRef
        from nmp.customization_common.schemas.model_entity import (
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
            deployment_config=DeploymentParams(lora_enabled=False),
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
                platform=MagicMock(),
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

        # A LoRA job's config must target the base model; the compiler now resolves it.
        platform = MagicMock()
        platform.models.get_deployment_config = AsyncMock(
            return_value=_response(
                _resolved_config(model_entity_id="default/base", model_name="base", model_namespace="default")
            )
        )

        original_fetch = compiler_mod.fetch_model_entity
        compiler_mod.fetch_model_entity = AsyncMock(return_value=_compiler_model_entity())
        try:
            job_spec = await platform_job_config_compiler(
                workspace="default",
                job_spec=spec,
                platform=platform,
            )
        finally:
            compiler_mod.fetch_model_entity = original_fetch

        me_step = next(s for s in job_spec["steps"] if s["name"] == "model-entity-creation")
        assert me_step["config"]["deployment_config"] == "my-config"
