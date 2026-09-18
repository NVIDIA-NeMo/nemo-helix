# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler tests: public-spec → TrainingStepConfig mapping, executor selection,
and the 4-step PlatformJobSpec shape."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.deployment import DeploymentParams, ToolCallParams
from nemo_platform_plugin.integrations import IntegrationsSpec, MlflowIntegration, WandbIntegration
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.models.types import ModelEntity
from nmp.common.entities.utils import get_random_id
from nmp.customization_common.schemas.values import OutputNameType
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from nmp.rl.app.jobs.compiler import (
    _build_download_config,
    _build_model_entity_config,
    _build_training_step,
    _build_training_step_config,
    platform_job_config_compiler,
)
from nmp.rl.app.jobs.training.schemas import OptimizerType, TrainingType
from nmp.rl.entities.values import FinetuningType
from nmp.rl.schemas import (
    DPOTraining,
    GRPOTraining,
    OutputResponse,
    ParallelismParams,
    RlJobOutput,
)


def _make_model_entity(fileset: str | None = "default/base-model") -> ModelEntity:
    return ModelEntity(
        id=get_random_id("model"),
        workspace="default",
        name="base-model",
        fileset=fileset,
        trust_remote_code=False,
        finetuning_type=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sandbox_capable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster that can run sandboxed Gym, which GRPO compilation requires.

    `_build_grpo_training_step_config` refuses to compile without these, so any
    test that reaches the GRPO branch for some *other* reason has to set them.
    Collected here so that setup is stated once. The negative tests override the
    single value they are about and keep the rest.

    `raising=False` on RL `config` fields: those are read off the module-level
    object, and a test run without the RL service settings loaded may not have
    every attribute present. Platform fields always exist on `NemoPlatformConfig`.
    """
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.sandboxed_gym_default", True, raising=False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_cluster_capable", True)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.job_storage_pvc_claim", "nmp-job-storage", raising=False)


def _make_job_output(
    training: DPOTraining | GRPOTraining | None = None,
    integrations: IntegrationsSpec | None = None,
    *,
    environment: str | None = None,
) -> RlJobOutput:
    training = training or DPOTraining(type="dpo")
    return RlJobOutput(
        model="default/base-model",
        dataset="default/prefs",
        environment=environment,
        training=training,
        integrations=integrations,
        output=OutputResponse(name="my-dpo", type=OutputNameType.MODEL, fileset="my-dpo-fs"),
    )


# Job specs/steps/executors/containers are all TypedDicts (plain dicts at runtime), so
# these take Any rather than a specific TypedDict: the compiler returns them typed as
# PlatformJobStepSpecParam, which is not assignable to dict[str, Any].
def _container(step: Any) -> dict[str, Any]:
    return step["executor"]["container"]


def _provider(step: Any) -> str:
    return step["executor"]["provider"]


def _steps(spec: Any) -> list[Any]:
    """spec["steps"] is typed as an Iterable, so index through a concrete list."""
    return list(spec["steps"])


@pytest.fixture
def platform_clients() -> AsyncCustomizationPlatformClients:
    return AsyncCustomizationPlatformClients(files=AsyncMock(), models=AsyncMock())


# --------------------------------------------------------------------------- #
# _build_training_step_config: public DPOTraining → internal TrainingStepConfig
# --------------------------------------------------------------------------- #


def test_training_step_config_maps_exposed_knobs() -> None:
    t = DPOTraining(
        type="dpo",
        optimizer_type=OptimizerType.ADAM_WITH_FLAT_LR,
        adam_eps=3e-7,
        activation_checkpointing=True,
        keep_top_k=5,
        val_at_end=True,
        ref_policy_kl_penalty=0.2,
        max_grad_norm=2.0,
    )
    sc = _build_training_step_config(_make_job_output(t), trust_remote_code=True)

    # Optimizer knobs.
    assert sc.optimizer.optimizer_type is OptimizerType.ADAM_WITH_FLAT_LR
    assert sc.optimizer.eps == 3e-7
    # Memory / checkpoint / validation knobs.
    assert sc.parallelism.activation_checkpointing is True
    assert sc.schedule.keep_top_k == 5
    assert sc.schedule.val_at_end is True
    # DPO hyperparameters + passthrough.
    assert sc.training.training_type is TrainingType.DPO
    assert sc.training.finetuning_type is FinetuningType.ALL_WEIGHTS
    assert sc.training.dpo is not None
    assert sc.training.dpo.ref_policy_kl_penalty == 0.2
    assert sc.training.dpo.max_grad_norm == 2.0
    assert sc.model.trust_remote_code is True


def test_the_reporting_budget_reaches_the_training_step_config() -> None:
    """From the public DPOTraining to the config the training container reads.

    Every link in the chain defaults, so a dropped one reports at 200 rather than
    failing -- which is exactly the kind of regression nothing else here notices.
    """
    from nmp.customization_common.training.reporting import ProgressReportingConfig

    t = DPOTraining(
        type="dpo", progress_reporting=ProgressReportingConfig(time_series_metrics=["*_loss", "*_accuracy"])
    )
    sc = _build_training_step_config(_make_job_output(t), trust_remote_code=False)

    assert sc.schedule.progress_reporting.time_series_metrics == ["*_loss", "*_accuracy"]


def test_the_reporting_budget_reaches_the_training_step_config_for_grpo(sandbox_capable: None) -> None:
    """The same chain from GRPOTraining, which is a separate branch of the compiler.

    Written because it was broken: ``_build_grpo_training_step_config`` built its
    ``ScheduleConfig`` without ``progress_reporting`` while the DPO branch passed it,
    so the field was accepted and validated on the public spec and then dropped. The
    only symptom was a GRPO run reporting under the backend defaults no matter what
    was asked for -- nothing raised, and the two knobs still reached the compiled
    config because ``ProgressReportingConfig()`` supplies them.

    So this asserts the *value*, not the presence of the key: presence is what
    test_grpo_config's equivalent checks, and presence is exactly what stayed true
    while the wiring was gone.
    """
    from nmp.customization_common.training.reporting import ProgressReportingConfig

    t = GRPOTraining(
        type="grpo",
        progress_reporting=ProgressReportingConfig(
            time_series_metrics=["train_reward"],
            min_report_interval_seconds=30.0,
        ),
    )
    # environment is required for GRPO -- validate_for_training rejects None -- so
    # compiling without it would exercise a spec the service cannot produce, even
    # though _build_training_step_config is reached directly here and would not care.
    sc = _build_training_step_config(_make_job_output(t, environment="default/env"), trust_remote_code=False)

    assert sc.schedule.progress_reporting.time_series_metrics == ["train_reward"]
    assert sc.schedule.progress_reporting.min_report_interval_seconds == 30.0


def test_the_reporting_budget_defaults_when_unstated() -> None:
    sc = _build_training_step_config(_make_job_output(), trust_remote_code=False)

    assert sc.schedule.progress_reporting.time_series_metrics is None


def test_training_step_config_maps_integrations() -> None:
    """job_spec.integrations must reach the step config; otherwise W&B/MLflow are
    silently disabled because the driver's builders read customizer_config.integrations."""
    integrations = IntegrationsSpec(
        wandb=WandbIntegration(
            project="proj", name="run", entity="team", tags=["t1"], notes="n", base_url="https://wandb.example"
        ),
        mlflow=MlflowIntegration(
            experiment_name="exp", name="mlrun", tags={"k": "v"}, description="d", tracking_uri="http://mlflow:5000"
        ),
    )
    sc = _build_training_step_config(_make_job_output(integrations=integrations), trust_remote_code=False)

    assert sc.integrations.wandb is not None
    assert sc.integrations.wandb.project == "proj"
    assert sc.integrations.wandb.name == "run"
    assert sc.integrations.wandb.entity == "team"
    assert sc.integrations.wandb.base_url == "https://wandb.example"

    assert sc.integrations.mlflow is not None
    assert sc.integrations.mlflow.experiment_name == "exp"
    # public MLflow `name` maps to the step config's `run_name`
    assert sc.integrations.mlflow.run_name == "mlrun"
    assert sc.integrations.mlflow.tracking_uri == "http://mlflow:5000"
    assert sc.integrations.mlflow.tags == {"k": "v"}


def test_training_step_config_no_integrations_is_empty() -> None:
    sc = _build_training_step_config(_make_job_output(), trust_remote_code=False)
    assert sc.integrations.wandb is None
    assert sc.integrations.mlflow is None


def test_training_step_config_defaults_match_prior_hardcodes() -> None:
    sc = _build_training_step_config(_make_job_output(), trust_remote_code=False)
    assert sc.optimizer.optimizer_type is None
    assert sc.optimizer.eps == 1e-5
    assert sc.parallelism.activation_checkpointing is False
    assert sc.schedule.keep_top_k == 1
    # val_at_end defaults True → final checkpoint carries val metrics for best-checkpoint selection.
    assert sc.schedule.val_at_end is True


# --------------------------------------------------------------------------- #
# _build_training_step: executor selection by topology
# --------------------------------------------------------------------------- #


def test_single_node_uses_gpu_executor() -> None:
    job = _make_job_output(DPOTraining(type="dpo", parallelism=ParallelismParams(num_nodes=1, num_gpus_per_node=1)))
    step = _build_training_step(job, [], trust_remote_code=False, profile=None)
    assert step["name"] == "dpo-training"
    assert _provider(step) == "gpu"
    assert _container(step)["command"] == ["-m", "nmp.rl.tasks.training"]
    resources = step["executor"]["resources"]
    actual = resources.shm_size if hasattr(resources, "shm_size") else resources["shm_size"]
    assert actual == "8Gi"


def test_multi_node_gpu_shm_scales_with_gpus_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.multinode_shared_storage_path", "/shared", raising=False)
    job = _make_job_output(DPOTraining(type="dpo", parallelism=ParallelismParams(num_nodes=2, num_gpus_per_node=2)))
    step = _build_training_step(job, [], trust_remote_code=False, profile=None)
    resources = step["executor"]["resources"]
    actual = resources.shm_size if hasattr(resources, "shm_size") else resources["shm_size"]
    assert actual == "16Gi"


def test_multi_node_requires_shared_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.multinode_shared_storage_path", None, raising=False)
    job = _make_job_output(DPOTraining(type="dpo", parallelism=ParallelismParams(num_nodes=2, num_gpus_per_node=2)))
    with pytest.raises(PlatformJobCompilationError, match="shared filesystem"):
        _build_training_step(job, [], trust_remote_code=False, profile=None)


def test_multi_node_uses_distributed_executor_with_shared_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.multinode_shared_storage_path", "/shared", raising=False)
    job = _make_job_output(DPOTraining(type="dpo", parallelism=ParallelismParams(num_nodes=2, num_gpus_per_node=2)))
    step = _build_training_step(job, [], trust_remote_code=False, profile=None)
    assert _provider(step) == "gpu_distributed"

    # BASE_LOG_DIR is injected so Ray can coordinate the cross-node barrier.
    def _env_value(env: Any) -> Any:
        return env["value"] if isinstance(env, dict) else getattr(env, "value", None)

    assert any(_env_value(env) == "/shared" for env in step["environment"])


def test_explicit_profile_overrides_default() -> None:
    job = _make_job_output()
    step = _build_training_step(job, [], trust_remote_code=False, profile="custom-gpu")
    assert step["executor"]["profile"] == "custom-gpu"


# --------------------------------------------------------------------------- #
# platform_job_config_compiler: full 4-step spec
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_compiler_emits_four_steps(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    spec = await platform_job_config_compiler("default", _make_job_output(), platform_clients)

    steps = _steps(spec)
    names = [s["name"] for s in steps]
    assert names == ["model-and-dataset-download", "dpo-training", "model-upload", "model-entity-creation"]

    # CPU task steps share the lighter customizer-tasks image; the GPU step uses the training image.
    assert "nmp-customizer-tasks" in _container(steps[0])["image"]
    assert "nmp-rl-training" in _container(steps[1])["image"]
    assert "nmp-customizer-tasks" in _container(steps[2])["image"]
    assert _container(steps[0])["command"] == [
        "-m",
        "nmp.customization_common.tasks.file_io",
        "--service-source",
        "rl",
        "--service-name",
        "rl",
    ]
    assert _container(steps[3])["command"] == [
        "-m",
        "nmp.customization_common.tasks.model_entity",
        "--service-name",
        "rl",
    ]

    upload_meta = steps[2]["config"]["upload"][0]["metadata"]
    assert upload_meta is None


@pytest.mark.asyncio
async def test_compiler_rejects_model_without_fileset(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity(fileset=None)),
    )
    with pytest.raises(PlatformJobCompilationError, match="has no fileset"):
        await platform_job_config_compiler("default", _make_job_output(), platform_clients)


def test_grpo_download_includes_environment() -> None:
    job = _make_job_output(GRPOTraining(type="grpo"), environment="default/my-env")
    cfg = _build_download_config(job, _make_model_entity(), workspace="default")
    dests = [item.dest for item in cfg.download]
    assert "/var/run/scratch/job/environment" in dests
    assert len(cfg.download) == 3


def test_grpo_training_step_config_sandboxed(sandbox_capable: None) -> None:
    sc = _build_training_step_config(
        _make_job_output(GRPOTraining(type="grpo"), environment="default/env"),
        trust_remote_code=False,
    )
    assert sc.training.training_type is TrainingType.GRPO
    assert sc.gym is not None
    assert sc.gym.sandboxed is True
    assert sc.gym.sandbox_environment_path == "/job/environment"
    assert sc.training.grpo is not None
    assert sc.training.grpo.num_generations_per_prompt == 8
    assert sc.training.finetuning_type is FinetuningType.ALL_WEIGHTS
    assert sc.training.lora is None


def test_grpo_lora_training_step_config(sandbox_capable: None) -> None:
    from nmp.rl.schemas import LoRAParams

    job = RlJobOutput(
        model="default/base-model",
        dataset="default/prefs",
        environment="default/env",
        training=GRPOTraining(
            type="grpo",
            finetuning_type="lora",
            lora=LoRAParams(rank=32, alpha=64, use_triton=True),
            parallelism=ParallelismParams(num_nodes=1, num_gpus_per_node=1),
        ),
        output=OutputResponse(name="my-lora", type=OutputNameType.ADAPTER, fileset="my-lora-fs"),
    )
    sc = _build_training_step_config(job, trust_remote_code=False)
    assert sc.training.finetuning_type is FinetuningType.LORA
    assert sc.training.lora is not None
    assert sc.training.lora.rank == 32
    assert sc.training.lora.alpha == 64


def test_grpo_lora_model_entity_peft(sandbox_capable: None) -> None:
    from nmp.rl.app.jobs.compiler import _build_model_entity_config
    from nmp.rl.schemas import LoRAParams

    job = RlJobOutput(
        model="default/base-model",
        dataset="default/prefs",
        environment="default/env",
        training=GRPOTraining(type="grpo", finetuning_type="lora", lora=LoRAParams(rank=8, alpha=16)),
        output=OutputResponse(name="my-lora", type=OutputNameType.ADAPTER, fileset="my-lora-fs"),
    )
    cfg = _build_model_entity_config("default", job, trust_remote_code=False)
    assert cfg.peft is not None
    assert cfg.peft.rank == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.type is FinetuningType.LORA


def test_grpo_compile_succeeds_when_platform_sandbox_capable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.sandboxed_gym_default", True, raising=False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_cluster_capable", True)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.job_storage_pvc_claim", "nmp-job-storage", raising=False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_server_protocol", "http")

    sc = _build_training_step_config(
        _make_job_output(GRPOTraining(type="grpo"), environment="default/env"),
        trust_remote_code=False,
    )
    assert sc.gym is not None
    assert sc.gym.sandboxed is True
    assert sc.gym.sandbox_server_protocol == "http"


def test_grpo_compile_fails_closed_without_sandbox_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.sandboxed_gym_default", True, raising=False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_cluster_capable", False)
    with pytest.raises(PlatformJobCompilationError, match="sandbox_cluster_capable"):
        _build_training_step_config(
            _make_job_output(GRPOTraining(type="grpo"), environment="default/env"),
            trust_remote_code=False,
        )


def test_dpo_compiles_without_sandbox_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sandbox capability gates GRPO only; DPO must compile on a cluster without OpenSandbox.

    DPO runs no Gym environment, so the fail-closed check above must not reach it. If the
    gate ever moves somewhere shared, every DPO job on a sandbox-less cluster stops
    compiling -- and DPO is the path that has no need of a sandbox at all.
    """
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.sandboxed_gym_default", True, raising=False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_cluster_capable", False)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.job_storage_pvc_claim", None, raising=False)

    sc = _build_training_step_config(_make_job_output(), trust_remote_code=False)

    assert sc.training.training_type is TrainingType.DPO
    assert sc.gym is None


def test_grpo_training_step_injects_egress_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.platform_config.sandbox_cluster_capable", True)
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.config.job_storage_pvc_claim", "nmp-job-storage", raising=False)
    job = _make_job_output(GRPOTraining(type="grpo"), environment="default/env")
    step = _build_training_step(job, [], trust_remote_code=False, profile=None)
    assert step["name"] == "grpo-training"
    env_names = {env["name"] for env in step["environment"]}
    assert "NMP_VLLM_SERVICE_HOST" in env_names
    assert "NMP_BROKER_SERVICE_PORT" in env_names


# --------------------------------------------------------------------------- #
# deployment_config: auto-deploy after training
# --------------------------------------------------------------------------- #


def test_model_entity_config_omits_deployment_config_by_default() -> None:
    config = _build_model_entity_config("default", _make_job_output(), trust_remote_code=False)

    assert config.deployment_config is None


def test_model_entity_config_forwards_inline_deployment_config() -> None:
    job = _make_job_output().model_copy(
        update={"deployment_config": DeploymentParams(gpu=2, image_name="img", lora_enabled=True)}
    )
    config = _build_model_entity_config("default", job, trust_remote_code=False)

    assert isinstance(config.deployment_config, DeploymentParams)
    assert config.deployment_config.gpu == 2
    assert config.deployment_config.image_name == "img"
    assert config.deployment_config.lora_enabled is True


def test_model_entity_config_forwards_deployment_config_string_ref() -> None:
    job = _make_job_output().model_copy(update={"deployment_config": "shared/existing-cfg"})
    config = _build_model_entity_config("default", job, trust_remote_code=False)

    assert config.deployment_config == "shared/existing-cfg"


def _make_deployment_config(
    *,
    lora_enabled: bool = True,
    model_entity_id: str = "default/my-dpo",
    model_name: str = "my-dpo",
    model_namespace: str = "default",
) -> Any:
    return SimpleNamespace(
        workspace="default",
        name="existing-cfg",
        model_entity_id=model_entity_id,
        model_spec=SimpleNamespace(
            lora_enabled=lora_enabled,
            model_name=model_name,
            model_namespace=model_namespace,
        ),
    )


def _grpo_lora_job() -> RlJobOutput:
    return RlJobOutput(
        model="default/base-model",
        dataset="default/prefs",
        environment="default/my-env",
        training=GRPOTraining(type="grpo", finetuning_type=FinetuningType.LORA.value),
        output=OutputResponse(name="my-lora", type=OutputNameType.ADAPTER, fileset="my-lora-fs"),
    )


def _not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(status_code=404, request=httpx.Request("GET", "http://test")))


@pytest.fixture
def authorized(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Install an auth client that grants everything, as a live request would."""
    auth_client = AsyncMock()
    auth_client.has_permissions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: auth_client),
    )
    return auth_client


@pytest.mark.asyncio
async def test_inline_deployment_config_compiles_without_an_auth_context(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
) -> None:
    """Only tool_call_plugin is permission-gated; plain params must not demand auth."""
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.auth_client_context", SimpleNamespace(get=lambda: None))
    job = _make_job_output().model_copy(update={"deployment_config": DeploymentParams(gpu=2)})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"]["gpu"] == 2


@pytest.mark.asyncio
async def test_string_deployment_config_compiles_without_an_auth_context(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.auth_client_context", SimpleNamespace(get=lambda: None))
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(data=lambda: _make_deployment_config())
    )
    platform_clients.models.get_model = AsyncMock(
        return_value=SimpleNamespace(data=lambda: SimpleNamespace(workspace="default", name="my-dpo"))
    )
    job = _make_job_output().model_copy(update={"deployment_config": "shared/some-cfg"})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"] == "shared/some-cfg"


@pytest.mark.asyncio
async def test_tool_call_plugin_without_an_auth_context_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    monkeypatch.setattr("nmp.rl.app.jobs.compiler.auth_client_context", SimpleNamespace(get=lambda: None))
    job = _make_job_output().model_copy(
        update={
            "deployment_config": DeploymentParams(tool_call_config=ToolCallParams(tool_call_plugin="default/my-plugin"))
        }
    )

    with pytest.raises(PlatformJobCompilationError, match="No auth context available"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_inline_tool_call_plugin_requires_permission(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    authorized.has_permissions = AsyncMock(return_value=False)
    job = _make_job_output().model_copy(
        update={
            "deployment_config": DeploymentParams(tool_call_config=ToolCallParams(tool_call_plugin="default/my-plugin"))
        }
    )

    with pytest.raises(PlatformJobCompilationError, match="models.tool-call-plugin.set"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_lora_job_rejects_string_ref_without_lora_enabled(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(data=lambda: _make_deployment_config(lora_enabled=False))
    )
    job = _grpo_lora_job().model_copy(update={"deployment_config": "shared/base-cfg"})

    with pytest.raises(PlatformJobCompilationError, match="lora_enabled=false"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_lora_job_accepts_string_ref_with_lora_enabled(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
    sandbox_capable: None,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: _make_deployment_config(
                lora_enabled=True,
                model_entity_id="default/base-model",
                model_name="base-model",
            )
        )
    )
    job = _grpo_lora_job().model_copy(update={"deployment_config": "shared/base-cfg"})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"] == "shared/base-cfg"


@pytest.mark.asyncio
async def test_full_weight_job_accepts_a_config_pointing_at_the_unborn_output_model(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    """A config created before the run, pointing forward at the model it produces."""
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(data=lambda: _make_deployment_config())
    )
    platform_clients.models.get_model = AsyncMock(side_effect=_not_found())
    job = _make_job_output().model_copy(update={"deployment_config": "shared/some-cfg"})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"] == "shared/some-cfg"


@pytest.mark.asyncio
async def test_full_weight_job_rejects_a_config_for_a_different_model_entity(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: _make_deployment_config(
                model_entity_id="default/other", model_name="other", model_namespace="default"
            )
        )
    )
    platform_clients.models.get_model = AsyncMock(side_effect=_not_found())
    job = _make_job_output().model_copy(update={"deployment_config": "shared/some-cfg"})

    with pytest.raises(PlatformJobCompilationError, match="targets a different model entity"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_full_weight_retrain_rejects_a_config_for_a_different_model(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: _make_deployment_config(
                model_entity_id="default/other", model_name="other", model_namespace="default"
            )
        )
    )
    platform_clients.models.get_model = AsyncMock(
        return_value=SimpleNamespace(data=lambda: SimpleNamespace(workspace="default", name="my-dpo"))
    )
    job = _make_job_output().model_copy(update={"deployment_config": "shared/some-cfg"})

    with pytest.raises(PlatformJobCompilationError, match="targets a different model entity"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_full_weight_retrain_accepts_a_config_targeting_the_output_model(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(data=lambda: _make_deployment_config())
    )
    platform_clients.models.get_model = AsyncMock(
        return_value=SimpleNamespace(data=lambda: SimpleNamespace(workspace="default", name="my-dpo"))
    )
    job = _make_job_output().model_copy(update={"deployment_config": "shared/some-cfg"})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"] == "shared/some-cfg"


@pytest.mark.asyncio
async def test_inline_deployment_config_reaches_the_model_entity_step(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    job = _make_job_output().model_copy(update={"deployment_config": DeploymentParams(gpu=4)})

    spec = await platform_job_config_compiler("default", job, platform_clients)

    assert _steps(spec)[3]["config"]["deployment_config"]["gpu"] == 4


@pytest.mark.asyncio
async def test_lora_job_rejects_a_config_for_a_different_base_model(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    """The adapter is served from its base model's deployment, so the config must target it."""
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    platform_clients.models.get_deployment_config = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: _make_deployment_config(
                lora_enabled=True, model_entity_id="default/unrelated", model_name="unrelated"
            )
        )
    )
    job = _grpo_lora_job().model_copy(update={"deployment_config": "shared/other-cfg"})

    with pytest.raises(PlatformJobCompilationError, match="different model entity than the base model"):
        await platform_job_config_compiler("default", job, platform_clients)


@pytest.mark.asyncio
async def test_inline_lora_enabled_false_is_rejected_at_compile(
    monkeypatch: pytest.MonkeyPatch,
    platform_clients: AsyncCustomizationPlatformClients,
    authorized: AsyncMock,
) -> None:
    """RlJobInput rejects this at submit; the compiler takes RlJobOutput, so re-assert it."""
    monkeypatch.setattr(
        "nmp.rl.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_model_entity()),
    )
    job = _grpo_lora_job().model_copy(update={"deployment_config": DeploymentParams(lora_enabled=False)})

    with pytest.raises(PlatformJobCompilationError, match="lora_enabled must be true"):
        await platform_job_config_compiler("default", job, platform_clients)
