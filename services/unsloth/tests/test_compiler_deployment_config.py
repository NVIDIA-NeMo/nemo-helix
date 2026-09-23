# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time validation of ``deployment_config`` on unsloth jobs.

A referenced deployment config is resolved and checked against the job before any
GPU time is spent. Without this, a config naming an unrelated model compiles and
the model_entity task deploys *that* model when training finishes.

The axis that matters most here is unsloth's split between a standalone LoRA
adapter (``save_method="lora"``) and a merged checkpoint
(``save_method="merged_16bit"``): the first is served from its base model's
deployment, the second registers a model entity of its own. The two take opposite
branches, so both are covered for every rule.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.deployment import DeploymentParams, ToolCallParams
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.models.types import ModelEntity
from nhx.unsloth.app.jobs.compiler import platform_job_config_compiler
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobOutput,
)


def _base_model_entity() -> ModelEntity:
    return ModelEntity(
        id="model-base",
        workspace="default",
        name="base",
        fileset="default/base-fileset",
        trust_remote_code=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def _not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(status_code=404, request=httpx.Request("GET", "http://test")))


def _response(data: object) -> MagicMock:
    response = MagicMock()
    response.data.return_value = data
    return response


def _deployment_config(
    *,
    lora_enabled: bool = True,
    model_entity_id: str | None = "default/base",
    model_name: str | None = "base",
    model_namespace: str | None = "default",
) -> Any:
    """A resolved ``ModelDeploymentConfig``.

    Defaults name the base model. Pass ``None`` for both links to get the unbound
    template shape that merged and full-weight training have to use.
    """
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


def _lora_job(deployment_config: str | DeploymentParams | None) -> UnslothJobOutput:
    """A job producing a standalone LoRA adapter, served from the base deployment."""
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule=ScheduleSpec(max_steps=1),
        output=OutputResponse(name="my-adapter", type="adapter", save_method="lora", fileset="my-adapter"),
        deployment_config=deployment_config,
    )


def _merged_job(deployment_config: str | DeploymentParams | None) -> UnslothJobOutput:
    """A LoRA job whose merged save produces a standalone model entity."""
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule=ScheduleSpec(max_steps=1),
        output=OutputResponse(name="my-merged", type="model", save_method="merged_16bit", fileset="my-merged"),
        deployment_config=deployment_config,
    )


def _all_weights_job(deployment_config: str | DeploymentParams | None) -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base", load_in_4bit=False),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(finetuning_type="all_weights"),
        schedule=ScheduleSpec(max_steps=1),
        output=OutputResponse(name="my-model", type="model", save_method="lora", fileset="my-model"),
        deployment_config=deployment_config,
    )


@pytest.fixture
def platform() -> MagicMock:
    """Platform clients with the model entity fetch stubbed; per-test stubs layer on."""
    clients = MagicMock()
    clients.models.get_deployment_config = AsyncMock(side_effect=_not_found())
    clients.models.get_model = AsyncMock(side_effect=_not_found())
    # No adapter holds the output name yet, which is what every test here assumes.
    clients.models.get_adapter = AsyncMock(side_effect=_not_found())
    return clients


@pytest.fixture(autouse=True)
def stub_fetch_model_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_base_model_entity()),
    )


@pytest.fixture
def authorized(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Install an auth client that grants everything, as a live request would."""
    auth_client = AsyncMock()
    auth_client.has_permissions = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: auth_client),
    )
    return auth_client


def _deployment_config_of(spec: Any) -> Any:
    step = next(s for s in spec["steps"] if s["name"] == "model-entity-creation")
    return step["config"]["deployment_config"]


# ---------------------------------------------------------------------------
# No deployment requested
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_deployment_config_resolves_nothing(platform: MagicMock) -> None:
    spec = await platform_job_config_compiler("default", _lora_job(None), platform)

    assert _deployment_config_of(spec) is None
    platform.models.get_deployment_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolvable_string_ref_is_rejected(platform: MagicMock) -> None:
    with pytest.raises(HelixJobCompilationError, match="does not exist in workspace 'default'"):
        await platform_job_config_compiler("default", _lora_job("missing-cfg"), platform)


# ---------------------------------------------------------------------------
# Standalone LoRA adapter: served from the base model's deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lora_job_accepts_a_config_targeting_its_base_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(return_value=_response(_deployment_config()))

    spec = await platform_job_config_compiler("default", _lora_job("existing-cfg"), platform)

    assert _deployment_config_of(spec) == "existing-cfg"


@pytest.mark.asyncio
async def test_lora_job_rejects_a_config_for_a_different_base_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id="default/unrelated", model_name="unrelated"))
    )

    with pytest.raises(HelixJobCompilationError, match="different model entity than the base model"):
        await platform_job_config_compiler("default", _lora_job("other-cfg"), platform)


@pytest.mark.asyncio
async def test_lora_job_rejects_a_referenced_config_without_lora_enabled(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(return_value=_response(_deployment_config(lora_enabled=False)))

    with pytest.raises(HelixJobCompilationError, match="lora_enabled=false"):
        await platform_job_config_compiler("default", _lora_job("existing-cfg"), platform)


@pytest.mark.asyncio
async def test_lora_job_accepts_an_unbound_config(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id=None, model_name=None, model_namespace=None))
    )

    spec = await platform_job_config_compiler("default", _lora_job("template-cfg"), platform)

    assert _deployment_config_of(spec) == "template-cfg"


@pytest.mark.asyncio
async def test_lora_job_rejects_an_unbound_config_without_lora_enabled(platform: MagicMock) -> None:
    """Unbound-ness excuses the model link, not a deployment that cannot load adapters."""
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(
            _deployment_config(lora_enabled=False, model_entity_id=None, model_name=None, model_namespace=None)
        )
    )

    with pytest.raises(HelixJobCompilationError, match="lora_enabled=false"):
        await platform_job_config_compiler("default", _lora_job("template-cfg"), platform)


@pytest.mark.asyncio
async def test_lora_job_rejects_inline_lora_enabled_false(platform: MagicMock) -> None:
    """UnslothJobInput rejects this too, but the compiler takes UnslothJobOutput."""
    job = _lora_job(DeploymentParams(gpu=1, lora_enabled=False))

    with pytest.raises(HelixJobCompilationError, match="lora_enabled must be true"):
        await platform_job_config_compiler("default", job, platform)


# ---------------------------------------------------------------------------
# Merged save: a LoRA *training* that registers a model entity of its own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merged_job_accepts_a_referenced_config_without_lora_enabled(platform: MagicMock) -> None:
    """A merged save folds the adapter into the weights, so no adapter loading is needed.

    This is the branch most at risk of being mapped wrong: merged training sets
    ``finetuning_type="lora"``, but it is not a standalone adapter, so the
    LoRA-specific rules must not apply to it.
    """
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(
            _deployment_config(lora_enabled=False, model_entity_id=None, model_name=None, model_namespace=None)
        )
    )

    spec = await platform_job_config_compiler("default", _merged_job("template-cfg"), platform)

    assert _deployment_config_of(spec) == "template-cfg"


@pytest.mark.asyncio
async def test_merged_job_is_not_checked_against_the_base_model(platform: MagicMock) -> None:
    """A config naming the base model is wrong for a merged job, which makes its own entity."""
    platform.models.get_deployment_config = AsyncMock(return_value=_response(_deployment_config()))
    platform.models.get_model = AsyncMock(side_effect=_not_found())

    with pytest.raises(HelixJobCompilationError, match="targets a different model entity"):
        await platform_job_config_compiler("default", _merged_job("existing-cfg"), platform)


@pytest.mark.asyncio
async def test_merged_job_accepts_a_config_pointing_at_the_unborn_output_model(platform: MagicMock) -> None:
    """A config created before the run, naming forward at the model it will produce."""
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id="default/my-merged", model_name="my-merged"))
    )

    spec = await platform_job_config_compiler("default", _merged_job("forward-cfg"), platform)

    assert _deployment_config_of(spec) == "forward-cfg"
    # The entity's existence is never consulted -- only the config's target.
    platform.models.get_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_weights_job_accepts_a_config_pointing_at_the_unborn_output_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id="default/my-model", model_name="my-model"))
    )

    spec = await platform_job_config_compiler("default", _all_weights_job("forward-cfg"), platform)

    assert _deployment_config_of(spec) == "forward-cfg"


@pytest.mark.asyncio
async def test_merged_job_accepts_an_unbound_config_for_a_new_model_entity(platform: MagicMock) -> None:
    """The output entity does not exist on a first run, so it cannot be named up front."""
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id=None, model_name=None, model_namespace=None))
    )

    spec = await platform_job_config_compiler("default", _merged_job("template-cfg"), platform)

    assert _deployment_config_of(spec) == "template-cfg"
    platform.models.get_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_merged_retrain_accepts_a_config_targeting_the_output_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id="default/my-merged", model_name="my-merged"))
    )
    platform.models.get_model = AsyncMock(
        return_value=_response(SimpleNamespace(workspace="default", name="my-merged"))
    )

    spec = await platform_job_config_compiler("default", _merged_job("existing-cfg"), platform)

    assert _deployment_config_of(spec) == "existing-cfg"


@pytest.mark.asyncio
async def test_merged_retrain_rejects_a_config_for_a_different_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id="default/unrelated", model_name="unrelated"))
    )
    platform.models.get_model = AsyncMock(
        return_value=_response(SimpleNamespace(workspace="default", name="my-merged"))
    )

    with pytest.raises(HelixJobCompilationError, match="targets a different model entity"):
        await platform_job_config_compiler("default", _merged_job("existing-cfg"), platform)


# ---------------------------------------------------------------------------
# Full-weight training
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_weights_job_accepts_an_unbound_config(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(
        return_value=_response(_deployment_config(model_entity_id=None, model_name=None, model_namespace=None))
    )

    spec = await platform_job_config_compiler("default", _all_weights_job("template-cfg"), platform)

    assert _deployment_config_of(spec) == "template-cfg"


@pytest.mark.asyncio
async def test_all_weights_job_rejects_a_config_naming_another_model(platform: MagicMock) -> None:
    platform.models.get_deployment_config = AsyncMock(return_value=_response(_deployment_config()))
    platform.models.get_model = AsyncMock(side_effect=_not_found())

    with pytest.raises(HelixJobCompilationError, match="targets a different model entity"):
        await platform_job_config_compiler("default", _all_weights_job("existing-cfg"), platform)


@pytest.mark.asyncio
async def test_all_weights_job_accepts_inline_params_without_resolving(platform: MagicMock) -> None:
    spec = await platform_job_config_compiler("default", _all_weights_job(DeploymentParams(gpu=2)), platform)

    assert _deployment_config_of(spec)["gpu"] == 2
    platform.models.get_deployment_config.assert_not_awaited()


# ---------------------------------------------------------------------------
# tool_call_plugin: the one deployment field behind a permission
# ---------------------------------------------------------------------------


def _tool_call_job() -> UnslothJobOutput:
    return _lora_job(DeploymentParams(tool_call_config=ToolCallParams(tool_call_plugin="default/my-plugin")))


@pytest.mark.asyncio
async def test_tool_call_plugin_is_allowed_with_the_permission(platform: MagicMock, authorized: AsyncMock) -> None:
    spec = await platform_job_config_compiler("default", _tool_call_job(), platform)

    assert _deployment_config_of(spec)["tool_call_config"]["tool_call_plugin"] == "default/my-plugin"
    authorized.has_permissions.assert_awaited_once_with("default", ["models.tool-call-plugin.set"])


@pytest.mark.asyncio
async def test_tool_call_plugin_without_the_permission_is_rejected(
    platform: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_client = AsyncMock()
    auth_client.has_permissions = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: auth_client),
    )

    with pytest.raises(HelixJobCompilationError, match="models.tool-call-plugin.set"):
        await platform_job_config_compiler("default", _tool_call_job(), platform)


@pytest.mark.asyncio
async def test_tool_call_plugin_without_an_auth_context_is_rejected(
    platform: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: None),
    )

    with pytest.raises(HelixJobCompilationError, match="No auth context available"):
        await platform_job_config_compiler("default", _tool_call_job(), platform)


@pytest.mark.asyncio
async def test_inline_params_without_tool_call_plugin_need_no_auth_context(
    platform: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the gated field consults auth; everything else must compile without it."""
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: None),
    )

    spec = await platform_job_config_compiler("default", _lora_job(DeploymentParams(gpu=2)), platform)

    assert _deployment_config_of(spec)["gpu"] == 2


@pytest.mark.asyncio
async def test_string_ref_needs_no_auth_context(platform: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nhx.unsloth.app.jobs.compiler.auth_client_context",
        SimpleNamespace(get=lambda: None),
    )
    platform.models.get_deployment_config = AsyncMock(return_value=_response(_deployment_config()))

    spec = await platform_job_config_compiler("default", _lora_job("existing-cfg"), platform)

    assert _deployment_config_of(spec) == "existing-cfg"
