# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar
from unittest.mock import MagicMock

import pytest
import yaml
from nemo_guardrails_plugin.benchmarks.constants import (
    APP_MODEL_NAME,
    APP_PROVIDER,
    CS_MODEL_NAME,
    CS_PROVIDER,
    GUARDRAIL_CONFIG,
    NO_GUARDRAILS_VM_NAME,
    VM_NAME,
    WORKSPACE,
)
from nemo_guardrails_plugin.benchmarks.seeding import (
    build_guardrail_config_data,
    seed_benchmark,
)
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.guardrail.client import GuardrailClient
from nemo_platform_plugin.guardrail.types import CreateGuardrailConfigRequest
from nemo_platform_plugin.inference_middleware import BackendFormat
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import CreateModelProviderRequest, ModelProvider, ServedModelMapping
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ResponseT = TypeVar("ResponseT")
_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


class _ClientResponse(Generic[ResponseT]):
    def __init__(self, value: ResponseT) -> None:
        self._value = value

    def data(self) -> ResponseT:
        return self._value


def _make_provider(*, provider_name: str, served_model_name: str, entity_suffix: str = "entity") -> ModelProvider:
    return ModelProvider(
        id=f"provider-{provider_name}",
        name=provider_name,
        workspace=WORKSPACE,
        host_url="http://test-provider:8000",
        served_models=[
            ServedModelMapping(
                served_model_name=served_model_name,
                model_entity_id=f"{WORKSPACE}/{served_model_name.replace('/', '-')}-{entity_suffix}",
            )
        ],
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
    )


def _make_virtual_model(*, workspace: str, body: CreateVirtualModelRequest) -> VirtualModel:
    return VirtualModel(
        name=body.name,
        workspace=workspace,
        default_model_entity=body.default_model_entity,
        models=body.models,
        request_middleware=body.request_middleware,
        response_middleware=body.response_middleware,
        post_response_middleware=body.post_response_middleware,
        override_proxy=body.override_proxy,
    )


def _write_upstream_configs(ng_root: Path) -> Path:
    cs_dir = ng_root / "examples" / "configs" / "content_safety_local"
    cs_dir.mkdir(parents=True)
    (cs_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "type": "main",
                        "engine": "nim",
                        "model": "meta/llama-3.3-70b-instruct",
                        "parameters": {"base_url": "http://localhost:8000"},
                    },
                ],
                "rails": {"input": {"flows": ["content safety check input $model=content_safety"]}},
            }
        ),
        encoding="utf-8",
    )
    (cs_dir / "prompts.yml").write_text(
        yaml.safe_dump({"prompts": [{"task": "content_safety_check_input", "content": "..."}]}),
        encoding="utf-8",
    )
    return cs_dir


def _response(data: ResponseT) -> _ClientResponse[ResponseT]:
    return _ClientResponse(data)


@pytest.fixture
def sdk() -> NeMoPlatform:
    return NeMoPlatform(base_url="http://test:8000")


@pytest.fixture
def typed_client_mocks(monkeypatch) -> SimpleNamespace:
    mocks = SimpleNamespace(
        create_workspace=MagicMock(return_value=_response(SimpleNamespace(name=WORKSPACE))),
        create_provider=MagicMock(),
        get_provider=MagicMock(
            side_effect=lambda *, name, workspace=None: _response(
                _make_provider(
                    provider_name=name,
                    served_model_name=APP_MODEL_NAME if name == APP_PROVIDER else CS_MODEL_NAME,
                )
            )
        ),
        create_guardrail_config=MagicMock(return_value=_response(SimpleNamespace(name=GUARDRAIL_CONFIG))),
        create_virtual_model=MagicMock(
            side_effect=lambda *, workspace, body: _response(_make_virtual_model(workspace=workspace, body=body))
        ),
        get_virtual_model=MagicMock(),
    )
    monkeypatch.setattr(WorkspacesClient, "create_workspace", mocks.create_workspace)
    monkeypatch.setattr(ModelsClient, "create_provider", mocks.create_provider)
    monkeypatch.setattr(ModelsClient, "get_provider", mocks.get_provider)
    monkeypatch.setattr(GuardrailClient, "create_guardrail_config", mocks.create_guardrail_config)
    monkeypatch.setattr(VirtualModelsClient, "create_virtual_model", mocks.create_virtual_model)
    monkeypatch.setattr(VirtualModelsClient, "get_virtual_model", mocks.get_virtual_model)
    return mocks


# ---------------------------------------------------------------------------
# build_guardrail_config_data
# ---------------------------------------------------------------------------


class TestBuildGuardrailConfigData:
    def test_rewrites_models_and_inlines_prompts(self, tmp_path: Path) -> None:
        cs_dir = _write_upstream_configs(tmp_path)

        data = build_guardrail_config_data(
            source_config_dir=cs_dir,
            content_safety_model_entity="benchmark/cs-entity",
        )

        assert data["models"] == [{"type": "content_safety", "engine": "nim", "model": "benchmark/cs-entity"}]
        assert data["prompts"] == [{"task": "content_safety_check_input", "content": "..."}]
        # Non-models fields preserved.
        assert data["rails"]["input"]["flows"] == ["content safety check input $model=content_safety"]

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config.yml"):
            build_guardrail_config_data(
                source_config_dir=tmp_path / "nope",
                content_safety_model_entity="x/y",
            )


# ---------------------------------------------------------------------------
# seed_benchmark
# ---------------------------------------------------------------------------


class TestSeedBenchmark:
    def test_calls_sdk_with_expected_payloads(
        self, sdk: NeMoPlatform, typed_client_mocks: SimpleNamespace, tmp_path: Path
    ) -> None:
        ng_root = tmp_path / "NeMo-Guardrails"
        _write_upstream_configs(ng_root)
        generated_dir = tmp_path / "generated"

        seeded = seed_benchmark(
            sdk,
            nemoguardrails_repo_root=ng_root,
            generated_dir=generated_dir,
            provider_wait_timeout=1.0,
        )

        typed_client_mocks.create_workspace.assert_called_once_with(
            exist_ok=True,
            body=CreateWorkspaceRequest(
                name=WORKSPACE,
                description="Local IGW guardrails benchmark workspace",
            ),
        )
        # Both providers registered.
        provider_create_names = [c.kwargs["body"].name for c in typed_client_mocks.create_provider.call_args_list]
        assert sorted(provider_create_names) == sorted([APP_PROVIDER, CS_PROVIDER])
        for call in typed_client_mocks.create_provider.call_args_list:
            assert call.kwargs["workspace"] == WORKSPACE
            assert isinstance(call.kwargs["body"], CreateModelProviderRequest)
            assert call.kwargs["exist_ok"] is True

        # Guardrail config payload uses the discovered content-safety entity.
        gc_call = typed_client_mocks.create_guardrail_config.call_args
        assert gc_call.kwargs["workspace"] == WORKSPACE
        assert isinstance(gc_call.kwargs["body"], CreateGuardrailConfigRequest)
        assert gc_call.kwargs["body"].name == GUARDRAIL_CONFIG
        assert gc_call.kwargs["exist_ok"] is True
        cs_entity = seeded.cs_model_entity
        assert gc_call.kwargs["body"].data.models[0].model == cs_entity

        # Two VirtualModels are created: the guardrails VM (with middleware) and
        # a control VM (no middleware) used by the without-guardrails benchmark
        # variant.
        vm_calls = typed_client_mocks.create_virtual_model.call_args_list
        assert len(vm_calls) == 2

        guardrails_vm_call = vm_calls[0]
        guardrails_body = guardrails_vm_call.kwargs["body"]
        assert isinstance(guardrails_body, CreateVirtualModelRequest)
        assert guardrails_body.name == VM_NAME
        assert guardrails_body.default_model_entity == seeded.app_model_entity
        assert guardrails_body.models == [
            VirtualModelInferenceConfig(model=seeded.app_model_entity, backend_format=BackendFormat.OPENAI_CHAT)
        ]
        expected_middleware = [
            MiddlewareCall(
                name="nemo-guardrails",
                config_type="guardrail_config",
                config_id=f"{WORKSPACE}/{GUARDRAIL_CONFIG}",
            )
        ]
        assert guardrails_body.request_middleware == expected_middleware
        assert guardrails_body.response_middleware == expected_middleware

        control_vm_call = vm_calls[1]
        control_body = control_vm_call.kwargs["body"]
        assert control_body.name == NO_GUARDRAILS_VM_NAME
        assert control_body.default_model_entity == seeded.app_model_entity
        assert control_body.request_middleware == []
        assert control_body.response_middleware == []

    def test_generated_dir_contains_artifacts(
        self, sdk: NeMoPlatform, typed_client_mocks: SimpleNamespace, tmp_path: Path
    ) -> None:
        ng_root = tmp_path / "NeMo-Guardrails"
        _write_upstream_configs(ng_root)
        generated_dir = tmp_path / "generated"

        seed_benchmark(
            sdk,
            nemoguardrails_repo_root=ng_root,
            generated_dir=generated_dir,
            provider_wait_timeout=1.0,
        )

        assert (generated_dir / "app_provider.json").is_file()
        assert (generated_dir / "content_safety_provider.json").is_file()
        assert (generated_dir / "virtual_model.json").is_file()
        assert (generated_dir / "virtual_model_no_guardrails.json").is_file()

        request_payload = json.loads(
            (generated_dir / "content_safety_local_nmp_request.json").read_text(encoding="utf-8")
        )
        assert request_payload["name"] == GUARDRAIL_CONFIG
        assert request_payload["exist_ok"] is True
        assert request_payload["data"]["models"][0]["type"] == "content_safety"

    def test_returns_seeded_resources(
        self, sdk: NeMoPlatform, typed_client_mocks: SimpleNamespace, tmp_path: Path
    ) -> None:
        ng_root = tmp_path / "NeMo-Guardrails"
        _write_upstream_configs(ng_root)

        seeded = seed_benchmark(
            sdk,
            nemoguardrails_repo_root=ng_root,
            generated_dir=tmp_path / "generated",
            provider_wait_timeout=1.0,
        )

        assert seeded.workspace == WORKSPACE
        assert seeded.vm_ref == f"{WORKSPACE}/{VM_NAME}"
        assert seeded.no_guardrails_vm_name == NO_GUARDRAILS_VM_NAME
        assert seeded.guardrail_config_ref == f"{WORKSPACE}/{GUARDRAIL_CONFIG}"

    def test_raises_if_served_models_never_populated(
        self, sdk: NeMoPlatform, typed_client_mocks: SimpleNamespace, tmp_path: Path
    ) -> None:
        ng_root = tmp_path / "NeMo-Guardrails"
        _write_upstream_configs(ng_root)

        typed_client_mocks.get_provider.return_value = _response(
            ModelProvider(
                id="provider-empty",
                name=APP_PROVIDER,
                workspace=WORKSPACE,
                host_url="http://test-provider:8000",
                served_models=[],
                created_at=_TIMESTAMP,
                updated_at=_TIMESTAMP,
            )
        )
        typed_client_mocks.get_provider.side_effect = None

        with pytest.raises(TimeoutError, match="served model"):
            seed_benchmark(
                sdk,
                nemoguardrails_repo_root=ng_root,
                generated_dir=tmp_path / "generated",
                provider_wait_timeout=0.1,
            )
