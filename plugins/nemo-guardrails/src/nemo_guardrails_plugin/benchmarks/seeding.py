# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed NMP with resources required by the IGW guardrails benchmark."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_guardrails_plugin.benchmarks.constants import (
    APP_MODEL_NAME,
    APP_PROVIDER,
    APP_PROVIDER_URL,
    CS_MODEL_NAME,
    CS_PROVIDER,
    CS_PROVIDER_URL,
    GUARDRAIL_CONFIG,
    GUARDRAILS_MIDDLEWARE_CONFIG_TYPE,
    GUARDRAILS_MIDDLEWARE_NAME,
    NO_GUARDRAILS_VM_NAME,
    VM_NAME,
    WORKSPACE,
)
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.guardrail.client import GuardrailClient
from nemo_platform_plugin.guardrail.types import CreateGuardrailConfigRequest
from nemo_platform_plugin.inference_middleware import BackendFormat
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import CreateModelProviderRequest, ModelProvider
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

log = logging.getLogger(__name__)

_PROVIDER_WAIT_TIMEOUT_SECONDS = 60
_PROVIDER_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class SeededResources:
    workspace: str
    app_provider_name: str
    cs_provider_name: str
    app_model_entity: str
    cs_model_entity: str
    guardrail_config_name: str
    vm_name: str
    # Control VM with no middleware; otherwise identical to the guardrails VM.
    no_guardrails_vm_name: str

    @property
    def vm_ref(self) -> str:
        return f"{self.workspace}/{self.vm_name}"

    @property
    def no_guardrails_vm_ref(self) -> str:
        return f"{self.workspace}/{self.no_guardrails_vm_name}"

    @property
    def guardrail_config_ref(self) -> str:
        return f"{self.workspace}/{self.guardrail_config_name}"


def seed_benchmark(
    sdk: NeMoPlatform,
    *,
    nemoguardrails_repo_root: Path,
    generated_dir: Path,
    provider_wait_timeout: float = _PROVIDER_WAIT_TIMEOUT_SECONDS,
) -> SeededResources:
    """Create workspace, providers, GuardrailConfig, and VirtualModel.

    All ``create`` calls are idempotent (``exist_ok=True``) so this is safe to
    rerun against a reused NMP instance.
    """
    generated_dir.mkdir(parents=True, exist_ok=True)
    workspaces_client = client_from_platform(sdk, WorkspacesClient)
    models_client = client_from_platform(sdk, ModelsClient)
    guardrail_client = client_from_platform(sdk, GuardrailClient)
    virtual_models_client = client_from_platform(sdk, VirtualModelsClient)

    log.info("Creating workspace %s", WORKSPACE)
    workspaces_client.create_workspace(
        exist_ok=True,
        body=CreateWorkspaceRequest(name=WORKSPACE, description="Local IGW guardrails benchmark workspace"),
    ).data()

    log.info("Registering app mock provider %s", APP_PROVIDER)
    models_client.create_provider(
        workspace=WORKSPACE,
        body=CreateModelProviderRequest(
            name=APP_PROVIDER,
            host_url=APP_PROVIDER_URL,
            enabled_models=[APP_MODEL_NAME],
            description=f"Benchmark mock app LLM on {APP_PROVIDER_URL}",
        ),
        exist_ok=True,
    )

    log.info("Registering content-safety mock provider %s", CS_PROVIDER)
    models_client.create_provider(
        workspace=WORKSPACE,
        body=CreateModelProviderRequest(
            name=CS_PROVIDER,
            host_url=CS_PROVIDER_URL,
            enabled_models=[CS_MODEL_NAME],
            description=f"Benchmark mock content-safety LLM on {CS_PROVIDER_URL}",
        ),
        exist_ok=True,
    )

    log.info("Waiting for provider discovery")
    app_provider = _wait_for_served_model(
        models_client,
        provider_name=APP_PROVIDER,
        served_model_name=APP_MODEL_NAME,
        timeout_seconds=provider_wait_timeout,
    )
    cs_provider = _wait_for_served_model(
        models_client,
        provider_name=CS_PROVIDER,
        served_model_name=CS_MODEL_NAME,
        timeout_seconds=provider_wait_timeout,
    )

    app_entity = _extract_model_entity(app_provider, APP_MODEL_NAME, provider_name=APP_PROVIDER)
    cs_entity = _extract_model_entity(cs_provider, CS_MODEL_NAME, provider_name=CS_PROVIDER)

    _dump_model(generated_dir / "app_provider.json", app_provider)
    _dump_model(generated_dir / "content_safety_provider.json", cs_provider)

    log.info("Building GuardrailConfig payload from %s", nemoguardrails_repo_root)
    config_data = build_guardrail_config_data(
        source_config_dir=nemoguardrails_repo_root / "examples" / "configs" / "content_safety_local",
        content_safety_model_entity=cs_entity,
    )
    # Persist the same payload shape the old shell harness produced for debuggability.
    (generated_dir / "content_safety_local_nmp_request.json").write_text(
        json.dumps(
            {
                "name": GUARDRAIL_CONFIG,
                "description": "Benchmark content_safety_local config routed through IGW",
                "data": config_data,
                "exist_ok": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("Creating GuardrailConfig %s", GUARDRAIL_CONFIG)
    guardrail_client.create_guardrail_config(
        workspace=WORKSPACE,
        body=CreateGuardrailConfigRequest(
            name=GUARDRAIL_CONFIG,
            description="Benchmark content_safety_local config routed through IGW",
            data=config_data,
        ),
        exist_ok=True,
    )

    middleware_call = MiddlewareCall(
        name=GUARDRAILS_MIDDLEWARE_NAME,
        config_type=GUARDRAILS_MIDDLEWARE_CONFIG_TYPE,
        config_id=f"{WORKSPACE}/{GUARDRAIL_CONFIG}",
    )
    vm_models = [VirtualModelInferenceConfig(model=app_entity, backend_format=BackendFormat.OPENAI_CHAT)]

    log.info("Creating VirtualModel %s/%s", WORKSPACE, VM_NAME)
    vm = _create_virtual_model_exist_ok(
        virtual_models_client,
        workspace=WORKSPACE,
        body=CreateVirtualModelRequest(
            name=VM_NAME,
            default_model_entity=app_entity,
            models=vm_models,
            request_middleware=[middleware_call],
            response_middleware=[middleware_call],
        ),
    )
    _dump_model(generated_dir / "virtual_model.json", vm)

    # Control VM: identical to the guardrails VM but no middleware, so the
    # with-vs-without delta isolates middleware overhead.
    log.info("Creating control VirtualModel %s/%s", WORKSPACE, NO_GUARDRAILS_VM_NAME)
    no_guardrails_vm = _create_virtual_model_exist_ok(
        virtual_models_client,
        workspace=WORKSPACE,
        body=CreateVirtualModelRequest(
            name=NO_GUARDRAILS_VM_NAME,
            default_model_entity=app_entity,
            models=vm_models,
            request_middleware=[],
            response_middleware=[],
        ),
    )
    _dump_model(generated_dir / "virtual_model_no_guardrails.json", no_guardrails_vm)

    return SeededResources(
        workspace=WORKSPACE,
        app_provider_name=APP_PROVIDER,
        cs_provider_name=CS_PROVIDER,
        app_model_entity=app_entity,
        cs_model_entity=cs_entity,
        guardrail_config_name=GUARDRAIL_CONFIG,
        vm_name=VM_NAME,
        no_guardrails_vm_name=NO_GUARDRAILS_VM_NAME,
    )


def _wait_for_served_model(
    client: ModelsClient,
    *,
    provider_name: str,
    served_model_name: str,
    timeout_seconds: float,
) -> ModelProvider:
    """Poll a provider until ``served_models`` lists the expected entry.

    Gateway readiness alone is not enough: VirtualModel creation needs the
    discovered ``model_entity_id``, which only appears once the provider has
    enumerated its models.
    """
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            provider = client.get_provider(name=provider_name, workspace=WORKSPACE).data()
        except NotFoundError as exc:
            last_error = exc
            time.sleep(_PROVIDER_POLL_INTERVAL_SECONDS)
            continue
        served_models = provider.served_models or []
        for served_model in served_models:
            if served_model.served_model_name == served_model_name and served_model.model_entity_id:
                return provider
        time.sleep(_PROVIDER_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Provider {provider_name!r} did not surface served model "
        f"{served_model_name!r} within {timeout_seconds}s: {last_error}"
    )


def _create_virtual_model_exist_ok(
    client: VirtualModelsClient,
    *,
    workspace: str,
    body: CreateVirtualModelRequest,
) -> VirtualModel:
    try:
        return client.create_virtual_model(workspace=workspace, body=body).data()
    except ConflictError:
        return client.get_virtual_model(workspace=workspace, name=body.name).data()


def _extract_model_entity(provider: ModelProvider, served_model_name: str, *, provider_name: str) -> str:
    for served_model in provider.served_models or []:
        if served_model.served_model_name == served_model_name and served_model.model_entity_id:
            return served_model.model_entity_id
    raise RuntimeError(f"Provider {provider_name!r} does not expose served model {served_model_name!r}")


def build_guardrail_config_data(
    *,
    source_config_dir: Path,
    content_safety_model_entity: str,
) -> dict[str, Any]:
    """Read the upstream content_safety_local config and rewrite it for NMP.

    The upstream config references an HTTP base_url; in NMP we instead route by
    ``model_entity_id`` resolved via the inference gateway. Prompts are inlined
    from the sibling ``prompts.yml`` file.
    """
    config_yaml = source_config_dir / "config.yml"
    prompts_yaml = source_config_dir / "prompts.yml"
    if not config_yaml.is_file():
        raise FileNotFoundError(f"Expected guardrails config at {config_yaml}")
    if not prompts_yaml.is_file():
        raise FileNotFoundError(f"Expected guardrails prompts at {prompts_yaml}")

    config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping at {config_yaml}, got {type(config).__name__}")

    prompts = yaml.safe_load(prompts_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(prompts, dict):
        raise ValueError(f"Expected a YAML mapping at {prompts_yaml}, got {type(prompts).__name__}")

    config["models"] = [
        {
            "type": "content_safety",
            "engine": "nim",
            "model": content_safety_model_entity,
        }
    ]
    config["prompts"] = prompts.get("prompts", [])
    return config


def _dump_model(path: Path, model: Any) -> None:
    """Best-effort serialize an SDK response model to JSON."""
    if hasattr(model, "model_dump"):
        payload = model.model_dump(mode="json")
    elif hasattr(model, "to_dict"):
        payload = model.to_dict()
    else:
        payload = json.loads(json.dumps(model, default=str))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
