# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: Customizer outputs auto-deployed onto the model-free NIM (NIM 2.x).

A deployment that names the model-free NIM gets the configuration NIM 2.x reads instead of
the NIM 1.x environment variables: the weights path in ``NIM_MODEL_PATH``, tool calling as
``vllm serve`` flags in ``NIM_PASSTHROUGH_ARGS``, and the chat template written to a file.
These tests auto-deploy merged, LoRA and full-weight outputs onto it and check each of those.

The base model comes from Hugging Face and the dataset is generated, so the tests need
outbound network but no S3 assets.
"""

import json
import logging
import os
import time
from typing import Any

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
)
from nhx.testing.e2e.customizer import wait_for_model_spec

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer import stage_assets
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.platform("docker", "kubernetes"),
    pytest.mark.feature("gpu", "automodel"),
    pytest.mark.external_network,
]

MODEL_FREE_NIM_IMAGE = os.environ.get("E2E_MODEL_FREE_NIM_IMAGE", "nvcr.io/nim/nvidia/vllm-model-free-nim")
MODEL_FREE_NIM_TAG = os.environ.get("E2E_MODEL_FREE_NIM_TAG", "latest")
BASE_HF_REPO = "Qwen/Qwen3-0.6B"

# NIM loads LoRA adapters from a folder it polls, so an adapter is routable a short while
# after the base deployment is READY.
ADAPTER_LOAD_TIMEOUT_S = 600

_QUESTION = [{"role": "user", "content": "What is the capital of France?"}]
_NO_THINKING = {"enable_thinking": False}
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_CHAT_TEMPLATE_MARKER = "nhx-model-free-chat-template"
# Raises on every request, so a request error naming the marker proves NIM rendered this
# template rather than the model's own.
_FAILING_CHAT_TEMPLATE = "{{ raise_exception('" + _CHAT_TEMPLATE_MARKER + "') }}"

_QA_ROWS = [
    ("What is the capital of France?", "Paris."),
    ("What is 2 + 2?", "4."),
    ("What color is the sky on a clear day?", "Blue."),
    ("How many days are in a week?", "Seven."),
    ("What is the largest planet in the solar system?", "Jupiter."),
    ("What is the chemical symbol for water?", "H2O."),
    ("How many legs does a spider have?", "Eight."),
    ("What is the opposite of hot?", "Cold."),
]


@pytest.fixture(scope="module")
def model_free_base_entity(client: NemoClient, customizer_workspace: str) -> str:
    """Qwen3-0.6B registered from Hugging Face in the test workspace."""
    name = "qwen3-0-6b-model-free"
    FilesClient.from_client(client).create_fileset(
        workspace=customizer_workspace,
        body=CreateFilesetRequest(
            name=name,
            purpose=FilesetPurpose.MODEL,
            description=f"{BASE_HF_REPO} from Hugging Face",
            storage=HuggingfaceStorageConfig(repo_id=BASE_HF_REPO, repo_type="model"),
        ),
    )
    ModelsClient.from_client(client).create_model(
        workspace=customizer_workspace,
        body=CreateModelEntityRequest(name=name, fileset=f"{customizer_workspace}/{name}"),
    )
    wait_for_model_spec(client, customizer_workspace, name)
    return name


@pytest.fixture(scope="module")
def model_free_dataset(
    client: NemoClient,
    customizer_workspace: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """A small chat dataset; the tests check serving, not what the model learned."""
    local_dir = tmp_path_factory.mktemp("model-free-sft")
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in _QA_ROWS]
    (local_dir / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows * 4))
    (local_dir / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return stage_assets.stage_dataset_fileset(
        client,
        customizer_workspace,
        local_dir=local_dir,
        name_prefix="model-free-sft",
        files={"training.jsonl": "training.jsonl", "validation.jsonl": "validation.jsonl"},
    )


def test_lora_merged_auto_deploys_with_tool_calling(
    client: NemoClient,
    customizer_workspace: str,
    model_free_base_entity: str,
    model_free_dataset: str,
) -> None:
    ws = customizer_workspace
    output = _train(
        client,
        ws,
        model_free_base_entity,
        model_free_dataset,
        finetuning_type="lora_merged",
        deployment_config=_model_free_deployment_params(
            lora_enabled=False,
            tool_call_config={"tool_call_parser": "hermes", "auto_tool_choice": True},
        ),
    )

    deployment, config = _find_auto_deployment(client, ws, output)
    try:
        jobs.wait_until_routable(client, ws, deployment, config)
        model = ceval.base_model_field(ws, output)
        _assert_answers(_chat(client, ws, deployment, model=model, messages=_QUESTION))

        # vLLM rejects tool_choice "auto" unless --enable-auto-tool-choice and --tool-call-parser are
        # set, so a successful response shows the tool calling settings reached the server.
        response = _chat(
            client,
            ws,
            deployment,
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
        )
        logger.info("Tool calling response: %s", response["choices"][0]["message"])

        # tool_choice "auto" may answer in plain text, so force a call to check one comes back.
        response = _chat(
            client,
            ws,
            deployment,
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
        )
        tool_calls = response["choices"][0]["message"].get("tool_calls")
        assert isinstance(tool_calls, list) and tool_calls, response
    finally:
        jobs.delete_deployment(client, ws, deployment, config)


def test_lora_adapter_auto_deploys_on_its_base_model(
    client: NemoClient,
    customizer_workspace: str,
    model_free_base_entity: str,
    model_free_dataset: str,
) -> None:
    ws = customizer_workspace
    adapter = _train(
        client,
        ws,
        model_free_base_entity,
        model_free_dataset,
        finetuning_type="lora",
        deployment_config=_model_free_deployment_params(lora_enabled=True),
    )

    # A LoRA job deploys its base model and serves the adapter from it.
    deployment, config = _find_auto_deployment(client, ws, model_free_base_entity)
    try:
        jobs.wait_until_routable(client, ws, deployment, config)
        _assert_answers(
            _chat(client, ws, deployment, model=ceval.base_model_field(ws, model_free_base_entity), messages=_QUESTION)
        )
        _assert_answers(_chat_when_adapter_loads(client, ws, deployment, ceval.lora_model_field(ws, adapter)))
    finally:
        jobs.delete_deployment(client, ws, deployment, config)


def test_all_weights_auto_deploys(
    client: NemoClient,
    customizer_workspace: str,
    model_free_base_entity: str,
    model_free_dataset: str,
) -> None:
    ws = customizer_workspace
    output = _train(
        client,
        ws,
        model_free_base_entity,
        model_free_dataset,
        finetuning_type="all_weights",
        deployment_config=_model_free_deployment_params(lora_enabled=False),
    )

    deployment, config = _find_auto_deployment(client, ws, output)
    try:
        jobs.wait_until_routable(client, ws, deployment, config)
        _assert_answers(_chat(client, ws, deployment, model=ceval.base_model_field(ws, output), messages=_QUESTION))
    finally:
        jobs.delete_deployment(client, ws, deployment, config)


def test_deployment_chat_template_is_applied(
    client: NemoClient,
    customizer_workspace: str,
    model_free_base_entity: str,
) -> None:
    ws = customizer_workspace
    deployment, config = _deploy_model_free_nim(
        client, ws, model_free_base_entity, chat_template=_FAILING_CHAT_TEMPLATE
    )
    try:
        with pytest.raises(NemoHTTPError) as raised:
            _chat(client, ws, deployment, model=ceval.base_model_field(ws, model_free_base_entity), messages=_QUESTION)
        assert _CHAT_TEMPLATE_MARKER in f"{raised.value} {raised.value.body}", raised.value.body
    finally:
        jobs.delete_deployment(client, ws, deployment, config)


def _model_free_deployment_params(**overrides: Any) -> dict[str, Any]:
    return {
        "gpu": 1,
        "image_name": MODEL_FREE_NIM_IMAGE,
        "image_tag": MODEL_FREE_NIM_TAG,
        **overrides,
    }


def _train(
    client: NemoClient,
    workspace: str,
    base_entity: str,
    dataset: str,
    *,
    finetuning_type: str,
    deployment_config: dict[str, Any],
) -> str:
    """Run a few training steps and return the output model or adapter name."""
    output = unique_name(f"model-free-{finetuning_type}")
    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{workspace}/{base_entity}",
            "dataset": {"training": f"{workspace}/{dataset}"},
            "training": {"training_type": "sft", "finetuning_type": finetuning_type, "max_seq_length": 512},
            "schedule": {"epochs": 1, "max_steps": 5},
            "batch": {"global_batch_size": 4, "micro_batch_size": 1},
            "optimizer": {"learning_rate": 5e-5},
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            "output": {"name": output},
            "deployment_config": deployment_config,
        }
    )
    job_name, final = jobs.submit_and_wait_customization_job(client, "automodel", spec, workspace)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, workspace)
    return output


def _find_auto_deployment(
    client: NemoClient,
    workspace: str,
    model_entity: str,
    *,
    timeout: float = 600,
    poll_interval: float = 15,
) -> tuple[str, str]:
    """Return the deployment and config Customizer created to serve ``model_entity``."""
    models = ModelsClient.from_client(client)
    deadline = time.monotonic() + timeout
    while True:
        for deployment in models.list_deployments(workspace=workspace).items():
            config = models.get_deployment_config(name=deployment.config, workspace=workspace).data()
            if config.model_spec is not None and config.model_spec.model_name == model_entity:
                logger.info("Auto-deployment for %s: %s (config %s)", model_entity, deployment.name, config.name)
                assert config.executor_config is not None
                assert config.executor_config.image_name == MODEL_FREE_NIM_IMAGE
                return deployment.name, config.name
        if time.monotonic() > deadline:
            pytest.fail(f"No auto-deployment for {workspace}/{model_entity} after {timeout:.0f}s")
        time.sleep(poll_interval)


def _deploy_model_free_nim(
    client: NemoClient,
    workspace: str,
    model_entity: str,
    *,
    chat_template: str,
) -> tuple[str, str]:
    """Deploy ``model_entity`` on the model-free NIM and wait until it is routable."""
    config_name = unique_name("model-free-config")
    deployment_name = unique_name("model-free")
    models = ModelsClient.from_client(client)
    models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=config_name,
            engine=Engine.NIM,
            model_spec=ModelDeploymentConfigModelSpec(
                model_name=model_entity,
                model_namespace=workspace,
                chat_template=chat_template,
            ),
            executor_config=ContainerExecutorConfig(
                gpu=1, image_name=MODEL_FREE_NIM_IMAGE, image_tag=MODEL_FREE_NIM_TAG
            ),
        ),
    )
    models.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=deployment_name, config=config_name),
    )
    try:
        jobs.wait_until_routable(client, workspace, deployment_name, config_name)
    except BaseException:
        jobs.delete_deployment(client, workspace, deployment_name, config_name, wait_seconds=0)
        raise
    return deployment_name, config_name


def _chat(client: NemoClient, workspace: str, deployment: str, **body: Any) -> dict[str, Any]:
    response = (
        InferenceGatewayClient.from_client(client)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=deployment,
            workspace=workspace,
            body=JsonBody({"max_tokens": 64, "chat_template_kwargs": _NO_THINKING, **body}),
        )
        .data()
    )
    assert isinstance(response, dict), f"Expected a JSON object, got {type(response)}"
    return response


def _chat_when_adapter_loads(client: NemoClient, workspace: str, deployment: str, adapter_model: str) -> dict[str, Any]:
    deadline = time.monotonic() + ADAPTER_LOAD_TIMEOUT_S
    while True:
        try:
            return _chat(client, workspace, deployment, model=adapter_model, messages=_QUESTION)
        except NemoHTTPError as error:
            if time.monotonic() > deadline:
                raise
            logger.info("Adapter %s not served yet (HTTP %s), retrying", adapter_model, error.status_code)
            time.sleep(15)


def _assert_answers(response: dict[str, Any]) -> None:
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), response
    logger.info("Response: %s", content)
