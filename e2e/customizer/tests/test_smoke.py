# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke: each customization backend trains a few steps on a small dataset, then serves its output.

Every job passes an unbound vLLM template as its ``deployment_config``, so the job's own
model_entity step deploys the output. The test waits for that deployment, sends one request
through the inference gateway, and deletes it. Automodel SFT outputs are then also served on the
model-free NIM (NIM 2.x). Smoke checks that the path works, not quality.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.models.types import ToolCallConfig
from nemo_unsloth_plugin.schema import UnslothJobInput
from pydantic import BaseModel

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import assert_output_registered, unique_name

logger = logging.getLogger(__name__)

MAX_STEPS = 5
GRPO_MAX_STEPS = 3
JOB_TIMEOUT_SECONDS = 2700

ServeCheck = Callable[[NemoClient, str, str, str], None]

# NIM loads LoRA adapters from a folder it polls, so an adapter is routable a short while
# after the base deployment is READY.
ADAPTER_LOAD_TIMEOUT_S = 600
# A ChatML template behind a fixed preamble of this many words. Qwen's own template renders the
# question in a few dozen prompt tokens and this one in at least this many, so `prompt_tokens`
# shows the model-free NIM applied the deployment's template while the deployment still answers.
# The empty think block is what Qwen3's template emits for enable_thinking=False.
CHAT_TEMPLATE_PREAMBLE_WORDS = 256
PREAMBLE_CHAT_TEMPLATE = (
    "{% for _ in range(" + str(CHAT_TEMPLATE_PREAMBLE_WORDS) + ") %} nhx{% endfor %}\n"
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
WEATHER_QUESTION = [{"role": "user", "content": "What is the weather in Paris right now?"}]


def _rl_job_input() -> type[BaseModel]:
    """``RlJobInput``, skipping only the calling RL test when the RL plugin is not installed."""
    return pytest.importorskip("nemo_rl_plugin.schema").RlJobInput


def _train_and_serve(
    client: NemoClient,
    workspace: str,
    backend: str,
    spec: BaseModel,
    *,
    base_entity: str,
    output: str,
    lora: bool,
    check: ServeCheck = jobs.assert_chat_completion,
) -> None:
    job_name, final = jobs.submit_and_wait_customization_job(
        client, backend, spec, workspace, timeout=JOB_TIMEOUT_SECONDS
    )
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, workspace)
    assert_output_registered(client, workspace, base_model_name=base_entity, output_name=output)

    # A LoRA adapter is served from its base model's deployment; a full-weight output has its own.
    served_entity = base_entity if lora else output
    model_field = ceval.lora_model_field(workspace, output) if lora else ceval.base_model_field(workspace, output)
    deployment, config = jobs.wait_for_auto_deployment(client, workspace, served_entity)
    try:
        check(client, workspace, deployment, model_field)
    finally:
        jobs.delete_deployment(client, workspace, deployment, config)


@pytest.mark.platform("docker", "kubernetes")
@pytest.mark.feature("gpu", "automodel", "smoke")
@pytest.mark.parametrize("finetuning_type", ["lora", "all_weights"])
def test_automodel_sft_smoke(
    client: NemoClient,
    customizer_workspace: str,
    squad_smoke_fileset: str,
    automodel_base_entity: str,
    vllm_lora_template: str,
    finetuning_type: str,
) -> None:
    ws = customizer_workspace
    output = unique_name(f"qwen3-smoke-{finetuning_type}")
    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{ws}/{automodel_base_entity}",
            "dataset": {"training": f"{ws}/{squad_smoke_fileset}"},
            "training": {"training_type": "sft", "finetuning_type": finetuning_type, "max_seq_length": 1024},
            "schedule": {"epochs": 1, "max_steps": MAX_STEPS},
            "batch": {"global_batch_size": 8, "micro_batch_size": 1},
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            "output": {"name": output},
            "deployment_config": f"{ws}/{vllm_lora_template}",
        }
    )
    lora = finetuning_type == "lora"
    _train_and_serve(client, ws, "automodel", spec, base_entity=automodel_base_entity, output=output, lora=lora)
    _serve_on_model_free_nim(client, ws, base_entity=automodel_base_entity, output=output, lora=lora)


def _serve_on_model_free_nim(client: NemoClient, workspace: str, *, base_entity: str, output: str, lora: bool) -> None:
    """Serve the same trained output on the model-free NIM and check the settings NIM 2.x reads.

    A LoRA adapter is served from its base model, which also carries a custom chat template. A
    full-weight output is served with tool calling.
    """
    if lora:
        deployment, config = jobs.deploy_model_free_nim(
            client, workspace, base_entity, lora_enabled=True, chat_template=PREAMBLE_CHAT_TEMPLATE
        )
        try:
            base = ceval.base_model_field(workspace, base_entity)
            response = jobs.provider_chat(client, workspace, deployment, model=base)
            prompt_tokens = response["usage"]["prompt_tokens"]
            assert prompt_tokens >= CHAT_TEMPLATE_PREAMBLE_WORDS, (
                f"model-free NIM did not apply the deployment's chat template: {prompt_tokens} prompt tokens"
            )
            jobs.assert_chat_completion(client, workspace, deployment, base)
            _assert_chat_when_adapter_loads(client, workspace, deployment, ceval.lora_model_field(workspace, output))
        finally:
            jobs.delete_deployment(client, workspace, deployment, config)
        return

    deployment, config = jobs.deploy_model_free_nim(
        client,
        workspace,
        output,
        tool_call_config=ToolCallConfig(tool_call_parser="hermes", auto_tool_choice=True),
    )
    try:
        model = ceval.base_model_field(workspace, output)
        jobs.assert_chat_completion(client, workspace, deployment, model)
        # vLLM rejects tool_choice "auto" unless --enable-auto-tool-choice and --tool-call-parser
        # are set, so a successful response shows the tool calling settings reached NIM.
        jobs.provider_chat(
            client,
            workspace,
            deployment,
            model=model,
            messages=WEATHER_QUESTION,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        # "auto" may answer in plain text, so force a call to check one comes back.
        response = jobs.provider_chat(
            client,
            workspace,
            deployment,
            model=model,
            messages=WEATHER_QUESTION,
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=64,
        )
        tool_calls = response["choices"][0]["message"].get("tool_calls")
        assert isinstance(tool_calls, list) and tool_calls, response
    finally:
        jobs.delete_deployment(client, workspace, deployment, config)


def _assert_chat_when_adapter_loads(client: NemoClient, workspace: str, deployment: str, adapter_model: str) -> None:
    deadline = time.monotonic() + ADAPTER_LOAD_TIMEOUT_S
    while True:
        try:
            jobs.assert_chat_completion(client, workspace, deployment, adapter_model)
            return
        except NemoHTTPError as error:
            if time.monotonic() > deadline:
                raise
            logger.info("Adapter %s not served yet (HTTP %s), retrying", adapter_model, error.status_code)
            time.sleep(15)


@pytest.mark.platform("docker", "kubernetes")
@pytest.mark.feature("gpu", "unsloth", "smoke")
@pytest.mark.parametrize("finetuning_type", ["lora", "all_weights"])
def test_unsloth_sft_smoke(
    client: NemoClient,
    customizer_workspace: str,
    squad_smoke_fileset: str,
    unsloth_base_entity: str,
    vllm_lora_template: str,
    finetuning_type: str,
) -> None:
    ws = customizer_workspace
    output = unique_name(f"qwen3-unsloth-smoke-{finetuning_type}")
    output_spec: dict[str, Any] = {"name": output}
    if finetuning_type == "lora":
        output_spec["save_method"] = "lora"
    spec = UnslothJobInput.model_validate(
        {
            "model": {"name": f"{ws}/{unsloth_base_entity}", "max_seq_length": 1024, "load_in_4bit": False},
            "dataset": {
                "path": f"{ws}/{squad_smoke_fileset}",
                "apply_chat_template": True,
            },
            "training": {"finetuning_type": finetuning_type},
            "schedule": {"epochs": 1, "max_steps": MAX_STEPS},
            "output": output_spec,
            "deployment_config": f"{ws}/{vllm_lora_template}",
        }
    )
    _train_and_serve(
        client, ws, "unsloth", spec, base_entity=unsloth_base_entity, output=output, lora=finetuning_type == "lora"
    )


@pytest.mark.platform("kubernetes")
@pytest.mark.feature("gpu", "rl", "smoke")
def test_rl_dpo_smoke(
    client: NemoClient,
    customizer_workspace: str,
    helpsteer_dpo_smoke_fileset: str,
    rl_base_entity: str,
    vllm_lora_template: str,
) -> None:
    jobs.require_kubernetes_backend(client)
    ws = customizer_workspace
    output = unique_name("qwen3-dpo-smoke")
    spec = _rl_job_input().model_validate(
        {
            "model": f"{ws}/{rl_base_entity}",
            "dataset": f"{ws}/{helpsteer_dpo_smoke_fileset}",
            "training": {
                "type": "dpo",
                "max_steps": MAX_STEPS,
                "batch_size": 8,
                "micro_batch_size": 1,
                "max_seq_length": 1024,
                "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            },
            "output": {"name": output},
            "deployment_config": f"{ws}/{vllm_lora_template}",
        }
    )
    _train_and_serve(client, ws, "rl", spec, base_entity=rl_base_entity, output=output, lora=False)


# GRPO runs math_with_judge (wheels-v1 and native-v1) and the converted deepscaler-math
# adapter package, with each policy that should train. Sandbox internet is platform-wide,
# and turning it on stops forcing the wheels formats offline, so the native-v1 formats
# (which install from an index) run in a second phase with it on.
GRPO_POLICIES = {
    "am-fw": {"policy_backend": "automodel", "finetuning_type": "all_weights"},
    "am-lr": {"policy_backend": "automodel", "finetuning_type": "lora"},
}


def _grpo_smoke(
    client: NemoClient,
    workspace: str,
    *,
    cell: str,
    environment: str,
    dataset: str,
    policy: str,
    base_entity: str,
    deployment_template: str,
) -> None:
    jobs.require_kubernetes_backend(client)
    output = unique_name(f"qwen3-grpo-{cell}-{policy}")
    spec = _rl_job_input().model_validate(
        {
            "model": f"{workspace}/{base_entity}",
            "dataset": f"{workspace}/{dataset}",
            "environment": f"{workspace}/{environment}",
            "training": {
                "type": "grpo",
                **GRPO_POLICIES[policy],
                "max_steps": GRPO_MAX_STEPS,
                "num_prompts_per_step": 4,
                "num_generations_per_prompt": 4,
                "batch_size": 16,
                "micro_batch_size": 1,
                "learning_rate": 1e-6,
                "max_seq_length": 1024,
                "max_new_tokens": 512,
                "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            },
            "output": {"name": output},
            "deployment_config": f"{workspace}/{deployment_template}",
        }
    )
    lora = GRPO_POLICIES[policy]["finetuning_type"] == "lora"
    _train_and_serve(client, workspace, "rl", spec, base_entity=base_entity, output=output, lora=lora)


@pytest.mark.platform("kubernetes")
@pytest.mark.feature("gpu", "rl", "grpo", "smoke")
@pytest.mark.parametrize("policy", GRPO_POLICIES)
@pytest.mark.parametrize("env_format", ["wheels-v1", "adapter-wheels-v1"])
def test_rl_grpo_offline_formats_smoke(
    request: pytest.FixtureRequest,
    client: NemoClient,
    customizer_workspace: str,
    rl_base_entity: str,
    vllm_lora_template: str,
    env_format: str,
    policy: str,
) -> None:
    """Vendored wheels: the sandbox installs them with no package index."""
    if env_format == "wheels-v1":
        # Resolved only for this case, so an adapter-wheels run does not stage the math package.
        cell = "wv"
        environment = request.getfixturevalue("grpo_math_env_fileset")
        dataset = request.getfixturevalue("grpo_math_smoke_fileset")
    else:
        # Resolved only for this case, so a wheels-v1 run does not need the hub conversion.
        cell = "ds"
        environment = request.getfixturevalue("grpo_deepscaler_env_fileset")
        dataset = request.getfixturevalue("grpo_deepscaler_smoke_fileset")
    _grpo_smoke(
        client,
        customizer_workspace,
        cell=cell,
        environment=environment,
        dataset=dataset,
        policy=policy,
        base_entity=rl_base_entity,
        deployment_template=vllm_lora_template,
    )


@pytest.mark.platform("kubernetes")
@pytest.mark.feature("gpu", "rl", "grpo", "grpo-internet", "smoke")
@pytest.mark.parametrize("policy", GRPO_POLICIES)
@pytest.mark.parametrize("env_format", ["native-v1", "native-v1-ref"])
def test_rl_grpo_internet_formats_smoke(
    client: NemoClient,
    customizer_workspace: str,
    grpo_math_env_native_fileset: str,
    grpo_math_env_native_ref_fileset: str,
    grpo_math_smoke_fileset: str,
    rl_base_entity: str,
    vllm_lora_template: str,
    env_format: str,
    policy: str,
) -> None:
    """native-v1 installs the server's requirements from an index, so it needs sandbox internet.

    Reference-only does too: the image prebuilds no Gym venvs, so its built-in server installs
    on first use.
    """
    cell, environment = {
        "native-v1": ("nv", grpo_math_env_native_fileset),
        "native-v1-ref": ("nr", grpo_math_env_native_ref_fileset),
    }[env_format]
    _grpo_smoke(
        client,
        customizer_workspace,
        cell=cell,
        environment=environment,
        dataset=grpo_math_smoke_fileset,
        policy=policy,
        base_entity=rl_base_entity,
        deployment_template=vllm_lora_template,
    )


@pytest.mark.platform("docker", "kubernetes")
@pytest.mark.feature("gpu", "automodel", "smoke")
def test_automodel_embedding_smoke(
    client: NemoClient,
    customizer_workspace: str,
    nvdocs_embedding_smoke_fileset: str,
    embed_base_entity: str,
    vllm_template: str,
) -> None:
    ws = customizer_workspace
    output = unique_name("nemotron-embed-smoke")
    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{ws}/{embed_base_entity}",
            "dataset": {"training": f"{ws}/{nvdocs_embedding_smoke_fileset}"},
            "training": {
                "training_type": "sft",
                "recipe": "bi_encoder",
                "finetuning_type": "all_weights",
                "max_seq_length": 512,
                "retrieval": {"query_max_length": 512, "passage_max_length": 512, "train_n_passages": 5},
            },
            "schedule": {"max_steps": MAX_STEPS},
            "batch": {"global_batch_size": 32, "micro_batch_size": 4},
            "optimizer": {"learning_rate": 1e-5},
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
            "output": {"name": output},
            "deployment_config": f"{ws}/{vllm_template}",
        }
    )
    _train_and_serve(
        client,
        ws,
        "automodel",
        spec,
        base_entity=embed_base_entity,
        output=output,
        lora=False,
        check=jobs.assert_embeddings,
    )
