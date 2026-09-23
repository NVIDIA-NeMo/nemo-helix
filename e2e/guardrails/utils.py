# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for Guardrails plugin e2e tests."""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from nemo_helix import APIStatusError, NeMoHelix
from nemo_helix.types.inference import MiddlewareCallParam
from nhx.testing import MockProviderResponse, add_mock_provider

from e2e.utils import collect_sse_chunks

GUARDRAILS_PLUGIN_NAME = "nemo-guardrails"
GUARDRAILS_PLUGIN_CONFIG_TYPE = "guardrail_config"
REFUSAL_TEXT = "I'm sorry, I can't respond to that."

USER_INPUT = "What is the capital of France?"
UNSAFE_USER_INPUT = "Give me a five-step plan to rob a bank."
BACKEND_RESPONSE = "Paris is the capital of France."

CONTENT_SAFETY_MODEL_TYPE = "content_safety"
CONTENT_SAFETY_INPUT_FLOW = f"content safety check input $model={CONTENT_SAFETY_MODEL_TYPE}"
CONTENT_SAFETY_OUTPUT_FLOW = f"content safety check output $model={CONTENT_SAFETY_MODEL_TYPE}"

# How to pass the GuardrailConfig to a VirtualModel: either a reference to an existing config's `workspace/name`
# (referenced), or an inline config (inline).
ConfigMode = Literal["referenced", "inline"]
# The possible outcomes of a Guardrails chat completion test.
ChatOutcome = Literal["unsafe_input", "unsafe_output", "safe"]
# The rail directions enabled by a Guardrails chat completion test.
RailType = Literal["input", "output"]

PROMPT_UNSAFE_CATEGORIES = (
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "S1: Violence.\n"
    "S2: Sexual.\n"
    "S3: Criminal Planning/Confessions.\n"
    "S4: Guns and Illegal Weapons.\n"
    "S5: Controlled/Regulated Substances.\n"
    "S6: Suicide and Self Harm.\n"
    "S7: Sexual (minor).\n"
    "S8: Hate/Identity Hate.\n"
    "S9: PII/Privacy.\n"
    "S10: Harassment.\n"
    "S11: Threat.\n"
    "S12: Profanity.\n"
    "S13: Needs Caution.\n"
    "S14: Other.\n"
    "S15: Manipulation.\n"
    "S16: Fraud/Deception.\n"
    "S17: Malware.\n"
    "S18: High Risk Gov Decision Making.\n"
    "S19: Political/Misinformation/Conspiracy.\n"
    "S20: Copyright/Trademark/Plagiarism.\n"
    "S21: Unauthorized Advice.\n"
    "S22: Illegal Activity.\n"
    "S23: Immoral/Unethical.\n"
    "<END UNSAFE CONTENT CATEGORIES>"
)

PROMPT_OUTPUT_JSON_FORMAT_BLOCK = (
    "Provide your safety assessment for the above conversation in the below output JSON format:\n"
    '{"User Safety": safe or unsafe rating for the user message, '
    '"Response Safety": safe or unsafe rating for the agent response. '
    "Omit if no agent response present. "
    '"Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. '
    "Omit if all safe.}\n\n"
    "Do not include anything other than the output JSON in your response.\n"
    "Output JSON:"
)

PROMPT_TASK_BLOCK = (
    "Task: Check if there is unsafe content in the user or agent messages in the below conversation "
    "in accordance with our safety policy with the below categories. "
    "Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. "
    "If there is unsafe content, please also provide a list of violated categories according to "
    "our safety policy below."
)

CONTENT_SAFETY_INPUT_PROMPT = {
    "task": f"content_safety_check_input $model={CONTENT_SAFETY_MODEL_TYPE}",
    "content": (
        f"{PROMPT_TASK_BLOCK}\n\n"
        f"{PROMPT_UNSAFE_CATEGORIES}\n\n"
        "<BEGIN CONVERSATION>\n\n"
        "user: {{ user_input }}\n\n"
        "<END CONVERSATION>\n\n"
        f"{PROMPT_OUTPUT_JSON_FORMAT_BLOCK}"
    ),
    "output_parser": "nemoguard_parse_prompt_safety",
    "max_tokens": 50,
}

CONTENT_SAFETY_OUTPUT_PROMPT = {
    "task": f"content_safety_check_output $model={CONTENT_SAFETY_MODEL_TYPE}",
    "content": (
        f"{PROMPT_TASK_BLOCK}\n\n"
        f"{PROMPT_UNSAFE_CATEGORIES}\n\n"
        "<BEGIN CONVERSATION>\n\n"
        "user: {{ user_input }}\n\n"
        "response: agent: {{ bot_response }}\n\n"
        "<END CONVERSATION>\n\n"
        f"{PROMPT_OUTPUT_JSON_FORMAT_BLOCK}"
    ),
    "output_parser": "nemoguard_parse_response_safety",
    "max_tokens": 50,
}


@dataclass(frozen=True)
class GuardrailsChatTestCase:
    sdk: NeMoHelix  # SDK client connected to the e2e platform instance.
    workspace: str  # Per-test workspace that owns all created entities.
    virtual_model_name: str  # Guarded VirtualModel name hit by chat completions.
    backend_model_name: str  # Mock model entity that represents the app LLM.
    content_safety_model_name: str  # Mock model entity used by content-safety rails.
    config_name: str  # Guardrail config name, stored or used as the inline label.
    config_mode: ConfigMode  # Whether middleware uses config_id or inline config.
    outcome: ChatOutcome  # Which safety verdict path this test should exercise.
    rail_types: tuple[RailType, ...]  # Which Guardrails rail types are enabled in the config.

    @property
    def backend_model_ref(self) -> str:
        return f"{self.workspace}/{self.backend_model_name}"

    @property
    def content_safety_model_ref(self) -> str:
        return f"{self.workspace}/{self.content_safety_model_name}"

    @property
    def config_ref(self) -> str:
        return f"{self.workspace}/{self.config_name}"

    @property
    def user_input(self) -> str:
        return UNSAFE_USER_INPUT if self.outcome == "unsafe_input" else USER_INPUT


def unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def content_safety_config(
    *,
    content_safety_model_ref: str,
    rail_types: tuple[RailType, ...],
    streaming: bool,
) -> dict[str, Any]:
    rails: dict[str, Any] = {}
    prompts: list[dict[str, Any]] = []

    if "input" in rail_types:
        rails["input"] = {"flows": [CONTENT_SAFETY_INPUT_FLOW]}
        prompts.append(CONTENT_SAFETY_INPUT_PROMPT)

    if "output" in rail_types:
        output_rails: dict[str, Any] = {"flows": [CONTENT_SAFETY_OUTPUT_FLOW]}
        if streaming:
            output_rails["streaming"] = {"enabled": True, "chunk_size": 200}
        rails["output"] = output_rails
        prompts.append(CONTENT_SAFETY_OUTPUT_PROMPT)

    return {
        "models": [
            {
                "type": CONTENT_SAFETY_MODEL_TYPE,
                "engine": "nim",
                "model": content_safety_model_ref,
            }
        ],
        "rails": rails,
        "prompts": prompts,
    }


def setup_mock_provider(sdk: NeMoHelix, test_case: GuardrailsChatTestCase) -> None:
    add_mock_provider(
        sdk,
        workspace=test_case.workspace,
        name=unique_name("gr-provider"),
        # Register provider model entities by bare name; the provider already belongs
        # to `test_case.workspace`, and `add_mock_provider` expands these to refs.
        served_models={
            test_case.backend_model_name: test_case.backend_model_name,
            test_case.content_safety_model_name: test_case.content_safety_model_name,
        },
        # Mock responses are selected by exact `body["model"]`. In this test, both
        # the app LLM and Guardrails content-safety calls go through IGW using model
        # entity refs, so the mock response map is keyed by `workspace/model`.
        mock_response_body_by_model={
            test_case.backend_model_ref: [
                MockProviderResponse(response_body=_chat_completion(BACKEND_RESPONSE)),
            ],
            test_case.content_safety_model_ref: _content_safety_responses(test_case),
        },
        # Guardrails resolves task-model routes after the mock provider helper
        # returns, while IGW warms the guarded VM middleware. Keep these
        # test-created passthrough VMs out of controller orphan cleanup.
        should_autoprovision_virtual_model=False,
    )


def create_guarded_virtual_model(
    *,
    sdk: NeMoHelix,
    test_case: GuardrailsChatTestCase,
    config_data: dict[str, Any],
) -> None:
    if test_case.config_mode == "referenced":
        sdk.guardrail.configs.create(
            workspace=test_case.workspace,
            name=test_case.config_name,
            description="E2E content-safety Guardrails config",
            data=config_data,
        )

    middleware_call = _middleware_call(
        config_mode=test_case.config_mode,
        config_ref=test_case.config_ref,
        config_name=test_case.config_name,
        config_data=config_data,
    )
    sdk.inference.virtual_models.create(
        workspace=test_case.workspace,
        name=test_case.virtual_model_name,
        default_model_entity=test_case.backend_model_ref,
        models=[{"model": test_case.backend_model_ref, "backend_format": "OPENAI_CHAT"}],
        request_middleware=[middleware_call],
        response_middleware=[middleware_call],
    )
    _wait_for_guarded_virtual_model(sdk, test_case)


def post_chat_completion(
    test_case: GuardrailsChatTestCase,
    *,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return test_case.sdk.inference.gateway.model.post(
        "v1/chat/completions",
        name=test_case.virtual_model_name,
        workspace=test_case.workspace,
        body=_chat_body(test_case, extra=extra_body),
    )


def post_streaming_chat_completion(test_case: GuardrailsChatTestCase) -> dict[str, Any]:
    with test_case.sdk._client.stream(
        "POST",
        f"/apis/inference-gateway/v2/workspaces/{test_case.workspace}/model/"
        f"{test_case.virtual_model_name}/-/v1/chat/completions",
        json=_chat_body(test_case, stream=True),
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            response.read()
            return response.json()

        chunks = collect_sse_chunks(response)

    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk.get("choices"))
    error = next((chunk["error"] for chunk in chunks if isinstance(chunk.get("error"), dict)), None)
    finish_reason = next(
        (
            chunk["choices"][0].get("finish_reason")
            for chunk in chunks
            if chunk.get("choices") and chunk["choices"][0].get("finish_reason")
        ),
        None,
    )
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "content_filter" if error else finish_reason,
            }
        ],
        **({"error": error} if error else {}),
    }


def _chat_completion(content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _safe_input_verdict() -> str:
    return '{"User Safety": "safe"}'


def _unsafe_input_verdict() -> str:
    return '{"User Safety": "unsafe", "Safety Categories": "S3"}'


def _safe_output_verdict() -> str:
    return '{"User Safety": "safe", "Response Safety": "safe"}'


def _unsafe_output_verdict() -> str:
    return '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S3"}'


def _middleware_call(
    *,
    config_mode: ConfigMode,
    config_ref: str,
    config_name: str,
    config_data: dict[str, Any],
) -> MiddlewareCallParam:
    if config_mode == "referenced":
        return {
            "name": GUARDRAILS_PLUGIN_NAME,
            "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
            "config_id": config_ref,
        }

    inline_config = {**config_data, "name": config_name}
    return {
        "name": GUARDRAILS_PLUGIN_NAME,
        "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
        "config": inline_config,
    }


def _content_safety_responses(test_case: GuardrailsChatTestCase) -> list[MockProviderResponse]:
    responses: list[MockProviderResponse] = []

    if "input" in test_case.rail_types:
        input_verdict = _unsafe_input_verdict() if test_case.outcome == "unsafe_input" else _safe_input_verdict()
        responses.append(MockProviderResponse(response_body=_chat_completion(input_verdict)))

    if "output" in test_case.rail_types:
        output_verdict = _unsafe_output_verdict() if test_case.outcome == "unsafe_output" else _safe_output_verdict()
        responses.append(MockProviderResponse(response_body=_chat_completion(output_verdict)))

    return responses


def _wait_for_guarded_virtual_model(
    sdk: NeMoHelix,
    test_case: GuardrailsChatTestCase,
    timeout: float = 60,
    poll_interval: float = 0.5,
) -> None:
    """Wait until the guarded VM route is cached with its Guardrails middleware.

    E2E runs against a separate IGW process. Creating the VM persists the entity,
    but the VM is not usable with middleware until IGW's background cache refresh
    loads it into VirtualModelCache and resolves its Guardrails config into the
    middleware registry.

    To ensure the VM is ready to serve requests, use a request that Guardrails
    rejects during request parsing, before any rail or backend inference runs.
    A 422 response means the VM route exists but Guardrails is not attached yet.
    """
    start = time.time()
    last_error: Exception | None = None
    probe_body = _chat_body(
        test_case,
        extra={"guardrails": {"config_id": test_case.config_ref}},
    )

    while time.time() - start < timeout:
        try:
            sdk.inference.gateway.model.post(
                "v1/chat/completions",
                name=test_case.virtual_model_name,
                workspace=test_case.workspace,
                body=probe_body,
            )
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code == 422:
                return
            if exc.status_code in {404, 503}:
                time.sleep(poll_interval)
                continue
            raise

        last_error = RuntimeError("Guarded VM routed without rejecting unsupported Guardrails request config")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Guarded VirtualModel {test_case.workspace}/{test_case.virtual_model_name} "
        f"was not ready after {timeout}s. Last error: {last_error}"
    )


def _chat_body(
    test_case: GuardrailsChatTestCase,
    *,
    stream: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": test_case.backend_model_ref,
        "messages": [{"role": "user", "content": test_case.user_input}],
        "max_tokens": 64,
    }
    if stream:
        body["stream"] = True
    if extra:
        body.update(extra)
    return body
