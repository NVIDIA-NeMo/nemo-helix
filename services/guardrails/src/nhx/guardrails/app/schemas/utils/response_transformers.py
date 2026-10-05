# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Tuple

from nemoguardrails.rails.llm.options import GenerationResponse
from nhx.guardrails.api.schemas import (
    ChatCompletionResponseChoice,
    CompletionResponseChoice,
    UsageInfo,
)
from nhx.guardrails.entities.enums import (
    StatusEnum,
)
from nhx.guardrails.entities.values.chat import GuardrailChatCompletionResponse
from nhx.guardrails.entities.values.check import (
    GuardrailCheckResponse,
    RailStatus,
)
from nhx.guardrails.entities.values.common import GuardrailsDataOutput
from nhx.guardrails.entities.values.completions import (
    GuardrailCompletionResponse,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def create_guardrail_chat_completion_response_from_generation_response(
    response: GenerationResponse,
    config_ids: list[str] | None = None,
    log_options: dict | None = None,
) -> GuardrailChatCompletionResponse:
    """
    Convert a GenerationResponse to GuardrailChatCompletionResponse format.

    Args:
        response (GenerationResponse): The response from the generation process.
        config_ids (Optional[List[str]]): List of configuration IDs.
        log_options: The log options the user explicitly requested.
            When `None` or an empty dict, `log` is omitted from the response entirely.
            Individual fields within `log` are only included when their corresponding option is `True`.

    Returns:
        GuardrailChatCompletionResponse: The transformed chat completion response.
    """
    activated_rails = response.log.activated_rails if response.log else []

    # Determine if any rail blocked the response
    was_blocked = any(rail.stop for rail in activated_rails)

    # Build choices directly from response.response — this is always the guardrails output,
    # whether it's the main model's response or a refusal message.
    choices = [
        ChatCompletionResponseChoice.model_construct(
            index=i,
            message={"role": msg.get("role", "assistant"), "content": msg.get("content", "")},
            finish_reason="content_filter" if was_blocked else "stop",
        )
        for i, msg in enumerate(response.response)
    ]

    # Extract model name and usage from the generation rail.
    # We can't reliable extract this from `llm_metadata.response_metadata`, because it stores the *last*
    # LLM call in the pipeline (ex. if an output rail ran, it represents the output rail's LLM response).
    # If a generation rail is not present, there was most likely an error, or the user input was blocked
    # by an input rail, so we default to placeholder model and empty usage info.
    generation_rail = next(
        (rail for rail in activated_rails if rail.type == "generation"),
        None,
    )

    if generation_rail:
        llm_calls = [call for action in generation_rail.executed_actions for call in action.llm_calls]
        model = llm_calls[-1].llm_model_name if llm_calls else "-"
        prompt_tokens = completion_tokens = total_tokens = 0
        for c in llm_calls:
            prompt_tokens += c.prompt_tokens or 0
            completion_tokens += c.completion_tokens or 0
            total_tokens += c.total_tokens or 0
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    else:
        # If the input was blocked before generation, fall back to placeholder model and usage info.
        model = "-"
        usage = UsageInfo()

    # Only include `log` in the response when the user explicitly requested at least one log field.
    if log_options and any(log_options.values()):
        log_dict = _get_log_dict(response)
        # If the user didn't explicitly request `activated_rails`, remove it from the log.
        if not log_options.get("activated_rails", False) and isinstance(log_dict, dict):
            log_dict.pop("activated_rails", None)
    else:
        log_dict = None

    guardrails_data = GuardrailsDataOutput(
        config_ids=config_ids,
        output_data=response.output_data,
        log=log_dict,
    )

    return GuardrailChatCompletionResponse(
        choices=choices,
        model=model,
        usage=usage,
        guardrails_data=guardrails_data,
    )


def create_guardrail_completion_response_from_generation_response(
    response: GenerationResponse,
    config_ids: list[str] | None = None,
    log_options: dict | None = None,
) -> GuardrailCompletionResponse:
    """
    Convert a GenerationResponse to GuardrailCompletionResponse format.

    Args:
        response (GenerationResponse): The response from the generation process.
        config_ids (Optional[List[str]]): List of configuration IDs.
        log_options: The log options the user explicitly requested.  When `None`
            or an empty dict, `log` is omitted from the response entirely.

    Returns:
        GuardrailCompletionResponse: The transformed completion response.
    """
    activated_rails = response.log.activated_rails if response.log else []

    was_blocked = any(rail.stop for rail in activated_rails)

    # For completions, response.response is typically a string.
    # Handle both str and List[dict] for defensive purposes.
    if isinstance(response.response, str):
        choices = [
            CompletionResponseChoice.model_construct(
                index=0,
                text=response.response,
                finish_reason="content_filter" if was_blocked else "stop",
            )
        ]
    else:
        choices = [
            CompletionResponseChoice.model_construct(
                index=i,
                text=msg.get("content", ""),
                finish_reason="content_filter" if was_blocked else "stop",
            )
            for i, msg in enumerate(response.response)
        ]

    generation_rail = next(
        (rail for rail in activated_rails if rail.type == "generation"),
        None,
    )

    if generation_rail:
        llm_calls = [call for action in generation_rail.executed_actions for call in action.llm_calls]
        model = llm_calls[-1].llm_model_name if llm_calls else "-"
        prompt_tokens = completion_tokens = total_tokens = 0
        for c in llm_calls:
            prompt_tokens += c.prompt_tokens or 0
            completion_tokens += c.completion_tokens or 0
            total_tokens += c.total_tokens or 0
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    else:
        model = "-"
        usage = UsageInfo()

    if log_options and any(log_options.values()):
        log_dict = _get_log_dict(response)
        if not log_options.get("activated_rails", False) and isinstance(log_dict, dict):
            log_dict.pop("activated_rails", None)
    else:
        log_dict = None

    guardrails_data = GuardrailsDataOutput(
        config_ids=config_ids,
        output_data=response.output_data,
        log=log_dict,
    )

    return GuardrailCompletionResponse(
        choices=choices,
        model=model,
        usage=usage,
        guardrails_data=guardrails_data,
    )


def create_guardrail_check_response_from_generation_response(
    response: GenerationResponse,
    rails: List[str],
    exclude_activated_rails_options: bool,
) -> GuardrailCheckResponse:
    """
    Convert a GenerationResponse to GuardrailCheckResponse format.

    Args:
        response (GenerationResponse): The response from generation.
        rails (List[str]): List of exact rail names to check.

    Returns:
        GuardrailCheckResponse: The transformed guardrail check response.
    """

    rails_status = {}

    # Extract all blocked rails from the response
    overall_status, activated_rails_map = _extract_status_from_response(response)

    # Iterate through the rails in order and populate rails_status
    for rail in rails:
        if rail in activated_rails_map:
            rails_status[rail] = RailStatus(status=activated_rails_map[rail])
            # No need to include rails after the first blocked rail
            if rails_status[rail].status == StatusEnum.BLOCKED:
                break

    if exclude_activated_rails_options:
        # reset activated_rails
        response.log.activated_rails = []

    log_dict = _get_log_dict(response)

    # Construct GuardrailsDataOutput
    guardrails_data = GuardrailsDataOutput(
        config_ids=None,  # Since we're using rails: List[str], config_ids may not be applicable unless we support hasing
        output_data=response.output_data,
        log=log_dict,
    )

    return GuardrailCheckResponse(
        status=overall_status,
        rails_status=rails_status,
        guardrails_data=guardrails_data,
    )


def _extract_status_from_response(response: GenerationResponse) -> Tuple[StatusEnum, Dict[str, StatusEnum]]:
    """
    Extract the overall status and the map of activated rails and their rail status from the GenerationResponse.
    The overall status is BLOCKED if there is an exception or a blocked rail. Otherwise, the overall status is SUCCESS.

    Each activated rails is mapped to its status. The status is BLOCKED if the rail is stopped. Otherwise, the status is SUCCESS.
    However, if there is an exception but no blocked rails, each activated rail's status is set to UNKNOWN. This is to guard against unexpected state.

    Args:
        response (GenerationResponse): The response from generation.

    Returns:
        Tuple[StatusEnum, Dict[str, StatusEnum]]: tuple of overall status, map of activated rails and their status.
    """
    # First, find if there is an exception in the response
    exception_present: bool = False
    for msg in response.response:
        if msg.get("role") == "exception" and msg.get("content") is not None:
            exception_present = True
            break

    # Next, build a map of activated rails and their stop status
    blocked_rail_present = False
    activated_rails_map = {}
    if response.log and response.log.activated_rails:
        for rail in response.log.activated_rails:
            if not blocked_rail_present and rail.stop:
                blocked_rail_present = True
            if rail.stop:
                activated_rails_map[rail.name] = StatusEnum.BLOCKED
            elif rail.name not in activated_rails_map:
                activated_rails_map[rail.name] = StatusEnum.SUCCESS

    if exception_present and not blocked_rail_present:
        logger.warning("Unexpected state. No activated rails received for an exception", stack_info=True)
        for k in activated_rails_map:
            activated_rails_map[k] = StatusEnum.UNKNOWN

    overall_status = StatusEnum.BLOCKED if (exception_present or blocked_rail_present) else StatusEnum.SUCCESS

    return overall_status, activated_rails_map


def _get_log_dict(response):
    # to avoid validation error due to cross module type mismatch
    # see nhx#844
    if isinstance(response.log, BaseModel):
        log_dict = response.log.model_dump()
    else:
        log_dict = response.log

    return log_dict
