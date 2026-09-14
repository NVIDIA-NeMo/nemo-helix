# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from nemo_guardrails_plugin.constants import GUARDRAILS_DATA_MESSAGE_ROLE
from nemo_guardrails_plugin.rails import build_guardrails_data
from nemo_platform_plugin.guardrail.types import GenerationLogOptionsParam
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareError,
    InferenceResponse,
    ResponseResult,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import GenerationLog, GenerationResponse

logger = logging.getLogger(__name__)

GUARDRAILS_DATA_FIELD = "guardrails_data"

# Matches the "[<status>] <detail>" prefix langchain_nvidia_ai_endpoints prefixes
# on every error it raises. The status represents the underlying status code
# that we'd like to preserve when returning the error to the caller.
_LANGCHAIN_ERROR_MESSAGE_PREFIX = re.compile(r"^\s*\[(\d{3})\]\s*(.*)", re.DOTALL)
# Error codes embedded in the upstream error message that we surface to the caller.
_HTTP_ERROR_STATUS_RANGE = range(400, 600)


def extract_upstream_error(exc: BaseException) -> InferenceMiddlewareError | None:
    """Recover a genuine upstream status code hidden inside a rail-task LLM call failure.

    nemoguardrails wraps every rail-task LLM failure in the same generic
    ``LLMCallException``, so our middleware's catch-all exception handler
    would otherwise map all of them to a 503. This function unwraps the
    exception and, if the provider library recorded a real upstream status,
    propagates it as-is (4xx or 5xx) instead of masking it. If not found,
    the plugin falls back to a 503.

    Example, given this ``exc`` parameter::

        exc = LLMCallException(
            Exception('[400] At most 1 image(s) may be provided in one '
                       'request.'),
            detail="Error invoking LLM (model=..., provider=nvidia_ai_endpoints, endpoint=...)",
        )

    the return value is::

        InferenceMiddlewareError(
            "Error invoking LLM (model=..., provider=nvidia_ai_endpoints, endpoint=...): "
            "At most 1 image(s) may be provided in one request.",
            status_code=400,
        )

    Returns ``None`` if no status is found, so the caller's 503 fallback stands.
    """
    context: str | None = None
    # `exc` is always this outer LLMCallException in practice. It never has a
    # status of its own (see checks below), but its `.detail` (e.g. "Error
    # invoking LLM (model=..., provider=..., endpoint=...)") stores which call
    # failed, so it's kept as context. `candidate` becomes the wrapped
    # exception the provider library actually raised.
    candidate: BaseException = exc
    if isinstance(exc, LLMCallException):
        context = exc.detail
        if isinstance(exc.inner_exception, BaseException):
            candidate = exc.inner_exception

    # Signal 1: a genuine `status_code` attribute, e.g. set by `openai.APIStatusError`.
    status_code = getattr(candidate, "status_code", None)
    if isinstance(status_code, int) and status_code in _HTTP_ERROR_STATUS_RANGE:
        detail = getattr(candidate, "message", None) or str(candidate)
        return InferenceMiddlewareError(f"{context}: {detail}" if context else detail, status_code=status_code)

    # Signal 2: no structured status, so fall back to the "[<status>] ..."
    # message prefix that's all `langchain_nvidia_ai_endpoints` leaves behind.
    if match := _LANGCHAIN_ERROR_MESSAGE_PREFIX.match(str(candidate)):
        status_code = int(match.group(1))
        if status_code in _HTTP_ERROR_STATUS_RANGE:
            detail = _sanitized_client_error_detail(match.group(2).strip()) or str(candidate)
            return InferenceMiddlewareError(f"{context}: {detail}" if context else detail, status_code=status_code)

    return None


def _sanitized_client_error_detail(text: str) -> str:
    """Strip the "Unknown Error {...}" placeholder langchain_nvidia_ai_endpoints
    leaves in front of the real upstream JSON body.

    For example, given this ``text`` parameter:

        text = 'Unknown Error\\n{"message": "At most 1 image(s) may be provided in one request.", "code": 400}'

    the return value is:

        "At most 1 image(s) may be provided in one request."

    For the detail, prefer a nested ``message``/``error``/``detail``/``title`` field
    from that JSON. Falls back to ``text`` unchanged if it isn't valid JSON, or
    has none of those fields.
    """
    brace_index = text.find("{")
    if brace_index == -1:
        return text  # no JSON body at all

    try:
        body = json.loads(text[brace_index:])
    except json.JSONDecodeError:
        return text  # looked like JSON, wasn't

    # Attempt to extract the error detail from the JSON body
    if isinstance(body, dict):
        for key in ("message", "error", "detail", "title"):
            if isinstance(value := body.get(key), str) and value:
                return value

    return text  # valid JSON, but none of the detail fields we look for


def build_chat_completion_response_id() -> str:
    """Build a chat completion response ID consistent with OpenAI-style responses."""
    return f"chatcmpl-{uuid.uuid4()}"


def is_blocked_generation_response(log: GenerationLog) -> bool:
    """Return True if the activation log indicates the request was blocked by a guardrail."""
    return any(rail.stop is True for rail in log.activated_rails)


def extract_response_content(generation_response: GenerationResponse) -> str:
    """Return the post-rail text from a ``GenerationResponse``.

    ``GenerationResponse.response`` is OpenAI-shaped. When it is a message list,
    the library puts the rails result on an ``role=assistant`` entry — even for
    **input-only** runs, where that string is the post-rail ``$user_message``
    (e.g. PII-redacted user text), not a model reply from chat history.
    """
    response = generation_response.response
    if isinstance(response, list):
        for item in reversed(response):
            if item.get("role") == "assistant" and item.get("content"):
                return item.get("content", "")

        return ""

    return response


# ---------------------------------------------------------------------------
# Masking write-back helpers
#
# After input-only generate_async, the (possibly redacted) user text is returned
# as GenerationResponse.response with role=assistant (not role=user).
# Example: request user "Hi John" → response [{"role": "assistant", "content": "Hi <PERSON>"}].
# If the message content is redacted, we need to write it back onto the last user message
# in the request.
# ---------------------------------------------------------------------------


def _index_of_current_user_message(messages: list[dict[str, Any]]) -> int | None:
    """Return the final message index when the current turn is a user message."""
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        return len(messages) - 1
    return None


def apply_input_rail_modifications(
    messages: list[dict[str, Any]],
    generation_response: GenerationResponse,
) -> list[dict[str, Any]]:
    """Return messages with input-rail modifications applied to the last user turn.

    Post-rail user text is taken from ``GenerationResponse.response`` via
    ``extract_response_content``. If modified, it is written onto the last user message.

    Returns the same list object when there is nothing to change; otherwise, returns a
    shallow copy that replaces only the last user message dict.
    """
    last_user_index = _index_of_current_user_message(messages)
    if last_user_index is None:
        return messages

    last_user_message = messages[last_user_index]
    original_content = last_user_message.get("content")
    if not isinstance(original_content, str):
        logger.debug(
            "Skipping input-rail write-back; last user content is %s, not str",
            type(original_content).__name__,
        )
        return messages

    # NOTE: the input-rail result is labeled as an assistant message in the GenerationResponse,
    # but the content is actually the redacted/transformed user message.
    processed = extract_response_content(generation_response)
    # If the message content wasn't modified, return the original messages.
    if not isinstance(processed, str) or processed == original_content:
        return messages

    # If the message content was modified, write it back onto the last user message.
    updated_messages = list(messages)
    updated_messages[last_user_index] = {**last_user_message, "content": processed}
    return updated_messages


def apply_output_rail_modifications(
    original_content: str,
    generation_response: GenerationResponse,
) -> str:
    """Return assistant content after output rails, from ``response`` content."""
    processed = extract_response_content(generation_response)

    if not isinstance(processed, str) or processed == original_content:
        return original_content

    return processed


def build_assistant_message_from_response_result(response_result: ResponseResult) -> dict[str, Any]:
    """
    Build an assistant message object with the content from the given response.
    """
    assistant_message = {
        "role": "assistant",
        "content": "",
    }

    if isinstance(response_result, AsyncIterator):
        return assistant_message

    try:
        content = response_result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return assistant_message

    assistant_message["content"] = content
    return assistant_message


def build_blocked_immediate_response_body(
    config_id: str,
    request_body: dict[str, Any],
    generation_response: GenerationResponse,
    user_log_options: GenerationLogOptionsParam | None,
) -> dict[str, Any]:
    """
    Build the response body to return when an input or output rail blocks the request.
    """
    blocked: dict[str, Any] = {
        "id": build_chat_completion_response_id(),
        "object": "chat.completion",
        "model": request_body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": extract_response_content(generation_response)},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    guardrails_data = build_guardrails_data(
        config_id,
        input_generation_response=generation_response,
        user_log_options=user_log_options,
    )
    if guardrails_data is not None:
        blocked[GUARDRAILS_DATA_FIELD] = guardrails_data.model_dump(exclude_none=True)

    return blocked


def build_immediate_response(
    *,
    response_body: dict[str, Any],
) -> ImmediateResponse:
    """Build the ImmediateResponse returned by the process_request handler."""
    result = dict(response_body)
    annotations = {}
    # If we should include guardrails_data in the result, add it as an annotation. IGW handles
    # merging the annotations into the response before returning to the caller.
    if GUARDRAILS_DATA_FIELD in result:
        annotations[GUARDRAILS_DATA_FIELD] = result.pop(GUARDRAILS_DATA_FIELD)

    return ImmediateResponse(
        data=result,
        response_body_annotations=annotations,
    )


def build_blocked_output_response_body(
    config_id: str,
    original_response: dict[str, Any],
    generation_response: GenerationResponse,
    input_generation_response: GenerationResponse | None,
    user_log_options: GenerationLogOptionsParam | None,
    return_guardrails_data_as_choice: bool = False,
) -> dict[str, Any]:
    """
    Build the response body to return when an output rail blocks the model's response.

    Preserves the original response shape, but overwrites the choices with a blocked
    assistant message.
    """
    blocked_response: dict[str, Any] = {
        **original_response,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": extract_response_content(generation_response)},
                "finish_reason": "content_filter",
            }
        ],
    }

    guardrails_data = build_guardrails_data(
        config_id,
        input_generation_response=input_generation_response,
        output_generation_response=generation_response,
        user_log_options=user_log_options,
    )

    if return_guardrails_data_as_choice:
        blocked_response["choices"].append(
            {
                "index": 1,
                "message": {
                    "role": GUARDRAILS_DATA_MESSAGE_ROLE,
                    "content": guardrails_data.model_dump_json(exclude_none=True),
                },
            }
        )
    else:
        blocked_response[GUARDRAILS_DATA_FIELD] = guardrails_data.model_dump(exclude_none=True)

    return blocked_response


def build_output_response_body(
    config_id: str,
    original_response: dict[str, Any],
    generation_response: GenerationResponse | None,
    input_generation_response: GenerationResponse | None,
    user_log_options: GenerationLogOptionsParam | None,
    return_guardrails_data_as_choice: bool = False,
) -> dict[str, Any]:
    """
    Build the response body to return when the model's response passes, or skips, the output rails.
    """
    response: dict[str, Any] = dict(original_response)
    if "choices" not in response:
        raise InferenceMiddlewareError(
            "Guardrails response middleware expected upstream response to include a 'choices' field.",
            status_code=500,
        )

    choices = list(response.get("choices", [])) if isinstance(response.get("choices"), list) else []
    response["choices"] = choices

    # Output rails validate choices[0].message, so return only the choice that was checked.
    if generation_response is not None and choices:
        choice = {**choices[0], "index": 0}
        raw_message = choice.get("message")
        message: dict[str, Any] = dict(raw_message) if isinstance(raw_message, dict) else {}
        original_content = message.get("content")

        # If the output rail modified the message content, write it back onto the choice.
        if isinstance(original_content, str):
            message["content"] = apply_output_rail_modifications(original_content, generation_response)
            choice["message"] = message

        response["choices"] = [choice]

    guardrails_data = build_guardrails_data(
        config_id,
        input_generation_response=input_generation_response,
        output_generation_response=generation_response,
        user_log_options=user_log_options,
    )

    if return_guardrails_data_as_choice:
        response["choices"].append(
            {
                "index": len(response["choices"]),
                "message": {
                    "role": GUARDRAILS_DATA_MESSAGE_ROLE,
                    "content": guardrails_data.model_dump_json(exclude_none=True),
                },
            }
        )
    else:
        response[GUARDRAILS_DATA_FIELD] = guardrails_data.model_dump(exclude_none=True)

    return response


def build_inference_response(
    *,
    response: InferenceResponse,
    response_body: dict[str, Any],
    return_guardrails_data_as_choice: bool = False,
) -> InferenceResponse:
    """Build the InferenceResponse returned by the process_response handler."""
    result = dict(response_body)
    annotations = dict(response.response_body_annotations)

    # If returning guardrails_data as a choice, remove it from the result and annotations.
    if return_guardrails_data_as_choice:
        annotations.pop(GUARDRAILS_DATA_FIELD, None)
        result.pop(GUARDRAILS_DATA_FIELD, None)
    # If we should include guardrails_data in the result, add it as an annotation. IGW handles
    # merging the annotations into the response before returning to the caller.
    elif GUARDRAILS_DATA_FIELD in result:
        annotations[GUARDRAILS_DATA_FIELD] = result.pop(GUARDRAILS_DATA_FIELD)

    return InferenceResponse(
        result=result,
        headers=response.headers,
        response_body_annotations=annotations,
    )
