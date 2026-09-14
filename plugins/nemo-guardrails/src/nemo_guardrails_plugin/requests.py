# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nemo_guardrails_plugin.schemas import GuardrailsRequest
from nemo_platform_plugin.guardrail.types import GenerationLogOptionsParam
from nemo_platform_plugin.inference_middleware import InferenceMiddlewareError
from pydantic import ValidationError


def _format_validation_errors(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "Validation error")
        messages.append(f"{loc}: {msg}" if loc else msg)

    detail = "; ".join(messages)
    if len(messages) > 1:
        return f"{len(messages)} validation errors. {detail}"
    return detail


def sanitize_request_body_for_proxy(request_body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request body without Guardrails-specific fields."""
    proxied_request_body = dict(request_body)
    proxied_request_body.pop("guardrails", None)

    return proxied_request_body


def parse_guardrails_request(guardrails: Any) -> GuardrailsRequest | None:
    """Validate the request-time ``guardrails`` envelope for the plugin path."""
    if guardrails is None:
        return None

    try:
        return GuardrailsRequest.model_validate(guardrails)
    except ValidationError as exc:
        raise InferenceMiddlewareError(
            f"Invalid guardrails request options: {_format_validation_errors(exc)}",
            status_code=422,
        ) from exc


def extract_log_options_from_request(guardrails: GuardrailsRequest | None) -> GenerationLogOptionsParam | None:
    """Extract user-requested log options from the guardrails request field.

    Example guardrails value:
    {
        "options": {
            "log": {
                "activated_rails": true,
                "llm_calls": true
            }
        }
    }

    Returns None if not requested.
    """
    if guardrails is None or guardrails.options is None or guardrails.options.log is None:
        return None

    log_options = guardrails.options.log.model_dump(exclude_none=True)
    return log_options or None


def extract_return_choice_from_request(guardrails: GuardrailsRequest | None) -> bool:
    """Return whether guardrails_data should be emitted as an extra choice."""
    return guardrails.return_choice if guardrails is not None else False
