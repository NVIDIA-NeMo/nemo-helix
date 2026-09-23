# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
from enum import Enum

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, Field, field_validator

from nemo_evaluator_sdk.inference import InferenceFn, PreprocessRequest, deep_merge
from nemo_evaluator_sdk.session import session_cache, session_lock
from nemo_evaluator_sdk.values import Model

_logger = logging.getLogger(__name__)

#: Probes must survive a reasoning model's thinking budget; too small and truncated output looks
#: identical to an endpoint that ignored the encoding.
_PROBE_MAX_TOKENS = 4096

#: The marker name must stay unguessable: the probe never shows this schema to the model, so a
#: mode only passes when the server actually injected the grammar.
_DEFAULT_PROBE_SCHEMA: dict = {
    "type": "object",
    "properties": {"__nhx_probe_score": {"type": "integer"}},
    "required": ["__nhx_probe_score"],
    "additionalProperties": False,
}


def _default_probe_schema() -> dict:
    """Return a fresh copy; the constant must never reach a request payload."""
    return copy.deepcopy(_DEFAULT_PROBE_SCHEMA)


class StructuredOutputMode(str, Enum):
    OPENAI_RESPONSE_FORMAT = "openai_response_format"
    ROOT_GUIDED_JSON = "root_guided_json"
    NVEXT_GUIDED_JSON = "nvext_guided_json"
    UNSUPPORTED = "unsupported"


class StructuredOutput(BaseModel):
    name: str | None = None
    json_schema: dict = Field(alias="schema")
    strict: bool = False

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema(cls, value: dict):
        validator = validator_for(value)
        validator.check_schema(value)
        return value


class InferenceStructuredOutput(PreprocessRequest):
    """Format structured output request parameters based on provider mode."""

    def __init__(self, mode: StructuredOutputMode, structured_output: dict, *, resolved: bool = True):
        """Build the hook.

        Args:
            mode: Structured output encoding to emit.
            structured_output: ``{"schema": ..., "strict": ...}`` payload.
            resolved: Whether ``mode`` reflects a real endpoint probe. Callers passing a
                provisional default must set this False so unprobed use is reported.
        """
        if not structured_output:
            raise ValueError("structured_output cannot be empty")
        try:
            output = StructuredOutput(**structured_output)
            self._json_schema = output.json_schema
            self._strict = output.strict
            self.mode = mode
            self._resolved = resolved
            self._warned_unresolved = False
            self.inference_param = self._build_inference_param(mode)
        except SchemaError as e:
            raise ValueError("structured output contains invalid JSON schema") from e

    @property
    def json_schema(self) -> dict:
        return self._json_schema

    @property
    def resolved(self) -> bool:
        """Whether the mode reflects a real endpoint probe rather than a provisional default."""
        return self._resolved

    def set_mode(self, mode: StructuredOutputMode) -> None:
        """Set the encoding, marking it as reflecting a real endpoint probe."""
        self.mode = mode
        self._resolved = True
        self.inference_param = self._build_inference_param(mode)

    def _build_inference_param(self, mode: StructuredOutputMode) -> dict:
        if mode == StructuredOutputMode.OPENAI_RESPONSE_FORMAT:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": self._json_schema,
                        "strict": self._strict,
                    },
                }
            }
        if mode == StructuredOutputMode.ROOT_GUIDED_JSON:
            return {"extra_body": {"guided_json": self._json_schema}}
        if mode == StructuredOutputMode.NVEXT_GUIDED_JSON:
            return {"extra_body": {"nvext": {"guided_json": self._json_schema}}}
        if mode == StructuredOutputMode.UNSUPPORTED:
            return {}
        raise ValueError(f"Unsupported structured output mode: {mode}")

    def _apply_fallback_instruction(self, request: dict) -> dict:
        schema_str = json.dumps(self._json_schema, separators=(",", ":"))
        instruction = f"Return ONLY valid JSON and ensure it matches this JSON schema exactly: {schema_str}"
        if request.get("messages"):
            msg = request["messages"][0]
            if msg.get("role") == "system":
                request["messages"][0]["content"] = f"{instruction} {msg['content']}"
            else:
                request["messages"].insert(0, {"role": "system", "content": instruction})
        elif request.get("prompt"):
            request["prompt"] = f"{instruction} {request['prompt']}"
        return request

    def preprocess(self, request: dict, id: str | None = None) -> dict:
        _ = id  # Required by preprocess hook interface.
        if not self._resolved and not self._warned_unresolved:
            # Once per hook, not per row.
            self._warned_unresolved = True
            _logger.warning(
                "Structured output encoding was never resolved against the endpoint; sending %s as "
                "a guess. Call detect_structured_output_mode() (or "
                "resolve_target_structured_output_mode()) before generating, or the constraint may "
                "be silently dropped.",
                self.mode.value,
            )
        if self.mode == StructuredOutputMode.UNSUPPORTED:
            return self._apply_fallback_instruction(request)
        # Use merge instead of update to avoid overwriting nested dicts
        return deep_merge(request, self.inference_param)


def default_structured_output_mode(format: str | None = None) -> StructuredOutputMode:
    """Return the provisional mode used before preflight probes the endpoint.

    ``format`` is accepted for compatibility and ignored.
    """
    _ = format
    return StructuredOutputMode.OPENAI_RESPONSE_FORMAT


def _looks_like_unsupported_guided_json_error(message: str) -> bool:
    lowered = message.lower()
    signatures = (
        "guided_json is unsupported",
        "unexpected keyword argument 'guided_json'",
        "unexpected keyword argument 'nvext'",
        "extra_forbidden",
        "extra inputs are not permitted",
    )
    if any(sig in lowered for sig in signatures):
        return "guided_json" in lowered or "nvext" in lowered or "extra_body" in lowered
    return False


def _probe_was_truncated(response: dict) -> bool:
    """Return whether the probe ran out of output budget before producing content."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    return choices[0].get("finish_reason") == "length"


def _extract_chat_content(response: dict) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = choices[0].get("message", {})
    content = msg.get("content")
    return content if isinstance(content, str) else None


def _is_probe_valid_json(content: str, probe_schema: dict) -> bool:
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return False
    if not isinstance(obj, dict):
        return False
    try:
        validator = validator_for(probe_schema)
        validator.check_schema(probe_schema)
        validator(probe_schema).validate(obj)
        return True
    except Exception:
        return False


async def detect_structured_output_mode(
    *,
    format: str | None = None,
    model: Model,
    inference_fn: InferenceFn,
    api_key: str | None,
    probe_schema: dict | None = None,
    use_cache: bool = True,
) -> StructuredOutputMode:
    """Detect the structured output mode an endpoint actually honours.

    Every endpoint is probed with the same ordered candidate list regardless of the model's label,
    because support is a property of the endpoint. OpenAI ``response_format`` is tried first, then
    the two ``guided_json`` placements. An endpoint accepting none falls back to a prompt-level
    instruction, which is *not* enforced.

    Results are cached for the enclosing :func:`begin_evaluation_session`. ``format`` is accepted
    for compatibility and ignored.
    """
    _ = format

    cache = session_cache() if use_cache else None
    if cache is None:
        return await _probe_structured_output_mode(
            model=model, inference_fn=inference_fn, api_key=api_key, probe_schema=probe_schema
        )

    # Keyed on the endpoint, not the credential: a session is assumed to be single-principal.
    cache_key = ("structured_output_mode", model.url, model.name)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    async with session_lock():
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        mode = await _probe_structured_output_mode(
            model=model, inference_fn=inference_fn, api_key=api_key, probe_schema=probe_schema
        )
        cache[cache_key] = mode
        return mode


async def _probe_structured_output_mode(
    *,
    model: Model,
    inference_fn: InferenceFn,
    api_key: str | None,
    probe_schema: dict | None,
) -> StructuredOutputMode:
    """Probe an endpoint for the structured output encoding it honours, without consulting cache."""
    probe_schema = _default_probe_schema() if probe_schema is None else probe_schema

    # Must not name the schema: every mode would then pass and enforcement would silently vanish.
    probe_message = "Return ONLY a JSON object that matches the provided schema exactly. No prose or code fences."
    base_request = {
        "messages": [{"role": "user", "content": probe_message}],
        "temperature": 0,
        # Generous because reasoning models spend most of a small budget thinking and return
        # truncated or empty content, which is indistinguishable from an unhonoured encoding.
        "max_tokens": _PROBE_MAX_TOKENS,
    }
    # Deep-copied so `probe_schema` stays unreachable from the request: Pydantic copies only a
    # dict's top level, and `probe_schema` is what the response is validated against.
    try:
        probe = InferenceStructuredOutput(StructuredOutputMode.UNSUPPORTED, {"schema": copy.deepcopy(probe_schema)})
    except ValueError:
        return StructuredOutputMode.UNSUPPORTED

    candidates: list[StructuredOutputMode] = [
        StructuredOutputMode.OPENAI_RESPONSE_FORMAT,
        StructuredOutputMode.ROOT_GUIDED_JSON,
        StructuredOutputMode.NVEXT_GUIDED_JSON,
    ]
    for mode in candidates:
        probe.set_mode(mode)
        try:
            response = await inference_fn(model, {**base_request, **probe.inference_param}, 1, api_key=api_key)
            content = _extract_chat_content(response)
            if content and _is_probe_valid_json(content, probe_schema):
                return mode
            if _probe_was_truncated(response):
                _logger.warning(
                    "Structured output probe for %s hit the %d-token budget, so support for %s "
                    "could not be determined; enforcement may be dropped for this run.",
                    model.name,
                    _PROBE_MAX_TOKENS,
                    mode.value,
                )
        except Exception as e:
            if _looks_like_unsupported_guided_json_error(str(e)):
                continue
            # Probe failures should not abort evaluation startup. If no mode works,
            # caller will fall back to prompt-level strict JSON instruction.
            continue
    return StructuredOutputMode.UNSUPPORTED
