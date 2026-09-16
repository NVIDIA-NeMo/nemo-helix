# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat Completions ↔ Switchyard normalized ``LlmRequest`` / ``AggLlmResponse``.

libsy ``run_stream`` and ``ModelCall.respond`` speak Switchyard protocol JSON
(``instructions``, ``outputs``, top-level tool ``name``). IGW and providers speak
OpenAI Chat Completions. RC2 does not expose a Python translator, so the host
does this conversion. This is not OpenAI ↔ Anthropic translation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_OPENAI_CHAT = "openai_chat"


def openai_chat_to_llm_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions body into a libsy ``LlmRequest`` dict."""
    if _looks_like_llm_request(body):
        return deepcopy(body)

    instructions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = _openai_message_content(raw)
        if role in {"system", "developer"}:
            instructions.append({"role": role, "content": content or [{"type": "text", "text": ""}]})
            continue
        messages.append({"role": role, "content": content})

    request: dict[str, Any] = {
        "model": body.get("model"),
        "instructions": instructions,
        "messages": messages,
        "tools": [_openai_tool_to_definition(tool) for tool in body.get("tools") or [] if isinstance(tool, dict)],
        "stream": bool(body.get("stream", False)),
        "sampling": {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
        },
        "output": {
            "max_output_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "response_format": body.get("response_format"),
        },
        "reasoning": {},
        "extensions": {"fields": {}},
        "preservation": {},
    }
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, str) and tool_choice in {"auto", "required", "none"}:
        request["tool_choice"] = {"type": tool_choice}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            request["tool_choice"] = {"type": "tool", "data": {"name": name}}
    return request


def llm_request_to_openai_chat(request: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """Convert a libsy ``LlmRequest`` dict into an OpenAI Chat Completions body."""
    if not _looks_like_llm_request(request):
        body = deepcopy(request)
        if model is not None:
            body["model"] = model
        return body

    messages: list[dict[str, Any]] = []
    for instruction in request.get("instructions") or []:
        if not isinstance(instruction, dict):
            continue
        messages.append(
            {
                "role": instruction.get("role") or "system",
                "content": _blocks_to_openai_content(instruction.get("content") or []),
            }
        )
    for message in request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        messages.append(_llm_message_to_openai(message))

    body: dict[str, Any] = {
        "model": model if model is not None else request.get("model"),
        "messages": messages,
    }
    tools = [_definition_to_openai_tool(tool) for tool in request.get("tools") or [] if isinstance(tool, dict)]
    if tools:
        body["tools"] = tools
    sampling = request.get("sampling") if isinstance(request.get("sampling"), dict) else {}
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    if sampling.get("temperature") is not None:
        body["temperature"] = sampling["temperature"]
    if sampling.get("top_p") is not None:
        body["top_p"] = sampling["top_p"]
    if output.get("max_output_tokens") is not None:
        body["max_tokens"] = output["max_output_tokens"]
    if output.get("response_format") is not None:
        body["response_format"] = output["response_format"]
    if request.get("stream"):
        body["stream"] = True
    return body


def openai_chat_to_agg(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response into a libsy ``AggLlmResponse`` dict."""
    if "outputs" in payload and "choices" not in payload:
        return deepcopy(payload)

    outputs: list[dict[str, Any]] = []
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = _openai_message_content({**message, "role": message.get("role") or "assistant"})
        outputs.append(
            {
                "role": message.get("role") or "assistant",
                "content": content,
                "stop_reason": _stop_reason(choice.get("finish_reason")),
            }
        )
    usage_raw = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "outputs": outputs,
        "usage": {
            "input_tokens": usage_raw.get("prompt_tokens"),
            "output_tokens": usage_raw.get("completion_tokens"),
            "total_tokens": usage_raw.get("total_tokens"),
        },
        "extensions": {"fields": {}},
        "preservation": {},
    }


def apply_llm_request_to_openai_body(openai_body: dict[str, Any], llm_request: dict[str, Any]) -> dict[str, Any]:
    """Overlay routing-time IR rewrites onto the original OpenAI Chat body."""
    rewritten = llm_request_to_openai_chat(llm_request, model=openai_body.get("model"))
    merged = deepcopy(openai_body)
    if rewritten.get("messages"):
        merged["messages"] = rewritten["messages"]
    if "tools" in rewritten:
        merged["tools"] = rewritten["tools"]
    return merged


def _looks_like_llm_request(body: dict[str, Any]) -> bool:
    if "instructions" in body or "outputs" in body:
        return True
    tools = body.get("tools")
    if (
        isinstance(tools, list)
        and tools
        and isinstance(tools[0], dict)
        and "name" in tools[0]
        and "function" not in tools[0]
    ):
        return True
    return False


def _openai_message_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = _content_to_blocks(message.get("content"))
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            text = arguments
            try:
                import json

                parsed: Any = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                parsed = text
            arguments = parsed
        blocks.append(
            {
                "type": "tool_call",
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": arguments if arguments is not None else {},
            }
        )
    if message.get("role") == "tool" and message.get("tool_call_id"):
        blocks = [
            {
                "type": "tool_result",
                "tool_call_id": str(message["tool_call_id"]),
                "content": blocks or [{"type": "text", "text": ""}],
                "is_error": None,
            }
        ]
    return blocks


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "unknown", "provider": _OPENAI_CHAT, "raw": content}]
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": str(item.get("text") or "")})
        elif kind == "image_url":
            image = item.get("image_url") if isinstance(item.get("image_url"), dict) else {}
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "data": {"url": str(image.get("url") or ""), "detail": image.get("detail")},
                    },
                }
            )
        elif kind in {
            "text",
            "reasoning",
            "image",
            "audio",
            "video",
            "file",
            "tool_call",
            "tool_result",
            "refusal",
            "unknown",
        }:
            blocks.append(item)
        else:
            blocks.append({"type": "unknown", "provider": _OPENAI_CHAT, "raw": item})
    return blocks


def _blocks_to_openai_content(blocks: list[Any]) -> str | list[dict[str, Any]]:
    texts: list[str] = []
    rich: list[dict[str, Any]] = []
    only_text = True
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = str(block.get("text") or "")
            texts.append(text)
            rich.append({"type": "text", "text": text})
        elif kind == "image":
            only_text = False
            source = block.get("source") if isinstance(block.get("source"), dict) else {}
            data = source.get("data") if isinstance(source.get("data"), dict) else {}
            rich.append({"type": "image_url", "image_url": {"url": data.get("url"), "detail": data.get("detail")}})
        elif kind == "unknown" and isinstance(block.get("raw"), dict):
            only_text = False
            rich.append(block["raw"])
        elif kind not in {"tool_call", "tool_result"}:
            only_text = False
            rich.append({"type": "text", "text": str(block)})
    if only_text:
        return "".join(texts)
    return rich or "".join(texts)


def _llm_message_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    blocks = [block for block in message.get("content") or [] if isinstance(block, dict)]
    tool_calls = [block for block in blocks if block.get("type") == "tool_call"]
    tool_results = [block for block in blocks if block.get("type") == "tool_result"]
    other = [block for block in blocks if block.get("type") not in {"tool_call", "tool_result"}]
    if tool_results:
        result = tool_results[0]
        inner = result.get("content") if isinstance(result.get("content"), list) else []
        return {
            "role": "tool",
            "tool_call_id": result.get("tool_call_id"),
            "content": _blocks_to_openai_content(inner),
        }
    out: dict[str, Any] = {
        "role": message.get("role") or "user",
        "content": _blocks_to_openai_content(other) if other or not tool_calls else None,
    }
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": call.get("arguments")
                    if isinstance(call.get("arguments"), str)
                    else _json_dumps(call.get("arguments")),
                },
            }
            for call in tool_calls
        ]
        if not other:
            out["content"] = None
    return out


def _openai_tool_to_definition(tool: dict[str, Any]) -> dict[str, Any]:
    if "name" in tool and "function" not in tool:
        return tool
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return {
        "name": function.get("name") or "",
        "description": function.get("description"),
        "parameters": function.get("parameters") if function.get("parameters") is not None else {},
        "strict": function.get("strict"),
    }


def _definition_to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters") if tool.get("parameters") is not None else {},
            "strict": tool.get("strict"),
        },
    }


def _stop_reason(finish_reason: Any) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
    }
    if finish_reason in mapping:
        return mapping[finish_reason]
    return "unknown"


def _json_dumps(value: Any) -> str:
    import json

    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value)
