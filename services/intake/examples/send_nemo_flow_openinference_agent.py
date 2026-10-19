#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a minimal NeMo Flow agent and export OpenInference spans over OTLP/HTTP.

Mirrors send_langchain_openinference_agent.py but exercises the NeMo Flow
producer path: a scope with one tool call and one managed LLM call against
an OpenAI-compatible inference endpoint (defaults to NVIDIA Inference API).
The OpenInference subscriber posts the resulting spans directly to Intake's
OTLP ingest endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nvidia/nemotron-3.5-lightning"
SESSION_ID = "nemo-flow-openinference-smoke"
USER_ID = "local-user"
QUESTION = "What is 6 times 7? Use the multiply tool, then summarize."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    base_url = _normalize_base_url(os.getenv("NVIDIA_BASE_URL"))
    model = os.getenv("NVIDIA_MODEL") or DEFAULT_MODEL
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set NVIDIA_API_KEY before running this script. "
            "For a local OpenAI-compatible server that ignores auth, set it to any non-empty value."
        )

    _preflight(args.endpoint)

    from nemo_flow import OpenInferenceConfig, OpenInferenceSubscriber

    config = OpenInferenceConfig()
    config.transport = "http_binary"
    config.endpoint = args.endpoint
    config.service_name = "intake-nemo-flow-openinference-smoke"
    config.service_namespace = "intake"
    config.service_version = "0.1.0"
    config.instrumentation_scope = "nemo-flow-openinference"
    config.timeout_millis = 5_000
    # Resource attributes are retained on raw span attributes. `session.id`
    # becomes the Intake span `session_id` correlation key.
    config.resource_attributes = {
        "deployment.environment": "dev",
        "session.id": SESSION_ID,
        "user.id": USER_ID,
        "run.id": str(uuid.uuid4()),
    }

    subscriber = OpenInferenceSubscriber(config)
    subscriber.register("intake-nemo-flow-openinference")

    try:
        result = asyncio.run(
            _run_agent(
                base_url=base_url,
                api_key=api_key,
                model=model,
                question=QUESTION,
            )
        )
        print(_final_text(result))
        print(f"sent NeMo Flow OpenInference OTLP spans to {args.endpoint}")
        print(f"session_id={SESSION_ID}")
    finally:
        subscriber.force_flush()
        subscriber.shutdown()


async def _run_agent(
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
) -> dict:
    """Deterministic agent loop: tool call → LLM call.

    We don't ask the model to choose tools — keeping the sequence fixed makes
    the resulting span tree predictable so we can compare what NeMo Flow
    emits against what Intake's importer expects.
    """
    import nemo_flow

    async def multiply_tool(args: dict[str, Any]) -> dict[str, Any]:
        a = int(args.get("a", 0))
        b = int(args.get("b", 0))
        return {"a": a, "b": b, "product": a * b}

    async def call_provider(request: nemo_flow.LLMRequest) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": request.content["messages"],
            "temperature": request.content.get("temperature", 0.2),
            "max_tokens": request.content.get("max_tokens", 256),
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    with nemo_flow.scope.scope("agent-run", nemo_flow.ScopeType.Agent) as handle:
        tool_result = await nemo_flow.tools.execute(
            "multiply",
            {"a": 6, "b": 7},
            multiply_tool,
            handle=handle,
        )

        # Fold the tool's result directly into the user message so we don't
        # need a synthetic assistant turn with a tool_call_id to satisfy
        # strict OpenAI-spec gateways. The multiply tool span is still
        # captured separately by NeMo Flow above.
        request = nemo_flow.LLMRequest(
            {},
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a terse calculator that summarizes computed results.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{question}\n\n"
                            f"The multiply tool returned: {json.dumps(tool_result)}. "
                            "Summarize the answer in one short sentence."
                        ),
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 128,
            },
        )

        return await nemo_flow.llm.execute(
            "nvidia-inference",
            request,
            call_provider,
            handle=handle,
            model_name=model,
        )


def _final_text(result: Any) -> str:
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                return content
    return json.dumps(result, default=str)


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        return DEFAULT_BASE_URL
    return base_url.rstrip("/")


def _preflight(endpoint: str) -> None:
    openapi_url = _replace_path(endpoint, "/openapi.json")
    try:
        response = httpx.get(openapi_url, timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Cannot reach Intake at {openapi_url}: {exc}") from exc


def _replace_path(endpoint: str, path: str) -> str:
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


if __name__ == "__main__":
    main()
