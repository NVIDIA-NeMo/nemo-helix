# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a small OTLP/HTTP trace to the Intake spans POC endpoint."""

from __future__ import annotations

import argparse
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    _preflight(args.endpoint)
    provider = TracerProvider(resource=Resource.create({"service.name": "intake-spans-smoke"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=args.endpoint)))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("nhx.intake.spans.sample")

    with tracer.start_as_current_span("sample-chain") as chain:
        chain.set_attribute("openinference.span.kind", "CHAIN")
        chain.set_attribute("gen_ai.conversation.id", "sample-session")
        with tracer.start_as_current_span("sample-llm") as llm:
            llm.set_attribute("openinference.span.kind", "LLM")
            llm.set_attribute("gen_ai.conversation.id", "sample-session")
            llm.set_attribute("gen_ai.system", "openai")
            llm.set_attribute("gen_ai.request.model", "gpt-4o-mini")
            llm.set_attribute("gen_ai.usage.input_tokens", 12)
            llm.set_attribute("gen_ai.usage.output_tokens", 8)
            llm.set_attribute("gen_ai.usage.total_tokens", 20)
            llm.set_attribute("input.value", '{"messages":[{"role":"user","content":"hello"}]}')
            llm.set_attribute("output.value", '{"choices":[{"message":{"role":"assistant","content":"hi"}}]}')

    provider.force_flush()
    provider.shutdown()
    _verify_sample_span(args.endpoint)
    print(f"sent OTLP sample to {args.endpoint}")


def _preflight(endpoint: str) -> None:
    openapi_url = _replace_path(endpoint, "/openapi.json")
    try:
        response = httpx.get(openapi_url, timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Cannot reach Intake at {openapi_url}: {exc}") from exc


def _verify_sample_span(endpoint: str) -> None:
    spans_url = _spans_url(endpoint)
    last_response: httpx.Response | None = None
    for _ in range(10):
        response = httpx.get(spans_url, timeout=5.0)
        last_response = response
        if response.status_code >= 400:
            raise SystemExit(f"Intake stored-span check failed: {response.status_code} {response.text}")

        spans = response.json().get("data", [])
        if any(span.get("session_id") == "sample-session" for span in spans):
            return
        time.sleep(0.5)

    detail = last_response.text if last_response is not None else "<no response>"
    raise SystemExit(f"OTLP export completed, but sample-session spans were not visible at {spans_url}: {detail}")


def _spans_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    suffix = "/ingest/otlp/v1/traces"
    if not parts.path.endswith(suffix):
        raise SystemExit(f"Endpoint must end with {suffix!r} to verify the sample span.")
    query = urlencode([*parse_qsl(parts.query), ("filter[source]", "otel"), ("filter[session_id]", "sample-session")])
    return urlunsplit((parts.scheme, parts.netloc, parts.path[: -len(suffix)] + "/spans", query, ""))


def _replace_path(endpoint: str, path: str) -> str:
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


if __name__ == "__main__":
    main()
