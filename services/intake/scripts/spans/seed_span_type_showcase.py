#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed Intake with one richly-populated example of every telemetry type.

Purpose: give the Studio team real data for building views that template *per
type*. Where the reference doc
(``services/intake/docs/intake-telemetry-types.md``) describes what each type
*can* carry, this script makes each one exist so the frontend has something to
render.

What it produces (per target workspace):

* **Every span kind** — LLM, CHAIN, TOOL, RETRIEVER, EMBEDDING, AGENT,
  RERANKER, EVALUATOR, GUARDRAIL, UNKNOWN — emitted as OTLP/OpenInference spans.
* **Every span status** — success, error (with ``error_type``/``error_message``),
  cancelled.
* **LLM detail** — token usage details (cache_write / audio / reasoning), cost
  details, prompt id, structured message input/output, and a bucket of raw
  metadata (temperature, top_p, max_tokens, booleans) so the catch-all renders.
* **Trace shapes** — a deep multi-span agent trace, a single-span trace, an
  error trace, and an evaluation-context trace.
* **EvaluatorResult of every data_type** — NUMERIC (LLM-as-judge + comment),
  BOOLEAN (pass/fail code check, no LLM meta), CATEGORICAL, TEXT.
* **Annotation of every kind** — feedback, note, label (text + numeric),
  metadata — at both span and session level.

Spans go in over OTLP (OpenTelemetry SDK -> protobuf). Evaluator results and
annotations go in over their JSON REST endpoints. Span ids are generated
deterministically so re-running is idempotent (ClickHouse de-dupes on span id)
and the evaluator-result / annotation targets stay stable.

Usage::

    uv run services/intake/scripts/spans/seed_span_type_showcase.py \\
        --base-url http://127.0.0.1:8080 --workspace default

    # Seed both the default workspace and a throwaway comparison workspace:
    uv run services/intake/scripts/spans/seed_span_type_showcase.py \\
        --workspace default
    uv run services/intake/scripts/spans/seed_span_type_showcase.py \\
        --workspace old-trace-views

The target workspace must already exist (Intake validates workspace access).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

import httpx
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Span, Status, StatusCode

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_WORKSPACE = "default"
SERVICE_NAME = "intake-type-showcase"
PROJECT = "type-showcase"

MS = 1_000_000  # nanoseconds per millisecond


# ---------------------------------------------------------------------------
# Deterministic ids so re-runs are idempotent (ClickHouse de-dupes on span id).
# ---------------------------------------------------------------------------


class DeterministicIdGenerator(IdGenerator):
    """Generate stable ids from a seed + creation order.

    The SDK calls ``generate_trace_id`` once per root span and
    ``generate_span_id`` per span, in deterministic creation order, so the same
    script run twice produces the same ids. ``int.from_bytes`` results are
    forced non-zero (OTLP rejects all-zero ids).
    """

    def __init__(self, seed: str) -> None:
        self._seed = seed
        self._trace_n = 0
        self._span_n = 0

    def generate_span_id(self) -> int:
        self._span_n += 1
        return self._digest_int(f"span:{self._span_n}", 8)

    def generate_trace_id(self) -> int:
        self._trace_n += 1
        return self._digest_int(f"trace:{self._trace_n}", 16)

    def _digest_int(self, label: str, num_bytes: int) -> int:
        digest = hashlib.sha256(f"{self._seed}:{label}".encode()).digest()
        return int.from_bytes(digest[:num_bytes], "big") or 1


# ---------------------------------------------------------------------------
# Span emission helpers
# ---------------------------------------------------------------------------


class Seeder:
    """Emits spans on a shared clock and collects eval/annotation targets."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self._tracer = tracer
        # Anchor the timeline ~1h in the past so the Studio timeline looks recent.
        self._base_ns = time.time_ns() - 3_600 * 1_000_000_000
        self.eval_results: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str,
        session_id: str,
        start_ms: int,
        duration_ms: int,
        attributes: dict[str, Any] | None = None,
        status: str = "success",
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> Iterator[Span]:
        start_ns = self._base_ns + start_ms * MS
        cm = self._tracer.start_as_current_span(
            name,
            start_time=start_ns,
            end_on_exit=False,
            record_exception=False,
        )
        with cm as span:
            span.set_attribute("openinference.span.kind", kind)
            # Correlation key -> Intake span.session_id; also flows the trace summary.
            span.set_attribute("gen_ai.conversation.id", session_id)
            span.set_attribute("project.name", PROJECT)
            for key, value in (attributes or {}).items():
                span.set_attribute(key, value)
            if error_type is not None:
                span.set_attribute("exception.type", error_type)
            if error_message is not None:
                span.set_attribute("exception.message", error_message)
            if status == "error":
                span.set_status(Status(StatusCode.ERROR, error_message or "error"))
            elif status == "cancelled":
                # Intake derives CANCELLED from this source-only attribute.
                span.set_attribute("status", "cancelled")
            else:
                span.set_status(Status(StatusCode.OK))
            try:
                yield span
            finally:
                span.end(end_time=start_ns + duration_ms * MS)

    @staticmethod
    def span_id_hex(span: Span) -> str:
        # Matches Intake's external_span_id (bytes(span_id).hex(), 8 bytes -> 16 hex chars).
        return format(span.get_span_context().span_id, "016x")

    def add_eval_result(
        self,
        *,
        span: Span,
        session_id: str,
        name: str,
        data_type: str,
        value: float | None = None,
        string_value: str | None = None,
        comment: str | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "span_id": self.span_id_hex(span),
            "session_id": session_id,
            "name": name,
            "data_type": data_type,
        }
        if value is not None:
            body["value"] = value
        if string_value is not None:
            body["string_value"] = string_value
        if comment is not None:
            body["comment"] = comment
        self.eval_results.append(body)

    def add_annotation(self, body: dict[str, Any]) -> None:
        self.annotations.append(body)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten chat messages into OpenInference ``llm.*_messages`` attributes."""
    attrs: dict[str, Any] = {}
    for index, message in enumerate(messages):
        direction = message["_dir"]
        prefix = f"llm.{direction}_messages.{index}.message"
        attrs[f"{prefix}.role"] = message["role"]
        if "content" in message:
            attrs[f"{prefix}.content"] = message["content"]
        for tool_index, tool_call in enumerate(message.get("tool_calls", [])):
            tc_prefix = f"{prefix}.tool_calls.{tool_index}.tool_call"
            attrs[f"{tc_prefix}.function.name"] = tool_call["name"]
            attrs[f"{tc_prefix}.function.arguments"] = json.dumps(tool_call["arguments"])
    return attrs


def seed_agent_trace(seeder: Seeder) -> None:
    """Deep multi-span agent trace: AGENT > GUARDRAIL, CHAIN > (RETRIEVER >
    EMBEDDING, RERANKER), LLM, TOOL, then EVALUATOR + output GUARDRAIL.

    Exercises the hierarchy plus most span kinds in one tree.
    """
    session = "showcase-agent-research"
    with seeder.span(
        "research-agent",
        kind="AGENT",
        session_id=session,
        start_ms=0,
        duration_ms=8400,
        attributes={
            "gen_ai.agent.name": "research-assistant",
            "gen_ai.agent.id": "agent-research-001",
            "gen_ai.agent.version": "2.3.1",
            "input.value": "Summarize the latest on solid-state batteries with citations.",
            "output.value": "Solid-state batteries promise higher density and safety; "
            "key 2026 progress is in sulfide electrolytes. [1][2]",
        },
    ) as agent_span:
        seeder.add_annotation(
            {
                "kind": "feedback",
                "session_id": session,
                "span_id": seeder.span_id_hex(agent_span),
                "value": "positive",
            }
        )
        # Session-level human note (no span_id).
        seeder.add_annotation(
            {"kind": "note", "session_id": session, "text": "Cited sources checked out by a human reviewer."}
        )

        with seeder.span(
            "input-content-safety",
            kind="GUARDRAIL",
            session_id=session,
            start_ms=20,
            duration_ms=140,
            attributes={
                "gen_ai.system": "nvidia",
                "gen_ai.request.model": "llama-3.1-nemoguard-8b-content-safety",
                "guardrail.stage": "input",
                "guardrail.blocked": False,
                "guardrail.categories": json.dumps(["safe"]),
                "guardrail.confidence": 0.99,
                "input.value": "Summarize the latest on solid-state batteries with citations.",
                "output.value": json.dumps({"blocked": False, "categories": ["safe"]}),
            },
        ):
            pass

        with seeder.span(
            "rag-pipeline",
            kind="CHAIN",
            session_id=session,
            start_ms=180,
            duration_ms=4200,
            attributes={
                "input.value": "solid-state battery 2026 progress",
                "output.value": "3 documents retrieved and reranked",
                "chain.step_count": 3,
            },
        ):
            with seeder.span(
                "vector-retrieve",
                kind="RETRIEVER",
                session_id=session,
                start_ms=200,
                duration_ms=900,
                attributes={
                    "input.value": "solid-state battery 2026 progress",
                    # Retrieved docs are not promoted -> land in raw_attributes.
                    "retrieval.documents.0.document.id": "doc-7f2a",
                    "retrieval.documents.0.document.score": 0.88,
                    "retrieval.documents.0.document.content": "Sulfide electrolytes reached 25 Ah cells in Q1 2026.",
                    "retrieval.documents.1.document.id": "doc-3c91",
                    "retrieval.documents.1.document.score": 0.81,
                    "retrieval.documents.1.document.content": "Oxide electrolytes remain limited by interface resistance.",
                    "retrieval.top_k": 8,
                },
            ):
                with seeder.span(
                    "embed-query",
                    kind="EMBEDDING",
                    session_id=session,
                    start_ms=210,
                    duration_ms=120,
                    attributes={
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": "text-embedding-3-large",
                        "gen_ai.usage.input_tokens": 9,
                        "gen_ai.usage.total_tokens": 9,
                        "gen_ai.usage.cost": 0.0000012,
                        "embedding.embeddings.0.embedding.text": "solid-state battery 2026 progress",
                        "embedding.dimensions": 3072,
                        "input.value": "solid-state battery 2026 progress",
                    },
                ):
                    pass

            with seeder.span(
                "cross-encoder-rerank",
                kind="RERANKER",
                session_id=session,
                start_ms=1120,
                duration_ms=260,
                attributes={
                    "gen_ai.request.model": "bge-reranker-v2-m3",
                    "reranker.top_n": 3,
                    "reranker.documents.0.score": 0.94,
                    "reranker.documents.1.score": 0.71,
                    "input.value": "solid-state battery 2026 progress",
                    "output.value": json.dumps([{"id": "doc-7f2a", "score": 0.94}]),
                },
            ):
                pass

        with seeder.span(
            "generate-answer",
            kind="LLM",
            session_id=session,
            start_ms=4400,
            duration_ms=3600,
            attributes={
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-opus-4-7",
                "gen_ai.prompt.id": "rag-answer/v4",
                # Usage incl. detail breakdowns -> usage_details.
                "gen_ai.usage.input_tokens": 1840,
                "gen_ai.usage.output_tokens": 320,
                "gen_ai.usage.cached_tokens": 1200,
                "gen_ai.usage.total_tokens": 2160,
                "llm.token_count.prompt_details.cache_write": 640,
                "llm.token_count.completion_details.reasoning": 180,
                # Costs incl. an extra bucket -> cost_details.
                "gen_ai.usage.cost": 0.0426,
                "llm.cost.prompt": 0.0276,
                "llm.cost.completion": 0.0144,
                "llm.cost.cache_write": 0.0006,
                # Raw metadata bucket: sampling params + booleans.
                "llm.invocation_parameters": json.dumps({"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024}),
                "llm.temperature": 0.2,
                "llm.top_p": 0.9,
                "llm.max_tokens": 1024,
                "llm.is_streaming": True,
                "llm.cache_hit": True,
                **_messages(
                    [
                        {
                            "_dir": "input",
                            "role": "system",
                            "content": "You are a precise research assistant. Cite sources.",
                        },
                        {
                            "_dir": "input",
                            "role": "user",
                            "content": "Summarize solid-state battery progress with citations.",
                        },
                        {
                            "_dir": "output",
                            "role": "assistant",
                            "content": "Sulfide electrolytes drove 2026 gains. [1][2]",
                        },
                    ]
                ),
            },
        ) as llm_span:
            # LLM-as-judge style numeric score + rationale on the generation.
            seeder.add_eval_result(
                span=llm_span,
                session_id=session,
                name="groundedness/v2",
                data_type="NUMERIC",
                value=0.92,
                comment="Every claim maps to a retrieved document; citations are accurate.",
            )
            seeder.add_eval_result(
                span=llm_span,
                session_id=session,
                name="answer_relevance",
                data_type="CATEGORICAL",
                string_value="high",
            )
            seeder.add_annotation(
                {
                    "kind": "label",
                    "session_id": session,
                    "span_id": seeder.span_id_hex(llm_span),
                    "value_type": "numeric",
                    "value": 4,
                    "name": "helpfulness",
                }
            )

        with seeder.span(
            "fetch-citations",
            kind="TOOL",
            session_id=session,
            start_ms=5200,
            duration_ms=480,
            attributes={
                "gen_ai.tool.name": "citation_lookup",
                "tool.description": "Resolve citation ids to source URLs.",
                "gen_ai.tool.call.arguments": json.dumps({"ids": ["doc-7f2a", "doc-3c91"]}),
                "gen_ai.tool.call.result": json.dumps(
                    {"doc-7f2a": "https://example.com/sulfide", "doc-3c91": "https://example.com/oxide"}
                ),
            },
        ) as tool_span:
            seeder.add_annotation(
                {
                    "kind": "label",
                    "session_id": session,
                    "span_id": seeder.span_id_hex(tool_span),
                    "value_type": "text",
                    "value": "verified",
                }
            )

        with seeder.span(
            "judge-answer",
            kind="EVALUATOR",
            session_id=session,
            start_ms=5700,
            duration_ms=1900,
            attributes={
                # LLM-as-judge evaluator: carries model/provider/temperature.
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "evaluator.name": "faithfulness-judge",
                "evaluator.temperature": 0.0,
                "evaluator.score": 0.92,
                "input.value": "Question + answer + retrieved context",
                "output.value": json.dumps({"score": 0.92, "verdict": "faithful"}),
            },
        ) as judge_span:
            seeder.add_eval_result(
                span=judge_span,
                session_id=session,
                name="faithfulness/v1",
                data_type="NUMERIC",
                value=0.92,
                comment="LLM-as-judge: no unsupported claims detected.",
            )

        with seeder.span(
            "output-content-safety",
            kind="GUARDRAIL",
            session_id=session,
            start_ms=7700,
            duration_ms=120,
            attributes={
                "gen_ai.system": "nvidia",
                "gen_ai.request.model": "llama-3.1-nemoguard-8b-content-safety",
                "guardrail.stage": "output",
                "guardrail.blocked": False,
                "guardrail.confidence": 0.98,
                "input.value": "Sulfide electrolytes drove 2026 gains. [1][2]",
                "output.value": json.dumps({"blocked": False}),
            },
        ):
            pass


def seed_single_llm_trace(seeder: Seeder) -> None:
    """Minimal single-span LLM trace (no agent/chain wrapper)."""
    session = "showcase-single-llm"
    with seeder.span(
        "chat-completion",
        kind="LLM",
        session_id=session,
        start_ms=0,
        duration_ms=1250,
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 42,
            "gen_ai.usage.output_tokens": 18,
            "gen_ai.usage.total_tokens": 60,
            "gen_ai.usage.cost": 0.00021,
            "llm.temperature": 0.7,
            "input.value": json.dumps({"messages": [{"role": "user", "content": "What is 6 x 7?"}]}),
            "output.value": json.dumps({"choices": [{"message": {"role": "assistant", "content": "42."}}]}),
        },
    ) as llm_span:
        seeder.add_eval_result(
            span=llm_span,
            session_id=session,
            name="exact_match",
            data_type="BOOLEAN",
            value=1,
        )


def seed_error_trace(seeder: Seeder) -> None:
    """Trace with an erroring LLM span (drives error_type/error_message + error_count)."""
    session = "showcase-error"
    with seeder.span(
        "agent-turn",
        kind="CHAIN",
        session_id=session,
        start_ms=0,
        duration_ms=2200,
        attributes={"input.value": "Generate a 5000-word essay.", "output.value": ""},
    ):
        with seeder.span(
            "llm-call-timeout",
            kind="LLM",
            session_id=session,
            start_ms=100,
            duration_ms=2000,
            status="error",
            error_type="UpstreamTimeoutError",
            error_message="Upstream provider timed out after 60s (request id req_abc123).",
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 5200,
                "llm.temperature": 0.4,
                "input.value": json.dumps({"messages": [{"role": "user", "content": "Generate a 5000-word essay."}]}),
            },
        ):
            pass


def seed_cancelled_trace(seeder: Seeder) -> None:
    """Trace with a cancelled tool span (drives the cancelled status template)."""
    session = "showcase-cancelled"
    with seeder.span(
        "long-running-tool",
        kind="TOOL",
        session_id=session,
        start_ms=0,
        duration_ms=15000,
        status="cancelled",
        attributes={
            "gen_ai.tool.name": "web_crawl",
            "tool.description": "Crawl a site and index pages.",
            "gen_ai.tool.call.arguments": json.dumps({"url": "https://example.com", "max_pages": 5000}),
        },
    ):
        pass


def seed_eval_context_trace(seeder: Seeder) -> None:
    """Trace tagged with the evaluation identity so evaluation_context/experiment_context populate."""
    session = "showcase-eval-run"
    experiment_attrs = {
        "nemo.evaluation.name": "exp-type-showcase",
        "nemo.experiment.run_id": "run-01",
        "nemo.experiment.sha": "a1b2c3d4",
        "nemo.test_case.name": "case-0007",
        "nemo.experiment.metadata": json.dumps({"dataset": "showcase-bench", "split": "test", "seed": 7}),
    }
    with seeder.span(
        "eval-trajectory",
        kind="AGENT",
        session_id=session,
        start_ms=0,
        duration_ms=3000,
        attributes={
            "gen_ai.agent.name": "graded-agent",
            "input.value": "Solve case-0007.",
            "output.value": "Answer for case-0007.",
            **experiment_attrs,
        },
    ):
        with seeder.span(
            "graded-llm-call",
            kind="LLM",
            session_id=session,
            start_ms=100,
            duration_ms=2600,
            attributes={
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-sonnet-4-6",
                "gen_ai.usage.input_tokens": 300,
                "gen_ai.usage.output_tokens": 90,
                "gen_ai.usage.total_tokens": 390,
                "gen_ai.usage.cost": 0.004,
                "input.value": json.dumps({"messages": [{"role": "user", "content": "Solve case-0007."}]}),
                "output.value": json.dumps({"role": "assistant", "content": "Answer for case-0007."}),
                **experiment_attrs,
            },
        ) as graded_span:
            # A deterministic pass/fail code check: BOOLEAN, no LLM meta on the result.
            seeder.add_eval_result(
                span=graded_span,
                session_id=session,
                name="unit_test_pass",
                data_type="BOOLEAN",
                value=1,
            )
            # A free-text critique from a judge: TEXT.
            seeder.add_eval_result(
                span=graded_span,
                session_id=session,
                name="critique",
                data_type="TEXT",
                string_value="Correct answer but missing an edge case for empty input.",
            )
            seeder.add_eval_result(
                span=graded_span,
                session_id=session,
                name="severity",
                data_type="CATEGORICAL",
                string_value="minor",
            )
            seeder.add_annotation(
                {
                    "kind": "metadata",
                    "session_id": session,
                    "span_id": seeder.span_id_hex(graded_span),
                    "metadata": {"reviewer": "qa-team", "ticket": "NEMO-4821", "rerun": False},
                }
            )
            seeder.add_annotation(
                {
                    "kind": "feedback",
                    "session_id": session,
                    "span_id": seeder.span_id_hex(graded_span),
                    "value": "negative",
                }
            )


def seed_code_eval_trace(seeder: Seeder) -> None:
    """EVALUATOR span with NO LLM meta — a deterministic code check (pass/fail)."""
    session = "showcase-code-eval"
    with seeder.span(
        "pytest-verifier",
        kind="EVALUATOR",
        session_id=session,
        start_ms=0,
        duration_ms=540,
        attributes={
            # No model/provider: this is a code-based check, not LLM-as-judge.
            "evaluator.name": "pytest-suite",
            "evaluator.kind": "code",
            "evaluator.passed": True,
            "evaluator.tests_total": 12,
            "evaluator.tests_passed": 12,
            "input.value": "pytest tests/ -q",
            "output.value": "12 passed in 0.54s",
        },
    ) as eval_span:
        seeder.add_eval_result(
            span=eval_span,
            session_id=session,
            name="pytest",
            data_type="BOOLEAN",
            value=1,
            comment="12/12 tests passed.",
        )
        seeder.add_annotation(
            {
                "kind": "label",
                "session_id": session,
                "span_id": seeder.span_id_hex(eval_span),
                "value_type": "text",
                "value": "green",
                "name": "ci_status",
            }
        )


def seed_unknown_trace(seeder: Seeder) -> None:
    """Span whose openinference.span.kind is unrecognized -> normalizes to UNKNOWN."""
    session = "showcase-unknown"
    with seeder.span(
        "mystery-step",
        kind="WIDGET",  # not a known kind -> UNKNOWN
        session_id=session,
        start_ms=0,
        duration_ms=300,
        attributes={
            "input.value": "do the thing",
            "output.value": "did the thing",
            "custom.vendor_attribute": "some-value",
            "custom.count": 3,
        },
    ):
        pass


SCENARIOS = (
    seed_agent_trace,
    seed_single_llm_trace,
    seed_error_trace,
    seed_cancelled_trace,
    seed_eval_context_trace,
    seed_code_eval_trace,
    seed_unknown_trace,
)

# Root-trace session ids each scenario emits — one per SCENARIOS entry. `_verify`
# checks these are all ingested so a silent ingest failure can't pass as success.
EXPECTED_SHOWCASE_SESSIONS = frozenset(
    {
        "showcase-agent-research",
        "showcase-single-llm",
        "showcase-error",
        "showcase-cancelled",
        "showcase-eval-run",
        "showcase-code-eval",
        "showcase-unknown",
    }
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    workspace = args.workspace
    otlp_endpoint = f"{base_url}/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"
    _preflight(base_url, workspace)

    # Deterministic ids keyed by workspace so each workspace gets its own stable set.
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME, "project.name": PROJECT}),
        id_generator=DeterministicIdGenerator(seed=f"type-showcase:{workspace}"),
    )
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    tracer = provider.get_tracer("nhx.intake.spans.type_showcase")

    seeder = Seeder(tracer)
    print(f"=== Seeding span-type showcase into workspace '{workspace}' ===")
    for scenario in SCENARIOS:
        scenario(seeder)
        print(f"  [trace] {scenario.__name__}")
    provider.force_flush()
    provider.shutdown()
    print(f"emitted {len(SCENARIOS)} traces (spans) over OTLP")

    with httpx.Client(timeout=10.0) as client:
        _post_eval_results(client, base_url, workspace, seeder.eval_results)
        _post_annotations(client, base_url, workspace, seeder.annotations)

    _verify(base_url, workspace)
    print("=== Done ===")


def _post_eval_results(client: httpx.Client, base_url: str, workspace: str, results: list[dict[str, Any]]) -> None:
    url = f"{base_url}/apis/intake/v2/workspaces/{workspace}/evaluator-results"
    for body in results:
        response = client.post(url, json=body)
        response.raise_for_status()
    print(f"posted {len(results)} evaluator results")


def _post_annotations(client: httpx.Client, base_url: str, workspace: str, annotations: list[dict[str, Any]]) -> None:
    url = f"{base_url}/apis/intake/v2/workspaces/{workspace}/annotations"
    # The server assigns a fresh ann-{uuid} per POST, so re-running would stack
    # duplicate feedback/labels/notes. Clear existing annotations on the showcase
    # sessions first to keep reruns idempotent.
    sessions = {body["session_id"] for body in annotations if body.get("session_id")}
    deleted = 0
    for session_id in sorted(sessions):
        existing = client.get(url, params={"filter[session_id]": session_id, "page_size": 1000})
        existing.raise_for_status()
        for annotation in existing.json().get("data", []):
            annotation_id = annotation.get("annotation_id")
            if annotation_id is None:
                continue
            response = client.delete(f"{url}/{annotation_id}")
            response.raise_for_status()
            deleted += 1
    for body in annotations:
        response = client.post(url, json=body)
        response.raise_for_status()
    print(f"posted {len(annotations)} annotations (cleared {deleted} existing)")


def _preflight(base_url: str, workspace: str) -> None:
    try:
        response = httpx.get(_replace_path(base_url, "/openapi.json"), timeout=3.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Cannot reach NeMo Helix at {base_url}: {exc}") from exc
    # Confirm the workspace exists (Intake validates workspace access on ingest).
    probe = httpx.get(
        f"{base_url}/apis/intake/v2/workspaces/{workspace}/traces",
        params={"page_size": 1},
        timeout=5.0,
    )
    if probe.status_code == 404:
        raise SystemExit(
            f"Workspace '{workspace}' not found. Create it first, e.g.\n"
            f"  curl -s -X POST {base_url}/apis/entities/v2/workspaces "
            f'-H "content-type: application/json" -d \'{{"name":"{workspace}"}}\''
        )
    probe.raise_for_status()


def _verify(base_url: str, workspace: str) -> None:
    response = httpx.get(
        f"{base_url}/apis/intake/v2/workspaces/{workspace}/traces",
        params={"page_size": 100},
        timeout=10.0,
    )
    response.raise_for_status()
    sessions = {trace.get("session_id") for trace in response.json().get("data", [])}
    showcase = sorted(s for s in sessions if isinstance(s, str) and s.startswith("showcase-"))
    missing = sorted(EXPECTED_SHOWCASE_SESSIONS - sessions)
    if missing:
        raise SystemExit(
            "Verification failed: expected showcase trace session(s) not ingested: "
            f"{', '.join(missing)}. Visible showcase session(s): {', '.join(showcase) or '(none)'}"
        )
    print(f"verified {len(showcase)} showcase trace session(s) visible: {', '.join(showcase)}")


def _replace_path(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


if __name__ == "__main__":
    main()
