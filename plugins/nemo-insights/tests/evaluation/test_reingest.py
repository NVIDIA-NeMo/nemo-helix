# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for evaluation.reingest (doc->OTLP inversion, idempotency guard, round-trip diff).

Golden docs are real detailed span docs captured from the live dev platform
during the 2026-07-06 spike (workspace smoke-20260626-121437-5559-oracle),
verbatim except where a test says it tweaked a field. No network anywhere.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from evaluation import reingest

# --- stub catalog: mirrors the real span_attribute_catalog surface the inversion reads ---


@dataclass(frozen=True)
class _Field:
    value: str


@dataclass(frozen=True)
class _Spec:
    field: _Field
    source_keys: tuple[str, ...]


class _StubCatalog:
    ATTRIBUTE_SPECS = (
        _Spec(_Field("model"), ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")),
        _Spec(_Field("agent_name"), ("gen_ai.agent.name", "llm.agent.name", "agent.name")),
        _Spec(_Field("agent_version"), ("gen_ai.agent.version", "agent.version")),
        _Spec(_Field("evaluation_name"), ("nemo.evaluation.name", "nemo.experiment.id")),
        _Spec(_Field("test_case_name"), ("nemo.test_case.name", "nemo.test_case.id")),
        _Spec(_Field("input_tokens"), ("gen_ai.usage.input_tokens", "llm.token_count.prompt")),
        _Spec(_Field("prompt_cache_write_tokens"), ("llm.token_count.prompt_details.cache_write",)),
        _Spec(_Field("cost_total_usd"), ("gen_ai.usage.cost", "llm.cost.total")),
    )


CATALOG = _StubCatalog()

# --- golden docs (live spike capture) ---

AGENT_DOC = {
    "ingested_at": "2026-06-26T18:14:41.412889",
    "kind": "AGENT",
    "session_id": "smoke-20260626-121437-5559-task1-t0",
    "source": "otel",
    "span_id": "72f937bfcb667a9e",
    "started_at": "2026-06-26T18:14:41.406179",
    "status": "success",
    "workspace": "smoke-20260626-121437-5559-oracle",
    "agent_name": "smoke-20260626-121437-5559",
    "cost_details": {},
    "ended_at": "2026-06-26T18:14:41.408179",
    "evaluation_context": {
        "evaluation_name": "smoke-20260626-121437-5559-20260626-121438-a833",
        "metadata": {},
        "test_case_name": "1",
    },
    "input": "do 1",
    "name": "smoke-20260626-121437-5559",
    "output": "ok",
    "raw_attributes": json.dumps(
        {
            "service.name": "nemo-insights-evaluation",
            "openinference.span.kind": "AGENT",
            "tau2.task": '{"description": {"purpose": "p"}}',
            "otel.scope": '{"name":"evaluation","version":"1.0.0"}',
        }
    ),
    "trace_id": "c9f0d45dfb4defbf92958b76fa367354",
    "usage_details": {},
}

LLM_DOC = {
    "ingested_at": "2026-06-26T18:14:41.412889",
    "kind": "LLM",
    "session_id": "smoke-20260626-121437-5559-task1-t0",
    "source": "otel",
    "span_id": "76cc9cc7b048160a",
    "started_at": "2026-06-26T18:14:41.407179",
    "status": "success",
    "workspace": "smoke-20260626-121437-5559-oracle",
    "cost_details": {},
    "ended_at": "2026-06-26T18:14:41.408179",
    "evaluation_context": {
        "evaluation_name": "smoke-20260626-121437-5559-20260626-121438-a833",
        "metadata": {},
        "test_case_name": "1",
    },
    "model": "m",
    "name": "agent-2",
    "output": '{"content": "ok", "tool_calls": []}',
    "parent_span_id": "72f937bfcb667a9e",
    "raw_attributes": json.dumps(
        {
            "service.name": "nemo-insights-evaluation",
            "openinference.span.kind": "LLM",
            "output.mime_type": "application/json",
            "otel.scope": '{"name":"evaluation","version":"1.0.0"}',
        }
    ),
    "trace_id": "c9f0d45dfb4defbf92958b76fa367354",
    "usage_details": {},
}


# --- doc_to_otlp ---


def test_agent_doc_golden():
    otlp = reingest.doc_to_otlp(AGENT_DOC, CATALOG)
    assert otlp["trace_id"] == "c9f0d45dfb4defbf92958b76fa367354"
    assert otlp["span_id"] == "72f937bfcb667a9e"
    assert otlp["parent_span_id"] is None
    assert otlp["name"] == "smoke-20260626-121437-5559"
    assert otlp["start_ns"] == 1782497681406179000
    assert otlp["end_ns"] == 1782497681408179000
    assert otlp["status_error"] is False
    assert otlp["scope"] == {"name": "evaluation", "version": "1.0.0"}
    assert otlp["attributes"] == {
        # raw passthrough (otel.scope hoisted out — ingest re-stamps it from the protobuf scope)
        "service.name": "nemo-insights-evaluation",
        "openinference.span.kind": "AGENT",
        "tau2.task": '{"description": {"purpose": "p"}}',
        # source-only protocol fields, synthesized from doc columns
        "session.id": "smoke-20260626-121437-5559-task1-t0",
        "input.value": "do 1",
        "output.value": "ok",
        # catalog inversion: semantic columns re-emitted under their top-precedence source key
        "gen_ai.agent.name": "smoke-20260626-121437-5559",
        "nemo.evaluation.name": "smoke-20260626-121437-5559-20260626-121438-a833",
        "nemo.test_case.name": "1",
    }


def test_llm_doc_inverts_model_and_sets_parent():
    otlp = reingest.doc_to_otlp(LLM_DOC, CATALOG)
    assert otlp["parent_span_id"] == "72f937bfcb667a9e"
    assert otlp["attributes"]["gen_ai.request.model"] == "m"
    assert otlp["attributes"]["output.value"] == '{"content": "ok", "tool_calls": []}'
    assert "input.value" not in otlp["attributes"]  # doc had no input


def test_atif_doc_inverts_to_exact_direct_span():
    doc = {
        **AGENT_DOC,
        "source": "atif",
        "span_id": "span-abc123",
        "trace_id": "campaign-session",
        "session_id": "campaign-session",
    }

    source, direct = reingest.doc_to_direct_span(doc, CATALOG)

    assert source == "atif"
    assert direct["span_id"] == "span-abc123"
    assert direct["trace_id"] == "campaign-session"
    assert direct["started_at"].endswith("+00:00")
    assert direct["input"] == "do 1"
    assert direct["output"] == "ok"
    assert direct["attributes"]["otel.scope"] == '{"name":"evaluation","version":"1.0.0"}'
    assert direct["attributes"]["nemo.evaluation.name"] == "smoke-20260626-121437-5559-20260626-121438-a833"


def test_raw_alias_wins_over_inversion():
    doc = dict(LLM_DOC)
    doc["raw_attributes"] = json.dumps({"llm.model_name": "other"})
    otlp = reingest.doc_to_otlp(doc, CATALOG)
    # the passthrough alias already derives `model` at ingest — the inversion must not double-write
    assert "gen_ai.request.model" not in otlp["attributes"]
    assert otlp["attributes"]["llm.model_name"] == "other"


def test_error_and_cancelled_status():
    error_doc = {**AGENT_DOC, "status": "error"}
    assert reingest.doc_to_otlp(error_doc, CATALOG)["status_error"] is True
    cancelled_doc = {**AGENT_DOC, "status": "cancelled"}
    otlp = reingest.doc_to_otlp(cancelled_doc, CATALOG)
    assert otlp["status_error"] is False
    assert otlp["attributes"]["status"] == "cancelled"  # source-only key; the OTLP route to a cancelled row


def test_kind_synthesized_only_when_missing():
    doc = {**AGENT_DOC, "raw_attributes": json.dumps({})}
    assert reingest.doc_to_otlp(doc, CATALOG)["attributes"]["openinference.span.kind"] == "AGENT"
    unknown = {**doc, "kind": "UNKNOWN"}
    assert "openinference.span.kind" not in reingest.doc_to_otlp(unknown, CATALOG)["attributes"]


def test_evaluation_metadata_and_usage_details_invert():
    doc = {
        **LLM_DOC,
        "evaluation_context": {"evaluation_id": "e1", "metadata": {"num_trials": 2}},
        "usage_details": {"prompt_details.cache_write": 7},
        "input_tokens": 11,
    }
    attrs = reingest.doc_to_otlp(doc, CATALOG)["attributes"]
    assert attrs["nemo.experiment.metadata"] == '{"num_trials":2}'
    assert attrs["llm.token_count.prompt_details.cache_write"] == 7
    assert attrs["gen_ai.usage.input_tokens"] == 11


def test_deprecated_evaluation_context_fields_still_invert():
    doc = {
        **AGENT_DOC,
        "evaluation_context": {"evaluation_id": "legacy-evaluation", "test_case_id": "legacy-case"},
    }
    attrs = reingest.doc_to_otlp(doc, CATALOG)["attributes"]
    assert attrs["nemo.evaluation.name"] == "legacy-evaluation"
    assert attrs["nemo.test_case.name"] == "legacy-case"


def test_missing_trace_id_is_an_error():
    doc = {k: v for k, v in AGENT_DOC.items() if k != "trace_id"}
    with pytest.raises(ValueError, match="no trace_id"):
        reingest.doc_to_otlp(doc, CATALOG)


def test_iso_to_ns_treats_naive_as_utc():
    assert reingest._iso_to_ns("2026-06-26T18:14:41.406179") == 1782497681406179000
    assert reingest._iso_to_ns("2026-06-26T18:14:41.406179+00:00") == 1782497681406179000


# --- build_trace_request ---


def test_build_trace_requests_respects_serialized_size(monkeypatch):
    """ByteSize() drives batching: a small limit splits spans across requests."""
    monkeypatch.setattr(reingest, "OTLP_REQUEST_MAX_BYTES", 512)
    docs = [{**AGENT_DOC, "span_id": f"{i:016x}"} for i in range(5)]
    requests = reingest.build_trace_requests(docs, CATALOG)
    assert len(requests) > 1
    for request in requests:
        assert request.ByteSize() <= 512
        assert len(request.resource_spans[0].scope_spans[0].spans) >= 1


def test_build_trace_requests_accepts_exact_serialized_size_limit():
    one_span_size = reingest.build_trace_request([reingest.doc_to_otlp(AGENT_DOC, CATALOG)]).ByteSize()

    requests = reingest.build_trace_requests([AGENT_DOC], CATALOG, max_bytes=one_span_size)

    assert len(requests) == 1
    assert requests[0].ByteSize() == one_span_size
    with pytest.raises(RuntimeError, match=rf"exceeds {one_span_size - 1} bytes"):
        reingest.build_trace_requests([AGENT_DOC], CATALOG, max_bytes=one_span_size - 1)


def test_build_trace_requests_respects_span_count_limit(monkeypatch):
    monkeypatch.setattr(reingest, "OTLP_REQUEST_MAX_SPANS", 2)
    docs = [{**AGENT_DOC, "span_id": f"{i:016x}"} for i in range(5)]
    requests = reingest.build_trace_requests(docs, CATALOG)
    sizes = [len(req.resource_spans[0].scope_spans[0].spans) for req in requests]
    assert sizes == [2, 2, 1]


def test_build_trace_requests_rejects_oversized_single_span():
    huge = {**AGENT_DOC, "raw_attributes": json.dumps({"payload": "x" * (5 * 1024 * 1024)})}
    with pytest.raises(RuntimeError, match="exceeds"):
        reingest.build_trace_requests([huge], CATALOG)


def test_ingest_bundle_splits_on_serialized_size(tmp_path, quiet_platform, monkeypatch):
    """Small byte limit forces multiple OTLP posts even when span count is low."""
    monkeypatch.setattr(reingest, "OTLP_REQUEST_MAX_BYTES", 512)
    docs = [{**AGENT_DOC, "span_id": f"{i:016x}"} for i in range(5)]
    export_dir = _write_export(tmp_path, "ws-a", docs)
    quiet_platform["span_counts"] = [0, 5]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 5),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert len(quiet_platform["requests"]) > 1


def test_ingest_bundle_oversized_span_raises_before_post(tmp_path, quiet_platform):
    """One span above the byte limit must fail before the first export_trace_request call."""
    huge = {**AGENT_DOC, "raw_attributes": json.dumps({"payload": "x" * (5 * 1024 * 1024)})}
    export_dir = _write_export(tmp_path, "ws-a", [huge])
    quiet_platform["span_counts"] = [0]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    with pytest.raises(RuntimeError, match="exceeds"):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 1),
            workspace_map={"ws-a": "ws-b"},
            catalog=CATALOG,
        )
    assert quiet_platform["requests"] == []


def test_build_request_groups_by_scope_and_encodes_protocol_fields():
    spans = [reingest.doc_to_otlp(AGENT_DOC, CATALOG), reingest.doc_to_otlp(LLM_DOC, CATALOG)]
    spans.append({**spans[1], "span_id": "aabbccdd11223344", "scope": None, "status_error": True})
    request = reingest.build_trace_request(spans)
    assert len(request.resource_spans) == 1
    assert not request.resource_spans[0].resource.attributes  # resource layer rides in the raw passthrough
    scopes = request.resource_spans[0].scope_spans
    assert len(scopes) == 2  # evaluation scope + scope-less
    named = next(s for s in scopes if s.scope.name == "evaluation")
    assert named.scope.version == "1.0.0"
    assert len(named.spans) == 2
    root, child = named.spans
    assert root.trace_id == bytes.fromhex("c9f0d45dfb4defbf92958b76fa367354")
    assert root.parent_span_id == b""  # unset for roots — all-zero parents are rejected
    assert child.parent_span_id == bytes.fromhex("72f937bfcb667a9e")
    assert root.start_time_unix_nano == 1782497681406179000
    assert root.status.code == 0
    scopeless = next(s for s in scopes if s is not named)
    assert scopeless.spans[0].status.code == 2


# --- catalog loading ---


def test_load_catalog_from_checkout_layout(tmp_path):
    path = tmp_path / reingest.CATALOG_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class AttributeSpec:\n"
        "    field: str\n"
        "    source_keys: tuple[str, ...]\n"
        "ATTRIBUTE_SPECS = (AttributeSpec('model', ('gen_ai.request.model',)),)\n",
        encoding="utf-8",
    )
    catalog = reingest.load_catalog(tmp_path)
    # dataclass field resolution requires the module in sys.modules — this is the regression test
    assert catalog.ATTRIBUTE_SPECS[0].source_keys == ("gen_ai.request.model",)


def test_load_catalog_missing_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="nemo-helix checkout"):
        reingest.load_catalog(tmp_path / "nowhere")


# --- workspace collection counts (the per-collection idempotency guard reads these) ---


def _counting_client(seen: list, total: int = 7) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"pagination": {"total_results": total}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_annotation_count_reads_workspace_total_with_epoch_bound():
    seen: list[httpx.Request] = []
    with _counting_client(seen) as client:
        assert reingest.annotation_count("http://x/", "ws-a", client=client) == 7
    assert seen[0].url.path == "/apis/intake/v2/workspaces/ws-a/annotations"
    # explicit epoch bound: the read API injects a 30-day default lookback without one
    assert seen[0].url.params["filter[created_at][gte]"] == "1970-01-01T00:00:00Z"
    assert seen[0].url.params["page_size"] == "1"  # the total rides on pagination, not the page


def test_evaluator_result_count_reads_workspace_total_with_epoch_bound():
    seen: list[httpx.Request] = []
    with _counting_client(seen) as client:
        assert reingest.evaluator_result_count("http://x", "ws-a", client=client) == 7
    assert seen[0].url.path == "/apis/intake/v2/workspaces/ws-a/evaluator-results"
    assert seen[0].url.params["filter[created_at][gte]"] == "1970-01-01T00:00:00Z"


def test_collection_count_non_200_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="nope")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="count failed for ws-a"):
            reingest.annotation_count("http://x", "ws-a", client=client)


# --- ingest_bundle ---


def _write_export(tmp_path: Path, workspace: str, spans: list[dict], annotations=(), results=()) -> Path:
    ws_dir = tmp_path / "export" / workspace
    ws_dir.mkdir(parents=True)
    for name, docs in (("spans", spans), ("annotations", annotations), ("evaluator_results", results)):
        (ws_dir / f"{name}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return tmp_path / "export"


def _manifest(workspace: str, spans: int, annotations: int = 0, results: int = 0) -> dict:
    return {
        "workspaces": [workspace],
        "counts": {workspace: {"spans": spans, "annotations": annotations, "evaluator_results": results}},
        "min_start_time": datetime.now(timezone.utc).isoformat(),
    }


ANNOTATION_DOC = {
    "annotation_id": "ann-997a55ec332c4e9c9af8a78ea38e0437",
    "created_at": "2026-07-06T22:04:47.443000",
    "ingested_at": "2026-07-06T22:04:47.443000",
    "kind": "note",
    "session_id": "smoke-20260626-121437-5559-task1-t0",
    "text": "spike note",
    "workspace": "smoke-20260626-121437-5559-oracle",
}

RESULT_DOC = {
    "created_at": "2026-06-26T18:14:41.660000",
    "data_type": "NUMERIC",
    "evaluator_result_id": "eval-2b40eb397ed3ea04abc9c942d2eb42a5",
    "ingested_at": "2026-06-26T18:14:41.660000",
    "name": "reward",
    "session_id": "smoke-20260626-121437-5559-task2-t0",
    "span_id": "671629150455e141",
    "workspace": "smoke-20260626-121437-5559-oracle",
    "value": 0.0,
}


@pytest.fixture
def quiet_platform(monkeypatch):
    """Stub every network touchpoint; record what ingest_bundle would send.

    Tests preload one count sequence per collection (``span_counts`` /
    ``annotation_counts`` / ``result_counts``); each stubbed count helper pops
    its next value, so an unexpected count call fails loudly (IndexError).
    """
    calls = {
        "requests": [],
        "posts": [],
        "ensured": [],
        "counts": [],
        "events": [],
        "span_counts": [],
        "annotation_counts": [],
        "result_counts": [],
        "first_ids": [],
    }

    def _counter(name):
        def fake(base_url, workspace, *, client=None):
            calls["counts"].append((name, workspace))
            calls["events"].append(("count", name, workspace))
            return calls[f"{name}_counts"].pop(0)

        return fake

    def fake_first(base_url, workspace, *, client=None):
        calls["counts"].append(("first_span", workspace))
        calls["events"].append(("count", "first_span", workspace))
        # default (no preloaded id): probe unavailable -> count-only fallback
        return calls["first_ids"].pop(0) if calls["first_ids"] else None

    def fake_ensure(url, workspace, *, client=None):
        calls["ensured"].append(workspace)
        calls["events"].append(("ensure", workspace))

    def fake_export(url, workspace, request, *, client=None):
        calls["requests"].append((workspace, request))
        calls["events"].append(("write", "span", workspace))

    def fake_post(client, url, body):
        calls["posts"].append((url, body))
        endpoint = "annotation" if url.endswith("/annotations") else "result"
        workspace = url.split("/workspaces/", 1)[1].split("/", 1)[0]
        calls["events"].append(("write", endpoint, workspace))

    real_client = httpx.Client

    def reject_unexpected_request(request: httpx.Request) -> httpx.Response:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            calls["events"].append(("write", request.method, request.url.path))
            raise AssertionError(f"unexpected {request.method} mutation: {request.url}")
        raise AssertionError(f"unexpected HTTP request: {request.method} {request.url}")

    transport = httpx.MockTransport(reject_unexpected_request)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(reingest, "span_count", _counter("span"))
    monkeypatch.setattr(reingest, "annotation_count", _counter("annotation"))
    monkeypatch.setattr(reingest, "evaluator_result_count", _counter("result"))
    monkeypatch.setattr(reingest, "_first_span_id", fake_first, raising=False)
    monkeypatch.setattr(reingest, "ensure_workspace", fake_ensure)
    monkeypatch.setattr(reingest, "export_trace_request", fake_export)
    monkeypatch.setattr(reingest, "_post_created", fake_post)
    monkeypatch.setattr(reingest.httpx, "Client", client_factory)
    return calls


@pytest.fixture
def fake_docker(monkeypatch):
    """Record docker invocations instead of running them; rc=0 unless a test overrides."""
    runs: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        runs.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(reingest.subprocess, "run", fake_run)
    return runs


@pytest.mark.parametrize(
    ("counts_key", "message"),
    [
        ("span_counts", "spans"),
        ("annotation_counts", "annotations"),
        ("result_counts", "evaluator results"),
    ],
)
def test_require_empty_rejects_existing_target_data(tmp_path, quiet_platform, counts_key: str, message: str) -> None:
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [0]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    quiet_platform[counts_key] = [1]

    with pytest.raises(RuntimeError, match=message):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 1, 1, 1),
            workspace_map={"ws-a": "target"},
            catalog=CATALOG,
            require_empty=True,
        )

    assert quiet_platform["events"] == [
        ("ensure", "target"),
        ("count", "span", "target"),
        ("count", "annotation", "target"),
        ("count", "result", "target"),
    ]
    assert quiet_platform["requests"] == []
    assert quiet_platform["posts"] == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_require_empty_transport_rejects_all_http_mutations(quiet_platform, method: str) -> None:
    with reingest.httpx.Client() as client:
        with pytest.raises(AssertionError, match=f"unexpected {method} mutation"):
            client.request(method, "http://x/hidden-mutation")

    assert quiet_platform["events"] == [("write", method, "/hidden-mutation")]


def test_require_empty_direct_restore_uses_real_http_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    counts = {"spans": 0, "annotations": 0, "evaluator-results": 0}
    requests: list[tuple[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append((request.method, path))
        if request.method == "POST" and path == "/apis/entities/v2/workspaces":
            return httpx.Response(409)
        if request.method == "GET":
            collection = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"pagination": {"total_results": counts[collection]}, "data": []})
        if request.method == "POST" and path.endswith("/ingest/otlp/v1/traces"):
            counts["spans"] = 1
            return httpx.Response(200, json={"errors": []})
        if request.method == "POST" and path.endswith("/annotations"):
            counts["annotations"] = 1
            return httpx.Response(201, json={})
        if request.method == "POST" and path.endswith("/evaluator-results"):
            counts["evaluator-results"] = 1
            return httpx.Response(201, json={})
        raise AssertionError(f"unexpected request: {request.method} {path}")

    real_client = httpx.Client
    transport = httpx.MockTransport(handle)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(reingest.httpx, "Client", client_factory)

    outcome = reingest.ingest_bundle(
        "http://platform.example",
        export_dir,
        _manifest("ws-a", 1, 1, 1),
        workspace_map={"ws-a": "target"},
        catalog=CATALOG,
        require_empty=True,
        sleep=lambda seconds: None,
    )

    assert [request for request in requests if request[0] != "GET"] == [
        ("POST", "/apis/entities/v2/workspaces"),
        ("POST", "/apis/intake/v2/workspaces/target/ingest/otlp/v1/traces"),
        ("POST", "/apis/intake/v2/workspaces/target/annotations"),
        ("POST", "/apis/intake/v2/workspaces/target/evaluator-results"),
    ]
    assert outcome["ws-a"] == {
        "workspace": "target",
        "spans": {"ingested": 1, "skipped": 0},
        "annotations": {"ingested": 1, "skipped": 0},
        "evaluator_results": {"ingested": 1, "skipped": 0},
    }


def test_require_empty_ingests_all_nonempty_collections_in_order(tmp_path, quiet_platform) -> None:
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [0, 1]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]

    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 1, 1, 1),
        workspace_map={"ws-a": "target"},
        catalog=CATALOG,
        require_empty=True,
        sleep=lambda seconds: None,
    )

    assert quiet_platform["events"] == [
        ("ensure", "target"),
        ("count", "span", "target"),
        ("count", "annotation", "target"),
        ("count", "result", "target"),
        ("write", "span", "target"),
        ("count", "span", "target"),
        ("write", "annotation", "target"),
        ("write", "result", "target"),
    ]
    assert len(quiet_platform["requests"]) == 1
    assert [url.rsplit("/", 1)[-1] for url, _ in quiet_platform["posts"]] == [
        "annotations",
        "evaluator-results",
    ]
    assert outcome["ws-a"] == {
        "workspace": "target",
        "spans": {"ingested": 1, "skipped": 0},
        "annotations": {"ingested": 1, "skipped": 0},
        "evaluator_results": {"ingested": 1, "skipped": 0},
    }


def test_require_empty_all_empty_bundle_has_no_recounts_or_writes(tmp_path, quiet_platform) -> None:
    export_dir = _write_export(tmp_path, "ws-a", [])
    quiet_platform["span_counts"] = [0]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]

    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 0),
        workspace_map={"ws-a": "target"},
        catalog=CATALOG,
        require_empty=True,
    )

    assert quiet_platform["events"] == [
        ("ensure", "target"),
        ("count", "span", "target"),
        ("count", "annotation", "target"),
        ("count", "result", "target"),
    ]
    assert quiet_platform["requests"] == []
    assert quiet_platform["posts"] == []
    assert outcome["ws-a"] == {
        "workspace": "target",
        "spans": {"ingested": 0, "skipped": 0},
        "annotations": {"ingested": 0, "skipped": 0},
        "evaluator_results": {"ingested": 0, "skipped": 0},
    }


def test_ingest_bundle_zero_ingests_everything(tmp_path, quiet_platform):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [0, 2]  # guard sees empty; wait sees all spans
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 1, 1),
        workspace_map={"ws-a": "ws-a-state-v6"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert quiet_platform["ensured"] == ["ws-a-state-v6"]
    assert [ws for ws, _ in quiet_platform["requests"]] == ["ws-a-state-v6"]
    ann_url, ann_body = quiet_platform["posts"][0]
    assert ann_url.endswith("/workspaces/ws-a-state-v6/annotations")
    assert ann_body == {"kind": "note", "session_id": "smoke-20260626-121437-5559-task1-t0", "text": "spike note"}
    res_url, res_body = quiet_platform["posts"][1]
    assert res_url.endswith("/workspaces/ws-a-state-v6/evaluator-results")
    assert res_body == {
        "data_type": "NUMERIC",
        "name": "reward",
        "session_id": "smoke-20260626-121437-5559-task2-t0",
        "span_id": "671629150455e141",
        "value": 0.0,
    }
    assert outcome["ws-a"] == {
        "workspace": "ws-a-state-v6",
        "spans": {"ingested": 2, "skipped": 0},
        "annotations": {"ingested": 1, "skipped": 0},
        "evaluator_results": {"ingested": 1, "skipped": 0},
    }


def test_ingest_bundle_batches_spans(tmp_path, quiet_platform, monkeypatch):
    monkeypatch.setattr(reingest, "OTLP_REQUEST_MAX_SPANS", 2)
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC, {**LLM_DOC, "span_id": "aabbccdd11223344"}])
    quiet_platform["span_counts"] = [0, 3]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 3),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    sizes = [len(req.resource_spans[0].scope_spans[0].spans) for _, req in quiet_platform["requests"]]
    assert sizes == [2, 1]


def test_ingest_bundle_equal_count_skips(tmp_path, quiet_platform, capsys):
    """All three collections at manifest counts -> one 'already restored' skip, zero POSTs."""
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [1]
    quiet_platform["result_counts"] = [1]
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 1, 1),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert quiet_platform["requests"] == []
    assert quiet_platform["posts"] == []  # skipping MUST also skip annotation re-posts (they duplicate)
    out = capsys.readouterr().out
    assert out.count("already restored") == 1
    assert "ws-b: already restored (2 spans) — skipping" in out
    assert outcome["ws-a"] == {
        "workspace": "ws-b",
        "spans": {"ingested": 0, "skipped": 2},
        "annotations": {"ingested": 0, "skipped": 1},
        "evaluator_results": {"ingested": 0, "skipped": 1},
    }


def test_ingest_bundle_partial_count_hard_errors(tmp_path, quiet_platform):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    quiet_platform["span_counts"] = [1]
    with pytest.raises(RuntimeError, match="partially restored"):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 2),
            workspace_map={"ws-a": "ws-b"},
            catalog=CATALOG,
        )
    assert quiet_platform["requests"] == []


def test_ingest_bundle_heals_missing_evaluator_results(tmp_path, quiet_platform, capsys):
    """Interrupted restore (spans landed, evaluator results never posted): self-heals.

    Evaluator-result POSTs upsert on a server-derived stable id, so re-posting
    the full set is safe — the guard must post them instead of skipping the
    workspace (the old span-only guard silently left the fixture incomplete
    forever).
    """
    results = [RESULT_DOC, {**RESULT_DOC, "span_id": "aabbccdd11223344"}]
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [ANNOTATION_DOC], results)
    quiet_platform["span_counts"] = [2]  # spans complete
    quiet_platform["annotation_counts"] = [1]  # annotations complete
    quiet_platform["result_counts"] = [0]  # evaluator results missing
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 1, 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert quiet_platform["requests"] == []  # spans are NOT re-ingested
    urls = [url for url, _ in quiet_platform["posts"]]
    assert len(urls) == 2 and all(url.endswith("/workspaces/ws-b/evaluator-results") for url in urls)
    out = capsys.readouterr().out
    assert "healing evaluator results: posting 2" in out
    assert "already restored" not in out
    assert outcome["ws-a"] == {
        "workspace": "ws-b",
        "spans": {"ingested": 0, "skipped": 2},
        "annotations": {"ingested": 0, "skipped": 1},
        "evaluator_results": {"ingested": 2, "skipped": 0},
    }


def test_ingest_bundle_heals_partial_evaluator_results_by_reposting_all(tmp_path, quiet_platform, capsys):
    """Fewer evaluator results than the manifest -> re-post ALL of them (upsert-safe)."""
    results = [RESULT_DOC, {**RESULT_DOC, "span_id": "aabbccdd11223344"}]
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [], results)
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [1]  # one of two landed before the interrupt
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 0, 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert len(quiet_platform["posts"]) == 2  # all, not just the missing one
    assert "healing evaluator results: posting 2" in capsys.readouterr().out
    assert outcome["ws-a"]["evaluator_results"] == {"ingested": 2, "skipped": 0}


def test_ingest_bundle_surplus_evaluator_results_hard_error(tmp_path, quiet_platform):
    """More evaluator results than the manifest -> foreign data, never post into it."""
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [], [RESULT_DOC])
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [3]
    with pytest.raises(RuntimeError, match="foreign"):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 2, 0, 1),
            workspace_map={"ws-a": "ws-b"},
            catalog=CATALOG,
        )
    assert quiet_platform["posts"] == []


def test_ingest_bundle_heals_missing_annotations(tmp_path, quiet_platform, capsys):
    """Interrupted before any annotation landed: zero existing -> safe to post them all."""
    annotations = [ANNOTATION_DOC, {**ANNOTATION_DOC, "text": "second"}]
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], annotations, [RESULT_DOC])
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [1]
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 2, 1),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    urls = [url for url, _ in quiet_platform["posts"]]
    assert len(urls) == 2 and all(url.endswith("/workspaces/ws-b/annotations") for url in urls)
    assert "healing annotations: posting 2" in capsys.readouterr().out
    assert outcome["ws-a"]["annotations"] == {"ingested": 2, "skipped": 0}
    assert outcome["ws-a"]["evaluator_results"] == {"ingested": 0, "skipped": 1}


def test_ingest_bundle_partial_annotations_hard_error(tmp_path, quiet_platform):
    """0 < existing annotations < expected: re-posting would duplicate (fresh uuids) — hard error."""
    annotations = [ANNOTATION_DOC, {**ANNOTATION_DOC, "text": "second"}]
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], annotations)
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [1]
    with pytest.raises(RuntimeError, match="mint fresh") as exc:
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 2, 2),
            workspace_map={"ws-a": "ws-b"},
            catalog=CATALOG,
        )
    message = str(exc.value)
    assert "ws-b" in message  # names the workspace
    assert "duplicate" in message
    assert "delete" in message.lower() and "restore" in message  # recommends delete + re-restore
    assert quiet_platform["posts"] == []


def test_ingest_bundle_zero_span_workspace_annotations_idempotent(tmp_path, quiet_platform, capsys):
    """Finding 2: a 0-span workspace with annotations must not re-post them on a second restore.

    The old guard (`if expected and existing == expected`) short-circuited on a
    zero span count and re-posted (duplicated) the annotations every run.
    """
    annotations = [ANNOTATION_DOC, {**ANNOTATION_DOC, "text": "second"}]
    export_dir = _write_export(tmp_path, "ws-a", [], annotations)
    manifest = _manifest("ws-a", 0, 2)
    quiet_platform["span_counts"] = [0]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    first = reingest.ingest_bundle(
        "http://x",
        export_dir,
        manifest,
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert len(quiet_platform["posts"]) == 2  # first restore posts the annotations
    assert first["ws-a"]["annotations"] == {"ingested": 2, "skipped": 0}
    quiet_platform["span_counts"] = [0]
    quiet_platform["annotation_counts"] = [2]
    quiet_platform["result_counts"] = [0]
    second = reingest.ingest_bundle(
        "http://x",
        export_dir,
        manifest,
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert len(quiet_platform["posts"]) == 2  # second restore posts NOTHING new
    assert "ws-b: already restored (0 spans) — skipping" in capsys.readouterr().out
    assert second["ws-a"]["annotations"] == {"ingested": 0, "skipped": 2}


def test_ingest_bundle_requires_full_workspace_map(tmp_path, quiet_platform):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC])
    with pytest.raises(RuntimeError, match="no target for bundle workspace 'ws-a'"):
        reingest.ingest_bundle("http://x", export_dir, _manifest("ws-a", 1), workspace_map={}, catalog=CATALOG)


def test_ingest_bundle_validates_target_names(tmp_path, quiet_platform):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC])
    with pytest.raises(RuntimeError, match="naming rule"):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 1),
            workspace_map={"ws-a": "Bad--Name-"},
            catalog=CATALOG,
        )


# --- provider-neutral restore for non-OTLP sources ---


def test_ingest_bundle_restores_non_otel_sources_directly(tmp_path, quiet_platform):
    atif = {**AGENT_DOC, "source": "atif", "span_id": "span-abc123"}
    chat = {**LLM_DOC, "source": "chat-completions", "span_id": "chatcmpl-xyz"}
    export_dir = _write_export(tmp_path, "ws-a", [atif, chat, AGENT_DOC], [ANNOTATION_DOC])
    quiet_platform["span_counts"] = [0, 3]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]

    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 3, 1),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda _: None,
    )

    direct = [body for url, body in quiet_platform["posts"] if url.endswith("/ingest/spans")]
    assert {body["source"] for body in direct} == {"atif", "chat-completions", "otel"}
    assert outcome["ws-a"]["spans"] == {"ingested": 3, "skipped": 0}


def test_ingest_bundle_scans_all_span_files_before_writes(tmp_path, quiet_platform):
    _write_export(tmp_path, "ws-a", [AGENT_DOC])
    ws_dir = tmp_path / "export" / "ws-c"
    ws_dir.mkdir(parents=True)
    bad = {key: value for key, value in AGENT_DOC.items() if key != "source"}
    (ws_dir / "spans.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    manifest = {
        "workspaces": ["ws-a", "ws-c"],
        "counts": {"ws-a": {"spans": 1}, "ws-c": {"spans": 1}},
        "min_start_time": datetime.now(timezone.utc).isoformat(),
    }
    with pytest.raises(ValueError, match="no source"):
        reingest.ingest_bundle(
            "http://x",
            tmp_path / "export",
            manifest,
            workspace_map={"ws-a": "ws-a-fx", "ws-c": "ws-c-fx"},
            catalog=CATALOG,
        )
    assert quiet_platform["requests"] == []  # ws-a (clean) was NOT ingested either
    assert quiet_platform["ensured"] == []


# --- _wait_for_spans transient-error tolerance (B2) ---


def test_wait_for_spans_tolerates_transient_connect_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient blip")
        return httpx.Response(200, json={"pagination": {"total_results": 2}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reingest._wait_for_spans("http://x", "ws-a", 2, client=client, timeout_s=30.0, sleep=lambda s: None)
    assert calls["n"] == 2  # errored once, then saw the count and returned


def test_wait_for_spans_treats_non_200_as_transient():
    """span_count raises RuntimeError on non-200 — poll_visible's spirit: not-yet-visible."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="warming up")
        return httpx.Response(200, json={"pagination": {"total_results": 1}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reingest._wait_for_spans("http://x", "ws-a", 1, client=client, timeout_s=30.0, sleep=lambda s: None)
    assert calls["n"] == 2


def test_wait_for_spans_deadline_reports_last_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="gateway down"):
            reingest._wait_for_spans("http://x", "ws-a", 2, client=client, timeout_s=0.0, sleep=lambda s: None)


# --- first-span fingerprint probe on the skip path (B5) ---


def test_first_span_id_fetches_earliest_with_epoch_bound():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"span_id": "72f937bfcb667a9e"}], "pagination": {}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert reingest._first_span_id("http://x", "ws-b", client=client) == "72f937bfcb667a9e"
    params = seen[0].url.params
    assert seen[0].url.path == "/apis/intake/v2/workspaces/ws-b/spans"
    assert params["page_size"] == "1"
    assert params["sort"] == "started_at"  # ascending: the FIRST span
    assert params["filter[started_at][gte]"] == "1970-01-01T00:00:00Z"  # beat the 30-day default lookback


def test_first_span_id_transient_failure_returns_none():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("blip")

    with httpx.Client(transport=httpx.MockTransport(boom)) as client:
        assert reingest._first_span_id("http://x", "ws-b", client=client) is None

    def not_ready(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="nope")

    with httpx.Client(transport=httpx.MockTransport(not_ready)) as client:
        assert reingest._first_span_id("http://x", "ws-b", client=client) is None

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "pagination": {}})

    with httpx.Client(transport=httpx.MockTransport(empty)) as client:
        assert reingest._first_span_id("http://x", "ws-b", client=client) is None


def test_skip_path_rejects_different_corpus_with_matching_counts(tmp_path, quiet_platform):
    """Same span count, different first span id -> re-minted ref / foreign corpus: hard error."""
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [2]  # guard matches -> skip path
    quiet_platform["first_ids"] = ["ffffffffffffffff"]  # but the workspace holds something else
    with pytest.raises(RuntimeError, match="(?i)different"):
        reingest.ingest_bundle(
            "http://x",
            export_dir,
            _manifest("ws-a", 2, 1, 1),
            workspace_map={"ws-a": "ws-b"},
            catalog=CATALOG,
        )
    assert quiet_platform["requests"] == []
    assert quiet_platform["posts"] == []


def test_skip_path_probe_match_skips_as_before(tmp_path, quiet_platform, capsys):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    quiet_platform["span_counts"] = [2]
    quiet_platform["first_ids"] = ["72f937bfcb667a9e"]  # AGENT_DOC is earliest by started_at
    quiet_platform["annotation_counts"] = [1]
    quiet_platform["result_counts"] = [1]
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2, 1, 1),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert "already restored" in capsys.readouterr().out
    assert outcome["ws-a"]["spans"] == {"ingested": 0, "skipped": 2}


def test_skip_path_probe_tolerates_started_at_ties(tmp_path, quiet_platform, capsys):
    """Two docs share the min started_at: either id is an acceptable first span."""
    tied = {**LLM_DOC, "started_at": AGENT_DOC["started_at"]}
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, tied])
    quiet_platform["span_counts"] = [2]
    quiet_platform["first_ids"] = [tied["span_id"]]  # the OTHER member of the tie
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert "already restored" in capsys.readouterr().out


def test_skip_path_probe_compares_started_at_chronologically(tmp_path, quiet_platform, capsys):
    """Mixed timestamp serializations: the bundle's first span is the chronological min, not the lexicographic one."""
    # 14:14-05:00 is 19:14 UTC — lexicographically first but chronologically LATER than 18:14 UTC.
    later_by_offset = {**LLM_DOC, "started_at": "2026-06-26T14:14:41.406179-05:00"}
    earlier_utc = {**AGENT_DOC, "started_at": "2026-06-26T18:14:41.406179Z"}
    export_dir = _write_export(tmp_path, "ws-a", [earlier_utc, later_by_offset])
    quiet_platform["span_counts"] = [2]
    quiet_platform["first_ids"] = [AGENT_DOC["span_id"]]  # the live first span IS the chronological min
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert "already restored" in capsys.readouterr().out


def test_skip_path_probe_failure_falls_back_to_count_guard(tmp_path, quiet_platform, capsys):
    """Transient probe failure (fixture default: None) must NOT block the restore."""
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    quiet_platform["span_counts"] = [2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    outcome = reingest.ingest_bundle(
        "http://x",
        export_dir,
        _manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
    )
    assert ("first_span", "ws-b") in quiet_platform["counts"]  # the probe WAS attempted
    assert "already restored" in capsys.readouterr().out
    assert outcome["ws-a"]["spans"] == {"ingested": 0, "skipped": 2}


# --- TTL / lookback guardrail ---


def test_warn_if_stale_old_bundle(capsys):
    manifest = {"min_start_time": "2026-01-01T00:00:00+00:00"}
    message = reingest.warn_if_stale(manifest, now=datetime(2026, 7, 6, tzinfo=timezone.utc))
    assert message is not None
    assert "SYSTEM STOP TTL MERGES" in message
    assert "since" in message
    assert "SYSTEM STOP TTL MERGES" in capsys.readouterr().err


def test_warn_if_stale_fresh_or_unknown(capsys):
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    assert reingest.warn_if_stale({"min_start_time": "2026-06-26T18:14:41"}, now=now) is None
    assert reingest.warn_if_stale({"min_start_time": None}, now=now) is None
    assert capsys.readouterr().err == ""


# --- TTL merges stopped on stale loopback restores (B3) ---


def _stale_manifest(workspace: str, spans: int) -> dict:
    manifest = _manifest(workspace, spans)
    manifest["min_start_time"] = "2026-01-01T00:00:00+00:00"  # far past STALE_AFTER_DAYS
    return manifest


def test_stale_loopback_restore_stops_ttl_merges(tmp_path, monkeypatch, quiet_platform, fake_docker, capsys):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    container = "nhx-intake-clickhouse-a1b2c3d4e5f6"
    monkeypatch.setattr(reingest, "_local_clickhouse_container", lambda: container)
    quiet_platform["span_counts"] = [0, 2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://localhost:8080",
        export_dir,
        _stale_manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert fake_docker == [["docker", "exec", container, "clickhouse-client", "-q", "SYSTEM STOP TTL MERGES"]]
    err = capsys.readouterr().err
    assert "TTL merges" in err and "stopped" in err


def test_stale_remote_restore_warns_but_never_touches_docker(tmp_path, quiet_platform, fake_docker, capsys):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    quiet_platform["span_counts"] = [0, 2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://intake.example.com:8080",
        export_dir,
        _stale_manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert fake_docker == []  # the local docker ClickHouse is NOT the remote target
    assert "SYSTEM STOP TTL MERGES" in capsys.readouterr().err  # existing warning still shown


def test_stale_loopback_restore_with_external_clickhouse_never_touches_docker(
    tmp_path,
    monkeypatch,
    quiet_platform,
    fake_docker,
    capsys,
):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    monkeypatch.setenv("NHX_INTAKE_CLICKHOUSE_URL", "https://clickhouse.example.internal:8443")
    quiet_platform["span_counts"] = [0, 2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]

    reingest.ingest_bundle(
        "http://127.0.0.1:8080",
        export_dir,
        _stale_manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )

    assert fake_docker == []
    assert "points to external ClickHouse" in capsys.readouterr().err


def test_fresh_bundle_never_touches_ttl_merges(tmp_path, quiet_platform, fake_docker):
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    quiet_platform["span_counts"] = [0, 2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    reingest.ingest_bundle(
        "http://localhost:8080",
        export_dir,
        _manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert fake_docker == []


def test_local_clickhouse_container_resolves_managed_name_by_label(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="nhx-intake-clickhouse-a1b2c3d4e5f6\n", stderr="")

    monkeypatch.setenv("NHX_INTAKE_CLICKHOUSE_URL", "http://localhost:55123")
    monkeypatch.setattr(reingest.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(reingest.subprocess, "run", fake_run)

    assert reingest._local_clickhouse_container() == "nhx-intake-clickhouse-a1b2c3d4e5f6"
    assert "label=nhx.nvidia.com/managed-by=nemo-helix" in commands[0]
    assert "label=nhx.nvidia.com/component=intake-clickhouse" in commands[0]
    assert "publish=55123" in commands[0]


def test_local_clickhouse_container_rejects_multiple_managed_instances(monkeypatch):
    monkeypatch.setenv("NHX_INTAKE_CLICKHOUSE_URL", "http://127.0.0.1:55123")
    monkeypatch.setattr(reingest.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        reingest.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="first\nsecond\n", stderr=""),
    )

    with pytest.raises(RuntimeError, match="expected one.*found first, second"):
        reingest._local_clickhouse_container()


@pytest.mark.parametrize("failure", ["rc1", "oserror"])
def test_ttl_stop_failure_is_nonfatal_and_tells_user(tmp_path, quiet_platform, monkeypatch, capsys, failure):
    def failing_run(cmd, **kwargs):
        if failure == "oserror":
            raise FileNotFoundError("docker not installed")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such container")

    monkeypatch.setattr(reingest.subprocess, "run", failing_run)
    monkeypatch.setattr(reingest, "_local_clickhouse_container", lambda: "nhx-intake-clickhouse-a1b2c3d4e5f6")
    export_dir = _write_export(tmp_path, "ws-a", [AGENT_DOC, LLM_DOC])
    quiet_platform["span_counts"] = [0, 2]
    quiet_platform["annotation_counts"] = [0]
    quiet_platform["result_counts"] = [0]
    outcome = reingest.ingest_bundle(  # must not raise
        "http://127.0.0.1:8080",
        export_dir,
        _stale_manifest("ws-a", 2),
        workspace_map={"ws-a": "ws-b"},
        catalog=CATALOG,
        sleep=lambda s: None,
    )
    assert outcome["ws-a"]["spans"] == {"ingested": 2, "skipped": 0}
    err = capsys.readouterr().err
    assert "manually" in err and "SYSTEM STOP TTL MERGES" in err


# --- normalization / diff rules ---


def test_diff_normalizes_workspace_ingested_at_and_ordering():
    raw_attributes = AGENT_DOC["raw_attributes"]
    assert isinstance(raw_attributes, str)
    restored = {
        **AGENT_DOC,
        "workspace": "scratch-reingest-x",
        "ingested_at": "2026-07-06T09:00:00.000000",
        # same raw attributes, different key order + key order inside the doc doesn't matter either
        "raw_attributes": json.dumps(json.loads(raw_attributes), sort_keys=True),
    }
    assert reingest._diff_collection([AGENT_DOC], [restored], workspace="ws-a", collection="spans") == []


def test_diff_reports_value_changes_and_count_drift():
    broken = {**AGENT_DOC, "output": "DIFFERENT"}
    mismatches = reingest._diff_collection([AGENT_DOC], [broken], workspace="ws-a", collection="spans")
    assert mismatches == ["ws-a/spans 72f937bfcb667a9e.output: 'ok' != 'DIFFERENT'"]
    mismatches = reingest._diff_collection([AGENT_DOC], [], workspace="ws-a", collection="spans")
    assert mismatches == ["ws-a/spans: 1 exported vs 0 restored"]


def test_grouped_jsonl_diff_ignores_order_within_trace(tmp_path):
    original = tmp_path / "original.jsonl"
    restored = tmp_path / "restored.jsonl"
    original.write_text("\n".join(json.dumps(doc) for doc in (AGENT_DOC, LLM_DOC)) + "\n", encoding="utf-8")
    restored_docs = [
        {**doc, "workspace": "scratch-rt", "ingested_at": "2026-08-18T00:00:00Z"} for doc in (LLM_DOC, AGENT_DOC)
    ]
    restored.write_text("\n".join(json.dumps(doc) for doc in restored_docs) + "\n", encoding="utf-8")

    assert reingest._diff_grouped_jsonl(original, restored, workspace="ws-a", collection="spans") == []


def test_diff_normalizes_annotation_server_fields():
    restored = {
        **ANNOTATION_DOC,
        "annotation_id": "ann-fresh",
        "created_at": "2026-07-06T00:00:00",
        "ingested_at": "2026-07-06T00:00:00",
        "created_by": "someone",
        "workspace": "scratch-reingest-x",
    }
    assert reingest._diff_collection([ANNOTATION_DOC], [restored], workspace="ws-a", collection="annotations") == []


def test_diff_normalizes_evaluator_result_id():
    restored = {**RESULT_DOC, "evaluator_result_id": "eval-other", "workspace": "w2"}
    diff = reingest._diff_collection([RESULT_DOC], [restored], workspace="ws-a", collection="evaluator_results")
    assert diff == []
    changed = {**restored, "value": 1.0}
    diff = reingest._diff_collection([RESULT_DOC], [changed], workspace="ws-a", collection="evaluator_results")
    assert len(diff) == 1 and "value" in diff[0]


# --- roundtrip_diff wiring ---


def test_roundtrip_diff_maps_scratch_reexports_and_cleans_up(tmp_path, monkeypatch):
    export_dir = _write_export(tmp_path / "src", "ws-a", [AGENT_DOC], [ANNOTATION_DOC], [RESULT_DOC])
    manifest = _manifest("ws-a", 1, 1, 1)
    seen = {}

    def fake_ingest(base_url, exp_dir, mani, *, workspace_map, catalog, sleep):
        seen["map"] = workspace_map
        return {}

    def fake_export(base_url, workspaces, out_dir, *, since):
        seen["exported"] = (workspaces, since)
        restored = {**AGENT_DOC, "workspace": workspaces[0], "ingested_at": "2026-07-06T00:00:00"}
        ann = {**ANNOTATION_DOC, "annotation_id": "ann-new", "workspace": workspaces[0]}
        res = {**RESULT_DOC, "evaluator_result_id": "eval-new", "workspace": workspaces[0]}
        _write_export(out_dir, workspaces[0], [restored], [ann], [res])
        return {"workspaces": {}}

    monkeypatch.setattr(reingest, "ingest_bundle", fake_ingest)
    monkeypatch.setattr(reingest.export, "export_workspaces", fake_export)
    monkeypatch.setattr(reingest, "cleanup_scratch", lambda url, ws: seen.setdefault("cleaned", ws))

    mismatches = reingest.roundtrip_diff(
        "http://x",
        export_dir,
        manifest,
        scratch_prefix="scratch-reingest-",
        catalog=CATALOG,
        content_key="cafef00d",
    )
    assert mismatches == []
    assert seen["map"] == {"ws-a": "scratch-reingest-cafef00d-ws-a"}
    assert seen["exported"] == (["scratch-reingest-cafef00d-ws-a"], None)
    assert seen["cleaned"] == ["scratch-reingest-cafef00d-ws-a"]


def test_roundtrip_diff_handles_mixed_scoped_workspaces(tmp_path, monkeypatch):
    manifest = {
        "workspaces": ["ws-a", "ws-b"],
        "counts": {},
        "selections": {"ws-a": {"experiment": "run-a"}},
    }
    exports = []

    def fake_export(base_url, workspaces, out_dir, *, since, selection=None):
        exports.append((workspaces[0], selection))
        return {"workspaces": {}}

    monkeypatch.setattr(reingest, "ingest_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(reingest.export, "export_workspaces", fake_export)
    monkeypatch.setattr(reingest, "_diff_grouped_jsonl", lambda *args, **kwargs: [])
    monkeypatch.setattr(reingest, "cleanup_scratch", lambda *args: None)

    mismatches = reingest.roundtrip_diff(
        "http://x",
        tmp_path,
        manifest,
        scratch_prefix="scratch-reingest-",
        catalog=CATALOG,
        content_key="cafef00d",
    )
    assert mismatches == []
    assert exports == [
        ("scratch-reingest-cafef00d-ws-a", {"experiment": "run-a"}),
        ("scratch-reingest-cafef00d-ws-b", None),
    ]


def test_roundtrip_diff_cleans_up_even_when_ingest_fails(tmp_path, monkeypatch):
    export_dir = _write_export(tmp_path / "src", "ws-a", [AGENT_DOC])
    cleaned = []

    def boom(*args, **kwargs):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(reingest, "ingest_bundle", boom)
    monkeypatch.setattr(reingest, "cleanup_scratch", lambda url, ws: cleaned.extend(ws))
    with pytest.raises(RuntimeError, match="ingest exploded"):
        reingest.roundtrip_diff(
            "http://x",
            export_dir,
            _manifest("ws-a", 1),
            scratch_prefix="scratch-reingest-",
            catalog=CATALOG,
            content_key="cafef00d",
        )
    assert cleaned == ["scratch-reingest-cafef00d-ws-a"]


# --- scratch naming is content-scoped (B4a) ---


def test_scratch_workspace_map_scoped_by_content_key():
    manifest = _manifest("ws-a", 2)
    mapping = reingest._scratch_workspace_map(manifest, "scratch-rt-", "a1b2c3d4")
    assert mapping == {"ws-a": "scratch-rt-a1b2c3d4-ws-a"}
    assert all(reingest._WS_OK.fullmatch(name) for name in mapping.values())


def test_scratch_workspace_map_default_key_deterministic_from_manifest():
    manifest = _manifest("ws-a", 2)
    first = reingest._scratch_workspace_map(manifest, "scratch-rt-", None)
    again = reingest._scratch_workspace_map(dict(manifest), "scratch-rt-", None)
    assert first == again  # deterministic: same counts+bounds -> same names
    other = reingest._scratch_workspace_map(_manifest("ws-a", 3), "scratch-rt-", None)
    assert first != other  # different content -> different scratch names
    assert all(reingest._WS_OK.fullmatch(name) for name in first.values())
    assert first["ws-a"].startswith("scratch-rt-") and first["ws-a"].endswith("-ws-a")


def test_scratch_workspace_map_truncates_long_names_within_ws_rules():
    long_ws = "w" + "x" * 58  # a valid 59-char source workspace
    manifest = {"workspaces": [long_ws], "counts": {long_ws: {"spans": 1}}}
    mapping = reingest._scratch_workspace_map(manifest, "scratch-rt-", "a1b2c3d4")
    name = mapping[long_ws]
    assert len(name) <= 63
    assert name.startswith("scratch-rt-a1b2c3d4-")
    assert reingest._WS_OK.fullmatch(name)


def test_scratch_workspace_map_unsafe_content_key_is_hashed():
    manifest = _manifest("ws-a", 1)
    mapping = reingest._scratch_workspace_map(manifest, "scratch-rt-", "Not*Hex!")
    assert all(reingest._WS_OK.fullmatch(name) for name in mapping.values())
    # still content-scoped: a different key yields a different name
    other = reingest._scratch_workspace_map(manifest, "scratch-rt-", "Other*Key!")
    assert mapping != other


# --- cleanup_scratch respects the target (B4b) ---


def test_cleanup_scratch_remote_leaves_rows_and_record_and_reports(monkeypatch, fake_docker, capsys):
    """Remote target: local docker is NOT the target's ClickHouse — deleting the workspace
    record would orphan rows behind it. Leave BOTH; tell the user what was left."""

    class _NoHTTP:
        def __init__(self, *args, **kwargs):
            raise AssertionError("remote cleanup must not touch the entities API")

    monkeypatch.setattr(reingest.httpx, "Client", _NoHTTP)
    reingest.cleanup_scratch("http://intake.example.com:8080", ["scratch-rt-abc12345-ws-a", "scratch-rt-abc12345-ws-b"])
    assert fake_docker == []
    err = capsys.readouterr().err
    assert "scratch-rt-abc12345-ws-a" in err and "scratch-rt-abc12345-ws-b" in err
    assert "manual" in err.lower()


def test_cleanup_scratch_loopback_deletes_rows_and_keeps_reusable_record(monkeypatch, fake_docker):
    class _NoHTTP:
        def __init__(self, *args, **kwargs):
            raise AssertionError("row cleanup must keep the active workspace record")

    monkeypatch.setattr(reingest.httpx, "Client", _NoHTTP)
    monkeypatch.setattr(reingest, "_local_clickhouse_container", lambda: "nhx-intake-clickhouse-a1b2c3d4e5f6")
    reingest.cleanup_scratch("http://127.0.0.1:8080", ["scratch-rt-abc12345-ws-a"])
    tables = {cmd[-1].split("FROM intake.")[1].split(" ")[0] for cmd in fake_docker}
    assert tables == {"spans", "annotations", "evaluator_results", "trace_index"}


def test_cleanup_scratch_keeps_workspace_record_when_container_is_ambiguous(monkeypatch, capsys):
    class _NoHTTP:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unsafe cleanup must not delete workspace records")

    monkeypatch.setattr(reingest.httpx, "Client", _NoHTTP)

    def ambiguous_container() -> str:
        raise RuntimeError("multiple local ClickHouse containers found")

    monkeypatch.setattr(reingest, "_local_clickhouse_container", ambiguous_container)

    reingest.cleanup_scratch("http://127.0.0.1:8080", ["scratch-rt-abc12345-ws-a"])

    assert "rows and workspace records left intact" in capsys.readouterr().err


def test_cleanup_scratch_ignores_unlabeled_legacy_name_collision(monkeypatch, capsys):
    commands: list[list[str]] = []

    class _NoHTTP:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unowned container cleanup must not delete workspace records")

    def fake_run(command, **kwargs):
        commands.append(command)
        # An unrelated running nhx-intake-clickhouse exists, but it does not
        # carry the durable Intake ownership labels selected by this query.
        stdout = "nhx-intake-clickhouse\n" if any(part.startswith("name=") for part in command) else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("NHX_INTAKE_CLICKHOUSE_URL", "http://localhost:8123")
    monkeypatch.setattr(reingest.httpx, "Client", _NoHTTP)
    monkeypatch.setattr(reingest.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(reingest.subprocess, "run", fake_run)

    reingest.cleanup_scratch("http://127.0.0.1:8080", ["scratch-rt-abc12345-ws-a"])

    assert len(commands) == 1
    assert not any(command_part.startswith("name=") for command_part in commands[0])
    assert "rows and workspace records left intact" in capsys.readouterr().err


def test_cleanup_scratch_loopback_with_external_clickhouse_preserves_all_state(monkeypatch, capsys):
    class _NoHTTP:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unverified cleanup must not delete workspace records")

    monkeypatch.setenv("NHX_INTAKE_CLICKHOUSE_URL", "https://clickhouse.example.internal:8443")
    monkeypatch.setattr(reingest.httpx, "Client", _NoHTTP)

    def fail_docker_lookup(name: str) -> str:
        raise AssertionError("external ClickHouse must not inspect Docker")

    monkeypatch.setattr(reingest.shutil, "which", fail_docker_lookup)

    reingest.cleanup_scratch("http://127.0.0.1:8080", ["scratch-rt-abc12345-ws-a"])

    err = capsys.readouterr().err
    assert "points to external ClickHouse" in err
    assert "rows and workspace records left intact" in err


def test_roundtrip_diff_rejects_non_scratch_prefix(tmp_path):
    with pytest.raises(ValueError, match="scratch-"):
        reingest.roundtrip_diff("http://x", tmp_path, _manifest("ws-a", 1), scratch_prefix="prod-", catalog=CATALOG)


def test_roundtrip_diff_surfaces_mismatches(tmp_path, monkeypatch):
    export_dir = _write_export(tmp_path / "src", "ws-a", [AGENT_DOC])
    manifest = _manifest("ws-a", 1)

    def fake_export(base_url, workspaces, out_dir, *, since):
        broken = {**AGENT_DOC, "workspace": workspaces[0], "output": "MANGLED"}
        _write_export(out_dir, workspaces[0], [broken])
        return {"workspaces": {}}

    monkeypatch.setattr(reingest, "ingest_bundle", lambda *a, **k: {})
    monkeypatch.setattr(reingest.export, "export_workspaces", fake_export)
    monkeypatch.setattr(reingest, "cleanup_scratch", lambda url, ws: None)
    mismatches = reingest.roundtrip_diff(
        "http://x", export_dir, manifest, scratch_prefix="scratch-reingest-", catalog=CATALOG
    )
    assert mismatches == ["ws-a/spans 72f937bfcb667a9e.output: 'ok' != 'MANGLED'"]


# --- platform-root resolution / fixture workspace mapping / bundle since (Task 3 helpers) ---


def _checkout_with_catalog(root: Path) -> Path:
    catalog = root / reingest.CATALOG_RELPATH
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("ATTRIBUTE_SPECS = ()\n", encoding="utf-8")
    return root


def test_resolve_platform_root_explicit_beats_everything(tmp_path):
    root = reingest.resolve_platform_root(
        str(tmp_path / "explicit"),
        env={"NHX_PLATFORM_ROOT": "/env"},
        candidates=[tmp_path / "cand"],
    )
    assert root == tmp_path / "explicit"


def test_resolve_platform_root_env_beats_candidates(tmp_path):
    good = _checkout_with_catalog(tmp_path / "cand")
    root = reingest.resolve_platform_root(None, env={"NHX_PLATFORM_ROOT": "/env"}, candidates=[good])
    assert root == Path("/env")


def test_resolve_platform_root_first_candidate_with_catalog_wins(tmp_path):
    ci = _checkout_with_catalog(tmp_path / "ci" / "nemo-helix")
    home = _checkout_with_catalog(tmp_path / "home" / "nemo-helix")
    assert reingest.resolve_platform_root(None, env={}, candidates=[ci, home]) == ci


def test_resolve_platform_root_skips_catalogless_candidate(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    good = _checkout_with_catalog(tmp_path / "good")
    assert reingest.resolve_platform_root(None, env={}, candidates=[empty, good]) == good


def test_resolve_platform_root_nothing_found_exits_with_fix(tmp_path):
    with pytest.raises(SystemExit) as exc:
        reingest.resolve_platform_root(None, env={}, candidates=[tmp_path / "nowhere"])
    assert "--platform-root" in str(exc.value)
    assert "NHX_PLATFORM_ROOT" in str(exc.value)


def test_default_platform_roots_monorepo_then_legacy_checkout():
    plugin_root = Path(reingest.__file__).resolve().parent.parent
    assert reingest.default_platform_roots() == [
        plugin_root.parents[1],
        Path.home() / "workstation" / "nemo-helix",
    ]


def test_fixture_workspace_map_ref_suffix():
    assert reingest.fixture_workspace_map(["tau2-airline", "tau2-airline-oracle"], "state-v6") == {
        "tau2-airline": "tau2-airline-state-v6",
        "tau2-airline-oracle": "tau2-airline-oracle-state-v6",
    }


def test_fixture_workspace_map_rejects_invalid_target():
    # a trailing hyphen in the source yields a forbidden double hyphen in the target
    with pytest.raises(SystemExit, match="naming rule"):
        reingest.fixture_workspace_map(["ws-"], "state-v6")


def test_bundle_digest_is_sha256_prefix(tmp_path):
    import hashlib

    bundle = tmp_path / "b.tar.zst"
    bundle.write_bytes(b"bundle-bytes")
    assert reingest.bundle_digest(bundle) == hashlib.sha256(b"bundle-bytes").hexdigest()[:8]
    assert len(reingest.bundle_digest(bundle)) == 8


def test_manifest_since_floors_min_start_time():
    since = reingest.manifest_since({"min_start_time": "2026-06-01T12:00:00+00:00"})
    assert since == datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)


def test_manifest_since_naive_timestamp_is_utc():
    since = reingest.manifest_since({"min_start_time": "2026-06-01T12:00:00"})
    assert since == datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)


def test_manifest_since_missing_bound_is_epoch():
    assert reingest.manifest_since({}) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert reingest.manifest_since({"min_start_time": None}) == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_explicit_workspace_map_single_workspace() -> None:
    assert reingest.explicit_workspace_map(["run-scoped-workspace"], "stable-workspace") == {
        "run-scoped-workspace": "stable-workspace"
    }


def test_explicit_workspace_map_rejects_multi_workspace_bundle() -> None:
    with pytest.raises(SystemExit, match="single-workspace"):
        reingest.explicit_workspace_map(["realistic", "oracle"], "stable-workspace")
