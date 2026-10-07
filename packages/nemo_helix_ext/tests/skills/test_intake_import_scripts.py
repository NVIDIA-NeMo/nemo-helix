# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import importlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.intake.client import IntakeClient
from nhx.testing import mock_nemo_client

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-intake/scripts"
FIXTURES = Path(__file__).parent / "fixtures/observability"


@pytest.fixture(autouse=True)
def _import_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))


def _payload(provider: str) -> dict[str, Any]:
    if provider == "mlflow":
        return _json("mlflow-trace.json")
    if provider == "langsmith":
        return {**_json("langsmith-runs.json"), **_json("langsmith-feedback.json")}
    if provider == "phoenix":
        return {"spans": _json("phoenix-spans.json")["data"], **_json("phoenix-annotations.json")}
    if provider == "braintrust":
        return _json("braintrust-project-log.json")
    raise AssertionError(provider)


def _bundle(provider: str, *, include_feedback: bool = True) -> Any:
    module = importlib.import_module(f"import_{provider}")
    mapper = getattr(module, f"map_{provider}_export")
    return mapper(_payload(provider), project="fixture", include_feedback=include_feedback)


@pytest.mark.parametrize("provider", ["mlflow", "langsmith", "phoenix", "braintrust"])
def test_official_provider_fixture_maps_to_golden_without_field_gaps(provider: str) -> None:
    bundle = _bundle(provider)
    golden = _json(f"expected-{provider}.json")

    bundle.validate()
    assert len(bundle.spans) == golden["span_count"]
    span = bundle.spans[0]
    for key, value in golden["span"].items():
        assert span[key] == value
    for expectation in golden["raw_paths"]:
        assert _at_path(span["attributes"], expectation["path"]) == expectation["value"]
    assert span["attributes"][f"{provider}.signals"] == golden["signals"]
    assert [item["name"] for item in bundle.evaluator_results] == golden["evaluator_names"]
    assert [item["kind"] for item in bundle.annotations] == golden["annotation_kinds"]

    for coverage in bundle.coverage:
        dispositions = (coverage.mapped_fields, coverage.preserved_fields, coverage.ignored_fields)
        assert set().union(*dispositions) == coverage.all_fields
        assert not any(left & right for index, left in enumerate(dispositions) for right in dispositions[index + 1 :])


@pytest.mark.parametrize("provider", ["mlflow", "langsmith", "phoenix", "braintrust"])
def test_include_feedback_false_suppresses_all_signal_writes(provider: str) -> None:
    bundle = _bundle(provider, include_feedback=False)

    assert bundle.evaluator_results == []
    assert bundle.annotations == []
    assert all(f"{provider}.signals" not in span["attributes"] for span in bundle.spans)


def test_braintrust_repeated_event_versions_keep_latest_first() -> None:
    payload = _payload("braintrust")
    stale = {**payload["events"][0], "output": {"answer": "stale"}}
    payload["events"].append(stale)
    module = importlib.import_module("import_braintrust")

    bundle = module.map_braintrust_export(payload, project="fixture", include_feedback=True)

    assert len(bundle.spans) == 1
    assert bundle.spans[0]["output"] == {"answer": "Paris"}


def test_braintrust_epoch_zero_start_is_not_replaced_by_created_time() -> None:
    payload = _payload("braintrust")
    payload["events"][0]["metrics"] = {"start": 0, "duration": 1}
    module = importlib.import_module("import_braintrust")

    bundle = module.map_braintrust_export(payload, project="fixture", include_feedback=False)

    assert bundle.spans[0]["started_at"] == "1970-01-01T00:00:00+00:00"
    assert bundle.spans[0]["ended_at"] == "1970-01-01T00:00:01+00:00"


def test_braintrust_top_level_name_is_mapped_without_raw_duplication() -> None:
    payload = _payload("braintrust")
    payload["events"][0]["name"] = "top-level-name"
    payload["events"][0]["span_attributes"].pop("name")
    module = importlib.import_module("import_braintrust")

    bundle = module.map_braintrust_export(payload, project="fixture", include_feedback=False)

    assert bundle.spans[0]["name"] == "top-level-name"
    assert "name" not in bundle.spans[0]["attributes"]["braintrust.raw"]
    event_coverage = next(item for item in bundle.coverage if item.record == "event[0]")
    assert "name" in event_coverage.mapped_fields


def test_braintrust_fetch_does_not_assume_created_time_page_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_braintrust")
    responses = [
        _Response({"events": [{"id": "old", "created": "2025-01-01T00:00:00Z"}], "cursor": "next"}),
        _Response({"events": [{"id": "in-range", "created": "2026-08-01T12:00:00Z"}]}),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")
    args = argparse.Namespace(
        braintrust_base_url="https://api.braintrust.dev",
        project="project-id",
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        until=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    payload = module.fetch_braintrust(args)

    assert [event["id"] for event in payload["events"]] == ["in-range"]
    assert get.call_count == 2


def test_braintrust_fetch_accepts_null_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_braintrust")
    get = Mock(return_value=_Response({"events": [{"id": "one", "created": "2026-08-01T12:00:00Z", "metrics": None}]}))
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    payload = module.fetch_braintrust(_braintrust_args())

    assert payload["events"][0]["id"] == "one"


def test_braintrust_fetch_rejects_missing_event_id(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_braintrust")
    monkeypatch.setattr(
        module.requests,
        "get",
        Mock(return_value=_Response({"events": [{"created": "2026-08-01T12:00:00Z"}]})),
    )
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    with pytest.raises(ValueError, match=r"Braintrust events require `id`"):
        module.fetch_braintrust(_braintrust_args())


def test_braintrust_fetch_rejects_repeated_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_braintrust")
    get = Mock(
        side_effect=[
            _Response({"events": [{"id": "one", "created": "2026-08-01T12:00:00Z"}], "cursor": "same"}),
            _Response({"events": [{"id": "two", "created": "2026-08-01T13:00:00Z"}], "cursor": "same"}),
        ]
    )
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="Braintrust pagination returned a repeated cursor"):
        module.fetch_braintrust(_braintrust_args())

    assert get.call_count == 2


def test_braintrust_fetch_stops_on_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_braintrust")
    get = Mock(return_value=_Response({"events": [], "cursor": "unused"}))
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    payload = module.fetch_braintrust(_braintrust_args())

    assert payload == {"events": []}
    assert get.call_count == 1


def test_phoenix_fetch_pages_reads_multiple_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_phoenix")
    get = Mock(
        side_effect=[
            _Response({"data": [{"spanId": "one"}], "next_cursor": "next"}),
            _Response({"data": [{"spanId": "two"}]}),
        ]
    )
    monkeypatch.setattr(module.requests, "get", get)

    results = module._fetch_pages("https://phoenix.example.com/v1/spans", headers={}, params={})

    assert [item["spanId"] for item in results] == ["one", "two"]
    assert get.call_count == 2


def test_phoenix_fetch_pages_rejects_repeated_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("import_phoenix")
    get = Mock(
        side_effect=[
            _Response({"data": [{"spanId": "one"}], "next_cursor": "same"}),
            _Response({"data": [{"spanId": "two"}], "next_cursor": "same"}),
        ]
    )
    monkeypatch.setattr(module.requests, "get", get)

    with pytest.raises(RuntimeError, match="Phoenix pagination returned a repeated cursor"):
        module._fetch_pages("https://phoenix.example.com/v1/spans", headers={}, params={})

    assert get.call_count == 2


def test_mlflow_native_trace_id_fallback_is_canonicalized() -> None:
    payload = _payload("mlflow")
    native_trace_id = bytes(range(16))
    payload["traces"][0]["info"]["trace_id"] = ""
    for span in payload["traces"][0]["data"]["spans"]:
        span["trace_id"] = base64.b64encode(native_trace_id).decode()
    module = importlib.import_module("import_mlflow")

    bundle = module.map_mlflow_export(payload, project="fixture", include_feedback=True)

    assert {span["trace_id"] for span in bundle.spans} == {native_trace_id.hex()}
    assert bundle.evaluator_results[0]["span_id"] in {span["span_id"] for span in bundle.spans}


def test_phoenix_mapper_reports_missing_required_start_time() -> None:
    payload = _payload("phoenix")
    del payload["spans"][0]["start_time_unix_nano"]
    module = importlib.import_module("import_phoenix")

    with pytest.raises(ValueError, match="Phoenix spans require startTimeUnixNano"):
        module.map_phoenix_export(payload, project="fixture", include_feedback=False)


def test_nanoseconds_to_datetime_preserves_microseconds_without_float_rounding() -> None:
    common = importlib.import_module("_import_common")

    assert common.nanoseconds_to_datetime(1_000_000_000_123_456_789) == "2001-09-09T01:46:40.123456+00:00"


@pytest.mark.parametrize(
    ("provider", "field", "message"),
    [
        ("mlflow", "start_time_unix_nano", "MLflow spans require start_time_unix_nano"),
        ("langsmith", "start_time", "LangSmith runs require start_time"),
    ],
)
def test_provider_mapper_reports_missing_required_start_time(provider: str, field: str, message: str) -> None:
    payload = _payload(provider)
    if provider == "mlflow":
        del payload["traces"][0]["data"]["spans"][0][field]
    else:
        del payload["runs"][0][field]
    module = importlib.import_module(f"import_{provider}")
    mapper = getattr(module, f"map_{provider}_export")

    with pytest.raises(ValueError, match=message):
        mapper(payload, project="fixture", include_feedback=False)


def _intake_client(handler: Callable[[httpx.Request], httpx.Response]) -> IntakeClient:
    return mock_nemo_client(handler, IntakeClient, base_url="https://platform.example.com", workspace="default")


def _page(items: list[dict[str, Any]], *, page: int = 1, total_pages: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": items,
            "pagination": {
                "page": page,
                "page_size": 1000,
                "current_page_size": len(items),
                "total_pages": total_pages,
                "total_results": len(items),
            },
        },
    )


def _annotation_row(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": "annotation-1",
        "workspace": "default",
        "created_at": "2026-08-14T12:00:00Z",
        "ingested_at": "2026-08-14T12:00:00Z",
        **annotation,
    }


def test_intake_writer_builds_a_typed_client_from_the_cli_context(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201)

    nemo_client = mock_nemo_client(
        handle, NemoClient, base_url="https://platform.example.com", workspace="oauth-workspace"
    )
    build = Mock(return_value=nemo_client)
    monkeypatch.setattr(common, "build_nemo_client", build)
    span = {"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"}

    with common.IntakeWriter(base_url=None, workspace=None) as writer:
        monkeypatch.setattr(writer, "_verify_spans", Mock())
        assert writer.base_url == "https://platform.example.com"
        assert writer.workspace == "oauth-workspace"
        summary = writer.write(common.ImportBundle(source="langsmith", spans=[span]), batch_size=500)

    build.assert_called_once_with(base_url=None, access_token=None, timeout=60.0, retry=None)
    assert summary["spans"] == 1
    assert [request.method for request in captured] == ["POST"]
    assert captured[0].url.path == "/apis/intake/v2/workspaces/oauth-workspace/ingest/spans"
    payload = json.loads(captured[0].content)
    assert payload["source"] == "langsmith"
    assert payload["spans"][0]["span_id"] == "span-1"
    assert payload["spans"][0]["trace_id"] == "trace-1"


def test_intake_writer_batches_spans_by_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201)

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    spans = [
        {"span_id": f"span-{index}", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"} for index in range(3)
    ]

    writer.write(common.ImportBundle(source="langsmith", spans=spans), batch_size=2)

    assert [len(json.loads(request.content)["spans"]) for request in captured] == [2, 1]


def test_intake_writer_posts_evaluator_results(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path.endswith("/evaluator-results"):
            return httpx.Response(
                201,
                json={
                    "evaluator_result_id": "result-1",
                    "span_id": "span-1",
                    "session_id": "session-1",
                    "workspace": "default",
                    "name": "langsmith.score",
                    "value": 0.5,
                    "data_type": "NUMERIC",
                    "created_at": "2026-08-14T12:00:00Z",
                    "ingested_at": "2026-08-14T12:00:00Z",
                },
            )
        return httpx.Response(201)

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    result = {
        "span_id": "span-1",
        "session_id": "session-1",
        "name": "langsmith.score",
        "data_type": "NUMERIC",
        "value": 0.5,
        "comment": None,
    }
    bundle = common.ImportBundle(
        source="langsmith",
        spans=[{"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"}],
        evaluator_results=[result],
    )

    writer.write(bundle, batch_size=500)

    assert [request.url.path for request in captured] == [
        "/apis/intake/v2/workspaces/default/ingest/spans",
        "/apis/intake/v2/workspaces/default/evaluator-results",
    ]
    assert json.loads(captured[1].content) == {
        "span_id": "span-1",
        "session_id": "session-1",
        "name": "langsmith.score",
        "data_type": "NUMERIC",
        "value": 0.5,
        "comment": None,
    }


def test_intake_writer_reports_only_new_annotation_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    annotation = {
        "kind": "note",
        "span_id": "span-1",
        "session_id": "session-1",
        "text": "already imported",
    }
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.method == "GET":
            return _page([_annotation_row(annotation)])
        return httpx.Response(201)

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    bundle = common.ImportBundle(
        source="langsmith",
        spans=[{"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"}],
        annotations=[annotation],
    )

    summary = writer.write(bundle, batch_size=500)

    assert summary["annotations"] == 0
    assert [request.method for request in captured] == ["POST", "GET"]
    assert captured[1].url.path == "/apis/intake/v2/workspaces/default/annotations"
    assert json.loads(captured[1].url.params["filter"]) == {
        "session_id": "session-1",
        "kind": "note",
        "span_id": "span-1",
    }


def test_intake_writer_fetches_existing_annotations_once_per_target(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    first = {"kind": "note", "span_id": "span-1", "session_id": "session-1", "text": "first"}
    second = {**first, "text": "second"}
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.method == "GET":
            return _page([])
        return httpx.Response(201, json=_annotation_row(first))

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    bundle = common.ImportBundle(
        source="langsmith",
        spans=[{"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"}],
        annotations=[first, second],
    )

    summary = writer.write(bundle, batch_size=500)

    assert summary["annotations"] == 2
    assert [request.method for request in captured] == ["POST", "GET", "POST", "POST"]


def test_intake_writer_verifies_imported_spans_by_trace_and_source() -> None:
    common = importlib.import_module("_import_common")
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _page([_span_row("span-1", "trace-1")])

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))

    writer._verify_spans([{"span_id": "span-1", "trace_id": "trace-1"}], source="langsmith")

    assert json.loads(captured[0].url.params["filter"]) == {"trace_id": "trace-1", "source": "langsmith"}
    with pytest.raises(RuntimeError, match=r"did not find imported spans: \['span-2'\]"):
        writer._verify_spans(
            [{"span_id": "span-1", "trace_id": "trace-1"}, {"span_id": "span-2", "trace_id": "trace-1"}],
            source="langsmith",
        )


def test_intake_writer_pagination_enforces_the_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    pages_requested: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        pages_requested.append(str(page))
        return _page([_span_row(f"span-{page}", "trace-1")], page=page, total_pages=3)

    writer = common.IntakeWriter(base_url=None, workspace="default", client=_intake_client(handle))
    monkeypatch.setattr(common, "MAX_INTAKE_PAGES", 2)

    with pytest.raises(RuntimeError, match="pagination exceeded 2 pages"):
        writer._verify_spans([{"span_id": "span-9", "trace_id": "trace-1"}], source="langsmith")

    assert pages_requested == ["1", "2", "3"]


def test_intake_writer_rejects_a_non_origin_base_url() -> None:
    common = importlib.import_module("_import_common")
    client = mock_nemo_client(
        lambda _request: httpx.Response(201), IntakeClient, base_url="https://platform.example.com/prefix"
    )

    with pytest.raises(ValueError, match="origin without a path"):
        common.IntakeWriter(base_url=None, workspace=None, client=client)


def _span_row(span_id: str, trace_id: str) -> dict[str, Any]:
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "session_id": "session-1",
        "workspace": "default",
        "kind": "LLM",
        "source": "langsmith",
        "started_at": "2026-08-14T12:00:00Z",
        "ingested_at": "2026-08-14T12:00:00Z",
        "status": "success",
    }


class _Response:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.content = self.text.encode()

    def json(self) -> dict[str, Any]:
        return self._payload


def _braintrust_args() -> argparse.Namespace:
    return argparse.Namespace(
        braintrust_base_url="https://api.braintrust.dev",
        project="project-id",
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        until=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )


def _json(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _at_path(value: Any, path: list[str | int]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current
