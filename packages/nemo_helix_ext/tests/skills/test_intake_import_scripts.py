# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

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


def test_intake_writer_uses_sdk_factory_for_context_and_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    sdk = Mock(base_url="https://platform.example.com", workspace="oauth-workspace")
    sdk._client = Mock()
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(common, "create_client", factory)

    writer = common.IntakeWriter(base_url=None, workspace=None)

    assert writer.base_url == "https://platform.example.com"
    assert writer.workspace == "oauth-workspace"
    factory.assert_called_once_with(base_url=None, access_token=None, timeout=60.0, max_retries=0)
    verify_spans = Mock()
    monkeypatch.setattr(writer, "_verify_spans", verify_spans)
    span = {"span_id": "span-1", "trace_id": "trace-1", "started_at": "2026-08-14T12:00:00Z"}

    writer.write(common.ImportBundle(source="langsmith", spans=[span]), batch_size=500)

    sdk.intake.ingest.spans.create.assert_called_once_with(
        workspace="oauth-workspace",
        source="langsmith",
        spans=[span],
    )
    verify_spans.assert_called_once_with([span], source="langsmith")
    writer.close()
    sdk.close.assert_called_once_with()


def test_intake_writer_reports_only_new_annotation_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    annotation = {
        "kind": "note",
        "span_id": "span-1",
        "session_id": "session-1",
        "text": "already imported",
    }
    session = Mock()
    session.request.side_effect = [
        _Response({}, status_code=201),
        _Response({"data": [annotation], "pagination": {"page": 1, "total_pages": 1}}),
    ]
    writer = common.IntakeWriter(
        base_url="https://platform.example.com",
        workspace="default",
        session=session,
    )
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    bundle = common.ImportBundle(
        source="langsmith",
        spans=[{"span_id": "span-1", "trace_id": "trace-1"}],
        annotations=[annotation],
    )

    summary = writer.write(bundle, batch_size=500)

    assert summary["annotations"] == 0
    assert session.request.call_count == 2
    assert session.request.call_args_list[0].args[0] == "POST"
    assert session.request.call_args_list[1].args[0] == "GET"


def test_intake_writer_fetches_existing_annotations_once_per_target(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    first = {"kind": "note", "span_id": "span-1", "session_id": "session-1", "text": "first"}
    second = {**first, "text": "second"}
    session = Mock()
    session.request.side_effect = [
        _Response({}, status_code=201),
        _Response({"data": [], "pagination": {"page": 1, "total_pages": 1}}),
        _Response({}, status_code=201),
        _Response({}, status_code=201),
    ]
    writer = common.IntakeWriter(
        base_url="https://platform.example.com",
        workspace="default",
        session=session,
    )
    monkeypatch.setattr(writer, "_verify_spans", Mock())
    bundle = common.ImportBundle(
        source="langsmith",
        spans=[{"span_id": "span-1", "trace_id": "trace-1"}],
        annotations=[first, second],
    )

    summary = writer.write(bundle, batch_size=500)

    assert summary["annotations"] == 2
    methods = [call.args[0] for call in session.request.call_args_list]
    assert methods == ["POST", "GET", "POST", "POST"]


def test_intake_writer_accepts_injected_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201)

    with httpx.Client(transport=httpx.MockTransport(handle)) as session:
        writer = common.IntakeWriter(
            base_url="https://platform.example.com",
            workspace="default",
            session=session,
        )
        monkeypatch.setattr(writer, "_verify_spans", Mock())

        summary = writer.write(
            common.ImportBundle(source="langsmith", spans=[{"span_id": "span-1", "trace_id": "trace-1"}]),
            batch_size=500,
        )

    assert summary["spans"] == 1
    assert [request.method for request in requests] == ["POST"]


def test_intake_writer_paginated_data_advances_pages_and_enforces_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    common = importlib.import_module("_import_common")
    session = Mock()
    session.request.side_effect = [
        _Response({"data": [{"span_id": "one"}], "pagination": {"page": 1, "total_pages": 3}}),
        _Response({"data": [{"span_id": "two"}], "pagination": {"page": 2, "total_pages": 3}}),
    ]
    writer = common.IntakeWriter(
        base_url="https://platform.example.com",
        workspace="default",
        session=session,
    )
    monkeypatch.setattr(common, "MAX_INTAKE_PAGES", 2)

    with pytest.raises(RuntimeError, match="pagination exceeded 2 pages"):
        list(writer._paginated_data("/spans", filters=[("filter[source]", "langsmith")]))

    pages = [dict(call.kwargs["params"])["page"] for call in session.request.call_args_list]
    assert pages == [1, 2]


def test_sdk_session_uses_public_sdk_request_methods() -> None:
    common = importlib.import_module("_import_common")
    client = Mock()
    adapter = common._SdkSession(client)

    adapter.request(
        "GET",
        "https://platform.example.com/apis/intake/v2/workspaces/default/spans",
        params=[("page", 1)],
        headers={},
        timeout=60,
        follow_redirects=False,
    )
    adapter.request(
        "POST",
        "https://platform.example.com/apis/intake/v2/workspaces/default/annotations",
        json={"kind": "note"},
        headers={},
        timeout=60,
        follow_redirects=False,
    )

    client.get.assert_called_once_with(
        "https://platform.example.com/apis/intake/v2/workspaces/default/spans",
        cast_to=httpx.Response,
        options={"params": [("page", 1)], "headers": {}, "timeout": 60, "follow_redirects": False},
    )
    client.post.assert_called_once_with(
        "https://platform.example.com/apis/intake/v2/workspaces/default/annotations",
        cast_to=httpx.Response,
        body={"kind": "note"},
        options={"headers": {}, "timeout": 60, "follow_redirects": False},
    )


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
