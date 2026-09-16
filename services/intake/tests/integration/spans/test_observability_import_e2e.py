# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official provider fixtures through adapters and the real ClickHouse-backed Intake API."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / "packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-intake/scripts"
FIXTURES = ROOT / "packages/nemo_helix_ext/tests/skills/fixtures/observability"

SPANS_URL = "/apis/intake/v2/workspaces/default/spans"
EVALUATORS_URL = "/apis/intake/v2/workspaces/default/evaluator-results"
ANNOTATIONS_URL = "/apis/intake/v2/workspaces/default/annotations"


@pytest.fixture(autouse=True)
def _import_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))


class _TestClientSession:
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        kwargs.pop("follow_redirects", None)
        kwargs.pop("timeout", None)
        parsed = urlsplit(url)
        return self.client.request(method, parsed.path, follow_redirects=False, **kwargs)


@pytest.mark.parametrize("provider", ["mlflow", "langsmith", "phoenix", "braintrust"])
def test_provider_fixture_full_import_is_lossless_and_replay_safe(client: TestClient, provider: str) -> None:
    common = importlib.import_module("_import_common")
    bundle = _bundle(provider)
    _rebase_span_times_inside_retention(bundle)
    golden = _json(f"expected-{provider}.json")
    writer = common.IntakeWriter(
        base_url="http://127.0.0.1:8080",
        workspace="default",
        session=_TestClientSession(client),
    )

    first = writer.write(bundle, batch_size=2)
    second = writer.write(bundle, batch_size=2)

    assert first["annotations"] == len(golden["annotation_kinds"])
    assert second == {**first, "annotations": 0}
    spans_response = client.get(SPANS_URL, params={"filter[source]": provider, "page_size": 1000})
    assert spans_response.status_code == 200, spans_response.text
    spans = spans_response.json()["data"]
    assert len(spans) == golden["span_count"]
    imported = next(item for item in spans if item["span_id"] == golden["span"]["span_id"])
    assert imported["trace_id"] == golden["span"]["trace_id"]
    assert imported["kind"] == golden["span"]["kind"]
    assert imported["status"] == golden["span"]["status"]
    assert json.loads(imported["input"]) == golden["span"]["input"]
    assert json.loads(imported["output"]) == golden["span"]["output"]
    for key, value in golden["read_fields"].items():
        assert imported[key] == value
    raw_attributes = json.loads(imported["raw_attributes"])
    for expectation in golden["raw_paths"]:
        assert _at_path(raw_attributes, expectation["path"]) == expectation["value"]
    assert raw_attributes[f"{provider}.signals"] == golden["signals"]

    evaluator_response = client.get(EVALUATORS_URL, params={"page_size": 1000})
    assert evaluator_response.status_code == 200, evaluator_response.text
    evaluator_names = sorted(item["name"] for item in evaluator_response.json()["data"])
    assert evaluator_names == sorted(golden["evaluator_names"])

    annotation_response = client.get(ANNOTATIONS_URL, params={"page_size": 1000})
    assert annotation_response.status_code == 200, annotation_response.text
    annotation_kinds = [item["kind"] for item in annotation_response.json()["data"]]
    assert sorted(annotation_kinds) == sorted(golden["annotation_kinds"])
    assert len(annotation_kinds) == len(golden["annotation_kinds"])


def _bundle(provider: str) -> Any:
    module = importlib.import_module(f"import_{provider}")
    payload = _payload(provider)
    mapper = getattr(module, f"map_{provider}_export")
    return mapper(payload, project="fixture", include_feedback=True)


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


def _json(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _at_path(value: Any, path: list[str | int]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current


def _rebase_span_times_inside_retention(bundle: Any) -> None:
    """Keep official payloads intact while moving only normalized write times inside ClickHouse TTL."""

    starts = [datetime.fromisoformat(span["started_at"].replace("Z", "+00:00")) for span in bundle.spans]
    source_anchor = min(starts)
    write_anchor = datetime.now(timezone.utc) - timedelta(days=30)
    for span, started_at in zip(bundle.spans, starts, strict=True):
        rebased_start = write_anchor + (started_at - source_anchor)
        ended_at = span.get("ended_at")
        span["started_at"] = rebased_start.isoformat()
        if ended_at is not None:
            parsed_end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            span["ended_at"] = (rebased_start + (parsed_end - started_at)).isoformat()
