# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route-level tests for the /tasks CRUD endpoints.

Drives the real FastAPI router + TaskService through a TestClient with an in-memory entity store.
Covers route wiring, the get_task_service dependency, and status-code mapping (201/204/404/409/422).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evaluator.api.dependencies import get_task_service
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    MetricInline,
    MetricRef,
    TaskInput,
    TaskInputs,
)
from nemo_evaluator.api.service.task_service import TaskService
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.api.v2 import tasks as tasks_routes
from nemo_helix_plugin.entity_client import NemoEntityConflictError


class _FakeMetricService:
    """Normalizes inline metrics to derived refs and resolves the ``default/stored-metric`` ref the
    route bodies submit, so task-create metric-ref validation passes for the happy-path tests."""

    async def store_derived_metric(self, metric: MetricInline, *, workspace: str) -> MetricRef:
        return MetricRef(f"{workspace}/derived.{metric.payload.digest}")

    async def get_metric(self, workspace: str, name: str) -> object | None:
        return object() if workspace == "default" and name in {"stored-metric", "other-metric"} else None


@pytest.fixture(autouse=True)
def verified_archive_projection(monkeypatch):
    # These tests isolate entity/API semantics; real archive verification has separate tests.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor.archive import NativeTask

    async def verify(spec, files_client):
        return NativeTask("fixture", spec.instruction, spec.config)

    monkeypatch.setattr("nemo_evaluator.api.service.task_service.verify_definition", verify)


@pytest.fixture
def client(entity_store) -> TestClient:
    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/v2/workspaces/{workspace}")
    service = TaskService(entity_store, _FakeMetricService(), files_client=Mock())
    app.dependency_overrides[get_task_service] = lambda: service
    return TestClient(app)


def _body(*, intent: str = "Answer the question.", tags: list[str] | None = None) -> dict:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=intent,
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric")],
        ),
        tags=tags or [],
    ).model_dump(mode="json")


_BASE = "/v2/workspaces/default/tasks"


def test_create_then_get(client: TestClient) -> None:
    resp = client.post(f"{_BASE}/task-1", json=_body())
    assert resp.status_code == 201
    assert resp.json()["name"] == "task-1"

    got = client.get(f"{_BASE}/task-1")
    assert got.status_code == 200
    body = got.json()
    assert body["spec"]["intent"] == "Answer the question."
    assert body["spec"]["metrics"] == ["default/stored-metric"]  # MetricRef serializes to a bare string


def test_create_preserves_arbitrary_input_fields(client: TestClient) -> None:
    body = _body()
    gym_row = {"input": [{"role": "user", "content": "What is 2+2?"}], "temperature": 0.2}
    body["spec"]["inputs"]["gym_row"] = gym_row

    response = client.post(f"{_BASE}/task-1", json=body)

    assert response.status_code == 201
    assert response.json()["spec"]["inputs"]["gym_row"] == gym_row


def test_create_preserves_structured_metadata_value(client: TestClient) -> None:
    body = _body()
    extras = {"expected": "B", "scores": [1.0, None], "verified": True}
    body["metadata"] = [{"key": "gym_row_extras", "value": extras}]

    response = client.post(f"{_BASE}/task-1", json=body)

    assert response.status_code == 201
    assert response.json()["metadata"] == [{"key": "gym_row_extras", "value": extras}]


def test_create_rejects_duplicate_metadata_keys(client: TestClient) -> None:
    # metadata is a key→value map as a list; duplicate keys are a 422, not a silent last-wins collapse.
    body = _body()
    body["metadata"] = [{"key": "suite", "value": "smoke"}, {"key": "suite", "value": "regression"}]
    assert client.post(f"{_BASE}/task-1", json=body).status_code == 422


def test_create_missing_metric_ref_returns_422(client: TestClient) -> None:
    body = _body()
    body["spec"]["metrics"] = ["default/missing-metric"]
    assert client.post(f"{_BASE}/task-1", json=body).status_code == 422


@pytest.mark.parametrize("method", ["post", "put"])
def test_write_without_a_spec_kind_returns_422(client: TestClient, method: str) -> None:
    """``spec`` is a discriminated union, so a raw body that omits ``kind`` has no variant to
    validate against. ``kind`` is therefore required in both definitions — a schema that defaulted
    it would tell a generated client it may be omitted, and every such request would 422."""
    body = _body()
    del body["spec"]["kind"]
    response = getattr(client, method)(f"{_BASE}/task-1", json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "union_tag_not_found"


@pytest.mark.parametrize("definition", [EvaluatorTaskDefinition, HarborTaskDefinition])
def test_published_schema_requires_the_kind_discriminator(definition: type) -> None:
    """The generated spec has to agree with the validator above: a default on ``kind`` would leave
    it out of ``required``, and a client generated from that spec would omit it."""
    assert "kind" in definition.model_json_schema()["required"]


def test_create_duplicate_returns_409(client: TestClient) -> None:
    assert client.post(f"{_BASE}/task-1", json=_body()).status_code == 201
    assert client.post(f"{_BASE}/task-1", json=_body()).status_code == 409


def test_create_rejects_invalid_name(client: TestClient) -> None:
    # NAME_PATTERN forbids slashes/spaces.
    assert client.post(f"{_BASE}/bad name", json=_body()).status_code == 422


def test_get_missing_returns_404(client: TestClient) -> None:
    assert client.get(f"{_BASE}/nope").status_code == 404


def test_list_returns_created_tasks(client: TestClient) -> None:
    client.post(f"{_BASE}/a", json=_body())
    client.post(f"{_BASE}/b", json=_body())

    resp = client.get(_BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert {t["name"] for t in body["data"]} == {"a", "b"}
    assert body["pagination"]["total_results"] == 2


def test_delete_then_get_404(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body())
    assert client.delete(f"{_BASE}/task-1").status_code == 204
    assert client.get(f"{_BASE}/task-1").status_code == 404


def test_delete_missing_returns_404(client: TestClient) -> None:
    assert client.delete(f"{_BASE}/nope").status_code == 404


def test_delete_conflict_returns_409() -> None:
    class _Service:
        async def delete_task(self, workspace: str, name: str) -> bool:
            raise NemoEntityConflictError("changed")

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_task_service] = lambda: _Service()
    client = TestClient(app)

    assert client.delete(f"{_BASE}/task-1").status_code == 409


# --- Publishing revisions (POST creates, PUT replaces) ------------------------


def test_create_publishes_revision_one(client: TestClient) -> None:
    """Every stored task has a revision from the moment it exists — there is no unpublished head,
    so no consumer has to define what one would mean."""
    created = client.post(f"{_BASE}/task-1", json=_body()).json()
    assert created["revision"] == 1
    assert created["tags"] == {"latest": 1}


def test_put_on_missing_task_creates_it(client: TestClient) -> None:
    """Upsert, so a publisher makes one call without first checking existence — that check is both
    a round trip and a race between two publishers of the same task."""
    response = client.put(f"{_BASE}/task-1", json=_body())
    assert response.status_code == 201
    assert response.json()["revision"] == 1


def test_put_with_changed_content_publishes_a_new_revision(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body())
    response = client.put(f"{_BASE}/task-1", json=_body(intent="Do something else."))
    assert response.status_code == 201
    assert response.json()["revision"] == 2
    assert response.json()["tags"]["latest"] == 2


def test_put_with_identical_content_publishes_nothing(client: TestClient) -> None:
    """Strict idempotency: PUT twice leaves exactly the same state, and says so with 200."""
    client.post(f"{_BASE}/task-1", json=_body())
    response = client.put(f"{_BASE}/task-1", json=_body())
    assert response.status_code == 200
    assert response.json()["revision"] == 1


def test_put_applies_tags_without_publishing(client: TestClient) -> None:
    """Re-PUTting unchanged content is how an existing revision gets tagged."""
    client.post(f"{_BASE}/task-1", json=_body())
    response = client.put(f"{_BASE}/task-1", json=_body(tags=["blessed"]))
    assert response.status_code == 200
    assert response.json()["tags"] == {"latest": 1, "blessed": 1}


def test_post_on_existing_task_still_conflicts(client: TestClient) -> None:
    """POST keeps its guard: a typo'd name must not silently overwrite someone else's task."""
    client.post(f"{_BASE}/task-1", json=_body())
    assert client.post(f"{_BASE}/task-1", json=_body(intent="Different.")).status_code == 409


def test_get_returns_the_current_revision(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body())
    client.put(f"{_BASE}/task-1", json=_body(intent="Do something else."))
    got = client.get(f"{_BASE}/task-1").json()
    assert got["revision"] == 2
    assert got["spec"]["intent"] == "Do something else."


# --- Reading and tagging a specific revision ---------------------------------


def test_list_revisions_newest_first(client: TestClient) -> None:
    """How a caller discovers what it can pin to."""
    client.post(f"{_BASE}/task-1", json=_body())
    client.put(f"{_BASE}/task-1", json=_body(intent="Second."))

    revisions = client.get(f"{_BASE}/task-1/revisions").json()["data"]
    assert [r["revision"] for r in revisions] == [2, 1]
    assert all(len(r["content_hash"]) == 64 for r in revisions)
    assert revisions[0]["tags"] == ["latest"]


def test_list_revisions_missing_task_returns_404(client: TestClient) -> None:
    assert client.get(f"{_BASE}/nope/revisions").status_code == 404


def test_get_by_digest_returns_the_published_content(client: TestClient) -> None:
    """The point of pinning: a consumer holding a digest reads what was published, not what is
    current."""
    first = client.post(f"{_BASE}/task-1", json=_body()).json()
    digest = client.get(f"{_BASE}/task-1/revisions").json()["data"][0]["content_hash"]
    client.put(f"{_BASE}/task-1", json=_body(intent="Newer."))

    pinned = client.get(f"{_BASE}/task-1/revisions/{digest}").json()
    assert pinned["spec"]["intent"] == first["spec"]["intent"]
    assert pinned["revision"] == 1
    assert client.get(f"{_BASE}/task-1").json()["spec"]["intent"] == "Newer."


def test_get_by_tag_resolves(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body(tags=["blessed"]))
    client.put(f"{_BASE}/task-1", json=_body(intent="Newer."))
    assert client.get(f"{_BASE}/task-1/revisions/blessed").json()["revision"] == 1


def test_get_by_unknown_revision_returns_404(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body())
    assert client.get(f"{_BASE}/task-1/revisions/{'c' * 64}").status_code == 404


def test_tag_an_existing_revision(client: TestClient) -> None:
    """Blessing a revision usually happens after it has been evaluated, not at publish time."""
    client.post(f"{_BASE}/task-1", json=_body())
    digest = client.get(f"{_BASE}/task-1/revisions").json()["data"][0]["content_hash"]
    client.put(f"{_BASE}/task-1", json=_body(intent="Newer."))

    tagged = client.put(f"{_BASE}/task-1/tags/blessed", params={"revision": digest})
    assert tagged.status_code == 200
    assert tagged.json()["tags"]["blessed"] == 1
    assert tagged.json()["tags"]["latest"] == 2, "tagging must not disturb latest"


def test_cannot_move_latest_by_hand(client: TestClient) -> None:
    """``latest`` is machine-managed; moving it would break the forward-only guarantee."""
    client.post(f"{_BASE}/task-1", json=_body())
    digest = client.get(f"{_BASE}/task-1/revisions").json()["data"][0]["content_hash"]
    assert client.put(f"{_BASE}/task-1/tags/latest", params={"revision": digest}).status_code == 422


def test_tag_unknown_revision_returns_404(client: TestClient) -> None:
    client.post(f"{_BASE}/task-1", json=_body())
    assert client.put(f"{_BASE}/task-1/tags/blessed", params={"revision": "c" * 64}).status_code == 404


def test_concurrent_replace_returns_409_not_500(entity_store) -> None:
    """A lost optimistic lock is a retryable client conflict. Before this was mapped it fell to the
    catch-all and surfaced as a 500, wrongly implying a server fault."""

    async def _stale(entity, *, original_name=None):
        raise NemoEntityConflictError("modified by another request")

    app = FastAPI()
    app.include_router(tasks_routes.router, prefix="/v2/workspaces/{workspace}")
    service = TaskService(entity_store, _FakeMetricService(), files_client=Mock())
    app.dependency_overrides[get_task_service] = lambda: service
    client = TestClient(app)

    client.post(f"{_BASE}/task-1", json=_body())
    entity_store.update = _stale

    assert client.put(f"{_BASE}/task-1", json=_body(intent="Newer.")).status_code == 409


def test_list_includes_harbor_tasks(client: TestClient) -> None:
    """Both kinds are one record type, so the listing must serialize either."""
    client.post(f"{_BASE}/evaluator-task", json=_body())
    client.post(
        f"{_BASE}/harbor-task",
        json=TaskInput(
            spec=HarborTaskDefinition(
                kind="harbor",
                native_task_id="fixture",
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                source=HarborArchiveSource(
                    fileset_ref="default/harbor#packages/o-n/abc/files",
                    files_hash="a" * 64,
                ),
            )
        ).model_dump(mode="json"),
    )

    response = client.get(_BASE)

    assert response.status_code == 200
    assert {t["name"]: t["spec"]["kind"] for t in response.json()["data"]} == {
        "evaluator-task": "evaluator",
        "harbor-task": "harbor",
    }


def _with_metadata(body: dict, **metadata: object) -> dict:
    return {**body, "metadata": [{"key": key, "value": value} for key, value in metadata.items()]}


@pytest.fixture
def filterable(client: TestClient) -> TestClient:
    alice = _with_metadata(
        _body(intent="Score the arithmetic answer.", tags=["v1.2"]), owner="alice", suite="smoke", level=1
    )
    alice["spec"]["metrics"] = ["default/stored-metric", "default/other-metric"]
    client.post(f"{_BASE}/alice-smoke", json=alice)
    client.post(f"{_BASE}/bob-smoke", json=_with_metadata(_body(), owner="bob", suite="smoke", level=2))
    client.post(f"{_BASE}/untagged", json=_body())
    client.post(
        f"{_BASE}/alice-harbor",
        json=_with_metadata(
            TaskInput(
                spec=HarborTaskDefinition(
                    kind="harbor",
                    native_task_id="fixture",
                    harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                    source=HarborArchiveSource(
                        fileset_ref="default/harbor#packages/o-n/abc/files",
                        files_hash="a" * 64,
                    ),
                )
            ).model_dump(mode="json"),
            owner="alice",
        ),
    )
    return client


def _names(client: TestClient, params: dict[str, str]) -> set[str]:
    response = client.get(_BASE, params=params)
    assert response.status_code == 200, response.text
    return {task["name"] for task in response.json()["data"]}


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"filter[kind]": "harbor"}, {"alice-harbor"}),
        ({"filter[kind]": "evaluator"}, {"alice-smoke", "bob-smoke", "untagged"}),
        ({"filter[metadata.owner]": "alice"}, {"alice-smoke", "alice-harbor"}),
        ({"filter[metadata][owner]": "alice"}, {"alice-smoke", "alice-harbor"}),
        ({"filter[metadata][owner]": "alice", "filter[metadata][suite]": "smoke"}, {"alice-smoke"}),
        ({"filter[metadata.owner][$in]": "alice,bob"}, {"alice-smoke", "bob-smoke", "alice-harbor"}),
        ({"filter[metadata][owner][$in]": "alice,bob"}, {"alice-smoke", "bob-smoke", "alice-harbor"}),
        ({"filter": '{"metadata": {"owner": {"$in": ["bob"]}, "suite": "smoke"}}'}, {"bob-smoke"}),
        ({"filter[kind]": "evaluator", "filter[metadata.owner]": "alice"}, {"alice-smoke"}),
        ({"filter": '{"metadata.level": 2}'}, {"bob-smoke"}),
        ({"filter": '{"$not": {"metadata.owner": "alice"}}'}, {"bob-smoke", "untagged"}),
        ({"filter[metadata.owner]": "carol"}, set()),
        ({"filter[metadata.owner][$like]": "ALI"}, {"alice-smoke", "alice-harbor"}),
        ({"filter[metadata][owner][$startsWith]": "al"}, {"alice-smoke", "alice-harbor"}),
        ({"filter[metadata.owner][$endsWith]": "ob"}, {"bob-smoke"}),
        ({"filter": '{"metadata.level": {"$gt": 1}}'}, {"bob-smoke"}),
        ({"filter[metadata.owner][$nin]": "alice"}, {"bob-smoke"}),
        ({"filter[metadata][owner][$in]": "alice,bob"}, {"alice-smoke", "bob-smoke", "alice-harbor"}),
    ],
)
def test_list_filters_by_kind_and_metadata(filterable: TestClient, params: dict[str, str], expected: set[str]) -> None:
    assert _names(filterable, params) == expected


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"filter[intent][$like]": "arithmetic"}, {"alice-smoke"}),
        ({"filter[native_task_id]": "fixture"}, {"alice-harbor"}),
        ({"filter[metrics]": "default/other-metric"}, {"alice-smoke"}),
        ({"filter[metrics]": "other-metric"}, {"alice-smoke"}),
        ({"filter[metrics]": "default/stored-metric"}, {"alice-smoke", "bob-smoke", "untagged"}),
        ({"filter[metrics]": "elsewhere/other-metric"}, set()),
        ({"filter[tags]": "v1.2"}, {"alice-smoke"}),
        ({"filter[tags]": "v1"}, set()),
        ({"filter[tags]": "latest"}, {"alice-smoke", "bob-smoke", "untagged", "alice-harbor"}),
        ({"filter[tags][$in]": "v1.2,nope"}, {"alice-smoke"}),
    ],
)
def test_list_filters_by_intent_harbor_id_metric_and_tag(
    filterable: TestClient, params: dict[str, str], expected: set[str]
) -> None:
    """A bare metric ref takes the path workspace, and a dotted tag is one tag, not a ``v1`` prefix."""
    assert _names(filterable, params) == expected


def test_metadata_filter_matches_key_and_value_on_the_same_annotation(filterable: TestClient) -> None:
    """``alice-smoke`` has owner=alice and suite=smoke; neither annotation is ``suite=alice``."""
    assert _names(filterable, {"filter[metadata.suite]": "alice"}) == set()


@pytest.mark.parametrize(
    "params",
    [
        {"filter[metadata.owner][$contains]": "ali"},
        {"filter[metadata][owner][$hasKey]": "ali"},
        {"filter": '{"metadata": {"owner": {"$bogus": "x"}}}'},
        {"filter[metadata.owner][$in]": ""},
        {"filter[metadata]": "alice"},
        {"filter[metadata.]": "alice"},
        {"filter": '{"metadata.owner": {"nested": "object"}}'},
        {"filter[spec.kind]": "harbor"},
        {"filter[metrics][$like]": "metric"},
        {"filter[tags][$like]": "v1"},
        {"filter": '{"tags": 3}'},
        {"filter[tasks]": "default/task-a"},
    ],
)
def test_list_rejects_unsupported_metadata_and_field_filters(client: TestClient, params: dict[str, str]) -> None:
    assert client.get(_BASE, params=params).status_code == 400
