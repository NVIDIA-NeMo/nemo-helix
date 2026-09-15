# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the plugin jobs ``api_factory`` filter pipeline.

These tests build a synthetic plugin ``/jobs`` route via ``job_route_factory``
and assert that the unified filter syntax (bracket notation with operators
like ``$like``, ``$gte``, ``$in``) flows through ``make_filter_dep`` correctly
and gets forwarded to the SDK.

Replaces the previous behavior where ``filter[name][$like]=foo`` returned 422
because the local ``get_jobs_list_filter`` dep model-validated against the raw
deep-object dict.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from nemo_platform_plugin.dependencies import get_entity_client, get_sdk_client
from nemo_platform_plugin.jobs.api_factory import job_route_factory
from pydantic import BaseModel, ConfigDict
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _patch_client_from_platform():
    """The factory calls ``client_from_platform(sdk, AsyncJobsClient)``; the test's
    ``_CapturingSdk`` exposes the captured client as ``sdk.jobs_client``, so route
    it through here."""
    with patch(
        "nemo_platform_plugin.jobs.api_factory.client_from_platform",
        side_effect=lambda sdk, _cls: sdk.jobs_client,
    ):
        yield


def _forwarded_filter(sdk: _CapturingSdk) -> dict:
    """Decode the JSON ``filter`` param the factory pushed through ``query_params``.

    The factory bypasses the client's typed ``filter`` handling because the
    querystring serializer mangles ``$and``-style list-of-dict values. It sends
    a JSON-encoded filter via the ``filter`` query param instead — tests parse
    it back here so assertions stay shape-driven, not string-driven.
    """
    return json.loads(sdk.list_kwargs["query_params"]["filter"])


class _Spec(BaseModel):
    foo: str = "bar"


def _fake_compiler(workspace, original_spec, transformed_spec, entity_client, job_name, sdk):
    return {"steps": []}


def _fake_page():
    """A ``PageResult``-like object for the typed client's ``list_jobs().page()``."""
    return SimpleNamespace(
        items=[],
        metadata={
            "page": 1,
            "page_size": 10,
            "current_page_size": 0,
            "total_pages": 1,
            "total_results": 0,
        },
    )


def _fake_list_response():
    return SimpleNamespace(page=_fake_page)


class _CapturingSdk:
    """Captures kwargs passed to ``JobsClient.list_jobs(...)`` for assertion.

    The factory now calls ``client_from_platform(sdk, AsyncJobsClient).list_jobs(...)``.
    ``_build_app`` patches ``client_from_platform`` to return this object's
    ``jobs_client`` so the ``list_jobs`` kwargs are captured here.
    """

    def __init__(self) -> None:
        self.list_kwargs: dict[str, Any] = {}

        async def _list_jobs(**kwargs: Any) -> Any:
            self.list_kwargs = kwargs
            return _fake_list_response()

        self.jobs_client = SimpleNamespace(list_jobs=_list_jobs)


def _build_app() -> tuple[FastAPI, _CapturingSdk]:
    app = FastAPI()
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_Spec,
        platform_job_config_compiler=_fake_compiler,
    )
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")

    sdk = _CapturingSdk()
    app.dependency_overrides[get_sdk_client] = lambda: sdk
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    return app, sdk


class TestPluginJobsFilter:
    """The list_jobs route must accept unified filter operators end-to-end."""

    def test_like_operator_returns_200(self):
        """Studio's ``filter[name][$like]=foo`` must not 422 (the regression)."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[name][$like]": "foo"})
        assert resp.status_code == 200, resp.text
        # User filter is AND-composed with the service source predicate so
        # logical roots like $or stay scoped (a flat dict merge would lose
        # source under a logical root).
        assert _forwarded_filter(sdk) == {"$and": [{"name": {"$like": "foo"}}, {"source": {"$eq": "widgets"}}]}
        # Typed ``filter`` kwarg is intentionally not used — see comment above.
        assert "filter" not in sdk.list_kwargs

    def test_bare_eq_value(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[name]": "exact-name"})
        assert resp.status_code == 200, resp.text
        # Bracket notation with a bare value implicitly becomes $eq.
        assert _forwarded_filter(sdk) == {"$and": [{"name": {"$eq": "exact-name"}}, {"source": {"$eq": "widgets"}}]}

    def test_datetime_gte_and_lte(self):
        app, sdk = _build_app()
        client = TestClient(app)
        ts = datetime(2026, 1, 1).isoformat()
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[created_at][$gte]": ts, "filter[created_at][$lte]": ts},
        )
        assert resp.status_code == 200, resp.text
        forwarded = _forwarded_filter(sdk)
        # Multi-field user filter already lives under $and; source is appended.
        assert "$and" in forwarded
        clauses = forwarded["$and"]
        assert any("created_at" in c and "$gte" in c.get("created_at", {}) for c in clauses)
        assert any("created_at" in c and "$lte" in c.get("created_at", {}) for c in clauses)
        assert {"source": {"$eq": "widgets"}} in clauses

    def test_in_operator_on_status(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[status][$in]": "active,completed"},
        )
        assert resp.status_code == 200, resp.text
        assert _forwarded_filter(sdk) == {
            "$and": [
                {"status": {"$in": ["active", "completed"]}},
                {"source": {"$eq": "widgets"}},
            ]
        }

    def test_unknown_field_is_400(self):
        """Allowlist gate enforced by make_filter_dep — typos should fail loudly."""
        app, _ = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[bogus]": "x"})
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]

    def test_no_filter_still_forwards_source(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs")
        assert resp.status_code == 200, resp.text
        # No user filter — source predicate stands alone (no $and wrap needed).
        assert _forwarded_filter(sdk) == {"source": {"$eq": "widgets"}}

    def test_logical_root_does_not_drop_source(self):
        """Regression: a flat dict merge silently dropped ``source`` when the
        user filter had a logical root ($or/$and/$not). Tree-level $and
        composition keeps source scoped at the conjunction."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter": '{"$or":[{"name":"a"},{"name":"b"}]}'},
        )
        assert resp.status_code == 200, resp.text
        forwarded = _forwarded_filter(sdk)
        assert "$and" in forwarded
        clauses = forwarded["$and"]
        # User's $or is preserved as one branch; source predicate is the other.
        assert any("$or" in c for c in clauses)
        assert {"source": {"$eq": "widgets"}} in clauses

    def test_spec_subpath_is_accepted_and_forwarded(self):
        """``spec`` is a namespace: ``spec.<path>`` addresses the stored job spec."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[spec.target.format]": "generic"},
        )
        assert resp.status_code == 200, resp.text
        assert _forwarded_filter(sdk) == {
            "$and": [
                {"spec.target.format": {"$eq": "generic"}},
                {"source": {"$eq": "widgets"}},
            ]
        }

    def test_spec_subpath_null_branch_survives_a_logical_root(self):
        """The exact shape the Model Evaluations table sends.

        ``$nin`` alone drops rows whose spec names no target, so the null branch is
        what keeps offline runs; it has to survive $and composition with source.
        Passed as a raw JSON ``filter=`` param rather than bracket notation, which
        the query serializer mangles for the array-of-objects a logical operator
        produces.
        """
        app, sdk = _build_app()
        client = TestClient(app)
        user_filter = {
            "$or": [
                {"spec.target.format": {"$nin": ["generic", "nemo_agent_toolkit"]}},
                {"spec.target.format": {"$eq": None}},
            ]
        }
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter": json.dumps(user_filter)})
        assert resp.status_code == 200, resp.text
        clauses = _forwarded_filter(sdk)["$and"]
        assert user_filter in clauses
        assert {"source": {"$eq": "widgets"}} in clauses

    def test_dotted_path_outside_a_namespace_is_400(self):
        """Only declared namespaces take sub-paths; everything else stays flat."""
        app, _ = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[status.value]": "active"})
        assert resp.status_code == 400
        assert "status.value" in resp.json()["detail"]

    def test_response_filter_echoes_user_facing_shape(self):
        app, _ = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[name][$like]": "foo"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Response filter field reflects the user query, not the source-injected forward tree.
        assert body["filter"] == {"name": {"$like": "foo"}}

    def test_text_filter_syntax(self):
        """Raw ``filter=name~"foo"`` text syntax is also accepted."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter": 'name~"foo"'})
        assert resp.status_code == 200, resp.text
        assert _forwarded_filter(sdk) == {"$and": [{"name": {"$like": "foo"}}, {"source": {"$eq": "widgets"}}]}


class TestPluginJobsFilterValueValidation:
    """``make_filter_dep`` validates field names; this layer enforces the
    value-side schema (``status`` enum, ``created_at``/``updated_at``
    operators) so invalid filters fail at the plugin boundary instead of
    silently filtering to zero results or 500-ing in the entity store."""

    def test_invalid_status_enum_value_is_400(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[status]": "definitely-not-a-status"},
        )
        assert resp.status_code == 400
        assert "definitely-not-a-status" in resp.json()["detail"]
        # SDK never invoked — validation runs before the forward call.
        assert sdk.list_kwargs == {}

    def test_invalid_status_value_in_in_list_is_400(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[status][$in]": "active,bogus"},
        )
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]
        assert sdk.list_kwargs == {}

    def test_unsupported_operator_on_status_is_400(self):
        """``$like`` doesn't make sense on an enum field."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[status][$like]": "active"},
        )
        assert resp.status_code == 400
        assert "status" in resp.json()["detail"]
        assert sdk.list_kwargs == {}

    def test_unsupported_operator_on_datetime_is_400(self):
        """``$eq`` / ``$like`` are nonsensical for datetime fields — only range ops."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[created_at][$eq]": "2026-01-01T00:00:00Z"},
        )
        assert resp.status_code == 400
        assert "created_at" in resp.json()["detail"]
        assert sdk.list_kwargs == {}

    def test_invalid_datetime_value_is_400(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[created_at][$gte]": "not-a-timestamp"},
        )
        assert resp.status_code == 400
        assert "created_at" in resp.json()["detail"]
        assert sdk.list_kwargs == {}

    def test_valid_status_in_list_passes(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter[status][$in]": "active,completed"},
        )
        assert resp.status_code == 200, resp.text
        assert _forwarded_filter(sdk) == {
            "$and": [
                {"status": {"$in": ["active", "completed"]}},
                {"source": {"$eq": "widgets"}},
            ]
        }

    def test_in_with_non_list_non_string_value_is_400(self):
        """Raw JSON ``{"status":{"$in":null}}`` — $in expects a list. (String values
        auto-split on commas via ``_normalize_value`` so they don't reach this gate;
        non-string scalars do.)"""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter": '{"status":{"$in":null}}'},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "list" in detail and "status" in detail
        assert sdk.list_kwargs == {}

    def test_eq_with_list_value_is_400(self):
        """Raw JSON ``{"status":{"$eq":["active"]}}`` — $eq expects a scalar."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter": '{"status":{"$eq":["active"]}}'},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "scalar" in detail and "status" in detail
        assert sdk.list_kwargs == {}

    def test_valid_status_with_not_wrap_is_validated(self):
        """Wrap status in ``$not`` — value validation must still fire on the inner clause."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter": '{"$not":{"status":"bogus"}}'},
        )
        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]
        assert sdk.list_kwargs == {}


class TestForwardedFilterSurvivesSdkSerialization:
    """Codex adversarial-review concern: the kwargs-capturing tests above prove
    *what* the factory hands to the typed Jobs client, but not whether the old SDK's
    querystring serializer can encode it onto the wire without mangling.

    The typed client forwards ``filter`` as a single JSON-string query param
    (not through the Stainless deep-object serializer, which mangled
    ``$and``-style list-of-dict values). These tests take that forwarded value
    and run it through ``make_filter_dep``'s parsing path, asserting the
    resulting ``FilterOperation`` tree matches what the plugin composed.

    A regression that reverted to a typed ``filter=`` dict with logical-array
    values would produce repr-joined garbage on the wire and fail to round-trip
    here.
    """

    @staticmethod
    def _round_trip(sdk: _CapturingSdk) -> dict:
        # The migrated factory forwards a single JSON-encoded ``filter`` string
        # via ``query_params`` — no deep-object array serialization involved.
        filter_value = (sdk.list_kwargs.get("query_params") or {}).get("filter")
        assert filter_value is not None, "expected a forwarded filter param"
        # Decode just like core jobs make_filter_dep would: see leading ``{``,
        # route through parse_json_filter.
        from nemo_platform_plugin.api.filter import parse_json_filter

        operation = parse_json_filter(filter_value)
        return operation.to_dict()

    def test_user_filter_round_trip(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs", params={"filter[name][$like]": "foo"})
        assert resp.status_code == 200, resp.text
        assert self._round_trip(sdk) == {"$and": [{"name": {"$like": "foo"}}, {"source": {"$eq": "widgets"}}]}

    def test_logical_root_round_trip(self):
        """The ``$or``-at-root case that originally motivated the JSON-encoding
        switch — list-of-dict values under a logical operator must survive the
        wire encoding."""
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get(
            "/apis/widgets/v2/workspaces/default/jobs",
            params={"filter": '{"$or":[{"name":"a"},{"name":"b"}]}'},
        )
        assert resp.status_code == 200, resp.text
        decoded = self._round_trip(sdk)
        assert "$and" in decoded
        clauses = decoded["$and"]
        # Original $or is preserved verbatim as one branch; source the other.
        assert {"$or": [{"name": {"$eq": "a"}}, {"name": {"$eq": "b"}}]} in clauses
        assert {"source": {"$eq": "widgets"}} in clauses

    def test_no_filter_round_trip(self):
        app, sdk = _build_app()
        client = TestClient(app)
        resp = client.get("/apis/widgets/v2/workspaces/default/jobs")
        assert resp.status_code == 200, resp.text
        assert self._round_trip(sdk) == {"source": {"$eq": "widgets"}}


# ---------------------------------------------------------------------------
# Create-path spec serialization
# ---------------------------------------------------------------------------


class _BlobPayload(BaseModel):
    """Mirrors ``MetricCloudpicklePayload``: a bytes field whose model owns the base64 JSON contract."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    blob: bytes


class _BlobSpec(BaseModel):
    """A job spec that carries raw bytes (e.g. a cloudpickle metric bundle)."""

    blob_payload: _BlobPayload


def _blob_compiler(workspace, original_spec, transformed_spec, entity_client, job_name, sdk):
    # A minimal-but-valid PlatformJobSpec (one step is required).
    return {"steps": [{"name": "s1", "executor": {"provider": "cpu", "container": {"image": "img"}}}]}


class _CreateCapturingSdk:
    """Fake jobs client whose ``create_job`` mirrors the real typed client's transport.

    The typed client serializes the request body with ``body.model_dump_json(exclude_unset=True)``
    (see ``client/endpoint.py``). A spec dumped python-mode leaves raw bytes in the opaque ``spec``
    dict, so that serialization raises ``PydanticSerializationError`` on the non-utf8 pickle marker —
    exactly the create-path bug. Captures the serialized body for assertion.
    """

    def __init__(self) -> None:
        self.body_json: str | None = None

        async def _create_job(*, workspace: str, body: Any) -> Any:
            self.body_json = body.model_dump_json(exclude_unset=True)
            stored = json.loads(self.body_json)
            job_resp = SimpleNamespace(
                id="job-1",
                name=body.name or "job-1",
                description=None,
                workspace=workspace,
                created_at=None,
                updated_at=None,
                spec=stored["spec"],
                status=None,
                status_details=None,
                error_details=None,
                ownership=None,
                custom_fields=None,
            )
            return SimpleNamespace(data=lambda: job_resp)

        self.jobs_client = SimpleNamespace(create_job=_create_job)


def _build_create_app() -> tuple[FastAPI, _CreateCapturingSdk]:
    app = FastAPI()
    router = job_route_factory(
        service_name="widgets",
        job_type="Widget",
        job_input=_BlobSpec,
        platform_job_config_compiler=_blob_compiler,
    )
    app.include_router(router, prefix="/apis/widgets/v2/workspaces/{workspace}")

    sdk = _CreateCapturingSdk()
    app.dependency_overrides[get_sdk_client] = lambda: sdk
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    return app, sdk


class TestPluginJobsCreateSerialization:
    """A created job spec carrying raw bytes (cloudpickle metric bundle) must serialize for transport."""

    def test_create_serializes_spec_with_bytes_payload(self):
        # Regression: the factory coerces the transformed spec to a dict for the request body. A
        # python-mode dump leaves raw bytes that the typed client can't JSON-encode (invalid utf-8 on
        # the pickle marker \x80). The spec is bound for JSON transport + storage, so it must be dumped
        # json-safe — the payload model's ser_json_bytes="base64" then base64-encodes the blob.
        app, sdk = _build_create_app()
        client = TestClient(app)
        raw = b"\x80\x04not-utf8\xff"
        resp = client.post(
            "/apis/widgets/v2/workspaces/default/jobs",
            json={"spec": {"blob_payload": {"blob": base64.b64encode(raw).decode()}}},
        )
        assert resp.status_code == 201, resp.text
        # The body handed to the jobs client serialized cleanly, with the blob base64-encoded (not raw
        # bytes) inside the opaque ``spec`` dict.
        assert sdk.body_json is not None
        stored_blob = json.loads(sdk.body_json)["spec"]["blob_payload"]["blob"]
        assert stored_blob == base64.b64encode(raw).decode()
