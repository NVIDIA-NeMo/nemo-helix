# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for model search through the models HTTP API.

These tests exercise the full flow:
  client → models service (parse_json_search → _build_search_query → to_dict())
         → entities service (re-parses the serialized JSON)

This is distinct from the entities integration tests which hit the entities
service directly and never call to_dict(). These tests are the only ones that
catch serialization bugs in LogicalOperation.to_dict() (e.g. the $not bug
where to_dict() produced a list instead of a dict).
"""

import uuid

from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_model(test_clients: ClientContext, name: str, **kwargs) -> dict:
    body = {"name": name, **kwargs}
    response = test_clients.test_client.post(f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models", json=body)
    assert response.status_code == 201, f"Failed to create model: {response.text}"
    return response.json()


def _list_models(test_clients: ClientContext, filter: str) -> list[dict]:
    response = test_clients.test_client.get(
        f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models",
        params={"filter": filter},
    )
    assert response.status_code == 200, f"Filter failed: {response.text}"
    return response.json()["data"]


class TestModelSearchNotNull:
    """Regression tests for $not:$eq:null search through the models HTTP API.

    These tests catch bugs in LogicalOperation.to_dict() because they exercise
    the full round-trip: parse → to_dict() → re-parse in entities service.
    """

    def test_not_null_single_field(self, test_clients: ClientContext):
        """$not:$eq:null on a single field returns only models where that field is set."""
        with_endpoint = _uid("with-ep")
        without_endpoint = _uid("no-ep")

        _create_model(
            test_clients,
            with_endpoint,
            api_endpoint={"url": "http://localhost:8000", "model_id": "llama"},
        )
        _create_model(test_clients, without_endpoint)

        results = _list_models(
            test_clients,
            filter='{"data.api_endpoint":{"$not":{"$eq":null}}}',
        )
        names = {m["name"] for m in results}

        assert with_endpoint in names
        assert without_endpoint not in names

    def test_not_null_combined_with_name_search(self, test_clients: ClientContext):
        """$not:$eq:null ANDed with a name $like — exercises to_dict() round-trip."""
        match = _uid("llama-ep")
        no_ep = _uid("llama-noep")
        other = _uid("mistral-ep")

        _create_model(test_clients, match, api_endpoint={"url": "http://llama/v1", "model_id": "llama"})
        _create_model(test_clients, no_ep)
        _create_model(test_clients, other, api_endpoint={"url": "http://mistral/v1", "model_id": "mistral"})

        results = _list_models(
            test_clients,
            filter=f'{{"$and":[{{"name":{{"$like":"{match}"}}}},{{"data.api_endpoint":{{"$not":{{"$eq":null}}}}}}]}}',
        )
        names = {m["name"] for m in results}

        assert names == {match}

    def test_two_not_nulls_combined(self, test_clients: ClientContext):
        """Two $not:$eq:null conditions ANDed — both must serialize as dicts, not lists."""
        both = _uid("both")
        ep_only = _uid("ep-only")
        prompt_only = _uid("prompt-only")
        neither = _uid("neither")

        _create_model(
            test_clients,
            both,
            api_endpoint={"url": "http://localhost/v1", "model_id": "m"},
            prompt={"system_prompt": "You are helpful."},
        )
        _create_model(test_clients, ep_only, api_endpoint={"url": "http://localhost/v1", "model_id": "m"})
        _create_model(test_clients, prompt_only, prompt={"system_prompt": "You are helpful."})
        _create_model(test_clients, neither)

        results = _list_models(
            test_clients,
            filter='{"$and":[{"data.api_endpoint":{"$not":{"$eq":null}}},{"data.prompt":{"$not":{"$eq":null}}}]}',
        )
        names = {m["name"] for m in results}

        assert both in names
        assert ep_only not in names
        assert prompt_only not in names
        assert neither not in names


class TestModelAdaptersRelationshipFilter:
    """Regression tests for the `adapters` relationship filter through the models HTTP API.

    The `adapters` field is a relationship (child `adapter` entities joined via
    `parent=model.id`), not a JSON column. The models-service filter translation
    must forward `adapters` as-is so the entities service's relationship-aware
    parser can resolve it. Prefixing to `data.adapters` produces a 500
    error.
    """

    def test_adapters_exists_does_not_500(self, test_clients: ClientContext):
        """`$exists` on adapters must forward the relationship key to the entities service."""
        _create_model(test_clients, _uid("no-adapters"))

        results = _list_models(test_clients, filter='{"adapters":{"$exists":true}}')
        assert isinstance(results, list)

    def test_studio_custom_models_default_filter(self, test_clients: ClientContext):
        """Studio's default Custom Models filter: $or of has-base-model or has-adapters."""
        with_base = _uid("with-base")
        without_base = _uid("without-base")

        _create_model(test_clients, with_base, base_model="llama-3-8b")
        _create_model(test_clients, without_base)

        results = _list_models(
            test_clients,
            filter='{"$or":[{"data.base_model":{"$not":{"$eq":null}}},{"adapters":{"$exists":true}}]}',
        )
        names = {m["name"] for m in results}

        assert with_base in names
        assert without_base not in names
