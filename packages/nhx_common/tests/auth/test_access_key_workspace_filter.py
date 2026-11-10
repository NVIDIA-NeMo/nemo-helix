# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth.access_key_workspace_filter import (
    INVALID_FILTER_DETAIL,
    MIXED_FILTER_DETAIL,
    WORKSPACE_FILTER_MISMATCH_DETAIL,
    workspace_filter_error,
)
from nhx.common.auth.client import AuthClient, AuthorizationResult
from nhx.common.auth.dependencies import get_auth_client
from nhx.common.auth.middleware import AuthorizationMiddleware
from nhx.common.config import AuthConfig, Configuration
from starlette.datastructures import QueryParams

MODELS_PATH = "/apis/models/v2/workspaces/team-a/models"


@pytest.fixture(autouse=True)
def _cleanup_config_overrides():
    yield
    Configuration.clear_overrides()


@pytest.mark.parametrize(
    "query",
    [
        "",
        "page=2",
        "filter[workspace]=team-a",
        "filter[name]=llama",
        "filter=" + json.dumps({"workspace": "team-a", "name": "llama"}),
        "filter=" + json.dumps({"$or": [{"$not": {"name": "x"}}, {"workspace": "team-a"}]}),
    ],
    ids=["none", "unrelated", "same-workspace", "other-field", "json-same-workspace", "negation-stays-in-its-branch"],
)
def test_filter_within_path_workspace_is_allowed(query):
    assert workspace_filter_error(MODELS_PATH, QueryParams(query)) is None


@pytest.mark.parametrize(
    "query",
    [
        "filter[workspace]=team-b",
        "filter=" + json.dumps({"workspace": "team-b"}),
        "filter=" + json.dumps({"$or": [{"workspace": "team-a"}, {"workspace": "team-b"}]}),
        "filter=" + json.dumps({"workspace": {"$in": ["team-a"]}}),
        "filter=" + json.dumps({"$not": {"workspace": "team-a"}}),
        "filter=" + json.dumps({"$not": {"$or": [{"workspace": "team-a"}]}}),
        "filter=" + json.dumps({"$not": {"$not": {"workspace": "team-a"}}}),
    ],
    ids=["bracket", "json", "nested-or", "non-eq-operator", "negated-path-workspace", "not-or", "not-not"],
)
def test_filter_naming_another_workspace_is_a_mismatch(query):
    assert workspace_filter_error(MODELS_PATH, QueryParams(query)) == WORKSPACE_FILTER_MISMATCH_DETAIL


@pytest.mark.parametrize(
    "query",
    ["filter][workspace]=team-b", "filte]r[workspace]=team-b"],
    ids=["leading-close-bracket", "bracket-inside-name"],
)
def test_malformed_bracket_keys_the_parser_accepts_are_checked(query):
    """parse_deep_object reads these as filter[workspace], so the guard must too."""
    assert workspace_filter_error(MODELS_PATH, QueryParams(query)) == WORKSPACE_FILTER_MISMATCH_DETAIL


@pytest.mark.parametrize("query", ["filter={not json", "filter=status:active"], ids=["json", "text"])
def test_unparseable_filter_is_refused_as_invalid(query):
    assert workspace_filter_error(MODELS_PATH, QueryParams(query)) == INVALID_FILTER_DETAIL


def test_workspace_filter_on_workspace_less_path_is_a_mismatch():
    assert (
        workspace_filter_error("/apis/models/v2/models", QueryParams("filter[workspace]=team-a"))
        == WORKSPACE_FILTER_MISMATCH_DETAIL
    )


def _app(config: AuthConfig) -> FastAPI:
    app = FastAPI()

    @app.get("/apis/models/v2/workspaces/{workspace}/models")
    async def list_models(workspace: str, auth_client: AuthClient = Depends(get_auth_client)):
        return {"workspace": workspace, "access_key_workspaces": auth_client.access_key_workspaces}

    Configuration.set_override(config)
    app.add_middleware(AuthorizationMiddleware, service_name="models")
    return app


@pytest.mark.parametrize(
    ("access_key_workspaces", "query", "status_code"),
    [
        (["team-a"], "filter[workspace]=team-b", 400),
        (["team-a"], "filter[workspace]=team-a", 200),
        (None, "filter[workspace]=team-b", 200),
    ],
    ids=["limited-other-workspace", "limited-same-workspace", "unlimited-unchanged"],
)
def test_middleware_rejects_workspace_filter_override_for_limited_key(access_key_workspaces, query, status_code):
    config = AuthConfig(enabled=True, policy_decision_point_base_url="http://localhost:8181")
    client = TestClient(_app(config), raise_server_exceptions=False)

    with patch("nhx.common.auth.client.AuthClient.authorize_request") as authorize:
        authorize.return_value = AuthorizationResult(allowed=True, access_key_workspaces=access_key_workspaces)
        response = client.get(f"{MODELS_PATH}?{query}", headers={"X-NHX-Principal-Id": "service-account:ci"})

    assert response.status_code == status_code
    if status_code == 200:
        assert response.json()["access_key_workspaces"] == access_key_workspaces
    else:
        assert "filter[workspace]" in response.json()["detail"]


def test_raw_and_bracket_filters_together_are_refused():
    query = QueryParams("filter=" + json.dumps({"name": "x"}) + "&filter[workspace]=team-b")

    assert workspace_filter_error(MODELS_PATH, query) == MIXED_FILTER_DETAIL
