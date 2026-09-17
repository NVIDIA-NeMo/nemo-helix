# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Auth service IAM endpoint definitions."""

from typing import get_origin

import pytest
from nemo_platform_plugin.client.types import BinaryContent, Paginated
from nemo_platform_plugin.iam import endpoints
from nemo_platform_plugin.iam.types import (
    AuthzErrorResponse,
    AuthzRequest,
    AuthzResponse,
    RoleBinding,
    RoleBindingDeleteResponse,
    RoleBindingInput,
)
from pydantic import ValidationError


def test_role_binding_endpoint_contracts() -> None:
    listed = endpoints.list_role_bindings(query_params={"filter[principal][$like]": "service:%"})
    created = endpoints.create_role_binding(body=RoleBindingInput(principal="user@example.com", role="Viewer"))
    fetched = endpoints.get_role_binding(name="rb-123")
    revoked = endpoints.revoke_role_binding(name="rb-123")

    assert listed.method == "GET"
    assert listed.path_template == "/apis/auth/v2/iam/role-bindings"
    assert listed.query_params == {"filter[principal][$like]": "service:%"}
    assert get_origin(listed.response_type) is Paginated
    assert created.method == "POST"
    assert created.query_params is None
    assert created.response_type is RoleBinding
    assert fetched.path_params == {"name": "rb-123"}
    assert fetched.response_type is RoleBinding
    assert revoked.method == "DELETE"
    assert revoked.query_params is None
    assert revoked.response_type is RoleBindingDeleteResponse


def test_role_binding_endpoints_send_only_supplied_query_params() -> None:
    """Server-side defaults (page, page_size, sort, wait_role_propagation) are not replicated client-side."""
    assert endpoints.list_role_bindings().query_params is None
    assert endpoints.list_role_bindings(query_params={"page": 2, "filter": 'role:"Viewer"'}).query_params == {
        "page": 2,
        "filter": 'role:"Viewer"',
    }
    assert endpoints.create_role_binding(
        body=RoleBindingInput(principal="user@example.com", role="Viewer"),
        query_params={"wait_role_propagation": False},
    ).query_params == {"wait_role_propagation": False}
    assert endpoints.revoke_role_binding(name="rb-123", query_params={"wait_role_propagation": False}).query_params == {
        "wait_role_propagation": False
    }


def test_authz_and_bundle_endpoint_contracts() -> None:
    request = AuthzRequest(input={"principal_id": "user@example.com"})
    authz = endpoints.evaluate_authorization(entrypoint="allow", body=request)
    bundle = endpoints.get_opa_bundle()

    assert authz.path_template == "/apis/auth/v2/authz/{entrypoint}"
    assert authz.path_params == {"entrypoint": "allow"}
    assert authz.response_type is AuthzResponse
    assert bundle.path_template == "/apis/auth/v2/iam/opa-bundle.tar.gz"
    assert bundle.response_type is BinaryContent
    assert bundle.additional_success_status_codes == (304,)


def test_authz_error_models_fastapi_detail_envelope() -> None:
    error = AuthzErrorResponse.model_validate(
        {"detail": {"error": "Invalid entrypoint: missing", "valid_entrypoints": ["allow"]}}
    )

    assert error.detail.error == "Invalid entrypoint: missing"
    assert error.detail.valid_entrypoints == ["allow"]
    with pytest.raises(ValidationError):
        AuthzErrorResponse.model_validate({"detail": "policy failed"})
