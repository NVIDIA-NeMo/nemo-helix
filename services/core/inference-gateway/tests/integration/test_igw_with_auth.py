# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Inference Gateway AuthZ.

These tests verify:
- Unauthenticated requests are rejected (401) for all three gateway route types
- Viewer can access all gateway routes (inference.gateway.*.exec for proxies, inference.providers.read for provider ready)
- Editor can access all gateway routes (same exec permissions)
- Users without a role in the workspace are denied (403) for all route types including list models
- Read/write scopes (X-NHX-Scopes) are enforced: read-only scopes allow GET but deny POST

Uses the create_test_client pattern with auth_enabled=True and mock provider mode.
Scope tests use patched_authz_data (like models tests) to ensure IGW endpoints have
explicit scope requirements for deterministic testing.
"""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.core.auth.app.bundle import build_authorization_data as _real_build_authorization_data
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.core.models.service import ModelsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    add_mock_provider,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)

# ctx is module-scoped (expensive to boot, auth_enabled=True): keep all classes in this
# file on one xdist worker so they share it instead of each worker re-provisioning it
# from scratch.
pytestmark = pytest.mark.xdist_group("igw_with_auth")


@pytest.fixture(scope="module")
def ctx() -> Generator[ClientContext, None, None]:
    """ClientContext with IGW + Models services (auth enabled, mock provider mode).

    ModelsService is needed because add_mock_provider creates providers via
    the Models API (e.g. /apis/models/v2/workspaces/{workspace}/providers).
    """
    with create_test_client(
        InferenceGatewayService,
        ModelsService,
        auth_enabled=True,
        igw_mock_provider_mode=True,
        client_type=ClientContext,
    ) as ctx:
        yield ctx


def _admin_workspaces(ctx: ClientContext) -> WorkspacesClient:
    """Typed workspaces client acting as the platform admin."""
    return WorkspacesClient(base_url="http://testserver", http_client=ctx.test_client).with_headers(
        {"X-NHX-Principal-Id": TEST_ADMIN_EMAIL, "X-NHX-Principal-Email": TEST_ADMIN_EMAIL}
    )


@pytest.mark.integration
class TestIGWUnauthenticated:
    """Unauthenticated requests should be rejected for all gateway route types."""

    def test_openai_proxy_without_auth_fails(self, ctx: ClientContext):
        response = ctx.test_client.post(
            "/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_openai_list_models_without_auth_fails(self, ctx: ClientContext):
        response = ctx.test_client.get(
            "/apis/inference-gateway/v2/workspaces/default/openai/-/v1/models",
        )
        assert response.status_code == 401

    def test_model_proxy_without_auth_fails(self, ctx: ClientContext):
        response = ctx.test_client.post(
            "/apis/inference-gateway/v2/workspaces/default/model/test-model/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_provider_proxy_without_auth_fails(self, ctx: ClientContext):
        response = ctx.test_client.post(
            "/apis/inference-gateway/v2/workspaces/default/provider/test-provider/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_provider_ready_without_auth_fails(self, ctx: ClientContext):
        response = ctx.test_client.get(
            "/apis/inference-gateway/v2/workspaces/default/provider/test-provider/ready",
        )
        assert response.status_code == 401


MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
}


@pytest.mark.integration
class TestIGWViewerAccess:
    """Viewer role should be able to access all gateway routes."""

    def test_viewer_can_list_openai_models(self, ctx: ClientContext):
        workspace = short_unique_name("igw-vl")
        viewer_email = unique_email("viewer")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=viewer_email,
            roles=["Viewer"],
        )
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers={
                "X-NHX-Principal-Id": viewer_email,
                "X-NHX-Principal-Email": viewer_email,
            },
        )
        assert response.status_code == 200

    def test_viewer_can_call_openai_chat_completions(self, ctx: ClientContext):
        workspace = short_unique_name("igw-vc")
        viewer_email = unique_email("viewer")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=viewer_email,
            roles=["Viewer"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": viewer_email,
                "X-NHX-Principal-Email": viewer_email,
            },
        )
        assert response.status_code == 200

    def test_viewer_can_call_model_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-vm")
        viewer_email = unique_email("viewer")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=viewer_email,
            roles=["Viewer"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/v1/chat/completions",
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": viewer_email,
                "X-NHX-Principal-Email": viewer_email,
            },
        )
        assert response.status_code == 200

    def test_viewer_can_call_provider_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-vp")
        viewer_email = unique_email("viewer")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        provider = add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name="test-prov",
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=viewer_email,
            roles=["Viewer"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": viewer_email,
                "X-NHX-Principal-Email": viewer_email,
            },
        )
        assert response.status_code == 200

    def test_viewer_can_check_provider_ready(self, ctx: ClientContext):
        workspace = short_unique_name("igw-vr")
        viewer_email = unique_email("viewer")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        provider = add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name="ready-prov",
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=viewer_email,
            roles=["Viewer"],
        )
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/ready",
            headers={
                "X-NHX-Principal-Id": viewer_email,
                "X-NHX-Principal-Email": viewer_email,
            },
        )
        # 200 (ready), 404 (not ready), or 503 — either way, not 401/403
        assert response.status_code in (200, 404, 503)


@pytest.mark.integration
class TestIGWEditorAccess:
    """Editor role should be able to access all gateway routes (same as Viewer for exec)."""

    def test_editor_can_list_openai_models(self, ctx: ClientContext):
        workspace = short_unique_name("igw-el")
        editor_email = unique_email("editor")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers={
                "X-NHX-Principal-Id": editor_email,
                "X-NHX-Principal-Email": editor_email,
            },
        )
        assert response.status_code == 200

    def test_editor_can_call_openai_chat_completions(self, ctx: ClientContext):
        workspace = short_unique_name("igw-ec")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": editor_email,
                "X-NHX-Principal-Email": editor_email,
            },
        )
        assert response.status_code == 200

    def test_editor_can_call_model_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-em")
        editor_email = unique_email("editor")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/v1/chat/completions",
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": editor_email,
                "X-NHX-Principal-Email": editor_email,
            },
        )
        assert response.status_code == 200

    def test_editor_can_call_provider_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-ep")
        editor_email = unique_email("editor")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        provider = add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name="test-editor-prov",
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": editor_email,
                "X-NHX-Principal-Email": editor_email,
            },
        )
        assert response.status_code == 200

    def test_editor_can_check_provider_ready(self, ctx: ClientContext):
        workspace = short_unique_name("igw-er")
        editor_email = unique_email("editor")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        provider = add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name="ready-editor-prov",
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(
            admin_sdk,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/ready",
            headers={
                "X-NHX-Principal-Id": editor_email,
                "X-NHX-Principal-Email": editor_email,
            },
        )
        # 200 (ready), 404 (not ready), or 503 — either way, not 401/403
        assert response.status_code in (200, 404, 503)


@pytest.mark.integration
class TestIGWUnauthorizedWorkspace:
    """Users without a role in the workspace should be denied (403) on all gateway route types."""

    def test_no_role_denied_openai_list_models(self, ctx: ClientContext):
        workspace = short_unique_name("igw-nl")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403

    def test_no_role_denied_openai_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-no")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403

    def test_no_role_denied_model_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-nm")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/model/any-model/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403

    def test_no_role_denied_provider_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-np")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/any-provider/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403

    def test_no_role_denied_provider_ready(self, ctx: ClientContext):
        workspace = short_unique_name("igw-nr")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/any-provider/ready",
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403


# --- Scope test helpers (mirrors models test pattern with patched_authz_data) ---

# Scope strings for X-NHX-Scopes header (space-separated per OAuth2)
SCOPES_READ_ONLY = "inference:read platform:read"
SCOPES_READ_WRITE = "inference:read inference:write platform:read platform:write"


async def _build_authorization_data_igw_scope_explicit(entities_client=None):
    """Ensure IGW endpoints have explicit scope requirements for scope tests.

    Uses real build but explicitly sets IGW endpoint scopes so tests are deterministic
    and independent of static-authz.yaml drift. Matches production config:
    - GET (list models, provider ready): inference:read, platform:read
    - POST (chat completions, etc.): inference:write, platform:write
    """
    data = await _real_build_authorization_data(entities_client)
    endpoints = data["authz"]["endpoints"]

    # IGW OpenAI list models (GET)
    pattern = "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models"
    if pattern in endpoints and "get" in endpoints[pattern]:
        endpoints[pattern]["get"]["scopes"] = ["inference:read", "platform:read"]

    # IGW OpenAI trailing (POST/PUT/PATCH/DELETE)
    pattern = "/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}"
    if pattern in endpoints:
        for method in ("post", "put", "patch", "delete"):
            if method in endpoints[pattern]:
                endpoints[pattern][method]["scopes"] = ["inference:write", "platform:write"]

    # IGW provider ready (GET)
    pattern = "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/ready"
    if pattern in endpoints and "get" in endpoints[pattern]:
        endpoints[pattern]["get"]["scopes"] = ["inference:read", "platform:read"]

    return data


@contextmanager
def patched_authz_data(build_fn):
    """Patch build_authorization_data in both the bundle and embedded PDP modules."""
    with (
        patch("nhx.core.auth.app.bundle.build_authorization_data", side_effect=build_fn),
        patch("nhx.core.auth.app.embedded_pdp.data.build_authorization_data", side_effect=build_fn),
    ):
        yield


def _auth_headers(email: str, scopes: str | None = None) -> dict[str, str]:
    """Build auth headers for IGW requests, optionally including scopes."""
    h: dict[str, str] = {
        "X-NHX-Principal-Id": email,
        "X-NHX-Principal-Email": email,
    }
    if scopes:
        h["X-NHX-Scopes"] = scopes
    return h


@pytest.mark.integration
class TestIGWScopeChecks:
    """Verify read/write scopes (X-NHX-Scopes) are enforced for IGW routes.

    Uses patched_authz_data (like models granular permission tests) to ensure IGW
    endpoints have explicit scope requirements. When X-NHX-Scopes is present with
    platform scopes, the PDP validates them:
    - GET (list models, provider ready): requires inference:read, platform:read
    - POST (chat completions, etc.): requires inference:write, platform:write
    """

    def test_read_only_scopes_allow_get_list_models(self, ctx: ClientContext):
        with patched_authz_data(_build_authorization_data_igw_scope_explicit):
            workspace = short_unique_name("igw-sr")
            viewer_email = unique_email("viewer")

            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=viewer_email,
                roles=["Viewer"],
            )

            response = ctx.test_client.get(
                f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
                headers=_auth_headers(viewer_email, SCOPES_READ_ONLY),
            )
            assert response.status_code == 200

    def test_read_only_scopes_deny_post_chat_completions(self, ctx: ClientContext):
        with patched_authz_data(_build_authorization_data_igw_scope_explicit):
            workspace = short_unique_name("igw-sw")
            viewer_email = unique_email("viewer")
            model_name = short_unique_name("mdl")

            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
            add_mock_provider(
                admin_sdk,
                workspace=workspace,
                name=model_name,
                mock_response_body=MOCK_CHAT_RESPONSE,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=viewer_email,
                roles=["Viewer"],
            )

            response = ctx.test_client.post(
                f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
                json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers(viewer_email, SCOPES_READ_ONLY),
            )
            assert response.status_code == 403

    def test_read_write_scopes_allow_post_chat_completions(self, ctx: ClientContext):
        with patched_authz_data(_build_authorization_data_igw_scope_explicit):
            workspace = short_unique_name("igw-srw")
            viewer_email = unique_email("viewer")
            model_name = short_unique_name("mdl")

            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
            add_mock_provider(
                admin_sdk,
                workspace=workspace,
                name=model_name,
                mock_response_body=MOCK_CHAT_RESPONSE,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=viewer_email,
                roles=["Viewer"],
            )

            response = ctx.test_client.post(
                f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
                json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers(viewer_email, SCOPES_READ_WRITE),
            )
            assert response.status_code == 200

    def test_read_only_scopes_allow_provider_ready(self, ctx: ClientContext):
        with patched_authz_data(_build_authorization_data_igw_scope_explicit):
            workspace = short_unique_name("igw-spr")
            viewer_email = unique_email("viewer")

            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
            provider = add_mock_provider(
                admin_sdk,
                workspace=workspace,
                name="ready-prov",
                mock_response_body=MOCK_CHAT_RESPONSE,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=viewer_email,
                roles=["Viewer"],
            )

            response = ctx.test_client.get(
                f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/ready",
                headers=_auth_headers(viewer_email, SCOPES_READ_ONLY),
            )
            # 200 (ready), 404 (not ready), or 503 — read-only scopes allow the call
            assert response.status_code in (200, 404, 503)


@pytest.mark.integration
class TestIGWServicePrincipalAccess:
    """Service principals (service:*) are evaluated by the PDP; policy allows these calls.

    The evaluator calls the IGW with X-NHX-Principal-Id: service:evaluator instead of
    a user JWT. These tests verify that service principals are allowed through without
    workspace membership and without a Bearer token.
    """

    SERVICE_PRINCIPAL_EVALUATOR = "service:evaluator"

    def test_service_principal_can_list_openai_models(self, ctx: ClientContext):
        workspace = short_unique_name("igw-svc-l")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        # Intentionally no workspace membership granted to service:evaluator

        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers={"X-NHX-Principal-Id": self.SERVICE_PRINCIPAL_EVALUATOR},
        )
        assert response.status_code == 200

    def test_service_principal_can_call_openai_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-svc-o")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        # Intentionally no workspace membership granted to service:evaluator

        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-NHX-Principal-Id": self.SERVICE_PRINCIPAL_EVALUATOR},
        )
        assert response.status_code == 200

    def test_service_principal_can_call_model_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-svc-m")
        model_name = short_unique_name("mdl")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )

        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/v1/chat/completions",
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-NHX-Principal-Id": self.SERVICE_PRINCIPAL_EVALUATOR},
        )
        assert response.status_code == 200

    def test_service_principal_can_call_provider_proxy(self, ctx: ClientContext):
        workspace = short_unique_name("igw-svc-p")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        provider = add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name="svc-prov",
            mock_response_body=MOCK_CHAT_RESPONSE,
        )

        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider.name}/-/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-NHX-Principal-Id": self.SERVICE_PRINCIPAL_EVALUATOR},
        )
        assert response.status_code == 200

    def test_regular_user_without_role_is_still_denied(self, ctx: ClientContext):
        """Contrast test: a non-service principal without workspace role is still denied."""
        workspace = short_unique_name("igw-svc-d")
        norole_email = unique_email("norole")

        as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()

        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers={
                "X-NHX-Principal-Id": norole_email,
                "X-NHX-Principal-Email": norole_email,
            },
        )
        assert response.status_code == 403


@pytest.mark.integration
class TestIGWDelegatedServicePrincipalAccess:
    """A *delegated* service principal (service:* + on-behalf-of) is scoped to the OBO user.

    Unlike a bare service principal (which takes the ServiceSystem bypass), a service
    principal that carries X-NHX-Principal-On-Behalf-Of must only reach workspaces the
    delegated user can reach. This is the agent-deployment case: the deployed agent's
    auth-proxy sidecar stamps service:agents on-behalf-of the deployment's creator.
    """

    SERVICE_PRINCIPAL_AGENTS = "service:agents"

    @staticmethod
    def _delegated_headers(on_behalf_of: str) -> dict[str, str]:
        return {
            "X-NHX-Principal-Id": TestIGWDelegatedServicePrincipalAccess.SERVICE_PRINCIPAL_AGENTS,
            "X-NHX-Principal-On-Behalf-Of": on_behalf_of,
            "X-NHX-Principal-On-Behalf-Of-Email": on_behalf_of,
        }

    def test_delegated_denied_when_obo_user_lacks_role(self, ctx: ClientContext):
        # The OBO user has NO role in the workspace: the service bypass must not apply.
        workspace = short_unique_name("igw-obo-d")
        obo_email = unique_email("obo-norole")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=short_unique_name("mdl"),
            mock_response_body=MOCK_CHAT_RESPONSE,
        )

        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers=self._delegated_headers(obo_email),
        )
        assert response.status_code == 403

    def test_delegated_allowed_when_obo_user_has_role(self, ctx: ClientContext):
        # The OBO user is granted a role in the workspace: the delegated call is allowed.
        workspace = short_unique_name("igw-obo-a")
        obo_email = unique_email("obo-viewer")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(admin_sdk, workspace=workspace, principal=obo_email, roles=["Viewer"])

        response = ctx.test_client.get(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models",
            headers=self._delegated_headers(obo_email),
        )
        assert response.status_code == 200

    def test_delegated_openai_proxy_denied_when_obo_user_lacks_role(self, ctx: ClientContext):
        workspace = short_unique_name("igw-obo-po")
        model_name = short_unique_name("mdl")
        obo_email = unique_email("obo-norole")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )

        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
            headers=self._delegated_headers(obo_email),
        )
        assert response.status_code == 403

    def test_delegated_openai_proxy_allowed_when_obo_user_has_role(self, ctx: ClientContext):
        workspace = short_unique_name("igw-obo-pa")
        model_name = short_unique_name("mdl")
        obo_email = unique_email("obo-editor")

        admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
        _admin_workspaces(ctx).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        add_mock_provider(
            admin_sdk,
            workspace=workspace,
            name=model_name,
            mock_response_body=MOCK_CHAT_RESPONSE,
        )
        grant_workspace_role(admin_sdk, workspace=workspace, principal=obo_email, roles=["Editor"])

        response = ctx.test_client.post(
            f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions",
            json={"model": f"{workspace}/{model_name}", "messages": [{"role": "user", "content": "hi"}]},
            headers=self._delegated_headers(obo_email),
        )
        assert response.status_code == 200
