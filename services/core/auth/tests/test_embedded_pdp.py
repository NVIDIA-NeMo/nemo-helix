# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WASM policy engine."""

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, ClassVar

import pytest
import yaml
from nmp.core.auth.app.embedded_pdp import (
    OPAPolicy,
    PolicyEngineError,
    evaluate,
    get_policy,
    get_valid_entrypoints,
    reload_policy,
    set_policy_data,
    validate_entrypoint,
)


@pytest.fixture
def static_authz_data():
    """Load the static authorization data."""
    path = Path(__file__).parent.parent / "src/nmp/core/auth/assets/static-authz.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def test_access_key_lifecycle_routes_are_available_to_authenticated_owners(
    static_authz_data: dict[str, Any],
) -> None:
    endpoints = static_authz_data["authz"]["endpoints"]

    for action in ["suspend", "unsuspend"]:
        rule = endpoints[f"/apis/auth/v2/access-keys/{{jti}}/{action}"]["post"]
        # No RBAC permission needed (self-service), but scope still gates self-escalation.
        assert rule == {"permissions": [], "scopes": ["auth:write", "platform:write"]}


@pytest.fixture
def minimal_authz_data():
    """Minimal authorization data for testing."""
    return {
        "authz": {
            "principals": {
                "test@example.com": {
                    "workspaces": {
                        "test-workspace": ["Editor"],
                        "system": ["PlatformAdmin"],
                    }
                },
                "viewer@example.com": {
                    "workspaces": {
                        "test-workspace": ["Viewer"],
                    }
                },
            },
            "roles": {
                "Viewer": {"permissions": ["models.read", "datasets.read"]},
                "Editor": {"permissions": ["models.read", "models.create", "models.update", "datasets.read"]},
                "PlatformAdmin": {"permissions": ["*"]},
                "ServiceSystem": {"permissions": ["*"]},
            },
            "endpoints": {
                "/apis/models/v2/workspaces/{workspace_id}/models": {
                    "get": {"permissions": ["models.read"]},
                    "post": {"permissions": ["models.create"]},
                },
            },
            "workspaces": {
                "public-workspace": {},
            },
        }
    }


@pytest.fixture(autouse=True)
def reset_policy():
    """Reset policy state between tests."""
    import nmp.core.auth.app.embedded_pdp.engine as pe

    pe._reset_policy_state_for_testing()
    yield
    pe._reset_policy_state_for_testing()


class TestEntrypointValidation:
    """Tests for entrypoint validation."""

    def test_valid_entrypoints(self):
        assert validate_entrypoint("allow")
        assert validate_entrypoint("has_permissions")
        assert validate_entrypoint("has_role")

    def test_invalid_entrypoint(self):
        assert not validate_entrypoint("invalid")
        assert not validate_entrypoint("")
        assert not validate_entrypoint("ALLOW")

    def test_get_valid_entrypoints(self):
        entrypoints = get_valid_entrypoints()
        assert "allow" in entrypoints
        assert "has_permissions" in entrypoints
        assert "has_role" in entrypoints
        assert len(entrypoints) == 3


class TestPolicyLoading:
    """Tests for policy loading."""

    def test_policy_loads(self):
        policy = get_policy()
        assert policy is not None
        assert isinstance(policy, OPAPolicy)

    def test_policy_reused_within_thread(self):
        policy1 = get_policy()
        policy2 = get_policy()
        assert policy1 is policy2

    def test_policy_runtime_is_thread_local(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        main_policy = get_policy()
        policies = Queue()
        errors = Queue()

        def load_policy_in_thread():
            try:
                policies.put(get_policy())
            except BaseException as exc:
                errors.put(exc)

        thread = Thread(target=load_policy_in_thread)
        thread.start()
        thread.join()

        assert errors.empty(), errors.get()
        worker_policy = policies.get_nowait()
        assert worker_policy is not main_policy
        assert worker_policy._owner_thread_id != main_policy._owner_thread_id

    def test_policy_runtime_rejects_cross_thread_use(self):
        policy = get_policy()
        errors = Queue()

        def evaluate_leaked_policy():
            try:
                policy.evaluate({})
            except BaseException as exc:
                errors.put(exc)

        thread = Thread(target=evaluate_leaked_policy)
        thread.start()
        thread.join()

        error = errors.get_nowait()
        assert isinstance(error, RuntimeError)
        assert "OPAPolicy used from a different thread" in str(error)

    def test_policy_load_forwards_auto_build_config(self, monkeypatch: pytest.MonkeyPatch):
        import nmp.core.auth.app.embedded_pdp.engine as pe
        from nmp.common.config import Configuration
        from nmp.core.auth.config import AuthServiceConfig

        calls: list[bool] = []

        class FakePolicy:
            def __init__(self, *_args, **_kwargs):
                pass

            def set_data(self, _data):
                pass

        monkeypatch.setattr(
            pe,
            "ensure_embedded_policy_wasm",
            lambda *, auto_build: calls.append(auto_build) or Path("/tmp/policy.wasm"),
        )
        monkeypatch.setattr(pe, "OPAPolicy", FakePolicy)
        Configuration.set_override(AuthServiceConfig(embedded_pdp_auto_build_wasm=False))
        try:
            pe.get_policy()
        finally:
            Configuration.clear_override(AuthServiceConfig)

        assert calls == [False]

    def test_reload_policy(self):
        policy1 = get_policy()
        reload_policy()
        policy2 = get_policy()
        assert policy1 is not policy2


class TestAllowEntrypoint:
    """Tests for the 'allow' entrypoint."""

    def test_health_live_endpoint_always_allowed(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate("allow", {"path": "/health/live", "method": "GET"})
        assert result["allowed"] is True

    def test_health_ready_endpoint_always_allowed(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate("allow", {"path": "/health/ready", "method": "GET"})
        assert result["allowed"] is True

    def test_status_endpoint_always_allowed(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate("allow", {"path": "/status", "method": "GET"})
        assert result["allowed"] is True

    def test_metrics_endpoint_always_allowed(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate("allow", {"path": "/metrics", "method": "GET"})
        assert result["allowed"] is True

    def test_unauthenticated_request_denied(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is False

    def test_authenticated_user_with_permission(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "test@example.com",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is True

    def test_service_principal_bypass(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "service:test-service",
                "path": "/apis/models/v2/workspaces/any-workspace/models",
                "method": "POST",
            },
        )
        assert result["allowed"] is True

    def test_platform_admin_bypass(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "test@example.com",  # Has PlatformAdmin in system workspace
                "path": "/apis/models/v2/workspaces/any-workspace/anything",
                "method": "DELETE",
            },
        )
        assert result["allowed"] is True


class TestHasPermissionsEntrypoint:
    """Tests for the 'has_permissions' entrypoint."""

    def test_user_has_permission(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_permissions",
            {
                "principal_id": "test@example.com",
                "workspace": "test-workspace",
                "permissions": ["models.read"],
            },
        )
        assert result["allowed"] is True

    def test_user_lacks_permission(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_permissions",
            {
                "principal_id": "viewer@example.com",
                "workspace": "test-workspace",
                "permissions": ["models.create"],
            },
        )
        assert result["allowed"] is False

    def test_service_principal_bypass(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_permissions",
            {
                "principal_id": "service:any-service",
                "workspace": "test-workspace",
                "permissions": ["anything"],
            },
        )
        assert result["allowed"] is True

    def test_unknown_user(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_permissions",
            {
                "principal_id": "unknown@example.com",
                "workspace": "test-workspace",
                "permissions": ["models.read"],
            },
        )
        assert result["allowed"] is False


class TestHasRoleEntrypoint:
    """Tests for the 'has_role' entrypoint."""

    def test_user_has_role(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_role",
            {
                "principal_id": "test@example.com",
                "workspace": "test-workspace",
                "role": "Editor",
            },
        )
        assert result["has_role"] is True

    def test_user_lacks_role(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_role",
            {
                "principal_id": "viewer@example.com",
                "workspace": "test-workspace",
                "role": "Editor",
            },
        )
        assert result["has_role"] is False

    def test_unknown_user(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "has_role",
            {
                "principal_id": "unknown@example.com",
                "workspace": "test-workspace",
                "role": "Viewer",
            },
        )
        assert result["has_role"] is False


class TestInvalidEntrypoint:
    """Tests for invalid entrypoint handling."""

    def test_invalid_entrypoint_raises(self, minimal_authz_data):
        set_policy_data(minimal_authz_data)
        with pytest.raises(PolicyEngineError) as exc_info:
            evaluate("invalid", {})
        assert "Invalid entrypoint" in str(exc_info.value)


class TestResourceLimits:
    """Tests for fuel and memory limits."""

    def test_fuel_exhaustion_raises_policy_error(self, minimal_authz_data):
        """Verify that an absurdly low fuel limit triggers PolicyEngineError."""
        import nmp.core.auth.app.embedded_pdp.engine as pe
        from nmp.common.config import Configuration
        from nmp.core.auth.config import AuthServiceConfig

        Configuration.set_override(AuthServiceConfig(embedded_pdp_cpu_limit=0, embedded_pdp_memory_limit_mb=32))
        try:
            pe.reload_policy()
            set_policy_data(minimal_authz_data)

            with pytest.raises(PolicyEngineError, match="exceeded fuel limit"):
                evaluate(
                    "allow",
                    {
                        "principal_id": "test@example.com",
                        "path": "/apis/models/v2/workspaces/test-workspace/models",
                        "method": "GET",
                    },
                )
        finally:
            Configuration.clear_override(AuthServiceConfig)
            pe._reset_policy_state_for_testing()

    def test_default_fuel_allows_normal_evaluation(self, minimal_authz_data):
        """Verify that the default fuel limit is sufficient for normal evaluations."""
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "test@example.com",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is True

    def test_memory_limit_applied(self):
        """Verify that memory limits are set on the WASM store."""
        path = Path(__file__).parent.parent / "src/nmp/core/auth/assets/policy.wasm"
        policy = OPAPolicy(str(path), fuel_limit=100_000_000, memory_limit_mb=16)
        assert policy.store is not None
        assert policy.fuel_limit == 100_000_000

    def test_custom_fuel_limit(self, minimal_authz_data):
        """Verify that a custom fuel limit works when sufficient."""
        import nmp.core.auth.app.embedded_pdp.engine as pe
        from nmp.common.config import Configuration
        from nmp.core.auth.config import AuthServiceConfig

        Configuration.set_override(AuthServiceConfig(embedded_pdp_cpu_limit=50, embedded_pdp_memory_limit_mb=32))
        try:
            pe.reload_policy()
            set_policy_data(minimal_authz_data)

            result = evaluate(
                "allow",
                {
                    "principal_id": "test@example.com",
                    "path": "/apis/models/v2/workspaces/test-workspace/models",
                    "method": "GET",
                },
            )
        finally:
            Configuration.clear_override(AuthServiceConfig)
            pe._reset_policy_state_for_testing()
        assert result["allowed"] is True


class TestDefaultDenyWithoutData:
    """Tests that the PDP denies all requests when policy data is not loaded."""

    AUTHENTICATED_REQUEST = {
        "principal_id": "user@example.com",
        "path": "/apis/models/v2/workspaces/test-workspace/models",
        "method": "GET",
    }

    def test_evaluate_without_set_data_raises(self):
        """OPAPolicy.evaluate() must raise when set_data() was never called."""
        path = Path(__file__).parent.parent / "src/nmp/core/auth/assets/policy.wasm"
        policy = OPAPolicy(str(path))
        with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
            policy.evaluate(self.AUTHENTICATED_REQUEST)

    def test_module_evaluate_without_data_raises(self):
        """Module-level evaluate() must raise when no policy data is set."""
        with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
            evaluate("allow", self.AUTHENTICATED_REQUEST)

    def test_empty_endpoints_denies_authenticated_get(self):
        """Authenticated GET is denied when endpoints dict is empty."""
        set_policy_data({"authz": {"endpoints": {}, "principals": {}, "roles": {}}})
        result = evaluate("allow", self.AUTHENTICATED_REQUEST)
        assert result["allowed"] is False

    def test_empty_endpoints_denies_authenticated_post(self):
        """Authenticated POST is denied when endpoints dict is empty."""
        set_policy_data({"authz": {"endpoints": {}, "principals": {}, "roles": {}}})
        result = evaluate(
            "allow",
            {
                "principal_id": "user@example.com",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "POST",
            },
        )
        assert result["allowed"] is False

    def test_empty_endpoints_denies_service_principal(self):
        """Service principals are denied when endpoints dict is empty."""
        set_policy_data({"authz": {"endpoints": {}, "principals": {}, "roles": {}}})
        result = evaluate(
            "allow",
            {
                "principal_id": "service:entity-store",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is False

    def test_empty_endpoints_denies_platform_admin(self):
        """Platform admins are denied when endpoints dict is empty."""
        set_policy_data(
            {
                "authz": {
                    "endpoints": {},
                    "principals": {"admin@example.com": {"workspaces": {"system": ["PlatformAdmin"]}}},
                    "roles": {"PlatformAdmin": {"permissions": ["*"]}},
                }
            }
        )
        result = evaluate(
            "allow",
            {
                "principal_id": "admin@example.com",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is False

    def test_health_endpoints_allowed_with_empty_endpoints(self):
        """Health endpoints remain accessible even with empty endpoint data."""
        set_policy_data({"authz": {"endpoints": {}, "principals": {}, "roles": {}}})
        for health_path in ["/health/live", "/health/ready", "/status", "/metrics"]:
            result = evaluate("allow", {"path": health_path, "method": "GET"})
            assert result["allowed"] is True, f"{health_path} should be allowed"

    def test_normal_operation_after_data_loaded(self, minimal_authz_data):
        """Normal authorization works correctly once data is loaded."""
        set_policy_data(minimal_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "test@example.com",
                "path": "/apis/models/v2/workspaces/test-workspace/models",
                "method": "GET",
            },
        )
        assert result["allowed"] is True


class TestWithStaticAuthzData:
    """Tests using the full static authorization data."""

    def test_loads_full_data(self, static_authz_data):
        set_policy_data(static_authz_data)
        result = evaluate("allow", {"path": "/health/live", "method": "GET"})
        assert result["allowed"] is True

    def test_viewer_role_has_permissions(self, static_authz_data):
        # Add a test principal with Viewer role
        static_authz_data["authz"]["principals"] = {"viewer@test.com": {"workspaces": {"my-ws": ["Viewer"]}}}
        set_policy_data(static_authz_data)

        result = evaluate(
            "has_permissions",
            {
                "principal_id": "viewer@test.com",
                "workspace": "my-ws",
                "permissions": ["models.read"],
            },
        )
        assert result["allowed"] is True

    def test_job_runner_can_upload_otlp_logs_without_editor(self, static_authz_data):
        static_authz_data["authz"]["principals"] = {
            "viewer@test.com": {"workspaces": {"my-ws": ["Viewer"]}},
            "editor@test.com": {"workspaces": {"my-ws": ["Editor"]}},
            "job-runner@test.com": {"workspaces": {"my-ws": ["Viewer", "JobRunner"]}},
        }
        set_policy_data(static_authz_data)

        path = "/apis/files/v2/workspaces/my-ws/filesets/job-fileset-test/otlp/v1/logs"
        viewer_result = evaluate(
            "allow",
            {
                "principal_id": "viewer@test.com",
                "method": "POST",
                "path": path,
            },
        )
        editor_result = evaluate(
            "allow",
            {
                "principal_id": "editor@test.com",
                "method": "POST",
                "path": path,
            },
        )
        job_runner_result = evaluate(
            "allow",
            {
                "principal_id": "job-runner@test.com",
                "method": "POST",
                "path": path,
            },
        )

        assert viewer_result["allowed"] is False
        assert editor_result["allowed"] is False
        assert job_runner_result["allowed"] is True


class TestIntakeAuthorization:
    """Verify active Intake endpoints are workspace-scoped in static authz data."""

    def _setup_principals(self, static_authz_data):
        static_authz_data["authz"]["principals"] = {
            "viewer@test.com": {"workspaces": {"my-ws": ["Viewer"]}},
            "editor@test.com": {"workspaces": {"my-ws": ["Editor"]}},
        }
        static_authz_data["authz"]["workspaces"] = {"my-ws": {}, "other-ws": {}}
        set_policy_data(static_authz_data)

    def test_viewer_can_list_spans_in_workspace(self, static_authz_data):
        self._setup_principals(static_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "viewer@test.com",
                "method": "GET",
                "path": "/apis/intake/v2/workspaces/my-ws/spans",
            },
        )
        assert result["allowed"] is True

    def test_viewer_cannot_create_annotations(self, static_authz_data):
        self._setup_principals(static_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "viewer@test.com",
                "method": "POST",
                "path": "/apis/intake/v2/workspaces/my-ws/annotations",
            },
        )
        assert result["allowed"] is False

    def test_editor_can_create_annotations(self, static_authz_data):
        self._setup_principals(static_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "editor@test.com",
                "method": "POST",
                "path": "/apis/intake/v2/workspaces/my-ws/annotations",
            },
        )
        assert result["allowed"] is True

    def test_viewer_cannot_read_other_workspace(self, static_authz_data):
        self._setup_principals(static_authz_data)
        result = evaluate(
            "allow",
            {
                "principal_id": "viewer@test.com",
                "method": "GET",
                "path": "/apis/intake/v2/workspaces/other-ws/spans",
            },
        )
        assert result["allowed"] is False

    def test_ingest_is_write_scoped(self, static_authz_data):
        self._setup_principals(static_authz_data)
        write_scoped = evaluate(
            "allow",
            {
                "principal_id": "editor@test.com",
                "method": "POST",
                "path": "/apis/intake/v2/workspaces/my-ws/ingest/chat-completions",
                "scopes": ["intake:write"],
            },
        )
        assert write_scoped["allowed"] is True

        read_scoped = evaluate(
            "allow",
            {
                "principal_id": "editor@test.com",
                "method": "POST",
                "path": "/apis/intake/v2/workspaces/my-ws/ingest/chat-completions",
                "scopes": ["intake:read"],
            },
        )
        assert read_scoped["allowed"] is False


class TestGenericEntitiesApiBlocked:
    """Verify that Viewer/Editor roles cannot access the generic Entities API.

    The entities.* permissions are intentionally not assigned to any role,
    so only PlatformAdmin and service principals can access these endpoints.
    """

    # /apis/entities/v2/entities/{id} is excluded because it has no workspace segment,
    # so the cross-workspace GET rule allows any authenticated user (tracked in #3992).
    ENTITY_ENDPOINTS: ClassVar[list[tuple[str, str]]] = [
        ("GET", "/apis/entities/v2/workspaces/my-ws/entities/evaluation_config"),
        ("POST", "/apis/entities/v2/workspaces/my-ws/entities/evaluation_config"),
        ("GET", "/apis/entities/v2/workspaces/my-ws/entities/guardrail_config/cfg-1"),
        ("PUT", "/apis/entities/v2/workspaces/my-ws/entities/role_binding/rb-1"),
        ("DELETE", "/apis/entities/v2/workspaces/my-ws/entities/evaluation_config/eval-1"),
    ]

    def _setup_principals(self, static_authz_data, principals):
        static_authz_data["authz"]["principals"] = principals
        set_policy_data(static_authz_data)

    @pytest.mark.parametrize("method,path", ENTITY_ENDPOINTS)
    def test_viewer_denied(self, static_authz_data, method, path):
        self._setup_principals(
            static_authz_data,
            {
                "viewer@test.com": {"workspaces": {"my-ws": ["Viewer"]}},
            },
        )
        result = evaluate("allow", {"principal_id": "viewer@test.com", "method": method, "path": path})
        assert result["allowed"] is False, f"Viewer should be denied {method} {path}"

    @pytest.mark.parametrize("method,path", ENTITY_ENDPOINTS)
    def test_editor_denied(self, static_authz_data, method, path):
        self._setup_principals(
            static_authz_data,
            {
                "editor@test.com": {"workspaces": {"my-ws": ["Editor"]}},
            },
        )
        result = evaluate("allow", {"principal_id": "editor@test.com", "method": method, "path": path})
        assert result["allowed"] is False, f"Editor should be denied {method} {path}"

    @pytest.mark.parametrize("method,path", ENTITY_ENDPOINTS)
    def test_admin_allowed(self, static_authz_data, method, path):
        self._setup_principals(
            static_authz_data,
            {
                "admin@test.com": {"workspaces": {"system": ["PlatformAdmin"]}},
            },
        )
        result = evaluate("allow", {"principal_id": "admin@test.com", "method": method, "path": path})
        assert result["allowed"] is True, f"PlatformAdmin should be allowed {method} {path}"

    @pytest.mark.parametrize("method,path", ENTITY_ENDPOINTS)
    def test_service_principal_allowed(self, static_authz_data, method, path):
        set_policy_data(static_authz_data)
        result = evaluate("allow", {"principal_id": "service:evaluator", "method": method, "path": path})
        assert result["allowed"] is True, f"Service principal should be allowed {method} {path}"

    def test_viewer_can_still_list_workspaces(self, static_authz_data):
        """Non-entity endpoints under /apis/entities/ are unaffected."""
        # Listing workspaces is a no-{workspace} GET, so its permission is checked in the
        # system workspace. A viewer holds workspaces.list there via a system binding (on a
        # real platform that's the seeded wildcard Viewer@system).
        self._setup_principals(
            static_authz_data,
            {
                "viewer@test.com": {"workspaces": {"my-ws": ["Viewer"], "system": ["Viewer"]}},
            },
        )
        result = evaluate(
            "allow", {"principal_id": "viewer@test.com", "method": "GET", "path": "/apis/entities/v2/workspaces"}
        )
        assert result["allowed"] is True

    def test_viewer_can_still_read_workspace(self, static_authz_data):
        """Non-entity endpoints under /apis/entities/ are unaffected."""
        self._setup_principals(
            static_authz_data,
            {
                "viewer@test.com": {"workspaces": {"my-ws": ["Viewer"]}},
            },
        )
        result = evaluate(
            "allow", {"principal_id": "viewer@test.com", "method": "GET", "path": "/apis/entities/v2/workspaces/my-ws"}
        )
        assert result["allowed"] is True

    def test_workspace_create_uses_system_scoped_permission(self, static_authz_data):
        static_authz_data["authz"]["principals"] = {
            "*": {"workspaces": {"system": ["WorkspaceCreator"]}},
            "admin@test.com": {"workspaces": {"system": ["PlatformAdmin"]}},
            "group:ml-leads": {"workspaces": {"system": ["WorkspaceCreator"]}},
        }
        set_policy_data(static_authz_data)

        wildcard_allowed = evaluate(
            "allow",
            {
                "principal_id": "plain-user@test.com",
                "method": "POST",
                "path": "/apis/entities/v2/workspaces",
                "scopes": ["entities:write", "platform:write"],
            },
        )
        assert wildcard_allowed["allowed"] is True

        static_authz_data["authz"]["principals"] = {
            "admin@test.com": {"workspaces": {"system": ["PlatformAdmin"]}},
            "group:ml-leads": {"workspaces": {"system": ["WorkspaceCreator"]}},
        }
        set_policy_data(static_authz_data)

        plain_denied = evaluate(
            "allow",
            {
                "principal_id": "plain-user@test.com",
                "method": "POST",
                "path": "/apis/entities/v2/workspaces",
                "scopes": ["entities:write", "platform:write"],
            },
        )
        group_allowed = evaluate(
            "allow",
            {
                "principal_id": "lead-user@test.com",
                "principal_groups": ["group:ml-leads"],
                "method": "POST",
                "path": "/apis/entities/v2/workspaces",
                "scopes": ["entities:write", "platform:write"],
            },
        )

        assert plain_denied["allowed"] is False
        assert group_allowed["allowed"] is True


class TestPerAreaScopes:
    """Validate that every endpoint in static-authz.yaml has per-area scopes.

    Each endpoint should have both an area-specific scope (e.g. models:read)
    and the corresponding platform catch-all scope (platform:read/platform:write),
    unless the endpoint has intentionally empty scopes (e.g. workspace creation).
    """

    # Map URL prefix to expected scope area
    AREA_MAP: ClassVar[dict[str, str]] = {
        "/apis/audit/": "audit",
        "/apis/auth/": "auth",
        "/apis/data-designer/": "data-designer",
        "/apis/entities/": "entities",
        "/apis/files/": "files",
        "/apis/guardrails/": "guardrails",
        "/apis/inference-gateway/": "inference",
        "/apis/intake/": "intake",
        "/apis/jobs/": "jobs",
        "/apis/models/": "models",
        "/apis/safe-synthesizer/": "safe-synthesizer",
        "/apis/secrets/": "secrets",
    }

    READ_METHODS: ClassVar[set[str]] = {"get", "head"}
    WRITE_METHODS: ClassVar[set[str]] = {"post", "put", "patch", "delete"}

    # POST endpoints that are semantically read operations (e.g. query endpoints).
    # These correctly use :read scopes despite being POST methods.
    READ_POST_ENDPOINTS: ClassVar[set[str]] = {
        "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query",
    }

    def _get_area(self, endpoint_path: str) -> str:
        for prefix, area in self.AREA_MAP.items():
            if endpoint_path.startswith(prefix):
                return area
        return ""

    def _expected_scope_type(self, method: str, path: str) -> str:
        """Determine whether an endpoint is a read or write operation.

        Most methods map directly: GET/HEAD → read, POST/PUT/PATCH/DELETE → write.
        Some POST endpoints are semantically reads (e.g. query endpoints).
        """
        method_lower = method.lower()
        if method_lower in self.READ_METHODS:
            return "read"
        if method_lower == "post" and path in self.READ_POST_ENDPOINTS:
            return "read"
        if method_lower in self.WRITE_METHODS:
            return "write"
        return ""

    def test_all_endpoints_have_area_scope(self, static_authz_data):
        """Every endpoint with non-empty scopes must include the area-specific scope."""
        endpoints = static_authz_data["authz"]["endpoints"]
        missing = []

        for path, methods in endpoints.items():
            area = self._get_area(path)
            assert area, f"No area mapping for endpoint: {path}"

            for method, config in methods.items():
                scopes = config.get("scopes", [])
                if not scopes:
                    continue

                scope_type = self._expected_scope_type(method, path)
                if not scope_type:
                    continue

                expected_area_scope = f"{area}:{scope_type}"
                if expected_area_scope not in scopes:
                    missing.append(f"{method.upper()} {path} missing {expected_area_scope}")

        assert not missing, "Endpoints missing per-area scopes:\n" + "\n".join(missing)

    def test_all_endpoints_have_platform_scope(self, static_authz_data):
        """Every endpoint with non-empty scopes must include a platform:* catch-all scope."""
        endpoints = static_authz_data["authz"]["endpoints"]
        missing = []

        for path, methods in endpoints.items():
            for method, config in methods.items():
                scopes = config.get("scopes", [])
                if not scopes:
                    continue

                scope_type = self._expected_scope_type(method, path)
                if not scope_type:
                    continue

                expected = f"platform:{scope_type}"
                if expected not in scopes:
                    missing.append(f"{method.upper()} {path} missing {expected}")

        assert not missing, "Endpoints missing platform scope:\n" + "\n".join(missing)

    def test_scope_read_write_consistency(self, static_authz_data):
        """GET/HEAD should have :read scopes, mutation methods should have :write scopes."""
        endpoints = static_authz_data["authz"]["endpoints"]
        inconsistent = []

        for path, methods in endpoints.items():
            for method, config in methods.items():
                scopes = config.get("scopes", [])
                if not scopes:
                    continue

                method_lower = method.lower()
                platform_scopes = [s for s in scopes if ":" in s]

                # Some POST endpoints are semantically reads (e.g. query endpoints)
                is_semantic_read = method_lower == "post" and path in self.READ_POST_ENDPOINTS

                if method_lower in self.READ_METHODS or is_semantic_read:
                    for scope in platform_scopes:
                        if scope.endswith(":write"):
                            inconsistent.append(f"{method.upper()} {path} has write scope: {scope}")
                elif method_lower in self.WRITE_METHODS:
                    for scope in platform_scopes:
                        if scope.endswith(":read"):
                            inconsistent.append(f"{method.upper()} {path} has read scope: {scope}")

        assert not inconsistent, "Inconsistent read/write scopes:\n" + "\n".join(inconsistent)

    def test_workspace_creation_requires_write_scopes_and_permission(self, static_authz_data):
        """POST /apis/entities/v2/workspaces should participate in normal write-scope RBAC."""
        ws_endpoint = static_authz_data["authz"]["endpoints"].get("/apis/entities/v2/workspaces", {})
        post_config = ws_endpoint.get("post", {})
        assert post_config.get("scopes") == ["entities:write", "platform:write"]
        assert post_config.get("permissions") == ["workspaces.create"]

    def test_audit_workspace_endpoints_have_audit_scopes(self, static_authz_data):
        """All workspace-scoped audit endpoints should have audit:read/audit:write scopes."""
        endpoints = static_authz_data["authz"]["endpoints"]
        missing = []

        for path, methods in endpoints.items():
            if not path.startswith("/apis/audit/v2/workspaces/"):
                continue
            for method, config in methods.items():
                scopes = config.get("scopes", [])
                has_audit = any(s.startswith("audit:") for s in scopes)
                if not has_audit:
                    missing.append(f"{method.upper()} {path}")

        assert not missing, "Workspace audit endpoints missing audit:* scope:\n" + "\n".join(missing)


class TestWasmNativeBuiltins:
    """The embedded engine stubs host-provided builtins (env::opa_builtin*) to return 0.

    Any rego that calls an SDK-dependent builtin (sprintf, glob.match, ...) therefore
    silently evaluates to undefined in production: allow rules fail closed, but DENY
    rules fail OPEN — and ``opa test`` cannot catch it because the Go evaluator
    implements every builtin. These tests pin the policy to wasm-native builtins only.
    """

    def test_policy_wasm_requires_no_host_builtins(self):
        """The compiled policy must not depend on any host-provided builtin."""
        policy = get_policy()
        required = policy._read_json(policy.exports["builtins"](policy.store))
        assert required == {}, (
            f"policy.wasm requires host builtins {list(required)} which the embedded "
            "engine stubs out — rewrite the policy using wasm-native builtins only "
            "(a deny rule depending on a stubbed builtin silently never fires)."
        )

    def test_namespace_fence_denies_subpaths_in_wasm(self, static_authz_data):
        """Regression: the fence's subpath arm was written with sprintf and never fired in WASM."""
        static_authz_data["authz"].setdefault("config", {})["denied_plugin_prefixes"] = ["/apis/brokenplugin"]
        set_policy_data(static_authz_data)
        cases = [
            ("/apis/brokenplugin/sub/route", False),  # subpath fenced (the bug: this was allowed)
            ("/apis/brokenplugin", False),  # bare prefix fenced
            ("/apis/brokenplugin-extra/x", True),  # sibling prefix not collaterally fenced
        ]
        for path, expect in cases:
            result = evaluate("allow", {"principal_id": "service:probe", "method": "GET", "path": path})
            assert result["allowed"] is expect, f"GET {path} as service:probe: expected allowed={expect}"
