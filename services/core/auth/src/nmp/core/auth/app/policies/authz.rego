# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.contains
import future.keywords.if
import future.keywords.in

import data.authz.extract_method
import data.authz.extract_path
import data.authz.scope_check_passed
import data.common.endpoint_scan
import data.common.get_applicable_principals
import data.common.has_permissions
import data.common.req_callers
import data.common.req_deny
import data.common.req_permissions

# Main entry point - returns result with X-NMP-Authorized header
#
# Example input:
# {
#   "principal_id": "user@example.com",
#   "method": "GET",
#   "path": "/v1/models"
# }
#
# Example output:
# {
#   "allowed": true,
#   "headers": {
#     "X-NMP-Authorized": "true"
#   }
# }
allow := result if {
	allow_request
	not deny_request
	result := {"allowed": true, "headers": {"X-NMP-Authorized": "true"}}
} else := result if {
	result := {"allowed": false, "headers": {"X-NMP-Authorized": "false"}}
}

# ALLOW REQUEST RULES

# Default deny
default allow_request := false

# Platform admin bypass - has access to everything, but must still pass scope_check_passed
# first so a PlatformAdmin's own scoped Access Key stays scoped down.
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	scope_check_passed

	# Check if any principal is a platform admin
	some principal in applicable_principals
	"PlatformAdmin" in data.authz.principals[principal].workspaces.system
}

# Service principals on paths that do not match any configured API pattern (e.g. legacy /v1/...).
# Known paths are authorized via the ServiceSystem role (wildcard permission) and has_permissions.
allow_request if {
	principal_id := extract_principal_id
	startswith(principal_id, "service:")
	endpoint_scan == ""
}

# Allow if any applicable principal has required permissions and scopes (if provided)
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	# Check scopes first (faster)
	scope_check_passed

	path := extract_path
	method := extract_method
	required_permissions := req_permissions
	count(required_permissions) > 0

	workspace_scan != ""
	workspace := workspace_scan

	# Skip this rule for wildcard workspace - use cross-workspace rule instead
	workspace != "-"

	# Check if any principal has the required permissions
	some principal in applicable_principals
	has_permissions(principal, workspace, required_permissions)
}

# Wildcard workspace "-" with mutating methods: permission-based authorization.
# GET/HEAD for "-" use cross-workspace rules above; mutating methods were previously
# allowed only via unconditional service bypass (service:* defaults to ServiceSystem with "*").
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	# Check scopes first (faster)
	scope_check_passed

	path := extract_path
	method := extract_method
	required_permissions := req_permissions
	count(required_permissions) > 0

	workspace_scan != ""
	workspace := workspace_scan
	workspace == "-"
	method in ["POST", "PUT", "PATCH", "DELETE"]

	some principal in applicable_principals
	startswith(principal, "service:")
	has_permissions(principal, workspace, required_permissions)
}

# Workspace-less endpoints with required permissions: check against the system workspace.
# Covers IAM APIs, workspace creation, and any future workspace-less permission-gated
# endpoint without needing path-specific rules.
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0
	scope_check_passed
	path := extract_path
	method := extract_method
	required_permissions := req_permissions
	count(required_permissions) > 0
	workspace_scan == ""
	some principal in applicable_principals
	has_permissions(principal, "system", required_permissions)
}

# Allow cross-workspace LIST operations (GET/HEAD without workspace in path) for authenticated
# users — the workspace-filtered list case (results scoped to the caller's accessible
# workspaces; an empty list when they have none).
# Only applies when the endpoint declares NO required permissions. A permission-stamped
# no-workspace GET must instead satisfy its permission (rule below); otherwise the stamped
# permission is decorative, which is how the bundle-download endpoint had to be special-cased.
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	# Check scopes first (faster)
	scope_check_passed

	method := extract_method
	method in ["GET", "HEAD"]
	path := extract_path

	# Ensure the path matches a known endpoint pattern.
	# normalize_endpoint is undefined for unknown paths, failing the rule (deny by default).
	endpoint_scan != ""

	# Match if no workspace can be extracted from path (undefined = no workspace placeholder)
	workspace_scan == ""

	# Permissionless only: an endpoint with no `permissions` (empty or absent) keeps the
	# "any authenticated user" behavior; a permissioned one falls through to the rule below.
	not req_has_permissions
}

# A permission-stamped no-workspace GET/HEAD must satisfy its declared permission in the system
# workspace (the home for non-workspace-scoped resources, matching the IAM rule above), so the
# permission is enforced rather than decorative.
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	scope_check_passed

	method := extract_method
	method in ["GET", "HEAD"]
	path := extract_path
	endpoint_scan != ""
	workspace_scan == ""

	required_permissions := req_permissions
	count(required_permissions) > 0

	some principal in applicable_principals
	has_permissions(principal, "system", required_permissions)
}

# True when the matched endpoint declares one or more required permissions for this method.
# Undefined required-permissions (an endpoint with no `permissions` key) makes count() undefined,
# so this is false there — absent/empty permissions are treated alike (no permission required).
req_has_permissions if {
	count(req_permissions) > 0
}

# Allow cross-workspace LIST operations with "-" wildcard workspace
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	# Check scopes first (faster)
	scope_check_passed

	method := extract_method
	method in ["GET", "HEAD"]
	path := extract_path

	# Match if workspace is "-" wildcard
	workspace_scan != ""
	workspace := workspace_scan
	workspace == "-"
}

# Allow if endpoint explicitly has no required permissions
# but still require authentication (at least one principal).
#
# SECURITY: We check the endpoint config directly instead of using get_required_permissions,
# because we need to distinguish between:
#   - endpoints explicitly configured with `permissions: []` → allow
#   - endpoints not in the config at all (unknown) → deny (fail-closed)
# If normalize_endpoint cannot match the path, it is undefined, the rule body fails,
# and access is denied.
allow_request if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0

	# Check scopes first (faster)
	scope_check_passed
	path := extract_path
	method := extract_method
	req_permissions == []
}

# DENY REQUEST RULES

# Default allow (deny_request overrides allow_request when true)
default deny_request := false

# Explicit deny marker (data.authz.endpoints[...].deny == true) — the fail-closed signal for
# unruled or invalid plugin routes. As a deny_request it overrides every allow rule, including
# the ServiceSystem "*" wildcard and the PlatformAdmin bypass, so an un-annotated plugin route
# can never fall through to the service: no-match bypass and become accessible.
deny_request if {
	req_deny
}

# Fence a degraded plugin's entire namespace. The bundle records /apis/<plugin> prefixes for
# plugins whose authz could not be derived at all (load / enumeration failure) — their routes
# may still be mounted by the runner, so deny every path under the prefix rather than let it
# fall through the service: no-match bypass. Undefined config key ⇒ no prefixes ⇒ inert.
deny_request if {
	some prefix in object.get(data.authz.config, "denied_plugin_prefixes", [])
	path_under_denied_prefix(split(extract_path, "?")[0], prefix)
}

# A path is fenced if it sits under the prefix (/apis/<plugin>/...) OR equals it exactly
# (the bare /apis/<plugin> route). The trailing-slash form alone misses the bare prefix.
#
# WASM constraint: only natively-compiled builtins may be used here. The embedded PDP
# stubs SDK-provided builtins (env::opa_builtin*) to return 0, so a deny arm written with
# e.g. sprintf silently never fires in production while `opa test` (full Go evaluator)
# still passes. Boundary check via startswith + substring/count, all wasm-native.
path_under_denied_prefix(path, prefix) if path == prefix

path_under_denied_prefix(path, prefix) if {
	startswith(path, prefix)
	substring(path, count(prefix), 1) == "/"
}

# Deny direct secret value access for non-service principals (including PlatformAdmin).
# Secret values must only be accessed through the service delegation pattern, where a
# service principal reads the value on behalf of a user with secrets.access permission.
# Matches: /apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access
deny_request if {
	path := extract_path
	base_path := split(path, "?")[0]
	path_parts := split(base_path, "/")
	count(path_parts) == 9
	path_parts[4] == "workspaces"
	path_parts[6] == "secrets"
	path_parts[8] == "access"

	principal_id := extract_principal_id
	not startswith(principal_id, "service:")
}

# OPA policy bundle download: system-scoped iam.bundle.read only (see static-authz endpoints).
# Without this deny, the cross-workspace GET rule would allow any authenticated user for this path.
default bundle_access_ok := false

bundle_access_ok if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0
	scope_check_passed
	some principal in applicable_principals
	has_permissions(principal, "system", ["iam.bundle.read"])
}

deny_request if {
	path := extract_path
	base_path := split(path, "?")[0]
	base_path == "/apis/auth/v2/iam/opa-bundle.tar.gz"
	not bundle_access_ok
}

# Nested Entities APIs (not workspace list/create or single-workspace CRUD): only service
# principals and PlatformAdmin (same as previous middleware: IAM paths stayed service-only).
default nested_entities_internal_only := false

nested_entities_internal_only if {
	path := extract_path
	base_path := split(path, "?")[0]
	startswith(base_path, "/apis/entities/v2/")
	not entities_workspace_object_path(base_path, extract_method)
}

deny_request if {
	nested_entities_internal_only
	principal_id := extract_principal_id
	not startswith(principal_id, "service:")
	not platform_admin_in_system
}

# Caller-kind enforcement for service-only routes.
#
# A route declares allowed caller kinds via the optional `callers` list on its
# endpoint config (see endpoint_callers). A route is "service-only" iff it allows
# service principals but NOT principals:
#   callers: ["service_principal"]
# When `callers` is absent, endpoint_callers is undefined and service_only_route is
# false — the route keeps today's PRINCIPAL-default semantics (no new restriction).
# Routes that list "principal" (alone or with "service_principal") are NOT service-only,
# so human callers remain allowed there.
default service_only_route := false

service_only_route if {
	callers := req_callers
	"service_principal" in callers
	not "principal" in callers
}

# Deny a human (non-service) caller on a service-only route. This is a deny_request so it
# overrides the allow rules — including the ServiceSystem "*" wildcard — otherwise humans
# would leak onto service-only routes. Service principals (id starts with "service:") are
# unaffected and stay allowed. A human PlatformAdmin keeps its global bypass here: an admin
# retains access to every route, service-only routes included.
deny_request if {
	service_only_route
	principal_id := extract_principal_id
	not startswith(principal_id, "service:")
	not platform_admin_in_system
}

# Caller-kind enforcement for principal-only routes — the symmetric counterpart of the
# service-only deny above. A route is "principal-only" iff it allows principals but NOT service
# principals:
#   callers: ["principal"]
# When `callers` is absent, endpoint_callers is undefined and this is false: a route with no
# `callers` keeps the PRINCIPAL-default semantics and imposes no new restriction on service
# principals. Routes listing "service_principal" (alone or with "principal") are NOT principal-only.
default principal_only_route := false

principal_only_route if {
	callers := req_callers
	"principal" in callers
	not "service_principal" in callers
}

# Deny a service principal on a principal-only route. Without this, callers=["principal"] was
# one-directional — it kept humans in but never kept service principals out (they passed via the
# ServiceSystem "*" wildcard), so `callers` could not actually scope a route to human users.
deny_request if {
	principal_only_route
	principal_id := extract_principal_id
	startswith(principal_id, "service:")
}

# True when any applicable principal has PlatformAdmin in the system workspace (see allow_request).
default platform_admin_in_system := false

platform_admin_in_system if {
	applicable_principals := get_applicable_principals
	count(applicable_principals) > 0
	some principal in applicable_principals
	"PlatformAdmin" in data.authz.principals[principal].workspaces.system
}

entities_workspace_object_path(base_path, method) if {
	parts := [p | p := split(base_path, "/")[_]; p != ""]
	count(parts) == 4
	parts[0] == "apis"
	parts[1] == "entities"
	parts[2] == "v2"
	parts[3] == "workspaces"
	lower(method) in ["get", "post"]
}

entities_workspace_object_path(base_path, method) if {
	parts := [p | p := split(base_path, "/")[_]; p != ""]
	count(parts) == 5
	parts[0] == "apis"
	parts[1] == "entities"
	parts[2] == "v2"
	parts[3] == "workspaces"
	lower(method) in ["get", "put", "delete"]
}

# All workspace sub-resources bypass the blanket nested-Entities hard deny and go through the
# endpoint permission system instead. The effective access level is determined by static-authz.yaml:
# members and projects grant user roles access; entities.* does not, so entity CRUD is still denied
# for non-service callers, but via a permission check rather than an OPA hard deny. This makes it
# easy to open access later by updating role definitions without touching the OPA policy.
# Cross-workspace queries use workspace "-"; exclude those so they retain the blanket internal-only
# restriction.
entities_workspace_object_path(base_path, method) if {
	parts := [p | p := split(base_path, "/")[_]; p != ""]
	count(parts) >= 6
	parts[0] == "apis"
	parts[1] == "entities"
	parts[2] == "v2"
	parts[3] == "workspaces"
	parts[4] != "-"
	sub := parts[5]
	sub in ["members", "projects", "entities"]
	lower(method) in ["get", "post", "put", "patch", "delete", "head"]
}

# Health check endpoints - always allow (must match middleware HEALTH_ENDPOINTS)
allow_request if {
	path := extract_path
	path in ["/health/live", "/health/ready", "/status", "/metrics"]
}

# Deny all non-health requests when no endpoint data is loaded (fail-closed).
# Defense in depth: the WASM engine also blocks evaluation when data is not set,
# but this rule catches the case where set_data() was called with empty/partial data.
# Health endpoints are excluded so Kubernetes probes still work during startup.
deny_request if {
	count(data.authz.endpoints) == 0
	path := extract_path
	not path in ["/health/live", "/health/ready", "/status", "/metrics", "/cluster-info"]
}
