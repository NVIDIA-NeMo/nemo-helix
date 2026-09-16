# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Tests for caller-kind enforcement of service-only routes.
#
# Data contract: an endpoint method may declare an optional `callers` list of
# caller-kind strings ("principal", "service_principal"). A route is "service-only"
# when it lists "service_principal" but NOT "principal". On such routes, human
# (non-service) callers are denied — overriding the permission allows — except a
# PlatformAdmin, who keeps its global bypass and stays allowed. Absence of `callers`
# preserves today's PRINCIPAL-default behavior (no new restriction).

caller_kind_test_data := {
	"roles": {
		"Viewer": {
			"permissions": ["models.read", "models.list", "jobs.read", "jobs.list"],
		},
		"Editor": {
			"includes": ["Viewer"],
			"permissions": ["models.create", "jobs.create"],
		},
		"PlatformAdmin": {
			"includes": ["Editor"],
			"permissions": ["iam.read", "iam.create"],
		},
		# Default role for service:* principals with no explicit bindings.
		"ServiceSystem": {
			"permissions": ["*"],
		},
	},
	"endpoints": {
		# Service-only route: only service principals are allowed.
		"/apis/jobs/v2/workspaces/{workspace}/internal-jobs/{name}": {
			"get": {"permissions": ["jobs.read"], "callers": ["service_principal"]},
		},
		# Mixed route: both humans and services are allowed.
		"/apis/models/v2/workspaces/{workspace}/models/{name}": {
			"get": {"permissions": ["models.read"], "callers": ["principal", "service_principal"]},
		},
		# No `callers` key: legacy PRINCIPAL-default behavior.
		"/apis/models/v2/workspaces/{workspace}/models": {
			"get": {"permissions": ["models.list"]},
		},
		# Principal-only route: only human principals are allowed (callers: ["principal"]).
		"/apis/models/v2/workspaces/{workspace}/human-only/{name}": {
			"get": {"permissions": ["models.read"], "callers": ["principal"]},
		},
	},
	"workspaces": {
		"system": {},
		"ws1": {},
	},
	"principals": {
		"platform-admin@example.com": {
			"workspaces": {"system": ["PlatformAdmin"]},
		},
		# Human user with permissions sufficient for every endpoint under test.
		"user@example.com": {
			"workspaces": {"ws1": ["Viewer"]},
		},
	},
}

# --- Service-only route ---

# A normal human principal (with the required permission) is DENIED on a service-only route.
test_service_only_route_denies_human_principal if {
	result := authz.allow
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/jobs/v2/workspaces/ws1/internal-jobs/job-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == false
}

# A service principal is ALLOWED on the same service-only route.
test_service_only_route_allows_service_principal if {
	result := authz.allow
		with input as {
			"principal_id": "service:jobs-controller",
			"method": "GET",
			"path": "/apis/jobs/v2/workspaces/ws1/internal-jobs/job-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# A human PlatformAdmin is ALLOWED on a service-only route — its global admin bypass is not
# clawed back here (only non-admin humans are denied, per
# test_service_only_route_denies_human_principal above).
test_service_only_route_allows_platform_admin if {
	result := authz.allow
		with input as {
			"principal_id": "platform-admin@example.com",
			"method": "GET",
			"path": "/apis/jobs/v2/workspaces/ws1/internal-jobs/job-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# --- Mixed route (callers: ["principal", "service_principal"]) ---

# A human principal is ALLOWED on a route that lists both caller kinds.
test_mixed_route_allows_human_principal if {
	result := authz.allow
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models/m-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# A service principal is ALLOWED on the same mixed route.
test_mixed_route_allows_service_principal if {
	result := authz.allow
		with input as {
			"principal_id": "service:customizer",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models/m-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# --- Principal-only route (callers: ["principal"]) ---

# A human principal with the required permission is ALLOWED on a principal-only route.
test_principal_only_route_allows_human_principal if {
	result := authz.allow
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/human-only/m-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# A service principal is DENIED on the same principal-only route — even though ServiceSystem
# grants "*" — closing the previously one-directional caller enforcement.
test_principal_only_route_denies_service_principal if {
	result := authz.allow
		with input as {
			"principal_id": "service:customizer",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/human-only/m-1",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == false
}

# principal_only_route is TRUE for a route that lists only "principal".
test_principal_only_route_helper_true_for_principal_only if {
	authz.principal_only_route
		with input as {
			"principal_id": "service:customizer",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/human-only/m-1",
		}
		with data.authz.endpoints as caller_kind_test_data.endpoints
}

# principal_only_route is FALSE for a mixed route (which also lists "service_principal").
test_principal_only_route_helper_false_for_mixed if {
	not authz.principal_only_route
		with input as {
			"principal_id": "service:customizer",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models/m-1",
		}
		with data.authz.endpoints as caller_kind_test_data.endpoints
}

# --- No `callers` key (legacy default) ---

# A human principal with the required permission is ALLOWED (the new deny does not fire).
test_no_callers_key_allows_human_principal if {
	result := authz.allow
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# A service principal is ALLOWED on the no-callers route (unchanged from today).
test_no_callers_key_allows_service_principal if {
	result := authz.allow
		with input as {
			"principal_id": "service:customizer",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models",
		}
		with data.authz.roles as caller_kind_test_data.roles
		with data.authz.endpoints as caller_kind_test_data.endpoints
		with data.authz.workspaces as caller_kind_test_data.workspaces
		with data.authz.principals as caller_kind_test_data.principals

	result.allowed == true
}

# --- Helper-level checks: service_only_route detection ---

# service_only_route is TRUE for a route that lists only "service_principal".
test_service_only_route_helper_true_for_service_only if {
	authz.service_only_route
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/jobs/v2/workspaces/ws1/internal-jobs/job-1",
		}
		with data.authz.endpoints as caller_kind_test_data.endpoints
}

# service_only_route is FALSE for a mixed route that also lists "principal".
test_service_only_route_helper_false_for_mixed if {
	not authz.service_only_route
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models/m-1",
		}
		with data.authz.endpoints as caller_kind_test_data.endpoints
}

# service_only_route is FALSE when the route has no `callers` key.
test_service_only_route_helper_false_for_absent_callers if {
	not authz.service_only_route
		with input as {
			"principal_id": "user@example.com",
			"method": "GET",
			"path": "/apis/models/v2/workspaces/ws1/models",
		}
		with data.authz.endpoints as caller_kind_test_data.endpoints
}
