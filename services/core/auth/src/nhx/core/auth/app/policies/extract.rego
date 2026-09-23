# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import data.authz.allow
import data.authz.has_permissions
import data.authz.has_role

import future.keywords.if
import future.keywords.in

import data.common.path_matches_pattern

# Input Extraction Helpers
#
# This module provides helper functions to extract information from authorization requests.
# It supports two input formats:
#
# 1. Direct format (from middleware):
#    {
#      "principal_id": "user@example.com",
#      "method": "GET",
#      "path": "/v1/workspaces/my-workspace",
#      "scopes": ["models:read", "platform:read"]
#    }
#
# 2. Envoy format (from Envoy External Authorization):
#    {
#      "attributes": {
#        "request": {
#          "http": {
#            "headers": {"x-nhx-principal-id": "user@example.com"},
#            "method": "GET",
#            "path": "/v1/workspaces/my-workspace"
#          }
#        }
#      }
#    }

# Normalize Envoy/header-style input once. HTTP header names are case-insensitive,
# and neither clients nor proxies owe us a canonical capitalization.
request_headers := normalized if {
	headers := input.attributes.request.http.headers
	normalized := {lower(key): value |
		some key
		value := headers[key]
	}
} else := {}

# Extract method from either format
extract_method := method if {
	# Direct format
	input.method
	method := input.method
} else := method if {
	# Envoy format
	input.attributes.request.http.method
	method := input.attributes.request.http.method
}

# Extract path from either format
extract_path := path if {
	# Direct format
	input.path
	path := input.path
} else := path if {
	# Envoy format
	input.attributes.request.http.path
	path := input.attributes.request.http.path
}

# Extract scopes from either format
extract_scopes := scopes if {
	# Direct format
	input.scopes
	scopes := input.scopes
} else := scopes if {
	# Envoy format - try x-nhx-scopes header (space-separated)
	request_headers["x-nhx-scopes"]
	scopes := split(request_headers["x-nhx-scopes"], " ")
}

# Extract principal_id from either format
extract_principal_id := principal_id if {
	# Direct format
	input.principal_id
	principal_id := input.principal_id
} else := principal_id if {
	# Envoy format - try x-nhx-principal-id header
	request_headers["x-nhx-principal-id"]
	principal_id := request_headers["x-nhx-principal-id"]
}

# Extract stable actor account_id from either format
extract_actor_account_id := account_id if {
	input.actor_account_id
	account_id := input.actor_account_id
} else := account_id if {
	request_headers["x-nhx-actor-account-id"]
	account_id := request_headers["x-nhx-actor-account-id"]
} else := ""

# Extract actor authorization aliases from either format
extract_actor_aliases := aliases if {
	input.actor_aliases
	aliases := input.actor_aliases
} else := aliases if {
	request_headers["x-nhx-actor-aliases"]
	aliases := split(request_headers["x-nhx-actor-aliases"], ",")
} else := []

# Extract caller kind from either format
extract_caller_kind := caller_kind if {
	input.caller_kind
	caller_kind := input.caller_kind
} else := ""

# Extract principal_email from either format
extract_principal_email := email if {
	# Direct format
	input.principal_email
	email := input.principal_email
} else := email if {
	# Envoy format - try x-nhx-principal-email header
	request_headers["x-nhx-principal-email"]
	email := request_headers["x-nhx-principal-email"]
} else := ""

# Extract principal_groups from either format
extract_principal_groups := groups if {
	# Direct format
	input.principal_groups
	groups := input.principal_groups
} else := groups if {
	# Envoy format - try x-nhx-principal-groups header (comma-separated)
	request_headers["x-nhx-principal-groups"]
	groups := split(request_headers["x-nhx-principal-groups"], ",")
} else := []

# Extract on-behalf-of principal id from either format
extract_on_behalf_of_principal_id := principal_id if {
	input.on_behalf_of_principal_id
	principal_id := input.on_behalf_of_principal_id
} else := principal_id if {
	request_headers["x-nhx-principal-on-behalf-of"]
	principal_id := request_headers["x-nhx-principal-on-behalf-of"]
} else := ""

# Extract stable subject account_id from either format
extract_subject_account_id := account_id if {
	input.subject_account_id
	account_id := input.subject_account_id
} else := account_id if {
	request_headers["x-nhx-subject-account-id"]
	account_id := request_headers["x-nhx-subject-account-id"]
} else := ""

# Extract subject authorization aliases from either format
extract_subject_aliases := aliases if {
	input.subject_aliases
	aliases := input.subject_aliases
} else := aliases if {
	request_headers["x-nhx-subject-aliases"]
	aliases := split(request_headers["x-nhx-subject-aliases"], ",")
} else := []

# Extract workspace from path by matching against defined endpoint patterns
# Looks for workspace/workspace placeholder in matching pattern:
# - {workspace}, {workspace_id}, or {workspace_id} in any position
# - {id} when it comes right after /workspaces/ or /workspaces/ (e.g., /v1/workspaces/{id}/members)
extract_workspace_from_path(path) := workspace if {
	# Remove query parameters
	base_path := split(path, "?")[0]
	path_parts := split(base_path, "/")

	# Find matching endpoint pattern
	some pattern in object.keys(data.authz.endpoints)
	path_matches_pattern(base_path, pattern)
	pattern_parts := split(pattern, "/")

	# Find the segment that represents workspace/workspace
	# Try explicit workspace/workspace placeholders
	some i in numbers.range(0, count(pattern_parts) - 1)
	pattern_parts[i] in ["{workspace}", "{workspace_id}", "{workspace_id}"]

	# Extract the corresponding value from the path
	workspace := path_parts[i]
} else := workspace if {
	# For /v2/workspaces/{workspace_id}/... paths
	base_path := split(path, "?")[0]
	path_parts := split(base_path, "/")

	# Find matching endpoint pattern
	some pattern in object.keys(data.authz.endpoints)
	path_matches_pattern(base_path, pattern)
	pattern_parts := split(pattern, "/")

	# Check if pattern is /v2/workspaces/{...}/...
	some i in numbers.range(0, count(pattern_parts) - 1)
	i >= 2
	pattern_parts[i - 1] == "workspaces"
	startswith(pattern_parts[i], "{")
	endswith(pattern_parts[i], "}")

	# Extract the workspace ID value (used as workspace)
	workspace := path_parts[i]
} else := workspace if {
	# For /v1/workspaces/{id}/... paths, {id} is the workspace identifier
	base_path := split(path, "?")[0]
	path_parts := split(base_path, "/")

	# Find matching endpoint pattern
	some pattern in object.keys(data.authz.endpoints)
	path_matches_pattern(base_path, pattern)
	pattern_parts := split(pattern, "/")

	# Check if pattern is /v1/workspaces/{id}/...
	some i in numbers.range(0, count(pattern_parts) - 1)
	i >= 2
	pattern_parts[i - 1] == "workspaces"
	startswith(pattern_parts[i], "{")
	endswith(pattern_parts[i], "}")

	# Extract the workspace ID value
	workspace := path_parts[i]
}

# Request-scoped workspace, memoized once per evaluation (see common.endpoint_scan).
# The function above stays intact for the policy tests; allow rules use this 0-arg rule.
workspace_scan := w if {
	w := extract_workspace_from_path(extract_path)
} else := ""
