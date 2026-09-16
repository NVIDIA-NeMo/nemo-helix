# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package common

import future.keywords.if

import data.authz.extract_method
import data.authz.extract_path
import data.authz.extract_principal_email
import data.authz.extract_principal_groups
import data.authz.extract_principal_id

# PERMISSIONS HELPERS

# Roles bound to a principal in a workspace. Service principals (service:*) default to
# ServiceSystem when they have no explicit bindings in policy data (see static-authz.yaml).
effective_roles(principal, workspace) := roles if {
	startswith(principal, "service:")
	explicit := object.get(object.get(data.authz.principals, principal, {}).workspaces, workspace, [])
	count(explicit) > 0
	roles := explicit
} else := roles if {
	startswith(principal, "service:")
	roles := ["ServiceSystem"]
} else := roles if {
	roles := object.get(object.get(data.authz.principals, principal, {}).workspaces, workspace, [])
}

# True if the role permission set grants the required permission (including "*" = all).
permission_satisfied(role_permissions, required_perm) if {
	"*" in role_permissions
}

permission_satisfied(role_permissions, required_perm) if {
	required_perm in role_permissions
}

# Check if user has required permissions for the operation
# When multiple permissions are required, ALL must be satisfied
# Supports workspace-scoped permissions (format: "workspace/permission")
has_permissions(principal, workspace, required_permissions) if {
	satisfied := [perm |
		perm := required_permissions[_]
		has_specific_permission(principal, workspace, perm)
	]
	count(satisfied) == count(required_permissions)
}

# Get required permissions for an endpoint/method combination.
# Returns undefined (not []) for unknown endpoints so callers cannot accidentally
# treat "not configured" the same as "no permissions required".
get_required_permissions(path, method) := perms if {
	endpoint := normalize_endpoint(path)
	method_lower := lower(method)
	perms := data.authz.endpoints[endpoint][method_lower].permissions
}

# Get the allowed caller kinds for an endpoint/method combination.
# Uses the same most-specific endpoint match as get_required_permissions.
# Returns undefined (not []) when the matched endpoint has no `callers` key, so
# callers can treat absence as the default (PRINCIPAL) semantics — no new restriction.
endpoint_callers(path, method) := callers if {
	endpoint := normalize_endpoint(path)
	method_lower := lower(method)
	callers := data.authz.endpoints[endpoint][method_lower].callers
}

# True when the matched endpoint carries an explicit `deny: true` marker — the fail-closed
# signal emitted for unruled or invalid plugin routes. Undefined (not false) otherwise so it
# only fires where the marker is present.
endpoint_denied(path, method) if {
	endpoint := normalize_endpoint(path)
	data.authz.endpoints[endpoint][lower(method)].deny == true
}

# Check specific permission (for middleware to check special permissions)
# Supports workspace-scoped permissions (format: "workspace/permission")
# For permissions with format "workspace/permission":
#   - Extract the workspace and base permission
#   - Check if principal has a role in that workspace with the base permission
# For permissions without "/":
#   - Use the workspace (existing behavior)
#   - Falls back to checking wildcard principal "*" for workspace access
has_specific_permission(principal, workspace, required_perm) if {
	# Check if permission has explicit workspace (format: workspace/permission)
	contains(required_perm, "/")
	parts := split(required_perm, "/")
	count(parts) == 2
	target_workspace := parts[0]
	base_permission := parts[1]

	some role in effective_roles(principal, target_workspace)
	role_permissions := get_role_permissions(role)
	permission_satisfied(role_permissions, base_permission)
} else if {
	# No explicit workspace - use context workspace (existing behavior)
	not contains(required_perm, "/")
	some role in effective_roles(principal, workspace)
	role_permissions := get_role_permissions(role)
	permission_satisfied(role_permissions, required_perm)
} else if {
	# Fallback: check if wildcard principal "*" has this permission in the workspace
	# This enables public/shared access via role bindings for "*"
	not contains(required_perm, "/")
	some role in data.authz.principals["*"].workspaces[workspace]
	role_permissions := get_role_permissions(role)
	permission_satisfied(role_permissions, required_perm)
}

# Helper function to check if all required permissions are satisfied
all_permissions_satisfied(required_permissions, role_permissions) if {
	# For each required permission, check if it exists in role permissions
	satisfied := [perm |
		perm := required_permissions[_]
		perm in role_permissions
	]

	# All permissions are satisfied if the count matches
	count(satisfied) == count(required_permissions)
}

# Get all permissions for a role (including inherited)
# Supports up to 5 levels of role inheritance
# Note: OPA doesn't support recursion, so we unroll the inheritance chain manually
get_role_permissions(role_name) := permissions if {
	# Collect all roles in the inheritance chain (up to 5 levels)
	all_roles := get_all_roles_in_chain(role_name)

	# Gather all permissions from all roles in the chain
	permissions := {perm |
		some r in all_roles
		role := data.authz.roles[r]
		role_perms := object.get(role, "permissions", [])
		some perm in role_perms
	}
}

# Get all roles in the inheritance chain (up to 5 levels deep)
# Returns a set containing the role and all its ancestors
get_all_roles_in_chain(role_name) := all_roles if {
	# Level 0: The role itself
	level_0 := {role_name}

	# Level 1: Direct includes
	level_1 := {included |
		some r in level_0
		role := data.authz.roles[r]
		includes := object.get(role, "includes", [])
		some included in includes
	}

	# Level 2: Includes of includes
	level_2 := {included |
		some r in level_1
		role := data.authz.roles[r]
		includes := object.get(role, "includes", [])
		some included in includes
	} - (level_0 | level_1) # Avoid duplicates

	# Level 3: Third level includes
	level_3 := {included |
		some r in level_2
		role := data.authz.roles[r]
		includes := object.get(role, "includes", [])
		some included in includes
	} - ((level_0 | level_1) | level_2)

	# Level 4: Fourth level includes
	level_4 := {included |
		some r in level_3
		role := data.authz.roles[r]
		includes := object.get(role, "includes", [])
		some included in includes
	} - (((level_0 | level_1) | level_2) | level_3)

	# Combine all levels
	all_roles := (((level_0 | level_1) | level_2) | level_3) | level_4
}

# Get all applicable principals (id, email, groups)
# Returns a set of all principal identifiers that should be checked
get_applicable_principals := principals if {
	# Start with empty set
	base := set()

	# Add principal ID if present and non-empty
	principal_id := extract_principal_id
	id_set := {principal_id | principal_id != ""}

	# Add email if present and non-empty
	email_set := {email |
		email := extract_principal_email
		email != ""
	}

	# Add groups if present (filter out empty strings)
	groups_set := {g |
		groups := extract_principal_groups
		g := groups[_]
		g != ""
	}

	# Combine all sets
	principals := ((base | id_set) | email_set) | groups_set

	# Ensure we have at least one principal
	count(principals) > 0
} else := set()

# PATH MATCHING HELPERS

# Check if a path matches a pattern
# Used for matching request paths against endpoint patterns with placeholders
#
# For patterns containing /-/:
#   - Wildcard suffix (single placeholder like {trailing_uri}): matches any trailing segments.
#   - Structured suffix (e.g., v1/models or v1/models/{name}): each suffix segment must match
#     the corresponding trailing path segment exactly (or be a placeholder).
#
# This distinction prevents /openai/-/v1/models from incorrectly matching
# /openai/-/v1/chat/completions — only the catch-all /-/{trailing_uri} pattern matches.

# /-/ pattern with a wildcard suffix (single placeholder matches any trailing path)
path_matches_pattern(path, pattern) if {
	# Pattern must contain /-/ but not at the beginning
	contains(pattern, "/-/")
	startswith(pattern, "/-/") == false

	# Split pattern at first /-/ into prefix and suffix.
	# e.g. ".../openai/-/v1/models" → prefix=".../openai", suffix_raw="v1/models"
	# Uses array.slice+concat (not [1]) to handle multiple /-/ occurrences.
	pattern_prefix_with_sep := split(pattern, "/-/")[0]
	suffix_raw := concat("/", array.slice(split(pattern, "/-/"), 1, count(split(pattern, "/-/"))))
	suffix_parts := split(suffix_raw, "/")

	# Wildcard: suffix is a single placeholder like {trailing_uri}
	count(suffix_parts) == 1
	startswith(suffix_parts[0], "{")
	endswith(suffix_parts[0], "}")

	# Check if path starts with the prefix pattern
	path_parts := split(path, "/")
	prefix_pattern_parts := split(pattern_prefix_with_sep, "/")

	# Path must be at least as long as prefix + separator + 1 segment
	count(path_parts) >= count(prefix_pattern_parts) + 2

	# Match prefix segments
	every i in numbers.range(0, count(prefix_pattern_parts) - 1) {
		segment_matches(path_parts[i], prefix_pattern_parts[i])
	}

	# Next segment must be the separator "-"
	path_parts[count(prefix_pattern_parts)] == "-"

	# At least one trailing segment must exist after the separator
	count(path_parts) > count(prefix_pattern_parts) + 1
} else if {
	# /-/ pattern with a structured suffix (specific segments after the separator).
	# For example, the pattern .../openai/-/v1/models must NOT match the path
	# .../openai/-/v1/chat/completions. Only the catch-all /-/{trailing_uri} should.
	# We enforce this by matching each suffix segment against the trailing path.
	contains(pattern, "/-/")
	startswith(pattern, "/-/") == false

	# Split pattern at first /-/ into prefix and suffix (same as wildcard branch above).
	pattern_prefix_with_sep := split(pattern, "/-/")[0]
	suffix_raw := concat("/", array.slice(split(pattern, "/-/"), 1, count(split(pattern, "/-/"))))
	suffix_parts := split(suffix_raw, "/")

	path_parts := split(path, "/")
	prefix_pattern_parts := split(pattern_prefix_with_sep, "/")

	# Index where trailing segments begin (right after the "-" separator)
	trailing_start := count(prefix_pattern_parts) + 1

	# Trailing path must have exactly as many segments as the suffix pattern
	count(path_parts) - trailing_start == count(suffix_parts)

	# Path must be at least as long as prefix + separator + 1 segment
	count(path_parts) >= count(prefix_pattern_parts) + 2

	# Match prefix segments
	every i in numbers.range(0, count(prefix_pattern_parts) - 1) {
		segment_matches(path_parts[i], prefix_pattern_parts[i])
	}

	# Next segment must be the separator "-"
	path_parts[count(prefix_pattern_parts)] == "-"

	# Match each trailing segment against the suffix pattern
	every j in numbers.range(0, count(suffix_parts) - 1) {
		segment_matches(path_parts[trailing_start + j], suffix_parts[j])
	}
} else if {
	# Standard matching: same number of segments, each must match
	path_parts := split(path, "/")
	pattern_parts := split(pattern, "/")

	# Must have same number of segments
	count(path_parts) == count(pattern_parts)

	# Each segment must match
	every i in numbers.range(0, count(path_parts) - 1) {
		segment_matches(path_parts[i], pattern_parts[i])
	}
}

# Check if a path segment matches a pattern segment
# Handles both exact matches and placeholder patterns like {workspace}, {id}
segment_matches(path_segment, pattern_segment) if {
	# Exact match
	path_segment == pattern_segment
} else if {
	# Pattern is a placeholder like {workspace}, {id}, etc.
	startswith(pattern_segment, "{")
	endswith(pattern_segment, "}")
	path_segment != ""
}

# Normalize endpoint path by matching against defined endpoint patterns
# Removes query parameters and matches path against configured endpoint patterns
# Returns the matching pattern for looking up permissions and scopes
# If multiple patterns match, returns the most specific one (fewest placeholders)
normalize_endpoint(path) := pattern if {
	base_path := split(path, "?")[0]
	matching_patterns := {p |
		some p in object.keys(data.authz.endpoints)
		path_matches_pattern(base_path, p)
	}

	# Find the pattern with the fewest placeholders (most specific)
	# Count placeholders by counting segments that start with {
	pattern_scores := {p: count([seg |
		some seg in split(p, "/")
		startswith(seg, "{")
	]) |
		some p in matching_patterns
	}

	# Get the minimum score (fewest placeholders)
	min_score := min([score | some score in [pattern_scores[p] | some p in matching_patterns]])

	# Return a pattern with the minimum score
	some pattern in matching_patterns
	pattern_scores[pattern] == min_score
}

# --- Request-scoped memoization -------------------------------------------------
# extract_path and extract_method are 0-arg rules, and OPA caches complete-rule
# results for the lifetime of a single query. normalize_endpoint scans every
# configured endpoint pattern (O(endpoints)); binding it to a 0-arg rule here makes
# that scan run ONCE per evaluation instead of once per call site. The allow/deny
# rules reference these instead of re-calling the path/method helper functions.
# The functions above are kept intact — the policy tests call them with explicit
# paths/methods, which must not be tied to the live request path.
endpoint_scan := e if {
	e := normalize_endpoint(extract_path)
} else := ""

req_method_lower := lower(extract_method)

req_permissions := data.authz.endpoints[endpoint_scan][req_method_lower].permissions

req_callers := data.authz.endpoints[endpoint_scan][req_method_lower].callers

req_deny if data.authz.endpoints[endpoint_scan][req_method_lower].deny == true

# UTILITY HELPERS

# Helper to format boolean as string for headers
format_bool(value) := "true" if value

else := "false"
