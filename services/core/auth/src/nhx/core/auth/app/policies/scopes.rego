# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import data.authz.allow
import data.authz.has_permissions
import data.authz.has_role

import future.keywords.contains
import future.keywords.if
import future.keywords.in

import data.authz.extract_scopes
import data.common.endpoint_scan
import data.common.normalize_endpoint
import data.common.req_method_lower

# Scope Checking Helpers
#
# This module provides helper functions for OAuth2 scope validation.
# Scopes provide coarse-grained authorization at the API level, while
# permissions provide fine-grained authorization at the resource level.
#
# Platform scopes (containing ":") are enforced, while standard OIDC scopes
# (openid, profile, email, offline_access) are ignored for authorization.
#
# IMPORTANT: Optional Scope Mechanism
# ------------------------------------
# The scope checking mechanism is designed to be OPTIONAL to simplify setup for customers
# who don't want to use scopes to limit API access. This allows the platform to work with:
# - Tokens that don't have any scopes
# - Tokens that only have OIDC scopes (openid, profile, email, offline_access)
# - Simple authentication setups where only permissions matter
#
# Behavior:
# - If no scopes are provided in the token → scope check passes (relies only on permissions)
# - If only OIDC scopes are provided → scope check passes (OIDC scopes are ignored)
# - If platform scopes (containing ":") are provided → they are validated against endpoint requirements
#
# This can be made configurable if customers want to enforce strict scope checking by
# rejecting tokens without platform scopes.

# Check if scopes are valid for the request (variant 1: no scopes provided)
# If no scopes provided at all, skip check to allow tokens without scopes (optional scope mechanism)

# TODO(v2): make this behavior configurable from the platform config, i.e., ignore scopes
scope_check_passed if {
	not extract_scopes
}

# Check if scopes are valid for the request (variant 2: no platform scopes)
# Handles both empty scope arrays and arrays with only OIDC scopes
# If no platform scopes are found, skip check to allow tokens without platform scopes (optional scope mechanism)
scope_check_passed if {
	scopes := extract_scopes
	platform_scopes := [s | s := scopes[_]; contains(s, ":")]
	count(platform_scopes) == 0
}

# Check if scopes are valid for the request (variant 3: platform scopes validation)
# Extract platform scopes and validate them against endpoint requirements
scope_check_passed if {
	scopes := extract_scopes
	platform_scopes := [s | s := scopes[_]; contains(s, ":")]
	count(platform_scopes) > 0
	req_has_required_scopes(platform_scopes)
}

# Get required scopes for an endpoint/method combination
get_required_scopes(path, method) := scopes if {
	endpoint := normalize_endpoint(path)
	method_lower := lower(method)
	scopes := data.authz.endpoints[endpoint][method_lower].scopes
} else := []

# Check if user has at least one of the required scopes (variant 1: no scopes provided)
# Returns true if provided_scopes is null (optional scope mechanism)
has_required_scopes(path, method, provided_scopes) if {
	provided_scopes == null
}

# Check if user has at least one of the required scopes (variant 2: no scopes required)
# Returns true if no scopes are required for the endpoint
has_required_scopes(path, method, provided_scopes) if {
	required_scopes := get_required_scopes(path, method)
	count(required_scopes) == 0
}

# Check if user has at least one of the required scopes (variant 3: validate provided scopes)
# Returns true if user has at least one of the required scopes
# This function assumes provided_scopes is defined (not undefined)
has_required_scopes(path, method, provided_scopes) if {
	required_scopes := get_required_scopes(path, method)
	some required_scope in required_scopes
	required_scope in provided_scopes
}

# Cached required-scopes for the request endpoint (mirror of
# get_required_scopes(extract_path, extract_method), but using the memoized endpoint).
req_required_scopes := scopes if {
	scopes := data.authz.endpoints[endpoint_scan][req_method_lower].scopes
} else := []

req_has_required_scopes(provided_scopes) if {
	count(req_required_scopes) == 0
}

req_has_required_scopes(provided_scopes) if {
	some required_scope in req_required_scopes
	required_scope in provided_scopes
}
