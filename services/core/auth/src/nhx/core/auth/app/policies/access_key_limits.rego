# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.if
import future.keywords.in

import data.authz.extract_scopes

# Workspace-limited Scoped Access Keys.
#
# The auth service sets input.access_key_workspaces before evaluation (app/access_key_workspace_limits.py):
#   null                  -> no key limit applies
#   ["team-a", "team-b"]  -> the only workspaces the request may touch
#   []                    -> nothing is allowed
# A limit is checked on every request and overrides every allow rule, including the HelixAdmin bypass.

access_key_limited if is_array(input.access_key_workspaces)

access_key_workspace_allowed(workspace) if {
	workspace != ""
	workspace != "-"
	workspace in input.access_key_workspaces
}

# A limited key must name an allowed workspace in the path. Workspace-less ("") and cross-workspace ("-")
# routes are denied.
deny_request if {
	access_key_limited
	not access_key_workspace_allowed(workspace_scan)
}

# A limited key with scopes gets strict scope checks. Scope families no route declares are not ignored
# (as they are for other tokens), and a route with no scopes is not open to it.
deny_request if {
	access_key_limited
	some scope in extract_scopes
	family := scope_family(scope)
	not configured_scope_family(family)
}

deny_request if {
	access_key_limited
	count([scope | some scope in extract_scopes; scope_family(scope)]) > 0
	count(req_required_scopes) == 0
}
