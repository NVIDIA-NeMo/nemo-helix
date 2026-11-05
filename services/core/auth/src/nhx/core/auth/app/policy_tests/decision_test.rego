# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_decision_test

import future.keywords.if
import data.authz

roles := {
    "Reader": {"permissions": ["vault.read"]},
    "Producer": {"permissions": ["vault.access"]},
    "DelegatedService": {"permissions": ["vault.access"]},
    "ServiceSystem": {"permissions": ["*"]},
    "HelixAdmin": {"permissions": ["*"]},
}
principals := {
    "alice": {"workspaces": {"team": ["Reader"]}},
    "actor-admins": {"workspaces": {"system": ["HelixAdmin"]}},
}
operations := {
    "vault.retrieve": {
        "owner": "vault", "predicate": "permissions", "permissions": ["vault.access"],
        "callers": ["service_principal"], "scopes": ["vault:read"],
        "on_behalf_of": {"actor_permissions": ["vault.access"], "permissions": ["vault.read"]},
    },
    "entities.auth_context.access": {
        "owner": "entities", "predicate": "auth_context", "permissions": [],
        "callers": ["service_principal"], "scopes": [],
        "on_behalf_of": {"actor_permissions": [], "permissions": []},
    },
    "entities.workspaces.visible": {
        "owner": "entities", "predicate": "workspace_visibility", "permissions": [],
        "callers": ["principal", "service_principal"], "scopes": [],
    },
}
request := {
    "protocol_version": 2, "operation": "vault.retrieve",
    "target": {"kind": "resource", "workspace": "team", "resource_type": "vault", "name": "credential"},
    "principal_id": "runtime:step", "caller_kind": "service_principal",
    "on_behalf_of_principal_id": "alice", "verified_workload_delegation": true,
    "principal_context_digest": "bound", "facts": {"kind": "none"}, "scopes": ["vault:read"],
}

evaluate(document, role_data, principal_data) := result if {
    result := authz.decision with input as document
        with data.authz.roles as role_data
        with data.authz.principals as principal_data
        with data.authz.operations as operations
        with data.authz.scope_families as {"vault": true, "other": true}
        with data.authz.auth_context_writers as {"service:jobs": ["job"]}
}

test_independent_actor_and_user_requirements_allow if {
    evaluate(request, roles, principals).allowed
}
test_actor_grant_cannot_replace_missing_subject_grant if {
    not evaluate(request, roles, {}).allowed
}
test_subject_grant_cannot_replace_missing_actor_grant if {
    no_actor := object.union(roles, {"DelegatedService": {"permissions": []}})
    not evaluate(request, no_actor, principals).allowed
}
test_unverified_delegation_does_not_receive_actor_grant if {
    not evaluate(object.union(request, {"verified_workload_delegation": false}), roles, principals).allowed
}
test_actor_groups_do_not_satisfy_user_half if {
    doc := object.union(request, {"actor_groups": ["actor-admins"]})
    not evaluate(doc, roles, {}).allowed
}
test_subject_group_grant_is_respected if {
    doc := object.union(request, {"on_behalf_of_principal_id": "bob", "subject_groups": ["readers"]})
    groups := {"readers": {"workspaces": {"team": ["Reader"]}}}
    evaluate(doc, roles, groups).allowed
}
test_cross_workspace_subject_permission_is_required if {
    doc := object.union(request, {"target": object.union(request.target, {"workspace": "other-team"})})
    not evaluate(doc, roles, principals).allowed
}
test_scope_is_required_even_for_subject_admin if {
    doc := object.union(request, {"scopes": ["other:read"]})
    admins := {"alice": {"workspaces": {"system": ["HelixAdmin"]}}}
    not evaluate(doc, roles, admins).allowed
}
test_direct_human_admin_is_not_service_caller if {
    doc := object.remove(object.union(request, {
        "principal_id": "alice", "caller_kind": "principal",
    }), ["on_behalf_of_principal_id"])
    admins := {"alice": {"workspaces": {"system": ["HelixAdmin"]}}}
    not evaluate(doc, roles, admins).allowed
}
test_unknown_operation_fails_closed if {
    not evaluate(object.union(request, {"operation": "unknown.operation"}), roles, principals).allowed
}
test_obo_field_denial_wins_over_service_and_subject_admin if {
    doc := object.union(request, {
        "operation": "entities.auth_context.access", "principal_id": "service:jobs",
        "target": {"kind": "field", "workspace": "team", "resource_type": "job", "field": "auth_context"},
    })
    admins := {"alice": {"workspaces": {"system": ["HelixAdmin"]}}}
    result := evaluate(doc, roles, admins)
    result.decision_type == "field"
    result.read == false
    result.write == false
}
test_direct_registered_platform_field_control if {
    doc := object.remove(object.union(request, {
        "operation": "entities.auth_context.access", "principal_id": "service:jobs",
        "target": {"kind": "field", "workspace": "team", "resource_type": "job", "field": "auth_context"},
    }), ["on_behalf_of_principal_id"])
    result := evaluate(doc, roles, principals)
    result.read
    result.write
}
test_workspace_scope_uses_subject_even_when_actor_is_admin if {
    doc := object.union(request, {
        "operation": "entities.workspaces.visible", "actor_groups": ["actor-admins"],
        "target": {"kind": "collection", "workspace": "-", "resource_type": "entity"},
    })
    result := evaluate(doc, roles, principals)
    result.access == {"kind": "restricted", "workspaces": ["team"]}
}
test_workspace_empty_set_remains_restricted_empty if {
    doc := object.union(request, {
        "operation": "entities.workspaces.visible",
        "target": {"kind": "collection", "workspace": "-", "resource_type": "entity"},
    })
    evaluate(doc, roles, {}).access == {"kind": "restricted", "workspaces": []}
}


test_permission_query_combines_subject_groups if {
    query := {
        "principal_id": "alice", "principal_groups": ["first", "second"],
        "workspace": "team", "permissions": ["vault.read", "vault.inspect"],
    }
    result := authz.has_permissions with input as query
        with data.authz.roles as {
            "First": {"permissions": ["vault.read"]},
            "Second": {"permissions": ["vault.inspect"]},
        }
        with data.authz.principals as {
            "first": {"workspaces": {"team": ["First"]}},
            "second": {"workspaces": {"team": ["Second"]}},
        }
    result.allowed
}

test_permission_query_never_combines_actor_and_subject_grants if {
    query := {
        "principal_id": "service:jobs", "caller_kind": "service_principal",
        "on_behalf_of_principal_id": "alice", "subject_groups": ["readers"],
        "workspace": "team", "permissions": ["vault.read", "vault.inspect"],
    }
    result := authz.has_permissions with input as query
        with data.authz.roles as {
            "Reader": {"permissions": ["vault.read"]},
            "ServiceSystem": {"permissions": ["*"]},
        }
        with data.authz.principals as {"readers": {"workspaces": {"team": ["Reader"]}}}
    not result.allowed
}
