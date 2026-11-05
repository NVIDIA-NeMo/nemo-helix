# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.if
import future.keywords.in
import future.keywords.every

import data.common

# These sets never combine actor and represented-user membership.
operation_delegated if extract_on_behalf_of_principal_id != ""

operation_subject_id := extract_on_behalf_of_principal_id if operation_delegated
else := extract_principal_id

actor_principals := {p |
    values := array.concat(
        [extract_principal_id, extract_actor_account_id, operation_actor_email],
        array.concat(extract_actor_aliases, operation_actor_groups),
    )
    p := values[_]
    p != ""
    p != "*"
}

operation_actor_email := object.get(input, "actor_email", object.get(request_headers, "x-nhx-principal-email", "")) if operation_delegated
else := object.get(input, "actor_email", operation_legacy_email)

operation_actor_groups := object.get(input, "actor_groups", operation_header_actor_groups) if operation_delegated
else := object.get(input, "actor_groups", operation_legacy_groups)

operation_header_actor_groups := split(value, ",") if {
    value := object.get(request_headers, "x-nhx-principal-groups", "")
    value != ""
} else := []

# Legacy direct-query bodies used principal_* for the represented user.
# Envoy carries actor and OBO claims in distinct headers; never union them.
operation_subject_email := object.get(input, "subject_email",
    object.get(input, "principal_email", object.get(request_headers, "x-nhx-principal-on-behalf-of-email", "")))

operation_header_subject_groups := split(value, ",") if {
    value := object.get(request_headers, "x-nhx-principal-on-behalf-of-groups", "")
    value != ""
} else := []
operation_subject_groups := object.get(input, "subject_groups",
    object.get(input, "principal_groups", operation_header_subject_groups))

subject_principals := principals if {
    operation_delegated
    principals := {p |
    values := array.concat(
        [extract_on_behalf_of_principal_id, extract_subject_account_id,
            operation_subject_email],
        array.concat(extract_subject_aliases, operation_subject_groups),
    )
    p := values[_]
    p != ""
    p != "*"
    }
} else := actor_principals

operation_effective_subject_email := operation_subject_email if operation_delegated
else := operation_actor_email

operation_subject_owner_ids := {identifier |
    identifier := [operation_subject_id, operation_effective_subject_email][_]
    identifier != ""
}

operation_platform_actor if common.valid_service_principal(extract_principal_id)

operation_admin(principals) if common.principal_set_admin(principals)

operation_subject_human if {
    not common.valid_service_principal(extract_on_behalf_of_principal_id)
    not startswith(extract_on_behalf_of_principal_id, "service-account:")
    operation_delegated
}
operation_subject_human if {
    not operation_delegated
    not common.valid_service_principal(extract_principal_id)
    not startswith(extract_principal_id, "service-account:")
}

operation_contract := rule if {
    object.get(input, "operation", "route") == "route"
    rule := data.authz.endpoints[common.endpoint_scan][operation_route_method]
} else := rule if {
    name := input.operation
    name != "route"
    rule := data.authz.operations[name]
}

operation_workspace := workspace if {
    object.get(input, "operation", "route") == "route"
    workspace_scan != ""
    workspace := workspace_scan
} else := workspace if {
    object.get(input, "operation", "route") == "route"
    workspace := "system"
} else := object.get(input.target, "workspace", "system")

operation_direct_permission_at(principals, workspace, permission) if {
    common.principal_set_permission(principals, workspace, permission, false, [])
}
operation_direct_permission(principals, permission) if {
    operation_direct_permission_at(principals, operation_workspace, permission)
}

# Only the authenticated workload-delegation contract supplies these actor defaults.
operation_additional_actor_permissions := common.get_role_permissions("DelegatedService") if {
    operation_delegated
    object.get(input, "verified_workload_delegation", false)
    not operation_platform_actor
    common.request_caller_kind == "service_principal"
    operation_contract.on_behalf_of
} else := []

operation_actor_permissions(required) if {
    common.principal_set_permissions(actor_principals, operation_workspace, required, true, operation_additional_actor_permissions)
}
operation_subject_permissions(required) if {
    common.principal_set_permissions(subject_principals, operation_workspace, required, false, [])
}

operation_callers := object.get(operation_contract, "callers", ["principal", "service_principal"])

# A service-only contract requires a direct platform identity unless its
# explicit delegated branch succeeds below. A caller-kind label is insufficient.
operation_direct_caller_allowed if {
    common.request_caller_kind in operation_callers
    "principal" in operation_callers
}
operation_direct_caller_allowed if {
    common.request_caller_kind == "service_principal"
    "service_principal" in operation_callers
    operation_platform_actor
}

operation_branch_passed if {
    not operation_delegated
    operation_direct_caller_allowed
    operation_actor_permissions(object.get(operation_contract, "permissions", []))
}
operation_branch_passed if {
    operation_delegated
    common.request_caller_kind == "service_principal"
    "service_principal" in operation_callers
    requirements := operation_contract.on_behalf_of
    operation_actor_permissions(requirements.actor_permissions)
    operation_subject_permissions(requirements.permissions)
}
operation_branch_passed if {
    operation_delegated
    not operation_contract.on_behalf_of
    common.request_caller_kind == "service_principal"
    "principal" in operation_callers
    operation_subject_permissions(object.get(operation_contract, "permissions", []))
}

operation_supplied_scopes := extract_scopes if extract_scopes
else := []

operation_scopes_passed if {
    supplied := operation_supplied_scopes
    relevant := [scope | scope := supplied[_]; is_platform_scope(scope)]
    count(relevant) == 0
}
operation_scopes_passed if count(object.get(operation_contract, "scopes", [])) == 0
operation_scopes_passed if {
    some scope in object.get(operation_contract, "scopes", [])
    scope in operation_supplied_scopes
}

operation_predicate := object.get(operation_contract, "predicate", "permissions")
operation_predicate_passed if operation_predicate == "permissions"
operation_predicate_passed if {
    operation_predicate == "auth_context"
    operation_platform_actor
    not operation_delegated
}
operation_predicate_passed if {
    operation_predicate in ["iam", "internal_service", "entity_storage"]
    operation_platform_actor
}
operation_predicate_passed if {
    operation_predicate == "entity_storage"
    not operation_delegated
    operation_admin(subject_principals)
}
operation_predicate_passed if operation_predicate == "workspace_visibility"
operation_predicate_passed if {
    operation_predicate == "workspace_deletion"
    operation_platform_actor
    not operation_delegated
}
operation_predicate_passed if {
    operation_predicate == "workspace_deletion"
    operation_admin(subject_principals)
}
operation_predicate_passed if {
    operation_predicate == "workspace_deletion"
    input.facts.created_by in operation_subject_owner_ids
}
operation_predicate_passed if {
    operation_predicate == "secret_binding"
    operation_platform_actor
    operation_delegated
    input.facts.principal_id == operation_subject_id
    not common.valid_service_principal(operation_subject_id)
}
operation_predicate_passed if {
    operation_predicate in ["builder_submit", "builder_read"]
    operation_subject_human
}
operation_predicate_passed if {
    operation_predicate == "builder_complete"
    operation_delegated
    input.facts.created_by == extract_on_behalf_of_principal_id
}
operation_control_permissions := ["iam.create", "iam.delete"] if input.facts.action == "update"
else := ["iam.create"] if input.facts.action == "create"
else := ["iam.read"] if input.facts.action == "read"
else := ["iam.list"] if input.facts.action == "list"
else := ["iam.delete"] if input.facts.action == "delete"

operation_predicate_passed if {
    operation_predicate == "control_entity"
    not input.target.resource_type in ["role_binding", "access_key"]
}
operation_predicate_passed if {
    operation_predicate == "control_entity"
    input.target.resource_type in ["role_binding", "access_key"]
    operation_subject_permissions(operation_control_permissions)
    operation_control_boundaries_passed
}

operation_control_boundaries_passed if operation_workspace != "-"
operation_control_boundaries_passed if {
    operation_workspace == "-"
    workspaces := {workspace |
        some principal in subject_principals
        some workspace
        data.authz.principals[principal].workspaces[workspace]
    }
    every workspace in workspaces {
        every permission in operation_control_permissions {
            operation_direct_permission_at(subject_principals, workspace, permission)
        }
    }
}
operation_predicate_passed if {
    operation_predicate == "access_key"
    operation_subject_human
    input.facts.service_bound
    operation_admin(subject_principals)
}
operation_predicate_passed if {
    operation_predicate == "access_key"
    operation_subject_human
    not input.facts.service_bound
    input.facts.owner == operation_subject_id
}

operation_route_restrictions_passed if object.get(input, "operation", "route") != "route"
operation_route_restrictions_passed if {
    object.get(input, "operation", "route") == "route"
    not deny_request
}

default operation_allowed := false
operation_allowed if {
    operation_contract
    data.authz.decision_contract_version == 2
    operation_route_restrictions_passed
    not operation_fenced
    not object.get(operation_contract, "deny", false)
    operation_branch_passed
    operation_scopes_passed
    operation_predicate_passed
}

# Public probe handling is part of the canonical route decision too.
operation_allowed if {
    data.authz.decision_contract_version == 2
    object.get(input, "operation", "route") == "route"
    lower(extract_method) in ["get", "head"]
    split(extract_path, "?")[0] in ["/health/live", "/health/ready", "/status", "/metrics", "/cluster-info"]
    not deny_request
    not operation_fenced
}

operation_binding := {
    "protocol_version": 2,
    "operation": input.operation,
    "target": input.target,
    "principal_context_digest": input.principal_context_digest,
    "policy_revision": operation_policy_revision,
    "reason": "policy_decision",
    "actor_account_id": object.get(input, "actor_account_id", ""),
    "actor_aliases": object.get(input, "actor_aliases", []),
    "subject_account_id": object.get(input, "subject_account_id", ""),
    "subject_aliases": object.get(input, "subject_aliases", []),
}

operation_workspace_unrestricted if {
    operation_allowed
    operation_platform_actor
    not operation_delegated
}
operation_workspace_unrestricted if {
    operation_allowed
    operation_admin(subject_principals)
}

operation_visible_workspaces := sort({workspace |
    some principal in subject_principals | {"*"}
    some workspace
    roles := data.authz.principals[principal].workspaces[workspace]
    count(roles) > 0
    workspace != ""
})

operation_workspace_access := {"kind": "denied"} if not operation_allowed
else := {"kind": "unrestricted"} if operation_workspace_unrestricted
else := {"kind": "restricted", "workspaces": operation_visible_workspaces}

operation_field_write if {
    operation_allowed
    not operation_delegated
    input.target.resource_type in data.authz.auth_context_writers[extract_principal_id]
}

decision := object.union(operation_binding, {
    "decision_type": "field", "field": "auth_context",
    "read": operation_allowed, "write": operation_field_write_allowed,
}) if {
    operation_predicate == "auth_context"
} else := object.union(operation_binding, {
    "decision_type": "workspaces", "access": operation_workspace_access,
}) if {
    operation_predicate == "workspace_visibility"
} else := object.union(operation_binding, {
    "decision_type": "operation", "allowed": operation_allowed,
})

default operation_field_write_allowed := false
operation_field_write_allowed if operation_field_write


operation_policy_revision := revision if { revision := data.authz.config.policy_revision }
else := "unversioned"

# Quarantine is mandatory for both route and named-operation entry paths.
operation_fenced if {
    some prefix in data.authz.config.denied_plugin_prefixes
    path_under_denied_prefix(extract_path, prefix)
}
operation_fenced if {
    some prefix in data.authz.config.denied_plugin_prefixes
    prefix == concat("", ["/apis/", operation_contract.owner])
}

operation_legacy_email := email if { email := extract_principal_email } else := ""
operation_legacy_groups := groups if { groups := extract_principal_groups } else := []

operation_route_method := "get" if {
    lower(extract_method) == "head"
    not data.authz.endpoints[common.endpoint_scan].head
} else := lower(extract_method)

# A declared cross-workspace collection may return an empty, policy-filtered
# result. Item routes and explicit dual contracts cannot use this admission.
operation_collection_read if {
    object.get(input, "operation", "route") == "route"
    workspace_scan == "-"
    lower(extract_method) in ["get", "head"]
    not operation_contract.on_behalf_of
    pattern := common.endpoint_scan
    not endswith(pattern, "}")
    count([part | part := split(pattern, "{")[_]]) == 2
}

operation_branch_passed if {
    operation_collection_read
    count(subject_principals) > 0
    not operation_delegated
    operation_direct_caller_allowed
}
operation_branch_passed if {
    operation_collection_read
    count(subject_principals) > 0
    operation_delegated
    common.request_caller_kind == "service_principal"
    "principal" in operation_callers
}

# Static trust for reading role-binding facts while assembling the next snapshot.
# The caller still receives a normal canonical decision for its actual operation.
operation_predicate_passed if {
    operation_predicate == "policy_bootstrap"
    extract_principal_id == "service:auth"
    not operation_delegated
    input.target.resource_type == "role_binding"
}
operation_predicate_passed if {
    operation_predicate == "policy_bootstrap"
    extract_principal_id == "service:auth"
    not operation_delegated
    input.target.kind == "request"
    lower(input.target.method) in ["get", "head"]
    endswith(input.target.path, "/entities/role_binding")
}
