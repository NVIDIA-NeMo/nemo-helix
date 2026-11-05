# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared policy contracts. These models validate decisions; they do not resolve grants."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DelegatedRequirements(PolicyModel):
    actor_permissions: list[str]
    permissions: list[str]


OperationPredicate = Literal[
    "permissions",
    "auth_context",
    "workspace_visibility",
    "iam",
    "internal_service",
    "entity_storage",
    "policy_bootstrap",
    "workspace_deletion",
    "secret_binding",
    "builder_submit",
    "builder_read",
    "builder_complete",
    "control_entity",
    "access_key",
]
OPERATION_PREDICATE_ADAPTER: TypeAdapter[OperationPredicate] = TypeAdapter(OperationPredicate)


class OperationRule(PolicyModel):
    owner: str
    predicate: OperationPredicate = "permissions"
    permissions: list[str]
    callers: list[Literal["principal", "service_principal"]]
    scopes: list[str]
    on_behalf_of: DelegatedRequirements | None = None


class RequestTarget(PolicyModel):
    kind: Literal["request"] = "request"
    method: str
    path: str


class ResourceTarget(PolicyModel):
    kind: Literal["resource"] = "resource"
    workspace: str
    resource_type: str
    name: str


class CollectionTarget(PolicyModel):
    kind: Literal["collection"] = "collection"
    workspace: str
    resource_type: str


class FieldTarget(PolicyModel):
    kind: Literal["field"] = "field"
    workspace: str
    resource_type: str
    field: Literal["auth_context"] = "auth_context"


DecisionTarget = Annotated[RequestTarget | ResourceTarget | CollectionTarget | FieldTarget, Field(discriminator="kind")]


class NoResourceFacts(PolicyModel):
    kind: Literal["none"] = "none"


class LifecycleFacts(PolicyModel):
    kind: Literal["lifecycle"] = "lifecycle"
    created_by: str


class EntityFacts(PolicyModel):
    kind: Literal["entity"] = "entity"
    action: Literal["create", "read", "list", "update", "delete"]


class ImageFacts(PolicyModel):
    kind: Literal["image"] = "image"
    created_by: str


class AccessKeyFacts(PolicyModel):
    kind: Literal["access_key"] = "access_key"
    owner: str
    service_bound: bool


class CreatorFacts(PolicyModel):
    kind: Literal["creator"] = "creator"
    principal_id: str


ResourceFacts = Annotated[
    NoResourceFacts | LifecycleFacts | EntityFacts | ImageFacts | AccessKeyFacts | CreatorFacts,
    Field(discriminator="kind"),
]


class DecisionRequest(PolicyModel):
    protocol_version: Literal[2] = 2
    operation: str
    target: DecisionTarget
    facts: ResourceFacts = NoResourceFacts()


class DecisionEvaluationInput(DecisionRequest):
    """Trusted service-to-PDP envelope, never populated from a public request body."""

    protocol_version: Literal[2]
    principal_id: str
    principal_context_digest: str
    actor_email: str = ""
    actor_groups: list[str] = Field(default_factory=list)
    actor_account_id: str = ""
    actor_aliases: list[str] = Field(default_factory=list)
    on_behalf_of_principal_id: str = ""
    subject_email: str = ""
    subject_groups: list[str] = Field(default_factory=list)
    subject_account_id: str = ""
    subject_aliases: list[str] = Field(default_factory=list)
    caller_kind: Literal["principal", "service_principal"]
    verified_workload_delegation: bool = False
    scopes: list[str] = Field(default_factory=list)
    identity_resolution: dict[str, JsonValue] = Field(default_factory=dict)
    method: str = ""
    path: str = ""

    @model_validator(mode="after")
    def bind_route_target(self) -> Self:
        if self.operation == "route":
            if (
                not isinstance(self.target, RequestTarget)
                or self.method != self.target.method
                or self.path != self.target.path
            ):
                raise ValueError("Route evaluation must match its typed method and path target")
        return self


class DecisionBinding(PolicyModel):
    protocol_version: Literal[2]
    operation: str
    target: DecisionTarget
    principal_context_digest: str
    policy_revision: str
    reason: Literal["policy_decision", "auth_disabled"]
    actor_account_id: str = ""
    actor_aliases: list[str] = Field(default_factory=list)
    subject_account_id: str = ""
    subject_aliases: list[str] = Field(default_factory=list)


class OperationDecision(DecisionBinding):
    target: Annotated[RequestTarget | ResourceTarget, Field(discriminator="kind")]
    decision_type: Literal["operation"]
    allowed: bool


class UnrestrictedWorkspaces(PolicyModel):
    kind: Literal["unrestricted"]


class RestrictedWorkspaces(PolicyModel):
    kind: Literal["restricted"]
    workspaces: list[str]


class DeniedWorkspaces(PolicyModel):
    kind: Literal["denied"]


WorkspaceAccess = Annotated[
    UnrestrictedWorkspaces | RestrictedWorkspaces | DeniedWorkspaces, Field(discriminator="kind")
]


class WorkspaceDecision(DecisionBinding):
    target: CollectionTarget
    decision_type: Literal["workspaces"]
    access: WorkspaceAccess


class FieldDecision(DecisionBinding):
    target: FieldTarget
    decision_type: Literal["field"]
    field: Literal["auth_context"]
    read: bool
    write: bool


PolicyDecision = Annotated[OperationDecision | WorkspaceDecision | FieldDecision, Field(discriminator="decision_type")]
POLICY_DECISION_ADAPTER: TypeAdapter[PolicyDecision] = TypeAdapter(PolicyDecision)


def validate_decision_result(document: DecisionEvaluationInput, payload: object) -> PolicyDecision:
    """Validate an untrusted response before enforcing its decision."""
    decision = POLICY_DECISION_ADAPTER.validate_python(payload)
    if (
        decision.operation != document.operation
        or decision.target != document.target
        or decision.principal_context_digest != document.principal_context_digest
        or not decision.policy_revision
    ):
        raise ValueError("PDP decision does not match its request context and target")
    return decision
