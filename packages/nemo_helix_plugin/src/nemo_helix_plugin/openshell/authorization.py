# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that configured provider secret references are readable as metadata."""

import hashlib

from nemo_helix_plugin.auth import AuthContext, platform_auth_enabled
from nemo_helix_plugin.auth.decision_client import query_decision
from nemo_helix_plugin.auth.decisions import CreatorFacts, DecisionEvaluationInput, OperationDecision, ResourceTarget
from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def authorize_provider_references(
    bindings: dict[str, GatewayProviderBinding], *, workspace: str, secrets_client: AsyncSecretsClient
) -> None:
    """Read each secret's metadata with the supplied client's identity, without fetching values.

    This does not separately check the workload creator's permission to read values;
    the owning backend performs that check before signing a registration request.
    """

    for ref in sorted({ref for binding in bindings.values() for ref in binding.credentials.values()}):
        secret_workspace, name = parse_qualified_name(ref, default_workspace=workspace)
        await secrets_client.get_secret(workspace=secret_workspace, name=name)


async def require_creator_secret_access(
    *, workspace: str, name: str, creator: AuthContext, actor_service: str, secrets_client: AsyncSecretsClient
) -> None:
    """Evaluate the protected creator snapshot through the ordinary OBO identity fields."""
    delegated = bool(creator.principal_on_behalf_of)
    principal_id = creator.principal_on_behalf_of if delegated else creator.principal_id
    target = ResourceTarget(workspace=workspace, resource_type="secret", name=name)
    facts = CreatorFacts(principal_id=principal_id or "")
    document = DecisionEvaluationInput(
        protocol_version=2,
        operation="secrets.binding.authorize",
        target=target,
        facts=facts,
        principal_id=f"service:{actor_service}",
        caller_kind="service_principal",
        on_behalf_of_principal_id=principal_id or "",
        subject_email=(creator.principal_on_behalf_of_email if delegated else creator.principal_email) or "",
        subject_groups=(creator.principal_on_behalf_of_groups if delegated else creator.principal_groups) or [],
        subject_account_id=(creator.principal_on_behalf_of_account_id if delegated else creator.principal_account_id)
        or "",
        subject_aliases=creator.principal_on_behalf_of_authz_aliases if delegated else creator.principal_authz_aliases,
        principal_context_digest="",
    )
    digest = hashlib.sha256(document.model_dump_json().encode()).hexdigest()
    decision = await query_decision(secrets_client, document.model_copy(update={"principal_context_digest": digest}))
    if not isinstance(decision, OperationDecision) or not decision.allowed:
        raise PermissionError("The workload creator cannot use this secret.")


async def authorize_secret_binding(
    *, workspace: str, name: str, creator: AuthContext | None, actor_service: str, secrets_client: AsyncSecretsClient
) -> None:
    """Authorize the captured creator before signing, then verify the metadata binding."""
    if creator is not None:
        if not platform_auth_enabled():
            raise PermissionError("Secret binding requires a configured creator permission check.")
        await require_creator_secret_access(
            workspace=workspace,
            name=name,
            creator=creator,
            actor_service=actor_service,
            secrets_client=secrets_client,
        )
    secret = (await secrets_client.get_secret(workspace=workspace, name=name)).data()
    if (secret.workspace, secret.name) != (workspace, name):
        raise PermissionError("Secret metadata does not match the requested binding.")
