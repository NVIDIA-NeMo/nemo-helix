# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder request fixtures use production declarations and compiled policy."""

import asyncio
from dataclasses import dataclass

import httpx
from nemo_builder_plugin._perms import BuildPerms, ContainerImagePerms
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.service import BuilderService
from nemo_helix_plugin.auth import RuntimeAuthClient
from nemo_helix_plugin.auth.decisions import DecisionRequest, ImageFacts, ResourceTarget
from nhx.testing.policy import PolicyHarness

_POLICY = PolicyHarness()
_POLICY.merge_service(BuilderService())
_POLICY.document["authz"]["roles"]["TestBuilder"] = {
    "permissions": [BuildPerms.CREATE.id, ContainerImagePerms.READ.id, ContainerImagePerms.COMPLETE.id]
}
for user in ["alice", "bob", "carol"]:
    _POLICY.document["authz"]["principals"][user] = {
        "workspaces": {workspace: ["TestBuilder"] for workspace in ["ws-a", "team-a", "default"]}
    }


@dataclass(frozen=True)
class Caller:
    auth: bool
    actor: str | None = None
    submitter: str | None = None


def authorizer(caller: Caller, *, http_client: httpx.AsyncClient | None = None) -> RuntimeAuthClient:
    return _POLICY.auth_client(
        actor=caller.actor or "",
        subject=caller.submitter,
        enabled=caller.auth,
        service_name="builder",
        http_client=http_client,
    )


def allows(caller: Caller, operation: str, row: ContainerImage | None = None) -> bool:
    async def check() -> bool:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_POLICY.respond)) as transport:
            client = authorizer(caller, http_client=transport)
            request = DecisionRequest(
                operation=operation,
                target=ResourceTarget(
                    workspace=row.workspace if row is not None else "ws-a",
                    resource_type="container_image",
                    name=row.name if row is not None else "",
                ),
            )
            if row is not None:
                request = request.model_copy(update={"facts": ImageFacts(created_by=row.created_by or "")})
            return await client.has_operation(request)

    return asyncio.run(check())
