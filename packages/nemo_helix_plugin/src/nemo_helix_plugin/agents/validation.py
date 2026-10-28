# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operation-time checks for registered agents."""

from typing import Any

from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import (
    InternalServerError,
    NemoResponseValidationError,
    NemoTransportError,
    NotFoundError,
    PermissionDeniedError,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobDependencyUnavailableError
from nemo_helix_plugin.refs import parse_entity_ref


async def require_registered_agent_config(agent: str | None, *, workspace: str, async_sdk: AsyncHelixClient) -> None:
    """Reject a name-only agent before submitting work that requires its config."""
    if agent is None:
        return
    ref = parse_entity_ref(agent, workspace)
    client = client_from_platform(async_sdk, AsyncAgentsClient)
    try:
        registered = (await client.get_agent(workspace=ref.workspace, name=ref.name)).data()
    except NotFoundError as exc:
        raise ValueError(f"Agent '{ref.workspace}/{ref.name}' not found.") from exc
    except PermissionDeniedError as exc:
        raise PermissionError(f"Access denied to agent '{ref.workspace}/{ref.name}'.") from exc
    except (InternalServerError, NemoResponseValidationError, NemoTransportError) as exc:
        raise HelixJobDependencyUnavailableError(
            "Unable to validate the agent: the Agents service is unavailable."
        ) from exc
    require_agent_config(registered.config, name=ref.name, workspace=ref.workspace)


def require_agent_config(config: dict[str, Any], *, name: str, workspace: str) -> None:
    """Reject an empty config already resolved from a registered agent."""
    if not config:
        raise ValueError(
            f"Agent '{workspace}/{name}' has no configuration. "
            "This operation requires an agent registered with a non-empty config."
        )
