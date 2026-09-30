# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission checks for dependent resources in Models API endpoints.

Cross-service checks (secrets, filesets) use the request-scoped typed clients
to retrieve the resource. The call goes through the HTTP stack and AuthZ
middleware, so a 403 is raised automatically if the user lacks access.

Same-service checks (deployments, deployment configs, model entities) use
AuthClient.has_permissions to check access without a round-trip HTTP call
to our own API.
"""

from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_helix_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.auth import AuthClient
from nhx.common.entities.utils import parse_entity_ref


async def check_secret_access(secrets: AsyncSecretsClient, secret_name: str, workspace: str) -> None:
    """Check that the current user can access the referenced secret.

    Raises:
        PermissionError: If the user cannot access the secret.
        ValueError: If the secret doesn't exist.
    """
    try:
        await secrets.get_secret(name=secret_name, workspace=workspace)
    except ClientPermissionDeniedError:
        raise PermissionError(f"Access denied to secret '{secret_name}' in workspace '{workspace}'") from None
    except ClientNotFoundError:
        raise ValueError(f"Secret '{secret_name}' not found in workspace '{workspace}'") from None


async def check_fileset_access(files: AsyncFilesClient, fileset: str, workspace: str) -> FilesetOutput:
    """Check that the current user can access the referenced fileset.

    Retrieves fileset metadata via the Files API; AuthZ middleware enforces
    access. fileset format is 'workspace/fileset_name' or just 'fileset_name'.

    Raises:
        PermissionError: If the user cannot access the fileset.
        ValueError: If the fileset does not exist.
    """
    _fs_ref = parse_entity_ref(fileset, default_workspace=workspace)
    fs_workspace, fs_name = _fs_ref.workspace, _fs_ref.name
    try:
        fs = (await files.get_fileset(workspace=fs_workspace, name=fs_name)).data()
        return fs
    except ClientPermissionDeniedError:
        raise PermissionError(f"Access denied to fileset '{fileset}'") from None
    except ClientNotFoundError:
        raise ValueError(f"Fileset '{fileset}' not found in workspace '{fs_workspace}'") from None


async def check_deployment_access(auth_client: AuthClient, deployment_id: str, workspace: str) -> None:
    """Check that the current user can access the referenced deployment.

    Raises:
        PermissionError: If the user cannot access the deployment's workspace.
    """
    deploy_workspace = parse_entity_ref(deployment_id, default_workspace=workspace).workspace
    if not await auth_client.has_permissions(deploy_workspace, ["inference.deployments.read"]):
        raise PermissionError(f"Access denied to deployment '{deployment_id}'")


async def check_deployment_config_access(auth_client: AuthClient, config_name: str, workspace: str) -> None:
    """Check that the current user can access the referenced deployment config.

    Raises:
        PermissionError: If the user cannot access the config's workspace.
    """
    if not await auth_client.has_permissions(workspace, ["inference.deployment-configs.read"]):
        raise PermissionError(f"Access denied to deployment config '{config_name}' in workspace '{workspace}'")


async def check_model_entity_access(auth_client: AuthClient, model_entity_id: str, workspace: str) -> None:
    """Check that the current user can access the referenced model entity.

    model_entity_id can be 'workspace/name' or just 'name' (defaults to request workspace).

    Raises:
        PermissionError: If the user cannot access the model entity's workspace.
    """
    entity_workspace = parse_entity_ref(model_entity_id, default_workspace=workspace).workspace
    if not await auth_client.has_permissions(entity_workspace, ["models.read"]):
        raise PermissionError(f"Access denied to model entity '{model_entity_id}'")


async def can_set_trust_remote_code(auth_client: AuthClient, workspace: str) -> bool:
    """Return True if the active caller has permission to set trust_remote_code outside of the allow_list."""
    if await auth_client.has_permissions(workspace, ["models.trust-remote-code.set"]):
        return True
    return False


async def can_set_tool_call_plugin(auth_client: AuthClient, workspace: str) -> bool:
    """Return True if the active caller has permission to set tool_call_plugin."""
    if await auth_client.has_permissions(workspace, ["models.tool-call-plugin.set"]):
        return True
    return False
