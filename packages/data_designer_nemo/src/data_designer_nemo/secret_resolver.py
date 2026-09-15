# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from data_designer.engine.errors import SecretResolutionError
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_platform_plugin.secrets.client import AsyncSecretsClient, SecretsClient

logger = logging.getLogger(__name__)


async def validate_secret(sdk: AsyncNeMoPlatform, secret: str, default_workspace: str) -> None:
    """Validate a secret reference with an async SDK instance.
    The SDK instance should carry end user authentication headers,
    so that this function validate existence and access in API
    endpoints and the job config compiler, *prior to* starting
    Data Designer library engine execution (which requires the
    NMPSecretResolver).
    """
    workspace, name = _parse_secret_reference(secret, default_workspace)
    secrets = client_from_platform(sdk, AsyncSecretsClient)
    try:
        await secrets.access_secret(name=name, workspace=workspace)
    except NotFoundError as e:
        raise NDDInvalidConfigError(f"Could not find secret {name!r} in workspace {workspace!r}") from e
    except PermissionDeniedError as e:
        raise NDDInvalidConfigError(f"Access denied to workspace {workspace!r}") from e
    except Exception as e:
        logger.exception("Error accessing secret", extra={"secret_name": name, "workspace": workspace})
        raise NDDInternalError(
            f"An unexpected error occurred while accessing secret {name!r} in workspace {workspace!r}: {e}"
        ) from e


class NMPSecretResolver:
    """An implementation of the Data Designer library's SecretResolver protocol
    that considers the provided `secret` string a NeMo Platform Secret reference. Providing
    only this secret resolver (and not a composite secret resolver with other types,
    e.g. EnvVar or PlainText resolvers) ensures that in this NeMo Platform context, the DD
    library only accepts NeMo Platform secrets in fields treated as secrets by the library.

    Public ``.resolve(secret) -> str`` is sync because the DD engine library is
    sync. Secrets should be validated in advance using :func:`validate_secret`.
    """

    def __init__(self, sdk: NeMoPlatform, default_workspace: str):
        self._sdk = sdk
        self._default_workspace = default_workspace

    def resolve(self, secret: str) -> str:
        try:
            workspace, name = _parse_secret_reference(secret, self._default_workspace)
            secrets = client_from_platform(self._sdk, SecretsClient)
            result = secrets.access_secret(name=name, workspace=workspace).data()
            return result.value
        except Exception as e:
            raise SecretResolutionError(f"Error resolving secret {secret!r}: {e}") from e


def _parse_secret_reference(secret: str, default_workspace: str) -> tuple[str, str]:
    """Parse a secret reference into workspace and name.

    Args:
        secret: Secret reference in format "name" or "workspace/name"
        default_workspace: Default workspace to use if not specified

    Returns:
        Tuple of (workspace, name)

    Raises:
        NDDInvalidConfigError: If the secret reference is malformed
    """
    match secret.split("/"):
        case [name]:
            return default_workspace, name
        case [workspace, name]:
            return workspace, name
        case _:
            raise NDDInvalidConfigError(f"The secret {secret!r} is formatted incorrectly")
