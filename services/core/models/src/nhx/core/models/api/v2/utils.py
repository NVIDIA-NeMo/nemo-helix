# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for API v2."""

from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import Runtime
from nhx.common.entities.client import EntityNotFoundError
from nhx.core.models.config import get_platform_config

# Let the registered SDK handlers preserve not-found status and detail.
NOT_FOUND_ERRORS = (NotFoundError, EntityNotFoundError)


def deployments_enabled() -> bool:
    """Check if deployments are enabled."""
    cfg = get_platform_config()
    return cfg.runtime != Runtime.NONE


ERR_DEPLOYMENTS_NOT_ENABLED = "Deployments are not enabled because the backend runtime is set to 'none'"
