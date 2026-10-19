# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization exceptions for NeMo Helix services."""

from typing import Optional

from nemo_helix_plugin.authz_format import (
    InvalidPermissionFormatError as InvalidPermissionFormatError,
)
from nemo_helix_plugin.authz_format import (
    InvalidScopeFormatError as InvalidScopeFormatError,
)


class AuthorizationError(Exception):
    """Exception for when a principal is not authorized to perform an operation."""

    def __init__(self, message: str, entity_type: Optional[str] = None, workspace_id: Optional[str] = None):
        self.message = message
        self.entity_type = entity_type
        self.workspace_id = workspace_id
        super().__init__(self.message)


class InvalidPrincipalHeader(ValueError):
    """Raised when an X-NHX-Principal-* header value fails validation.

    The middleware translates this into a 400 Bad Request response.
    """
