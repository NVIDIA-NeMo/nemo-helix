# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upstream request errors preserved by the platform's exception handlers."""

from nemo_helix_plugin.client.errors import (
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

# Do not treat arbitrary ValueError/ValidationError as bad client input.
LIST_REQUEST_ERRORS = (
    NotFoundError,
    NemoEntityNotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    PermissionDeniedError,
)

LOOKUP_REQUEST_ERRORS = (NotFoundError, PermissionDeniedError)
