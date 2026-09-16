# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.entities.client import EntityBase
from pydantic import Field, PrivateAttr


class PlatformSecret(EntityBase):
    """A platform secret, which represents a secret entity within the platform."""

    description: str | None = Field(None, description="An optional description of the secret")

    # Secret data is stored as a private attribute and not exposed in API responses
    _data: str = PrivateAttr()

    # The encrypted data encryption key (DEK) used to encrypt the secret data. Can only be decrypted by the platform's key encryption key (KEK)
    _encrypted_dek: str = PrivateAttr()

    # This defines which secret provider was used to encrypt/decrypt the dek
    _secret_provider: str = PrivateAttr()  # E.g. v1, v2, ...
