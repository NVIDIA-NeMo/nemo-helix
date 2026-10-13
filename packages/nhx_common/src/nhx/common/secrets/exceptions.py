# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions for secrets management."""


class SecretError(Exception):
    """Custom exception for secret management errors."""


class SecretNotFoundError(SecretError):
    """Custom exception for secret not found errors."""


class SecretAccessDeniedError(SecretError):
    """Custom exception for secret access denied errors."""


class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
