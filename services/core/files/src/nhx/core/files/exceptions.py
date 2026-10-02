# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions for the Files service."""


class StorageBackendError(Exception):
    """Raised when a storage backend has an error."""


class StorageAccessError(StorageBackendError):
    """Raised when access to a storage backend is denied.

    This is used for authentication/authorization failures when accessing
    external storage (e.g., gated repos, invalid tokens).
    """


class StorageConfigError(StorageBackendError):
    """Raised when storage configuration is invalid.

    This is used when the user-provided storage config points to a resource
    that doesn't exist (e.g., repo not found, revision not found).
    """


class StorageUnavailableError(StorageBackendError):
    """Raised when a storage backend is unavailable.

    This is used for upstream service outages, rate limiting, or timeouts
    when accessing external storage (e.g., HuggingFace 5xx, 429, timeouts).
    """


class NotFoundError(Exception):
    """Error when the resource isn't found."""


class InactivityTimeoutError(Exception):
    """Raised when no data is received within the configured timeout period."""


class InvalidRangeError(Exception):
    """Raised when the HTTP GET range was invalid."""


class InvalidPathError(Exception):
    """Raised when the provided `path` was invalid."""


class InvalidFilterError(Exception):
    """Raised when the provided filters are invalid."""
