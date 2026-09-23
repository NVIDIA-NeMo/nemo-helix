# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and shared fixtures for error_mapping tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path so we can import the package
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# =============================================================================
# SHARED TEST EXCEPTION CLASSES
# =============================================================================


class BaseTestError(Exception):
    """Base exception for all test exceptions."""

    def __init__(self, message: str = "", detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class NetworkError(BaseTestError):
    """Network-related errors."""

    pass


class CudaError(BaseTestError):
    """GPU/CUDA errors."""

    pass


class TimeoutError_(BaseTestError):
    """Timeout errors (avoiding builtin name)."""

    pass


class InternalError(BaseTestError):
    """Internal/unexpected errors."""

    pass


class DatasetError(BaseTestError):
    """Dataset-related errors."""

    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def exception_registry():
    """Standard exception registry for tests."""
    return {
        "BaseTestError": BaseTestError,
        "NetworkError": NetworkError,
        "CudaError": CudaError,
        "TimeoutError": TimeoutError_,
        "InternalError": InternalError,
        "DatasetError": DatasetError,
    }


@pytest.fixture
def sample_factory_registry():
    """Sample factory registry for tests."""

    def network_error_factory(exc):
        return NetworkError(message=f"Network failed: {exc}", detail="Factory-created")

    def cuda_error_factory(exc):
        return CudaError(message=f"GPU error: {exc}", detail="Try reducing batch size")

    return {
        "network_factory": network_error_factory,
        "cuda_factory": cuda_error_factory,
    }


@pytest.fixture
def basic_rules_config():
    """Basic rules configuration for tests."""
    return {
        "rules": [
            {"exact": "Connection refused", "exception": "NetworkError"},
            {"contains": "CUDA", "exception": "CudaError"},
            {"regex": r"timeout", "exception": "TimeoutError"},
        ]
    }


@pytest.fixture
def complex_rules_config():
    """Complex rules configuration with various matchers."""
    return {
        "rules": [
            # Type-based
            {"type_name": "TimeoutError", "exception": "TimeoutError"},
            # Composite
            {
                "and": [
                    {"type_name": "RuntimeError"},
                    {"contains": "CUDA"},
                ],
                "exception": "CudaError",
                "error_details": "GPU memory issue",
            },
            # Or matcher
            {
                "or": [
                    {"contains": "network"},
                    {"contains": "connection"},
                ],
                "exception": "NetworkError",
            },
            # Keywords
            {"all_keywords": ["dataset", "format"], "exception": "DatasetError"},
            # Fallback
            {"contains": "error", "exception": "InternalError"},
        ]
    }
