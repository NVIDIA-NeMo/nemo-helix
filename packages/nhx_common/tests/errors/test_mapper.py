# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ExceptionConverter.
"""

import pytest
from nhx.common.errors.converter import ExceptionConverter
from nhx.common.errors.matchers import (
    ContainsMatcher,
    OrMatcher,
)

# =============================================================================
# TEST EXCEPTION CLASSES
# =============================================================================


class CustomError(Exception):
    """Custom error for testing."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class NetworkError(CustomError):
    """Network error for testing."""

    pass


class TimeoutError_(CustomError):
    """Timeout error for testing (avoiding builtin TimeoutError)."""

    pass


class CudaError(CustomError):
    """CUDA error for testing."""

    pass


# =============================================================================
# BASIC CONVERTER TESTS
# =============================================================================


class TestExceptionConverterBasic:
    """Basic tests for ExceptionConverter."""

    def test_empty_converter(self):
        converter = ExceptionConverter()
        assert converter.rule_count == 0
        assert converter.convert(ValueError("test")) is None

    def test_add_rule(self):
        converter = ExceptionConverter()
        converter.add_rule(ContainsMatcher("test"), lambda e: CustomError(str(e)))
        assert converter.rule_count == 1

    def test_add_rule_chaining(self):
        converter = (
            ExceptionConverter()
            .add_rule(ContainsMatcher("a"), lambda _e: CustomError("a"))
            .add_rule(ContainsMatcher("b"), lambda _e: CustomError("b"))
        )
        assert converter.rule_count == 2

    def test_convert_match(self):
        converter = ExceptionConverter()
        converter.add_rule(ContainsMatcher("network"), lambda e: NetworkError(str(e)))

        result = converter.convert(ValueError("network error"))
        assert isinstance(result, NetworkError)
        assert result.message == "network error"

    def test_convert_no_match(self):
        converter = ExceptionConverter()
        converter.add_rule(ContainsMatcher("network"), lambda e: NetworkError(str(e)))

        result = converter.convert(ValueError("something else"))
        assert result is None

    def test_convert_first_match_wins(self):
        """When multiple rules match, first one wins."""
        converter = ExceptionConverter()
        converter.add_rule(ContainsMatcher("error"), lambda _e: NetworkError("first"))
        converter.add_rule(ContainsMatcher("error"), lambda _e: CudaError("second"))

        result = converter.convert(ValueError("some error"))
        assert isinstance(result, NetworkError)
        assert result.message == "first"


# =============================================================================
# CONVENIENCE METHOD TESTS
# =============================================================================


class TestConverterConvenienceMethods:
    """Tests for convenience methods (add_exact, add_regex, etc.)."""

    def test_add_exact(self):
        converter = ExceptionConverter()
        converter.add_exact("Connection refused", lambda e: NetworkError(str(e)))

        assert converter.convert(ValueError("Connection refused")) is not None
        assert converter.convert(ValueError("Connection timeout")) is None

    def test_add_regex(self):
        converter = ExceptionConverter()
        converter.add_regex(r"^Timeout after \d+ seconds$", lambda e: TimeoutError_(str(e)))

        assert converter.convert(ValueError("Timeout after 30 seconds")) is not None
        assert converter.convert(ValueError("Timeout after X seconds")) is None

    def test_add_contains(self):
        converter = ExceptionConverter()
        converter.add_contains("CUDA", lambda e: CudaError(str(e)))

        assert converter.convert(RuntimeError("CUDA out of memory")) is not None
        assert converter.convert(RuntimeError("CPU out of memory")) is None

    def test_add_type(self):
        converter = ExceptionConverter()
        converter.add_type(ValueError, lambda e: CustomError(str(e)))

        assert converter.convert(ValueError("test")) is not None
        assert converter.convert(TypeError("test")) is None


# =============================================================================
# RAISE_CONVERTED_OR_ORIGINAL TESTS
# =============================================================================


class TestRaiseConvertedOrOriginal:
    """Tests for raise_converted_or_original method."""

    def test_match_raises_converted(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("network error")
        with pytest.raises(NetworkError) as exc_info:
            converter.raise_converted_or_original(original)

        assert isinstance(exc_info.value, NetworkError)

    def test_sets_cause(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("network error")
        with pytest.raises(NetworkError) as exc_info:
            converter.raise_converted_or_original(original)

        # Verify __cause__ is set to the original exception
        assert exc_info.value.__cause__ is original

    def test_no_match_raises_original(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("something else")
        with pytest.raises(ValueError) as exc_info:
            converter.raise_converted_or_original(original)

        assert exc_info.value is original


# =============================================================================
# RAISE_CONVERTED_OR_DEFAULT TESTS
# =============================================================================


class TestRaiseConvertedOrDefault:
    """Tests for raise_converted_or_default method."""

    def test_match_raises_converted(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("network error")
        with pytest.raises(NetworkError) as exc_info:
            converter.raise_converted_or_default(original, lambda _e: CustomError("fallback"))

        assert isinstance(exc_info.value, NetworkError)

    def test_match_sets_cause(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("network error")
        with pytest.raises(NetworkError) as exc_info:
            converter.raise_converted_or_default(original, lambda _e: CustomError("fallback"))

        assert exc_info.value.__cause__ is original

    def test_no_match_raises_default(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("something else")
        with pytest.raises(CustomError) as exc_info:
            converter.raise_converted_or_default(original, lambda e: CustomError("fallback", detail=str(e)))

        assert exc_info.value.message == "fallback"
        assert exc_info.value.detail == "something else"

    def test_no_match_sets_cause_on_default(self):
        converter = ExceptionConverter()
        converter.add_contains("network", lambda e: NetworkError(str(e)))

        original = ValueError("something else")
        with pytest.raises(CustomError) as exc_info:
            converter.raise_converted_or_default(original, lambda _e: CustomError("fallback"))

        # Verify __cause__ is set even for default handler
        assert exc_info.value.__cause__ is original


# =============================================================================
# COMPLEX SCENARIOS
# =============================================================================


class TestConverterComplexScenarios:
    """Tests for complex converter scenarios."""

    def test_multiple_rules_different_types(self):
        converter = (
            ExceptionConverter()
            .add_contains("CUDA", lambda e: CudaError(str(e)))
            .add_contains("network", lambda e: NetworkError(str(e)))
            .add_regex(r"timeout", lambda e: TimeoutError_(str(e)))
        )

        assert isinstance(converter.convert(RuntimeError("CUDA error")), CudaError)
        assert isinstance(converter.convert(RuntimeError("network error")), NetworkError)
        assert isinstance(converter.convert(RuntimeError("connection timeout")), TimeoutError_)
        assert converter.convert(RuntimeError("something else")) is None

    def test_handler_receives_original_exception(self):
        """Verify handler receives the original exception."""
        received_exception = None

        def capture_handler(e):
            nonlocal received_exception
            received_exception = e
            return CustomError(str(e))

        converter = ExceptionConverter()
        converter.add_contains("test", capture_handler)

        original = ValueError("test message")
        converter.convert(original)

        assert received_exception is original

    def test_with_custom_matcher(self):
        """Test with a custom OrMatcher."""
        converter = ExceptionConverter()
        converter.add_rule(
            OrMatcher(
                [
                    ContainsMatcher("CUDA"),
                    ContainsMatcher("GPU"),
                ]
            ),
            lambda e: CudaError(str(e)),
        )

        assert isinstance(converter.convert(RuntimeError("CUDA error")), CudaError)
        assert isinstance(converter.convert(RuntimeError("GPU memory")), CudaError)
        assert converter.convert(RuntimeError("CPU error")) is None

    def test_initialization_with_rules(self):
        """Test initializing converter with rules list."""
        rules = [
            (ContainsMatcher("network"), lambda e: NetworkError(str(e))),
            (ContainsMatcher("CUDA"), lambda e: CudaError(str(e))),
        ]
        converter = ExceptionConverter(rules=rules)

        assert converter.rule_count == 2
        assert isinstance(converter.convert(RuntimeError("network")), NetworkError)
        assert isinstance(converter.convert(RuntimeError("CUDA")), CudaError)
