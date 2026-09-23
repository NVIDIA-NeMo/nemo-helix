# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for RulesLoader.
"""

import json
import tempfile

import pytest
from nhx.common.errors.loader import RulesLoader

# =============================================================================
# TEST EXCEPTION CLASSES
# =============================================================================


class CustomError(Exception):
    """Base custom error for testing."""

    def __init__(self, message: str = "", detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class NetworkError(CustomError):
    """Network error for testing."""

    pass


class TimeoutError_(CustomError):
    """Timeout error for testing."""

    pass


class CudaError(CustomError):
    """CUDA error for testing."""

    pass


class InternalError(CustomError):
    """Internal error for testing."""

    pass


# Exception registry for tests
EXCEPTION_REGISTRY = {
    "CustomError": CustomError,
    "NetworkError": NetworkError,
    "TimeoutError": TimeoutError_,
    "CudaError": CudaError,
    "InternalError": InternalError,
}


# =============================================================================
# FROM_DICT TESTS
# =============================================================================


class TestFromDict:
    """Tests for from_dict loading."""

    def test_empty_rules(self):
        config = {"rules": []}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)
        assert mapper.rule_count == 0

    def test_exact_matcher(self):
        config = {"rules": [{"exact": "Connection refused", "exception": "NetworkError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        result = mapper.convert(ValueError("Connection refused"))
        assert isinstance(result, NetworkError)

    def test_regex_matcher(self):
        config = {"rules": [{"regex": r"^Timeout after \d+ seconds$", "exception": "TimeoutError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("Timeout after 30 seconds")) is not None
        assert mapper.convert(ValueError("Timeout after X seconds")) is None

    def test_contains_matcher(self):
        config = {"rules": [{"contains": "CUDA", "exception": "CudaError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(RuntimeError("CUDA out of memory")) is not None

    def test_starts_with_matcher(self):
        config = {"rules": [{"starts_with": "Error:", "exception": "InternalError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("Error: something")) is not None
        assert mapper.convert(ValueError("Warning: something")) is None

    def test_ends_with_matcher(self):
        config = {"rules": [{"ends_with": "try again", "exception": "NetworkError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("Connection failed, try again")) is not None
        assert mapper.convert(ValueError("Connection failed")) is None

    def test_type_name_matcher(self):
        config = {"rules": [{"type_name": "ValueError", "exception": "InternalError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("test")) is not None
        assert mapper.convert(TypeError("test")) is None

    def test_type_matcher_builtin(self):
        config = {"rules": [{"type": "ValueError", "exception": "InternalError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("test")) is not None

    def test_all_keywords_matcher(self):
        config = {"rules": [{"all_keywords": ["CUDA", "memory"], "exception": "CudaError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(RuntimeError("CUDA out of memory")) is not None
        assert mapper.convert(RuntimeError("CUDA error")) is None

    def test_any_keywords_matcher(self):
        config = {"rules": [{"any_keywords": ["CUDA", "GPU"], "exception": "CudaError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(RuntimeError("CUDA error")) is not None
        assert mapper.convert(RuntimeError("GPU memory")) is not None
        assert mapper.convert(RuntimeError("CPU error")) is None

    def test_or_matcher(self):
        config = {
            "rules": [
                {
                    "or": [
                        {"contains": "CUDA"},
                        {"contains": "GPU"},
                    ],
                    "exception": "CudaError",
                }
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(RuntimeError("CUDA error")) is not None
        assert mapper.convert(RuntimeError("GPU memory")) is not None

    def test_and_matcher(self):
        config = {
            "rules": [
                {
                    "and": [
                        {"type_name": "RuntimeError"},
                        {"contains": "CUDA"},
                    ],
                    "exception": "CudaError",
                }
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(RuntimeError("CUDA error")) is not None
        assert mapper.convert(ValueError("CUDA error")) is None

    def test_not_matcher(self):
        config = {"rules": [{"not": {"contains": "warning"}, "exception": "InternalError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert mapper.convert(ValueError("error")) is not None
        assert mapper.convert(ValueError("warning")) is None

    def test_cause_matcher(self):
        config = {
            "rules": [
                {
                    # Use type_name for string matching (doesn't need registry lookup)
                    "cause": {"type_name": "OSError"},
                    "exception": "InternalError",
                }
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        # Create exception with cause
        cause = OSError("file not found")
        exc = RuntimeError("wrapped")
        exc.__cause__ = cause

        assert mapper.convert(exc) is not None

        # Also test no match when cause is different type
        cause2 = ValueError("wrong type")
        exc2 = RuntimeError("wrapped2")
        exc2.__cause__ = cause2
        assert mapper.convert(exc2) is None

    def test_cause_matcher_recursive(self):
        """Test recursive cause matching from config."""
        config = {"rules": [{"cause": {"type_name": "ValueError", "recursive": True}, "exception": "InternalError"}]}
        converter = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        # Build chain: outer -> middle -> root
        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Should find ValueError deep in the chain
        assert converter.convert(outer) is not None

    def test_cause_matcher_recursive_with_matcher_key(self):
        """Test recursive cause matching using explicit 'matcher' key."""
        config = {
            "rules": [
                {"cause": {"matcher": {"type_name": "ValueError"}, "recursive": True}, "exception": "InternalError"}
            ]
        }
        converter = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        # Build chain: outer -> middle -> root
        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Should find ValueError deep in the chain
        assert converter.convert(outer) is not None

    def test_attribute_matcher(self):
        config = {"rules": [{"attribute": {"name": "errno", "value": 2}, "exception": "InternalError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        exc = OSError("test")
        exc.errno = 2
        assert mapper.convert(exc) is not None

        exc2 = OSError("test")
        exc2.errno = 13
        assert mapper.convert(exc2) is None


# =============================================================================
# ERROR_DETAILS TESTS
# =============================================================================


class TestErrorDetails:
    """Tests for error_details field."""

    def test_error_details_passed_to_exception(self):
        config = {
            "rules": [
                {"contains": "network", "exception": "NetworkError", "error_details": "Could not connect to server"}
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        result = mapper.convert(ValueError("network error"))
        assert isinstance(result, NetworkError)
        assert result.message == "Could not connect to server"

    def test_no_error_details_uses_original(self):
        config = {"rules": [{"contains": "network", "exception": "NetworkError"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        result = mapper.convert(ValueError("network connection failed"))
        assert isinstance(result, NetworkError)
        assert result.message == "network connection failed"


# =============================================================================
# HANDLER REGISTRY TESTS
# =============================================================================


class TestHandlerRegistry:
    """Tests for custom handler support."""

    def test_custom_handler(self):
        def custom_network_handler(exc):
            return NetworkError(message=f"Network issue: {exc}", detail="custom detail")

        handler_registry = {"custom_network": custom_network_handler}

        config = {"rules": [{"contains": "network", "exception": "NetworkError", "handler": "custom_network"}]}
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY, handler_registry=handler_registry)

        result = mapper.convert(ValueError("network error"))
        assert isinstance(result, NetworkError)
        assert "Network issue:" in result.message
        assert result.detail == "custom detail"

    def test_unknown_handler_raises(self):
        config = {"rules": [{"contains": "test", "exception": "NetworkError", "handler": "nonexistent_handler"}]}

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "Unknown handler" in str(exc_info.value)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestLoaderErrorHandling:
    """Tests for loader error handling."""

    def test_missing_exception_field(self):
        config = {
            "rules": [
                {"contains": "test"}  # Missing "exception"
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "missing 'exception' field" in str(exc_info.value)

    def test_unknown_exception_class(self):
        config = {"rules": [{"contains": "test", "exception": "NonexistentError"}]}

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "Unknown exception class" in str(exc_info.value)

    def test_missing_matcher_field(self):
        config = {
            "rules": [
                {"exception": "NetworkError"}  # No matcher
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "Rule must have one of" in str(exc_info.value)

    def test_handler_and_error_details_mutually_exclusive(self):
        """handler and error_details cannot be specified together."""

        def dummy_handler(exc):
            return NetworkError(str(exc))

        handler_registry = {"my_handler": dummy_handler}

        config = {
            "rules": [
                {
                    "contains": "test",
                    "exception": "NetworkError",
                    "handler": "my_handler",
                    "error_details": "This should not be allowed",
                }
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY, handler_registry=handler_registry)

        assert "cannot have both 'handler' and 'error_details'" in str(exc_info.value)

    def test_invalid_all_keywords_type(self):
        config = {"rules": [{"all_keywords": "not a list", "exception": "NetworkError"}]}

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "'all_keywords' must be a list" in str(exc_info.value)

    def test_invalid_or_type(self):
        config = {"rules": [{"or": "not a list", "exception": "NetworkError"}]}

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "'or' must be a list" in str(exc_info.value)

    def test_invalid_attribute_config(self):
        config = {
            "rules": [
                {"attribute": {"name": "errno"}, "exception": "NetworkError"}  # Missing value
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        assert "'attribute' requires 'name' and 'value'" in str(exc_info.value)


# =============================================================================
# FROM_JSON TESTS
# =============================================================================


class TestFromJson:
    """Tests for from_json loading."""

    def test_load_from_json_file(self):
        config = {"rules": [{"contains": "CUDA", "exception": "CudaError"}]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            mapper = RulesLoader.from_json(f.name, EXCEPTION_REGISTRY)

            assert mapper.rule_count == 1
            assert mapper.convert(RuntimeError("CUDA error")) is not None

    def test_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            RulesLoader.from_json("/nonexistent/path.json", EXCEPTION_REGISTRY)


# =============================================================================
# DEFAULT HANDLER TESTS
# =============================================================================


class TestDefaultHandler:
    """Tests for custom default handler."""

    def test_custom_default_handler(self):
        def my_default_handler(exc_class, original, error_details):
            return exc_class(message=error_details or "Default message", detail=f"Wrapped: {original}")

        config = {"rules": [{"contains": "test", "exception": "NetworkError"}]}

        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY, default_handler=my_default_handler)

        result = mapper.convert(ValueError("test error"))
        assert isinstance(result, NetworkError)
        assert result.message == "Default message"
        assert "Wrapped:" in result.detail


# =============================================================================
# MULTIPLE RULES TESTS
# =============================================================================


class TestMultipleRules:
    """Tests for multiple rules."""

    def test_first_match_wins(self):
        config = {
            "rules": [
                {"contains": "error", "exception": "NetworkError"},
                {"contains": "error", "exception": "CudaError"},
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        result = mapper.convert(ValueError("some error"))
        assert isinstance(result, NetworkError)

    def test_rules_in_order(self):
        config = {
            "rules": [
                {"exact": "CUDA out of memory", "exception": "CudaError"},
                {"contains": "CUDA", "exception": "InternalError"},
            ]
        }
        mapper = RulesLoader.from_dict(config, EXCEPTION_REGISTRY)

        # Exact match should win
        result1 = mapper.convert(RuntimeError("CUDA out of memory"))
        assert isinstance(result1, CudaError)

        # Contains match for other CUDA errors
        result2 = mapper.convert(RuntimeError("CUDA initialization failed"))
        assert isinstance(result2, InternalError)
