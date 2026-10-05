# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for exception matchers.
"""

import subprocess

import pytest
from nhx.common.errors.matchers import (
    AllKeywordsMatcher,
    AnyKeywordMatcher,
    AttributeMatcher,
    CauseMatcher,
    CompositeMatcher,
    ContainsMatcher,
    EndsWithMatcher,
    ExactMatcher,
    ExceptionTypeMatcher,
    ExceptionTypeNameMatcher,
    NotMatcher,
    OrMatcher,
    RegexMatcher,
    StartsWithMatcher,
)

# =============================================================================
# BASIC MATCHERS
# =============================================================================


class TestExactMatcher:
    """Tests for ExactMatcher."""

    def test_exact_match(self):
        matcher = ExactMatcher("Connection refused")
        assert matcher.matches(ValueError("Connection refused"))

    def test_exact_no_match(self):
        matcher = ExactMatcher("Connection refused")
        assert not matcher.matches(ValueError("Connection timeout"))

    def test_exact_match_requires_exact_whitespace(self):
        """Whitespace differences should NOT match (truly exact)."""
        matcher = ExactMatcher("Error:  multiple   spaces")
        assert matcher.matches(ValueError("Error:  multiple   spaces"))  # Exact match
        assert not matcher.matches(ValueError("Error: multiple spaces"))  # Different whitespace

    def test_exact_match_preserves_newlines(self):
        """Newlines are preserved (truly exact)."""
        matcher = ExactMatcher("Line one\nLine two")
        assert matcher.matches(ValueError("Line one\nLine two"))  # Exact match
        assert not matcher.matches(ValueError("Line one Line two"))  # Newline vs space

    def test_exact_match_case_sensitive(self):
        """Matching should be case-sensitive."""
        matcher = ExactMatcher("Error")
        assert not matcher.matches(ValueError("error"))


class TestRegexMatcher:
    """Tests for RegexMatcher."""

    def test_regex_match_simple(self):
        matcher = RegexMatcher(r"timeout")
        assert matcher.matches(ValueError("Connection timeout occurred"))

    def test_regex_match_pattern(self):
        matcher = RegexMatcher(r"^Timeout after \d+ seconds$")
        assert matcher.matches(ValueError("Timeout after 30 seconds"))

    def test_regex_no_match(self):
        matcher = RegexMatcher(r"^Timeout after \d+ seconds$")
        assert not matcher.matches(ValueError("Connection refused"))

    def test_regex_multiline(self):
        matcher = RegexMatcher(
            r"error.*occurred",
        )
        assert matcher.matches(ValueError("An error has occurred"))

    def test_regex_special_chars(self):
        matcher = RegexMatcher(r"Failed to open \[.*\]")
        assert matcher.matches(ValueError("Failed to open [config.yaml]"))


class TestContainsMatcher:
    """Tests for ContainsMatcher."""

    def test_contains_match(self):
        matcher = ContainsMatcher("CUDA")
        assert matcher.matches(RuntimeError("CUDA out of memory"))

    def test_contains_no_match(self):
        matcher = ContainsMatcher("CUDA")
        assert not matcher.matches(RuntimeError("CPU out of memory"))

    def test_contains_case_sensitive(self):
        matcher = ContainsMatcher("CUDA")
        assert not matcher.matches(RuntimeError("cuda out of memory"))

    def test_contains_substring(self):
        matcher = ContainsMatcher("out of")
        assert matcher.matches(RuntimeError("CUDA out of memory"))


class TestStartsWithMatcher:
    """Tests for StartsWithMatcher."""

    def test_starts_with_match(self):
        matcher = StartsWithMatcher("Error:")
        assert matcher.matches(ValueError("Error: Something went wrong"))

    def test_starts_with_no_match(self):
        matcher = StartsWithMatcher("Error:")
        assert not matcher.matches(ValueError("Warning: Something happened"))

    def test_starts_with_exact(self):
        matcher = StartsWithMatcher("Error")
        assert matcher.matches(ValueError("Error"))


class TestEndsWithMatcher:
    """Tests for EndsWithMatcher."""

    def test_ends_with_match(self):
        matcher = EndsWithMatcher("try again")
        assert matcher.matches(ValueError("Operation failed, try again"))

    def test_ends_with_no_match(self):
        matcher = EndsWithMatcher("try again")
        assert not matcher.matches(ValueError("Operation succeeded"))

    def test_ends_with_exact(self):
        matcher = EndsWithMatcher("failed")
        assert matcher.matches(ValueError("failed"))


# =============================================================================
# TYPE-BASED MATCHERS
# =============================================================================


class TestExceptionTypeMatcher:
    """Tests for ExceptionTypeMatcher."""

    def test_type_match_exact(self):
        matcher = ExceptionTypeMatcher(ValueError)
        assert matcher.matches(ValueError("test"))

    def test_type_match_subclass(self):
        """Should match subclasses too (isinstance behavior)."""
        matcher = ExceptionTypeMatcher(Exception)
        assert matcher.matches(ValueError("test"))

    def test_type_no_match(self):
        matcher = ExceptionTypeMatcher(ValueError)
        assert not matcher.matches(TypeError("test"))

    def test_type_match_timeout_expired(self):
        matcher = ExceptionTypeMatcher(subprocess.TimeoutExpired)
        exc = subprocess.TimeoutExpired(cmd="test", timeout=10)
        assert matcher.matches(exc)


class TestExceptionTypeNameMatcher:
    """Tests for ExceptionTypeNameMatcher."""

    def test_type_name_match(self):
        matcher = ExceptionTypeNameMatcher("ValueError")
        assert matcher.matches(ValueError("test"))

    def test_type_name_no_match(self):
        matcher = ExceptionTypeNameMatcher("ValueError")
        assert not matcher.matches(TypeError("test"))

    def test_type_name_does_not_match_subclass(self):
        """Should NOT match subclasses (exact class name only)."""
        matcher = ExceptionTypeNameMatcher("Exception")
        assert not matcher.matches(ValueError("test"))

    def test_type_name_custom_exception(self):
        class CustomError(Exception):
            pass

        matcher = ExceptionTypeNameMatcher("CustomError")
        assert matcher.matches(CustomError("test"))


# =============================================================================
# KEYWORD MATCHERS
# =============================================================================


class TestAllKeywordsMatcher:
    """Tests for AllKeywordsMatcher."""

    def test_all_keywords_match(self):
        matcher = AllKeywordsMatcher(["CUDA", "memory"])
        assert matcher.matches(RuntimeError("CUDA out of memory"))

    def test_all_keywords_partial_no_match(self):
        """Must have ALL keywords."""
        matcher = AllKeywordsMatcher(["CUDA", "memory", "GPU"])
        assert not matcher.matches(RuntimeError("CUDA out of memory"))

    def test_all_keywords_single(self):
        matcher = AllKeywordsMatcher(["CUDA"])
        assert matcher.matches(RuntimeError("CUDA error"))

    def test_all_keywords_empty_list_raises(self):
        """Empty keyword list should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AllKeywordsMatcher([])
        assert "requires at least one keyword" in str(exc_info.value)


class TestAnyKeywordMatcher:
    """Tests for AnyKeywordMatcher."""

    def test_any_keyword_match_first(self):
        matcher = AnyKeywordMatcher(["CUDA", "GPU", "device"])
        assert matcher.matches(RuntimeError("CUDA error"))

    def test_any_keyword_match_middle(self):
        matcher = AnyKeywordMatcher(["CUDA", "GPU", "device"])
        assert matcher.matches(RuntimeError("GPU memory full"))

    def test_any_keyword_match_last(self):
        matcher = AnyKeywordMatcher(["CUDA", "GPU", "device"])
        assert matcher.matches(RuntimeError("device not found"))

    def test_any_keyword_no_match(self):
        matcher = AnyKeywordMatcher(["CUDA", "GPU", "device"])
        assert not matcher.matches(RuntimeError("CPU error"))

    def test_any_keyword_empty_list_raises(self):
        """Empty keyword list should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AnyKeywordMatcher([])
        assert "requires at least one keyword" in str(exc_info.value)


# =============================================================================
# COMPOSITE MATCHERS
# =============================================================================


class TestCompositeMatcher:
    """Tests for CompositeMatcher (AND logic)."""

    def test_composite_all_match(self):
        matcher = CompositeMatcher(
            [
                ContainsMatcher("error"),
                ContainsMatcher("CUDA"),
            ]
        )
        assert matcher.matches(RuntimeError("CUDA error occurred"))

    def test_composite_partial_no_match(self):
        matcher = CompositeMatcher(
            [
                ContainsMatcher("error"),
                ContainsMatcher("CUDA"),
            ]
        )
        assert not matcher.matches(RuntimeError("CPU error occurred"))

    def test_composite_with_type(self):
        matcher = CompositeMatcher(
            [
                ExceptionTypeNameMatcher("RuntimeError"),
                ContainsMatcher("distributed"),
            ]
        )
        assert matcher.matches(RuntimeError("distributed training failed"))
        assert not matcher.matches(ValueError("distributed training failed"))

    def test_composite_empty(self):
        """Empty composite should match everything."""
        matcher = CompositeMatcher([])
        assert matcher.matches(RuntimeError("anything"))


class TestOrMatcher:
    """Tests for OrMatcher."""

    def test_or_first_match(self):
        matcher = OrMatcher(
            [
                ContainsMatcher("CUDA"),
                ContainsMatcher("GPU"),
            ]
        )
        assert matcher.matches(RuntimeError("CUDA error"))

    def test_or_second_match(self):
        matcher = OrMatcher(
            [
                ContainsMatcher("CUDA"),
                ContainsMatcher("GPU"),
            ]
        )
        assert matcher.matches(RuntimeError("GPU memory"))

    def test_or_no_match(self):
        matcher = OrMatcher(
            [
                ContainsMatcher("CUDA"),
                ContainsMatcher("GPU"),
            ]
        )
        assert not matcher.matches(RuntimeError("CPU error"))

    def test_or_empty(self):
        """Empty or should match nothing."""
        matcher = OrMatcher([])
        assert not matcher.matches(RuntimeError("anything"))


class TestNotMatcher:
    """Tests for NotMatcher."""

    def test_not_inverts_match(self):
        matcher = NotMatcher(ContainsMatcher("warning"))
        assert not matcher.matches(ValueError("This is a warning"))

    def test_not_inverts_no_match(self):
        matcher = NotMatcher(ContainsMatcher("warning"))
        assert matcher.matches(ValueError("This is an error"))


# =============================================================================
# SPECIAL MATCHERS
# =============================================================================


class TestCauseMatcher:
    """Tests for CauseMatcher."""

    def test_cause_match(self):
        matcher = CauseMatcher(ExceptionTypeMatcher(TimeoutError))

        cause = TimeoutError("timed out")
        exc = RuntimeError("Operation failed")
        exc.__cause__ = cause

        assert matcher.matches(exc)

    def test_cause_no_match_different_type(self):
        matcher = CauseMatcher(ExceptionTypeMatcher(TimeoutError))

        cause = ValueError("bad value")
        exc = RuntimeError("Operation failed")
        exc.__cause__ = cause

        assert not matcher.matches(exc)

    def test_cause_no_match_no_cause(self):
        matcher = CauseMatcher(ExceptionTypeMatcher(TimeoutError))
        exc = RuntimeError("Operation failed")
        assert not matcher.matches(exc)

    def test_cause_nested_matcher(self):
        """CauseMatcher with a ContainsMatcher inside."""
        matcher = CauseMatcher(ContainsMatcher("timeout"))

        cause = Exception("Connection timeout")
        exc = RuntimeError("wrapped")
        exc.__cause__ = cause

        assert matcher.matches(exc)

    def test_cause_recursive_finds_deep_cause(self):
        """Recursive CauseMatcher should find causes deep in the chain."""
        matcher = CauseMatcher(ExceptionTypeMatcher(ValueError), recursive=True)

        # Build a chain: outer -> middle -> root
        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Non-recursive would not find ValueError (immediate cause is TypeError)
        non_recursive = CauseMatcher(ExceptionTypeMatcher(ValueError), recursive=False)
        assert not non_recursive.matches(outer)

        # Recursive should find it
        assert matcher.matches(outer)

    def test_cause_recursive_default_is_true(self):
        """Default recursive=True should traverse the entire cause chain."""
        matcher = CauseMatcher(ExceptionTypeMatcher(ValueError))  # default recursive=True

        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Should match because recursive=True finds ValueError deep in the chain
        assert matcher.matches(outer)

    def test_cause_non_recursive_only_checks_immediate(self):
        """Non-recursive matcher should only check immediate cause."""
        matcher = CauseMatcher(ExceptionTypeMatcher(ValueError), recursive=False)

        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Should not match because immediate cause is TypeError, not ValueError
        assert not matcher.matches(outer)

    def test_cause_recursive_matches_immediate_cause_too(self):
        """Recursive matcher should still match immediate cause."""
        matcher = CauseMatcher(ExceptionTypeMatcher(TypeError), recursive=True)

        root = ValueError("root cause")
        middle = TypeError("middle")
        middle.__cause__ = root
        outer = RuntimeError("outer")
        outer.__cause__ = middle

        # Should match because immediate cause is TypeError
        assert matcher.matches(outer)


class TestAttributeMatcher:
    """Tests for AttributeMatcher."""

    def test_attribute_match(self):
        matcher = AttributeMatcher("errno", 2)

        exc = OSError("No such file")
        exc.errno = 2

        assert matcher.matches(exc)

    def test_attribute_no_match_wrong_value(self):
        matcher = AttributeMatcher("errno", 2)

        exc = OSError("Permission denied")
        exc.errno = 13

        assert not matcher.matches(exc)

    def test_attribute_no_match_missing_attr(self):
        matcher = AttributeMatcher("errno", 2)
        exc = ValueError("no errno")
        assert not matcher.matches(exc)

    def test_attribute_match_string_value(self):
        class CodedError(Exception):
            """Exception with a typed code attribute."""

            code: str

        matcher = AttributeMatcher("code", "E001")

        exc = CodedError("Custom error")
        exc.code = "E001"

        assert matcher.matches(exc)


# =============================================================================
# COMPLEX COMBINATIONS
# =============================================================================


class TestComplexCombinations:
    """Tests for complex matcher combinations."""

    def test_nested_or_and(self):
        """Test OR containing ANDs."""
        matcher = OrMatcher(
            [
                CompositeMatcher(
                    [
                        ExceptionTypeNameMatcher("RuntimeError"),
                        ContainsMatcher("CUDA"),
                    ]
                ),
                CompositeMatcher(
                    [
                        ExceptionTypeNameMatcher("MemoryError"),
                        ContainsMatcher("GPU"),
                    ]
                ),
            ]
        )

        assert matcher.matches(RuntimeError("CUDA out of memory"))
        assert not matcher.matches(ValueError("CUDA error"))

    def test_not_with_or(self):
        """Test NOT with OR inside."""
        matcher = NotMatcher(
            OrMatcher(
                [
                    ContainsMatcher("warning"),
                    ContainsMatcher("info"),
                ]
            )
        )

        assert matcher.matches(ValueError("This is an error"))
        assert not matcher.matches(ValueError("This is a warning"))
        assert not matcher.matches(ValueError("This is info"))
