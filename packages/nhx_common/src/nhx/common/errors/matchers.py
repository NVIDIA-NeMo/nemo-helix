# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Exception matchers for the error mapping.

Matchers answer one question: "Does this exception match my criteria?"

Each matcher implements a different matching strategy:
- ExactMatcher: message equals a specific string
- RegexMatcher: message matches a regex pattern
- ContainsMatcher: message contains a substring
- ExceptionTypeMatcher: exception is an instance of a specific type
- CompositeMatcher (and/or/not): combine multiple matchers

Usage: Matchers are typically created from YAML rules by RulesLoader,
then used by ExceptionConverter to find the right error conversion for an exception.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ExceptionMatcher(ABC):
    """Abstract base class for exception matching strategies."""

    @abstractmethod
    def matches(self, exception: Exception) -> bool:
        """Return True if the exception matches this rule."""
        ...


# =============================================================================
# BASIC MATCHERS
# =============================================================================


@dataclass
class ExactMatcher(ExceptionMatcher):
    """Match if str(exception) exactly equals the pattern."""

    pattern: str

    def matches(self, exception: Exception) -> bool:
        return str(exception) == self.pattern


@dataclass
class RegexMatcher(ExceptionMatcher):
    """Match if the regex pattern is found in str(exception)."""

    pattern: str
    _compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        try:
            self._compiled = re.compile(self.pattern)
        except re.error as e:
            raise ValueError(
                f"RegexMatcher failed to compile pattern '{self.pattern}': {e}. "
                f"Check your regex syntax in the error mapping rules."
            ) from e

    def matches(self, exception: Exception) -> bool:
        return self._compiled.search(str(exception)) is not None


@dataclass
class ContainsMatcher(ExceptionMatcher):
    """Match if the pattern is a substring of str(exception)."""

    pattern: str

    def matches(self, exception: Exception) -> bool:
        return self.pattern in str(exception)


@dataclass
class StartsWithMatcher(ExceptionMatcher):
    """Match if str(exception) starts with the pattern."""

    pattern: str

    def matches(self, exception: Exception) -> bool:
        return str(exception).startswith(self.pattern)


@dataclass
class EndsWithMatcher(ExceptionMatcher):
    """Match if str(exception) ends with the pattern."""

    pattern: str

    def matches(self, exception: Exception) -> bool:
        return str(exception).endswith(self.pattern)


# =============================================================================
# TYPE-BASED MATCHERS
# =============================================================================


@dataclass
class ExceptionTypeMatcher(ExceptionMatcher):
    """Match by exception type (isinstance check)."""

    exception_type: type[Exception]

    def matches(self, exception: Exception) -> bool:
        return isinstance(exception, self.exception_type)


@dataclass
class ExceptionTypeNameMatcher(ExceptionMatcher):
    """
    Match by exception class name (string), without needing the actual class.

    Useful when you can't or don't want to import the exception class.
    Example: Match "OutOfMemoryError" without importing torch.cuda.OutOfMemoryError
    """

    type_name: str

    def matches(self, exception: Exception) -> bool:
        return type(exception).__name__ == self.type_name


# =============================================================================
# KEYWORD MATCHERS
# =============================================================================


@dataclass
class AllKeywordsMatcher(ExceptionMatcher):
    """Match if ALL keywords are present in str(exception)."""

    keywords: list[str]

    def __post_init__(self):
        if not self.keywords:
            raise ValueError(
                "AllKeywordsMatcher requires at least one keyword. An empty list would match all exceptions."
            )

    def matches(self, exception: Exception) -> bool:
        msg = str(exception)
        return all(kw in msg for kw in self.keywords)


@dataclass
class AnyKeywordMatcher(ExceptionMatcher):
    """Match if ANY keyword is present in str(exception)."""

    keywords: list[str]

    def __post_init__(self):
        if not self.keywords:
            raise ValueError("AnyKeywordMatcher requires at least one keyword. An empty list would never match.")

    def matches(self, exception: Exception) -> bool:
        msg = str(exception)
        return any(kw in msg for kw in self.keywords)


# =============================================================================
# COMPOSITE MATCHERS
# =============================================================================


@dataclass
class CompositeMatcher(ExceptionMatcher):
    """Combine multiple matchers with AND logic (all must match)."""

    matchers: list[ExceptionMatcher]

    def matches(self, exception: Exception) -> bool:
        return all(m.matches(exception) for m in self.matchers)


@dataclass
class OrMatcher(ExceptionMatcher):
    """Match if ANY of the matchers match (OR logic)."""

    matchers: list[ExceptionMatcher]

    def matches(self, exception: Exception) -> bool:
        return any(m.matches(exception) for m in self.matchers)


@dataclass
class NotMatcher(ExceptionMatcher):
    """Match if the inner matcher does NOT match (negation)."""

    matcher: ExceptionMatcher

    def matches(self, exception: Exception) -> bool:
        return not self.matcher.matches(exception)


# =============================================================================
# SPECIAL MATCHERS
# =============================================================================


@dataclass
class CauseMatcher(ExceptionMatcher):
    """
    Match based on the exception's __cause__ (chained exception).

    Useful when exceptions are wrapped and you want to match the original cause.

    Args:
        cause_matcher: The matcher to apply to cause exceptions.
        recursive: If True (default), search the entire cause chain.
                   If False, only check the immediate __cause__.

    Example:
        # Match anywhere in cause chain (default)
        CauseMatcher(ExceptionTypeMatcher(TimeoutError))

        # Match only immediate cause
        CauseMatcher(ExceptionTypeMatcher(TimeoutError), recursive=False)
    """

    cause_matcher: ExceptionMatcher
    recursive: bool = True

    def matches(self, exception: Exception) -> bool:
        cause = exception.__cause__
        while cause is not None:
            if self.cause_matcher.matches(cause):
                return True
            if not self.recursive:
                break
            cause = cause.__cause__
        return False


@dataclass
class AttributeMatcher(ExceptionMatcher):
    """
    Match based on an exception attribute value.

    Example: AttributeMatcher("errno", 2) matches exceptions where exc.errno == 2
    """

    attribute: str
    value: Any

    def matches(self, exception: Exception) -> bool:
        return getattr(exception, self.attribute, None) == self.value


def normalize_whitespace(s: str) -> str:
    """
    Normalize whitespace for stable matching.

    Strips leading/trailing whitespace and collapses multiple whitespace
    characters (including newlines) into single spaces.

    Useful for matching error messages that may have inconsistent formatting.
    """
    return " ".join(str(s).strip().split())


__all__ = [
    "ExceptionMatcher",
    # Basic
    "ExactMatcher",
    "RegexMatcher",
    "ContainsMatcher",
    "StartsWithMatcher",
    "EndsWithMatcher",
    # Type-based
    "ExceptionTypeMatcher",
    "ExceptionTypeNameMatcher",
    # Keyword
    "AllKeywordsMatcher",
    "AnyKeywordMatcher",
    # Composite
    "CompositeMatcher",
    "OrMatcher",
    "NotMatcher",
    # Special
    "CauseMatcher",
    "AttributeMatcher",
    # Utilities
    "normalize_whitespace",
]
