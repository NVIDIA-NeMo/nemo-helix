# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Exception converter - the core rule engine.

The ExceptionConverter holds a list of (matcher, handler) rules and evaluates
them in order. First match wins.
"""

from typing import Generic, NoReturn

from nhx.common.errors.matchers import (
    ContainsMatcher,
    ExactMatcher,
    ExceptionMatcher,
    ExceptionTypeMatcher,
    RegexMatcher,
)
from nhx.common.errors.types import Handler, TException


class ExceptionConverter(Generic[TException]):
    """
    Converts exceptions by matching them against rules and creating
    domain-specific exception types.

    Rules are evaluated in order; first match wins.

    Example:
        converter = ExceptionConverter[MyError](
            default_handler=lambda e: InternalError(str(e))
        )
        converter.add_regex(r"timeout", lambda e: TimeoutError(str(e)))
        converter.add_contains("permission denied", lambda e: PermissionError(str(e)))

        try:
            risky_operation()
        except Exception as e:
            converter.raise_converted_or_default(e)  # Uses pre-set default_handler

    Example to create a converter from YAML - see RulesLoader for more details:
        converter = RulesLoader.from_yaml("rules.yaml", registry)
        converter.raise_converted_or_original(exception)
    """

    def __init__(
        self,
        rules: list[tuple[ExceptionMatcher, Handler[TException]]] | None = None,
        default_handler: Handler[TException] | None = None,
    ) -> None:
        self._rules: list[tuple[ExceptionMatcher, Handler[TException]]] = list(rules) if rules else []
        self._default_handler = default_handler

    def add_rule(
        self,
        matcher: ExceptionMatcher,
        handler: Handler[TException],
    ) -> "ExceptionConverter[TException]":
        """Add a rule. Returns self for chaining."""
        self._rules.append((matcher, handler))
        return self

    def add_exact(
        self,
        pattern: str,
        handler: Handler[TException],
    ) -> "ExceptionConverter[TException]":
        """Add a rule that matches if str(exception) exactly equals the pattern."""
        return self.add_rule(ExactMatcher(pattern), handler)

    def add_regex(
        self,
        pattern: str,
        handler: Handler[TException],
    ) -> "ExceptionConverter[TException]":
        """Add a rule that matches if the pattern is found in str(exception)."""
        return self.add_rule(RegexMatcher(pattern), handler)

    def add_contains(
        self,
        pattern: str,
        handler: Handler[TException],
    ) -> "ExceptionConverter[TException]":
        """Add a rule that matches if the pattern is a substring of str(exception)."""
        return self.add_rule(ContainsMatcher(pattern), handler)

    def add_type(
        self,
        exception_type: type[Exception],
        handler: Handler[TException],
    ) -> "ExceptionConverter[TException]":
        """Add a rule that matches by exception type."""
        return self.add_rule(ExceptionTypeMatcher(exception_type), handler)

    def convert(self, exception: Exception) -> TException | None:
        """
        Attempt to convert the exception.
        Returns the converted exception if a rule matches, None otherwise.
        """
        for matcher, handler in self._rules:
            if matcher.matches(exception):
                return handler(exception)
        return None

    def raise_converted_or_original(self, exception: Exception) -> NoReturn:
        """
        Convert and raise with proper exception chaining.

        Always raises:
        - If a rule matches: raises the converted exception with __cause__ set to original
        - If no rule matches: raises the original exception
        """
        result = self.convert(exception)
        if result is not None:
            raise result from exception
        raise exception

    def raise_converted_or_default(
        self,
        exception: Exception,
        default_handler: Handler[TException] | None = None,
    ) -> NoReturn:
        """
        Convert and raise with proper exception chaining, using default handler as fallback.

        Always raises:
        - If a rule matches: raises the converted exception with __cause__ set to original
        - If no rule matches: raises default_handler(exception) with __cause__ set to original

        Args:
            exception: The exception to convert.
            default_handler: Optional override for the default handler. If not provided,
                uses the default_handler set in the constructor.

        Raises:
            ValueError: If no default_handler is available (neither passed nor set in constructor).
        """
        result = self.convert(exception)
        if result is not None:
            raise result from exception

        handler = default_handler or self._default_handler
        if handler is None:
            raise ValueError(
                "No default_handler provided. Either pass it to raise_converted_or_default() "
                "or set it in the ExceptionConverter constructor."
            )
        raise handler(exception) from exception

    @property
    def rule_count(self) -> int:
        """Return the number of rules loaded."""
        return len(self._rules)


__all__ = ["ExceptionConverter"]
