# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rules loader - loads exception mapping rules from YAML/JSON configuration.
"""

import builtins
import json
import logging
import subprocess
import types
from pathlib import Path
from typing import Any, Callable, ClassVar

import yaml
from nhx.common.errors.converter import ExceptionConverter
from nhx.common.errors.matchers import (
    AllKeywordsMatcher,
    AnyKeywordMatcher,
    AttributeMatcher,
    CauseMatcher,
    CompositeMatcher,
    ContainsMatcher,
    EndsWithMatcher,
    ExactMatcher,
    ExceptionMatcher,
    ExceptionTypeMatcher,
    ExceptionTypeNameMatcher,
    NotMatcher,
    OrMatcher,
    RegexMatcher,
    StartsWithMatcher,
)
from nhx.common.errors.types import DefaultExceptionHandler, ExceptionRegistry, Handler, HandlerRegistry

logger = logging.getLogger(__name__)


class RulesLoader:
    """
    Load exception mapping rules from YAML or JSON configuration.

    Rule Structure
    --------------
    Each rule has: ONE matcher field + exception fields.

    - Matcher field: Defines the condition - when should this rule trigger?
    - Exception fields: Define what to do when the rule matches.

    ```yaml
    - <matcher_field>: <value>       # MATCHER: when to trigger (pick ONE)
      exception: MyException         # EXCEPTION: class to convert to (REQUIRED)
      error_details: "User message"  # EXCEPTION: optional message for default handler
      handler: my_handler            # EXCEPTION: optional custom handler
    ```

    Exception Fields (apply to ALL rules)
    ----------------------------------
    exception     : (REQUIRED) Exception class name from exception_registry
    error_details : (optional) User-friendly message passed to default handler
    handler       : (optional) Custom handler name from handler_registry

    Note: `error_details` and `handler` are MUTUALLY EXCLUSIVE. Specifying both
    will raise a ValueError. Use `error_details` for simple message overrides,
    or `handler` for custom conversion logic with a user friendly message defined.

    Matcher Fields (pick ONE per rule)
    ----------------------------------
    Message matching:
        exact        : str       - Message equals this string exactly
        regex        : str       - Message matches this regex pattern
        contains     : str       - Message contains this substring
        starts_with  : str       - Message starts with this prefix
        ends_with    : str       - Message ends with this suffix
        all_keywords : [str]     - Message contains ALL of these keywords
        any_keywords : [str]     - Message contains ANY of these keywords

    Type matching:
        type         : str       - Exception is instance of this type (from registry or builtins)
        type_name    : str       - Exception class name equals this string

    Composite (nest other matchers):
        and          : [matcher] - ALL sub-matchers must match
        or           : [matcher] - ANY sub-matcher must match
        not          : matcher   - Sub-matcher must NOT match

    Special:
        cause        : matcher or {matcher, recursive} - Match the exception's __cause__
                       Use recursive: true to search entire cause chain
        attribute    : {name, value} - Exception has attribute with specific value

    Examples
    --------
    ```yaml
    rules:
      # Simple exact match
      - exact: "Connection refused"
        exception: NetworkError
        error_details: "Could not connect to the server"

      # Regex with custom handler
      - regex: "^Timeout after \\d+ seconds$"
        exception: TimeoutError
        handler: timeout_with_duration

      # Match by exception type
      - type: FileNotFoundError
        exception: DatasetNotFoundError

      # Match if message contains any of these
      - any_keywords: ["CUDA", "GPU", "device"]
        exception: CudaError

      # Composite: match RuntimeError containing "distributed"
      - and:
          - type_name: RuntimeError
          - contains: "distributed"
        exception: DistributedError

      # Match chained exception
      - cause:
          type: TimeoutError
          recursive: true
        exception: TrainingTimeoutError
        error_details: "Training timed out. Please try again."
    ```
    """

    # Default modules to search for exception types (builtins like ValueError, etc.)
    DEFAULT_FALLBACK_MODULES: ClassVar[list[types.ModuleType]] = [builtins, subprocess]

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        exception_registry: ExceptionRegistry,
        *,
        handler_registry: HandlerRegistry | None = None,
        default_handler: DefaultExceptionHandler | None = None,
        fallback_exception: type[Exception] | None = None,
        fallback_modules: list[types.ModuleType] | None = None,
    ) -> ExceptionConverter:
        """
        Load converter rules from a YAML file.

        Args:
            yaml_path: Path to the YAML rules file.
            exception_registry: Dict mapping exception class names (strings) to actual classes.
            handler_registry: Optional dict mapping handler names to handler callables.
            default_handler: Handler for creating exceptions. Used both for matched rules
                without a custom handler, and for the fallback when no rule matches.
                Signature: (exc_class, original_exception, error_details) -> Exception
            fallback_exception: Exception class to use when no rule matches. If provided,
                raise_converted_or_default() will use default_handler(fallback_exception, original, None).
            fallback_modules: Optional list of modules to search for exception types not in
                the registry. Defaults to [builtins, subprocess]. Example: [builtins, torch.cuda, ssl]

        Returns:
            Configured ExceptionConverter ready to use.

        Raises:
            FileNotFoundError: If yaml_path doesn't exist.
            ValueError: If a rule references an unknown exception class or handler.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {yaml_path}")

        with open(path) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in rules file '{yaml_path}': {e}") from e

        return cls._build_converter(
            config, exception_registry, handler_registry, default_handler, fallback_exception, fallback_modules
        )

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        exception_registry: ExceptionRegistry,
        *,
        handler_registry: HandlerRegistry | None = None,
        default_handler: DefaultExceptionHandler | None = None,
        fallback_exception: type[Exception] | None = None,
        fallback_modules: list[types.ModuleType] | None = None,
    ) -> ExceptionConverter:
        """
        Load converter rules from a JSON file.

        Args:
            json_path: Path to the JSON rules file.
            exception_registry: Dict mapping exception class names to actual classes.
            handler_registry: Optional dict mapping handler names to handler callables.
            default_handler: Handler for creating exceptions. Used both for matched rules
                without a custom handler, and for the fallback when no rule matches.
                Signature: (exc_class, original_exception, error_details) -> Exception
            fallback_exception: Exception class to use when no rule matches. If provided,
                raise_converted_or_default() will use default_handler(fallback_exception, original, None).
            fallback_modules: Optional list of modules to search for exception types not in
                the registry. Defaults to [builtins, subprocess]. Example: [builtins, torch.cuda, ssl]

        Returns:
            Configured ExceptionConverter ready to use.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {json_path}")

        with open(path) as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in rules file '{json_path}': {e}") from e

        return cls._build_converter(
            config, exception_registry, handler_registry, default_handler, fallback_exception, fallback_modules
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        exception_registry: ExceptionRegistry,
        *,
        handler_registry: HandlerRegistry | None = None,
        default_handler: DefaultExceptionHandler | None = None,
        fallback_exception: type[Exception] | None = None,
        fallback_modules: list[types.ModuleType] | None = None,
    ) -> ExceptionConverter:
        """
        Load converter rules from a dictionary (useful for testing or inline config).

        Args:
            config: Dict with "rules" key containing list of rule definitions.
            exception_registry: Dict mapping exception class names to actual classes.
            handler_registry: Optional dict mapping handler names to handler callables.
            default_handler: Handler for creating exceptions. Used both for matched rules
                without a custom handler, and for the fallback when no rule matches.
                Signature: (exc_class, original_exception, error_details) -> Exception
            fallback_exception: Exception class to use when no rule matches. If provided,
                raise_converted_or_default() will use default_handler(fallback_exception, original, None).
            fallback_modules: Optional list of modules to search for exception types not in
                the registry. Defaults to [builtins, subprocess]. Example: [builtins, torch.cuda, ssl]

        Returns:
            Configured ExceptionConverter ready to use.
        """
        return cls._build_converter(
            config, exception_registry, handler_registry, default_handler, fallback_exception, fallback_modules
        )

    @classmethod
    def _build_converter(
        cls,
        config: dict[str, Any],
        exception_registry: ExceptionRegistry,
        handler_registry: HandlerRegistry | None,
        default_handler: DefaultExceptionHandler | None,
        fallback_exception: type[Exception] | None,
        fallback_modules: list[types.ModuleType] | None,
    ) -> ExceptionConverter:
        """Internal: build converter from parsed config."""
        if handler_registry is None:
            handler_registry = {}

        if default_handler is None:
            default_handler = cls._default_exception_handler

        if fallback_modules is None:
            fallback_modules = cls.DEFAULT_FALLBACK_MODULES

        # Create the no-match handler if fallback_exception is provided
        no_match_handler: Handler | None = None
        if fallback_exception is not None:
            # Capture default_handler and fallback_exception in closure
            def no_match_handler(
                original: Exception,
                _exc_class: type[Exception] = fallback_exception,
                _handler: Callable = default_handler,
            ) -> Exception:
                return _handler(_exc_class, original, None)

        converter: ExceptionConverter = ExceptionConverter(default_handler=no_match_handler)

        for rule in config.get("rules", []):
            exc_name = rule.get("exception")
            if not exc_name:
                raise ValueError(f"Rule missing 'exception' field: {rule}")

            if exc_name not in exception_registry:
                raise ValueError(f"Unknown exception class '{exc_name}'. Available: {list(exception_registry.keys())}")

            exc_class = exception_registry[exc_name]

            # Determine which handler to use for this rule
            handler_name = rule.get("handler")
            error_details = rule.get("error_details")

            # Validate: handler and error_details are mutually exclusive
            if handler_name and error_details:
                raise ValueError(
                    f"Rule cannot have both 'handler' and 'error_details'. "
                    f"Use 'handler' for custom logic, or 'error_details' for simple message override. "
                    f"Rule: {rule}"
                )

            if handler_name:
                # Use named handler from registry
                if handler_name not in handler_registry:
                    raise ValueError(f"Unknown handler '{handler_name}'. Available: {list(handler_registry.keys())}")
                handler = handler_registry[handler_name]
            else:
                # Use default handler with error_details
                handler = cls._make_default_handler(exc_class, error_details, default_handler)

            # Build the matcher from the rule
            matcher = cls._build_matcher(rule, exception_registry, fallback_modules)
            converter.add_rule(matcher, handler)

        if converter.rule_count == 0:
            logger.warning("No rules loaded into ExceptionConverter. All exceptions will pass through unconverted.")

        return converter

    @classmethod
    def _build_matcher(
        cls,
        rule: dict[str, Any],
        exception_registry: ExceptionRegistry,
        fallback_modules: list[types.ModuleType],
    ) -> ExceptionMatcher:
        """
        Build an ExceptionMatcher from a rule dictionary.

        Supports nested/recursive matchers for composite types (or, and, not, cause).
        """
        # Basic matchers
        if "exact" in rule:
            return ExactMatcher(rule["exact"])

        if "regex" in rule:
            return RegexMatcher(rule["regex"])

        if "contains" in rule:
            return ContainsMatcher(rule["contains"])

        if "starts_with" in rule:
            return StartsWithMatcher(rule["starts_with"])

        if "ends_with" in rule:
            return EndsWithMatcher(rule["ends_with"])

        # Type-based matchers
        if "type" in rule:
            type_name = rule["type"]
            type_class = cls._resolve_exception_type(type_name, exception_registry, fallback_modules)
            return ExceptionTypeMatcher(type_class)

        if "type_name" in rule:
            return ExceptionTypeNameMatcher(rule["type_name"])

        # Keyword matchers
        if "all_keywords" in rule:
            keywords = rule["all_keywords"]
            if not isinstance(keywords, list):
                raise ValueError(f"'all_keywords' must be a list: {rule}")
            return AllKeywordsMatcher(keywords)

        if "any_keywords" in rule:
            keywords = rule["any_keywords"]
            if not isinstance(keywords, list):
                raise ValueError(f"'any_keywords' must be a list: {rule}")
            return AnyKeywordMatcher(keywords)

        # Composite matchers (recursive)
        if "or" in rule:
            sub_rules = rule["or"]
            if not isinstance(sub_rules, list):
                raise ValueError(f"'or' must be a list of matchers: {rule}")
            sub_matchers = [cls._build_matcher(r, exception_registry, fallback_modules) for r in sub_rules]
            return OrMatcher(sub_matchers)

        if "and" in rule:
            sub_rules = rule["and"]
            if not isinstance(sub_rules, list):
                raise ValueError(f"'and' must be a list of matchers: {rule}")
            sub_matchers = [cls._build_matcher(r, exception_registry, fallback_modules) for r in sub_rules]
            return CompositeMatcher(sub_matchers)

        if "not" in rule:
            sub_rule = rule["not"]
            if not isinstance(sub_rule, dict):
                raise ValueError(f"'not' must be a matcher dict: {rule}")
            sub_matcher = cls._build_matcher(sub_rule, exception_registry, fallback_modules)
            return NotMatcher(sub_matcher)

        # Special matchers
        if "cause" in rule:
            cause_config = rule["cause"]
            if not isinstance(cause_config, dict):
                raise ValueError(f"'cause' must be a dict: {rule}")

            # Check for recursive option
            recursive = cause_config.get("recursive", False)

            # If 'matcher' key exists, use it; otherwise treat the whole dict as the matcher
            if "matcher" in cause_config:
                sub_rule = cause_config["matcher"]
                if not isinstance(sub_rule, dict):
                    raise ValueError(f"'cause.matcher' must be a matcher dict: {rule}")
            else:
                # Filter out 'recursive' to get the matcher fields
                sub_rule = {k: v for k, v in cause_config.items() if k != "recursive"}

            cause_matcher = cls._build_matcher(sub_rule, exception_registry, fallback_modules)
            return CauseMatcher(cause_matcher, recursive=recursive)

        if "attribute" in rule:
            attr_config = rule["attribute"]
            if not isinstance(attr_config, dict):
                raise ValueError(f"'attribute' must be a dict with 'name' and 'value': {rule}")
            if "name" not in attr_config or "value" not in attr_config:
                raise ValueError(f"'attribute' requires 'name' and 'value' fields: {rule}")
            return AttributeMatcher(attr_config["name"], attr_config["value"])

        # No recognized matcher key found
        matcher_keys = [
            "exact",
            "regex",
            "contains",
            "starts_with",
            "ends_with",
            "type",
            "type_name",
            "all_keywords",
            "any_keywords",
            "or",
            "and",
            "not",
            "cause",
            "attribute",
        ]
        raise ValueError(f"Rule must have one of {matcher_keys}: {rule}")

    @staticmethod
    def _make_default_handler(
        exc_class: type[Exception],
        error_details: str | None,
        default_handler: DefaultExceptionHandler,
    ) -> Handler:
        """Create a handler function using the default handler with captured exc_class and error_details."""

        def handler(
            original: Exception,
            _exc_class: type[Exception] = exc_class,
            _error_details: str | None = error_details,
            _handler: Callable = default_handler,
        ) -> Exception:
            return _handler(_exc_class, original, _error_details)

        return handler

    @staticmethod
    def _default_exception_handler(
        exc_class: type[Exception],
        original: Exception,
        error_details: str | None,
    ) -> Exception:
        """
        Built-in default handler for creating exceptions.

        Tries to call exc_class with (message, detail) positional arguments.
        Falls back to just (message,) if that fails.

        If error_details is None, empty, or whitespace-only, uses the original
        exception message instead.
        """
        has_details = error_details is not None and error_details.strip()
        message = error_details if has_details else str(original)
        detail = str(original)

        try:
            return exc_class(message, detail)
        except TypeError:
            return exc_class(message)

    @staticmethod
    def _resolve_exception_type(
        type_name: str,
        exception_registry: ExceptionRegistry,
        fallback_modules: list[types.ModuleType],
    ) -> type[Exception]:
        """
        Resolve an exception type name to an actual class.

        Searches in order:
        1. exception_registry (user-provided mapping)
        2. fallback_modules (defaults to [builtins, subprocess])
        """
        if type_name in exception_registry:
            return exception_registry[type_name]

        for module in fallback_modules:
            if hasattr(module, type_name):
                resolved_cls = getattr(module, type_name)
                if isinstance(resolved_cls, type) and issubclass(resolved_cls, BaseException):
                    return resolved_cls

        module_names = [m.__name__ for m in fallback_modules]
        raise ValueError(
            f"Cannot resolve exception type '{type_name}'. "
            f"Add it to the exception_registry or ensure it exists in fallback_modules: {module_names}"
        )


__all__ = ["RulesLoader"]
