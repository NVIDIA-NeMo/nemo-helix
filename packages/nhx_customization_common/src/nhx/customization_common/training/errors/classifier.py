# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify training failures with the rules in ``rules.yaml``.

A rule names a stable cause ``id``, the user-facing error class, and whether a rerun of the same job
could succeed. A rule marked ``symptom`` describes something that usually follows another failure, such
as a collective timeout after a rank died. A failure is classified by its cause when one matched, and by
a symptom only when nothing else did.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nhx.common.errors import RulesLoader
from nhx.common.errors.matchers import ExceptionMatcher
from nhx.customization_common.training.errors.exceptions import (
    EXCEPTION_REGISTRY,
    CustomizerTrainingError,
    ErrorDetails,
    InternalError,
    format_exception_string,
)
from nhx.customization_common.training.errors.parser import fallback_error, find_exceptions

logger = logging.getLogger(__name__)

RULES_PATH = Path(__file__).parent / "rules.yaml"

# Status details a failed task reports, which a rerun reads from the failed attempt.
RESUMABLE = "resumable"
NON_RESUMABLE_REASON = "non_resumable_reason"

_RULE_FIELDS = {"id", "rerunnable", "symptom", "exception", "error_details"}
_MATCHER_FIELDS = {
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
}


@dataclass(frozen=True)
class Rule:
    id: str
    rerunnable: bool
    symptom: bool
    exception: type[CustomizerTrainingError]
    error_details: str | None
    matcher: ExceptionMatcher

    def build(self, original: Exception) -> CustomizerTrainingError:
        # Without a message of its own, the rule keeps the original one, which is specific (our dataset
        # validation errors, for example).
        message = self.error_details or f"{self.exception.user_message} ({original})"
        return self.exception(
            message=message,
            detail=format_exception_string(original),
            cause=self.id,
            rerunnable=self.rerunnable,
        )


def _parse_rule(raw: dict[str, Any]) -> Rule:
    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError(f"Rule needs an 'id': {raw}")
    unknown = set(raw) - _RULE_FIELDS - _MATCHER_FIELDS
    if unknown:
        raise ValueError(f"Rule '{rule_id}' has unknown fields {sorted(unknown)}")
    matchers = set(raw) & _MATCHER_FIELDS
    if len(matchers) != 1:
        raise ValueError(f"Rule '{rule_id}' needs exactly one matcher, has {sorted(matchers)}")
    if not isinstance(raw.get("rerunnable"), bool):
        raise ValueError(f"Rule '{rule_id}' needs 'rerunnable: true' or 'rerunnable: false'")
    exception = EXCEPTION_REGISTRY.get(raw.get("exception", ""))
    if exception is None:
        raise ValueError(f"Rule '{rule_id}' names an unknown exception '{raw.get('exception')}'")
    registry: dict[str, type[Exception]] = dict(EXCEPTION_REGISTRY)
    return Rule(
        id=rule_id,
        rerunnable=raw["rerunnable"],
        symptom=bool(raw.get("symptom", False)),
        exception=exception,
        error_details=raw.get("error_details"),
        matcher=RulesLoader.build_matcher(raw, registry, RulesLoader.DEFAULT_FALLBACK_MODULES),
    )


class FailureClassifier:
    """Match failures against rules, in file order; the first matching rule wins."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        counts = Counter(rule.id for rule in rules)
        duplicates = sorted(rule_id for rule_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate rule ids: {duplicates}")
        self.rules = tuple(rules)

    @classmethod
    def from_yaml(cls, path: Path = RULES_PATH) -> "FailureClassifier":
        config = yaml.safe_load(path.read_text())
        return cls([_parse_rule(raw) for raw in config.get("rules", [])])

    def match(self, exception: Exception) -> Rule | None:
        for rule in self.rules:
            if rule.matcher.matches(exception):
                return rule
        return None

    def _select(self, candidates: Sequence[Exception]) -> CustomizerTrainingError:
        """Classify by the latest candidate that matches a cause, else a symptom, else the latest one.

        ``candidates`` run from the earliest failure to the latest.
        """
        symptom: tuple[Exception, Rule] | None = None
        for candidate in reversed(candidates):
            rule = self.match(candidate)
            if rule is None:
                continue
            if not rule.symptom:
                return rule.build(candidate)
            if symptom is None:
                symptom = (candidate, rule)
        if symptom is not None:
            candidate, rule = symptom
            return rule.build(candidate)
        latest = candidates[-1]
        return InternalError(
            message=f"{InternalError.user_message} ({format_exception_string(latest)})",
            detail=format_exception_string(latest),
        )

    def classify(self, exception: Exception) -> CustomizerTrainingError:
        """Classify an exception raised in this process, including the exceptions it was raised from."""
        if isinstance(exception, CustomizerTrainingError):
            return exception
        chain: list[Exception] = []
        current: BaseException | None = exception
        while isinstance(current, Exception) and current not in chain:
            chain.append(current)
            current = current.__cause__ or current.__context__
        return self._select(list(reversed(chain)))

    def classify_output(self, lines: Iterable[str], returncode: int) -> CustomizerTrainingError:
        """Classify a failed subprocess from the exceptions it printed."""
        lines = list(lines)
        candidates = [parsed.to_exception() for parsed in find_exceptions(lines)]
        if not candidates:
            candidates = [fallback_error(lines, returncode).to_exception()]
        return self._select(candidates)


_classifier: FailureClassifier | None = None


def get_classifier() -> FailureClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FailureClassifier.from_yaml()
        logger.info(f"Loaded {len(_classifier.rules)} training error rules")
    return _classifier


def classify_exception(exception: Exception) -> CustomizerTrainingError:
    """Classify a training failure, falling back to an internal error if the rules cannot be loaded."""
    try:
        return get_classifier().classify(exception)
    except Exception as e:  # noqa: BLE001 - the last-resort reporter must always return an error
        logger.warning(f"Could not classify the training failure: {format_exception_string(e)}")
        return InternalError(
            message=f"{InternalError.user_message} ({format_exception_string(exception)})",
            detail=format_exception_string(exception),
        )


def classify_output(lines: Iterable[str], returncode: int) -> CustomizerTrainingError:
    """Classify a failed subprocess, falling back to an internal error if the rules cannot be loaded."""
    lines = list(lines)
    try:
        return get_classifier().classify_output(lines, returncode)
    except Exception as e:  # noqa: BLE001 - the training failure must not be replaced by this one
        logger.warning(f"Could not classify the training failure: {format_exception_string(e)}")
        parsed = fallback_error(lines, returncode)
        return InternalError(
            message=f"{InternalError.user_message} ({parsed.exception_type}: {parsed.message})",
            detail=f"{parsed.exception_type}: {parsed.message}",
        )


def create_error_details(exception: Exception) -> ErrorDetails:
    """Return the error details a failed task reports."""
    return classify_exception(exception).to_error_details()


def failure_status_details(error: CustomizerTrainingError) -> dict[str, Any]:
    """Return the status details that tell a rerun whether this failure would repeat."""
    if error.rerunnable:
        return {RESUMABLE: True}
    return {RESUMABLE: False, NON_RESUMABLE_REASON: error.message}
