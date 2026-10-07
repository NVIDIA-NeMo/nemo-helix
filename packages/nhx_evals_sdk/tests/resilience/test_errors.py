# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jinja2.exceptions import UndefinedError as JinjaUndefinedError
from nhx_evals_sdk.execution.values import EvaluationError, EvaluationPhase
from nhx_evals_sdk.resilience.errors import (
    find_cause,
    first_failure_cause,
    get_evaluation_error,
    iter_leaf_causes,
    normalize_evaluation_failure,
)


class _DomainError(Exception):
    """Domain-specific error used to exercise typed lookup."""


class TestIterLeafCauses:
    def test_plain_exception_is_its_own_leaf(self):
        exc = ValueError("boom")
        assert list(iter_leaf_causes(exc)) == [exc]

    def test_flat_exception_group_yields_all_children(self):
        a = ValueError("a")
        b = RuntimeError("b")
        c = KeyError("c")
        group = ExceptionGroup("flat", [a, b, c])
        assert list(iter_leaf_causes(group)) == [a, b, c]

    def test_nested_group_yields_dfs_order(self):
        a = ValueError("a")
        b = RuntimeError("b")
        c = KeyError("c")
        d = TypeError("d")
        group = ExceptionGroup(
            "outer",
            [a, ExceptionGroup("inner", [b, c]), d],
        )
        assert list(iter_leaf_causes(group)) == [a, b, c, d]

    def test_empty_group_yields_group_itself(self):
        # Defensive: degenerate groups fall back to yielding the group node.
        group = ExceptionGroup("empty", [ValueError("placeholder")])
        # Replace .exceptions with an empty tuple via a subclass-like stand-in.

        class _EmptyGroup(Exception):
            exceptions = ()

        empty = _EmptyGroup()
        assert list(iter_leaf_causes(empty)) == [empty]
        # And a group with a real child still yields the child.
        assert list(iter_leaf_causes(group)) == list(group.exceptions)


class TestFirstFailureCause:
    def test_plain_exception(self):
        exc = ValueError("boom")
        assert first_failure_cause(exc) is exc

    def test_takes_first_leaf_of_group(self):
        a = ValueError("a")
        b = RuntimeError("b")
        group = ExceptionGroup("g", [a, b])
        assert first_failure_cause(group) is a

    def test_descends_into_first_nested_group(self):
        leaf = ValueError("leaf")
        inner = ExceptionGroup("inner", [leaf, RuntimeError("other")])
        outer = ExceptionGroup("outer", [inner, RuntimeError("sibling")])
        assert first_failure_cause(outer) is leaf


class TestFindCause:
    def test_finds_leading_match(self):
        target = _DomainError("hit")
        group = ExceptionGroup("g", [target, RuntimeError("noise")])
        assert find_cause(group, _DomainError) is target

    def test_finds_non_leading_match(self):
        target = _DomainError("hit")
        group = ExceptionGroup("g", [RuntimeError("a"), target, RuntimeError("b")])
        assert find_cause(group, _DomainError) is target

    def test_finds_deeply_nested_match(self):
        target = _DomainError("deep")
        group = ExceptionGroup(
            "outer",
            [
                RuntimeError("a"),
                ExceptionGroup("mid", [RuntimeError("b"), ExceptionGroup("inner", [target])]),
            ],
        )
        assert find_cause(group, _DomainError) is target

    def test_returns_none_when_no_match(self):
        group = ExceptionGroup("g", [RuntimeError("a"), ValueError("b")])
        assert find_cause(group, _DomainError) is None

    def test_returns_none_for_plain_exception_without_match(self):
        assert find_cause(ValueError("x"), _DomainError) is None

    def test_matches_plain_exception(self):
        exc = _DomainError("plain")
        assert find_cause(exc, _DomainError) is exc


class TestNormalizeEvaluationFailure:
    def test_jinja_undefined_leaf_is_rendered_as_templating_error(self):
        exc = ExceptionGroup("g", [JinjaUndefinedError("'x' is undefined")])
        err = normalize_evaluation_failure(exc)
        assert "templating error" in str(err)
        assert "'x' is undefined" in str(err)

    def test_non_jinja_error_uses_generic_prefix(self):
        err = normalize_evaluation_failure(RuntimeError("boom"))
        assert "with error" in str(err)
        assert "boom" in str(err)


class TestEvaluationErrorOrNormalized:
    def test_evaluation_error_uses_metric_key_context(self):
        target = EvaluationError(
            index=3,
            message="bad template",
            phase=EvaluationPhase.METRIC_SCORING,
            metric_key="exact-match",
        )

        assert target.metric_key == "exact-match"
        assert str(target) == (
            "Evaluation failed during metric scoring for metric 'exact-match' on row 3: bad template"
        )

    def test_returns_leading_evaluation_error(self):
        target = EvaluationError(index=2, message="bad template")
        group = ExceptionGroup("g", [target, RuntimeError("noise")])
        assert get_evaluation_error(group) is target

    def test_returns_non_leading_evaluation_error(self):
        target = EvaluationError(index=5, message="bad template")
        group = ExceptionGroup("g", [RuntimeError("a"), target, RuntimeError("b")])
        assert get_evaluation_error(group) is target

    def test_returns_deeply_nested_evaluation_error(self):
        target = EvaluationError(index=7, message="bad template")
        group = ExceptionGroup(
            "outer",
            [
                RuntimeError("sibling"),
                ExceptionGroup("mid", [RuntimeError("noise"), ExceptionGroup("inner", [target])]),
            ],
        )
        assert get_evaluation_error(group) is target

    def test_returns_plain_evaluation_error(self):
        target = EvaluationError(index=0, message="x")
        assert get_evaluation_error(target) is target

    def test_normalizes_when_no_evaluation_error(self):
        err = get_evaluation_error(RuntimeError("pipeline down"))
        assert isinstance(err, RuntimeError)
        assert not isinstance(err, EvaluationError)
        assert "pipeline down" in str(err)
        assert "with error" in str(err)

    def test_normalizes_jinja_templating_error(self):
        err = get_evaluation_error(ExceptionGroup("g", [JinjaUndefinedError("'x' is undefined")]))
        assert isinstance(err, RuntimeError)
        assert not isinstance(err, EvaluationError)
        assert "templating error" in str(err)
        assert "'x' is undefined" in str(err)
