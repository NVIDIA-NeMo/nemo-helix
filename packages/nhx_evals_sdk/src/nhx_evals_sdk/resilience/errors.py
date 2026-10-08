# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception helpers shared by evaluator resilience/task flows."""

from collections.abc import Iterator
from typing import TypeVar

from jinja2.exceptions import UndefinedError as JinjaUndefinedError
from nhx_evals_sdk.execution.values import EvaluationError

E = TypeVar("E", bound=BaseException)


def iter_leaf_causes(exc: BaseException) -> Iterator[BaseException]:
    """Yield all leaf exceptions from an exception-group tree in left-to-right DFS order.

    Recurses into any exception exposing a non-empty tuple ``.exceptions``
    attribute — matches ``BaseExceptionGroup`` on Python >= 3.11 and the
    ``exceptiongroup`` backport. Group nesting reflects how tasks were grouped
    (e.g. nested ``TaskGroup``s), not causation — leaves are independent,
    concurrent failures.

    Example:
        Nested group — each branch is fully traversed (children before
        siblings) before moving to the next sibling at the parent level::

            group = ExceptionGroup(
                "outer",
                [ValueError("a"),
                 ExceptionGroup("inner", [RuntimeError("b"), KeyError("c")]),
                 TypeError("d")],
            )
            list(iter_leaf_causes(group))
            # [ValueError("a"), RuntimeError("b"), KeyError("c"), TypeError("d")]
    """
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        exceptions = getattr(current, "exceptions", None)
        if not isinstance(exceptions, tuple) or not exceptions:
            yield current
            continue
        if any(not isinstance(child, BaseException) for child in exceptions):
            # Defensive: a non-exception child means this node is effectively a leaf.
            yield current
            continue
        # Push reversed so the leftmost child is popped first (preserves DFS left→right).
        stack.extend(reversed(exceptions))


def first_failure_cause(exc: BaseException) -> BaseException:
    """Return the first leaf failure from an exception/exception-group tree.

    Use case: surface *a* concrete failure in a log line or wrapped message,
    without caring which sibling comes first.
    """
    return next(iter_leaf_causes(exc), exc)


def find_cause(exc: BaseException, cls: type[E]) -> E | None:
    """Return the first leaf of type ``cls`` anywhere in the exception tree, or ``None``.

    Unlike :func:`first_failure_cause`, which returns only the first leaf in
    traversal order, ``find_cause`` walks every leaf of any ``ExceptionGroup``
    via :func:`iter_leaf_causes`. That means a matching exception is found even
    when it is not the leading sibling or sits under a non-leading branch. The
    return type is narrowed to ``cls | None`` via ``TypeVar``, avoiding a
    duplicate ``isinstance`` check at the call site. Common with
    ``asyncio.TaskGroup``, where sibling order depends on task scheduling and
    is not deterministic.

    Example:
        Nested group — the ``EvaluationError`` is buried under a non-leading
        branch, so ``first_failure_cause`` misses it::

            group = ExceptionGroup("outer", [
                RuntimeError("sibling"),
                ExceptionGroup("inner", [
                    RuntimeError("noise"),
                    EvaluationError(index=5, message="template error"),
                ]),
            ])
            first_failure_cause(group)          # RuntimeError("sibling")
            find_cause(group, EvaluationError)  # EvaluationError(index=5, ...)
    """
    for leaf in iter_leaf_causes(exc):
        if isinstance(leaf, cls):
            return leaf
    return None


def normalize_evaluation_failure(
    exc: BaseException,
    *,
    prefix: str = "Metric evaluation has failed",
) -> RuntimeError:
    """Convert queue/task execution failures into the public evaluator error shape."""
    root = first_failure_cause(exc)
    if isinstance(root, JinjaUndefinedError):
        return RuntimeError(f"{prefix} due to templating error: {str(root)}")
    if isinstance(exc, JinjaUndefinedError):
        return RuntimeError(f"{prefix} due to templating error: {str(exc)}")
    return RuntimeError(f"{prefix} with error: {str(root) or root.__class__.__name__}")


def get_evaluation_error(exc: BaseException) -> EvaluationError | RuntimeError:
    """Classify a pipeline failure into an ``EvaluationError`` or normalized ``RuntimeError``.

    Returns the first ``EvaluationError`` found anywhere in the exception tree
    (via :func:`find_cause`) if present; otherwise falls back to
    :func:`normalize_evaluation_failure`. Callers decide how to re-raise.
    """
    evaluation_error = find_cause(exc, EvaluationError)
    if evaluation_error is not None:
        return evaluation_error
    return normalize_evaluation_failure(exc)
