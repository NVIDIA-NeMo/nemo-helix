# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Number-check metric runtime implementation."""

import math
import re
import sys
from fractions import Fraction

from nhx_evals_sdk.metrics.protocol import (
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nhx_evals_sdk.metrics.template_rendering import (
    TemplateSample,
    build_template_context,
    render_template_or_raise,
    template_metric_repr,
)
from nhx_evals_sdk.values.metrics import NumberCheck, NumberCheckOperation
from nhx_evals_sdk.values.protocol import OUTPUT_DETAIL, REASON_DETAIL

__all__ = ["NumberCheckMetric", "NumberCheckOperation"]

_OUT_OF_FLOAT_RANGE_REASON = "operand_out_of_float_range"


def _parse_number_answer(answer: str) -> int | float:
    """Parse the last numeric value from text; return ``NaN`` when parsing fails.

    An integer with more digits than ``sys.get_int_max_str_digits()`` allows becomes signed infinity rather
    than ``NaN``, so it still compares correctly against in-range values instead of marking the row as
    unparseable. ``int()`` counts leading zeros toward that limit, so they are stripped before the length
    check to keep a zero-padded value exact. The pattern matches digits from every script, so the zeros
    stripped are all matched characters whose value is 0, not only ASCII ``"0"``.
    """
    numbers = re.findall(r"[+-]?[\.\d]*\d+", answer)
    if not numbers:
        return float("nan")

    last_number = numbers[-1]
    decimal = last_number.count(".")
    if decimal == 1:
        return float(last_number)
    if decimal == 0:
        try:
            return int(last_number)
        except ValueError:
            sign = last_number[0] if last_number[0] in "+-" else ""
            digits = last_number[len(sign) :]
            zero_chars = "".join(c for c in set(digits) if int(c) == 0)
            unpadded_digits = digits.lstrip(zero_chars) or "0"
            if len(unpadded_digits) <= sys.get_int_max_str_digits():
                return int(sign + unpadded_digits)
            return -math.inf if sign == "-" else math.inf
    return float("nan")


def _parse_operand(value: object) -> int | float:
    """Parse a rendered template value as a number.

    A bare template can render a native integer that ``str()`` refuses under the same digit limit as
    ``int()``; such a value becomes signed infinity.
    """
    try:
        text = str(value)
    except ValueError:
        if isinstance(value, int):
            return math.inf if value > 0 else -math.inf
        raise
    return _parse_number_answer(text)


def _absolute_difference(left: int | float, right: int | float) -> int | float | Fraction:
    """Return ``abs(left - right)``, including when an integer operand is too large to convert to float.

    Subtracting a float and such an integer raises ``OverflowError``, so the difference is computed with
    ``Fraction`` instead. ``Fraction`` cannot represent infinity, and ``math.isinf`` raises on such an integer.
    """
    try:
        return abs(left - right)
    except OverflowError:
        if any(isinstance(value, float) and math.isinf(value) for value in (left, right)):
            return math.inf
        return abs(Fraction(left) - Fraction(right))


def _out_of_range_diagnostics(output_name: str, left: int | float, right: int | float) -> list[MetricDiagnostic]:
    """Describe each operand beyond float range without including its digits."""
    diagnostics: list[MetricDiagnostic] = []
    for side, value in (("left", left), ("right", right)):
        if isinstance(value, float) and math.isinf(value):
            representation, compared_as = "infinity", "signed infinity"
        elif isinstance(value, int) and abs(value) > sys.float_info.max:
            representation, compared_as = "exact_integer", "an exact integer"
        else:
            continue
        diagnostics.append(
            MetricDiagnostic(
                message=f"The {side} operand is outside float range and was compared as {compared_as}.",
                details={
                    OUTPUT_DETAIL: output_name,
                    REASON_DETAIL: _OUT_OF_FLOAT_RANGE_REASON,
                    "operand": side,
                    "representation": representation,
                },
            )
        )
    return diagnostics


class NumberCheckMetric(NumberCheck):
    """Numeric-comparison metric with template-driven operands."""

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(self.type.value)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        item = input.row.data
        sample: TemplateSample = input.candidate
        context = build_template_context(item, sample)
        metric_repr = template_metric_repr(self)
        left_value = render_template_or_raise(
            template_name="left_template",
            template=self.left_template,
            context=context,
            item=item,
            sample=sample,
            metric_repr=metric_repr,
        )
        right_value = render_template_or_raise(
            template_name="right_template",
            template=self.right_template,
            context=context,
            item=item,
            sample=sample,
            metric_repr=metric_repr,
        )

        left_number = _parse_operand(left_value)
        right_number = _parse_operand(right_value)
        diagnostics = _out_of_range_diagnostics(self.type.value, left_number, right_number)
        # Preserve the legacy behavior: if either side fails to parse as a
        # number, return NaN instead of raising.
        if isinstance(left_number, float) and math.isnan(left_number):
            return MetricResult(
                outputs=[MetricOutput(name=self.type.value, value=left_number)], diagnostics=diagnostics
            )
        if isinstance(right_number, float) and math.isnan(right_number):
            return MetricResult(
                outputs=[MetricOutput(name=self.type.value, value=right_number)], diagnostics=diagnostics
            )

        # Perform the requested numeric comparison on the parsed operands.
        if self.operation in ["equals", "=="]:
            score = left_number == right_number
        elif self.operation in ["!=", "<>", "not equals"]:
            score = left_number != right_number
        elif self.operation in [">=", "gte", "greater than or equal"]:
            score = left_number >= right_number
        elif self.operation in [">", "gt", "greater than"]:
            score = left_number > right_number
        elif self.operation in ["<=", "lte", "less than or equal"]:
            score = left_number <= right_number
        elif self.operation in ["<", "lt", "less than"]:
            score = left_number < right_number
        elif self.operation == "absolute difference":
            if self.epsilon is None:
                raise ValueError("epsilon value is required with operation absolute difference")
            score = _absolute_difference(left_number, right_number) <= self.epsilon
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")

        return MetricResult(
            outputs=[MetricOutput(name=self.type.value, value=1.0 if score else 0.0)], diagnostics=diagnostics
        )
