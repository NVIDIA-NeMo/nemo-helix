# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from collections.abc import Iterator
from typing import Literal

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.number_check import NumberCheckMetric, _parse_number_answer
from nhx_evals_sdk.values import MetricResult
from nhx_evals_sdk.values.protocol import OUTPUT_DETAIL, REASON_DETAIL
from pydantic_core import ValidationError

NumberOperation = Literal[
    "equals",
    "==",
    "!=",
    "<>",
    "not equals",
    ">=",
    "gte",
    "greater than or equal",
    ">",
    "gt",
    "greater than",
    "<=",
    "lte",
    "less than or equal",
    "<",
    "lt",
    "less than",
    "absolute difference",
]

EQUALS_OPS: tuple[NumberOperation, ...] = ("equals", "==")
NOT_EQUALS_OPS: tuple[NumberOperation, ...] = ("!=", "<>", "not equals")
GTE_OPS: tuple[NumberOperation, ...] = (">=", "gte", "greater than or equal")
GT_OPS: tuple[NumberOperation, ...] = (">", "gt", "greater than")
LTE_OPS: tuple[NumberOperation, ...] = ("<=", "lte", "less than or equal")
LT_OPS: tuple[NumberOperation, ...] = ("<", "lt", "less than")

INT_DIGIT_LIMIT = 4300
BEYOND_FLOAT_RANGE = "9" * 400
OVER_DIGIT_LIMIT = "9" * 5000


class TestParseNumberAnswer:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("value is 42", 42),
            ("value is -3.5", -3.5),
        ],
    )
    def test_parse_number(self, answer, expected):
        assert _parse_number_answer(answer) == expected

    def test_parse_number_nan_when_no_number(self):
        assert math.isnan(_parse_number_answer("no number"))

    def test_parse_number_nan_when_multiple_decimals(self):
        assert math.isnan(_parse_number_answer("1.2.3"))

    @pytest.mark.parametrize(
        ("answer", "expected", "desc"),
        [
            ("1234", 1234, "integer"),
            ("my reasoning 123 and final answer is 1.2", 1.2, "float"),
            ("-3", -3, "negative integer"),
            ("-3.23 is the answer", -3.23, "negative float"),
            ("The values are 10.5, -3.14, and .75.", 0.75, "last decimal and ignores period"),
        ],
    )
    def test_parse_number_answer(self, answer: str, expected: int | float, desc: str):
        assert _parse_number_answer(answer) == expected, desc

    @pytest.mark.parametrize(
        ("answer", "desc"),
        [
            (".", "period is not matched as a number"),
            ("-", "hyphen is not matched as a number"),
            ("127.0.0.1", "IP is not a valid number"),
        ],
    )
    def test_parse_number_answer_nan(self, answer: str, desc: str):
        assert math.isnan(_parse_number_answer(answer)), desc


class TestNumberCheckMetric:
    @pytest.mark.parametrize(
        ("desc", "params"),
        [
            ("operation", {"operation": "not-supported", "left_template": "", "right_template": ""}),
            ("right_template", {"operation": "==", "left_template": ""}),
            (
                "epsilon value can only be used with absolute difference",
                {"operation": ">", "left_template": "1", "right_template": "2", "epsilon": 5},
            ),
            (
                "epsilon value is required",
                {"operation": "absolute difference", "left_template": "1", "right_template": "2"},
            ),
        ],
    )
    def test_validation_errors(self, desc: str, params: dict):
        with pytest.raises(ValidationError, match=desc):
            NumberCheckMetric.model_validate(params)

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = NumberCheckMetric(
            operation="equals",
            left_template="{{item.reference_answer}}",
            right_template="{{output_text}}",
        )
        result = await compute_scores(metric, {"reference_answer": "1"}, {"output_text": "1"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    def test_absolute_difference_requires_epsilon(self):
        with pytest.raises(ValueError, match="epsilon value is required with operation absolute difference"):
            NumberCheckMetric(
                operation="absolute difference",
                left_template="{{item.left}}",
                right_template="{{item.right}}",
            )

    def test_epsilon_only_allowed_for_absolute_difference(self):
        with pytest.raises(ValueError, match="epsilon value can only be used with absolute difference operation"):
            NumberCheckMetric(
                operation="equals",
                left_template="{{item.left}}",
                right_template="{{item.right}}",
                epsilon=0.25,
            )

    @pytest.mark.parametrize(
        ("operation", "left", "right", "expected"),
        [
            ("equals", "3", "3", True),
            ("not equals", "3", "4", True),
            ("gte", "4", "3", True),
            ("gt", "4", "3", True),
            ("lte", "3", "3", True),
            ("lt", "2", "3", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_operations(self, operation, left, right, expected):
        metric = NumberCheckMetric(
            operation=operation,
            left_template="{{item.left}}",
            right_template="{{item.right}}",
        )
        assert (await compute_scores(metric, {"left": left, "right": right}, {})).outputs[0].value == (
            1.0 if expected else 0.0
        )

    @pytest.mark.asyncio
    async def test_absolute_difference(self):
        metric = NumberCheckMetric(
            operation="absolute difference",
            left_template="{{item.left}}",
            right_template="{{item.right}}",
            epsilon=0.25,
        )
        assert (await compute_scores(metric, {"left": "3.0", "right": "3.1"}, {})).outputs[0].value == 1.0
        assert (await compute_scores(metric, {"left": "3.0", "right": "3.4"}, {})).outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_absolute_difference_raises_when_epsilon_removed_after_init(self):
        metric = NumberCheckMetric(
            operation="absolute difference",
            left_template="{{item.left}}",
            right_template="{{item.right}}",
            epsilon=0.25,
        )
        metric.epsilon = None

        with pytest.raises(ValueError, match="epsilon value is required"):
            await compute_scores(metric, {"left": "3.0", "right": "3.1"}, {})

    @pytest.mark.asyncio
    async def test_nan_short_circuit(self):
        metric = NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        assert math.isnan((await compute_scores(metric, {"left": "abc", "right": "1"}, {})).outputs[0].value)
        assert math.isnan((await compute_scores(metric, {"left": "1", "right": "abc"}, {})).outputs[0].value)

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises(self):
        metric = NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        metric.operation = "unsupported"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="Unsupported operation"):
            await compute_scores(metric, {"left": "1", "right": "1"}, {})

    @pytest.mark.asyncio
    async def test_compute_scores_and_score_names(self):
        metric = NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        result = await compute_scores(metric, {"left": "1", "right": "1"}, {})
        assert result.outputs[0].name == "number-check"
        assert output_names(metric) == ["number-check"]

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_template_field(self):
        metric = NumberCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, {"left": "1"}, {})
        assert "could not render its 'right_template' template for this row" in str(exc_info.value)
        assert "missing_key='right'" in str(exc_info.value)

    def test_run_sync(self):
        metric = NumberCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{item.actual}}",
        )
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[{"expected": "1", "actual": "1"}, {"expected": "1", "actual": "2"}],
        )
        assert len(result.row_scores) == 2


class TestNumberCheckMetricConfig:
    def test_absolute_difference_requires_epsilon(self):
        with pytest.raises(ValueError, match="epsilon value is required with operation absolute difference"):
            NumberCheckMetric(
                operation="absolute difference",
                left_template="{{item.left}}",
                right_template="{{item.right}}",
            )

    def test_epsilon_only_allowed_for_absolute_difference(self):
        with pytest.raises(
            ValueError, match="epsilon value can only be used with absolute difference operation: equals"
        ):
            NumberCheckMetric(
                operation="equals",
                left_template="{{item.left}}",
                right_template="{{item.right}}",
                epsilon=0.25,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            1.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "1.0"},
            1.0,
            "int equals float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            0.0,
            "float 1 != 1.2",
        ),
        (
            {"reference_answer": "-0.123"},
            {"output_text": "I first thought 123 but then my final answer is -0.123."},
            1.0,
            "negative float",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            0.0,
            "positive and negative float are not equal",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            0.0,
            "simple numbers ref is less than output",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_equals(item: dict, sample: dict, expected_score: float, desc: str):
    for op in EQUALS_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            0.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "1.0"},
            0.0,
            "int equals float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            1.0,
            "float 1 != 1.2",
        ),
        (
            {"reference_answer": "-0.123"},
            {"output_text": "I first thought 123 but then my final answer is -0.123."},
            0.0,
            "negative float equals",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            1.0,
            "positive and negative float are not equal",
        ),
        ({"reference_answer": "1"}, {"output_text": "23"}, 1.0, "simple numbers ref is greater than output"),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_not_equals(item: dict, sample: dict, expected_score: float, desc: str):
    for op in NOT_EQUALS_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            1.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            0.0,
            "float 1 < 1.2",
        ),
        (
            {"reference_answer": "-0.1"},
            {"output_text": "I first thought 123 but then my final answer is -0.22."},
            1.0,
            "negative float -0.1 > -0.22",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            1.0,
            "positive float greater than negative float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            0.0,
            "simple numbers ref is less than output",
        ),
        (
            {"reference_answer": "23"},
            {"output_text": "1"},
            1.0,
            "simple numbers ref is greater than output",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_gte(item: dict, sample: dict, expected_score: float, desc: str):
    for op in GTE_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            0.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            0.0,
            "float 1 < 1.2",
        ),
        (
            {"reference_answer": "-0.1"},
            {"output_text": "I first thought 123 but then my final answer is -0.22."},
            1.0,
            "negative float -0.1 > -0.22",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            1.0,
            "positive float greater than negative float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            0.0,
            "simple numbers ref is less than output",
        ),
        (
            {"reference_answer": "23"},
            {"output_text": "1"},
            1.0,
            "simple numbers ref is greater than output",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_gt(item: dict, sample: dict, expected_score: float, desc: str):
    for op in GT_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


# @pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            1.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            1.0,
            "float 1 < 1.2",
        ),
        (
            {"reference_answer": "-0.1"},
            {"output_text": "I first thought 123 but then my final answer is -0.22."},
            0.0,
            "negative float -0.1 > -0.22",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            0.0,
            "positive float greater than negative float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            1.0,
            "simple numbers ref is less than output",
        ),
        (
            {"reference_answer": "23"},
            {"output_text": "1"},
            0.0,
            "simple numbers ref is greater than output",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        ({"reference_answer": ""}, {"output_text": "1"}, float("nan"), "no number parsed reference_answer"),
    ],
)
async def test_lte(item: dict, sample: dict, expected_score: float, desc: str):
    for op in LTE_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


# @pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            0.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            1.0,
            "float 1 < 1.2",
        ),
        (
            {"reference_answer": "-0.1"},
            {"output_text": "I first thought 123 but then my final answer is -0.22."},
            0.0,
            "negative float -0.1 > -0.22",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            0.0,
            "positive float greater than negative float",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            1.0,
            "simple numbers ref is less than output",
        ),
        (
            {"reference_answer": "23"},
            {"output_text": "1"},
            0.0,
            "simple numbers ref is greater than output",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_lt(item: dict, sample: dict, expected_score: float, desc: str):
    for op in LT_OPS:
        metric = NumberCheckMetric(
            operation=op,
            left_template="{{item.reference_answer | trim}}",
            right_template="{{output_text | trim}}",
        )

        metric_result = await compute_scores(metric, item, sample)
        assert len(metric_result.outputs) == 1
        score = metric_result.outputs[0]
        assert score.name == "number-check"
        if math.isnan(expected_score):
            assert math.isnan(score.value)
        else:
            assert score.value == expected_score, desc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "sample", "expected_score", "desc"),
    [
        (
            {"reference_answer": "context 1.\n"},
            {"output_text": "my reasoning 123 and final answer is 1 with more words after to be ignored"},
            1.0,
            "last number is parsed equals",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "my reasoning 123 and final answer is 1.2"},
            1.0,
            "abs(1 - 1.2) <= 1",
        ),
        (
            {"reference_answer": "0"},
            {"output_text": "my reasoning 123 and final answer is 2.2"},
            0.0,
            "abs(0 - 2.2) > 1",
        ),
        (
            {"reference_answer": "-0.1"},
            {"output_text": "I first thought 123 but then my final answer is -0.22."},
            1.0,
            "negative floats -0.1 - -0.22 <= 1",
        ),
        (
            {"reference_answer": "0.123"},
            {"output_text": "-0.123."},
            1.0,
            "positive float and negative float == 1",
        ),
        (
            {"reference_answer": "1.123"},
            {"output_text": "-2.123."},
            0.0,
            "positive float and negative float > 1",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "23"},
            0.0,
            "simple numbers abs(1-23) > 2",
        ),
        (
            {"reference_answer": "23"},
            {"output_text": "1"},
            0.0,
            "simple numbers abs(1-23) > 2",
        ),
        (
            {"reference_answer": "1"},
            {"output_text": "no number here"},
            float("nan"),
            "no number parsed output_text",
        ),
        (
            {"reference_answer": "no number here"},
            {"output_text": "1"},
            float("nan"),
            "no number parsed reference_answer",
        ),
    ],
)
async def test_abs_diff(item: dict, sample: dict, expected_score: float, desc: str):
    metric = NumberCheckMetric(
        operation="absolute difference",
        left_template="{{item.reference_answer | trim}}",
        right_template="{{output_text | trim}}",
        epsilon=2,
    )

    metric_result = await compute_scores(metric, item, sample)
    assert len(metric_result.outputs) == 1
    score = metric_result.outputs[0]
    assert score.name == "number-check"
    if math.isnan(expected_score):
        assert math.isnan(score.value)
    else:
        assert score.value == expected_score, desc


@pytest.fixture
def int_digit_limit() -> Iterator[int]:
    """Pin the integer-string digit limit, which ``PYTHONINTMAXSTRDIGITS`` can change."""
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(INT_DIGIT_LIMIT)
    yield INT_DIGIT_LIMIT
    sys.set_int_max_str_digits(previous)


async def _check(
    operation: NumberOperation, left: object, right: object, epsilon: int | float | None = None
) -> MetricResult:
    metric = NumberCheckMetric(
        operation=operation, left_template="{{item.left}}", right_template="{{item.right}}", epsilon=epsilon
    )
    return await compute_scores(metric, {"left": left, "right": right}, {})


def _out_of_range_operands(result: MetricResult) -> list[tuple[object, object]]:
    return [
        ((diagnostic.details or {}).get("operand"), (diagnostic.details or {}).get("representation"))
        for diagnostic in result.diagnostics
    ]


class TestOperandsBeyondFloatRange:
    @pytest.mark.parametrize(
        ("operation", "expected"),
        [("equals", 0.0), ("not equals", 1.0), (">", 1.0), ("<", 0.0)],
    )
    @pytest.mark.asyncio
    async def test_compares_exactly_against_small_number(self, operation: NumberOperation, expected: float):
        assert (await _check(operation, BEYOND_FLOAT_RANGE, "42")).outputs[0].value == expected

    @pytest.mark.asyncio
    async def test_identical_values_compare_equal(self):
        assert (await _check("equals", BEYOND_FLOAT_RANGE, BEYOND_FLOAT_RANGE)).outputs[0].value == 1.0

    @pytest.mark.parametrize(
        ("left", "operation", "right", "expected"),
        [
            ("-" + BEYOND_FLOAT_RANGE, "<", "42", 1.0),
            ("-" + BEYOND_FLOAT_RANGE, ">", "42", 0.0),
            ("-" + BEYOND_FLOAT_RANGE, "<", "-" + "9" * 399, 1.0),
        ],
        ids=["below-small", "not-above-small", "below-shorter-negative"],
    )
    @pytest.mark.asyncio
    async def test_negative_values_order_correctly(
        self, left: str, operation: NumberOperation, right: str, expected: float
    ):
        assert (await _check(operation, left, right)).outputs[0].value == expected

    @pytest.mark.parametrize(("epsilon", "expected"), [(1, 0.0), (10**400, 1.0)], ids=["small-epsilon", "huge-epsilon"])
    @pytest.mark.asyncio
    async def test_absolute_difference_with_finite_float_is_exact(self, epsilon: int, expected: float):
        result = await _check("absolute difference", "1" + "0" * 400, "42.5", epsilon=epsilon)
        assert result.outputs[0].value == expected

    @pytest.mark.asyncio
    async def test_absolute_difference_with_infinite_float(self):
        result = await _check("absolute difference", BEYOND_FLOAT_RANGE, BEYOND_FLOAT_RANGE + ".5", epsilon=1)
        assert result.outputs[0].value == 0.0

    @pytest.mark.parametrize(
        ("operation", "right", "epsilon", "expected"),
        [("equals", "42", None, 0.0), (">", "42", None, 1.0), ("absolute difference", "42.5", 1, 0.0)],
    )
    @pytest.mark.asyncio
    async def test_integer_just_above_float_max_keeps_score_and_gains_diagnostic(
        self, operation: NumberOperation, right: str, epsilon: int | None, expected: float
    ):
        result = await _check(operation, str(int(sys.float_info.max) + 1), right, epsilon=epsilon)
        assert result.outputs[0].value == expected
        assert _out_of_range_operands(result) == [("left", "exact_integer")]


@pytest.mark.usefixtures("int_digit_limit")
class TestIntegerDigitLimit:
    @pytest.mark.asyncio
    async def test_integer_over_limit_scores_without_raising(self):
        assert (await _check("equals", OVER_DIGIT_LIMIT, "42")).outputs[0].value == 0.0

    @pytest.mark.parametrize("sign", ["", "-"], ids=["positive", "negative"])
    @pytest.mark.parametrize("padding", ["", "0" * 10], ids=["unpadded", "padded-past-limit"])
    @pytest.mark.asyncio
    async def test_distinct_integers_at_limit_stay_distinguishable(self, padding: str, sign: str):
        left_digits = "1" + "0" * (INT_DIGIT_LIMIT - 1)
        right_digits = "1" + "0" * (INT_DIGIT_LIMIT - 2) + "1"
        left = sign + padding + left_digits
        right = sign + padding + right_digits
        assert _parse_number_answer(left) == int(sign + left_digits)
        assert (await _check("equals", left, right)).outputs[0].value == 0.0
        assert (await _check("equals", left, left)).outputs[0].value == 1.0

    @pytest.mark.parametrize("sign", ["", "-"], ids=["positive", "negative"])
    @pytest.mark.asyncio
    async def test_distinct_integers_over_limit_compare_equal_as_infinity(self, sign: str):
        left = sign + "1" + "0" * INT_DIGIT_LIMIT
        right = sign + "2" + "0" * INT_DIGIT_LIMIT
        assert (await _check("equals", left, right)).outputs[0].value == 1.0

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0" * 5000 + "7", 7),
            ("\u0660" * 5000 + "\u0667", 7),
            ("0\u0660" * 2500 + "7", 7),
            ("-" + "0" * 5000, 0),
        ],
        ids=["ascii", "arabic-indic", "mixed-script", "negative-zero"],
    )
    def test_zero_padding_preserves_small_integers(self, text: str, expected: int):
        value = _parse_number_answer(text)
        assert isinstance(value, int)
        assert value == expected

    @pytest.mark.asyncio
    async def test_zero_padding_keeps_integers_beyond_float_precision_exact(self):
        padded = "0" * 5000 + "9007199254740993"
        assert (await _check("equals", padded, "9007199254740993")).outputs[0].value == 1.0
        assert (await _check("equals", padded, "9007199254740992")).outputs[0].value == 0.0

    @pytest.mark.parametrize(
        ("value", "operation", "expected"),
        [(10**5000, ">", 1.0), (10**5000, "equals", 0.0), (-(10**5000), "<", 1.0)],
        ids=["positive-greater", "positive-equals", "negative-less"],
    )
    @pytest.mark.asyncio
    async def test_huge_native_integers_from_bare_templates(
        self, value: int, operation: NumberOperation, expected: float
    ):
        result = await _check(operation, value, 42)
        assert result.outputs[0].value == expected
        assert _out_of_range_operands(result) == [("left", "infinity")]


class TestOutOfRangeDiagnostics:
    @pytest.mark.usefixtures("int_digit_limit")
    @pytest.mark.asyncio
    async def test_identify_side_and_representation_without_digits(self):
        result = await _check("equals", BEYOND_FLOAT_RANGE, OVER_DIGIT_LIMIT)
        assert _out_of_range_operands(result) == [("left", "exact_integer"), ("right", "infinity")]
        for diagnostic in result.diagnostics:
            assert diagnostic.details is not None
            assert diagnostic.details[OUTPUT_DETAIL] == "number-check"
            assert diagnostic.details[REASON_DETAIL] == "operand_out_of_float_range"
            assert not any(character.isdigit() for character in diagnostic.model_dump_json())

    @pytest.mark.usefixtures("int_digit_limit")
    @pytest.mark.parametrize(("left", "right"), [("1", "1"), ("0" * 5000 + "7", "7")], ids=["ordinary", "zero-padded"])
    @pytest.mark.asyncio
    async def test_absent_for_values_within_float_range(self, left: str, right: str):
        result = await _check("equals", left, right)
        assert result.outputs[0].value == 1.0
        assert result.diagnostics == []


@pytest.mark.usefixtures("int_digit_limit")
@pytest.mark.parametrize(
    ("operation", "epsilon", "problem_row"),
    [
        pytest.param("equals", None, {"expected": "42", "actual": BEYOND_FLOAT_RANGE}, id="400-digit-string"),
        pytest.param("equals", None, {"expected": "42", "actual": OVER_DIGIT_LIMIT}, id="5000-digit-string"),
        pytest.param(
            "absolute difference",
            0.5,
            {"expected": "42.5", "actual": BEYOND_FLOAT_RANGE},
            id="400-digit-absolute-difference",
        ),
        pytest.param("equals", None, {"expected": 42, "actual": 10**5000}, id="positive-native-integer"),
        pytest.param("equals", None, {"expected": 42, "actual": -(10**5000)}, id="negative-native-integer"),
    ],
)
def test_run_scores_every_row_with_oversized_operand(
    operation: NumberOperation, epsilon: float | None, problem_row: dict
):
    metric = NumberCheckMetric(
        operation=operation, left_template="{{item.expected}}", right_template="{{item.actual}}", epsilon=epsilon
    )
    dataset = [{"expected": "42", "actual": "42"}, {"expected": "42", "actual": "7"}, problem_row]

    result = Evaluator().run_sync(metrics=[metric], dataset=dataset)

    scores = [outputs[0].value for row in result.row_scores for outputs in row.metrics.values()]
    assert scores == [1.0, 0.0, 0.0]
    assert [bool(row.metric_diagnostics) for row in result.row_scores] == [False, False, True]


class TestUnsupportedNumberFormats:
    @pytest.mark.parametrize(
        "answer",
        ["1,000", "The total is 12,345.67", "-1,000,000", "6.02e23", "1.5E-3", "1e+05", "5.e3"],
    )
    def test_comma_between_digits_or_scientific_notation_is_rejected(self, answer: str):
        assert math.isnan(_parse_number_answer(answer))

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("1, 2, and 3", 3), ("The answer is 42, final", 42), ("3 eggs", 3), ("version 2e", 2)],
    )
    def test_commas_and_letters_elsewhere_still_parse(self, answer: str, expected: int):
        assert _parse_number_answer(answer) == expected

    @pytest.mark.asyncio
    async def test_rejected_answer_scores_nan(self):
        assert math.isnan((await _check("equals", "1,000", "1000")).outputs[0].value)

    @pytest.mark.asyncio
    async def test_native_float_printed_in_scientific_notation_is_rejected(self):
        assert math.isnan((await _check("equals", 1e16, "42")).outputs[0].value)
