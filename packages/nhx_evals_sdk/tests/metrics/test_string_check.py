# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.string_check import MetricResult, StringCheckMetric, StringCheckOperation

NOT_EQUALS_OPS: tuple[StringCheckOperation, ...] = ("!=", "<>", "not equals")


class TestStringCheckMetric:
    @pytest.mark.parametrize(
        ("operation", "left", "right", "expected"),
        [
            ("equals", "abc", "abc", True),
            ("not equals", "abc", "def", True),
            ("contains", "abc", "b", True),
            ("not contains", "abc", "z", True),
            ("startswith", "abc", "a", True),
            ("endswith", "abc", "c", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_operations(self, operation, left, right, expected):
        metric = StringCheckMetric(
            operation=operation,
            left_template="{{item.left}}",
            right_template="{{item.right}}",
        )
        assert (await compute_scores(metric, {"left": left, "right": right}, {})).outputs[0].value == (
            1.0 if expected else 0.0
        )

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises(self):
        metric = StringCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        metric.operation = "unknown"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="Unsupported operation"):
            await compute_scores(metric, {"left": "a", "right": "a"}, {})

    @pytest.mark.asyncio
    async def test_validates_rendered_types(self):
        metric = StringCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        with pytest.raises(TypeError, match="The left value must be a string"):
            await compute_scores(metric, {"left": 1, "right": "1"}, {})
        with pytest.raises(TypeError, match="The right value must be a string"):
            await compute_scores(metric, {"left": "1", "right": 1}, {})

    @pytest.mark.asyncio
    async def test_compute_scores_and_score_names(self):
        metric = StringCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        result = await compute_scores(metric, {"left": "x", "right": "x"}, {})
        assert result.outputs[0].name == "string-check"
        assert output_names(metric) == ["string-check"]

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_template_field(self):
        metric = StringCheckMetric(operation="equals", left_template="{{item.left}}", right_template="{{item.right}}")
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, {"left": "x"}, {})
        assert "could not render its 'right_template' template for this row" in str(exc_info.value)
        assert "missing_key='right'" in str(exc_info.value)

    def test_run_sync(self):
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{item.actual}}",
        )
        result = Evaluator().run_sync(metrics=[metric], dataset=[{"expected": "x", "actual": "x"}])
        assert len(result.row_scores) == 1

    def test_init_valid_params(self):
        """Test that __init__ works with valid params."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        assert metric.operation == "equals"
        assert metric.left_template == "{{item.expected}}"
        assert metric.right_template == "{{sample.output_text}}"
        assert metric.type.value == "string-check"

    @pytest.mark.asyncio
    async def test_compute_scores_equals_true(self):
        """Test compute_scores with equals operation returning true."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        item = {"expected": "hello"}
        sample = {"output_text": "hello"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "string-check"
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )
        result = await compute_scores(metric, {"expected": "a"}, {"output_text": "a"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_equals_false(self):
        """Test compute_scores with equals operation returning false."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        item = {"expected": "hello"}
        sample = {"output_text": "goodbye"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_double_equals(self):
        """Test compute_scores with == operator."""
        metric = StringCheckMetric(
            operation="==",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        item = {"expected": "hello"}
        sample = {"output_text": "hello"}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_not_equals(self):
        """Test compute_scores with not equals operation."""
        metric = StringCheckMetric(
            operation="!=",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        item = {"expected": "hello"}
        sample = {"output_text": "goodbye"}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_not_equals_variants(self):
        """Test compute_scores with all not equals variants."""
        for op in NOT_EQUALS_OPS:
            metric = StringCheckMetric(
                operation=op,
                left_template="{{item.expected}}",
                right_template="{{sample.output_text}}",
            )

            item = {"expected": "hello"}
            sample = {"output_text": "goodbye"}

            result = await compute_scores(metric, item, sample)
            assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_contains_true(self):
        """Test compute_scores with contains operation returning true."""
        metric = StringCheckMetric(
            operation="contains",
            left_template="{{item.haystack}}",
            right_template="{{item.needle}}",
        )

        item = {"haystack": "hello world", "needle": "world"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_contains_false(self):
        """Test compute_scores with contains operation returning false."""
        metric = StringCheckMetric(
            operation="contains",
            left_template="{{item.haystack}}",
            right_template="{{item.needle}}",
        )

        item = {"haystack": "hello world", "needle": "goodbye"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_not_contains(self):
        """Test compute_scores with not contains operation."""
        metric = StringCheckMetric(
            operation="not contains",
            left_template="{{item.haystack}}",
            right_template="{{item.needle}}",
        )

        item = {"haystack": "hello world", "needle": "goodbye"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_startswith_true(self):
        """Test compute_scores with startswith operation returning true."""
        metric = StringCheckMetric(
            operation="startswith",
            left_template="{{item.text}}",
            right_template="{{item.prefix}}",
        )

        item = {"text": "hello world", "prefix": "hello"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_startswith_false(self):
        """Test compute_scores with startswith operation returning false."""
        metric = StringCheckMetric(
            operation="startswith",
            left_template="{{item.text}}",
            right_template="{{item.prefix}}",
        )

        item = {"text": "hello world", "prefix": "world"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_endswith_true(self):
        """Test compute_scores with endswith operation returning true."""
        metric = StringCheckMetric(
            operation="endswith",
            left_template="{{item.text}}",
            right_template="{{item.suffix}}",
        )

        item = {"text": "hello world", "suffix": "world"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_endswith_false(self):
        """Test compute_scores with endswith operation returning false."""
        metric = StringCheckMetric(
            operation="endswith",
            left_template="{{item.text}}",
            right_template="{{item.suffix}}",
        )

        item = {"text": "hello world", "suffix": "hello"}
        sample = {}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_case_sensitive(self):
        """Test compute_scores is case sensitive."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )

        item = {"expected": "Hello"}
        sample = {"output_text": "hello"}

        result = await compute_scores(metric, item, sample)

        # Should be case sensitive
        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_with_jinja_filters(self):
        """Test compute_scores works with Jinja filters in templates."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected | trim}}",
            right_template="{{sample.output_text | trim}}",
        )

        item = {"expected": "  hello  "}
        sample = {"output_text": "hello"}

        result = await compute_scores(metric, item, sample)

        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_metric_method_unsupported_operation(self):
        """Test the metric method raises error for unsupported operation."""
        metric = StringCheckMetric(
            operation="equals",
            left_template="{{item.expected}}",
            right_template="{{sample.output_text}}",
        )
        # Manually override operation to test error handling; the literal type forbids it,
        # which is exactly the runtime case under test.
        metric.operation = cast(Any, "unsupported")

        item = {"expected": "hello"}
        sample = {"output_text": "hello"}

        with pytest.raises(ValueError, match="Unsupported operation"):
            await compute_scores(metric, item, sample)
