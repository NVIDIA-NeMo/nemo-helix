# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.f1 import F1Metric, MetricResult


class TestF1Metric:
    @pytest.mark.parametrize(
        ("item", "sample", "expected"),
        [
            ({"reference": "cat sat"}, {"output_text": "cat sat"}, 1.0),
            ({"reference": "cat sat"}, {"output_text": "dog ran"}, 0.0),
            ({"reference": ""}, {"output_text": ""}, 1.0),
            ({"reference": ""}, {"output_text": "x"}, 0.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_metric_default_candidate(self, item, sample, expected):
        metric = F1Metric(reference="{{item.reference}}")
        assert (await compute_scores(metric, item, sample)).outputs[0].value == expected

    @pytest.mark.asyncio
    async def test_metric_with_custom_candidate(self):
        metric = F1Metric(reference="{{item.reference}}", candidate="{{item.pred}}")
        assert (
            await compute_scores(metric, {"reference": "a b c", "pred": "a b"}, {"output_text": "ignored"})
        ).outputs[0].value == pytest.approx(0.6666666666666666)

    @pytest.mark.asyncio
    async def test_metric_validates_rendered_types(self):
        metric = F1Metric(reference="{{item.reference}}")
        with pytest.raises(TypeError, match="The reference must be a string"):
            await compute_scores(metric, {"reference": 1}, {"output_text": "1"})
        with pytest.raises(TypeError, match="The candidate must be a string"):
            await compute_scores(metric, {"reference": "1"}, {"output_text": 1})

    @pytest.mark.asyncio
    async def test_compute_scores(self):
        metric = F1Metric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": "cat"}, {"output_text": "cat"})
        assert result.outputs[0].name == "f1"
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = F1Metric(reference="{{item.reference}}", candidate="{{sample.output_text}}")
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, item={"prompt": "hello"}, sample={"output_text": "hi"})
        assert "could not render its 'reference' template for this row" in str(exc_info.value)
        assert "missing_key='reference'" in str(exc_info.value)

    def test_score_names(self):
        metric = F1Metric(reference="{{item.reference}}")
        assert output_names(metric) == ["f1"]

    def test_run_sync(self):
        metric = F1Metric(reference="{{item.reference}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[{"reference": "a", "prediction": "a"}, {"reference": "a", "prediction": "b"}],
        )
        assert len(result.row_scores) == 2

    def test_init_valid_params(self):
        """Test that __init__ works with valid params."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        assert metric.reference == "{{item.reference}}"
        assert metric.candidate is None
        assert metric.type.value == "f1"

    def test_init_with_prediction(self):
        """Test that __init__ works with custom prediction template."""
        metric = F1Metric(
            reference="{{item.reference}}",
            candidate="{{sample.custom_output}}",
        )

        assert metric.candidate == "{{sample.custom_output}}"

    @pytest.mark.asyncio
    async def test_compute_scores_exact_match(self):
        """Test compute_scores with exact match."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat"}
        sample = {"output_text": "The cat sat on the mat"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "f1"
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = F1Metric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": "a"}, {"output_text": "a"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_partial_overlap(self):
        """Test compute_scores with partial token overlap."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat"}
        sample = {"output_text": "The cat slept on the floor"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "f1"
        # Partial overlap should give score between 0 and 1
        assert 0.0 < result.outputs[0].value < 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_no_overlap(self):
        """Test compute_scores with no token overlap."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "hello world"}
        sample = {"output_text": "goodbye universe"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_both_empty(self):
        """Test compute_scores with both strings empty."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": ""}
        sample = {"output_text": ""}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        # Empty strings that agree should return 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_one_empty(self):
        """Test compute_scores with one string empty."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": ""}
        sample = {"output_text": "some text"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        # One empty, one not should return 0
        assert result.outputs[0].value == 0

    @pytest.mark.asyncio
    async def test_compute_scores_case_insensitive(self):
        """Test compute_scores is case insensitive."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The Cat Sat"}
        sample = {"output_text": "the cat sat"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_ignores_punctuation(self):
        """Test compute_scores ignores punctuation."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "Hello, world!"}
        sample = {"output_text": "Hello world"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_ignores_articles(self):
        """Test compute_scores ignores articles (a, an, the)."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on a mat"}
        sample = {"output_text": "cat sat on mat"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_duplicate_tokens(self):
        """Test compute_scores handles duplicate tokens correctly."""
        metric = F1Metric(
            reference="{{item.reference}}",
        )

        item = {"reference": "cat cat dog"}
        sample = {"output_text": "cat dog dog"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        # Should handle duplicate tokens with Counter intersection
        assert 0.0 < result.outputs[0].value < 1.0
