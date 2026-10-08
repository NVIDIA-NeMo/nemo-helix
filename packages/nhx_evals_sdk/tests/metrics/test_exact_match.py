# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric, MetricResult


class TestExactMatchMetric:
    def test_score_names(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")
        assert output_names(metric) == ["exact-match"]

    @pytest.mark.asyncio
    async def test_metric_returns_score(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")
        assert (await compute_scores(metric, {"reference": "The Cat"}, {"output_text": "the cat"})).outputs[
            0
        ].value == 1.0

    def test_input_schema_tracks_reference_template(self):
        metric = ExactMatchMetric(reference="{{reference}}")

        assert metric.input_schema().schema_ == {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"reference": {}},
            "required": ["reference"],
        }

    def test_init_valid_params(self):
        """Test that __init__ works with valid params."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        assert metric.reference == "{{item.reference}}"
        assert metric.candidate is None
        assert metric.type.value == "exact-match"

    def test_init_with_candidate(self):
        """Test that __init__ works with custom candidate template."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
            candidate="{{sample.custom_output}}",
        )

        assert metric.candidate == "{{sample.custom_output}}"

    @pytest.mark.asyncio
    async def test_compute_scores_exact_match(self):
        """Test compute_scores with exact match."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "exact-match"
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": "a"}, {"output_text": "a"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_case_insensitive(self):
        """Test compute_scores is case insensitive."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The Cat Sat On The Mat."}
        sample = {"output_text": "the cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_ignores_punctuation(self):
        """Test compute_scores ignores punctuation."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The cat sat on the mat!"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_ignores_articles(self):
        """Test compute_scores ignores articles (a, an, the)."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on a mat."}
        sample = {"output_text": "cat sat on mat"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_no_match(self):
        """Test compute_scores with no match."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The dog ran in the park."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 0

    @pytest.mark.asyncio
    async def test_compute_scores_with_custom_candidate(self):
        """Test compute_scores with custom candidate template."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
            candidate="{{item.custom_output}}",
        )

        item = {"reference": "The cat sat on the mat.", "custom_output": "The cat sat on the mat."}
        sample = {"output_text": "This should be ignored."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_whitespace_handling(self):
        """Test compute_scores handles extra whitespace."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The   cat    sat on the mat."}
        sample = {"output_text": "The cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1

    @pytest.mark.asyncio
    async def test_compute_scores_empty_strings(self):
        """Test compute_scores handles empty strings."""
        metric = ExactMatchMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": ""}
        sample = {"output_text": ""}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value == 1
