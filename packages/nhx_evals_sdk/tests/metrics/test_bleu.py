# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from metrics.helpers import compute_corpus_scores, compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.bleu import BLEUMetric, MetricResult


class TestBLEUMetric:
    @pytest.mark.asyncio
    async def test_metric_default_candidate(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        score = (
            (await compute_scores(metric, {"reference": "the cat sat"}, {"output_text": "the cat sat"}))
            .outputs[0]
            .value
        )
        assert score == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_metric_with_custom_candidate(self):
        metric = BLEUMetric(references=["{{item.reference}}"], candidate="{{item.pred}}")
        score = (
            (
                await compute_scores(
                    metric, {"reference": "the cat sat", "pred": "the cat sat"}, {"output_text": "ignored"}
                )
            )
            .outputs[0]
            .value
        )
        assert score == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_metric_scores(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        assert (await compute_scores(metric, {"reference": "the cat"}, {"output_text": "the cat"})).outputs[
            0
        ].value == pytest.approx(100.0)
        assert (await compute_scores(metric, {"reference": "the cat"}, {"output_text": "dog barked"})).outputs[
            0
        ].value == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_metric_validates_reference_and_candidate_types(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        with pytest.raises(TypeError, match="The reference must be a string"):
            await compute_scores(metric, {"reference": 1}, {"output_text": "1"})

        bad_candidate = BLEUMetric(references=["{{item.reference}}"], candidate="{{item.pred}}")
        with pytest.raises(TypeError, match="The candidate must be a string"):
            await compute_scores(bad_candidate, {"reference": "x", "pred": 1}, {"output_text": "ignored"})

    @pytest.mark.asyncio
    async def test_metric_default_candidate_non_string_raises_value_error(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        with pytest.raises(TypeError, match="The candidate must be a string"):
            await compute_scores(metric, {"reference": "x"}, {"output_text": 123})

    @pytest.mark.asyncio
    async def test_metric_default_candidate_missing_output_text_raises_clear_error(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        with pytest.raises(ValueError, match=r"candidate=\.\.\."):
            await compute_scores(metric, {"reference": "x"}, {})

    @pytest.mark.asyncio
    async def test_compute_scores_and_corpus_scores(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        row_result = await compute_scores(metric, {"reference": "the cat"}, {"output_text": "the cat"})
        assert row_result.outputs[0].name == "sentence"

        corpus_result = await compute_corpus_scores(
            metric,
            items=[{"reference": "the cat"}, {"reference": "a dog"}],
            samples=[{"output_text": "the cat"}, {"output_text": "a dog"}],
        )
        assert corpus_result is not None
        assert corpus_result.outputs[0].name == "corpus"

    @pytest.mark.asyncio
    async def test_corpus_uses_candidate_template_and_validates_type(self):
        metric = BLEUMetric(references=["{{item.reference}}"], candidate="{{item.pred}}")
        corpus_result = await compute_corpus_scores(
            metric,
            items=[{"reference": "the cat", "pred": "the cat"}],
            samples=[{"output_text": "ignored"}],
        )
        assert corpus_result is not None

        bad_metric = BLEUMetric(references=["{{item.reference}}"], candidate="{{item.pred}}")
        with pytest.raises(TypeError, match="The candidate must be a string"):
            await compute_corpus_scores(
                bad_metric,
                items=[{"reference": "the cat", "pred": 1}],
                samples=[{"output_text": "ignored"}],
            )

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = BLEUMetric(references=["{{item.reference}}"], candidate="{{sample.output_text}}")
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, item={"prompt": "hello"}, sample={"output_text": "hi"})
        assert "could not render its 'references[0]' template for this row" in str(exc_info.value)
        assert "missing_key='reference'" in str(exc_info.value)

    def test_score_names(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        assert output_names(metric) == ["sentence"]

    def test_run_sync_adds_corpus_score(self):
        metric = BLEUMetric(references=["{{item.reference}}"], candidate="{{item.pred}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {"reference": "the cat sat", "pred": "the cat sat"},
                {"reference": "a dog ran", "pred": "a dog ran"},
            ],
        )
        aggregate_names = {score.name for score in result.aggregate_scores.scores}
        assert "bleu.sentence" in aggregate_names
        assert "bleu.corpus" in aggregate_names

    @pytest.mark.asyncio
    async def test_compute_scores_exact_match(self):
        """Test compute_scores with exact match between candidate and reference."""
        metric = BLEUMetric(
            references=["{{item.reference}}"],
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "sentence"
        # Exact match should give a perfect or near-perfect BLEU score
        assert result.outputs[0].value > 90.0

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = BLEUMetric(references=["{{item.reference}}"])
        result = await compute_scores(metric, {"reference": "a"}, {"output_text": "a"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_partial_match(self):
        """Test compute_scores with partial match."""
        metric = BLEUMetric(
            references=["{{item.reference}}"],
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The dog sat on the floor."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "sentence"
        # Partial match should give a lower score
        assert 0.0 <= result.outputs[0].value < 100.0

    @pytest.mark.asyncio
    async def test_compute_scores_no_match(self):
        """Test compute_scores with no match."""
        metric = BLEUMetric(
            references=["{{item.reference}}"],
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "Completely different sentence with no overlap."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "sentence"
        # No match should give a very low score
        assert result.outputs[0].value >= 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_with_custom_candidate(self):
        """Test compute_scores with custom candidate template."""
        metric = BLEUMetric(
            references=["{{item.reference}}"],
            candidate="{{item.custom_output}}",
        )

        item = {"reference": "The cat sat on the mat.", "custom_output": "The cat sat on the mat."}
        sample = {"output_text": "This should be ignored."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].value > 90.0

    @pytest.mark.asyncio
    async def test_compute_corpus_scores(self):
        """Test compute_corpus_scores with multiple items."""
        metric = BLEUMetric(
            references=["{{item.reference}}"],
        )

        items = [
            {"reference": "The cat sat on the mat."},
            {"reference": "The dog ran in the park."},
        ]
        samples = [
            {"output_text": "The cat sat on the mat."},
            {"output_text": "The dog ran in the park."},
        ]

        result = await compute_corpus_scores(metric, items, samples)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        assert result.outputs[0].name == "corpus"
        # Good matches should give high corpus score
        assert result.outputs[0].value > 90.0

    @pytest.mark.asyncio
    async def test_compute_scores_multiple_references(self):
        """Test compute_scores with multiple references."""
        metric = BLEUMetric(
            references=["{{item.reference1}}", "{{item.reference2}}"],
        )

        item = {"reference1": "The cat sat on the mat.", "reference2": "A cat was sitting on the mat."}
        sample = {"output_text": "The cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 1
        # With multiple references, should still compute properly
        assert result.outputs[0].value >= 0.0
