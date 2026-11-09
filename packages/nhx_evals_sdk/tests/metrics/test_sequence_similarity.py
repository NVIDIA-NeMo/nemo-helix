# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import difflib

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.sequence_similarity import SequenceSimilarityMetric
from nhx_evals_sdk.metrics.types import MetricsUnion
from pydantic import TypeAdapter

# A receipt label in the CORD-v2 tag format, long enough (345 characters) for SequenceMatcher's autojunk
# rule to apply when it is the second argument.
RECEIPT_LABEL = (
    "<s_total><s_total_price>38,000</s_total_price><s_changeprice>12,000</s_changeprice>"
    "<s_cashprice>50,000</s_cashprice></s_total><s_menu><s_price>18,000</s_price><s_nm>ICED TEA</s_nm>"
    "<s_cnt>2</s_cnt><sep/><s_price>12,000</s_price><s_nm>FRIED RICE</s_nm><s_cnt>1</s_cnt><sep/>"
    "<s_price>8,000</s_price><s_nm>SPRING ROLL</s_nm><s_cnt>1</s_cnt></s_menu>"
)
RECEIPT_PREDICTION = (
    "<s_total><s_total_price>38,000</s_total_price></s_total><s_menu><s_nm>ICED TEA</s_nm>"
    "<s_cnt>2</s_cnt><s_price>18,000</s_price></s_menu>"
)


class TestSequenceSimilarityMetric:
    @pytest.mark.parametrize(
        ("reference", "output_text", "expected"),
        [
            ("cat sat", "cat sat", 1.0),
            ("abc", "xyz", 0.0),
            ("", "", 1.0),
            ("", "x", 0.0),
            ("abcd", "abce", 0.75),
        ],
    )
    @pytest.mark.asyncio
    async def test_scores_default_candidate(self, reference, output_text, expected):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": reference}, {"output_text": output_text})
        assert result.outputs[0].value == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_matches_difflib_with_candidate_first_and_autojunk_on(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": RECEIPT_LABEL}, {"output_text": RECEIPT_PREDICTION})

        expected = difflib.SequenceMatcher(None, RECEIPT_PREDICTION, RECEIPT_LABEL).ratio()
        assert result.outputs[0].value == expected
        assert result.outputs[0].value == pytest.approx(0.5291666666666667)
        # Swapping the arguments, or turning autojunk off, gives a different score for this pair.
        assert expected != difflib.SequenceMatcher(None, RECEIPT_LABEL, RECEIPT_PREDICTION).ratio()
        assert expected != difflib.SequenceMatcher(None, RECEIPT_PREDICTION, RECEIPT_LABEL, autojunk=False).ratio()

    @pytest.mark.asyncio
    async def test_does_not_normalize(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        for output_text in ("Paris ", "paris"):
            result = await compute_scores(metric, {"reference": "Paris"}, {"output_text": output_text})
            assert result.outputs[0].value < 1.0

    @pytest.mark.asyncio
    async def test_candidate_template_can_trim(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}", candidate="{{sample.output_text | trim}}")
        result = await compute_scores(metric, {"reference": "Paris"}, {"output_text": "\nParis \n"})
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_custom_candidate_overrides_output_text(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}", candidate="{{item.prediction}}")
        result = await compute_scores(metric, {"reference": "abc", "prediction": "abc"}, {"output_text": "ignored"})
        assert result.outputs[0].value == 1.0

    @pytest.mark.asyncio
    async def test_validates_rendered_types(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        with pytest.raises(TypeError, match="The reference must be a string"):
            await compute_scores(metric, {"reference": 1}, {"output_text": "1"})
        with pytest.raises(TypeError, match="The candidate must be a string"):
            await compute_scores(metric, {"reference": "1"}, {"output_text": 1})

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(metric, item={"prompt": "hello"}, sample={"output_text": "hi"})
        assert "could not render its 'reference' template for this row" in str(exc_info.value)
        assert "missing_key='reference'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_output_text(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        with pytest.raises(ValueError, match="SequenceSimilarityMetric has missing `candidate` field"):
            await compute_scores(metric, {"reference": "a"}, {})

    def test_score_names(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}")
        assert metric.type.value == "sequence-similarity"
        assert output_names(metric) == ["sequence-similarity"]

    def test_round_trips_through_metrics_union(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}", candidate="{{sample.output_text | trim}}")
        restored = TypeAdapter(MetricsUnion).validate_python(metric.model_dump(mode="json"))
        assert isinstance(restored, SequenceSimilarityMetric)
        assert restored == metric

    def test_input_schema_lists_template_fields(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}", candidate="{{item.prediction}}")
        schema = metric.input_schema().schema_
        assert set(schema["required"]) == {"reference", "prediction"}

    def test_run_sync(self):
        metric = SequenceSimilarityMetric(reference="{{item.reference}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[{"reference": "abcd", "prediction": "abcd"}, {"reference": "abcd", "prediction": "abce"}],
        )
        assert [row.metrics["sequence-similarity"][0].value for row in result.row_scores] == [1.0, 0.75]
        aggregate = result.aggregate_scores.scores[0]
        assert aggregate.name == "sequence-similarity.sequence-similarity"
        assert aggregate.mean == pytest.approx(0.875)
