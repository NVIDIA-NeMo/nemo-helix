# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.rouge import MetricResult, ROUGEMetric


class _FakeRougeScore:
    """Minimal ROUGE score object for unit tests."""

    def __init__(self, fmeasure: float):
        self.fmeasure = fmeasure


class _FakeRougeScorer:
    """Deterministic scorer used to isolate unit tests from NLTK/scipy imports."""

    def __init__(self, scores: dict[str, _FakeRougeScore]):
        self._scores = scores

    def score(self, *_args, **_kwargs) -> dict[str, _FakeRougeScore]:
        return self._scores


class TestROUGEMetric:
    def test_cached_scorer_initialization_uses_lazy_import(self, monkeypatch: pytest.MonkeyPatch):
        class _StubRougeScorer:
            def __init__(self, keys: list[str], *, use_stemmer: bool):
                self.keys = keys
                self.use_stemmer = use_stemmer

        fake_pkg = types.ModuleType("rouge_score")
        setattr(fake_pkg, "rouge_scorer", types.SimpleNamespace(RougeScorer=_StubRougeScorer))
        monkeypatch.setitem(sys.modules, "rouge_score", fake_pkg)

        metric = ROUGEMetric(reference="{{item.reference}}")
        scorer = metric._scorer

        assert isinstance(scorer, _StubRougeScorer)
        assert scorer.keys == ["rouge1", "rouge2", "rouge3", "rougeL"]
        assert scorer.use_stemmer is True

    def test_score_names(self):
        metric = ROUGEMetric(reference="{{item.reference}}")
        assert output_names(metric) == ["rouge_1_score", "rouge_2_score", "rouge_3_score", "rouge_L_score"]

    @pytest.mark.asyncio
    async def test_compute_scores_default_candidate(self):
        metric = ROUGEMetric(reference="{{item.reference}}")
        metric.__dict__["_scorer"] = _FakeRougeScorer(
            {
                "rouge1": _FakeRougeScore(1.0),
                "rouge2": _FakeRougeScore(1.0),
                "rouge3": _FakeRougeScore(1.0),
                "rougeL": _FakeRougeScore(1.0),
            }
        )
        result = await compute_scores(metric, {"reference": "the cat sat"}, {"output_text": "the cat sat"})
        assert {score.name for score in result.outputs} == {
            "rouge_1_score",
            "rouge_2_score",
            "rouge_3_score",
            "rouge_L_score",
        }

    @pytest.mark.asyncio
    async def test_compute_scores_custom_candidate(self):
        metric = ROUGEMetric(reference="{{item.reference}}", candidate="{{item.pred}}")
        metric.__dict__["_scorer"] = _FakeRougeScorer(
            {
                "rouge1": _FakeRougeScore(1.0),
                "rouge2": _FakeRougeScore(1.0),
                "rouge3": _FakeRougeScore(1.0),
                "rougeL": _FakeRougeScore(1.0),
            }
        )
        result = await compute_scores(metric, {"reference": "the cat sat", "pred": "the cat sat"}, {"output_text": "x"})
        assert all(score.value == pytest.approx(1.0) for score in result.outputs)

    @pytest.mark.asyncio
    async def test_metric_scores(self):
        metric = ROUGEMetric(reference="{{item.reference}}")
        metric.__dict__["_scorer"] = _FakeRougeScorer(
            {
                "rouge1": _FakeRougeScore(1.0),
                "rouge2": _FakeRougeScore(0.0),
                "rouge3": _FakeRougeScore(0.0),
                "rougeL": _FakeRougeScore(0.0),
            }
        )
        assert (await compute_scores(metric, {"reference": "the cat"}, {"output_text": "the cat"})).outputs[
            0
        ].value == 1.0
        metric.__dict__["_scorer"] = _FakeRougeScorer(
            {
                "rouge1": _FakeRougeScore(0.0),
                "rouge2": _FakeRougeScore(0.0),
                "rouge3": _FakeRougeScore(0.0),
                "rougeL": _FakeRougeScore(0.0),
            }
        )
        assert (await compute_scores(metric, {"reference": "the cat"}, {"output_text": "a dog"})).outputs[
            0
        ].value == 0.0
        assert isinstance(
            (await compute_scores(metric, {"reference": "the cat"}, {"output_text": "the cat"})).outputs[0].value, float
        )

    def test_metric_validates_rendered_types(self):
        metric = ROUGEMetric(reference="{{item.reference}}")
        metric.__dict__["_scorer"] = _FakeRougeScorer(
            {
                "rouge1": _FakeRougeScore(1.0),
                "rouge2": _FakeRougeScore(1.0),
                "rouge3": _FakeRougeScore(1.0),
                "rougeL": _FakeRougeScore(1.0),
            }
        )
        with pytest.raises(TypeError, match="The reference must be a string"):
            metric._metric({"reference": 1}, {"output_text": "1"})

        bad_candidate = ROUGEMetric(reference="{{item.reference}}", candidate="{{item.pred}}")
        bad_candidate.__dict__["_scorer"] = metric.__dict__["_scorer"]
        with pytest.raises(TypeError, match="The candidate must be a string"):
            bad_candidate._metric({"reference": "x", "pred": 1}, {"output_text": "x"})

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = ROUGEMetric(reference="{{item.reference}}", candidate="{{sample.output_text}}")

        with pytest.raises(ValueError) as exc_info:
            await compute_scores(
                metric,
                item={"prompt": "Summarize this"},
                sample={"output_text": "summary"},
            )

        assert "could not render its 'reference' template for this row" in str(exc_info.value)
        assert "missing_key='reference'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_default_candidate_missing_output_text(self):
        metric = ROUGEMetric(reference="{{item.reference}}")

        with pytest.raises(ValueError) as exc_info:
            await compute_scores(
                metric,
                item={"reference": "Paris"},
                sample={"some_other_field": "Paris"},
            )

        assert str(exc_info.value) == (
            "ROUGEMetric has missing `candidate` field.\n"
            "For offline evaluation, `candidate=...` field is required when constructing ROUGEMetric.\n"
            "For online evaluation, this usually means the evaluated model produced no output."
        )

    def test_run_sync(self, monkeypatch: pytest.MonkeyPatch):
        class _StubRougeScorer:
            def __init__(self, *_args, **_kwargs):
                pass

            def score(self, *_args, **_kwargs) -> dict[str, _FakeRougeScore]:
                return {
                    "rouge1": _FakeRougeScore(1.0),
                    "rouge2": _FakeRougeScore(1.0),
                    "rouge3": _FakeRougeScore(1.0),
                    "rougeL": _FakeRougeScore(1.0),
                }

        fake_pkg = types.ModuleType("rouge_score")
        setattr(fake_pkg, "rouge_scorer", types.SimpleNamespace(RougeScorer=_StubRougeScorer))
        monkeypatch.setitem(sys.modules, "rouge_score", fake_pkg)

        metric = ROUGEMetric(reference="{{item.reference}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[{"reference": "the cat sat", "prediction": "the cat sat"}],
        )
        assert len(result.row_scores) == 1

    def test_init_valid_params(self):
        """Test that __init__ works with valid params."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        assert metric.reference == "{{item.reference}}"
        assert metric.candidate is None
        assert metric.type.value == "rouge"

    def test_init_with_candidate(self):
        """Test that __init__ works with custom candidate template."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
            candidate="{{sample.custom_output}}",
        )

        assert metric.candidate == "{{sample.custom_output}}"

    @pytest.mark.asyncio
    async def test_compute_scores_exact_match(self):
        """Test compute_scores with exact match."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The cat sat on the mat."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        # Should return all 4 ROUGE scores
        assert len(result.outputs) == 4
        score_names = {s.name for s in result.outputs}
        assert score_names == {"rouge_1_score", "rouge_2_score", "rouge_3_score", "rouge_L_score"}
        # All scores should be high for exact match
        for score in result.outputs:
            assert score.value >= 0.9

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = ROUGEMetric(reference="{{item.reference}}")
        result = await compute_scores(metric, {"reference": "a"}, {"output_text": "a"})
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_partial_match(self):
        """Test compute_scores with partial match."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "The cat sat on the mat."}
        sample = {"output_text": "The dog sat on the floor."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 4
        # All scores should be between 0 and 1
        for score in result.outputs:
            assert 0.0 <= score.value <= 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_no_match(self):
        """Test compute_scores with no match."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        item = {"reference": "hello world"}
        sample = {"output_text": "goodbye universe"}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 4
        # All scores should be very low or 0 for no match
        for score in result.outputs:
            assert score.value < 0.1

    @pytest.mark.asyncio
    async def test_compute_scores_with_custom_candidate(self):
        """Test compute_scores with custom candidate template."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
            candidate="{{item.custom_output}}",
        )

        item = {"reference": "The cat sat on the mat.", "custom_output": "The cat sat on the mat."}
        sample = {"output_text": "This should be ignored."}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 4
        # All scores should be high for exact match
        for score in result.outputs:
            assert score.value >= 0.9

    @pytest.mark.asyncio
    async def test_compute_scores_rouge_1(self):
        """Test ROUGE-1 (unigram) scoring."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        # Simple case where we can predict ROUGE-1
        item = {"reference": "cat dog"}
        sample = {"output_text": "cat bird"}

        result = await compute_scores(metric, item, sample)

        # ROUGE-1 should detect unigram overlap (cat)
        rouge_1_score = next(s for s in result.outputs if s.name == "rouge_1_score")
        assert rouge_1_score.value > 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_rouge_2(self):
        """Test ROUGE-2 (bigram) scoring."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        # Case with bigram overlap
        item = {"reference": "the cat sat"}
        sample = {"output_text": "the cat slept"}

        result = await compute_scores(metric, item, sample)

        # ROUGE-2 should detect bigram overlap (the cat)
        rouge_2_score = next(s for s in result.outputs if s.name == "rouge_2_score")
        assert rouge_2_score.value > 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_rouge_l(self):
        """Test ROUGE-L (longest common subsequence) scoring."""
        metric = ROUGEMetric(
            reference="{{item.reference}}",
        )

        # Case with LCS
        item = {"reference": "the quick brown fox"}
        sample = {"output_text": "the brown fox"}

        result = await compute_scores(metric, item, sample)

        # ROUGE-L should detect LCS
        rouge_l_score = next(s for s in result.outputs if s.name == "rouge_L_score")
        assert rouge_l_score.value > 0.0
