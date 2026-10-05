# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from jinja2 import UndefinedError
from nemo_evaluator_sdk import Evaluator
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput
from nemo_evaluator_sdk.values.results import (
    AggregatedMetricResult,
    AggregateRangeScore,
    AggregateRubricScore,
    EvaluationResult,
    Histogram,
    Percentiles,
    RowScore,
    RubricScoreStat,
    ScoreStats,
)
from pytest_mock import MockerFixture


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_range_score(name: str, *, count: int, nan_count: int = 0) -> AggregateRangeScore:
    return AggregateRangeScore(
        name=name,
        count=count,
        nan_count=nan_count,
        sum=1.0,
        mean=0.5,
        min=0.0,
        max=1.0,
        std_dev=0.1,
        variance=0.01,
        percentiles=Percentiles(
            p10=0.1,
            p20=0.2,
            p30=0.3,
            p40=0.4,
            p50=0.5,
            p60=0.6,
            p70=0.7,
            p80=0.8,
            p90=0.9,
            p100=1.0,
        ),
        histogram=Histogram(bins=[]),
    )


def _metric_input(item: dict[str, str], sample: dict[str, str]) -> MetricInput:
    return build_metric_input(item, sample, 0)


class TestExactMatchMetric:
    def test_top_level_exports_include_evaluators(self):
        assert Evaluator is not None

    def test_evaluate_with_inline_rows(self):
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {"expected": "blue", "model_output": "Blue"},
                {"expected": "Jupiter", "model_output": "Saturn"},
            ],
        )

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].mean == 0.5
        assert result.row_scores[0].sample == {}

    def test_evaluate_with_file_path(self, tmp_path: Path):
        dataset_path = tmp_path / "eval.jsonl"
        _write_jsonl(
            dataset_path,
            [
                {"expected": "4", "prediction": "4"},
                {"expected": "10", "prediction": "11"},
            ],
        )

        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(metrics=[metric], dataset=dataset_path)

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].count == 2

    def test_evaluate_with_file_path_that_contains_glob_metacharacters(self, tmp_path: Path):
        dataset_path = tmp_path / "eval[1].jsonl"
        _write_jsonl(dataset_path, [{"expected": "4", "prediction": "4"}])

        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(metrics=[metric], dataset=dataset_path)

        assert len(result.row_scores) == 1
        assert result.aggregate_scores.scores[0].mean == 1.0

    def test_evaluate_ignores_other_files_in_directory(self, tmp_path: Path):
        _write_jsonl(tmp_path / "train.jsonl", [{"expected": "4", "prediction": "4"}])
        _write_jsonl(tmp_path / "ignored.jsonl", [{"expected": "10", "prediction": "11"}])

        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(metrics=[metric], dataset=tmp_path / "train.jsonl")

        assert len(result.row_scores) == 1
        assert result.aggregate_scores.scores[0].mean == 1.0

    def test_evaluate_with_glob_path(self, tmp_path: Path):
        _write_jsonl(tmp_path / "train.jsonl", [{"expected": "4", "prediction": "4"}])
        _write_jsonl(tmp_path / "validation.jsonl", [{"expected": "Paris", "prediction": "Paris"}])
        _write_jsonl(tmp_path / "ignored.csv", [{"expected": "10", "prediction": "11"}])

        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(metrics=[metric], dataset=tmp_path / "*.jsonl")

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].mean == 1.0

    def test_evaluate_with_nested_glob_path(self, tmp_path: Path):
        _write_jsonl(tmp_path / "splits" / "train.jsonl", [{"expected": "4", "prediction": "4"}])
        _write_jsonl(
            tmp_path / "splits" / "nested" / "validation.jsonl", [{"expected": "Paris", "prediction": "Paris"}]
        )

        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}")
        result = Evaluator().run_sync(metrics=[metric], dataset=tmp_path / "splits" / "**" / "*.jsonl")

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].mean == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_async_matches_sync_behavior(self):
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")
        result = await Evaluator().run(
            metrics=[metric],
            dataset=[
                {"expected": "blue", "model_output": "Blue"},
                {"expected": "Jupiter", "model_output": "Saturn"},
            ],
        )

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].mean == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_runs_inside_active_event_loop(self):
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {"expected": "blue", "model_output": "Blue"},
                {"expected": "Jupiter", "model_output": "Saturn"},
            ],
        )

        assert len(result.row_scores) == 2
        assert result.aggregate_scores.scores[0].mean == 0.5

    @pytest.mark.asyncio
    async def test_default_candidate_uses_output_text(self):
        metric = ExactMatchMetric(reference="{{item.expected}}")
        match = await metric.compute_scores(_metric_input({"expected": "blue"}, {"output_text": "Blue"}))
        mismatch = await metric.compute_scores(_metric_input({"expected": "Jupiter"}, {"output_text": "Saturn"}))

        assert match.outputs[0].value == 1.0
        assert mismatch.outputs[0].value == 0.0


class TestComputeScores:
    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = ExactMatchMetric(reference="{{item.reference}}", candidate="{{sample.output_text}}")

        with pytest.raises(ValueError) as exc_info:
            await metric.compute_scores(
                _metric_input({"prompt": "What is the capital of France?"}, {"output_text": "Paris"})
            )

        assert str(exc_info.value) == (
            "ExactMatchMetric(reference='{{item.reference}}', candidate='{{sample.output_text}}') "
            "could not render its 'reference' template for this row.\n"
            "Available item keys=['prompt']. \n"
            "Available sample keys=['output_text'].\n"
            "Dataset item has missing_key='reference' but the 'reference' template references it.\n"
            "If a field_mapping supplies this field, its path did not resolve for this row; check the "
            "mapping rather than adding a dataset column of this name, which would be scored in its place."
        )

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_candidate_field(self):
        metric = ExactMatchMetric(reference="{{item.reference}}", candidate="{{sample.prediction}}")

        with pytest.raises(ValueError) as exc_info:
            await metric.compute_scores(_metric_input({"reference": "Paris"}, {"output_text": "Paris"}))

        assert str(exc_info.value) == (
            "ExactMatchMetric(reference='{{item.reference}}', candidate='{{sample.prediction}}') "
            "could not render its 'candidate' template for this row.\n"
            "Available item keys=['reference']. \n"
            "Available sample keys=['output_text'].\n"
            "Dataset item has missing_key='prediction' but the 'candidate' template references it.\n"
            "If a field_mapping supplies this field, its path did not resolve for this row; check the "
            "mapping rather than adding a dataset column of this name, which would be scored in its place."
        )

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_bare_reference_name(self):
        metric = ExactMatchMetric(reference="{{reference}}", candidate="{{sample.output_text}}")

        with pytest.raises(ValueError) as exc_info:
            await metric.compute_scores(
                _metric_input({"prompt": "What is the capital of France?"}, {"output_text": "Paris"})
            )

        assert str(exc_info.value) == (
            "ExactMatchMetric(reference='{{reference}}', candidate='{{sample.output_text}}') "
            "could not render its 'reference' template for this row.\n"
            "Available item keys=['prompt']. \n"
            "Available sample keys=['output_text'].\n"
            "Dataset item has missing_key='reference' but the 'reference' template references it.\n"
            "If a field_mapping supplies this field, its path did not resolve for this row; check the "
            "mapping rather than adding a dataset column of this name, which would be scored in its place."
        )

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_default_candidate_missing_output_text(self):
        metric = ExactMatchMetric(reference="{{item.reference}}")

        with pytest.raises(ValueError) as exc_info:
            await metric.compute_scores(_metric_input({"reference": "Paris"}, {"some_other_field": "Paris"}))

        assert str(exc_info.value) == (
            "ExactMatchMetric has missing `candidate` field.\n"
            "For offline evaluation, `candidate=...` field is required when constructing ExactMatchMetric.\n"
            "For online evaluation, this usually means the evaluated model produced no output."
        )

    @pytest.mark.asyncio
    async def test_includes_original_jinja_error_when_missing_key_cannot_be_inferred(self, mocker: MockerFixture):
        metric = ExactMatchMetric(reference="{{item.reference}}", candidate="{{sample.output_text}}")
        mocker.patch(
            "nemo_evaluator_sdk.metrics.template_rendering.render_template",
            side_effect=UndefinedError("custom undefined message"),
        )

        with pytest.raises(ValueError, match=r"jinja_error='custom undefined message'"):
            await metric.compute_scores(
                _metric_input({"prompt": "What is the capital of France?"}, {"output_text": "Paris"})
            )


class TestOfflineEvaluationResult:
    @pytest.fixture
    def result(self):
        metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.model_output}}")
        return Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {"expected": "blue", "model_output": "Blue"},
                {"expected": "Jupiter", "model_output": "Saturn"},
            ],
        )

    def test_to_records_rows(self, result):
        records = result.to_records(view="rows")
        assert len(records) == 2
        assert records[0]["output.exact-match.exact-match"] == 1.0
        assert records[1]["output.exact-match.exact-match"] == 0.0
        assert records[0]["item.expected"] == "blue"
        assert records[1]["item.expected"] == "Jupiter"
        assert records[0]["item.model_output"] == "Blue"
        assert "sample.model_output" not in records[0]

    def test_to_records_aggregate(self, result):
        records = result.to_records(view="aggregate")
        assert len(records) == 1
        assert records[0]["name"] == "exact-match.exact-match"
        assert records[0]["score_type"] == "range"
        assert records[0]["percentiles.p50"] == 0.5

    def test_to_table_returns_pyarrow_table(self, result):
        table = result.to_table(view="rows")
        assert table.num_rows == 2
        assert "output.exact-match.exact-match" in table.column_names

    def test_to_pandas_returns_dataframe_when_available(self, result):
        pd = pytest.importorskip("pandas")
        dataframe = result.to_pandas(view="rows")
        assert isinstance(dataframe, pd.DataFrame)
        assert "output.exact-match.exact-match" in dataframe.columns

    def test_format_summary_and_str(self, result):
        formatted = result.format_summary(max_rows=1)
        assert "BenchmarkEvaluationResult(rows=2, aggregate_scores=1, ok=2)" in formatted
        assert "Aggregate scores" in formatted
        assert "Row preview for metric 'exact-match' (first 1 of 2)" in formatted
        assert "Error details" not in formatted
        assert str(result).startswith("BenchmarkEvaluationResult(rows=2, aggregate_scores=1, ok=2)")

    def test_to_records_rows_includes_error_status_and_response_payload(self):
        result = EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=None,
                    item={"prompt": "hello"},
                    sample={"response": {"id": "r1"}},
                    metrics={},
                    requests=[],
                    metric_errors={"judge": "bad output"},
                )
            ],
            aggregate_scores=AggregatedMetricResult(scores=[]),
        )

        records = result.to_records(view="rows")

        assert records == [
            {
                "row_index": 0,
                "status": "error",
                "item.prompt": "hello",
                "sample.response.id": "r1",
                "error": "judge: bad output",
            }
        ]

    def test_to_records_rows_marks_scored_rows_with_errors_as_error(self):
        result = EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=3,
                    item={"prompt": "hello"},
                    sample={"output_text": "world"},
                    metrics={"judge": [MetricOutput(name="judge", value=0.5)]},
                    requests=[],
                    metric_errors={"judge": "bad but scored"},
                )
            ],
            aggregate_scores=AggregatedMetricResult(scores=[]),
        )

        records = result.to_records(view="rows")

        assert records == [
            {
                "row_index": 3,
                "status": "error",
                "item.prompt": "hello",
                "sample.output_text": "world",
                "error": "judge: bad but scored",
                "output.judge": 0.5,
            }
        ]

    def test_to_records_aggregate_includes_mode_category_and_histogram_json(self):
        result = EvaluationResult(
            row_scores=[],
            aggregate_scores=AggregatedMetricResult(
                scores=[
                    AggregateRubricScore(
                        name="judge",
                        count=2,
                        nan_count=0,
                        sum=3.0,
                        mean=1.5,
                        min=1.0,
                        max=2.0,
                        std_dev=0.5,
                        variance=0.25,
                        rubric_distribution=[
                            RubricScoreStat(label="good", value=2, count=1),
                            RubricScoreStat(label="ok", value=1, count=1),
                        ],
                        mode_category="good",
                    )
                ]
            ),
        )

        records = result.to_records(view="aggregate")

        assert records == [
            {
                "name": "judge",
                "count": 2,
                "nan_count": 0,
                "sum": 3.0,
                "mean": 1.5,
                "min": 1.0,
                "max": 2.0,
                "median": None,
                "std_dev": 0.5,
                "variance": 0.25,
                "sample_std_dev": None,
                "sample_variance": None,
                "score_type": "rubric",
                "rubric_distribution": [
                    {"label": "good", "description": None, "value": 2, "count": 1},
                    {"label": "ok", "description": None, "value": 1, "count": 1},
                ],
                "mode_category": "good",
            }
        ]

    def test_to_records_rejects_unknown_view(self, result):
        with pytest.raises(ValueError, match="Unsupported view 'bad'"):
            result.to_records(view="bad")

    def test_format_summary_handles_scored_and_unscored_error_rows(self):
        result = EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=3,
                    item={"prompt": "first"},
                    sample={"response": {"id": "hidden"}, "output_text": "text"},
                    metrics={"judge": [MetricOutput(name="judge", value=0.5)]},
                    requests=[],
                    metric_errors={"judge": "bad but scored"},
                ),
                RowScore(
                    row_index=4,
                    item={"prompt": "second"},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors={"judge": "totally failed"},
                ),
            ],
            aggregate_scores=AggregatedMetricResult(scores=[_make_range_score("judge", count=1, nan_count=1)]),
        )

        formatted = result.format_summary(max_rows=1, max_error_rows=1)

        assert "EvaluationResult(rows=2, aggregate_scores=1, error=2)" in formatted
        assert "row_index | status" in formatted
        assert "3         | error" in formatted
        assert "sample.response" not in formatted
        assert "Error details (1 of 2 failed rows)" in formatted
        assert "[row 3]" in formatted
        assert "bad but scored" in formatted
        assert "... 1 more failed rows omitted" in formatted

    def test_print_summary_writes_formatted_output(self, result, capsys: pytest.CaptureFixture[str]):
        result.print_summary(max_rows=1)

        captured = capsys.readouterr()
        assert "EvaluationResult(rows=2, aggregate_scores=1, ok=2)" in captured.out

    def test_format_summary_without_preview_or_aggregates(self):
        result = EvaluationResult(
            row_scores=[
                RowScore(
                    row_index=0,
                    item={"prompt": "hello"},
                    sample={},
                    metrics={},
                    requests=[],
                    metric_errors={"judge": "failed"},
                )
            ],
            aggregate_scores=AggregatedMetricResult(scores=[]),
        )

        formatted = result.format_summary(max_rows=0, max_error_rows=1)

        assert "Aggregate scores" in formatted
        assert "(no rows)" in formatted
        assert "Row preview" not in formatted
        assert "Error details (1 of 1 failed rows)" in formatted


def test_score_stats_serializes_every_nan_statistic_as_a_string() -> None:
    """NaN is not valid JSON, so ScoreStats renders it as "NaN" — including the sample statistics.

    sample_variance/sample_stddev were added alongside the population pair; leaving them off the
    serializer would emit a bare NaN token where every sibling stat emits a quoted string.
    """
    nan = float("nan")
    stats = ScoreStats(
        count=1,
        sum=nan,
        sum_squared=nan,
        min=nan,
        max=nan,
        mean=nan,
        variance=nan,
        stddev=nan,
        sample_variance=nan,
        sample_stddev=nan,
        stderr=nan,
    )

    payload = stats.model_dump()
    counts_and_lists = {"count", "nan_count", "rubric_distribution"}
    numeric = {key: value for key, value in payload.items() if key not in counts_and_lists}
    assert numeric == dict.fromkeys(numeric, "NaN"), f"unserialized NaN in: {numeric}"

    # And the result is actually JSON-encodable, which a bare NaN would not be under strict mode.
    json.dumps(payload, allow_nan=False)
