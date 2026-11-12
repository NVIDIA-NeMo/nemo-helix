# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.answer_match import AnswerMatchMetric, SelectUnmatchedError
from nhx_evals_sdk.metrics.protocol import MetricOutputSpec
from nhx_evals_sdk.metrics.types import MetricsUnion
from pydantic import TypeAdapter, ValidationError


def _metric(**kwargs: Any) -> AnswerMatchMetric:
    config: dict[str, Any] = {"reference": "{{item.answer}}", "comparisons": {"exact": {"type": "equals"}}}
    config.update(kwargs)
    return AnswerMatchMetric.model_validate(config)


async def _scores(metric: AnswerMatchMetric, item: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    result = await compute_scores(metric, item, sample)
    return {output.name: output.value for output in result.outputs}


def _response(content: Any, **message: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content, **message}}]}


THINK = {"after": "</think>", "trace_starts_after": "<think>", "found_as": "closed"}


class TestExtract:
    @pytest.mark.parametrize(
        ("extract", "text", "answer", "found"),
        [
            (None, "  <think>x</think> 42 ", "<think>x</think> 42", None),
            (THINK, "<think>x</think> 42", "42", True),
            (THINK, "42", "42", False),
            # The first delimiter, or the last: a second think block stays in the answer, or is skipped.
            (THINK, "<think>a</think><think>b</think>42", "<think>b</think>42", True),
            ({**THINK, "occurrence": "last"}, "<think>a</think><think>b</think>42", "42", True),
            ({"regex": r"Answer:\s*(.+)", "found_as": "closed"}, "so\nAnswer: 42", "42", True),
            (
                {"regex": r"Answer:\s*(.+)", "occurrence": "last", "found_as": "closed"},
                "Answer: 1\nAnswer: 42",
                "42",
                True,
            ),
            ({"regex": r"Answer:\s*(?P<value>.+)", "group": "value", "found_as": "closed"}, "Answer: 42", "42", True),
            # Without a match, the answer is the whole text.
            ({"regex": r"Answer:\s*(.+)", "found_as": "closed"}, "42", "42", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_extracts_the_answer(self, extract, text, answer, found):
        metric = _metric(extract=extract)
        scores = await _scores(metric, {"answer": answer}, {"output_text": text})
        assert scores["exact"] is True
        assert scores.get("closed") is found

    @pytest.mark.asyncio
    async def test_raw_source_rewraps_the_reasoning(self):
        """With a reasoning parser, `raw` rebuilds the completion, so the trace is available and closed."""
        metric = _metric(
            extract={**THINK, "source": "raw"},
            comparisons={
                "exact": {"type": "equals"},
                "python": {"type": "parses", "language": "python", "input": "trace"},
            },
        )
        sample = {"response": _response("42", reasoning_content="x = 6 * 7"), "output_text": "42"}
        assert await _scores(metric, {"answer": "42"}, sample) == {"exact": True, "python": True, "closed": True}

    @pytest.mark.asyncio
    async def test_output_source_reads_the_output_text_only(self):
        metric = _metric(
            extract=THINK,
            comparisons={
                "exact": {"type": "equals"},
                "python": {"type": "parses", "language": "python", "input": "trace"},
            },
        )
        sample = {"response": _response("42", reasoning_content="x = ("), "output_text": "42"}
        # The reasoning isn't read: the trace is empty (which parses), and no delimiter was found.
        assert await _scores(metric, {"answer": "42"}, sample) == {"exact": True, "python": True, "closed": False}

    @pytest.mark.asyncio
    async def test_candidate_template_overrides_the_source(self):
        metric = _metric(candidate="{{item.completion}}", extract={**THINK, "source": "raw"})
        sample = {"response": _response("wrong", reasoning_content="r")}
        scores = await _scores(metric, {"answer": "42", "completion": "<think>r</think>42"}, sample)
        assert (scores["exact"], scores["closed"]) == (True, True)


class TestComparisons:
    @pytest.mark.parametrize(
        ("normalize", "answer", "reference", "expected"),
        [
            ({}, "Net sales", "net sales", False),
            ({"casefold": True}, "Net sales", "net sales", True),
            ({"casefold": True, "collapse_whitespace": True}, " net \n sales ", "net sales", True),
            ({"strip": ".!"}, "net sales.", "net sales", True),
            ({"unicode": "NFKC"}, "１２", "12", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_equals_after_normalizing(self, normalize, answer, reference, expected):
        scores = await _scores(_metric(normalize=normalize), {"answer": reference}, {"output_text": answer})
        assert scores["exact"] is expected

    @pytest.mark.parametrize(
        ("number", "answer", "reference", "expected"),
        [
            ({"relative_tolerance": 0.05}, "104", "100", True),
            ({"relative_tolerance": 0.05}, "106", "100", False),
            ({"absolute_tolerance": 0.01}, "3.149", "3.14", True),
            ({"absolute_tolerance": 0.01}, "3.2", "3.14", False),
            # Characters to ignore, and scale words.
            ({"ignore": ",$"}, "$1,000", "1000", True),
            ({}, "1,000", "1000", False),
            ({"scales": {"%": "percent", "percent": "percent"}}, "12.5 percent", "12.5%", True),
            ({"scales": {"%": "percent"}, "require_reference_scale": True}, "12.5", "12.5%", False),
            ({"scales": {"%": "percent"}}, "12.5", "12.5%", True),
            # Text that doesn't parse: false, or an equals comparison.
            ({}, "yes", "yes", False),
            ({"otherwise": "equals"}, "yes", "yes", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_number(self, number, answer, reference, expected):
        metric = _metric(comparisons={"close": {"type": "number", **number}})
        scores = await _scores(metric, {"answer": reference}, {"output_text": answer})
        assert scores["close"] is expected

    @pytest.mark.parametrize(
        ("answer", "reference", "expected"),
        [
            ("2018; 2019", "2019 | 2018", True),
            ("2018##2019##2019", "2019 | 2018", False),
            ("2018", "2019 | 2018", False),
            # The text is split before it is normalized: NFKC turns a full-width semicolon into ";" only
            # inside the span, so it stays one span.
            ("a\uff1bb", "a | b", False),
            ("a\uff1bb", "A\uff1bB", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_span_set(self, answer, reference, expected):
        metric = _metric(
            normalize={"unicode": "NFKC", "casefold": True},
            comparisons={"spans": {"type": "span-set", "separators": ["|", "##", ";"]}},
        )
        scores = await _scores(metric, {"answer": reference}, {"output_text": answer})
        assert scores["spans"] is expected

    @pytest.mark.asyncio
    async def test_token_f1_uses_the_pattern(self):
        metric = _metric(
            normalize={"casefold": True},
            comparisons={
                "words": {"type": "token-f1"},
                "alnum": {"type": "token-f1", "pattern": "[a-z0-9]+"},
            },
        )
        scores = await _scores(metric, {"answer": "net_sales revenue"}, {"output_text": "net sales"})
        assert scores["words"] == pytest.approx(0.0)
        assert scores["alnum"] == pytest.approx(2 * (2 / 2) * (2 / 3) / ((2 / 2) + (2 / 3)))

    @pytest.mark.parametrize(
        ("language", "text", "expected"),
        [
            ("python", "a = (1 + 2)", True),
            ("python", "a = (1 +", False),
            ("json", '{"a": 1}', True),
            ("json", "{a}", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_parses(self, language, text, expected):
        metric = _metric(comparisons={"valid": {"type": "parses", "language": language}})
        assert (await _scores(metric, {"answer": ""}, {"output_text": text}))["valid"] is expected


def _routed(**select: Any) -> AnswerMatchMetric:
    return _metric(
        normalize={"casefold": True},
        comparisons={
            "exact": {"type": "equals", "report": False},
            "close": {"type": "number", "absolute_tolerance": 0.01, "report": False},
            "f1": {"type": "token-f1", "report": False},
        },
        label_outputs={"kind": "{{item.kind}}"},
        select=select,
    )


class TestSelect:
    @pytest.mark.parametrize(
        ("kind", "answer", "reference", "expected"),
        [
            ("number", "3.141", "3.14", 1.0),
            ("free-text", "net sales", "net sales revenue", 0.8),
            ("label", "Yes", "yes", 1.0),
            ("label", "no", "yes", 0.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_picks_a_comparison_per_row(self, kind, answer, reference, expected):
        metric = _routed(
            correct={"on": "kind", "cases": {"number": "close", "*-text": "f1"}, "default": "exact"},
        )
        scores = await _scores(metric, {"answer": reference, "kind": kind}, {"output_text": answer})
        # A token-f1 among the cases makes the output continuous, so booleans become 1.0 or 0.0.
        assert isinstance(scores["correct"], float)
        assert scores["correct"] == pytest.approx(expected)
        assert scores["kind"] == kind

    @pytest.mark.asyncio
    async def test_the_first_matching_case_wins(self):
        metric = _routed(correct={"on": "kind", "cases": {"multi-span": "exact", "*span": "close"}})
        scores = await _scores(metric, {"answer": "1", "kind": "multi-span"}, {"output_text": "1.001"})
        assert scores["correct"] is False

    @pytest.mark.asyncio
    async def test_an_unmatched_row_omits_the_output(self):
        metric = _routed(correct={"on": "kind", "cases": {"number": "close"}})
        scores = await _scores(metric, {"answer": "1", "kind": "text"}, {"output_text": "1"})
        assert "correct" not in scores
        assert MetricOutputSpec.boolean("correct", required=False) in metric.output_spec()

    @pytest.mark.asyncio
    async def test_an_unmatched_row_fails_before_the_candidate_is_read(self):
        metric = _routed(correct={"on": "kind", "cases": {"number": "close"}, "unmatched": "error"})
        with pytest.raises(SelectUnmatchedError, match=r"select 'correct': the label kind='text' matches no case"):
            await compute_scores(metric, {"answer": "1", "kind": "text"}, {})


class TestConfig:
    def test_outputs_in_order_with_their_kinds(self):
        metric = _metric(
            extract=THINK,
            comparisons={
                "exact": {"type": "equals"},
                "f1": {"type": "token-f1"},
                "close": {"type": "number", "report": False},
            },
            label_outputs={"task": "{{item.task}}"},
            select={"correct": {"on": "task", "cases": {"a": "close"}, "default": "exact"}},
        )
        assert metric.type.value == "answer-match"
        assert output_names(metric) == ["exact", "f1", "closed", "correct", "task"]
        assert metric.output_spec() == [
            MetricOutputSpec.boolean("exact"),
            MetricOutputSpec.continuous_score("f1"),
            MetricOutputSpec.boolean("closed"),
            MetricOutputSpec.boolean("correct"),
            MetricOutputSpec.label("task"),
        ]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"extract": {}}, "exactly one of `after` or `regex`"),
            ({"extract": {"after": "x", "regex": "y"}}, "exactly one of `after` or `regex`"),
            ({"extract": {"regex": "("}}, "does not compile"),
            ({"extract": {"regex": "a(b)", "group": 2}}, "group 2 doesn't exist"),
            ({"extract": {"regex": "a(b)", "group": "name"}}, "no group named 'name'"),
            ({"extract": {"regex": "a(b)", "source": "raw"}}, "needs `after`"),
            ({"comparisons": {}}, "at least 1 item"),
            ({"comparisons": {"s": {"type": "span-set", "separators": [""]}}}, "must not be empty"),
            ({"comparisons": {"t": {"type": "token-f1", "pattern": "("}}}, "does not compile"),
            ({"comparisons": {"p": {"type": "parses", "language": "python", "input": "trace"}}}, "needs `extract`"),
            ({"comparisons": {"exact": {"type": "equals", "report": False}}}, "emits no scores"),
            ({"label_outputs": {"exact": "{{item.x}}"}}, "must be unique"),
            ({"select": {"s": {"on": "task", "cases": {"a": "exact"}}}}, "not a label output"),
            (
                {"label_outputs": {"task": "{{item.task}}"}, "select": {"s": {"on": "task", "cases": {"a": "nope"}}}},
                "comparisons that don't exist: \\['nope'\\]",
            ),
        ],
    )
    def test_rejects_inconsistent_configs(self, change, message):
        with pytest.raises(ValidationError, match=message):
            _metric(**change)

    @pytest.mark.parametrize("name", ["", "finance.qa", "-qa", "a b", "x" * 65])
    def test_rejects_names_that_cannot_key_scores(self, name):
        with pytest.raises(ValidationError, match="name"):
            _metric(name=name)

    def test_round_trips_through_metrics_union(self):
        metric = _metric(
            name="qa",
            extract={**THINK, "source": "raw"},
            comparisons={"exact": {"type": "equals"}, "close": {"type": "number", "relative_tolerance": 0.05}},
            label_outputs={"task": "{{item.task}}"},
            select={"correct": {"on": "task", "cases": {"a": "close"}, "default": "exact"}},
        )
        for dumped in (metric.model_dump(mode="json"), metric.model_dump(mode="json", exclude_unset=True)):
            restored = TypeAdapter(MetricsUnion).validate_python(dumped)
            assert isinstance(restored, AnswerMatchMetric)
            assert restored == metric

    def test_input_schema_lists_template_fields(self):
        metric = _metric(candidate="{{item.completion}}", label_outputs={"task": "{{item.task}}"})
        assert set(metric.input_schema().schema_["required"]) == {"answer", "task", "completion"}

    def test_the_name_keys_the_scores(self):
        dataset = [{"answer": "Paris", "output": "paris"}]
        named = _metric(name="capital", candidate="{{item.output}}", normalize={"casefold": True})
        unnamed = _metric(candidate="{{item.output}}")

        benchmark = Evaluator().run_sync(metrics=[named, unnamed, named], dataset=dataset)
        assert list(benchmark.row_scores[0].metrics) == ["capital", "answer-match", "capital_2"]
        means = {score.name: score.mean for score in benchmark.aggregate_scores.scores}
        assert means == {"capital.exact": 1.0, "answer-match.exact": 0.0, "capital_2.exact": 1.0}

    def test_run_sync_aggregates_and_keeps_labels_out(self):
        metric = _routed(correct={"on": "kind", "cases": {"number": "close"}})
        result = Evaluator().run_sync(
            metrics=[metric.model_copy(update={"candidate": "{{item.output}}"})],
            dataset=[
                {"kind": "number", "answer": "3.14", "output": "3.141"},
                {"kind": "number", "answer": "3.14", "output": "3.2"},
                {"kind": "text", "answer": "pi", "output": "pi"},
            ],
        )
        aggregates = {score.name: score for score in result.aggregate_scores.scores}
        correct = aggregates["answer-match.correct"]
        assert (correct.mean, correct.count, correct.nan_count) == (0.5, 2, 1)
        assert "answer-match.kind" not in aggregates
