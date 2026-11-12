# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import random
from typing import Any

import pytest
from metrics import finance_qa_cookbook as cookbook
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics import finance_qa
from nhx_evals_sdk.metrics.finance_qa import FinanceQAMetric
from nhx_evals_sdk.metrics.types import MetricsUnion
from pydantic import TypeAdapter

TASKS = {
    "TAT-DQA/span": "visual_ocr",
    "TAT-DQA/multi-span": "visual_ocr",
    "TAT-DQA/arithmetic": "financial_reasoning",
    "TAT-DQA/count": "financial_reasoning",
    "CodeFinQA": "program_synthesis",
    "CodeTAT-QA": "program_synthesis",
}
INSTRUCTION = (
    "Return a grounded reasoning trace inside <think>...</think>, followed by only the final answer. "
    "Preserve the requested financial unit or scale."
)


def _metric(**kwargs: Any) -> FinanceQAMetric:
    return FinanceQAMetric(reference="{{item.answer}}", task="{{item.task}}", **kwargs)


def _response(content: Any, completion_tokens: int | None = None, **message: Any) -> dict[str, Any]:
    response: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": content, **message}}]}
    if completion_tokens is not None:
        response["usage"] = {"completion_tokens": completion_tokens}
    return response


async def _scores(metric: FinanceQAMetric, item: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    result = await compute_scores(metric, item, sample)
    return {output.name: output.value for output in result.outputs}


class TestTaskScoring:
    @pytest.mark.parametrize(
        ("task", "prediction", "reference", "task_correct", "answer_f1"),
        [
            # Span: a normalized exact match, with token F1 for partial credit.
            ("TAT-DQA/span", "Net Sales.", "net sales", True, 1.0),
            ("TAT-DQA/span", "net sales revenue", "net sales", False, 0.8),
            # Multi-span: the span lists in any order, split on |, ## or ;.
            ("TAT-DQA/multi-span", "2019 | 2018", "2018; 2019", True, 1.0),
            ("TAT-DQA/multi-span", "2019##2018##2017", "2018 | 2019", False, 0.8),
            # Arithmetic and count: numbers within 5% relative error, with matching scale.
            ("TAT-DQA/arithmetic", "$1,050", "1000", True, 1.0),
            ("TAT-DQA/arithmetic", "1,051", "1000", False, 0.0),
            ("TAT-DQA/arithmetic", "12.5%", "12.4 percent", True, 1.0),
            ("TAT-DQA/arithmetic", "12.5", "12.5%", False, 0.0),
            ("TAT-DQA/arithmetic", "12.5%", "12.5", True, 1.0),
            ("TAT-DQA/count", "3", "3", True, 1.0),
            ("CodeFinQA", "-0.25 million", "-0.24 million", True, 1.0),
            # Non-numeric answers fall back to a normalized exact match.
            ("CodeTAT-QA", "Yes", "yes.", True, 1.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_hand_written_cases(self, task, prediction, reference, task_correct, answer_f1):
        scores = await _scores(_metric(), {"answer": reference, "task": task}, {"output_text": prediction})
        assert scores["task_correct"] is task_correct
        assert scores["answer_f1"] == pytest.approx(answer_f1)

    @pytest.mark.asyncio
    async def test_matches_the_cookbook_on_generated_rows(self):
        """Each task is scored with its cookbook token limit, as one job per capability does."""
        rng = random.Random(1087)
        metrics = {limit: _metric(max_new_tokens=limit) for limit in (256, 512)}
        for _ in range(600):
            task = rng.choice(list(TASKS))
            reference, prediction = _answer_pair(rng, task)
            response = _random_response(rng, prediction)
            expected = cookbook.cookbook_row(
                task=task,
                task_family=TASKS[task],
                answer=reference,
                response=response,
                task_limit=_task_limit(task),
                tolerance=0.05,
            )
            scores = await _scores(
                metrics[_task_limit(task)], {"answer": reference, "task": task}, {"response": response}
            )
            context = f"task={task!r} reference={reference!r} response={response!r}"
            for name in ("exact", "relaxed", "task_correct", "answer_f1", "trace_closed"):
                assert scores[name] == expected[name], f"{name}: {context}"
            assert scores.get("program_syntax_valid") == expected["program_syntax_valid"], context
            if "usage" in response:
                assert scores["hit_max_new_tokens"] == expected["hit_max_new_tokens"], context
            else:
                # The cookbook counts a response without usage as under the limit; the metric omits the score.
                assert "hit_max_new_tokens" not in scores, context
                assert expected["hit_max_new_tokens"] is False
            assert scores["task"] == task

    @pytest.mark.parametrize(
        ("task", "reference", "response"),
        [
            # A second think block in the content: the cookbook keeps it in the answer, with or without a parser.
            ("TAT-DQA/count", "3", _response("<think>\nThree rows.\n</think>\n3", 120, reasoning_content="Count.")),
            ("TAT-DQA/count", "3", _response("<think>a</think>\n<think>b</think>\n3", 120)),
            # Full-width separators: the cookbook splits spans before NFKC turns them into ASCII.
            ("TAT-DQA/multi-span", "2019 | 2018", _response("2018\uff1b2019", 50, reasoning_content="r")),
            ("TAT-DQA/multi-span", "2018\uff1b2019", _response("2018\uff1b2019", 50, reasoning_content="r")),
            ("TAT-DQA/multi-span", "2018 ; 2019", _response("2019 \uff5c 2018", 50, reasoning_content="r")),
            # A zero reference: the tolerance has a 1e-6 floor.
            ("TAT-DQA/arithmetic", "0", _response("0.0000005", 50, reasoning_content="r")),
            ("TAT-DQA/arithmetic", "0", _response("0.00001", 50, reasoning_content="r")),
            ("CodeFinQA", "0.0", _response("-0.0000009", 50, reasoning_content="answer = 0")),
        ],
    )
    @pytest.mark.asyncio
    async def test_matches_the_cookbook_on_edge_rows(self, task, reference, response):
        """Rows the generator doesn't produce, where a wrong step order or constant would score differently."""
        expected = cookbook.cookbook_row(
            task=task,
            task_family=TASKS[task],
            answer=reference,
            response=response,
            task_limit=_task_limit(task),
            tolerance=0.05,
        )
        scores = await _scores(
            _metric(max_new_tokens=_task_limit(task)), {"answer": reference, "task": task}, {"response": response}
        )
        assert {name: scores.get(name) for name in expected} == expected


class TestCompletionParsing:
    @pytest.mark.parametrize(
        ("response", "answer", "trace_closed"),
        [
            (_response("42", reasoning_content="x = 6 * 7"), "42", True),
            (_response("42", reasoning="x = 6 * 7"), "42", True),
            (_response("x = 6 * 7\n</think>\n42"), "42", True),
            (_response("<think>x = 6 * 7</think>42"), "42", True),
            (_response([{"type": "text", "text": "4"}, {"type": "text", "text": "2"}]), "42", False),
            (_response("42"), "42", False),
            (_response(None, reasoning_content="x = 6 *"), "", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_scores_the_text_after_think(self, response, answer, trace_closed):
        scores = await _scores(_metric(), {"answer": "42", "task": "TAT-DQA/count"}, {"response": response})
        assert scores["exact"] is (answer == "42")
        assert scores["trace_closed"] is trace_closed

    @pytest.mark.asyncio
    async def test_a_truncated_trace_is_scored_not_rejected(self):
        """Without ``</think>``, the whole output is the prediction and the trace is open."""
        scores = await _scores(
            _metric(max_new_tokens=256),
            {"answer": "42", "task": "TAT-DQA/arithmetic"},
            {"response": _response("<think>First, revenue grew by", completion_tokens=256)},
        )
        assert scores["exact"] is False
        assert scores["trace_closed"] is False
        assert scores["hit_max_new_tokens"] is True

    @pytest.mark.asyncio
    async def test_reasoning_content_wins_over_reasoning(self):
        response = _response("1", reasoning_content="a = 1", reasoning="not python (")
        scores = await _scores(_metric(), {"answer": "1", "task": "CodeFinQA"}, {"response": response})
        assert scores["program_syntax_valid"] is True

    @pytest.mark.asyncio
    async def test_output_text_is_the_raw_completion_without_a_response(self):
        scores = await _scores(
            _metric(), {"answer": "42", "task": "TAT-DQA/count"}, {"output_text": "<think>6 * 7</think> 42"}
        )
        assert (scores["exact"], scores["trace_closed"]) == (True, True)

    @pytest.mark.asyncio
    async def test_candidate_template_overrides_the_response(self):
        metric = _metric(candidate="{{item.completion}}")
        scores = await _scores(
            metric,
            {"answer": "42", "task": "TAT-DQA/count", "completion": "</think>42"},
            {"response": _response("7")},
        )
        assert scores["exact"] is True

    @pytest.mark.asyncio
    async def test_raises_clear_error_without_a_completion(self):
        with pytest.raises(ValueError, match="FinanceQAMetric has missing `candidate` field"):
            await compute_scores(_metric(), {"answer": "1", "task": "TAT-DQA/count"}, {})

    @pytest.mark.asyncio
    async def test_end_token_reasoning_is_wrapped_once(self):
        """``reasoning.end_token`` moves ``<think>...</think>`` into reasoning_content; the trace still closes once."""
        response = _response("0.25", reasoning_content="<think>answer = 30 / 120</think>")
        scores = await _scores(_metric(), {"answer": "0.25", "task": "CodeFinQA"}, {"response": response})
        assert (scores["exact"], scores["program_syntax_valid"], scores["trace_closed"]) == (True, True, True)


class TestOptionalScores:
    @pytest.mark.parametrize(
        ("task", "reasoning", "expected"),
        [
            ("CodeFinQA", "revenue = 120\ncost = 80\nanswer = revenue - cost", True),
            ("CodeTAT-QA", "answer = (120 - 80", False),
            ("CodeTAT-QA", "answer = 1\x00", False),
            ("TAT-DQA/arithmetic", "answer = 120 - 80", None),
        ],
    )
    @pytest.mark.asyncio
    async def test_program_syntax_only_on_program_synthesis_tasks(self, task, reasoning, expected):
        response = _response("40", reasoning_content=reasoning)
        scores = await _scores(_metric(), {"answer": "40", "task": task}, {"response": response})
        assert scores.get("program_syntax_valid") is expected

    @pytest.mark.parametrize("error", [MemoryError("Parser stack overflowed"), RecursionError("too deep")])
    @pytest.mark.asyncio
    async def test_a_trace_beyond_the_parser_limits_is_invalid(self, monkeypatch, error):
        """Deep nesting makes ast.parse raise MemoryError or RecursionError; the depth that does varies by Python."""

        def parse(source: str) -> None:
            raise error

        monkeypatch.setattr(finance_qa, "parse", parse)
        response = _response("40", reasoning_content="answer = " + "(" * 50 + "40" + ")" * 50)
        scores = await _scores(_metric(), {"answer": "40", "task": "CodeFinQA"}, {"response": response})
        assert (scores["program_syntax_valid"], scores["task_correct"]) == (False, True)

    @pytest.mark.parametrize(
        ("max_new_tokens", "completion_tokens", "expected"),
        [(512, 511, False), (512, 512, True), (256, 300, True), (None, 512, None), (512, None, None)],
    )
    @pytest.mark.asyncio
    async def test_hit_max_new_tokens(self, max_new_tokens, completion_tokens, expected):
        scores = await _scores(
            _metric(max_new_tokens=max_new_tokens),
            {"answer": "1", "task": "TAT-DQA/count"},
            {"response": _response("1", completion_tokens=completion_tokens)},
        )
        assert scores.get("hit_max_new_tokens") is expected

    @pytest.mark.parametrize(("completion_tokens", "expected"), [(None, False), (512.0, True), (True, None)])
    @pytest.mark.asyncio
    async def test_usage_is_read_as_the_cookbook_reads_it(self, completion_tokens, expected):
        """A null count is 0, a float count is truncated, and a bool is not a count."""
        response = _response("1")
        response["usage"] = {"completion_tokens": completion_tokens}
        scores = await _scores(
            _metric(max_new_tokens=512), {"answer": "1", "task": "TAT-DQA/count"}, {"response": response}
        )
        assert scores.get("hit_max_new_tokens") is expected


class TestRows:
    """The importer's rows: the task is the prompt's first line, and the answer is the assistant turn."""

    @pytest.mark.asyncio
    async def test_task_template_reads_an_image_row(self):
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "images/page-1.png"}},
                        {"type": "text", "text": f"Task: TAT-DQA/multi-span\n{INSTRUCTION}\nQuestion: Which years?"},
                    ],
                },
                {"role": "assistant", "content": "2019 | 2018"},
            ]
        }
        metric = FinanceQAMetric(
            reference="{{ item.messages[-1].content }}",
            task="{{ item.messages[0].content[1].text.splitlines()[0].removeprefix('Task: ') }}",
        )
        scores = await _scores(metric, row, {"response": _response("2018 ## 2019", reasoning_content="Both.")})
        assert scores["task"] == "TAT-DQA/multi-span"
        assert (scores["exact"], scores["task_correct"]) == (False, True)

    @pytest.mark.asyncio
    async def test_task_template_reads_a_text_row(self):
        row = {
            "messages": [
                {"role": "user", "content": f"Task: CodeFinQA\n{INSTRUCTION}\n\nContext:\n...\n\nQuestion:\nWhat?"},
                {"role": "assistant", "content": "0.25"},
            ]
        }
        metric = FinanceQAMetric(
            reference="{{ item.messages[-1].content }}",
            task="{{ item.messages[0].content.splitlines()[0].removeprefix('Task: ') }}",
        )
        scores = await _scores(metric, row, {"response": _response("25%", reasoning_content="answer = 1 / 4")})
        assert scores["task"] == "CodeFinQA"
        assert scores["program_syntax_valid"] is True
        assert scores["relaxed"] is False  # The reference has no scale, so 25% parses as 25.

    @pytest.mark.asyncio
    async def test_numeric_reference_is_read_as_text(self):
        scores = await _scores(_metric(), {"answer": 1000, "task": "TAT-DQA/arithmetic"}, {"output_text": "1,000"})
        assert scores["exact"] is False
        assert scores["relaxed"] is True

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_task_field(self):
        with pytest.raises(ValueError) as exc_info:
            await compute_scores(_metric(), {"answer": "1"}, {"output_text": "1"})
        assert "could not render its 'task' template for this row" in str(exc_info.value)

    @pytest.mark.parametrize("task", ["Task: CodeFinQA", "TAT-DQA/Span", "TAT-DQA/span ", "CodeFinQA\n"])
    @pytest.mark.asyncio
    async def test_raises_for_an_unknown_task(self, task):
        """A task template that renders the wrong text fails instead of changing how the row is scored."""
        with pytest.raises(ValueError, match="which is not a TAT-DQA task"):
            await compute_scores(_metric(), {"answer": "1", "task": task}, {"output_text": "1"})

    @pytest.mark.asyncio
    async def test_custom_program_synthesis_tasks_are_known(self):
        metric = _metric(program_synthesis_tasks=["FinCode"])
        scores = await _scores(metric, {"answer": "1", "task": "FinCode"}, {"output_text": "<think>a = 1</think>1"})
        assert scores["program_syntax_valid"] is True


class TestConfig:
    def test_score_names(self):
        metric = _metric()
        assert metric.type.value == "finance-qa"
        assert output_names(metric) == [
            "exact",
            "relaxed",
            "task_correct",
            "answer_f1",
            "trace_closed",
            "program_syntax_valid",
            "task",
        ]
        assert "hit_max_new_tokens" in output_names(_metric(max_new_tokens=512))

    def test_round_trips_through_metrics_union(self):
        metric = _metric(candidate="{{item.completion}}", tolerance=0.01, max_new_tokens=512)
        restored = TypeAdapter(MetricsUnion).validate_python(metric.model_dump(mode="json"))
        assert isinstance(restored, FinanceQAMetric)
        assert restored == metric

    def test_input_schema_lists_template_fields(self):
        metric = _metric(candidate="{{item.completion}}")
        assert set(metric.input_schema().schema_["required"]) == {"answer", "task", "completion"}

    def test_run_sync_aggregates(self):
        metric = _metric(candidate="{{item.completion}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {"task": "CodeFinQA", "answer": "40", "completion": "<think>a = 120 - 80</think>40"},
                {"task": "CodeFinQA", "answer": "40", "completion": "<think>a = (</think>50"},
                {"task": "TAT-DQA/span", "answer": "net sales", "completion": "Net sales"},
            ],
        )
        rows = [{output.name: output.value for output in row.metrics["finance-qa"]} for row in result.row_scores]
        assert [row["task"] for row in rows] == [
            "CodeFinQA",
            "CodeFinQA",
            "TAT-DQA/span",
        ]
        aggregates = {score.name: score for score in result.aggregate_scores.scores}
        assert aggregates["finance-qa.task_correct"].mean == pytest.approx(2 / 3)
        assert aggregates["finance-qa.trace_closed"].sum == 2
        syntax = aggregates["finance-qa.program_syntax_valid"]
        assert (syntax.mean, syntax.count, syntax.nan_count) == (0.5, 2, 1)
        assert "finance-qa.task" not in aggregates
        assert "finance-qa.hit_max_new_tokens" not in aggregates


def _task_limit(task: str) -> int:
    """The cookbook's per-task completion limit (evaluation.py lines 428-431, with its default 768)."""
    return min(768, 256 if task.endswith(("arithmetic", "count")) else 512)


def _number(rng: random.Random) -> float:
    return round(rng.uniform(-5000, 5000), rng.choice([0, 1, 2, 3]))


def _format_number(rng: random.Random, value: float, scale: str) -> str:
    text = rng.choice([f"{value:,}", f"{value}", f"{value:.2f}", f"{value:,.1f}"])
    if rng.random() < 0.2:
        text = f"${text}" if not text.startswith("-") else f"-${text[1:]}"
    if rng.random() < 0.1:
        text = text.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    if scale:
        text = f"{text}{rng.choice(['', ' '])}{scale}"
    return rng.choice([text, f"{text}.", f" {text} ", text.upper()])


def _answer_pair(rng: random.Random, task: str) -> tuple[str, str]:
    """A reference and a prediction that is often right, often close, and sometimes unrelated."""
    words = ["revenue", "Net Sales", "cost of goods", "2019", "2018", "EBITDA", "the total", "Q4"]
    if task.endswith("span"):
        count = rng.randint(2, 3) if task.endswith("multi-span") else 1
        spans = rng.sample(words, count)
        reference = rng.choice([" | ", "##", "; "]).join(spans)
        predicted = rng.sample(spans, len(spans)) if rng.random() < 0.6 else rng.sample(words, count)
        if rng.random() < 0.3:
            predicted = [span.upper() for span in predicted]
        return reference, rng.choice([" | ", "##", ";", ", "]).join(predicted)
    if rng.random() < 0.1:
        word = rng.choice(words)
        return word, rng.choice([word, word.lower() + ".", f"{word}?!", rng.choice(words)])
    value = _number(rng)
    scale = rng.choice(["", "", "%", "percent", "thousand", "million", "billion"])
    predicted = value * rng.choice([1, 1, 1.03, 0.96, 1.051, 0.94, 10])
    predicted_scale = scale if rng.random() < 0.8 else rng.choice(["", "%", "million"])
    return _format_number(rng, value, scale), _format_number(rng, round(predicted, 4), predicted_scale)


def _random_response(rng: random.Random, prediction: str) -> dict[str, Any]:
    reasoning = rng.choice(["revenue = 120\nanswer = revenue * 2", "answer = (120", "Read the table.", ""])
    tokens = rng.choice([None, 100, 255, 256, 300, 511, 512])
    form = rng.randrange(6)
    if form == 0:
        return _response(prediction, tokens, reasoning_content=reasoning)
    if form == 1:
        return _response(prediction, tokens, reasoning=reasoning)
    if form == 2:
        return _response(f"{reasoning}\n</think>\n{prediction}", tokens)
    if form == 3:
        return _response(f"<think>\n{reasoning}\n</think>\n\n{prediction}", tokens)
    if form == 4:
        return _response(f"<think>\n{reasoning} and then", tokens)
    return _response([{"type": "text", "text": prediction}], tokens)


def test_generated_rows_cover_every_outcome():
    """The generator above reaches both outcomes of every boolean score, so the comparison is not vacuous."""
    rng = random.Random(1087)
    seen: dict[str, set[Any]] = {}
    for _ in range(600):
        task = rng.choice(list(TASKS))
        reference, prediction = _answer_pair(rng, task)
        row = cookbook.cookbook_row(
            task=task,
            task_family=TASKS[task],
            answer=reference,
            response=_random_response(rng, prediction),
            task_limit=_task_limit(task),
            tolerance=0.05,
        )
        for name, value in row.items():
            if name != "answer_f1":
                seen.setdefault(name, set()).add(value)
            elif not math.isclose(value, round(value)):
                seen.setdefault("partial_f1", set()).add(True)
    assert all(values >= {True, False} for name, values in seen.items() if name != "partial_f1"), seen
    assert seen["program_syntax_valid"] == {True, False, None}
    assert seen["partial_f1"] == {True}
