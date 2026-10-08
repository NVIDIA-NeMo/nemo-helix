# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from metrics.helpers import compute_scores, output_names
from nhx_evals_sdk.execution.evaluator import Evaluator
from nhx_evals_sdk.metrics.tool_calling import MetricResult, ToolCallingMetric


def _response(tool_calls):
    return {"choices": [{"message": {"tool_calls": tool_calls}}]}


class TestToolCallingMetric:
    def test_score_names(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        assert output_names(metric) == ["function_name_accuracy", "function_name_and_args_accuracy"]

    @pytest.mark.asyncio
    async def test_metric(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {"response": _response([{"function": {"name": "sum", "arguments": '{"x": 1}'}}])}
        assert (await compute_scores(metric, item, sample)).outputs[0].value == 1.0

    def test_metric_uses_item_response_when_sample_missing(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {
            "reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}],
            "response": _response([{"function": {"name": "sum", "arguments": '{"x": 1}'}}]),
        }
        scores = metric._metric(item, {})
        assert scores["function_name_accuracy"] == 1.0
        assert scores["function_name_and_args_accuracy"] == 1.0

    def test_metric_raises_when_response_missing(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        with pytest.raises(ValueError, match="No response found in sample or item"):
            metric._metric({"reference": []}, {})

    @pytest.mark.parametrize(
        ("response_data", "expected_error"),
        [
            ("not-a-dict", "expected response dict"),
            ({"unexpected": "value"}, "expected non-empty choices list"),
            ({"choices": "not-a-list"}, "expected non-empty choices list"),
            ({"choices": []}, "expected non-empty choices list"),
            ({"choices": [{}]}, "expected choices\\[0\\]\\.message"),
        ],
    )
    def test_metric_raises_for_malformed_response_data(self, response_data, expected_error):
        metric = ToolCallingMetric(reference="{{item.reference}}")

        with pytest.raises(ValueError, match=expected_error) as exc_info:
            metric._metric({"reference": [], "response": response_data}, {})

        assert repr(response_data) in str(exc_info.value)

    def test_metric_handles_missing_tool_calls_key(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [], "response": {"choices": [{"message": {}}]}}
        scores = metric._metric(item, {})
        assert scores["function_name_accuracy"] == 1.0
        assert scores["function_name_and_args_accuracy"] == 1.0

    def test_metric_fn_name_mismatch(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {"response": _response([{"function": {"name": "subtract", "arguments": '{"x": 1}'}}])}
        scores = metric._metric(item, sample)
        assert scores["function_name_accuracy"] == 0.0
        assert scores["function_name_and_args_accuracy"] == 0.0

    def test_metric_args_mismatch(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {"response": _response([{"function": {"name": "sum", "arguments": '{"x": 2}'}}])}
        scores = metric._metric(item, sample)
        assert scores["function_name_accuracy"] == 1.0
        assert scores["function_name_and_args_accuracy"] == 0.0

    def test_metric_json_decode_error_returns_nan(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {"response": _response([{"function": {"name": "sum", "arguments": "{bad json"}}])}
        scores = metric._metric(item, sample)
        assert scores["function_name_accuracy"] == 1.0
        assert math.isnan(scores["function_name_and_args_accuracy"])

    def test_metric_ignores_tool_calls_without_function_name(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {
            "response": _response(
                [
                    {"function": {"arguments": '{"x": 1}'}},
                    {"function": {"name": "sum", "arguments": '{"x": 1}'}},
                ]
            )
        }
        scores = metric._metric(item, sample)
        assert scores["function_name_accuracy"] == 1.0
        assert scores["function_name_and_args_accuracy"] == 1.0

    def test_metric_treats_missing_arguments_as_empty_dict(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {}}}]}
        sample = {"response": _response([{"function": {"name": "sum", "arguments": None}}])}
        scores = metric._metric(item, sample)
        assert scores["function_name_accuracy"] == 1.0
        assert scores["function_name_and_args_accuracy"] == 1.0

    @pytest.mark.parametrize(
        "bad_reference",
        [
            [{"not_function": {}}],
            [{"function": {"arguments": {}}}],
            [{"function": {"name": "sum"}}],
            ["not-a-dict"],
        ],
    )
    def test_metric_raises_for_malformed_ground_truth(self, bad_reference):
        """Malformed ground-truth entries produce a clear ValueError."""
        metric = ToolCallingMetric(reference="{{item.reference}}")
        with pytest.raises(ValueError, match="Invalid reference template"):
            metric._metric(
                {"reference": bad_reference},
                {"response": _response([])},
            )

    def test_metric_validates_ground_truth_type(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        with pytest.raises(TypeError, match="The reference must render to a list of OpenAI-style tool calls"):
            metric._metric({"reference": {"function": {"name": "sum", "arguments": {}}}}, {"response": _response([])})

    @pytest.mark.asyncio
    async def test_compute_scores_score_names(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        item = {"reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}]}
        sample = {"response": _response([{"function": {"name": "sum", "arguments": '{"x": 1}'}}])}
        result = await compute_scores(metric, item, sample)
        assert {score.name for score in result.outputs} == {"function_name_accuracy", "function_name_and_args_accuracy"}

    @pytest.mark.asyncio
    async def test_raises_clear_error_for_missing_reference_field(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")

        with pytest.raises(ValueError) as exc_info:
            await compute_scores(
                metric,
                item={"prompt": "call sum"},
                sample={"response": _response([])},
            )

        assert "could not render its 'reference' template for this row" in str(exc_info.value)
        assert "missing_key='reference'" in str(exc_info.value)

    def test_run_sync(self):
        metric = ToolCallingMetric(reference="{{item.reference}}")
        result = Evaluator().run_sync(
            metrics=[metric],
            dataset=[
                {
                    "reference": [{"function": {"name": "sum", "arguments": {"x": 1}}}],
                    "response": _response([{"function": {"name": "sum", "arguments": '{"x": 1}'}}]),
                }
            ],
        )
        assert len(result.row_scores) == 1

    @pytest.mark.asyncio
    async def test_compute_scores_exact_match(self):
        """Test compute_scores with exact match of function names and arguments."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": '{"base": 10, "height": 5}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 2
        score_names = {s.name for s in result.outputs}
        assert score_names == {"function_name_accuracy", "function_name_and_args_accuracy"}
        # Both should be 1.0 for exact match
        for score in result.outputs:
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_score_names_match_compute_scores(self):
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )
        item = {
            "expected_tool_calls": [{"function": {"name": "calculate_area", "arguments": {"base": 10, "height": 5}}}]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "calculate_area", "arguments": '{"base": 10, "height": 5}'}}
                            ]
                        }
                    }
                ]
            }
        }
        result = await compute_scores(metric, item, sample)
        assert {score.name for score in result.outputs} == set(output_names(metric))

    @pytest.mark.asyncio
    async def test_compute_scores_reference_without_tojson_preserves_list(self):
        """Regression test: `{{tool_calls}}` should render to a list, not a string."""
        metric = ToolCallingMetric(
            reference="{{tool_calls}}",
        )

        tool_calls = [
            {
                "function": {
                    "name": "calculate_area",
                    "arguments": {"base": 10, "height": 5},
                }
            }
        ]
        item = {"tool_calls": tool_calls}
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": '{"base": 10, "height": 5}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        for score in result.outputs:
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_function_name_match_only(self):
        """Test compute_scores with function name match but different arguments."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": '{"base": 20, "height": 10}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        function_name_accuracy = next(s for s in result.outputs if s.name == "function_name_accuracy")
        function_name_and_args_accuracy = next(s for s in result.outputs if s.name == "function_name_and_args_accuracy")

        # Function names match
        assert function_name_accuracy.value == 1.0
        # But arguments don't match
        assert function_name_and_args_accuracy.value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_no_match(self):
        """Test compute_scores with no function name match."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "different_function",
                                        "arguments": '{"base": 10, "height": 5}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        # Both should be 0.0 when function names don't match
        for score in result.outputs:
            assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_no_tool_calls(self):
        """Test compute_scores when no tool calls are present."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                }
            ]
        }
        sample = {"response": {"choices": [{"message": {}}]}}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        # Both should be 0.0 when no tool calls present
        for score in result.outputs:
            assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_multiple_tool_calls_order_insensitive(self):
        """Test compute_scores is order insensitive for multiple tool calls."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "function_a",
                        "arguments": {"arg": 1},
                    }
                },
                {
                    "function": {
                        "name": "function_b",
                        "arguments": {"arg": 2},
                    }
                },
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "function_b",
                                        "arguments": '{"arg": 2}',
                                    }
                                },
                                {
                                    "function": {
                                        "name": "function_a",
                                        "arguments": '{"arg": 1}',
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        # Both should be 1.0 even though order is different
        for score in result.outputs:
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_compute_scores_invalid_json_arguments_returns_nan(self):
        """Test compute_scores returns NaN for args score when JSON parsing fails."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": "not valid json",
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        function_name_accuracy = next(s for s in result.outputs if s.name == "function_name_accuracy")
        function_name_and_args_accuracy = next(s for s in result.outputs if s.name == "function_name_and_args_accuracy")

        # Function names still match
        assert function_name_accuracy.value == 1.0
        # But arguments can't be parsed, so should be NaN
        assert math.isnan(function_name_and_args_accuracy.value)

    @pytest.mark.asyncio
    async def test_compute_scores_case_sensitive(self):
        """Test compute_scores is case sensitive for function names."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "Calculate_Area",
                        "arguments": {"base": 10},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": '{"base": 10}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        result = await compute_scores(metric, item, sample)

        # Case sensitivity means this should not match
        for score in result.outputs:
            assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_compute_scores_value(self):
        """Test the metric method returns a numeric score."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "calculate_area",
                        "arguments": {"base": 10},
                    }
                }
            ]
        }
        sample = {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculate_area",
                                        "arguments": '{"base": 10}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

        score = (await compute_scores(metric, item, sample)).outputs[0].value
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_offline_evaluation_response_in_item(self):
        """Test offline evaluation where response data is in dataset row (item) not sample.

        This enables offline evaluation of tool-calling by including the model response
        directly in the dataset, without requiring live inference.
        """
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        # Response is in item (dataset row) not sample - enables offline evaluation
        item = {
            "expected_tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "NYC"},
                    }
                }
            ],
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "NYC"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        }
        # Empty sample - no live inference
        sample: dict = {}

        result = await compute_scores(metric, item, sample)

        assert isinstance(result, MetricResult)
        assert len(result.outputs) == 2
        for score in result.outputs:
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_offline_evaluation_no_response_raises_error(self):
        """Test that missing response in both sample and item raises clear error."""
        metric = ToolCallingMetric(
            reference="{{item.expected_tool_calls}}",
        )

        item = {
            "expected_tool_calls": [{"function": {"name": "test", "arguments": {}}}],
        }
        sample: dict = {}

        with pytest.raises(ValueError, match="No response found"):
            await compute_scores(metric, item, sample)
