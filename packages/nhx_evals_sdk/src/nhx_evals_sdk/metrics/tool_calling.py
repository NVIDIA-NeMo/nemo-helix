# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling metric runtime implementation."""

import json
import logging
from collections.abc import Mapping
from typing import ClassVar, cast

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import (
    TemplateSample,
    build_template_context,
    render_template_or_raise,
    sample_template_payload,
    template_metric_repr,
)
from nhx_evals_sdk.values.metrics import ToolCalling

__all__ = ["ToolCallingMetric"]

_logger = logging.getLogger(__name__)


class ToolCallingMetric(ToolCalling):
    """Tool-calling accuracy metric for structured function calls.

    A metric that supports checks of tool calling:
    - function names
    - function names and args
    and produces respective scores.

    Important. This metric:
    - is case sensitive (for all scores)
    - is order insensitive, so parallel multiple tool calls may appear out of order
    - expects the ground truth to be formatted in OpenAI-compliant tool calling format
    - requires function names with dots (``.``) to be normalized to underscores
      (``_``), since dots are not supported in OpenAI function names

    Evaluator-driven runs should usually source ``reference`` from ``item``.
    """

    _score_names: ClassVar[list[str]] = ["function_name_accuracy", "function_name_and_args_accuracy"]

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return [MetricOutputSpec.continuous_score(score_name) for score_name in self._score_names]

    def _metric(self, item: dict, sample: TemplateSample) -> dict[str, float]:
        """Compute raw tool-calling scores for one item/sample pair."""
        sample_payload = sample_template_payload(sample)
        context = build_template_context(item, sample)
        ground_truth = render_template_or_raise(
            template_name="reference",
            template=self.reference,
            context=context,
            item=item,
            sample=sample,
            metric_repr=template_metric_repr(self),
        )
        if not isinstance(ground_truth, list):
            raise TypeError("The reference must render to a list of OpenAI-style tool calls.")

        try:
            # TODO: Preserve duplicate tool-call multiplicity. The current
            # name-keyed matching keeps existing behavior for this pass.
            gt_fn_names: list[str] = []
            gt_fn_args_by_name: dict[str, object] = {}
            for gt in ground_truth:
                if not isinstance(gt, Mapping):
                    raise TypeError(f"expected reference item to be a mapping, got {type(gt).__name__}")

                gt_mapping = cast(Mapping[str, object], gt)
                function_payload = gt_mapping["function"]
                if not isinstance(function_payload, Mapping):
                    raise TypeError(
                        f"expected reference function payload to be a mapping, got {type(function_payload).__name__}"
                    )

                function_mapping = cast(Mapping[str, object], function_payload)
                function_name = function_mapping["name"]
                if not isinstance(function_name, str):
                    raise TypeError(
                        f"expected reference function name to be a string, got {type(function_name).__name__}"
                    )

                gt_fn_names.append(function_name)
                gt_fn_args_by_name[function_name] = function_mapping["arguments"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid reference template - expected each item to have function.name and function.arguments: {e}"
            ) from e

        # Parse tool calls: check sample (online) first, then item (offline).
        response_data = sample_payload.get("response") or item.get("response")
        if not response_data:
            raise ValueError("No response found in sample or item - tool-calling metric requires model response data")
        if not isinstance(response_data, dict):
            raise ValueError(f"Invalid response format: expected response dict, got {response_data!r}.")

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid response format: expected non-empty choices list in response {response_data!r}.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            raise ValueError(f"Invalid response format: expected choices[0].message in response {response_data!r}.")

        message = first_choice["message"]

        if not message.get("tool_calls"):
            _logger.info("No tool calls found in %s", sample_payload)
            message["tool_calls"] = []

        tool_calls = message["tool_calls"]
        pred_fn_name = [call["function"]["name"] for call in tool_calls if "name" in call.get("function", {})]

        fn_names_match = set(gt_fn_names) == set(pred_fn_name)
        fn_name_accuracy_score = 1.0 if fn_names_match else 0.0

        try:
            pred_fn_args_by_name: dict[str, dict] = {}
            for pred in tool_calls:
                function_payload = pred.get("function", {})
                if "name" not in function_payload:
                    continue

                args = function_payload.get("arguments")
                if args is None:
                    parsed_args = {}
                else:
                    parsed_args = json.loads(args)

                pred_fn_args_by_name[function_payload["name"]] = parsed_args

            _logger.debug("Comparing %s and %s", gt_fn_args_by_name, pred_fn_args_by_name)

            all_args_match = True
            if not fn_names_match:
                # If function names do not match, the combined name-and-args
                # score must also fail automatically.
                fn_name_and_args_accuracy_score = 0.0
            else:
                # Compare parsed arguments for each ground-truth function name.
                for gt_fn_name, gt_fn_args in gt_fn_args_by_name.items():
                    pred_fn_args = pred_fn_args_by_name.get(gt_fn_name)

                    if pred_fn_args != gt_fn_args:
                        all_args_match = False
                        break

                fn_name_and_args_accuracy_score = 1.0 if all_args_match else 0.0
        except (json.JSONDecodeError, TypeError):
            # If the model hallucinated malformed JSON arguments, preserve the
            # legacy behavior and report NaN for the args-sensitive score.
            _logger.warning("Failed parsing tool calling function args: %s", tool_calls)
            fn_name_and_args_accuracy_score = float("nan")

        return {
            "function_name_accuracy": fn_name_accuracy_score,
            "function_name_and_args_accuracy": fn_name_and_args_accuracy_score,
        }

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        item = input.row.data
        sample = input.candidate
        scores = self._metric(item, sample)
        return MetricResult(
            outputs=[MetricOutput(name=score_name, value=score) for score_name, score in scores.items()]
        )
