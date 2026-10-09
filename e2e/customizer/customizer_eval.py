# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic base-vs-tuned uplift eval for customizer GPU e2e tests."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHAT_USER_PROMPT_TEMPLATE: dict[str, Any] = {"messages": "{{ item.messages[:-1] }}"}
CHAT_REFERENCE_TEMPLATE = "{{ item.messages[-1].content }}"
MetricKind = str


def assert_chat_row(row: dict[str, Any], index: int | None = None) -> None:
    label = f"row {index}" if index is not None else "row"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{label}: expected a 'messages' list with a prompt turn + final assistant label")
    last = messages[-1]
    if not isinstance(last, dict):
        raise ValueError(f"{label}: expected final messages[-1] to be a dict, got {type(last).__name__}")
    if last.get("role") != "assistant":
        raise ValueError(f"{label}: expected final messages[-1] role='assistant' (the label to score)")


def load_chat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert_chat_row(row, index=index)
        rows.append(row)
    return rows


def provider_route_url(base_url: str, workspace: str, deployment_name: str) -> str:
    return f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/provider/{deployment_name}/-/v1"


def base_model_field(workspace: str, entity: str) -> str:
    return f"{workspace}/{entity}"


def lora_model_field(workspace: str, adapter: str) -> str:
    return f"{workspace}--{adapter}"


def _build_target(base_url: str, workspace: str, deployment_name: str, model_field: str):
    from nhx_evals_sdk.enums import ModelFormat
    from nhx_evals_sdk.values.models import Model

    return Model(
        url=provider_route_url(base_url, workspace, deployment_name),
        name=model_field,
        format=ModelFormat.NVIDIA_NIM,
    )


def _build_config(max_tokens: int, parallelism: int, limit_samples: int | None, enable_thinking: bool):
    from nhx_evals_sdk.values import InferenceParams, RunConfigOnlineModel

    inference_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0}
    if not enable_thinking:
        inference_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return RunConfigOnlineModel(
        parallelism=parallelism,
        limit_samples=limit_samples,
        inference=InferenceParams(**inference_kwargs),
    )


def _metric(kind: MetricKind):
    if kind == "f1":
        from nhx_evals_sdk.metrics.f1 import F1Metric

        return F1Metric(reference=CHAT_REFERENCE_TEMPLATE)
    if kind == "exact_match":
        from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric

        return ExactMatchMetric(reference=CHAT_REFERENCE_TEMPLATE)
    if kind == "rouge":
        from nhx_evals_sdk.metrics.rouge import ROUGEMetric

        return ROUGEMetric(reference=CHAT_REFERENCE_TEMPLATE)
    raise ValueError(f"Unknown metric kind: {kind!r} (expected 'f1', 'exact_match', or 'rouge')")


def score_rows(
    rows: Sequence[dict[str, Any]],
    base_url: str,
    workspace: str,
    deployment_name: str,
    model_field: str,
    *,
    metric: MetricKind = "f1",
    max_tokens: int = 64,
    parallelism: int = 8,
    limit_samples: int | None = None,
    enable_thinking: bool = False,
) -> float:
    from nhx_evals_sdk import Evaluator

    for index, row in enumerate(rows):
        assert_chat_row(row, index=index)

    target = _build_target(base_url, workspace, deployment_name, model_field)
    config = _build_config(max_tokens, parallelism, limit_samples, enable_thinking)
    result = Evaluator().run_sync(
        metrics=[_metric(metric)],
        dataset=list(rows),
        target=target,
        prompt_template=CHAT_USER_PROMPT_TEMPLATE,
        config=config,
    )
    score = result.aggregate_scores.scores[0]
    if score.mean is None:
        raise RuntimeError(
            f"eval {model_field}: metric {metric} returned no mean score (n={len(rows)}) via "
            f"{deployment_name}. All rows likely failed inference — treating as an eval-pipeline "
            "failure rather than a 0.0 score."
        )
    mean = score.mean
    logger.info(
        "eval %s: metric=%s mean=%.4f (n=%d) via %s",
        model_field,
        metric,
        mean,
        len(rows),
        deployment_name,
    )
    return round(float(mean), 4)


@dataclass
class UpliftResult:
    metric: MetricKind
    base_score: float
    tuned_score: float
    base_label: str = "base"
    tuned_label: str = "tuned"
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def uplift(self) -> float:
        return round(self.tuned_score - self.base_score, 4)

    def assert_ok(self, require_uplift: bool = False, tolerance: float = 0.02) -> None:
        if require_uplift:
            assert self.tuned_score > self.base_score, (
                f"expected strict uplift on {self.metric}: "
                f"tuned={self.tuned_score} !> base={self.base_score} (uplift={self.uplift})"
            )
        else:
            assert self.tuned_score >= self.base_score - tolerance, (
                f"{self.metric} regressed beyond tolerance {tolerance}: "
                f"tuned={self.tuned_score} < base={self.base_score} (uplift={self.uplift})"
            )
