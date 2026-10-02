<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Metric Selection

Read this file when mapping rubric criteria to SDK metrics, choosing
deterministic vs LLM judges, or working with RAG/agentic/tool-calling metrics.

## Choosing The Right Metric

| Evaluation goal | Prefer | SDK class | Watch for |
| --- | --- | --- | --- |
| Exact label, enum, or string regression | `exact-match` or `string-check` | `ExactMatchMetric`, `StringCheckMetric` | Normalize whitespace/case with Jinja filters if needed |
| Numeric correctness or thresholds | `number-check` | `NumberCheckMetric` | Non-numeric values score `NaN` |
| Reference text similarity | `f1`, `bleu`, or `rouge` | `F1Metric`, `BLEUMetric`, `ROUGEMetric` | Similarity is not factual correctness |
| Flexible semantic quality | `llm-judge` | `LLMJudgeMetric` | Use explicit rubrics and parser-compatible judge output |
| Zero-config semantic quality | `llm-judge` without custom prompt/parsers | `LLMJudgeMetric` with score definitions only | Validate default prompt and parser behavior on tiny data |
| RAG retrieval coverage | `context_recall`, `context_precision`, `context_relevance`, `context_entity_recall` | RAGAS metric classes | Required columns differ by metric |
| RAG grounding or hallucination | `faithfulness`, `response_groundedness`, `noise_sensitivity` | RAGAS metric classes | Requires judge model config |
| RAG answer relevance | `response_relevancy` | `ResponseRelevancyMetric` | Requires judge and embeddings model config |
| Agent final outcome | `agent_goal_accuracy`, `answer_accuracy`, `topic_adherence` | RAGAS agentic metric classes | Most require a judge model |
| Agent tool/function calls | `tool_call_accuracy` or `tool-calling` | `ToolCallAccuracyMetric`, `ToolCallingMetric` | Ground truth and response shape must match the metric |
| Custom business scoring | `remote` or `nemo-agent-toolkit-remote` | `RemoteMetric`, `NemoAgentToolkitRemoteMetric` | Smoke test endpoint auth, payload, timeout, and parser path |
| Repeatable model comparison | Multi-metric SDK run or platform benchmark job | `Evaluator.run(metrics=[...])` or benchmark APIs | Record metric list, dataset, model config, params, and results |
| Bring-your-own benchmark reproduction | Fixed judge plus explicit artifact protocol | SDK harness around generation, judge predictions, and aggregation | Keep generation quality separate from judge-quality evaluation |

## Composable Primitive Mapping

When converting rubric criteria into an eval, use the PRD's primitive mindset
and map it onto the current SDK surface:

| PRD primitive | Use when | Current SDK surface |
| --- | --- | --- |
| `exact` | Criterion checks for a term, phrase, regex-like pattern, or exact output | `ExactMatchMetric`, `StringCheckMetric` |
| `numeric` | Criterion expects a calculated value, threshold, tolerance, or count | `NumberCheckMetric` |
| `llm` | Criterion needs semantic reasoning, style judgment, communication quality, or subjective quality | `LLMJudgeMetric` with `RangeScore` or `RubricScore` |
| `code` | Criterion needs domain-specific computation or validation logic | Custom harness code, `RemoteMetric`, or a reviewed Python checker around SDK results |
| `composite` | One criterion benefits from deterministic checks plus an LLM fallback or weighted subchecks | Multiple SDK metrics plus agent-owned aggregation, or a custom wrapper until a native composite primitive exists |

Prefer deterministic primitives before LLM calls. Use LLM judges for nuance,
not for checks that can be expressed as exact, string, numeric, or code logic.

## RAG And Agentic Metrics

Use RAGAS-backed metric classes from `nemo_evaluator_sdk.metrics.ragas` when the
task is about retrieval, grounding, or agent behavior. Keep dataset columns
aligned with the metric:

| Metric family | Common required fields |
| --- | --- |
| Retrieval quality | `user_input`, `retrieved_contexts`, often `reference` |
| Grounding/hallucination | `user_input`, `response`, `retrieved_contexts`, sometimes `reference` |
| Response relevancy | `user_input`, `response`, `retrieved_contexts` plus embeddings model config |
| Tool call accuracy | `user_input`, `reference_tool_calls` or metric-specific tool-call fields |
| Agent goal/answer/topic | Conversation or final response fields plus judge model config |

When unsure, read the SDK metric class and its tests before creating a large
run.
