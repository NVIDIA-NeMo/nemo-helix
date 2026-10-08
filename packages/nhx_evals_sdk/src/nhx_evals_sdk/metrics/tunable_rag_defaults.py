# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default rubric text and JSON format instructions for tunable RAG evaluation.

Ported from https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/main/packages/nvidia_nat_langchain/src/nat/plugins/langchain/eval/tunable_rag_evaluator.py
"""

from __future__ import annotations

DEFAULT_SCORING_INSTRUCTIONS = (
    "Every score is a decimal number on a 0.0 to 1.0 scale, where 0.0 is the lowest and 1.0 is the highest. "
    "Never use a 0-10 or 0-100 scale. "
    "The coverage score is a measure of how well the generated answer covers the critical aspects mentioned in the "
    "expected answer. A low coverage score indicates that the generated answer misses critical aspects of the "
    "expected answer. A middle coverage score indicates that the generated answer covers some of the must-haves "
    "of the expected answer but lacks other details. A high coverage score indicates that all of the expected "
    "aspects are present in the generated answer. The correctness score is a measure of how well the generated "
    "answer matches the expected answer. A low correctness score indicates that the generated answer is incorrect "
    "or does not match the expected answer. A middle correctness score indicates that the generated answer is "
    "correct but lacks some details. A high correctness score indicates that the generated answer is exactly the "
    "same as the expected answer. The relevance score is a measure of how well the generated answer is relevant "
    "to the question. A low relevance score indicates that the generated answer is not relevant to the question. "
    "A middle relevance score indicates that the generated answer is somewhat relevant to the question. A high "
    "relevance score indicates that the generated answer is exactly relevant to the question. The reasoning is a "
    "1-2 sentence explanation for the scoring."
)

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "coverage": 0.5,
    "correctness": 0.3,
    "relevance": 0.2,
}

DEFAULT_SCORING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage_score": {"type": "number"},
        "correctness_score": {"type": "number"},
        "relevance_score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["coverage_score", "correctness_score", "relevance_score", "reasoning"],
    "additionalProperties": False,
}

CUSTOM_SCORING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


def build_evaluation_prompt(
    *,
    judge_llm_prompt: str,
    instruction: str,
    answer_description: str,
    generated_answer: str,
    default_scoring: bool,
) -> str:
    """Build the judge user prompt (format instructions are passed via structured output)."""
    if default_scoring:
        return (
            "You are an intelligent assistant that responds strictly in JSON format. "
            f"Judge based on the following scoring rubric: {DEFAULT_SCORING_INSTRUCTIONS}"
            f"{judge_llm_prompt}\n"
            f"Here is the instruction: {instruction}"
            f"Here is the description of the expected answer: {answer_description}"
            f"Here is the generated answer: {generated_answer}"
        )
    return (
        f"You are an intelligent assistant that responds strictly in JSON format. {judge_llm_prompt}\n"
        f"Here is the instruction: {instruction}"
        f"Here is the description of the expected answer: {answer_description}"
        f"Here is the generated answer: {generated_answer}"
    )


def normalize_score_weights(weights: dict[str, float] | None) -> tuple[float, float, float]:
    """Normalize coverage/correctness/relevance weights to sum to 1."""
    source = weights or DEFAULT_SCORE_WEIGHTS
    coverage = float(source.get("coverage", 1 / 3))
    correctness = float(source.get("correctness", 1 / 3))
    relevance = float(source.get("relevance", 1 / 3))
    total = coverage + correctness + relevance
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return coverage / total, correctness / total, relevance / total
