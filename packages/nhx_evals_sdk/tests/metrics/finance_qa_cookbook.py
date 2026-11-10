# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cookbook's finance scorer, kept as a test oracle for FinanceQAMetric.

Copied from NVIDIA-NeMo/Nemotron (Apache-2.0) at 18c74c09a,
usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-automodel/scripts/evaluation.py, with only line
wrapping changed: the scoring functions (lines 11-71) and the response parsing (283-306). ``cookbook_row``
repeats the per-row scoring in ``evaluate_route`` (445-467) as a function.
"""

import ast
import re
import unicodedata


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?")


def number_and_scale(value: object) -> tuple[float | None, str]:
    text = normalize(value).replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    match = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*(%|percent|thousand|million|billion)?",
        text,
    )
    if not match:
        return None, ""
    scale = match.group(2) or ""
    if scale == "%":
        scale = "percent"
    return float(match.group(1)), scale


def relaxed(prediction: str, reference: str, tolerance: float) -> bool:
    (left, left_scale), (right, right_scale) = number_and_scale(prediction), number_and_scale(reference)
    if left is not None and right is not None:
        if right_scale and left_scale != right_scale:
            return False
        return abs(left - right) <= max(1e-6, abs(right) * tolerance)
    return normalize(prediction) == normalize(reference)


def answer_tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(value))


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    predicted, expected = Counter(answer_tokens(prediction)), Counter(answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        return 1.0
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def unordered_spans(value: object) -> list[str]:
    return sorted(normalize(part) for part in re.split(r"\s*(?:\||##|;)\s*", str(value)) if normalize(part))


def task_score(task: str, prediction: str, reference: str, tolerance: float) -> tuple[bool, float]:
    if task.endswith("multi-span"):
        matched = unordered_spans(prediction) == unordered_spans(reference)
        return matched, token_f1(prediction, reference)
    if task.endswith("span"):
        score = token_f1(prediction, reference)
        return normalize(prediction) == normalize(reference), score
    matched = relaxed(prediction, reference, tolerance)
    return matched, float(matched)


def split_raw_completion(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if "</think>" not in text:
        return text, ""
    reasoning, answer = text.split("</think>", 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]
    return answer.strip(), reasoning.strip()


def completion_text(message: dict) -> tuple[str, str, str]:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    raw = f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else str(content)
    prediction, parsed_reasoning = split_raw_completion(raw)
    return prediction, parsed_reasoning, raw.strip()


def cookbook_row(
    *, task: str, task_family: str, answer: str, response: dict, task_limit: int, tolerance: float
) -> dict:
    """The per-row scores ``evaluate_route`` records for one response.

    ``task_limit`` is the row's completion limit, which ``evaluate_route`` computes per task (lines 428-431).
    """
    message = response["choices"][0]["message"]
    prediction, reasoning, raw = completion_text(message)
    syntax_valid = None
    if task_family == "program_synthesis":
        try:
            ast.parse(reasoning)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
    scored, answer_f1 = task_score(task, prediction, answer, tolerance)
    usage = response.get("usage") or {}
    return {
        "exact": normalize(prediction) == normalize(answer),
        "relaxed": relaxed(prediction, answer, tolerance),
        "task_correct": scored,
        "answer_f1": answer_f1,
        "trace_closed": bool(reasoning) or "</think>" in raw,
        "program_syntax_valid": syntax_valid,
        "hit_max_new_tokens": int(usage.get("completion_tokens") or 0) >= task_limit,
    }
