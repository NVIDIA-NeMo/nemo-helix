# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific metrics for the standalone SDK surface-discovery task."""

import ast
import re
from pathlib import Path

from evaluator_agent_eval.task_config import load_agentic_use_task_config
from evaluator_agent_eval.task_metric_utils import contains_all, extract_fenced_python_code, score_checks
from nhx_evals_sdk.values.results import MetricResult, MetricScore


class SurfaceDiscoveryMetric:
    """Verify the candidate found the standalone Evals SDK surface."""

    @property
    def type(self) -> str:
        return "agent_eval/surface_discovery"

    def score_names(self) -> list[str]:
        return ["task_success", "verification_score", "output_schema_valid"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        final_text = str(item.get("output_text", ""))
        code = _extract_python_code(final_text)
        required_terms_present = contains_all(final_text, _required_terms(item))
        imports_valid = bool(code) and _imports_sdk_symbols(code)
        dataset_valid = bool(code) and _contains_expected_dataset(code)
        metric_templates_valid = bool(code) and _uses_exact_match_templates(code)
        invocation_mode = _invocation_mode(code) if code else None
        invocation_valid = invocation_mode in {"sync", "async"}
        output_schema_valid = bool(
            code and imports_valid and dataset_valid and metric_templates_valid and invocation_valid
        )
        checks = [
            item.get("final_answer_extracted") is True,
            required_terms_present,
            code is not None,
            imports_valid,
            dataset_valid,
            metric_templates_valid,
            invocation_valid,
        ]
        verification_score = score_checks(checks)
        task_success = bool(
            item.get("final_answer_extracted") is True and required_terms_present and output_schema_valid
        )

        return MetricResult(
            scores=[
                MetricScore(name="task_success", value=float(task_success)),
                MetricScore(name="verification_score", value=verification_score),
                MetricScore(name="output_schema_valid", value=float(output_schema_valid)),
            ]
        )


def _extract_python_code(text: str) -> str | None:
    return extract_fenced_python_code(
        text,
        predicate=lambda code: "Evaluator" in code and "ExactMatchMetric" in code,
        scorer=_surface_discovery_code_score,
    )


def _surface_discovery_code_score(code: str) -> int:
    return sum(
        [
            "dataset" in code,
            "ExactMatchMetric(" in code,
            "{{item.expected}}" in code,
            "{{item.prediction}}" in code,
            ".run_sync(" in code or ".run(" in code,
        ]
    )


def _imports_sdk_symbols(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith("nhx_evals_sdk"):
            continue
        for alias in node.names:
            imported_names.add(alias.asname or alias.name)
            imported_names.add(alias.name)
    return "Evaluator" in imported_names and "ExactMatchMetric" in imported_names


def _contains_expected_dataset(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            value = _literal_value(node.value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "dataset" and isinstance(keyword.value, ast.List):
                    value = _literal_value(keyword.value)
                    break
        if value == EXPECTED_DATASET:
            return True
    return False


def _uses_exact_match_templates(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "ExactMatchMetric":
            continue
        keywords = {
            keyword.arg: keyword.value.value
            for keyword in node.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        }
        if keywords.get("reference") == "{{item.expected}}" and keywords.get("candidate") == "{{item.prediction}}":
            return True
    return False


def _invocation_mode(code: str) -> str | None:
    if ".run_sync(" in code:
        return "sync"
    if re.search(r"await\s+.+\.run\(", code):
        return "async"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name == "run_sync":
            return "sync"
        if call_name == "run" and _awaited_run_call(tree, node):
            return "async"
    return None


def _awaited_run_call(tree: ast.AST, target_call: ast.Call) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and node.value is target_call:
            return True
    return False


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _literal_value(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _required_terms(item: dict) -> list[str]:
    task_dir = item.get("task_dir")
    if not isinstance(task_dir, str):
        return []
    return load_agentic_use_task_config(Path(task_dir)).evaluator.expected.required_terms


EXPECTED_DATASET = [
    {"question": "2+2?", "expected": "4", "prediction": "4"},
    {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
]
