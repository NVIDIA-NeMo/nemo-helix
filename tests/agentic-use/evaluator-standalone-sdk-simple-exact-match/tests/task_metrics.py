# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific metrics for the simple exact-match Evals SDK task."""

import asyncio
import contextlib
import difflib
import io
import subprocess
from pathlib import Path

from evaluator_agent_eval.task_config import load_agentic_use_task_config
from evaluator_agent_eval.task_metric_utils import contains_all, run_python_file, score_checks
from nhx_evals_sdk import Evaluator, ExactMatchMetric
from nhx_evals_sdk.values.protocol import MetricDiagnostic, MetricOutput, MetricResult


class ExactMatchEvaluationMetric:
    """Verify the candidate demonstrated standalone SDK exact-match scoring."""

    @property
    def type(self) -> str:
        return "agent_eval/exact_match_evaluation"

    def score_names(self) -> list[str]:
        return ["task_success", "verification_score", "output_schema_valid"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        final_text = str(item.get("output_text", ""))
        solution_path = _solution_path(item)
        code = _read_solution(solution_path)
        process = await asyncio.to_thread(_run_solution, solution_path) if code else None
        has_summary_call = code is not None and ".print_summary(" in code
        code_ran = process is not None and process.returncode == 0
        expected_summary = await asyncio.to_thread(_render_expected_summary) if code_ran else ""
        actual_summary = process.stdout if process is not None else ""
        required_terms_present = code is not None and contains_all(
            f"{final_text}\n{code}\n{actual_summary}",
            _required_terms(item),
        )
        summary_matches = bool(expected_summary) and _normalize_summary(actual_summary) == _normalize_summary(
            expected_summary
        )
        output_schema_valid = code is not None and has_summary_call and code_ran and summary_matches
        task_success = bool(
            item.get("final_answer_extracted") is True and required_terms_present and output_schema_valid and code_ran
        )
        verification_score = score_checks(
            [
                item.get("final_answer_extracted") is True,
                required_terms_present,
                code is not None,
                code_ran,
                has_summary_call,
                summary_matches,
            ]
        )
        diagnostics: list[MetricDiagnostic] = []
        if code_ran and not summary_matches:
            diagnostics.append(
                MetricDiagnostic(
                    message="summary mismatch",
                    details={
                        "expected": expected_summary,
                        "actual": actual_summary,
                        "diff": _summary_diff(expected_summary, actual_summary),
                    },
                )
            )
        return MetricResult(
            outputs=[
                MetricOutput(name="task_success", value=float(task_success)),
                MetricOutput(name="verification_score", value=verification_score),
                MetricOutput(name="output_schema_valid", value=float(output_schema_valid)),
            ],
            diagnostics=diagnostics,
        )


def _solution_path(item: dict) -> Path | None:
    workspace_dir = item.get("workspace_dir")
    if not isinstance(workspace_dir, str):
        return None
    path = Path(workspace_dir) / "solution.py"
    try:
        path.resolve().relative_to(Path(workspace_dir).resolve())
    except ValueError:
        return None
    return path


def _read_solution(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    code = path.read_text(encoding="utf-8", errors="replace").strip()
    if not code:
        return None
    if "Evaluator" in code and "ExactMatchMetric" in code and "run_sync" in code:
        return code
    return None


def _run_solution(path: Path | None) -> subprocess.CompletedProcess[str]:
    return run_python_file(
        path,
        timeout=20,
        cwd=_repo_root(path),
        timeout_stderr="candidate exact-match code timed out",
        missing_stderr="missing solution.py",
    )


def _render_expected_summary() -> str:
    rows = [
        {"question": "2+2?", "expected": "4", "prediction": "4"},
        {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
    ]
    result = Evaluator().run_sync(
        metrics=ExactMatchMetric(
            reference="{{item.expected}}",
            candidate="{{item.prediction}}",
        ),
        dataset=rows,
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result.print_summary()
    return output.getvalue()


def _repo_root(path: Path | None) -> Path:
    candidates: list[Path] = []
    if path is not None:
        candidates.extend([path, *path.parents])
    candidates.extend([Path.cwd(), *Path(__file__).resolve().parents])
    for candidate in candidates:
        if (candidate / "packages" / "nhx_evals_sdk" / "src").exists():
            return candidate
    raise RuntimeError("Could not locate repo root containing packages/nhx_evals_sdk/src")


def _normalize_summary(text: str) -> str:
    return text.strip().replace("\r\n", "\n")


def _summary_diff(expected: str, actual: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            _normalize_summary(expected).splitlines(),
            _normalize_summary(actual).splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )


def _required_terms(item: dict) -> list[str]:
    task_dir = _task_dir(item)
    if task_dir is None:
        return []
    return load_agentic_use_task_config(task_dir).evaluator.expected.required_terms


def _task_dir(item: dict) -> Path | None:
    task_dir = item.get("task_dir")
    if not isinstance(task_dir, str):
        return None
    return Path(task_dir)
