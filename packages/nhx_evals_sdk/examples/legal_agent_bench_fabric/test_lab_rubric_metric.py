# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LabRubricMetric — lock the run_dir contract, diagnostics, and error handling.

These run WITHOUT LAB's source or scoring deps: ``score_rubric`` is stubbed to mimic LAB's real
behavior (it reads deliverables from ``run_dir / "output"``), which is exactly the contract the
metric must satisfy. The metric module is loaded by path (the examples dir is standalone, not a
package), mirroring tests/agent_eval/test_example_metrics.py.

Run from the repo root::

    uv run --frozen pytest \\
        packages/nhx_evals_sdk/examples/legal_agent_bench_fabric/test_lab_rubric_metric.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nhx_evals_sdk.execution.samples import build_metric_input
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor

_MODULE_PATH = Path(__file__).resolve().parent / "lab_rubric_metric.py"
_spec = importlib.util.spec_from_file_location("lab_rubric_metric", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
lab_rubric_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab_rubric_metric)
LabRubricMetric = lab_rubric_metric.LabRubricMetric

_REFERENCE = {
    "criteria": [
        {"id": "C-001", "title": "Names the parties", "match_criteria": "..."},
        {"id": "C-002", "title": "Cites the statute", "match_criteria": "..."},
    ],
    "task_title": "Draft an antitrust memo",
}


def _input(reference: dict[str, Any], workspace: Path):
    evidence = CandidateEvidence(descriptors={"workspace": EvidenceDescriptor(kind="filesystem", ref=str(workspace))})
    return build_metric_input({"reference": reference}, {"evidence": evidence}, index=0)


def _lab_like_score_rubric(captured: dict[str, Any]):
    """Mimic LAB's real score_rubric: read deliverables from run_dir/'output', all-pass grade."""

    def score_rubric(criteria, run_dir, judge, task_desc, parallel):  # noqa: ANN001, ARG001 - LAB's signature
        captured["run_dir"] = run_dir
        output_dir = Path(run_dir) / "output"  # EXACTLY what LAB's scoring.py does
        files = [p.name for p in output_dir.rglob("*") if p.is_file()] if output_dir.is_dir() else []
        # A criterion passes only if LAB actually sees a deliverable (i.e. run_dir was the workspace root).
        results = [
            {"id": c["id"], "title": c["title"], "verdict": "pass" if files else "fail", "reasoning": f"saw {files}"}
            for c in criteria
        ]
        n_pass = sum(1 for r in results if r["verdict"] == "pass")
        score = 1.0 if results and n_pass == len(results) else 0.0
        return SimpleNamespace(score=score, max_score=1.0, criteria_results=results)

    return score_rubric


@pytest.mark.asyncio
async def test_run_dir_is_workspace_root_so_lab_finds_output(tmp_path: Path) -> None:
    # Regression for the double-output bug: LAB's score_rubric appends /output, so the metric must hand it
    # the workspace ROOT. Passing <ws>/output made LAB read <ws>/output/output (empty) and fail everything.
    workspace = tmp_path / "ws"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "memo.docx").write_text("a real deliverable", encoding="utf-8")

    captured: dict[str, Any] = {}
    metric = LabRubricMetric(score_rubric=_lab_like_score_rubric(captured), judge=object())

    result = await metric.compute_scores(_input(_REFERENCE, workspace))
    outputs = {o.name: o.value for o in result.outputs}

    assert Path(captured["run_dir"]) == workspace  # the workspace root, NOT <ws>/output
    assert outputs["n_passed"] == 2 and outputs["n_criteria"] == 2
    assert outputs["criteria_pass_rate"] == 1.0 and outputs["all_pass"] is True and outputs["score"] == 1.0
    # per-criterion verdicts are surfaced as diagnostics (the component-level signal the aggregates hide)
    messages = " ".join(d.message for d in result.diagnostics)
    assert "C-001" in messages and "C-002" in messages
    assert any((d.details or {}).get("verdict") == "pass" for d in result.diagnostics)


@pytest.mark.asyncio
async def test_no_output_dir_scores_zero_without_running_scorer(tmp_path: Path) -> None:
    # The agent wrote no output/ dir: a legitimate all-fail (zero + diagnostic), and the scorer never runs.
    workspace = tmp_path / "ws"
    (workspace / "documents").mkdir(parents=True)  # inputs seeded, but no deliverables produced

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("score_rubric must not run when there is no output/ dir")

    metric = LabRubricMetric(score_rubric=_must_not_run, judge=object())

    result = await metric.compute_scores(_input(_REFERENCE, workspace))
    outputs = {o.name: o.value for o in result.outputs}

    assert outputs["score"] == 0.0 and outputs["n_passed"] == 0 and outputs["n_criteria"] == 2
    assert result.diagnostics and "output" in result.diagnostics[0].message


@pytest.mark.asyncio
async def test_scorer_failure_propagates_not_a_fake_zero(tmp_path: Path) -> None:
    # An infra/scorer failure must RAISE (the evaluator records an errored row), never masquerade as a
    # legitimate all-fail score of 0.0 — the most dangerous failure mode for an eval harness.
    workspace = tmp_path / "ws"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "memo.docx").write_text("x", encoding="utf-8")

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LAB scoring deps missing: pandoc/anthropic")

    metric = LabRubricMetric(score_rubric=_boom, judge=object())

    with pytest.raises(RuntimeError, match="LAB scoring deps missing"):
        await metric.compute_scores(_input(_REFERENCE, workspace))


def test_rejects_output_subdir_other_than_labs_hardcoded_output() -> None:
    # LAB's score_rubric always reads run_dir/"output". Any other value would grade nothing and report a
    # false zero, so the constructor must reject it rather than fail silently at scoring time.
    with pytest.raises(ValueError, match="output_subdir must be 'output'"):
        LabRubricMetric(score_rubric=lambda *a, **k: None, judge=object(), output_subdir="deliverables")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A brace inside a string value must not terminate the object early.
        ('{"verdict": "pass", "reasoning": "cites 15 U.S.C. } sec 1"}', "pass"),
        ('```json\n{"verdict": "fail", "reasoning": "no memo"}\n```', "fail"),
        ('Here is my verdict:\n{"verdict": "pass", "reasoning": "ok"}\nThanks!', "pass"),
    ],
)
def test_parse_json_handles_braces_in_strings_and_prose(raw: str, expected: str) -> None:
    assert lab_rubric_metric._parse_json(raw)["verdict"] == expected


def test_parse_json_raises_when_no_object_present() -> None:
    with pytest.raises(ValueError, match="No JSON found"):
        lab_rubric_metric._parse_json("the model refused to answer")
