# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade LAB deliverables with LAB's OWN scorer, wrapped in the Evaluator Metric protocol.

Rather than reimplement LAB's rubric, the **caller loads LAB's own `score_rubric` and builds a `judge`**
(see `prepare_lab_taskset.load_lab_score_rubric` / `build_lab_judge`) and passes them in; this metric just
orchestrates them. That gives exact fidelity for free — LAB's own code does the document→text extraction
(including pandoc `--track-changes` for redlines), uses LAB's exact `rubric_criterion` prompt, and honors
per-criterion `deliverables` / `evaluation_options`. Keeping the LAB-source loading in the caller (not the
metric) means the metric has **no filesystem / sys.path coupling** and stays portable — it can run in a
backend service. Under the SDK `Metric` protocol it: reads the agent's workspace from the trial's
`workspace` evidence, hands `score_rubric` the workspace *run directory* (LAB reads its `output/` subdir
itself) + criteria + the judge, and maps the result to a `MetricResult` (per-criterion verdicts as diagnostics).

LAB's `evaluation/scoring.py` public entry point (pinned commit):

    def score_rubric(criteria: list[dict], run_dir, judge, task_desc: str, parallel: int) -> RubricResult
    # RubricResult(score, max_score, criteria_results: list[dict]); all-pass: score==max_score iff every criterion passes.

    class Judge:  # evaluation/judge.py
        def __init__(self, model: str = "claude-sonnet-4-6")   # provider auto-detected from the name; reads env keys
        def evaluate_from_file(self, prompt_name: str, variables: dict) -> {"verdict": "pass"|"fail", "reasoning": str}

REQUIREMENTS (in the *eval* process, where this metric runs — not the agent sandbox): LAB's extraction
stack must be importable/on PATH — `pandoc` (binary), `libreoffice`/`soffice`, `python-docx`,
`python-redlines`, `pandas`, `openpyxl`, `pdfplumber`, `markitdown` — plus the judge provider SDK
(`openai`/`anthropic`/…). For an OpenAI-compatible judge endpoint, LAB's `Judge` uses the OpenAI SDK,
which honors `OPENAI_BASE_URL` + `OPENAI_API_KEY` from the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.metrics.protocol import (
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)

logger = logging.getLogger(__name__)

# The Fabric runner exposes the agent's final per-task file tree under this evidence key
# (agent_eval/runtimes/fabric/runtime.py: _WORKSPACE_EVIDENCE_KEY = "workspace").
WORKSPACE_EVIDENCE_KEY = "workspace"


class LabRubricMetric:
    """Score a trial's deliverables with LAB's own `score_rubric` (exact prompt + extraction)."""

    def __init__(
        self,
        *,
        score_rubric: Callable[..., Any],
        judge: Any,
        output_subdir: str = "output",
        parallel: int = 4,
        metric_type: str = "lab_rubric",
    ) -> None:
        # LAB's score_rubric hardcodes ``run_dir / "output"``, so any other value would silently grade
        # nothing and report a false zero (see _run_dir). Fail loudly instead.
        if output_subdir != "output":
            raise ValueError(f"output_subdir must be 'output' (LAB's scorer hardcodes it), got {output_subdir!r}")
        # Portability: the caller loads LAB's ``score_rubric`` (the scorer callable) and builds the
        # ``judge`` (any object with ``evaluate_from_file(name, variables)``) and passes them in — so this
        # metric has no filesystem / sys.path / LAB-source coupling and can run anywhere (e.g. a backend
        # service via a plugin). See prepare_lab_taskset.load_lab_score_rubric / build_lab_judge.
        self._score_rubric = score_rubric
        self._judge = judge
        self._output_subdir = output_subdir
        self._parallel = parallel
        self._type = metric_type

    @property
    def type(self) -> str:
        return self._type

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("score"),  # LAB all-pass reward (1.0 iff every criterion passes)
            MetricOutputSpec.continuous_score("criteria_pass_rate"),
            MetricOutputSpec.boolean("all_pass"),
            MetricOutputSpec.discrete_score("n_passed"),
            MetricOutputSpec.discrete_score("n_criteria"),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        reference = input.row.data.get("reference") or {}
        criteria = list(reference.get("criteria") or [])
        task_desc = str(reference.get("task_title") or "")
        n_criteria = len(criteria)

        if n_criteria == 0:
            return self._zero(n_criteria, note="task declares no rubric criteria")

        run_dir = await self._run_dir(input)
        if run_dir is None:
            # The agent produced no deliverables directory: a legitimate all-fail (nothing gradable), not
            # a scorer error. Return a zero score with a diagnostic that says so.
            return self._zero(n_criteria, note=f"agent produced no {self._output_subdir!r} deliverables directory")

        # Deliberately UNGUARDED: a scorer/judge failure must not masquerade as a legitimate all-fail.
        # Letting it raise lets AgentEvaluator record an ERRORED metric row (with a diagnostic), which stays
        # distinct from a real score=0.0. LAB's score_rubric is synchronous and does its own thread
        # parallelism, so keep it off the event loop.
        result = await asyncio.to_thread(
            self._score_rubric, criteria, str(run_dir), self._judge, task_desc, self._parallel
        )
        return self._to_result(result, n_criteria)

    # --- helpers ---------------------------------------------------------------------------------
    async def _run_dir(self, input: MetricInput) -> Path | None:
        """Return the workspace ROOT to hand LAB's ``score_rubric`` as its ``run_dir``.

        Critically, LAB's ``score_rubric`` reads deliverables from ``run_dir / "output"`` — it appends the
        ``output/`` segment itself (evaluation/scoring.py). So we must pass the workspace *root*, not
        ``<root>/output``: passing the latter makes LAB look in ``<root>/output/output``, find nothing, and
        grade "(No agent output found)" for every criterion. We still verify the agent actually produced an
        ``output/`` tree (``self._output_subdir`` must match LAB's hardcoded ``output``), returning None
        (-> zero score) when it did not.
        """
        evidence = input.candidate.evidence
        if evidence is None or evidence.get(WORKSPACE_EVIDENCE_KEY) is None:
            return None
        handle = await evidence.filesystem(WORKSPACE_EVIDENCE_KEY)
        if self._output_subdir and not handle.path(self._output_subdir).is_dir():
            logger.warning(
                "LabRubricMetric: no '%s/' deliverables dir in workspace; grading nothing", self._output_subdir
            )
            return None
        return handle.root

    def _to_result(self, result: Any, n_criteria: int) -> MetricResult:
        criteria_results = list(getattr(result, "criteria_results", []) or [])
        n_passed = sum(1 for c in criteria_results if str(c.get("verdict", "")).strip().lower() == "pass")
        total = len(criteria_results) or n_criteria
        all_pass = total > 0 and n_passed == total
        score = float(getattr(result, "score", 1.0 if all_pass else 0.0))
        pass_rate = (n_passed / total) if total else 0.0
        return MetricResult(
            outputs=[
                MetricOutput(name="score", value=score),
                MetricOutput(name="criteria_pass_rate", value=pass_rate),
                MetricOutput(name="all_pass", value=all_pass),
                MetricOutput(name="n_passed", value=n_passed),
                MetricOutput(name="n_criteria", value=total),
            ],
            diagnostics=self._diagnostics(criteria_results, n_passed, total),
        )

    @staticmethod
    def _diagnostics(criteria_results: list[dict[str, Any]], n_passed: int, total: int) -> list[MetricDiagnostic]:
        """Surface LAB's per-criterion verdicts (verdict + reasoning) — the component-level signal the flat
        aggregates hide. One summary finding, then one per criterion, recorded in MetricResult.diagnostics."""
        diagnostics = [
            MetricDiagnostic(
                message=f"{n_passed}/{total} rubric criteria passed",
                details={"n_passed": n_passed, "n_criteria": total},
            )
        ]
        for criterion in criteria_results:
            verdict = str(criterion.get("verdict", "")).strip().lower() or "unknown"
            title = str(criterion.get("title") or criterion.get("id") or "criterion")
            identifier = str(criterion.get("id") or "").strip()
            label = f"{identifier}: {title}" if identifier else title
            diagnostics.append(
                MetricDiagnostic(
                    message=f"[{verdict.upper()}] {label}",
                    details={
                        "id": criterion.get("id"),
                        "title": criterion.get("title"),
                        "verdict": verdict,
                        "reasoning": criterion.get("reasoning"),
                    },
                )
            )
        return diagnostics

    @staticmethod
    def _zero(n_criteria: int, *, note: str | None = None) -> MetricResult:
        return MetricResult(
            outputs=[
                MetricOutput(name="score", value=0.0),
                MetricOutput(name="criteria_pass_rate", value=0.0),
                MetricOutput(name="all_pass", value=False),
                MetricOutput(name="n_passed", value=0),
                MetricOutput(name="n_criteria", value=n_criteria),
            ],
            diagnostics=[MetricDiagnostic(message=note)] if note else [],
        )


class OpenAICompatibleJudge:
    """A LAB-compatible judge (``evaluate_from_file``) for any OpenAI-compatible endpoint.

    Reuses LAB's exact rubric prompts (passed in as ``{prompt_name: template}`` + ``str.format(**variables)``)
    and JSON extraction, but calls ``chat.completions`` instead of LAB's native routing — so a namespaced
    NVIDIA model id (e.g. ``nvidia/nemotron-3-super-120b-a12b``) against an OpenAI-compatible endpoint works. LAB's native
    Judge can't: it rejects non-``gpt-*`` names and uses the OpenAI *Responses* API, which NVIDIA doesn't serve.
    """

    def __init__(
        self,
        *,
        prompts: Mapping[str, str],
        model: str,
        base_url: str | None,
        api_key: str | None,
        max_attempts: int = 2,
        min_interval_s: float = 2.0,
    ) -> None:
        from openai import OpenAI

        self._prompts = dict(prompts)
        self._model = model
        self._max_attempts = max(1, max_attempts)
        # build.nvidia.com rate-limits large models (~40 req/min); throttle to stay under it. Keep the
        # timeout + retries BOUNDED so a stuck/unresponsive endpoint fails fast (worst case per criterion
        # ~= max_attempts * max_retries * timeout) instead of hanging for many minutes on a dead connection.
        self._min_interval_s = min_interval_s
        self._last_call = 0.0
        self._throttle_lock = threading.Lock()
        self._client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=90, max_retries=2)

    def _throttle(self) -> None:
        with self._throttle_lock:
            wait = self._min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def evaluate_from_file(self, prompt_name: str, variables: dict[str, Any]) -> dict[str, Any]:
        template = self._prompts[prompt_name]
        prompt = template.format(**variables)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        self._throttle()
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            # First attempt asks for strict JSON mode; later attempts drop response_format (some endpoints
            # reject it) and re-ask on a parse failure (transient truncation / format drift).
            call = {**kwargs, "response_format": {"type": "json_object"}} if attempt == 0 else kwargs
            try:
                response = self._client.chat.completions.create(**call)
                return _parse_json(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - retry API and JSON-parse failures alike
                last_exc = exc
        # Exhausted retries: raise so the metric errors this row (recorded distinctly from an all-fail),
        # rather than silently scoring a criterion the judge never actually evaluated.
        raise RuntimeError(
            f"judge {self._model!r} failed to produce a valid verdict for {prompt_name!r} after "
            f"{self._max_attempts} attempts: {last_exc}"
        ) from last_exc


def _parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response (code fences or balanced braces) — LAB's logic."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # raw_decode understands string literals, so a brace inside a value (e.g. {"reasoning": "a } here"})
    # does not end the object early the way a naive brace counter would.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"No JSON found in judge response: {text[:200]}")
