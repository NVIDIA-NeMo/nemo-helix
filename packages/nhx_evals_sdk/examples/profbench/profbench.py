# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProfBench loading, rubric scoring, and judging helpers."""

from __future__ import annotations

import html
import json
import os
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import nhx_evals_sdk.inference as inference
from nhx_evals_sdk.agent_eval.dashboard import write_dashboard as write_sdk_dashboard
from nhx_evals_sdk.agent_eval.results import AgentEvalResult
from nhx_evals_sdk.agent_eval.scores import AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nhx_evals_sdk.execution.metric_execution import (
    generate_online_sample,
    resolve_target_structured_output_mode,
)
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values import InferenceParams, Model, RunConfigOnlineModel
from nhx_evals_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROFBENCH_DATASET_URL = "https://huggingface.co/datasets/nvidia/ProfBench/resolve/main/test.jsonl"
PROFBENCH_METRIC_TYPE = "profbench_rubric"
PROFBENCH_METRIC_ID = "profbench"
PROFBENCH_DETAILS_OUTPUT = "profbench_details"
PROFBENCH_WEIGHT_POINTS = {
    "Critical": 4.0,
    "Major": 3.0,
    "Minor": 2.0,
    "Additional": 1.0,
}
PROFBENCH_BASELINE_RESPONSES = {
    "o3": "o3_response",
    "r1-0528": "r1-0528_response",
    "grok4": "grok4_response",
}
PROFBENCH_JUDGE_STRUCTURED_OUTPUT = {
    "schema": {
        "type": "object",
        "properties": {
            "fulfilled": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["fulfilled", "reason"],
        "additionalProperties": False,
    }
}

CriterionType = str | list[str]


class EvidenceLocator(BaseModel):
    """Concrete link to evidence for a score deduction."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    uri: str
    line: int | None = Field(default=None, ge=1)
    json_path: str | None = None
    excerpt: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _atif_requires_line(self) -> "EvidenceLocator":
        if self.kind.lower() == "atif" and self.line is None:
            raise ValueError("ATIF evidence locators require a line number")
        return self

    def href(self, *, base_dir: str | Path | None = None) -> str:
        """Return a browser-usable evidence link."""
        href, supports_line_fragment = self._href_base(base_dir=base_dir)
        if self.line is None or not supports_line_fragment:
            return href
        separator = "&" if "#" in href else "#"
        return f"{href}{separator}L{self.line}"

    def _href_base(self, *, base_dir: str | Path | None) -> tuple[str, bool]:
        if self.uri.startswith(("http://", "https://", "atif://")):
            href = self.uri
            return href, True

        local_path = _local_evidence_path(self.uri)
        if local_path is not None:
            if base_dir is not None:
                base_path = Path(base_dir).expanduser().resolve()
                resolved_path = local_path if local_path.is_absolute() else (base_path / local_path).resolve()
                return quote(Path(os.path.relpath(resolved_path, base_path)).as_posix(), safe="/"), False
            return local_path.expanduser().resolve().as_uri(), False

        return quote(self.uri), False


def _local_evidence_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme:
        return None
    return Path(uri)


class CriterionScore(BaseModel):
    """Per-criterion scoring result."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    criterion_type: CriterionType | None = None
    weight_name: str
    points: float
    fulfilled: bool
    evidence: list[EvidenceLocator] = Field(default_factory=list)
    judge_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfBenchRubricDetails(BaseModel):
    """ProfBench-specific rubric diagnostics emitted by the example metric."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    earned_points: float = Field(ge=0)
    max_points: float = Field(gt=0)
    model_id: str
    domain: str | None = None
    criterion_scores: list[CriterionScore]


class ProfBenchCriterion(BaseModel):
    """One ProfBench rubric criterion with source location metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    weight_name: str
    points: float
    criterion_type: CriterionType | None = None
    source_uri: str
    line_number: int
    json_path: str

    @classmethod
    def from_raw(
        cls,
        *,
        task_id: str,
        index: int,
        raw: dict[str, Any],
        source_uri: str,
        line_number: int,
    ) -> "ProfBenchCriterion":
        if "criterion_weight" not in raw:
            raise ValueError(f"ProfBench rubric {task_id}:criterion-{index + 1} is missing criterion_weight")
        weight_name = str(raw["criterion_weight"])
        if weight_name not in PROFBENCH_WEIGHT_POINTS:
            raise ValueError(f"Unknown ProfBench criterion_weight {weight_name!r} for {task_id}:criterion-{index + 1}")
        return cls(
            id=f"{task_id}:criterion-{index + 1}",
            description=str(raw["criterion_description"]),
            weight_name=weight_name,
            points=PROFBENCH_WEIGHT_POINTS[weight_name],
            criterion_type=raw.get("criterion_type"),
            source_uri=source_uri,
            line_number=line_number,
            json_path=f"$.rubrics[{index}]",
        )

    def source_locator(self) -> EvidenceLocator:
        return EvidenceLocator(
            kind="profbench",
            uri=self.source_uri,
            line=self.line_number,
            json_path=self.json_path,
            excerpt=self.description,
            label=self.id,
        )


class ProfBenchJudgeRequest(BaseModel):
    """Prompt material passed to a ProfBench rubric judge."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    response: str
    criterion_id: str
    criterion_description: str
    criterion_type: CriterionType | None = None
    weight_name: str


class ProfBenchJudgeDecision(BaseModel):
    """Yes/No rubric judge decision."""

    model_config = ConfigDict(extra="forbid")

    fulfilled: bool
    reason: str
    raw_response: dict[str, Any] | None = None


class ProfBenchJudge(Protocol):
    """Async rubric judge protocol used by the example metric."""

    def judge(self, request: ProfBenchJudgeRequest) -> Awaitable[ProfBenchJudgeDecision]: ...


class ProfBenchBenchmark(BaseModel):
    """Loaded ProfBench tasks and provided baseline trials."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tasks: list[AgentEvalTask]
    trials: list[AgentEvalTrial]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfBenchModelJudge:
    """Minimal Yes/No LLM judge for ProfBench criteria."""

    def __init__(
        self,
        *,
        model: Model,
        params: RunConfigOnlineModel | None = None,
        inference_fn: inference.InferenceFn | None = None,
        client: Any | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.params = params or _default_judge_params()
        self.inference_fn = inference_fn or inference.make_inference_request
        self.client = client
        self.default_headers = default_headers

    async def judge(self, request: ProfBenchJudgeRequest) -> ProfBenchJudgeDecision:
        preprocess_hooks, postprocess_hooks = inference.new_hooks(self.params)
        # Hooks are rebuilt per judge call, so this relies on detection being cached per endpoint.
        await resolve_target_structured_output_mode(
            preprocess_hooks=preprocess_hooks,
            model=self.model,
            inference_fn=self.inference_fn,
            params=self.params,
        )
        sample = await generate_online_sample(
            target=self.model,
            row={"prompt": _render_judge_prompt(request)},
            index=0,
            prompt_template={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON-only evaluator. Return the requested JSON object and no reasoning text.",
                    },
                    {"role": "user", "content": "{{item.prompt}}"},
                ]
            },
            params=self.params,
            inference_fn=self.inference_fn,
            client=self.client,
            preprocess_hooks=preprocess_hooks,
            postprocess_hooks=postprocess_hooks,
            default_headers=self.default_headers,
        )
        text = sample.get("output_text") or ""
        raw_response = sample.get("response")
        return _parse_judge_decision(text, raw_response=raw_response if isinstance(raw_response, dict) else None)


class ProfBenchRubricMetric:
    """Score ProfBench trials and emit evidence-backed point deductions."""

    def __init__(
        self,
        *,
        criteria: list[ProfBenchCriterion],
        judge: ProfBenchJudge | None = None,
        evidence_dir: Path | None = None,
    ) -> None:
        self.criteria = criteria
        self.judge = judge
        self.evidence_dir = evidence_dir

    @property
    def type(self) -> str:
        return PROFBENCH_METRIC_TYPE

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score(PROFBENCH_METRIC_ID),
            MetricOutputSpec.model(PROFBENCH_DETAILS_OUTPUT, ProfBenchRubricDetails),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        details = await self._score(input)
        return MetricResult(
            outputs=[
                MetricOutput(name=PROFBENCH_METRIC_ID, value=details.score),
                MetricOutput(name=PROFBENCH_DETAILS_OUTPUT, value=details),
            ]
        )

    async def _score(self, input: MetricInput) -> ProfBenchRubricDetails:
        if not self.criteria:
            raise ValueError("ProfBench metric requires at least one criterion")

        fulfilments = _baseline_fulfilments(input.candidate.metadata)
        output_text = input.candidate.output_text
        if output_text is None:
            raise ValueError("ProfBench trial has no output_text to score")

        earned_points = 0.0
        max_points = sum(criterion.points for criterion in self.criteria)
        criterion_scores: list[CriterionScore] = []

        for criterion in self.criteria:
            source_locator = criterion.source_locator()
            judge_locator: EvidenceLocator | None = None
            judge_reason: str | None = None
            score_source = "dataset_label"

            if criterion.id in fulfilments:
                fulfilled = fulfilments[criterion.id]
            else:
                if self.judge is None:
                    raise ValueError("ProfBench candidate scoring requires a judge when dataset labels are absent")
                score_source = "judge"
                inputs = input.row.data.get("inputs", {})
                task = input.row.data.get("task", {})
                judge_request = ProfBenchJudgeRequest(
                    task_id=str(task.get("id", "")) if isinstance(task, dict) else "",
                    prompt=str(inputs.get("instruction", "")) if isinstance(inputs, dict) else "",
                    response=output_text,
                    criterion_id=criterion.id,
                    criterion_description=criterion.description,
                    criterion_type=criterion.criterion_type,
                    weight_name=criterion.weight_name,
                )
                decision = await self.judge.judge(judge_request)
                fulfilled = decision.fulfilled
                judge_reason = decision.reason
                judge_locator = self._write_judge_artifact(
                    task_id=str(task.get("id", "")) if isinstance(task, dict) else "",
                    trial_id=str(input.candidate.metadata.get("trial_id", "trial")),
                    criterion_id=criterion.id,
                    request=judge_request,
                    decision=decision,
                )

            evidence = [source_locator]
            if judge_locator is not None:
                evidence.append(judge_locator)

            if fulfilled:
                earned_points += criterion.points

            criterion_scores.append(
                CriterionScore(
                    criterion_id=criterion.id,
                    description=criterion.description,
                    criterion_type=criterion.criterion_type,
                    weight_name=criterion.weight_name,
                    points=criterion.points,
                    fulfilled=fulfilled,
                    evidence=evidence,
                    judge_reason=judge_reason,
                    metadata={"score_source": score_source},
                )
            )

        model_id = str(
            input.candidate.metadata.get("model_id") or input.candidate.metadata.get("target_name") or "candidate"
        )
        inputs = input.row.data.get("inputs", {})
        domain = inputs.get("domain") if isinstance(inputs, dict) else None
        return ProfBenchRubricDetails(
            score=earned_points / max_points,
            earned_points=earned_points,
            max_points=max_points,
            model_id=model_id,
            domain=domain if isinstance(domain, str) else None,
            criterion_scores=criterion_scores,
        )

    def _write_judge_artifact(
        self,
        *,
        task_id: str,
        trial_id: str,
        criterion_id: str,
        request: ProfBenchJudgeRequest,
        decision: ProfBenchJudgeDecision,
    ) -> EvidenceLocator | None:
        if self.evidence_dir is None:
            return None

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        file_name = _safe_artifact_name(f"judge-{task_id}-{trial_id}-{criterion_id}.json")
        path = self.evidence_dir / file_name
        path.write_text(
            json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return EvidenceLocator(
            kind="judge", uri=str(path.resolve()), line=1, json_path="$.decision", excerpt=decision.reason
        )


def load_profbench(
    source: str | Path = PROFBENCH_DATASET_URL,
    *,
    limit: int | None = None,
    judge: ProfBenchJudge | None = None,
    evidence_dir: Path | None = None,
    include_cached_fulfilments: bool = True,
) -> ProfBenchBenchmark:
    """Load ProfBench from local JSONL or the Hugging Face raw URL."""
    source_uri, lines, metadata = _read_jsonl_source(source, evidence_dir=evidence_dir)
    tasks: list[AgentEvalTask] = []
    trials: list[AgentEvalTrial] = []

    for line_number, line in lines[:limit]:
        row = json.loads(line)
        task_id = str(row["task_id"])
        criteria = [
            ProfBenchCriterion.from_raw(
                task_id=task_id,
                index=index,
                raw=raw_rubric,
                source_uri=source_uri,
                line_number=line_number,
            )
            for index, raw_rubric in enumerate(row["rubrics"])
        ]
        task = AgentEvalTask(
            id=task_id,
            intent=str(row["prompt"]),
            inputs={"instruction": row["prompt"], "domain": row.get("domain")},
            metrics=[ProfBenchRubricMetric(criteria=criteria, judge=judge, evidence_dir=evidence_dir)],
            metadata={
                "benchmark": "ProfBench",
                "domain": row.get("domain"),
                "profbench_task_id": task_id,
                "source_uri": source_uri,
                "line_number": line_number,
            },
        )
        tasks.append(task)
        trials.extend(
            _recorded_trials(
                row=row,
                task_id=task_id,
                criteria=criteria,
                source_uri=source_uri,
                include_cached_fulfilments=include_cached_fulfilments,
            )
        )

    metadata.update(
        {
            "benchmark": "ProfBench",
            "dataset_url": PROFBENCH_DATASET_URL,
            "source": source_uri,
            "record_count": len(tasks),
            "baseline_models": sorted(PROFBENCH_BASELINE_RESPONSES),
        }
    )
    return ProfBenchBenchmark(tasks=tasks, trials=trials, metadata=metadata)


def profbench_details(output: MetricOutput) -> ProfBenchRubricDetails | None:
    """Return ProfBench details for a metric output, if present."""
    if output.name != PROFBENCH_DETAILS_OUTPUT:
        return None
    if isinstance(output.value, ProfBenchRubricDetails):
        return output.value
    return ProfBenchRubricDetails.model_validate(output.value)


def _recorded_trials(
    *,
    row: dict[str, Any],
    task_id: str,
    criteria: list[ProfBenchCriterion],
    source_uri: str,
    include_cached_fulfilments: bool,
) -> list[AgentEvalTrial]:
    trials: list[AgentEvalTrial] = []
    for model_id, response_field in PROFBENCH_BASELINE_RESPONSES.items():
        response_text = row.get(response_field)
        if not isinstance(response_text, str):
            continue

        metadata: dict[str, Any] = {
            "trial_id": f"{task_id}:{model_id}",
            "model_id": model_id,
            "profbench_response_field": response_field,
        }
        if include_cached_fulfilments:
            metadata["profbench_fulfilments"] = {
                criterion.id: _coerce_bool(
                    row["rubrics"][index].get(f"{model_id}_fulfilment"),
                    field=f"{model_id}_fulfilment for {criterion.id}",
                )
                for index, criterion in enumerate(criteria)
            }

        trials.append(
            AgentEvalTrial(
                id=f"{task_id}:{model_id}",
                task_id=task_id,
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text=response_text),
                evidence=CandidateEvidence(
                    descriptors={
                        "source": EvidenceDescriptor(
                            kind="profbench",
                            ref=source_uri,
                            format="jsonl",
                            metadata={"task_id": task_id, "response_field": response_field},
                        )
                    }
                ),
                metadata=metadata,
            )
        )
    return trials


def _baseline_fulfilments(metadata: dict[str, Any]) -> dict[str, bool]:
    raw = metadata.get("profbench_fulfilments")
    if not isinstance(raw, dict):
        return {}
    return {str(key): _coerce_bool(value, field=str(key)) for key, value in raw.items()}


def _read_jsonl_source(
    source: str | Path, *, evidence_dir: Path | None = None
) -> tuple[str, list[tuple[int, str]], dict[str, Any]]:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        request = Request(source_text, headers={"User-Agent": "nhx-evals-sdk"})
        with urlopen(request, timeout=60) as response:  # noqa: S310 - profbench dataset source is an operator-supplied http(s) URL
            body = response.read().decode("utf-8")
            headers = dict(response.headers.items())
        metadata = {
            "etag": headers.get("ETag"),
            "resolved_commit": headers.get("x-repo-commit"),
            "remote_source": source_text,
        }
        source_uri = source_text
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = evidence_dir / "profbench-dataset.jsonl"
            dataset_path.write_text(body, encoding="utf-8")
            source_uri = str(dataset_path.resolve())
            metadata["source_file"] = source_uri
        lines = [(index, line) for index, line in enumerate(body.splitlines(), start=1) if line.strip()]
        return source_uri, lines, metadata

    path = Path(source).expanduser().resolve()
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [(index, line) for index, line in enumerate(raw_lines, start=1) if line.strip()]
    return str(path), lines, {"source_file": str(path)}


def _render_judge_prompt(request: ProfBenchJudgeRequest) -> str:
    criterion_type = request.criterion_type
    if isinstance(criterion_type, list):
        criterion_type_text = ", ".join(str(value) for value in criterion_type)
    else:
        criterion_type_text = str(criterion_type or "unspecified")

    return (
        "You are judging whether a professional benchmark response satisfies one rubric criterion.\n"
        "Return only a compact JSON object with keys `fulfilled` (boolean) and `reason` (string). "
        "Do not include markdown, analysis, or explanatory text outside the JSON object.\n\n"
        f"Task prompt:\n{request.prompt}\n\n"
        f"Candidate response:\n{request.response}\n\n"
        f"Criterion id: {request.criterion_id}\n"
        f"Criterion type: {criterion_type_text}\n"
        f"Criterion weight: {request.weight_name}\n"
        f"Criterion:\n{request.criterion_description}\n"
    )


def _parse_judge_decision(text: str, *, raw_response: dict[str, Any] | None = None) -> ProfBenchJudgeDecision:
    """Parse the judge's structured-output JSON; treat anything unparseable as unfulfilled."""
    parsed = _parse_json_object(text.strip())
    if parsed is not None and isinstance(parsed.get("fulfilled"), bool):
        reason = parsed.get("reason")
        return ProfBenchJudgeDecision(
            fulfilled=parsed["fulfilled"],
            reason=reason if isinstance(reason, str) else "",
            raw_response=raw_response,
        )
    return ProfBenchJudgeDecision(
        fulfilled=False,
        reason="Judge response was not parseable JSON with a boolean 'fulfilled'; treating criterion as unfulfilled.",
        raw_response=raw_response,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _default_judge_params() -> RunConfigOnlineModel:
    return RunConfigOnlineModel(
        parallelism=1,
        inference=InferenceParams.model_validate(
            {
                "temperature": 0.0,
                "max_tokens": 256,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_budget": 0,
                },
            }
        ),
        structured_output=PROFBENCH_JUDGE_STRUCTURED_OUTPUT,
    )


def _coerce_bool(value: Any, *, field: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field} must be a boolean-like value; got {value!r}")


def _criterion_type_labels(value: CriterionType | None) -> list[str]:
    if value is None:
        return ["unknown"]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _safe_artifact_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


# --------------------------------------------------------------------------------------
# Dashboard rendering
#
# A minimal, dependency-free HTML report that demonstrates ProfBench's abilities:
# overall score, per-model and per-domain scores, criterion-type fulfilment, and an
# expandable per-trial table of criteria (pass/fail, weight, judge reason, evidence).
# The richer generic view is left to the SDK dashboard written alongside it.
# --------------------------------------------------------------------------------------

_DASHBOARD_STYLE = """
  body { margin:0; background:#f7f8fa; color:#15171a; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { padding:24px 32px; border-bottom:1px solid #d0d5dd; background:#fff; }
  h1 { margin:0 0 6px; font-size:26px; }
  h2 { margin:28px 0 12px; font-size:18px; }
  main { max-width:1100px; margin:0 auto; padding:24px 32px 48px; }
  .hero { display:flex; gap:24px; align-items:flex-end; flex-wrap:wrap; }
  .score { font-size:48px; font-weight:700; line-height:1; color:#0f766e; }
  .muted { color:#667085; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }
  .card { background:#fff; border:1px solid #d0d5dd; border-radius:8px; padding:14px; }
  .card strong { display:block; font-size:22px; margin-top:6px; }
  table { width:100%; border-collapse:collapse; background:#fff; border:1px solid #d0d5dd; }
  th, td { padding:9px 10px; border-bottom:1px solid #d0d5dd; text-align:left; vertical-align:top; }
  th { font-size:12px; text-transform:uppercase; color:#667085; background:#f8fafc; }
  .pass { color:#0f766e; font-weight:600; }
  .fail { color:#b42318; font-weight:600; }
  details { background:#fff; border:1px solid #d0d5dd; border-radius:8px; margin:10px 0; }
  summary { padding:12px 14px; cursor:pointer; }
  details table { border:0; border-top:1px solid #d0d5dd; }
  a { color:#0f766e; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { border:1px solid #d0d5dd; border-radius:999px; padding:2px 8px; color:#667085; text-decoration:none; }
""".strip()


def write_example_dashboards(result: AgentEvalResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write the generic SDK dashboard and the ProfBench-specific report side by side."""
    path = Path(output_dir)
    sdk_dashboard_path = write_sdk_dashboard(result, path / "sdk-report.html")
    dashboard_path = write_profbench_dashboard(result, path / "report.html")
    return sdk_dashboard_path, dashboard_path


def write_profbench_dashboard(result: AgentEvalResult, output_path: str | Path) -> Path:
    """Write the ProfBench-specific HTML report for an example run."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_profbench_dashboard(result, evidence_base_dir=path.parent), encoding="utf-8")
    return path


def render_profbench_dashboard(result: AgentEvalResult, *, evidence_base_dir: str | Path | None = None) -> str:
    """Render a ProfBench-aware HTML report from generic agent-eval results."""
    rows = _profbench_result_rows(result.scores)
    overall = _format_percent(_overall_score(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(result.run_id)} ProfBench Report</title>
  <style>{_DASHBOARD_STYLE}</style>
</head>
<body>
<header>
  <div class="hero">
    <div>
      <h1>ProfBench Agent Eval Report</h1>
      <div class="muted">Run {_e(result.run_id)} · {_e(result.summary.task_count)} tasks · \
{_e(result.summary.trial_count)} trials</div>
    </div>
    <div class="score">{overall}</div>
  </div>
</header>
<main>
  <section>
    <div class="grid">
      {_cards("Model", _scores_by_model(rows))}
      {_cards("Domain", _scores_by_domain(rows))}
      {_cards("Criterion", _criterion_fulfilment(rows))}
      <div class="card"><span class="muted">Failed criteria</span><strong>{_e(_failed_criteria_count(rows))}</strong></div>
    </div>
  </section>
  <section>
    <h2>Task Details</h2>
    {_task_details(rows, evidence_base_dir=evidence_base_dir)}
  </section>
</main>
</body>
</html>
"""


def _profbench_result_rows(
    scores: list[AgentEvalTaskScore],
) -> list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]:
    rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]] = []
    for task_score in scores:
        for output in task_score.outputs:
            details = profbench_details(output)
            if details is not None:
                rows.append((task_score, details))
    return rows


def _overall_score(rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]) -> float | None:
    if not rows:
        return None
    return _mean([details.score for _, details in rows])


def _cards(title: str, values: dict[str, float]) -> str:
    if not values:
        return f'<div class="card"><span class="muted">{_e(title)}</span><strong>n/a</strong></div>'
    return "".join(
        f'<div class="card"><span class="muted">{_e(title)} · {_e(name)}</span>'
        f"<strong>{_format_percent(score)}</strong></div>"
        for name, score in sorted(values.items())
    )


def _scores_by_model(rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for _, details in rows:
        values.setdefault(details.model_id, []).append(details.score)
    return _mean_by_key(values)


def _scores_by_domain(rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for _, details in rows:
        values.setdefault(details.domain or "unknown", []).append(details.score)
    return _mean_by_key(values)


def _criterion_fulfilment(rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for _, details in rows:
        for criterion in details.criterion_scores:
            for criterion_type in _criterion_type_labels(criterion.criterion_type):
                values.setdefault(criterion_type, []).append(1.0 if criterion.fulfilled else 0.0)
    return _mean_by_key(values)


def _failed_criteria_count(rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]]) -> int:
    return sum(1 for _, details in rows for criterion in details.criterion_scores if not criterion.fulfilled)


def _task_details(
    rows: list[tuple[AgentEvalTaskScore, ProfBenchRubricDetails]],
    *,
    evidence_base_dir: str | Path | None,
) -> str:
    rendered = []
    for task_score, details in rows:
        criterion_rows = "".join(
            "<tr>"
            f"<td>{_e(criterion.criterion_id)}</td>"
            f"<td>{_e(criterion.weight_name)}</td>"
            f"<td>{_e(_criterion_type_label(criterion.criterion_type))}</td>"
            f"<td>{criterion.points:g}</td>"
            f'<td class="{"pass" if criterion.fulfilled else "fail"}">{"yes" if criterion.fulfilled else "no"}</td>'
            f"<td>{_e(criterion.description)}</td>"
            f"<td>{_e(criterion.metadata.get('score_source', ''))}</td>"
            f"<td>{_e(criterion.judge_reason or '')}</td>"
            f"<td>{_evidence_links(criterion.evidence, evidence_base_dir=evidence_base_dir)}</td>"
            "</tr>"
            for criterion in details.criterion_scores
        )
        rendered.append(
            "<details>"
            f"<summary>{_e(task_score.task_id)} · {_e(details.model_id)} · {_format_percent(details.score)}</summary>"
            "<table><thead><tr><th>Criterion</th><th>Weight</th><th>Type</th><th>Points</th>"
            "<th>Fulfilled</th><th>Description</th><th>Source</th><th>Reason</th><th>Evidence</th></tr></thead>"
            f"<tbody>{criterion_rows}</tbody></table>"
            "</details>"
        )
    return "".join(rendered)


def _evidence_links(locators: list[EvidenceLocator], *, evidence_base_dir: str | Path | None) -> str:
    if not locators:
        return ""
    return (
        '<div class="chips">'
        + "".join(_evidence_link(locator, evidence_base_dir=evidence_base_dir) for locator in locators)
        + "</div>"
    )


def _evidence_link(locator: EvidenceLocator, *, evidence_base_dir: str | Path | None) -> str:
    label = locator.label or locator.kind
    if locator.line is not None:
        label = f"{label}:L{locator.line}"
    if locator.json_path:
        label = f"{label} {locator.json_path}"
    return f'<a class="chip" href="{_e(locator.href(base_dir=evidence_base_dir))}">{_e(label)}</a>'


def _mean_by_key(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: _mean(scores) for key, scores in values.items()}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _criterion_type_label(value: CriterionType | None) -> str:
    return ", ".join(_criterion_type_labels(value))


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)
