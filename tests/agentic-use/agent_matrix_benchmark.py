#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and compare agentic-use tasks across direct agent backends."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Literal

import nat_runner
from pydantic import BaseModel, ConfigDict, Field, model_validator

TASKS_DIR = Path(__file__).resolve().parent
NAT_RUNNER = TASKS_DIR / "nat_runner.py"
DEFAULT_BACKENDS = ("codex", "claude-code", "cursor-agent")
SUPPORTED_BACKENDS = frozenset({"codex", "claude-code", "cursor-agent"})
REPORT_JSON = "benchmark_summary.json"
REPORT_MARKDOWN = "benchmark_summary.md"
REPORT_HTML = "benchmark_report.html"
AgentBackend = Literal["codex", "claude-code", "cursor-agent"]
ConfigScalar = str | int | float | bool | None
ConfigValue = ConfigScalar | list[ConfigScalar]
LEADERBOARD_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("rank", "Rank", True),
    ("candidate", "Candidate", True),
    ("backend", "Backend", True),
    ("model", "Model", True),
    ("params", "Params", True),
    ("pass-rate", "Pass Rate", True),
    ("passed", "Passed", True),
    ("tokens", "Tokens", True),
    ("token-coverage", "Token Coverage", True),
    ("input", "Input", True),
    ("output", "Output", True),
    ("cache-read", "Cache Read", True),
    ("cache-write", "Cache Write", True),
    ("runtime", "Runtime", True),
)
LEADERBOARD_FROZEN_COLUMN_IDS = frozenset({"rank", "candidate"})


class CodexParams(BaseModel):
    """Codex-specific benchmark candidate parameters."""

    model_config = ConfigDict(extra="forbid")

    intelligence: Literal["low", "medium", "high", "xhigh"] | None = None
    speed: Literal["standard", "fast"] | None = None
    config: dict[str, ConfigValue] = Field(default_factory=dict)


class ClaudeCodeParams(BaseModel):
    """Claude Code-specific benchmark candidate parameters."""

    model_config = ConfigDict(extra="forbid")

    permission_mode: Literal["acceptEdits", "bypassPermissions", "default", "delegate", "dontAsk", "plan"] | None = None
    max_budget_usd: float | None = None


class CursorAgentParams(BaseModel):
    """Cursor Agent-specific benchmark candidate parameters."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["plan", "ask"] | None = None
    sandbox: Literal["enabled", "disabled"] | None = None


CandidateParams = CodexParams | ClaudeCodeParams | CursorAgentParams


class CandidateConfig(BaseModel):
    """Typed Python-config candidate definition."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    backend: AgentBackend
    model: str | None = None
    params: CandidateParams | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_backend_params(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw = _object_dict(value)
        backend = raw.get("backend")
        params = raw.get("params")
        if backend == "codex" and not isinstance(params, CodexParams):
            raw["params"] = CodexParams.model_validate(params or {})
        elif backend == "claude-code" and not isinstance(params, ClaudeCodeParams):
            raw["params"] = ClaudeCodeParams.model_validate(params or {})
        elif backend == "cursor-agent" and not isinstance(params, CursorAgentParams):
            raw["params"] = CursorAgentParams.model_validate(params or {})
        return raw

    @model_validator(mode="after")
    def _validate_backend_params(self) -> CandidateConfig:
        expected_types: dict[str, type[BaseModel]] = {
            "codex": CodexParams,
            "claude-code": ClaudeCodeParams,
            "cursor-agent": CursorAgentParams,
        }
        expected = expected_types[self.backend]
        if self.params is not None and not isinstance(self.params, expected):
            raise ValueError(f"{self.backend} candidates must use {expected.__name__}")
        return self

    def to_candidate(self) -> Candidate:
        params = self.params.model_dump(exclude_none=True) if self.params is not None else {}
        return Candidate(
            backend=self.backend,
            model=self.model,
            candidate_id=self.id or _candidate_id(backend=self.backend, model=self.model, params=params),
            params=params,
        )


class MatrixDefaults(BaseModel):
    """Optional operational defaults for a matrix config."""

    model_config = ConfigDict(extra="forbid")

    timeout: int | None = None
    skip_build: bool | None = None
    allow_dirty: bool | None = None
    parallel_candidates: int | None = None
    nhx_base_url: str | None = None
    anthropic_base_url: str | None = None


class AgentMatrixConfig(BaseModel):
    """Python-config schema for an agent/model benchmark matrix."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    manifest: str | None = None
    tasks: list[str] = Field(default_factory=list)
    all_tasks: bool = False
    defaults: MatrixDefaults = Field(default_factory=MatrixDefaults)
    candidates: list[CandidateConfig]

    @model_validator(mode="after")
    def _validate_task_selection(self) -> AgentMatrixConfig:
        selections = sum([self.manifest is not None, bool(self.tasks), self.all_tasks])
        if selections != 1:
            raise ValueError("Matrix config must set exactly one of manifest, tasks, or all_tasks")
        return self


@dataclass(frozen=True)
class Candidate:
    """One backend/model candidate in the agent matrix."""

    backend: str
    model: str | None
    candidate_id: str
    params: dict[str, object] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class MatrixRow:
    """One normalized candidate/task result row."""

    candidate_id: str
    agent_backend: str
    agent_model: str
    candidate_params: dict[str, object]
    task: str
    status: str
    passed: bool
    reward: float
    runtime_sec: float | None
    total_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_read_tokens: int | None
    cache_creation_tokens: int | None
    token_metrics_status: str
    verifier_scores: dict[str, object] | None
    result_path: str | None
    output_dir: str | None
    failure_excerpt: str | None
    provenance: dict[str, object]


@dataclass(frozen=True)
class CandidateSummary:
    """Aggregate result for one candidate."""

    candidate_id: str
    agent_backend: str
    agent_model: str
    candidate_params: dict[str, object]
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    total_tokens_sum: int | None
    prompt_tokens_sum: int | None
    completion_tokens_sum: int | None
    cache_read_tokens_sum: int | None
    cache_creation_tokens_sum: int | None
    token_metrics_coverage: float
    runtime_sec_sum: float | None
    rank: int | None = None


def parse_candidate(spec: str) -> Candidate:
    """Parse ``backend[:model]`` into a matrix candidate."""
    backend, separator, model = spec.partition(":")
    backend = backend.strip()
    model = model.strip() if separator else None
    if not backend or backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported agent backend {backend!r}; expected one of {sorted(SUPPORTED_BACKENDS)}")
    if separator and not model:
        raise ValueError(f"Agent candidate {spec!r} has an empty model")
    candidate_id = _candidate_id(backend=backend, model=model, params={})
    return Candidate(backend=backend, model=model, candidate_id=candidate_id)


def parse_candidates(specs: list[str] | None, model_specs: list[str] | None = None) -> list[Candidate]:
    """Parse backend/model candidate specs and reject duplicate ids.

    ``--agent backend[:model]`` remains the terse candidate form. ``--agent
    backend`` can also be paired with repeated ``--agent-model backend:model``
    entries to express a grid of sub-models under a selected backend.
    """
    if not specs and not model_specs:
        candidates = [parse_candidate(spec) for spec in DEFAULT_BACKENDS]
        _validate_unique_candidates(candidates)
        return candidates

    candidates: list[Candidate] = []
    default_backends: list[str] = []
    model_backends: set[str] = set()
    for spec in model_specs or []:
        candidate = parse_candidate(spec)
        if candidate.model is None:
            raise ValueError(f"Agent model spec {spec!r} must use backend:model")
        candidates.append(candidate)
        model_backends.add(candidate.backend)

    for spec in specs or []:
        candidate = parse_candidate(spec)
        if candidate.model is None:
            default_backends.append(candidate.backend)
        else:
            candidates.append(candidate)
            model_backends.add(candidate.backend)

    for backend in default_backends:
        if backend not in model_backends:
            candidates.append(parse_candidate(backend))

    _validate_unique_candidates(candidates)
    return candidates


def _validate_unique_candidates(candidates: list[Candidate]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            duplicates.append(candidate.candidate_id)
        seen.add(candidate.candidate_id)
    if duplicates:
        raise ValueError(f"Duplicate candidate id(s): {', '.join(sorted(set(duplicates)))}")


def build_nat_runner_command(
    *,
    candidate: Candidate,
    jobs_dir: Path,
    manifest: Path | None,
    tasks: list[str],
    all_tasks: bool,
    skip_build: bool,
    allow_dirty: bool,
    timeout: int,
    codex_auth_json: Path | None,
    nhx_base_url: str,
    anthropic_base_url: str,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the subprocess command used to run one matrix candidate."""
    command = [
        python_executable,
        str(NAT_RUNNER),
        "--agent-backend",
        candidate.backend,
        "--candidate-id",
        candidate.candidate_id,
        "--jobs-dir",
        str(jobs_dir),
        "--timeout",
        str(timeout),
        "--nhx-base-url",
        nhx_base_url,
    ]
    if candidate.model is not None:
        command.extend(["--agent-model", candidate.model])
    if candidate.params:
        command.extend(["--candidate-params", json.dumps(candidate.params, sort_keys=True)])
    if candidate.backend == "claude-code":
        command.extend(["--anthropic-base-url", anthropic_base_url])
    if candidate.backend == "codex" and codex_auth_json is not None:
        command.extend(["--codex-auth-json", str(codex_auth_json)])
    if skip_build:
        command.append("--skip-build")
    if allow_dirty:
        command.append("--allow-dirty")
    if all_tasks:
        command.append("--all")
    elif manifest is not None:
        command.extend(["--manifest", str(manifest)])
    else:
        command.extend(tasks)
    return command


def resolve_selected_tasks(*, manifest: Path | None, tasks: list[str], all_tasks: bool) -> list[str]:
    """Resolve the matrix task selection using nat_runner's task resolver."""
    if all_tasks:
        return nat_runner.resolve_tasks([])
    if manifest is not None:
        patterns = nat_runner._read_manifest(manifest)  # noqa: SLF001 - share nat_runner manifest semantics.
        return nat_runner.resolve_tasks(patterns)
    return nat_runner.resolve_tasks(tasks)


def load_matrix_config(path: Path) -> AgentMatrixConfig:
    """Load a Python matrix config file containing ``MATRIX``."""
    config_path = path.expanduser().resolve()
    module = _load_python_module(config_path)
    if not hasattr(module, "MATRIX"):
        raise ValueError(f"Matrix config {config_path} must define MATRIX")
    raw_config = module.MATRIX
    if isinstance(raw_config, BaseModel):
        raw_config = raw_config.model_dump()
    return AgentMatrixConfig.model_validate(raw_config)


def run_candidates(
    *,
    candidates: list[Candidate],
    run_dir: Path,
    selected_tasks: list[str],
    manifest: Path | None,
    tasks: list[str],
    all_tasks: bool,
    skip_build: bool,
    allow_dirty: bool,
    timeout: int,
    codex_auth_json: Path | None,
    nhx_base_url: str,
    anthropic_base_url: str,
    parallel_candidates: int = 1,
) -> list[dict[str, object]]:
    """Run all candidates and return subprocess metadata."""
    if parallel_candidates < 1:
        raise ValueError("parallel_candidates must be >= 1")

    run_metadata: list[dict[str, object]] = []
    total_candidates = len(candidates)
    if parallel_candidates == 1 or total_candidates <= 1:
        for index, candidate in enumerate(candidates, start=1):
            run_metadata.append(
                _run_candidate(
                    index=index,
                    total_candidates=total_candidates,
                    candidate=candidate,
                    run_dir=run_dir,
                    selected_tasks=selected_tasks,
                    manifest=manifest,
                    tasks=tasks,
                    all_tasks=all_tasks,
                    skip_build=skip_build,
                    allow_dirty=allow_dirty,
                    timeout=timeout,
                    codex_auth_json=codex_auth_json,
                    nhx_base_url=nhx_base_url,
                    anthropic_base_url=anthropic_base_url,
                )
            )
        return run_metadata

    worker_count = min(parallel_candidates, total_candidates)
    _log(f"Running candidates with parallelism={worker_count}; each candidate still runs its tasks serially.")
    metadata_by_index: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _run_candidate,
                index=index,
                total_candidates=total_candidates,
                candidate=candidate,
                run_dir=run_dir,
                selected_tasks=selected_tasks,
                manifest=manifest,
                tasks=tasks,
                all_tasks=all_tasks,
                skip_build=skip_build,
                allow_dirty=allow_dirty,
                timeout=timeout,
                codex_auth_json=codex_auth_json,
                nhx_base_url=nhx_base_url,
                anthropic_base_url=anthropic_base_url,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        for future in as_completed(futures):
            metadata = future.result()
            index_value = metadata.get("candidate_index")
            if isinstance(index_value, int):
                metadata_by_index[index_value] = metadata

    return [metadata_by_index[index] for index in sorted(metadata_by_index)]


def _run_candidate(
    *,
    index: int,
    total_candidates: int,
    candidate: Candidate,
    run_dir: Path,
    selected_tasks: list[str],
    manifest: Path | None,
    tasks: list[str],
    all_tasks: bool,
    skip_build: bool,
    allow_dirty: bool,
    timeout: int,
    codex_auth_json: Path | None,
    nhx_base_url: str,
    anthropic_base_url: str,
) -> dict[str, object]:
    """Run one candidate subprocess and return normalized metadata."""
    candidate_dir = run_dir / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    command = build_nat_runner_command(
        candidate=candidate,
        jobs_dir=candidate_dir,
        manifest=manifest,
        tasks=tasks,
        all_tasks=all_tasks,
        skip_build=skip_build,
        allow_dirty=allow_dirty,
        timeout=timeout,
        codex_auth_json=codex_auth_json,
        nhx_base_url=nhx_base_url,
        anthropic_base_url=anthropic_base_url,
    )
    _log(
        "Candidate "
        f"{index}/{total_candidates} START {candidate.candidate_id} "
        f"(backend={candidate.backend}, model={candidate.model or 'default'}, tasks={len(selected_tasks)})"
    )
    _log(f"Artifacts: {candidate_dir}")
    _log(f"Command: {' '.join(command)}")
    started_at = datetime.now(UTC).isoformat()
    started_monotonic = time.monotonic()
    completed = subprocess.run(command, check=False)
    finished_at = datetime.now(UTC).isoformat()
    duration_sec = time.monotonic() - started_monotonic
    _log(
        "Candidate "
        f"{index}/{total_candidates} DONE {candidate.candidate_id} "
        f"returncode={completed.returncode} elapsed={_format_seconds(duration_sec)}"
    )
    return {
        "candidate_id": candidate.candidate_id,
        "agent_backend": candidate.backend,
        "agent_model": candidate.model or "default",
        "candidate_params": candidate.params,
        "candidate_index": index,
        "jobs_dir": str(candidate_dir),
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": round(duration_sec, 3),
    }


def build_matrix_summary(
    *,
    run_dir: Path,
    candidates: list[Candidate],
    tasks: list[str],
    run_metadata: list[dict[str, object]] | None = None,
    manifest: Path | None = None,
) -> dict[str, object]:
    """Build the canonical matrix summary payload."""
    rows: list[MatrixRow] = []
    for candidate in candidates:
        rows.extend(_load_candidate_rows(run_dir=run_dir, candidate=candidate, tasks=tasks))

    summaries = _rank_candidate_summaries(_summarize_candidates(candidates=candidates, tasks=tasks, rows=rows))
    row_payloads = [asdict(row) for row in rows]
    summary_payloads = [asdict(summary) for summary in summaries]
    generated_at = datetime.now(UTC).isoformat()
    winner = summary_payloads[0]["candidate_id"] if summary_payloads else None

    return {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "manifest": str(manifest) if manifest is not None else None,
        "tasks": tasks,
        "winner": winner,
        "ranking": summary_payloads,
        "candidate_groups": _candidate_groups(summary_payloads),
        "rows": row_payloads,
        "run_metadata": run_metadata or [],
    }


def write_reports(summary: dict[str, object], output_dir: Path) -> dict[str, str]:
    """Write JSON, Markdown, and static HTML reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / REPORT_JSON,
        "markdown": output_dir / REPORT_MARKDOWN,
        "html": output_dir / REPORT_HTML,
    }
    paths["json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(summary), encoding="utf-8")
    paths["html"].write_text(render_html(summary), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def render_markdown(summary: dict[str, object]) -> str:
    """Render a compact Markdown benchmark report."""
    ranking = _list_of_dicts(summary.get("ranking"))
    candidate_groups = _dict_of_list_of_dicts(summary.get("candidate_groups"))
    rows = _list_of_dicts(summary.get("rows"))
    tasks = _string_list(summary.get("tasks"))
    lines = [
        "# Agent Matrix Benchmark",
        "",
        f"- Generated: `{summary.get('generated_at')}`",
        f"- Run dir: `{summary.get('run_dir')}`",
        f"- Winner: `{summary.get('winner') or 'n/a'}`",
        "",
        "## Ranking",
        "",
        "| Rank | Candidate | Backend | Model | Pass Rate | Passed | Tokens | Token Coverage | Input | Output | Cache Read | Cache Write | Runtime |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ranking:
        lines.append(
            "| {rank} | `{candidate_id}` | `{agent_backend}` | `{agent_model}` | {pass_rate:.3f} | "
            "{passed_tasks}/{total_tasks} | {tokens} | {token_coverage:.0%} | {input_tokens} | {output_tokens} | "
            "{cache_read_tokens} | {cache_creation_tokens} | {runtime} |".format(
                rank=candidate.get("rank"),
                candidate_id=candidate.get("candidate_id"),
                agent_backend=candidate.get("agent_backend"),
                agent_model=candidate.get("agent_model"),
                pass_rate=_float_value(candidate.get("pass_rate")),
                passed_tasks=candidate.get("passed_tasks"),
                total_tasks=candidate.get("total_tasks"),
                tokens=_format_number(candidate.get("total_tokens_sum")),
                token_coverage=_float_value(candidate.get("token_metrics_coverage")),
                input_tokens=_format_number(candidate.get("prompt_tokens_sum")),
                output_tokens=_format_number(candidate.get("completion_tokens_sum")),
                cache_read_tokens=_format_number(candidate.get("cache_read_tokens_sum")),
                cache_creation_tokens=_format_number(candidate.get("cache_creation_tokens_sum")),
                runtime=_format_seconds(candidate.get("runtime_sec_sum")),
            )
        )
    lines.extend(
        [
            "",
            "Token bucket note: `n/a` means the backend did not emit that token bucket; `0` means it emitted the bucket with value zero.",
        ]
    )

    if candidate_groups:
        lines.extend(["", "## Agent Model Groups", ""])
        for backend in sorted(candidate_groups):
            models = ", ".join(
                f"`{candidate.get('agent_model')}` ({_format_params(candidate.get('candidate_params'))})"
                for candidate in candidate_groups[backend]
            )
            lines.append(f"- `{backend}`: {models}")

    lines.extend(["", "## Task Matrix", "", _render_markdown_matrix(tasks=tasks, rows=rows)])
    failures = [row for row in rows if row.get("status") != "passed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            result_path = row.get("result_path") or ""
            lines.append(f"- `{row.get('candidate_id')}` / `{row.get('task')}`: {row.get('status')} `{result_path}`")
    return "\n".join(lines) + "\n"


def render_html(summary: dict[str, object]) -> str:
    """Render a self-contained browser-viewable benchmark report."""
    ranking = _list_of_dicts(summary.get("ranking"))
    candidate_groups = _dict_of_list_of_dicts(summary.get("candidate_groups"))
    rows = _list_of_dicts(summary.get("rows"))
    tasks = _string_list(summary.get("tasks"))
    winner = str(summary.get("winner") or "n/a")
    run_dir = str(summary.get("run_dir") or "")

    cards = "\n".join(_render_candidate_card(candidate) for candidate in ranking)
    group_cards = _render_candidate_groups(candidate_groups)
    leaderboard = _render_leaderboard(ranking)
    task_matrix = _render_html_task_matrix(tasks=tasks, ranking=ranking, rows=rows)
    provenance_badges = _render_provenance_badges(rows)
    token_chart = _render_bar_chart(ranking, value_key="total_tokens_sum", label="Total Tokens", css_class="tokens")
    runtime_chart = _render_bar_chart(
        ranking, value_key="runtime_sec_sum", label="Runtime Seconds", css_class="runtime"
    )
    failures = "\n".join(
        _render_failure_detail(row, run_dir=Path(run_dir)) for row in rows if row.get("status") != "passed"
    )
    if not failures:
        failures = '<p class="muted">No failures recorded.</p>'

    payload = html.escape(json.dumps(summary, indent=2), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Matrix Benchmark</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #657086;
      --line: #d9deea;
      --line-strong: #c9d1e2;
      --pass: #147a3f;
      --pass-bg: #e8f6ee;
      --fail: #b42318;
      --fail-bg: #fee9e7;
      --missing: #805b10;
      --missing-bg: #fff4d6;
      --accent: #4f46e5;
      --accent-soft: #eef2ff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px max(24px, 4vw);
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .page-shell {{
      max-width: 1480px;
      margin: 0 auto;
    }}
    main {{
      padding: 26px max(24px, 4vw) 52px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 32px;
      line-height: 1.1;
    }}
    h2 {{
      margin-top: 34px;
      margin-bottom: 12px;
      font-size: 21px;
    }}
    h3 {{
      font-size: 15px;
      line-height: 1.25;
    }}
    .meta, .muted {{
      color: var(--muted);
    }}
    .meta {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      font-size: 13px;
    }}
    .meta span {{
      overflow-wrap: anywhere;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(280px, 420px);
      gap: 24px;
      align-items: center;
    }}
    .winner {{
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff, #f8faff);
    }}
    .winner strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      color: var(--accent);
      overflow-wrap: anywhere;
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .score-card {{
      display: grid;
      min-height: 142px;
      align-content: space-between;
    }}
    .score-card h3 {{
      overflow-wrap: anywhere;
    }}
    .candidate-subtitle {{
      color: var(--muted);
      margin-top: 3px;
      overflow-wrap: anywhere;
    }}
    .metric {{
      margin-top: 10px;
      font-size: 30px;
      font-weight: 700;
    }}
    .passbar {{
      height: 8px;
      margin-top: 10px;
      border-radius: 999px;
      background: #eef1f7;
      overflow: hidden;
    }}
    .passbar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
      background: var(--accent);
    }}
    .model-groups {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .model-group {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }}
    .model-group h3 {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .model-row {{
      display: grid;
      grid-template-columns: minmax(0, 0.7fr) minmax(0, 1fr) minmax(0, 1.2fr) minmax(72px, auto);
      gap: 12px;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }}
    .model-row:last-child {{
      border-bottom: 0;
    }}
    .model-name, .candidate-name {{
      overflow-wrap: anywhere;
    }}
    .candidate-name {{
      font-weight: 650;
    }}
    .param-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-width: 0;
    }}
    .chip {{
      display: inline-flex;
      max-width: 100%;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #3730a3;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .chip.neutral {{
      background: #eef1f7;
      color: var(--muted);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
    }}
    .badge.dirty {{
      border-color: #fed7aa;
      background: #fff7ed;
      color: #9a3412;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .leaderboard-frame {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }}
    .leaderboard-frozen {{
      border-right: 1px solid var(--line-strong);
      background: var(--panel);
      z-index: 1;
    }}
    .leaderboard-frozen,
    .leaderboard-scroll-wrap {{
      min-width: 0;
    }}
    .leaderboard-scroll-wrap {{
      overflow-x: auto;
      overflow-y: hidden;
      border: 0;
      border-radius: 0;
      border-top-right-radius: 8px;
      border-bottom-right-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      min-width: 780px;
    }}
    .leaderboard-tools {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .table-hint {{
      font-size: 12px;
      color: var(--muted);
    }}
    .column-picker {{
      position: relative;
      margin: 0;
    }}
    .column-picker summary {{
      cursor: pointer;
      list-style: none;
      font-size: 13px;
      font-weight: 600;
      color: var(--accent);
    }}
    .column-picker summary::-webkit-details-marker {{
      display: none;
    }}
    .column-menu {{
      margin-top: 8px;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 12px 24px rgba(23, 32, 51, 0.08);
      min-width: min(720px, calc(100vw - 64px));
      max-width: 920px;
    }}
    .column-option {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text);
      min-width: 0;
    }}
    .column-option input {{
      margin: 0;
    }}
    .leaderboard-table {{
      width: max-content;
      min-width: 1320px;
    }}
    .leaderboard-table col.rank-col {{
      width: 72px;
    }}
    .leaderboard-table col.candidate-col {{
      width: 260px;
    }}
    .leaderboard-table col.backend-col {{
      width: 130px;
    }}
    .leaderboard-table col.model-col {{
      width: 220px;
    }}
    .leaderboard-table col.params-col {{
      width: 250px;
    }}
    .leaderboard-table col.runtime-col {{
      width: 90px;
    }}
    .leaderboard-table [data-col].is-hidden {{
      display: none;
    }}
    .leaderboard-table col.is-hidden {{
      visibility: collapse;
    }}
    .leaderboard-table th {{
      position: relative;
      white-space: nowrap;
    }}
    .leaderboard-frozen-table {{
      min-width: 332px;
      width: max-content;
    }}
    .leaderboard-frozen-table col.rank-col {{
      width: 72px;
    }}
    .leaderboard-frozen-table col.candidate-col {{
      width: 260px;
    }}
    .leaderboard-scroll-table {{
      width: max-content;
      min-width: 988px;
    }}
    .leaderboard-table th.resizable {{
      padding-right: 18px;
    }}
    .resize-handle {{
      position: absolute;
      top: 0;
      right: 0;
      width: 12px;
      height: 100%;
      cursor: col-resize;
      user-select: none;
    }}
    .resize-handle::after {{
      content: "";
      position: absolute;
      top: 22%;
      bottom: 22%;
      left: 5px;
      width: 2px;
      border-radius: 999px;
      background: #d1d8e8;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      background: #f0f3f9;
      font-weight: 600;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .numeric {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .matrix-table {{
      min-width: 1180px;
    }}
    .matrix-table th:first-child,
    .matrix-table td:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--panel);
      border-right: 1px solid var(--line-strong);
      min-width: 320px;
    }}
    .matrix-table th:first-child {{
      background: #f0f3f9;
    }}
    .task-result-cell {{
      min-width: 170px;
    }}
    .task-result-status {{
      margin-bottom: 8px;
    }}
    .status-legend {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .status-legend-details {{
      margin: 0 0 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 12px 12px;
    }}
    .status-legend-details summary {{
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
    }}
    .status-legend-item {{
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 10px;
      align-items: start;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      min-width: 0;
    }}
    .status-legend-item strong {{
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
    }}
    .status-legend-item span:last-child {{
      font-size: 12px;
      line-height: 1.35;
      color: var(--muted);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-width: 58px;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .passed {{ color: var(--pass); background: var(--pass-bg); }}
    .failed, .agent_failed, .verify_failed {{ color: var(--fail); background: var(--fail-bg); }}
    .missing {{ color: var(--missing); background: var(--missing-bg); }}
    .bars {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .token-buckets {{
      margin: 0;
      display: grid;
      grid-template-columns: minmax(72px, max-content) minmax(60px, 1fr);
      gap: 3px 12px;
      align-items: baseline;
      font-size: 12px;
    }}
    .token-buckets dt {{
      color: var(--muted);
    }}
    .token-buckets dd {{
      margin: 0;
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(110px, 170px) 1fr minmax(70px, max-content);
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}
    .bar-track {{
      display: block;
      height: 12px;
      background: #edf0f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      display: block;
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }}
    .score-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }}
    .score-chip {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
    }}
    .score-chip.good {{
      color: var(--pass);
      background: var(--pass-bg);
    }}
    .score-chip.bad {{
      color: var(--fail);
      background: var(--fail-bg);
    }}
    .score-chip.neutral {{
      color: var(--muted);
      background: #eef1f7;
    }}
    .failure {{
      margin-bottom: 12px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .failure-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .failure-header h3 {{
      margin: 0;
    }}
    .failure-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .diagnosis {{
      padding: 12px;
      border-radius: 8px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #7c2d12;
      margin-bottom: 12px;
    }}
    .diagnosis strong {{
      display: block;
      margin-bottom: 3px;
    }}
    .check-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .check-card {{
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fbfcff;
    }}
    .check-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .check-card strong {{
      font-size: 18px;
    }}
    .detail-table {{
      margin-top: 8px;
      min-width: 0;
    }}
    .detail-table table {{
      min-width: 0;
    }}
    .detail-table td:first-child {{
      width: 240px;
      color: var(--muted);
      font-weight: 650;
    }}
    .details-json {{
      max-height: 180px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #101828;
      color: #f8fafc;
      padding: 12px;
      border-radius: 6px;
      font-size: 12px;
      max-height: 240px;
      overflow: auto;
    }}
    .failure details {{
      margin-top: 10px;
    }}
    a {{ color: var(--accent); }}
    details {{
      margin-top: 24px;
    }}
    @media (max-width: 780px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .model-groups {{ grid-template-columns: 1fr; }}
      .model-row {{ grid-template-columns: 1fr; gap: 6px; }}
      th, td {{ font-size: 12px; padding: 8px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="page-shell hero">
      <div>
        <h1>Agent Matrix Benchmark</h1>
        <div class="meta">
          <span>Generated: {html.escape(str(summary.get("generated_at") or ""))}</span>
          <span>Run dir: {html.escape(run_dir)}</span>
          {provenance_badges}
        </div>
      </div>
      <div class="winner">
        <span class="muted">Overall winner</span>
        <strong>{html.escape(winner)}</strong>
      </div>
    </div>
  </header>
  <main>
    <div class="page-shell">
    <section class="score-grid">{cards}</section>

    <h2>Candidate Leaderboard</h2>
    {leaderboard}

    <h2>Task Matrix</h2>
    {_render_status_legend()}
    {task_matrix}

    <h2>Agents And Models</h2>
    {group_cards}

    <h2>Token And Runtime Comparison</h2>
    <div class="bars">
      <div class="panel">{token_chart}</div>
      <div class="panel">{runtime_chart}</div>
    </div>

    <h2>Failure Details</h2>
    {failures}

    <h2>Provenance</h2>
    <div class="panel">{_render_provenance(rows)}</div>

    <details>
      <summary>Raw summary JSON</summary>
      <pre>{payload}</pre>
    </details>
    </div>
  </main>
  <script>
    (() => {{
      const leaderboard = document.querySelector('[data-role="candidate-leaderboard"]');
      if (!leaderboard) return;
      const frozenTable = leaderboard.querySelector('[data-role="leaderboard-frozen-table"]');
      const scrollTable = leaderboard.querySelector('[data-role="leaderboard-scroll-table"]');

      const syncRowHeights = () => {{
        if (!frozenTable || !scrollTable) return;
        const frozenRows = [...frozenTable.querySelectorAll('tr')];
        const scrollRows = [...scrollTable.querySelectorAll('tr')];
        const rowCount = Math.min(frozenRows.length, scrollRows.length);
        for (let index = 0; index < rowCount; index += 1) {{
          const frozenRow = frozenRows[index];
          const scrollRow = scrollRows[index];
          frozenRow.style.height = '';
          scrollRow.style.height = '';
          const height = Math.max(frozenRow.getBoundingClientRect().height, scrollRow.getBoundingClientRect().height);
          frozenRow.style.height = `${{height}}px`;
          scrollRow.style.height = `${{height}}px`;
        }}
      }};

      const toggleColumn = (column, visible) => {{
        leaderboard.querySelectorAll(`[data-col="${{column}}"]`).forEach((element) => {{
          element.classList.toggle('is-hidden', !visible);
        }});
        syncRowHeights();
      }};

      leaderboard.querySelectorAll('[data-column-toggle]').forEach((toggle) => {{
        const apply = () => toggleColumn(toggle.value, toggle.checked);
        apply();
        toggle.addEventListener('change', apply);
      }});

      leaderboard.querySelectorAll('th.resizable').forEach((header) => {{
        const handle = header.querySelector('.resize-handle');
        const column = header.dataset.col;
        if (!handle || !column) return;
        handle.addEventListener('mousedown', (event) => {{
          event.preventDefault();
          const table = header.closest('table');
          const col = table ? table.querySelector(`col[data-col="${{column}}"]`) : null;
          const startX = event.clientX;
          const startWidth = (col || header).getBoundingClientRect().width;
          const minWidth = Number(header.dataset.minWidth || 96);
          const onMove = (moveEvent) => {{
            const width = Math.max(minWidth, startWidth + moveEvent.clientX - startX);
            if (col) {{
              col.style.width = `${{width}}px`;
            }} else {{
              header.style.width = `${{width}}px`;
            }}
            syncRowHeights();
          }};
          const onUp = () => {{
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
          }};
          window.addEventListener('mousemove', onMove);
          window.addEventListener('mouseup', onUp);
        }});
      }});
      window.addEventListener('resize', syncRowHeights);
      syncRowHeights();
    }})();
  </script>
</body>
</html>
"""


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run an agentic-use agent matrix benchmark.")
    parser.add_argument("--config", type=Path, help="Python matrix config file defining MATRIX.")
    parser.add_argument("tasks", nargs="*", metavar="TASK_OR_GLOB", help="Task names or glob patterns.")
    parser.add_argument("--all", action="store_true", help="Include all agentic-use tasks.")
    parser.add_argument(
        "--manifest", type=Path, help="Task manifest file. Relative paths resolve from tests/agentic-use/."
    )
    parser.add_argument(
        "--agent",
        action="append",
        help=(
            "Candidate backend[:model]. May be repeated. Use backend-only with --agent-model "
            "to run a model collection under one backend."
        ),
    )
    parser.add_argument(
        "--agent-model",
        action="append",
        help="Model candidate as backend:model. May be repeated, for example --agent-model codex:gpt-5.1.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=nat_runner.DEFAULT_JOBS_DIR / "agent-matrix",
        help="Root directory for matrix artifacts.",
    )
    parser.add_argument("--run-id", help="Optional run id. Defaults to a UTC timestamp.")
    parser.add_argument("--skip-build", action="store_true", help="Forward --skip-build to nat_runner.py.")
    parser.add_argument("--allow-dirty", action="store_true", help="Forward --allow-dirty to nat_runner.py.")
    parser.add_argument("--timeout", type=int, help="Agent timeout in seconds.")
    parser.add_argument(
        "--parallel-candidates",
        type=int,
        default=None,
        help=(
            "Number of candidate backends to run concurrently. Each candidate still runs its selected tasks serially. "
            "Values greater than 1 require --skip-build to avoid concurrent Docker tag rebuilds."
        ),
    )
    parser.add_argument("--nhx-base-url")
    parser.add_argument(
        "--anthropic-base-url",
        default=None,
    )
    parser.add_argument("--codex-auth-json", type=Path, help="Explicit Codex auth.json path for the codex backend.")
    args = parser.parse_args()

    if args.config is not None and (
        args.all or args.manifest is not None or args.tasks or args.agent or args.agent_model
    ):
        print(
            "ERROR: --config is authoritative for tasks and candidates; do not combine it with "
            "--all, --manifest, TASK_OR_GLOB, --agent, or --agent-model.",
            file=sys.stderr,
        )
        return 1
    if args.config is None:
        if args.all and args.manifest is not None:
            print("ERROR: --all and --manifest are mutually exclusive.", file=sys.stderr)
            return 1
        if args.tasks and args.manifest is not None:
            print("ERROR: Provide either TASK_OR_GLOB args or --manifest, not both.", file=sys.stderr)
            return 1
        if args.tasks and args.all:
            print("ERROR: TASK_OR_GLOB args and --all are mutually exclusive.", file=sys.stderr)
            return 1
        if not args.all and args.manifest is None and not args.tasks:
            print("ERROR: No tasks specified. Use --config, --manifest, --all, or TASK_OR_GLOB args.", file=sys.stderr)
            return 1

    try:
        if args.config is not None:
            config = load_matrix_config(args.config)
            candidates = [candidate.to_candidate() for candidate in config.candidates]
            manifest = Path(config.manifest) if config.manifest is not None else None
            tasks = config.tasks
            all_tasks = config.all_tasks
            timeout = args.timeout or config.defaults.timeout or nat_runner.DEFAULT_TIMEOUT
            skip_build = args.skip_build or bool(config.defaults.skip_build)
            allow_dirty = args.allow_dirty or bool(config.defaults.allow_dirty)
            parallel_candidates = (
                args.parallel_candidates
                if args.parallel_candidates is not None
                else config.defaults.parallel_candidates
            )
            if parallel_candidates is None:
                parallel_candidates = 1
            nhx_base_url = args.nhx_base_url or config.defaults.nhx_base_url or nat_runner.DEFAULT_LOCAL_NHX_BASE_URL
            anthropic_base_url = (
                args.anthropic_base_url
                or config.defaults.anthropic_base_url
                or os.environ.get("ANTHROPIC_BASE_URL", "https://inference-api.nvidia.com")
            )
        else:
            candidates = parse_candidates(args.agent, args.agent_model)
            manifest = args.manifest
            tasks = args.tasks
            all_tasks = args.all
            timeout = args.timeout or nat_runner.DEFAULT_TIMEOUT
            skip_build = args.skip_build
            allow_dirty = args.allow_dirty
            parallel_candidates = args.parallel_candidates if args.parallel_candidates is not None else 1
            nhx_base_url = args.nhx_base_url or nat_runner.DEFAULT_LOCAL_NHX_BASE_URL
            anthropic_base_url = args.anthropic_base_url or os.environ.get(
                "ANTHROPIC_BASE_URL", "https://inference-api.nvidia.com"
            )
        selected_tasks = resolve_selected_tasks(manifest=manifest, tasks=tasks, all_tasks=all_tasks)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if parallel_candidates < 1:
        print("ERROR: --parallel-candidates must be >= 1.", file=sys.stderr)
        return 1
    if parallel_candidates > 1 and not skip_build:
        print(
            "ERROR: --parallel-candidates > 1 requires --skip-build. "
            "Build the selected task images once, then rerun the matrix with --skip-build.",
            file=sys.stderr,
        )
        return 1

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.jobs_dir.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Run id: {run_id}")
    _log(f"Run dir: {run_dir}")
    _log(f"Selected tasks ({len(selected_tasks)}): {', '.join(selected_tasks)}")
    _log(
        "Candidates "
        f"({len(candidates)}): "
        + ", ".join(
            f"{candidate.candidate_id}[{candidate.backend}:{candidate.model or 'default'}]" for candidate in candidates
        )
    )
    matrix_started = time.monotonic()
    run_metadata = run_candidates(
        candidates=candidates,
        run_dir=run_dir,
        selected_tasks=selected_tasks,
        manifest=manifest,
        tasks=tasks,
        all_tasks=all_tasks,
        skip_build=skip_build,
        allow_dirty=allow_dirty,
        timeout=timeout,
        codex_auth_json=args.codex_auth_json.expanduser().resolve() if args.codex_auth_json else None,
        nhx_base_url=nhx_base_url,
        anthropic_base_url=anthropic_base_url,
        parallel_candidates=parallel_candidates,
    )
    summary = build_matrix_summary(
        run_dir=run_dir,
        candidates=candidates,
        tasks=selected_tasks,
        run_metadata=run_metadata,
        manifest=manifest,
    )
    reports = write_reports(summary, run_dir)
    _log(f"Matrix run complete elapsed={_format_seconds(time.monotonic() - matrix_started)}")
    _log("Wrote reports:")
    for name, path in reports.items():
        _log(f"  {name}: {path}")
    return 0


def _log(message: str) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[agent_matrix {timestamp}] {message}", flush=True)


def _load_python_module(path: Path) -> ModuleType:
    module_name = f"_agent_matrix_config_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load matrix config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _flatten_candidate_params(params: dict[str, object], *, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in params.items():
        if value in (None, {}, []):
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_candidate_params(_object_dict(value), prefix=name))
        elif isinstance(value, list):
            flattened[name] = ".".join(str(entry) for entry in value)
        else:
            flattened[name] = value
    return flattened


def _candidate_id(*, backend: str, model: str | None, params: dict[str, object]) -> str:
    parts = [backend]
    if model is not None:
        parts.append(model)
    for key, value in sorted(_flatten_candidate_params(params).items()):
        parts.extend([key, str(value)])
    raw = "-".join(parts)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    return sanitized or backend


def _load_candidate_rows(*, run_dir: Path, candidate: Candidate, tasks: list[str]) -> list[MatrixRow]:
    candidate_dir = run_dir / candidate.candidate_id
    payloads = _latest_results_by_task(candidate_dir)
    rows: list[MatrixRow] = []
    for task in tasks:
        payload = payloads.get(task)
        if payload is None:
            rows.append(
                MatrixRow(
                    candidate_id=candidate.candidate_id,
                    agent_backend=candidate.backend,
                    agent_model=candidate.model or "default",
                    candidate_params=candidate.params,
                    task=task,
                    status="missing",
                    passed=False,
                    reward=0.0,
                    runtime_sec=None,
                    total_tokens=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    cache_read_tokens=None,
                    cache_creation_tokens=None,
                    token_metrics_status="unavailable",
                    verifier_scores=None,
                    result_path=None,
                    output_dir=None,
                    failure_excerpt="No result.json was produced for this task.",
                    provenance={},
                )
            )
            continue
        rows.append(_row_from_payload(candidate=candidate, task=task, payload=payload))
    return rows


def _latest_results_by_task(candidate_dir: Path) -> dict[str, dict[str, object]]:
    latest: dict[str, tuple[float, dict[str, object]]] = {}
    if not candidate_dir.exists():
        return {}
    for result_path in candidate_dir.rglob("result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        task = payload.get("task")
        if not isinstance(task, str):
            continue
        normalized = dict(payload)
        normalized["_result_path"] = str(result_path)
        mtime = result_path.stat().st_mtime
        current = latest.get(task)
        if current is None or mtime > current[0]:
            latest[task] = (mtime, normalized)
    return {task: payload for task, (_mtime, payload) in latest.items()}


def _row_from_payload(*, candidate: Candidate, task: str, payload: dict[str, object]) -> MatrixRow:
    metric_dict = _object_dict(payload.get("metrics"))
    output_dir = _string_value(payload.get("output_dir"))
    result_path = _string_value(payload.get("_result_path"))
    passed = payload.get("passed") is True or _float_value(payload.get("reward")) >= 1.0
    status = _status_from_payload(payload=payload, passed=passed)
    return MatrixRow(
        candidate_id=_string_value(payload.get("candidate_id")) or candidate.candidate_id,
        agent_backend=_string_value(payload.get("agent_backend")) or candidate.backend,
        agent_model=_string_value(payload.get("agent_model")) or candidate.model or "default",
        candidate_params=_object_dict(payload.get("candidate_params")) or candidate.params,
        task=task,
        status=status,
        passed=passed,
        reward=_float_value(payload.get("reward")),
        runtime_sec=_optional_float(payload.get("runtime_sec")),
        total_tokens=_optional_int(metric_dict.get("total_tokens")),
        prompt_tokens=_optional_int(metric_dict.get("prompt_tokens")),
        completion_tokens=_optional_int(metric_dict.get("completion_tokens")),
        cache_read_tokens=_optional_int(metric_dict.get("cache_read_tokens")),
        cache_creation_tokens=_optional_int(metric_dict.get("cache_creation_tokens")),
        token_metrics_status=_string_value(metric_dict.get("token_metrics_status")) or "unavailable",
        verifier_scores=_object_dict(payload.get("verifier_scores")) or None,
        result_path=result_path,
        output_dir=output_dir,
        failure_excerpt=None if passed else _failure_excerpt(output_dir),
        provenance=_object_dict(payload.get("provenance")),
    )


def _summarize_candidates(
    *,
    candidates: list[Candidate],
    tasks: list[str],
    rows: list[MatrixRow],
) -> list[CandidateSummary]:
    summaries: list[CandidateSummary] = []
    by_candidate = {candidate.candidate_id: [] for candidate in candidates}
    for row in rows:
        by_candidate.setdefault(row.candidate_id, []).append(row)

    total_tasks = len(tasks)
    for candidate in candidates:
        candidate_rows = by_candidate.get(candidate.candidate_id, [])
        passed_tasks = sum(1 for row in candidate_rows if row.passed)
        token_values = [row.total_tokens for row in candidate_rows if row.total_tokens is not None]
        prompt_values = [row.prompt_tokens for row in candidate_rows if row.prompt_tokens is not None]
        completion_values = [row.completion_tokens for row in candidate_rows if row.completion_tokens is not None]
        cache_read_values = [row.cache_read_tokens for row in candidate_rows if row.cache_read_tokens is not None]
        cache_creation_values = [
            row.cache_creation_tokens for row in candidate_rows if row.cache_creation_tokens is not None
        ]
        runtime_values = [row.runtime_sec for row in candidate_rows if row.runtime_sec is not None]
        runtime_complete = len(runtime_values) == total_tasks
        summaries.append(
            CandidateSummary(
                candidate_id=candidate.candidate_id,
                agent_backend=candidate.backend,
                agent_model=candidate.model or "default",
                candidate_params=candidate.params,
                total_tasks=total_tasks,
                passed_tasks=passed_tasks,
                pass_rate=(passed_tasks / total_tasks) if total_tasks else 0.0,
                total_tokens_sum=sum(token_values) if token_values else None,
                prompt_tokens_sum=sum(prompt_values) if prompt_values else None,
                completion_tokens_sum=sum(completion_values) if completion_values else None,
                cache_read_tokens_sum=sum(cache_read_values) if cache_read_values else None,
                cache_creation_tokens_sum=sum(cache_creation_values) if cache_creation_values else None,
                token_metrics_coverage=(len(token_values) / total_tasks) if total_tasks else 0.0,
                runtime_sec_sum=sum(runtime_values) if runtime_complete else None,
            )
        )
    return summaries


def _rank_candidate_summaries(summaries: list[CandidateSummary]) -> list[CandidateSummary]:
    def sort_key(summary: CandidateSummary) -> tuple[float, int, float, float, str]:
        token_complete = summary.token_metrics_coverage >= 1.0
        runtime_value = summary.runtime_sec_sum if summary.runtime_sec_sum is not None else float("inf")
        token_value = summary.total_tokens_sum if summary.total_tokens_sum is not None else float("inf")
        return (
            -summary.pass_rate,
            0 if token_complete else 1,
            float(token_value),
            float(runtime_value),
            summary.candidate_id,
        )

    ranked = sorted(summaries, key=sort_key)
    return [
        CandidateSummary(
            candidate_id=summary.candidate_id,
            agent_backend=summary.agent_backend,
            agent_model=summary.agent_model,
            candidate_params=summary.candidate_params,
            total_tasks=summary.total_tasks,
            passed_tasks=summary.passed_tasks,
            pass_rate=summary.pass_rate,
            total_tokens_sum=summary.total_tokens_sum,
            prompt_tokens_sum=summary.prompt_tokens_sum,
            completion_tokens_sum=summary.completion_tokens_sum,
            cache_read_tokens_sum=summary.cache_read_tokens_sum,
            cache_creation_tokens_sum=summary.cache_creation_tokens_sum,
            token_metrics_coverage=summary.token_metrics_coverage,
            runtime_sec_sum=summary.runtime_sec_sum,
            rank=index,
        )
        for index, summary in enumerate(ranked, start=1)
    ]


def _status_from_payload(*, payload: dict[str, object], passed: bool) -> str:
    if passed:
        return "passed"
    agent = _string_value(payload.get("agent"))
    verify = _string_value(payload.get("verify"))
    if agent and agent != "ok":
        return "agent_failed"
    if verify and verify != "ok":
        return "verify_failed"
    return "failed"


def _failure_excerpt(output_dir: str | None) -> str | None:
    if output_dir is None:
        return None
    root = Path(output_dir)
    for relative in ("verifier/test-stdout.txt", "agent/nat_agent.stderr", "agent/nat_agent.log"):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text[-1600:]
    return None


def _render_markdown_matrix(*, tasks: list[str], rows: list[dict[str, object]]) -> str:
    candidates = sorted({str(row.get("candidate_id")) for row in rows})
    by_key = {(str(row.get("task")), str(row.get("candidate_id"))): row for row in rows}
    lines = [
        "| Task | " + " | ".join(f"`{candidate}`" for candidate in candidates) + " |",
        "|---|" + "|".join("---" for _candidate in candidates) + "|",
    ]
    for task in tasks:
        cells = []
        for candidate in candidates:
            row = by_key.get((task, candidate))
            cells.append(_status_symbol(str(row.get("status"))) if row else "MISSING")
        lines.append(f"| `{task}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_candidate_card(candidate: dict[str, object]) -> str:
    pass_rate = _float_value(candidate.get("pass_rate"))
    return """<article class="card score-card">
  <div>
  <h3>{candidate_id}</h3>
  <div class="candidate-subtitle">{backend} / {model}</div>
  </div>
  <div>
  <div class="metric">{pass_rate:.0%}</div>
  <div class="muted">{passed}/{total} passed</div>
  <div class="passbar"><span style="width: {pass_width:.1f}%"></span></div>
  </div>
</article>""".format(
        candidate_id=html.escape(str(candidate.get("candidate_id"))),
        backend=html.escape(str(candidate.get("agent_backend"))),
        model=html.escape(str(candidate.get("agent_model"))),
        pass_rate=pass_rate,
        pass_width=max(0.0, min(pass_rate * 100, 100.0)),
        passed=html.escape(str(candidate.get("passed_tasks"))),
        total=html.escape(str(candidate.get("total_tasks"))),
    )


def _render_leaderboard(candidates: list[dict[str, object]]) -> str:
    frozen_headers = "".join(
        _render_leaderboard_header(column_id, label, visible, frozen=True)
        for column_id, label, visible in LEADERBOARD_COLUMNS
        if column_id in LEADERBOARD_FROZEN_COLUMN_IDS
    )
    scroll_headers = "".join(
        _render_leaderboard_header(column_id, label, visible, frozen=False)
        for column_id, label, visible in LEADERBOARD_COLUMNS
        if column_id not in LEADERBOARD_FROZEN_COLUMN_IDS
    )
    frozen_rows = "\n".join(_render_frozen_ranking_row(candidate) for candidate in candidates)
    scroll_rows = "\n".join(_render_scroll_ranking_row(candidate) for candidate in candidates)
    toggles = "".join(
        """<label class="column-option">
  <input type="checkbox" data-column-toggle value="{column_id}" {checked}>
  <span>{label}</span>
</label>""".format(
            column_id=html.escape(column_id),
            checked="checked" if visible else "",
            label=html.escape(label),
        )
        for column_id, label, visible in LEADERBOARD_COLUMNS
        if column_id not in LEADERBOARD_FROZEN_COLUMN_IDS
    )
    return """<section data-role="candidate-leaderboard">
  <div class="leaderboard-tools">
    <p class="table-hint">
      Scroll horizontally for wide comparisons. Drag header dividers to resize columns.
      Token cells use n/a when the backend did not emit that bucket; 0 means the bucket was emitted with value zero.
    </p>
    <details class="column-picker">
      <summary>Columns</summary>
      <div class="column-menu">{toggles}</div>
    </details>
  </div>
  <div class="leaderboard-frame">
    <div class="leaderboard-frozen">
      <table class="leaderboard-table leaderboard-frozen-table" data-role="leaderboard-frozen-table">
        <colgroup>
          <col class="rank-col" data-col="rank">
          <col class="candidate-col" data-col="candidate">
        </colgroup>
        <thead><tr>{frozen_headers}</tr></thead>
        <tbody>{frozen_rows}</tbody>
      </table>
    </div>
    <div class="leaderboard-scroll-wrap table-wrap">
      <table class="leaderboard-table leaderboard-scroll-table" data-role="leaderboard-scroll-table">
        <colgroup>
          <col class="backend-col" data-col="backend">
          <col class="model-col" data-col="model">
          <col class="params-col" data-col="params">
          <col data-col="pass-rate">
          <col data-col="passed">
          <col data-col="tokens">
          <col data-col="token-coverage">
          <col data-col="input">
          <col data-col="output">
          <col data-col="cache-read">
          <col data-col="cache-write">
          <col class="runtime-col" data-col="runtime">
        </colgroup>
        <thead><tr>{scroll_headers}</tr></thead>
        <tbody>{scroll_rows}</tbody>
      </table>
    </div>
  </div>
</section>""".format(
        toggles=toggles,
        frozen_headers=frozen_headers,
        frozen_rows=frozen_rows,
        scroll_headers=scroll_headers,
        scroll_rows=scroll_rows,
    )


def _render_leaderboard_header(column_id: str, label: str, visible: bool, *, frozen: bool) -> str:
    classes: list[str] = []
    min_width = 96
    if column_id == "rank":
        classes.append("numeric")
        min_width = 72
    elif column_id == "candidate":
        classes.append("resizable")
        min_width = 180
    elif column_id in {"backend", "model", "params"}:
        classes.append("resizable")
        min_width = 120 if column_id != "params" else 160
    elif column_id in {"tokens", "input", "output", "cache-read", "cache-write"}:
        classes.extend(["numeric", "resizable"])
        min_width = 100
    elif column_id in {"pass-rate", "passed", "runtime"}:
        classes.append("numeric")
    class_attr = " ".join(classes)
    hidden_class = "" if visible else " is-hidden"
    handle = '<span class="resize-handle" aria-hidden="true"></span>' if "resizable" in classes else ""
    data_frozen = "true" if frozen else "false"
    return (
        '<th data-col="{column_id}" data-frozen="{data_frozen}" data-min-width="{min_width}" class="{classes}{hidden_class}">'
        "{label}{handle}</th>"
    ).format(
        column_id=html.escape(column_id),
        data_frozen=data_frozen,
        min_width=min_width,
        classes=html.escape(class_attr),
        hidden_class=hidden_class,
        label=html.escape(label),
        handle=handle,
    )


def _render_frozen_ranking_row(candidate: dict[str, object]) -> str:
    return """<tr>
  <td data-col="rank" class="numeric">{rank}</td>
  <td data-col="candidate"><strong>{candidate_id}</strong></td>
</tr>""".format(
        rank=html.escape(str(candidate.get("rank"))),
        candidate_id=html.escape(str(candidate.get("candidate_id"))),
    )


def _render_scroll_ranking_row(candidate: dict[str, object]) -> str:
    return """<tr>
  <td data-col="backend">{backend}</td>
  <td data-col="model">{model}</td>
  <td data-col="params"><div class="param-list">{params}</div></td>
  <td data-col="pass-rate" class="numeric">{pass_rate:.3f}</td>
  <td data-col="passed" class="numeric">{passed}/{total}</td>
  <td data-col="tokens" class="numeric">{tokens}</td>
  <td data-col="token-coverage" class="numeric">{token_coverage}</td>
  <td data-col="input" class="numeric">{input_tokens}</td>
  <td data-col="output" class="numeric">{output_tokens}</td>
  <td data-col="cache-read" class="numeric">{cache_read_tokens}</td>
  <td data-col="cache-write" class="numeric">{cache_creation_tokens}</td>
  <td data-col="runtime" class="numeric">{runtime}</td>
</tr>""".format(
        backend=html.escape(str(candidate.get("agent_backend"))),
        model=html.escape(str(candidate.get("agent_model"))),
        params=_render_param_chips(candidate.get("candidate_params")),
        pass_rate=_float_value(candidate.get("pass_rate")),
        passed=html.escape(str(candidate.get("passed_tasks"))),
        total=html.escape(str(candidate.get("total_tasks"))),
        tokens=_render_metric_cell(candidate.get("total_tokens_sum")),
        token_coverage=html.escape(f"{_float_value(candidate.get('token_metrics_coverage')):.0%}"),
        input_tokens=_render_metric_cell(candidate.get("prompt_tokens_sum")),
        output_tokens=_render_metric_cell(candidate.get("completion_tokens_sum")),
        cache_read_tokens=_render_metric_cell(candidate.get("cache_read_tokens_sum")),
        cache_creation_tokens=_render_metric_cell(candidate.get("cache_creation_tokens_sum")),
        runtime=html.escape(_format_seconds(candidate.get("runtime_sec_sum"))),
    )


def _render_candidate_groups(candidate_groups: dict[str, list[dict[str, object]]]) -> str:
    if not candidate_groups:
        return '<p class="muted">No candidate groups recorded.</p>'
    sections = ['<div class="model-groups">']
    for backend in sorted(candidate_groups):
        candidates = candidate_groups[backend]
        rows = []
        for candidate in candidates:
            rows.append(
                """<div class="model-row">
  <div class="model-name">{model}</div>
  <div class="candidate-name">{candidate}</div>
  <div class="param-list">{params}</div>
  <div class="numeric">{passed}/{total} ({pass_rate:.0%})</div>
</div>""".format(
                    model=html.escape(str(candidate.get("agent_model"))),
                    candidate=html.escape(str(candidate.get("candidate_id"))),
                    params=_render_param_chips(candidate.get("candidate_params")),
                    pass_rate=_float_value(candidate.get("pass_rate")),
                    passed=html.escape(str(candidate.get("passed_tasks"))),
                    total=html.escape(str(candidate.get("total_tasks"))),
                )
            )
        sections.append(
            """<section class="model-group">
  <h3>{backend}</h3>
  {rows}
</section>""".format(backend=html.escape(backend), rows="".join(rows))
        )
    sections.append("</div>")
    return "\n".join(sections)


def _render_html_task_matrix(
    *, tasks: list[str], ranking: list[dict[str, object]], rows: list[dict[str, object]]
) -> str:
    candidates = [str(candidate.get("candidate_id")) for candidate in ranking]
    by_key = {(str(row.get("task")), str(row.get("candidate_id"))): row for row in rows}
    head = "".join(f"<th>{html.escape(candidate)}</th>" for candidate in candidates)
    body_lines = []
    for task in tasks:
        cells = []
        for candidate in candidates:
            row = by_key.get((task, candidate))
            cells.append(_render_task_result_cell(row))
        body_lines.append(f"<tr><td><strong>{html.escape(task)}</strong></td>{''.join(cells)}</tr>")
    return (
        '<div class="table-wrap">'
        f'<table class="matrix-table"><thead><tr><th>Task</th>{head}</tr></thead>'
        f"<tbody>{''.join(body_lines)}</tbody></table></div>"
    )


def _render_task_result_cell(row: dict[str, object] | None) -> str:
    status = str(row.get("status")) if row else "missing"
    status_pill = '<span class="pill {status}" title="{description}">{symbol}</span>'.format(
        status=html.escape(status),
        description=html.escape(_status_description(status)),
        symbol=html.escape(_status_symbol(status)),
    )
    return (
        '<td class="task-result-cell">'
        f'<div class="task-result-status">{status_pill}</div>'
        f"{_render_token_bucket_cell(row)}"
        "</td>"
    )


def _render_token_bucket_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return '<span class="muted">missing</span>'
    if row.get("token_metrics_status") != "available":
        return '<span class="muted" title="This task did not emit token metrics.">n/a</span>'
    bucket_labels = [
        ("Total", row.get("total_tokens")),
        ("Input", row.get("prompt_tokens")),
        ("Output", row.get("completion_tokens")),
        ("Cache Read", row.get("cache_read_tokens")),
        ("Cache Write", row.get("cache_creation_tokens")),
    ]
    items = "".join(
        "<dt>{label}</dt><dd>{value}</dd>".format(
            label=html.escape(label),
            value=_render_metric_cell(value),
        )
        for label, value in bucket_labels
    )
    return f'<dl class="token-buckets">{items}</dl>'


def _render_status_legend() -> str:
    entries = [
        (
            "passed",
            "PASS",
            "Benchmark success",
            "The agent ran, verification ran, and the deterministic Evaluator SDK checks passed.",
        ),
        (
            "verify_failed",
            "VERIFY",
            "Verifier failed",
            "The agent completed, but the task-specific verifier rejected its output or artifacts.",
        ),
        (
            "agent_failed",
            "AGENT",
            "Agent failed",
            "The agent phase failed before the task could be successfully verified.",
        ),
        (
            "failed",
            "FAIL",
            "Failed",
            "The run failed without a more specific agent or verifier status.",
        ),
        (
            "missing",
            "MISSING",
            "No result",
            "No result.json was produced for this candidate/task pair.",
        ),
    ]
    return (
        '<details class="status-legend-details"><summary>Status Legend</summary><div class="status-legend">'
        + "".join(
            """<div class="status-legend-item">
  <span class="pill {css_class}">{pill}</span>
  <span><strong>{title}</strong>{description}</span>
</div>""".format(
                css_class=html.escape(css_class),
                pill=html.escape(pill),
                title=html.escape(title),
                description=html.escape(description),
            )
            for css_class, pill, title, description in entries
        )
        + "</div></details>"
    )


def _status_description(status: str) -> str:
    return {
        "passed": "The agent ran, verification ran, and the deterministic Evaluator SDK checks passed.",
        "verify_failed": "The agent completed, but the task-specific verifier rejected its output or artifacts.",
        "agent_failed": "The agent phase failed before the task could be successfully verified.",
        "failed": "The run failed without a more specific agent or verifier status.",
        "missing": "No result.json was produced for this candidate/task pair.",
    }.get(status, status)


def _render_bar_chart(candidates: list[dict[str, object]], *, value_key: str, label: str, css_class: str) -> str:
    values = [_optional_float(candidate.get(value_key)) for candidate in candidates]
    max_value = max([value for value in values if value is not None], default=0.0)
    rows = [f"<h3>{html.escape(label)}</h3>"]
    for candidate, value in zip(candidates, values, strict=True):
        width = 0.0 if not value or max_value <= 0 else (value / max_value) * 100
        rows.append(
            """<div class="bar-row">
  <span>{candidate}</span>
  <span class="bar-track"><span class="bar-fill {css_class}" style="width: {width:.1f}%"></span></span>
  <span class="numeric">{value}</span>
</div>""".format(
                candidate=html.escape(str(candidate.get("candidate_id"))),
                css_class=html.escape(css_class),
                width=width,
                value=html.escape(_format_metric_number(value)),
            )
        )
    return "\n".join(rows)


def _render_failure_detail(row: dict[str, object], *, run_dir: Path) -> str:
    output_dir = str(row.get("output_dir") or "")
    href = _artifact_href(output_dir, run_dir=run_dir)
    excerpt = html.escape(str(row.get("failure_excerpt") or "No failure excerpt available."))
    scores_payload = _object_dict(row.get("verifier_scores"))
    diagnosis = _render_failure_diagnosis(row=row, scores_payload=scores_payload)
    metric_outcome = _render_metric_outcome_summary(scores_payload)
    scores = _render_verifier_scores(scores_payload)
    status = str(row.get("status") or "failed")
    return f"""<section class="failure">
  <div class="failure-header">
    <h3>{html.escape(str(row.get("candidate_id")))} / {html.escape(str(row.get("task")))}</h3>
    <div class="failure-actions"><span class="pill {html.escape(status)}">{html.escape(_status_symbol(status))}</span> <a href="{html.escape(href)}">artifact directory</a></div>
  </div>
    {diagnosis}
    {metric_outcome}
  {scores}
  <details>
    <summary>Raw verifier log</summary>
    <pre>{excerpt}</pre>
  </details>
</section>"""


def _render_verifier_scores(value: object) -> str:
    scores_payload = _object_dict(value)
    aggregate_scores = scores_payload.get("aggregate_scores")
    if not isinstance(aggregate_scores, list) or not aggregate_scores:
        return '<p class="muted">No structured Evaluator SDK scores were captured for this run.</p>'

    score_dicts = [_object_dict(score) for score in aggregate_scores if isinstance(score, dict)]
    chips = [_render_score_chip(score) for score in score_dicts]
    failed_chips = [_render_score_chip(score) for score in score_dicts if _is_failed_score(score)]
    if not chips:
        return '<p class="muted">No structured Evaluator SDK scores were captured for this run.</p>'
    failed_section = ""
    if failed_chips:
        failed_section = (
            "<div><strong>Failed or blocking metrics</strong>"
            '<div class="score-list">' + "".join(failed_chips) + "</div></div>"
        )
    return (
        failed_section
        + '<details><summary>All metric scores</summary><div class="score-list">'
        + "".join(chips)
        + "</div></details>"
    )


def _render_failure_diagnosis(*, row: dict[str, object], scores_payload: dict[str, object]) -> str:
    status = str(row.get("status") or "failed")
    title = {
        "verify_failed": "Verifier rejected the completed agent attempt.",
        "agent_failed": "Agent execution failed before successful verification.",
        "failed": "Run failed.",
        "missing": "No result was produced.",
    }.get(status, "Run did not pass.")
    reason_parts = []
    task_success = _aggregate_score_mean(scores_payload, "task_success")
    output_schema_valid = _aggregate_score_mean(scores_payload, "output_schema_valid")
    verification_score = _aggregate_score_mean(scores_payload, "verification_score")
    if task_success == 0.0:
        reason_parts.append("Task success is false.")
    if output_schema_valid == 0.0:
        reason_parts.append("Output schema/contract is invalid.")
    if verification_score is not None:
        reason_parts.append(f"Verification score: {verification_score:.3f}.")
    if not reason_parts:
        reason_parts.append(_status_description(status))
    return '<div class="diagnosis"><strong>{title}</strong>{reason}</div>'.format(
        title=html.escape(title),
        reason=html.escape(" ".join(reason_parts)),
    )


def _render_metric_outcome_summary(scores_payload: dict[str, object]) -> str:
    task_success = _aggregate_score_mean(scores_payload, "task_success")
    verification_score = _aggregate_score_mean(scores_payload, "verification_score")
    output_schema_valid = _aggregate_score_mean(scores_payload, "output_schema_valid")
    if task_success is None and verification_score is None and output_schema_valid is None:
        return ""
    cards = [
        _render_check_card("Task Success", _format_score_pass(task_success)),
        _render_check_card(
            "Verification Score",
            "n/a" if verification_score is None else f"{verification_score:.3f}",
        ),
        _render_check_card("Output Contract", _format_score_pass(output_schema_valid)),
    ]
    return '<div class="check-grid">' + "".join(cards) + "</div>"


def _render_check_card(label: str, value: str) -> str:
    return '<div class="check-card"><span>{label}</span><strong>{value}</strong></div>'.format(
        label=html.escape(label),
        value=html.escape(value),
    )


def _render_score_chip(score: dict[str, object]) -> str:
    name = str(score.get("name") or "unknown")
    mean = _optional_float(score.get("mean"))
    if mean is None:
        css_class = "neutral"
        label = "n/a"
    else:
        css_class = _score_chip_class(name, mean)
        label = f"{mean:.3f}"
    short_name = name.split(".", maxsplit=1)[-1]
    return '<span class="score-chip {css_class}"><span>{name}</span><span>{value}</span></span>'.format(
        css_class=css_class,
        name=html.escape(_humanize_metric_name(short_name)),
        value=html.escape(label),
    )


def _is_failed_score(score: dict[str, object]) -> bool:
    name = str(score.get("name") or "unknown")
    mean = _optional_float(score.get("mean"))
    return mean is not None and _score_chip_class(name, mean) == "bad"


def _score_chip_class(name: str, mean: float) -> str:
    short_name = name.split(".", maxsplit=1)[-1]
    if short_name in {"surface_violation_count", "legacy_surface_hit_count", "failed_command_count"}:
        return "good" if mean == 0.0 else "bad"
    if short_name in {"tool_call_count", "recovery_event_count"}:
        return "neutral"
    return "good" if mean >= 1.0 else "bad"


def _format_boolish(value: object) -> str:
    if value is True:
        return "Pass"
    if value is False:
        return "Fail"
    if value is None:
        return "n/a"
    return str(value)


def _format_score_pass(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1.0:
        return "Pass"
    if value <= 0.0:
        return "Fail"
    return f"{value:.3f}"


def _aggregate_score_mean(scores_payload: dict[str, object], score_name: str) -> float | None:
    aggregate_scores = scores_payload.get("aggregate_scores")
    if not isinstance(aggregate_scores, list):
        return None
    for score in aggregate_scores:
        score_dict = _object_dict(score)
        name = str(score_dict.get("name") or "")
        if name.endswith(f".{score_name}"):
            return _optional_float(score_dict.get("mean"))
    return None


def _humanize_metric_name(name: str) -> str:
    return name.replace("_", " ").replace(".", " / ").title()


def _render_provenance(rows: list[dict[str, object]]) -> str:
    observed: dict[str, set[str]] = {}
    for row in rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            continue
        for key, value in provenance.items():
            if value is None:
                continue
            observed.setdefault(str(key), set()).add(str(value))
    if not observed:
        return '<p class="muted">No provenance recorded.</p>'
    lines = ["<table><thead><tr><th>Field</th><th>Observed Values</th></tr></thead><tbody>"]
    for key in sorted(observed):
        values = ", ".join(sorted(observed[key]))
        lines.append(f"<tr><td>{html.escape(key)}</td><td>{html.escape(values)}</td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _render_provenance_badges(rows: list[dict[str, object]]) -> str:
    for row in rows:
        provenance = _object_dict(row.get("provenance"))
        if not provenance:
            continue
        if _is_truthy_provenance_value(provenance.get("commit_dirty")):
            return (
                '<span class="badge dirty" title="At least one task reported commit_dirty=true in provenance.">'
                "dirty worktree</span>"
            )
    return ""


def _is_truthy_provenance_value(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _artifact_href(output_dir: str, *, run_dir: Path) -> str:
    if not output_dir:
        return "#"
    try:
        return os.path.relpath(output_dir, run_dir)
    except ValueError:
        return output_dir


def _status_symbol(status: str) -> str:
    return {
        "passed": "PASS",
        "failed": "FAIL",
        "agent_failed": "AGENT",
        "verify_failed": "VERIFY",
        "missing": "MISSING",
    }.get(status, status.upper())


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [_object_dict(entry) for entry in value if isinstance(entry, dict)]


def _dict_of_list_of_dicts(value: object) -> dict[str, list[dict[str, object]]]:
    if not isinstance(value, dict):
        return {}
    groups: dict[str, list[dict[str, object]]] = {}
    for key, entries in value.items():
        if isinstance(entries, list):
            groups[str(key)] = [_object_dict(entry) for entry in entries if isinstance(entry, dict)]
    return groups


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _candidate_groups(candidates: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        backend = str(candidate.get("agent_backend") or "unknown")
        groups.setdefault(backend, []).append(candidate)
    return groups


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_number(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "-"
    numeric = float(numeric)
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"
    if numeric >= 1_000:
        return f"{numeric / 1_000:.1f}k"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def _format_metric_number(value: object) -> str:
    return "n/a" if _optional_float(value) is None else _format_number(value)


def _render_metric_cell(value: object) -> str:
    if _optional_float(value) is None:
        return '<span title="This backend did not emit this token bucket.">n/a</span>'
    formatted = _format_number(value)
    if formatted == "0":
        return '<span title="This backend emitted this token bucket with value zero.">0</span>'
    return html.escape(formatted)


def _format_seconds(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "-"
    minutes = int(numeric) // 60
    seconds = int(numeric) % 60
    return f"{minutes}:{seconds:02d}"


def _format_params(value: object) -> str:
    params = _object_dict(value)
    flattened = _flatten_candidate_params(params)
    if not flattened:
        return "-"
    return ", ".join(f"{key}={flattened[key]}" for key in sorted(flattened))


def _render_param_chips(value: object) -> str:
    params = _object_dict(value)
    flattened = _flatten_candidate_params(params)
    if not flattened:
        return '<span class="chip neutral">default</span>'
    return "".join(
        f'<span class="chip">{html.escape(str(key))}={html.escape(str(flattened[key]))}</span>'
        for key in sorted(flattened)
    )


if __name__ == "__main__":
    raise SystemExit(main())
