# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT runner — NeMo Agent Toolkit as the agent under test.

Marker file: ``workflow.yml``. Delegates to ``tests/agentic-use/nat_runner.py``
(subprocess wrapper for the POC; the plan calls for a shared orchestration
library extracted with @gabwow — that's a v0 follow-up).

POC scope:
- Eval discovery (find dirs with ``workflow.yml``)
- Per-task execution via ``nat_runner.py`` subprocess
- Pass/fail + duration + tokens via reading ``result.json`` (PR #227 contract)
- LLM-analyzer fallback: pytest output + container logs read from the runner's
  output dir, *not* faked into a Claude Code session-JSONL shape.

Trace-driven analysis (clustering on tool-call patterns, skill usage) is a
known gap until NAT runs gain OpenTelemetry/Phoenix spans we can parse.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from nemo_agents_plugin.improvement.models import (
    BatchResult,
    EvalResult,
    EvalSpec,
    EvalStatus,
    EvalTiming,
    TokenUsage,
)

# Required files for a valid NAT eval task
NAT_REQUIRED_FILES = ("workflow.yml", "instruction.md", "tests/test_outputs.py")
NAT_EXCLUDED_DIRS = frozenset({"shared", "scripts", "reports", "agentic_flows", "example-test-template"})


def discover_nat_evals(evals_dir: Path) -> list[EvalSpec]:
    """Scan *evals_dir* for directories with ``workflow.yml``."""
    if not evals_dir.is_dir():
        return []
    evals_dir = evals_dir.resolve()
    specs: list[EvalSpec] = []
    for child in sorted(evals_dir.iterdir()):
        if not child.is_dir() or child.name in NAT_EXCLUDED_DIRS:
            continue
        if not (child / "workflow.yml").exists():
            continue
        if any(not (child / f).exists() for f in NAT_REQUIRED_FILES):
            continue
        # Use an EvalSpec with sensible NAT defaults; we don't have a task.toml
        # to read difficulty/category from, so this is best-effort.
        from nemo_agents_plugin.improvement.models import Difficulty

        specs.append(
            EvalSpec(
                name=child.name,
                path=child,
                difficulty=Difficulty.MEDIUM,
                category="",
                tags=[],
                agent_timeout_sec=600.0,
                verifier_timeout_sec=60.0,
            )
        )
    return specs


class NATRunner:
    """Runner for NAT-based eval tasks (NeMo Agent Toolkit agent).

    Locates ``nat_runner.py`` by env var ``NHX_AGENTS_NAT_RUNNER`` or by
    convention at ``<repo_root>/tests/agentic-use/nat_runner.py``.
    """

    name = "nat"
    supports_repeats: ClassVar[bool] = False

    def discover(self, evals_dir: Path) -> list[EvalSpec]:
        """Find NAT eval tasks (directories with ``workflow.yml``)."""
        return discover_nat_evals(evals_dir)

    async def run_batch(
        self,
        evals: list[EvalSpec],
        batch_dir: Path,
        *,
        concurrency: int = 4,
        skip_build: bool = False,
        project_root: Path | None = None,
        repeats: int = 1,
    ) -> BatchResult:
        """Run *evals* via subprocess ``nat_runner.py``; ``repeats > 1`` is logged and ignored."""
        # NAT runner doesn't yet implement repeats — flag clearly.
        if repeats > 1:
            from rich.console import Console

            Console().print(
                f"[yellow]NAT runner: --repeats {repeats} ignored — multi-trial support "
                "is Harbor-only in this POC. Running each eval once.[/yellow]"
            )
        # PR #227 reports concurrent NAT runs contend on per-task provider
        # seeding; >1 inflates wallclock by ~50%. Warn so callers can tune.
        if concurrency > 1:
            from rich.console import Console

            Console().print(
                f"[yellow]NAT runner: concurrency={concurrency} may inflate wallclock — "
                "parallel runs contend on AUT provider seeding (see PR #227). "
                "Lower to 1 or pass --no-aut-seed-providers if available.[/yellow]"
            )
        # NAT tasks inherit from nhx-agentic-base:latest, the shared image
        # built by the helper in _agentic_base.py. Trigger the build here so
        # NAT users don't get a silent image-not-found if Harbor hasn't been
        # run first.
        from nemo_agents_plugin.improvement._agentic_base import build_agentic_base_image

        if not skip_build:
            if project_root is None:
                raise RuntimeError("project_root is required when skip_build=False")
            await build_agentic_base_image(project_root)

        nat_runner_path = self._locate_nat_runner(project_root)
        if not nat_runner_path:
            raise RuntimeError(
                "NAT runner not found. Set NHX_AGENTS_NAT_RUNNER or ensure "
                "tests/agentic-use/nat_runner.py exists in the project."
            )

        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_id = batch_dir.name
        started_at = datetime.now(timezone.utc)

        sem = asyncio.Semaphore(max(1, concurrency))
        results: list[EvalResult] = []

        async def _run_one(spec: EvalSpec) -> EvalResult:
            async with sem:
                return await self._run_single(spec, batch_dir, nat_runner_path)

        tasks = [asyncio.create_task(_run_one(s)) for s in evals]
        for t in tasks:
            results.append(await t)

        return BatchResult(
            batch_id=batch_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            agent="nat",
            results=results,
        )

    def eval_input_paths(self, eval_dir: Path) -> list[Path]:
        """Files the loop must not modify for a NAT task."""
        return [
            eval_dir / "workflow.yml",
            eval_dir / "instruction.md",
            eval_dir / "tests",
            eval_dir / "environment",
        ]

    def _locate_nat_runner(self, project_root: Path | None) -> Path | None:
        """Resolve the ``nat_runner.py`` path from env var or project convention."""
        env = os.environ.get("NHX_AGENTS_NAT_RUNNER")
        if env:
            p = Path(env)
            return p if p.exists() else None
        if project_root:
            p = project_root / "tests" / "agentic-use" / "nat_runner.py"
            if p.exists():
                return p
        return None

    async def _run_single(self, spec: EvalSpec, batch_dir: Path, nat_runner_path: Path) -> EvalResult:
        """Execute one NAT task via subprocess and parse its ``result.json``."""
        out_dir = batch_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)

        started = datetime.now(timezone.utc)

        # PR #227 contract: --jobs-dir is an argparse flag, NAT_RUNNER_JOBS_DIR
        # env var is no longer read. nat_runner.py's default for --nhx-base-url
        # is localhost:8080 and it ignores shell NHX_BASE_URL — forward it
        # explicitly so callers can point at a non-default deployment.
        cmd: list[str] = [
            sys.executable,
            str(nat_runner_path),
            "--jobs-dir",
            str(out_dir),
            spec.name,
        ]
        nhx_base_url = os.environ.get("NHX_BASE_URL")
        if nhx_base_url:
            cmd[-1:-1] = ["--nhx-base-url", nhx_base_url]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=spec.agent_timeout_sec + spec.verifier_timeout_sec + 60
            )
        except asyncio.TimeoutError:
            # Drain pipes and reap the killed child so we don't leak the
            # process and we still get whatever stdout/stderr was buffered
            # before the kill — partial logs are the only post-mortem signal
            # for a timed-out run.
            proc.kill()
            stdout, stderr = await proc.communicate()
            finished = datetime.now(timezone.utc)
            duration = (finished - started).total_seconds()
            (out_dir / "stdout.log").write_bytes(stdout or b"")
            (out_dir / "stderr.log").write_bytes(stderr or b"")
            return EvalResult(
                eval_name=spec.name,
                status=EvalStatus.ERROR,
                timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                agent_timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                exception="nat_runner timeout",
                job_dir=out_dir,
            )

        finished = datetime.now(timezone.utc)
        duration = (finished - started).total_seconds()

        # Save stdout/stderr for the LLM analyzer fallback before parsing — if
        # parse fails we still want the diagnostic output on disk.
        (out_dir / "stdout.log").write_bytes(stdout or b"")
        (out_dir / "stderr.log").write_bytes(stderr or b"")

        # Exit code is authoritative: a non-zero return means the run is
        # untrustworthy regardless of whether result.json parses, since a
        # crashed runner can leave a partial/stale payload behind. Treating
        # such runs as FAIL would contaminate baselines with runner errors.
        if proc.returncode != 0:
            return EvalResult(
                eval_name=spec.name,
                status=EvalStatus.ERROR,
                timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                agent_timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                exception=f"nat_runner exited with code {proc.returncode}",
                job_dir=out_dir,
            )

        # PR #227 contract: read result.json (top of out_dir) for pass/fail,
        # runtime, and per-task token totals. The legacy reward.txt lives
        # under verifier/ and is no longer authoritative.
        result_file = out_dir / "result.json"
        if not result_file.exists():
            return EvalResult(
                eval_name=spec.name,
                status=EvalStatus.ERROR,
                timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                agent_timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                exception="nat_runner did not produce result.json",
                job_dir=out_dir,
            )

        try:
            payload = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return EvalResult(
                eval_name=spec.name,
                status=EvalStatus.ERROR,
                timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                agent_timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
                exception=f"result.json unreadable: {exc}",
                job_dir=out_dir,
            )

        passed = bool(payload.get("passed"))
        reward = float(payload.get("reward") or 0.0)
        runtime_sec = payload.get("runtime_sec")
        agent_duration = float(runtime_sec) if isinstance(runtime_sec, (int, float)) else duration

        metrics = payload.get("metrics") or {}
        prompt_tokens = metrics.get("prompt_tokens") or 0
        completion_tokens = metrics.get("completion_tokens") or 0
        cache_tokens = (metrics.get("cache_read_tokens") or 0) + (metrics.get("cache_creation_tokens") or 0)
        tokens = TokenUsage(
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens),
            cache_tokens=int(cache_tokens),
        )

        status = EvalStatus.PASS if passed else EvalStatus.FAIL

        return EvalResult(
            eval_name=spec.name,
            status=status,
            reward=reward,
            timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=duration),
            agent_timing=EvalTiming(started_at=started, finished_at=finished, duration_sec=agent_duration),
            tokens=tokens,
            exception=None,
            job_dir=out_dir,
        )
