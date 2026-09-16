# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AnalyzeBatchJob — analyze a batch of eval-suite results.

Registered under ``nemo.jobs`` as ``agents.analyze``.

The CLI command ``nemo agents analyze --spec '{...}'`` submits it to the
platform; the jobs controller dispatches a subprocess on the same host and the
result lands in ``nemo jobs list`` / Studio's Jobs view.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from nemo_agents_plugin.jobs.evaluate_suite import _require_absolute
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import PlatformJobSpec
from nemo_helix_plugin.jobs.exceptions import PlatformJobCompilationError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzeBatchConfig(BaseModel):
    batch: str = Field(description="Path to a batch directory produced by evaluate-suite.")
    format: Literal["md", "json"] = Field(default="md", description="Output format: md or json.")
    mechanical_only: bool = Field(default=False, description="Skip the LLM analysis pass.")
    anthropic_api_key_secret: str | None = Field(
        default=None,
        description=(
            "Name of a platform Secret holding the Anthropic API key.  When set on a platform submission, "
            "the value is injected as ``ANTHROPIC_API_KEY`` into the dispatched subprocess "
            "so the LLM gap-analysis pass can call Anthropic.  Ignored when "
            "``mechanical_only=True``.  Local (in-process) runs read ``ANTHROPIC_API_KEY`` "
            "from the calling shell as before."
        ),
    )
    anthropic_base_url: str | None = Field(
        default=None,
        description=(
            "Anthropic-compatible API base URL injected as ``ANTHROPIC_BASE_URL`` for platform "
            "submissions.  Local (in-process) runs read ``ANTHROPIC_BASE_URL`` from the calling "
            "shell as before."
        ),
    )


class AnalyzeBatchJob(NemoJob):
    """Analyze a batch of eval-suite results — mechanical clustering + LLM hypotheses."""

    name: ClassVar[str] = "analyze"
    description: ClassVar[str] = "Analyze a batch of eval-suite results (clusters, regressions, hypotheses)."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = AnalyzeBatchConfig

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: AnalyzeBatchConfig,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Single-step PlatformJobSpec running ``nemo_agents_plugin.tasks.analyze``."""
        from nemo_helix_plugin.jobs.api_factory import (
            EnvironmentVariable,
            EnvironmentVariableFromSecret,
            PlatformJobStep,
            SubprocessExecutionProviderSpec,
        )
        from nemo_helix_plugin.jobs.constants import (
            DEFAULT_JOB_STORAGE_PATH,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
        )

        _require_absolute(spec.batch, "batch")
        if not spec.mechanical_only and not spec.anthropic_api_key_secret:
            raise PlatformJobCompilationError(
                "'anthropic_api_key_secret' is required when submitting unless "
                "'mechanical_only' is True (the LLM analysis pass calls Anthropic, "
                "and the subprocess backend's sanitized env does not inherit "
                "ANTHROPIC_API_KEY)."
            )

        spec_dict = spec.model_dump(mode="json")
        spec_dict["workspace"] = workspace

        environment: list[EnvironmentVariable] = [
            EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH),
        ]
        if spec.anthropic_api_key_secret:
            environment.append(
                EnvironmentVariable(
                    name="ANTHROPIC_API_KEY",
                    from_secret=EnvironmentVariableFromSecret(name=spec.anthropic_api_key_secret),
                )
            )
        if spec.anthropic_base_url:
            environment.append(EnvironmentVariable(name="ANTHROPIC_BASE_URL", value=spec.anthropic_base_url))

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="analyze",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agents_plugin.tasks.analyze"],
                    ),
                    config=spec_dict,
                    environment=environment,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext | None = None) -> dict[str, Any]:
        # ``ctx`` is signature-typed so the framework's DI populates it on the
        # remote job path; programmatic local callers invoke ``run(spec)`` directly.
        del ctx
        from nemo_agents_plugin.improvement.analysis.llm import generate_gap_analysis
        from nemo_agents_plugin.improvement.analysis.mechanical import cluster_evals, mechanical_analysis
        from nemo_agents_plugin.improvement.baselines import load_baselines
        from nemo_agents_plugin.improvement.models import GapAnalysis, _serialize_dict
        from nemo_agents_plugin.improvement.runners._harbor_results import parse_batch_results
        from nemo_agents_plugin.improvement.traces.base import TraceParser
        from nemo_agents_plugin.improvement.traces.claude_code_parser import ClaudeCodeTraceParser

        cfg = AnalyzeBatchConfig.model_validate(config)
        batch_dir = Path(cfg.batch).resolve()
        if not batch_dir.is_dir():
            raise RuntimeError(f"Batch directory not found: {batch_dir}")

        batch = parse_batch_results(batch_dir)
        baselines_path = batch_dir.parent.parent / "baselines.json"
        baselines = load_baselines(baselines_path) if baselines_path.exists() else {}

        # Pick the trace parser at the call site. Today the only
        # implementation is ClaudeCodeTraceParser (session.jsonl); when
        # other parsers exist (e.g. for NAT IntermediateStep records) this
        # becomes a config-driven choice. Batches whose traces aren't
        # claude-code-shaped run mechanical-only — the runner-agnostic
        # signals (failing/slowest/regressions/baselines) still populate.
        parser: TraceParser | None = ClaudeCodeTraceParser() if batch.agent == "claude-code" else None
        if parser is None:
            logger.warning("No trace parser for agent %r — trace-derived analysis skipped.", batch.agent)

        if cfg.mechanical_only:
            mech = mechanical_analysis(batch, parser, baselines)
            clusters = cluster_evals(mech, baselines=baselines, batch=batch)
            ga = GapAnalysis(batch_id=batch.batch_id, mechanical=mech, clusters=clusters, hypotheses=[])
        else:
            ga = asyncio.run(generate_gap_analysis(batch=batch, parser=parser, baselines=baselines))

        if cfg.format == "json":
            return _serialize_dict(ga)

        # markdown
        from nemo_agents_plugin.improvement._analysis_reporting import generate_gap_report

        return {"format": "md", "report": generate_gap_report(ga)}
