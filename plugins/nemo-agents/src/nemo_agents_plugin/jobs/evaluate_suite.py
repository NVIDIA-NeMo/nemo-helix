# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvaluateSuiteJob — run a directory of containerized eval tasks against an agent.

Registered under ``nemo.jobs`` as ``agents.evaluate-suite``. The CLI command
``nemo agents evaluate-suite --spec '{...}'`` submits it to the platform; the
jobs controller dispatches a subprocess on the same host that runs the
platform (today: the user's laptop) and the result lands in ``nemo jobs list``
/ Studio's Jobs view.

For repeatable runs, use a YAML spec file:

    nemo agents evaluate-suite --spec-file .agent-improver.yml
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from nemo_agents_plugin.jobs.job_usage import evaluation_batch_token_usage
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _require_absolute(value: str | None, field: str) -> None:
    """Raise HelixJobCompilationError if *value* is set and not absolute.

    The platform's host-subprocess executor runs each step in an ephemeral
    work dir under ``/tmp/nhx-subprocess-jobs/<job>/<attempt>/<step>/task-<id>``,
    not the caller's project root.  Relative paths on the submit path resolve
    against that work dir, which is empty, and the job silently fails preflight
    (or worse, writes outputs to the work dir that get reaped after the job's
    TTL).  Reject relative paths up front with a clear error so a 422 surfaces
    at submit time instead.
    """
    if value is None:
        return
    if not Path(value).is_absolute():
        raise HelixJobCompilationError(
            f"{field!r} must be an absolute path when submitting via the platform "
            f"(got {value!r}); the subprocess executor's work dir is not the caller's "
            f"cwd. Re-submit with an absolute path."
        )


class EvaluateSuiteConfig(BaseModel):
    """Canonical config consumed by ``EvaluateSuiteJob.run``.

    ``agent`` and ``output`` are optional here so the local run path can
    fall back to ``Path.cwd()`` / ``Path.cwd()/runs/batch-<ts>``.  The
    submit path uses :class:`EvaluateSuiteSubmitConfig` (where both are
    required) as its wire schema, and ``compile()`` re-validates that
    both ended up non-``None`` before dispatch.
    """

    evals: str = Field(description="Path to the directory of eval tasks.")
    agent: str | None = Field(default=None, description="Agent root (defaults to repo root in local run).")
    runner: Literal["auto", "harbor", "nat"] = Field(default="auto", description="Eval runner.")
    prefer: Literal["harbor", "nat"] = Field(default="nat", description="Tiebreaker when both markers present.")
    concurrency: int = Field(default=4, ge=1, description="Parallel eval concurrency.")
    skip_build: bool = Field(default=False, description="Skip docker build (Harbor only).")
    output: str | None = Field(
        default=None, description="Output dir for batch artifacts (defaults to cwd/runs in local run)."
    )
    filter_glob: str | None = Field(default=None, description="Glob filter on eval names.")
    repeats: int = Field(default=1, ge=1, description="Trials per eval (median aggregation when >1).")
    anthropic_api_key_secret: str | None = Field(
        default=None,
        description=(
            "Name of a platform Secret holding the Anthropic API key.  When set on a platform submission, "
            "the value is injected as ``ANTHROPIC_API_KEY`` into the dispatched subprocess so "
            "the Harbor eval tasks' LLM-judge calls reach Anthropic.  Local (in-process) runs "
            "read ``ANTHROPIC_API_KEY`` from the calling shell as before."
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


class EvaluateSuiteSubmitConfig(EvaluateSuiteConfig):
    """Submit-side wire schema for ``EvaluateSuiteJob``.

    Narrows ``agent`` and ``output`` from ``str | None`` to ``str`` (no
    default) so the OpenAPI contract matches the runtime behaviour: the
    subprocess executor's work dir is not the caller's cwd, so the
    canonical ``EvaluateSuiteConfig`` fallback to ``Path.cwd()`` would
    silently land in ``/tmp/nhx-subprocess-jobs/.../task-*/``.  Making
    them required at the schema layer surfaces the requirement as a
    422 at submit time instead of a confusing dispatched-job failure.
    """

    agent: str = Field(description="Agent root.")
    output: str = Field(description="Output dir for batch artifacts.")


class EvaluateSuiteJob(NemoJob):
    """Run a suite of containerized eval tasks against an agent."""

    name: ClassVar[str] = "evaluate-suite"
    description: ClassVar[str] = "Run a directory of containerized eval tasks (Harbor or NAT) against an agent."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = EvaluateSuiteConfig
    input_spec_schema: ClassVar[type[BaseModel]] = EvaluateSuiteSubmitConfig

    @classmethod
    async def to_spec(  # ty: ignore[invalid-method-override]
        cls,
        input_spec: EvaluateSuiteSubmitConfig,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> EvaluateSuiteConfig:
        # input_spec_schema is a strict subclass — re-pack to drop the input
        # label so downstream consumers (compile, run) see the canonical type.
        return EvaluateSuiteConfig.model_validate(input_spec.model_dump())

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: EvaluateSuiteConfig,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        """Single-step HelixJobSpec running ``nemo_agents_plugin.tasks.evaluate_suite``.

        Dispatched by the platform's host-subprocess executor — same machine as the
        platform (and the user's docker daemon).  No dedicated container image.
        """
        from nemo_helix_plugin.jobs.api_factory import (
            EnvironmentVariable,
            EnvironmentVariableFromSecret,
            HelixJobStep,
            SubprocessExecutionProviderSpec,
        )
        from nemo_helix_plugin.jobs.constants import (
            DEFAULT_JOB_STORAGE_PATH,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
        )

        # Subprocess work dir is /tmp/nhx-subprocess-jobs/.../task-..., not the
        # caller's cwd.  Relative paths silently fail at preflight or stash
        # outputs in the ephemeral work dir; None defaults fall back to
        # ``Path.cwd()`` inside run() — same hazard.  Require both up front.
        _require_absolute(spec.evals, "evals")
        if spec.agent is None:
            raise HelixJobCompilationError(
                "'agent' is required when submitting (defaults to Path.cwd() in run(), "
                "which inside the subprocess executor is the empty task scratch dir)."
            )
        if spec.output is None:
            raise HelixJobCompilationError(
                "'output' is required when submitting (defaults to Path.cwd()/runs/... in "
                "run(), writing artifacts to the ephemeral subprocess scratch dir)."
            )
        _require_absolute(spec.agent, "agent")
        _require_absolute(spec.output, "output")

        # URL workspace is the auth boundary; overwrite any spec workspace
        # field if one is added later.  Today the config has no `workspace`
        # field so this just adds the key (Pydantic ``extra="ignore"`` on
        # the subprocess-side validate drops it).
        spec_dict = spec.model_dump(mode="json")
        spec_dict["workspace"] = workspace

        environment: list[EnvironmentVariable] = [
            EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH),
        ]
        if spec.anthropic_api_key_secret:
            # The subprocess backend's sanitized environment does not inherit
            # ANTHROPIC_API_KEY from the platform process.  Harbor eval tasks
            # call Anthropic for LLM-judge scoring, so inject from a Secret
            # to keep submitted jobs functional.
            environment.append(
                EnvironmentVariable(
                    name="ANTHROPIC_API_KEY",
                    from_secret=EnvironmentVariableFromSecret(name=spec.anthropic_api_key_secret),
                )
            )
        if spec.anthropic_base_url:
            environment.append(EnvironmentVariable(name="ANTHROPIC_BASE_URL", value=spec.anthropic_base_url))

        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="evaluate-suite",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agents_plugin.tasks.evaluate_suite"],
                    ),
                    config=spec_dict,
                    environment=environment,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext | None = None) -> dict:
        from nemo_agents_plugin.improvement import preflight
        from nemo_agents_plugin.improvement.runners.detect import detect_runner, get_runner

        cfg = EvaluateSuiteConfig.model_validate(config)
        evals_dir = Path(cfg.evals).resolve()
        agent_root = Path(cfg.agent).resolve() if cfg.agent else Path.cwd()
        output = (
            Path(cfg.output).resolve()
            if cfg.output
            else (Path.cwd() / "runs" / f"batch-{datetime.now(timezone.utc).strftime('%Y-%m-%d__%H-%M-%S')}")
        )

        # Preflight: fail fast before any slow work
        preflight.check_evals_dir(evals_dir)
        preflight.check_docker()

        runner = detect_runner(evals_dir, prefer=cfg.prefer) if cfg.runner == "auto" else get_runner(cfg.runner)
        logger.info("Using runner: %s (evals=%s)", runner.name, evals_dir)

        # Per-runner preflights
        if runner.name == "harbor":
            preflight.check_harbor()
            if not cfg.skip_build:
                preflight.check_dockerfile(agent_root)
        elif runner.name == "nat":
            preflight.check_nat_runner(agent_root)

        evals = runner.discover(evals_dir)
        if cfg.filter_glob:
            from fnmatch import fnmatch as _fn

            evals = [e for e in evals if _fn(e.name, cfg.filter_glob)]
        if not evals:
            # Raising instead of returning {"status": "no-evals-found"} so the
            # platform dispatcher marks the job as error rather than completed.
            # A silent success here is invisible in Studio's Jobs view.
            filter_clause = f" matching {cfg.filter_glob!r}" if cfg.filter_glob else ""
            raise RuntimeError(f"No evals found in {evals_dir} (runner={runner.name}){filter_clause}.")

        batch = asyncio.run(
            runner.run_batch(
                evals=evals,
                batch_dir=output,
                concurrency=cfg.concurrency,
                skip_build=cfg.skip_build,
                project_root=agent_root,
                repeats=cfg.repeats,
            )
        )
        if ctx is not None:
            expected_executions = len(evals) * (cfg.repeats if runner.name == "harbor" else 1)
            usage = evaluation_batch_token_usage(
                output,
                runner=runner.name,
                expected_executions=expected_executions,
            )
            if usage is not None:
                ctx.usage.report_totals(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        return {
            "status": "completed",
            "runner": runner.name,
            "batch_id": batch.batch_id,
            "batch_dir": str(output),
            "passed": batch.pass_count,
            "failed": batch.fail_count,
            "errors": batch.error_count,
            "total": len(batch.results),
        }
