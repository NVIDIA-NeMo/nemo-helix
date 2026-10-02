# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptimizeSkillsJob — the optimize-skills loop.

Registered under ``nemo.jobs`` as ``agents.optimize-skills``.

The CLI command ``nemo agents optimize-skills --spec '{...}'`` submits it to
the platform; the jobs controller dispatches a subprocess on the same host
that runs the platform and the result lands in ``nemo jobs list`` / Studio's
Jobs view.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from nemo_agents_plugin.jobs.evaluate_suite import _require_absolute
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OptimizeSkillsConfig(BaseModel):
    evals: str = Field(description="Path to the directory of eval tasks.")
    agent: str = Field(description="Agent root (where the loop runs from; skills live under it).")
    skills_path: str = Field(default=".agents/skills", description="Relative path inside agent where skills live.")
    filter_glob: str | None = Field(default=None, description="Glob filter on eval names (e.g. 'files-*').")
    iterations: int = Field(default=3, ge=1)
    concurrency: int = Field(default=2, ge=1)
    state: str | None = Field(default=None, description="Path to loop_state.json; default: <agent>/loop_state.json.")
    initial_batch: str | None = Field(default=None, description="Existing batch dir to seed from.")
    full_verification: bool = Field(default=False)
    open_pr: bool = Field(default=False)
    repeats: int = Field(default=1, ge=1)
    analyze_only: bool = Field(
        default=False,
        description=(
            "Analyze-only mode: consume an existing --initial-batch, generate suggestions, exit. "
            "Skips worktree, apply, verify, MR. Works with any AUT (Harbor or NAT)."
        ),
    )
    trace_parser: Literal["claude-code"] = Field(
        default="claude-code",
        description=(
            "Trace parser for the agent's traces. Today only 'claude-code' is registered "
            "(parses Claude Code session.jsonl). Future parsers — e.g. for NAT IntermediateStep "
            "records — extend this Literal."
        ),
    )
    anthropic_api_key_secret: str | None = Field(
        default=None,
        description=(
            "Name of a platform Secret holding the Anthropic API key.  When set on a platform submission, "
            "the value is injected as ``ANTHROPIC_API_KEY`` into the dispatched subprocess so "
            "the loop's ``check_anthropic_api`` / Claude analyzer preflights pass.  Local "
            "(in-process) runs read ``ANTHROPIC_API_KEY`` from the calling shell as before."
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


class OptimizeSkillsJob(NemoJob):
    """Run the optimize-skills loop end-to-end."""

    name: ClassVar[str] = "optimize-skills"
    description: ClassVar[str] = "Optimize an agent's skills against eval failures via a coding agent (Claude)."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = OptimizeSkillsConfig

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: OptimizeSkillsConfig,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        """Single-step HelixJobSpec running ``nemo_agents_plugin.tasks.optimize_skills``.

        Dispatched by the platform's host-subprocess executor — same machine as the
        platform (and the user's docker daemon / Claude CLI).  No dedicated container image.
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
        # caller's cwd; reject relative paths up front.
        _require_absolute(spec.evals, "evals")
        _require_absolute(spec.agent, "agent")
        _require_absolute(spec.state, "state")
        _require_absolute(spec.initial_batch, "initial_batch")

        # Mirror the run() guard so remote submission fails fast with a 422 instead of
        # spawning a doomed subprocess that errors inside analyze_only.
        if spec.analyze_only and not spec.initial_batch:
            raise HelixJobCompilationError(
                "'analyze_only' requires 'initial_batch' to point at an existing batch "
                "directory; the analyze-only path has no batch to consume otherwise."
            )

        # URL workspace is the auth boundary.  Config has no ``workspace``
        # field today; injecting the key is a no-op now and a guard against
        # future workspace-scoped fields.
        spec_dict = spec.model_dump(mode="json")
        spec_dict["workspace"] = workspace

        environment: list[EnvironmentVariable] = [
            EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH),
        ]
        if spec.anthropic_api_key_secret:
            # The subprocess backend's sanitized environment does not inherit
            # ANTHROPIC_API_KEY from the platform process.  Inject from a
            # platform Secret so ``preflight.check_anthropic_api`` and the
            # Claude coding-agent preflight pass inside the dispatched step.
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
                    name="optimize-skills",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agents_plugin.tasks.optimize_skills"],
                    ),
                    config=spec_dict,
                    environment=environment,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext | None = None) -> dict[str, Any]:
        from nemo_agents_plugin.improvement import preflight
        from nemo_agents_plugin.improvement.coding_agents.claude import ClaudeCodingAgent
        from nemo_agents_plugin.improvement.loop import run_analyze_only, run_loop
        from nemo_agents_plugin.improvement.models import _serialize_dict

        cfg = OptimizeSkillsConfig.model_validate(config)
        agent_root = Path(cfg.agent).resolve()
        evals_dir = Path(cfg.evals).resolve()
        state_path = Path(cfg.state).resolve() if cfg.state else agent_root / "loop_state.json"
        initial = Path(cfg.initial_batch).resolve() if cfg.initial_batch else None

        if cfg.analyze_only:
            if initial is None:
                raise ValueError("analyze_only=True requires initial_batch to point at an existing batch directory.")
            # Skip Docker/Harbor/Claude/forge preflight — those gate apply, not analysis.
            preflight.check_evals_dir(evals_dir)
            preflight.check_anthropic_api()
            state = asyncio.run(
                run_analyze_only(
                    evals_dir=evals_dir,
                    initial_batch_dir=initial,
                    skills_path=cfg.skills_path,
                    trace_parser=cfg.trace_parser,
                )
            )
            return _serialize_dict(state)

        # Preflight: fail fast before any slow work
        preflight.check_evals_dir(evals_dir)
        preflight.check_evals_inside_agent(agent_root, evals_dir)
        preflight.check_skills_path(agent_root, cfg.skills_path)
        preflight.check_docker()
        preflight.check_harbor()  # both runners need it for image inheritance
        # The loop accepts a Runner but rejects anything but harbor up-front
        # (see run_loop's runner-kind guard). It always invokes harbor with
        # skip_build=False, so the Dockerfile is always needed.
        preflight.check_dockerfile(agent_root)
        # Coding agent preflight (Claude CLI / OAuth for the apply step)
        ClaudeCodingAgent().preflight()
        # LLM analyzer preflight (Anthropic API for the gap-analysis step)
        preflight.check_anthropic_api()

        # MR/PR forge preflight — fail fast before slow eval work if the
        # branch won't be routable to a PR/MR at the end of the loop.
        if cfg.open_pr:
            preflight.check_forge_tooling()

        state = asyncio.run(
            run_loop(
                agent_root=agent_root,
                evals_dir=evals_dir,
                skills_path=cfg.skills_path,
                filter_glob=cfg.filter_glob,
                max_iterations=cfg.iterations,
                concurrency=cfg.concurrency,
                state_path=state_path,
                initial_batch_dir=initial,
                full_verification=cfg.full_verification,
                open_pr=cfg.open_pr,
                repeats=cfg.repeats,
                trace_parser=cfg.trace_parser,
            )
        )
        return _serialize_dict(state)
