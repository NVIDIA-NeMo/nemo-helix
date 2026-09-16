# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an agent on Harvey Labs' Legal Agent Benchmark (LAB) the NeMo Platform way.

Native `AgentEvalTask`s (built from LAB's public data) run through NeMo Fabric, and LAB's rubric is
scored by `LabRubricMetric`, which calls **LAB's own `evaluation/score_rubric`** over the trial's
`workspace` evidence — task -> trial -> metric -> score, execution and scoring decoupled.

LAB's three skills (docx/pptx/xlsx) are delivered as **task inputs** (seeded into each workspace under
`skills/<name>/`, manuals prepended to the instruction) — matching how LAB provides all skills at once,
and sidestepping Fabric's one-skill-per-runtime limitation. Input documents are seeded under
`documents/`; the agent writes deliverables under `output/`, which the metric grades.

Runner choices:

* `--runtime host`      — `FabricAgentRuntime` (host workspaces). The document toolchain (pandoc,
  libreoffice, python-docx, ...) that the skills rely on must be on the host.
* `--runtime container` — `FabricContainerRuntime` (Docker sandbox). Pass `--image` to supply a
  prebuilt image that includes the document toolchain (image *building* is your concern); without it,
  the stock Fabric image is used and the skills' scripts will fail for lack of tooling.

The judge is LAB's `Judge(model=--judge-model)`, which runs in **this (eval) process** and reads its
credential from the environment (for an OpenAI-compatible endpoint: `OPENAI_API_KEY` +
`OPENAI_BASE_URL`). LAB's extraction stack (pandoc, libreoffice, python-docx, python-redlines, pandas,
openpyxl, pdfplumber, markitdown) must also be available in this process.

Run from the repository root (codex agent on OpenAI via a ~/.codex login; judge on an OpenAI-compatible
endpoint such as NVIDIA). codex is the default harness because it is the only one whose shell tool runs
LAB's skill scripts under Fabric, and it is configured closed-book (no web search) to match LAB::

    NVIDIA_API_KEY=... \\
    .venv/bin/python -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_fabric.run_legal_agent_bench_fabric \\
        --runtime host --harness codex-cli --model gpt-5.5 \\
        --judge-model openai/gpt-oss-120b \\
        --judge-base-url https://integrate.api.nvidia.com/v1 --judge-api-key-env NVIDIA_API_KEY \\
        --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig

from .prepare_lab_taskset import build_lab_tasks, ensure_lab_source

logger = logging.getLogger(__name__)


def _fabric_config(harness: str, *, provider: str, model: str, api_key_env: str) -> dict:
    common = {
        "metadata": {"name": "lab-fabric-eval"},
        "models": {"default": {"provider": provider, "model": model, "api_key_env": api_key_env}},
    }
    # Adapter ids select the harness implementation directly; Fabric owns its execution mechanism.
    if harness == "deepagents":
        # LangChain Deep Agents — provider-agnostic; for provider=nvidia it targets NVIDIA's
        # OpenAI-compatible endpoint. The recommended harness for NVIDIA-hosted models.
        return {
            **common,
            "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
            "runtime": {"input_schema": "chat", "output_schema": "message"},
        }
    if harness == "codex-cli":
        # codex runs as the SDK adapter here; CLI-only settings (e.g. skip_git_repo_check) are rejected.
        # Closed-book for LAB fidelity (LAB is documents-only; its reference harness has no web tool):
        #   * config_overrides.web_search="disabled" turns off codex's native web-search tool, which runs
        #     at the model provider and is NOT gated by the sandbox network — it must be disabled explicitly.
        #   * sandbox="workspace-write" lets the skill scripts write deliverables; under it the shell's
        #     network_access defaults to false (set explicitly here), so shell commands have no egress.
        return {
            **common,
            "harness": {
                "adapter_id": "nvidia.fabric.codex",
                "settings": {
                    "sandbox": "workspace-write",
                    "config_overrides": {"web_search": "disabled", "sandbox_workspace_write.network_access": False},
                },
            },
        }
    return {  # hermes-sdk
        **common,
        "harness": {"adapter_id": "nvidia.fabric.hermes", "resolution": "preinstalled"},
        "runtime": {"input_schema": "chat", "output_schema": "message"},
    }


def _non_negative_int(raw: str) -> int:
    """argparse type for ``--limit``: reject negatives so 0 unambiguously means "no tasks"."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return value


def _build_runtime(args: argparse.Namespace):
    config = _fabric_config(
        args.harness, provider=args.agent_provider, model=args.model, api_key_env=args.agent_api_key_env
    )
    if args.runtime == "host":
        # LAB scoring uses the workspace deliverables, not the ATIF trajectory, so trajectory capture is
        # optional here. Disable it with --no-trajectory if your nemo-fabric's `enable_relay` signature
        # differs from the SDK's (newer Fabric dropped the `config=` kwarg for `observability=`).
        return FabricAgentRuntime(config=config, work_root=args.work_root, capture_trajectory=not args.no_trajectory)
    # Container: isolated sandbox. Pass a doc-tooling image via --image (build it yourself).
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
    from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers.docker import DockerSandboxProvider
    from nemo_evaluator_sdk.values.common import SecretRef

    if not args.image:
        logger.warning("No --image given: the stock Fabric image lacks LAB's doc toolchain; skill scripts will fail.")
    return FabricContainerRuntime(
        config,
        provider=DockerSandboxProvider(),
        secrets={args.agent_api_key_env: SecretRef(root=args.agent_api_key_env)},
        image=args.image,  # None -> build-if-missing stock image
    )


async def _main(args: argparse.Namespace) -> None:
    source_root = ensure_lab_source(args.source_dir, allow_download=not args.no_download)
    tasks = build_lab_tasks(
        source_root,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key_env=args.judge_api_key_env,
        limit=args.limit,
        task_ids=set(args.task_ids) if args.task_ids else None,
        judge_parallel=args.judge_parallel,
    )
    runtime = _build_runtime(args)

    result = await AgentEvaluator().run(
        tasks=tasks,
        target=runtime,
        config=AgentEvalRunConfig(work_dir=Path(args.output_dir), parallelism=args.parallelism),
    )
    result.persist()

    print(f"run_id: {result.run_id}  tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    print("Aggregate scores:")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: {aggregate.headline_value}")
    print(f"\nRun bundle (run.json, trials.jsonl, scores.jsonl, summary.json, report.html): {args.output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default="./data/lab-source", help="Where LAB source is downloaded/extracted.")
    p.add_argument("--no-download", action="store_true", help="Fail if LAB source is not already present.")
    p.add_argument("--runtime", choices=("host", "container"), default="host")
    # codex is the default: it is the only harness whose shell tool actually runs LAB's docx/pptx/xlsx skill
    # scripts under Fabric (deepagents' `execute` is inert with the host FilesystemBackend, so it cannot
    # produce document deliverables). codex is OpenAI-provider-locked, so the agent defaults are OpenAI.
    p.add_argument("--harness", choices=("codex-cli", "deepagents", "hermes-sdk"), default="codex-cli")
    p.add_argument("--model", default="gpt-5.5", help="Agent model slug (codex is OpenAI-only).")
    p.add_argument("--agent-provider", default="openai", help="Fabric model provider for the agent.")
    p.add_argument("--agent-api-key-env", default="OPENAI_API_KEY", help="Env var holding the agent model key.")
    p.add_argument(
        "--image", default=None, help="Container runtime: a prebuilt sandbox image with LAB's doc toolchain."
    )
    p.add_argument("--judge-model", default="openai/gpt-oss-120b", help="Judge model name.")
    p.add_argument(
        "--judge-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible judge endpoint (defaults to $OPENAI_BASE_URL). When set, grade via a "
        "chat.completions adapter using LAB's exact prompt; else use LAB's native prefix-routed Judge.",
    )
    p.add_argument("--judge-api-key-env", default="OPENAI_API_KEY", help="Env var holding the judge endpoint key.")
    p.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable Relay ATIF trajectory capture (host runtime). Use if nemo-fabric's enable_relay API "
        "differs from the SDK's; LAB scoring doesn't need the trajectory.",
    )
    p.add_argument("--work-root", default=None, help="Host runtime: root for per-task workspaces.")
    p.add_argument("--output-dir", default="./results/legal_agent_bench_fabric", help="Run bundle + report.html.")
    p.add_argument("--limit", type=_non_negative_int, default=None, help="Score only the first N tasks.")
    p.add_argument(
        "--task-ids", nargs="*", default=None, help="Flattened task ids to run (e.g. one per area for a diverse run)."
    )
    p.add_argument("--parallelism", type=int, default=4, help="Tasks scored concurrently.")
    p.add_argument(
        "--judge-parallel",
        type=int,
        default=1,
        help="Concurrent judge calls per task (default 1; build.nvidia.com rate-limits large models).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(_parse_args()))
