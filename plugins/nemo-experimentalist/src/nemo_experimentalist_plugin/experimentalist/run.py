# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable optimizer Experimentalist run orchestration."""

import logging
from pathlib import Path
from typing import TextIO

from nemo_experimentalist_plugin.entities import DatasetRef
from nemo_experimentalist_plugin.experimentalist.agent import build_experimentalist_agent
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import (
    make_experimentalist_backend,
)
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter, Verbosity
from nemo_experimentalist_plugin.experimentalist.runner import ExperimentRunner
from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryOptimizerConfig
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.nooa_model_client import (
    ConfiguredModelClients,
    ConfiguredModelRefs,
    activate_model_clients,
    resolve_model_clients,
)

logger = logging.getLogger(__name__)


def build_run_reporter(
    *,
    run_dir: Path,
    agent: str,
    insight: str | None,
    strategy: str = "evolutionary",
    sink: TextIO | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
) -> RunReporter:
    """Construct a reporter and announce the run start."""
    reporter = RunReporter(sink=sink, verbosity=verbosity)
    reporter.run_started(run_dir=run_dir, agent=agent, insight=insight, strategy=strategy)
    return reporter


async def run_experimentalist(
    *,
    agent: str | Path | None = None,
    ethos: str | None = None,
    insight: Path | str | None,
    train_dataset: DatasetRef,
    validation_dataset: DatasetRef,
    experiment_dir: Path,
    workspace: str,
    client: AsyncNemoClient,
    model_refs: ConfiguredModelRefs | None = None,
    config: EvolutionaryOptimizerConfig,
    task_template: DatasetRef | None = None,
    framework_skills_dirs: list[Path] | None = None,
) -> str:
    """Build and run the Experimentalist against an agent and dataset.

    Args:
        agent: Optional baseline agent for Mode 2, or an override for the agent
            referenced by ``insight``. A local directory path or a git ``url@ref``; a git
            source is fetched by the backend and enables opening a draft PR/MR for
            the winner against that ref.
        ethos: Optional URI of a markdown file describing the agent under test.
            Materialized by the backend and threaded to components that use it.
        insight: Optional Mode 1 insight — a local Insight file path or a platform
            insight id (fetched from the platform by the backend).
        train_dataset: Evaluator dataset reference for training.
        validation_dataset: Evaluator dataset reference for validation.
        task_template: Evaluator-specific task template used for production traces.
        experiment_dir: Working directory for optimization artifacts.
        workspace: Platform workspace.
        model_refs: Optional explicit default/fast Model Entity IDs. Unset uses the
            active Platform CLI context.
        client: Caller-owned Platform client.
        config: Evolutionary optimizer configuration.

    Returns:
        Terminal optimization summary.
    """
    # Logging is configured at the entry-point boundary (the root ``nemo`` CLI
    # callback runs ``configure_logging`` before dispatching this subcommand),
    # so this library function leaves root logging untouched.
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.resolve()
    # Leave ``agent`` unresolved: it may be a git ``url@ref`` (not a filesystem path).
    # The backend's get_agent_code handles both local dirs and git sources.
    # A local file path is resolved; a platform insight id (str) is forwarded verbatim.
    insight = insight.resolve() if isinstance(insight, Path) else insight

    reporter = build_run_reporter(
        run_dir=experiment_dir,
        agent=str(agent) if agent else "(from insight)",
        insight=str(insight) if insight is not None else None,
    )

    model_clients: ConfiguredModelClients | None = None
    try:
        backend = make_experimentalist_backend(
            client=client,
            experiments_output=str(experiment_dir),
            storage=config.storage,
        )
        # Every component resolves its models through the platform (#1159), so the resolved
        # clients have to stay active for the whole run, not just while the runner is built.
        model_clients = await resolve_model_clients(client, model_refs)
        with activate_model_clients(model_clients):
            result = await ExperimentRunner(
                backend=backend,
                strategy=build_experimentalist_agent(
                    working_dir=experiment_dir,
                    config=config,
                    framework_skills_dirs=framework_skills_dirs,
                ),
                config=config,
                workspace=workspace,
                root=experiment_dir,
                agent=agent,
                ethos=ethos,
                insight=insight,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                task_template=task_template,
                reporter=reporter,
            ).run()
    finally:
        if model_clients is not None:
            await model_clients.aclose()

    winner = result.winner
    reporter.run_finished(
        winner=winner.label if winner is not None else None,
        scores=dict(winner.rewards["validation"].metrics) if winner and winner.rewards["validation"].metrics else {},
        report_path=(experiment_dir / "eval-and-optimize" / "OPTIMIZATION.md") if winner is not None else None,
    )
    return result.summary
