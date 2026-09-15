# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable Eval Author run orchestration."""

from pathlib import Path
from typing import Protocol

from nemo_experimentalist_plugin.client import make_client
from nemo_experimentalist_plugin.entities import Dataset, DatasetRef, Task
from nemo_experimentalist_plugin.eval_author.models import EvalAuthorConfig, EvalAuthorResult
from nemo_experimentalist_plugin.experimentalist.components.dataset_staging import stage_eval_author_inputs
from nemo_experimentalist_plugin.experimentalist.components.evaluator.base import EvaluatorType
from nemo_experimentalist_plugin.experimentalist.components.evaluator.factory import DatasetFactory
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import (
    make_experimentalist_backend,
)
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter
from nemo_insights_plugin.entities import Insight
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.nooa_model_client import (
    ConfiguredModelClients,
    ConfiguredModelRefs,
    activate_model_clients,
    configured_model_refs,
    resolve_model_clients,
)


class _EvalAuthorAgent(Protocol):
    async def run(
        self,
        insight: Insight,
        agent_path: Path,
        task_template: Task,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        client: AsyncNemoClient,
    ) -> EvalAuthorResult: ...


async def run_eval_author(
    *,
    insight: Path | str,
    train_dataset: DatasetRef,
    validation_dataset: DatasetRef,
    task_template: DatasetRef,
    experiment_dir: Path,
    workspace: str,
    base_url: str | None,
    config: EvalAuthorConfig,
    agent: Path | str | None = None,
    evaluator_type: EvaluatorType = "harbor-native",
    model_refs: ConfiguredModelRefs | None = None,
) -> EvalAuthorResult:
    """Stage evaluation inputs, resolve one Insight, then run Eval Author.

    Args:
        insight: Local Insight path or platform Insight id.
        train_dataset: Training dataset to stage and augment.
        validation_dataset: Validation dataset to stage and augment.
        task_template: Local or Fileset-backed evaluator task template.
        experiment_dir: Working directory for authored artifacts.
        workspace: Platform workspace.
        base_url: Platform base URL. ``None`` uses the active platform context.
        config: Eval Author tuning parameters.
        agent: Optional agent source override. When absent, the Insight's agent is used.
        evaluator_type: Evaluator adapter used to parse datasets and task template.
        model_refs: Optional explicit default/fast Model Entity IDs. Unset uses
            the active Platform CLI context.

    Returns:
        EvalAuthorResult: containing the modified and newly created datasets, additional metrics
            and summary.
    """
    selected_model_refs = model_refs if model_refs is not None else configured_model_refs()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.resolve()
    insight_locator = str(insight.resolve()) if isinstance(insight, Path) else insight

    client = make_client(base_url)
    model_clients: ConfiguredModelClients | None = None
    try:
        model_clients = await resolve_model_clients(client, selected_model_refs)
        backend = make_experimentalist_backend(
            client=client,
            experiments_output=str(experiment_dir),
        )
        resolved_insight = await backend.get_insight(
            workspace=workspace,
            insight_id=insight_locator,
        )
        agent_ref = agent if agent is not None else resolved_insight.agent
        agent_path = experiment_dir / "eval_author" / "source-agent"
        await backend.get_agent_code(
            workspace=workspace,
            agent=agent_ref,
            dest=agent_path,
        )

        staged_inputs = await stage_eval_author_inputs(
            experiment_dir,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            task_template=task_template,
            client=client,
            workspace=workspace,
        )
        dataset_factory = DatasetFactory()
        parsed_train = dataset_factory.build_dataset(
            evaluator_type,
            staged_inputs.train_dataset,
        )
        parsed_validation = dataset_factory.build_dataset(
            evaluator_type,
            staged_inputs.validation_dataset,
        )
        parsed_template = dataset_factory.build_task_template(
            evaluator_type,
            staged_inputs.task_template,
        )
        with activate_model_clients(model_clients):
            eval_author = build_eval_author_agent(
                experiment_dir=experiment_dir,
                config=config,
            )
            return await eval_author.run(
                insight=resolved_insight,
                agent_path=agent_path,
                task_template=parsed_template,
                train_dataset=parsed_train,
                validation_dataset=parsed_validation,
                client=client,
            )
    finally:
        try:
            if model_clients is not None:
                await model_clients.aclose()
        finally:
            await client.close()


def build_eval_author_agent(
    *,
    experiment_dir: Path,
    config: EvalAuthorConfig,
    reporter: RunReporter | None = None,
) -> _EvalAuthorAgent:
    """Build the LLM-backed Eval Author agent lazily."""
    from nemo_experimentalist_plugin.eval_author.agent import build_eval_author_agent as _build_eval_author_agent

    return _build_eval_author_agent(
        experiment_dir=experiment_dir,
        config=config,
        reporter=reporter,
    )
