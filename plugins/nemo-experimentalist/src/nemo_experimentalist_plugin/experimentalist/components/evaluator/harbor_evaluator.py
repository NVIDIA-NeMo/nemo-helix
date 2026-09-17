# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor evaluator that delegates orchestration to the NeMo Evaluator SDK.

``HarborEvaluator`` builds Harbor's ``JobConfig`` and drives ``Job`` itself. This
evaluator hands that job to the SDK's ``HarborAgentTaskRunner`` instead: the SDK
owns the ``JobConfig``, the Harbor-valid result cache, and the scoped
agent import. Harbor still does the work underneath — the difference is who owns
the orchestration.

Results are read back off the Harbor job directory through the same
:func:`~...evaluator.harbor.trials_from_job_dir` adapter ``harbor_native`` uses.
**That sharing is the point**: one adapter over one source of truth is what makes
the two evaluator types produce equivalent :class:`TrialResult` objects, rather
than two parsers that have to be kept in agreement.

Scoring is deliberately left to Harbor. Its verifier already computes the rewards
and writes them to ``<trial>/result.json``; the SDK's metric layer only reads them
back, so running the trials through ``AgentEvaluator`` would add a scoring pass
whose output this evaluator discards. When ``harbor_native`` is eventually removed,
the natural move is to consume ``AgentEvalTrial`` directly — see
``temp/evaluator/plans/AgentEvalResult_as_shared_harbor_interface.md``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    discover_harbor_tasks,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_experimentalist_plugin.entities import Dataset, TrialResult, local_path_from_uri
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.evaluator.base import (
    EvaluatorConfig,
    EvaluatorType,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor import (
    DEFAULT_TRACE_ARTIFACT_SOURCE,
    HarborDataset,
    resolve_harbor_run_inputs,
    trials_from_job_dir,
)
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)


class HarborTaskNameError(ValueError):
    """A dataset task could not be mapped onto exactly one Harbor task name."""


class HarborRunnerConfig(EvaluatorConfig):
    """Configuration for the SDK-backed Harbor evaluator.

    Every field maps onto exactly one ``HarborRuntimeConfig`` field. Unknown keys
    are rejected rather than silently ignored: several plain-``HarborEvaluator``
    options (notably the full ``retry`` model) have no unambiguous SDK equivalent,
    so passing them here is a configuration error, not a no-op.

    ``agent_dir`` is deliberately absent — it is always derived from the candidate
    being evaluated, so a config cannot point the run at a different agent.

    One asymmetry to know when A/B-ing against ``HarborEvaluatorConfig``: at their
    defaults the two are equivalent (Harbor resolves an unset phase multiplier to
    the global ``timeout_multiplier``, which defaults to ``1.0``), but they diverge
    once tuned. This config exposes the global ``timeout_multiplier`` and leaves the
    phase multipliers unset so they inherit it; the plain config has no global knob
    and pins each phase to ``1.0``, which masks it. Set the phase multipliers
    explicitly on both sides when comparing non-default timeouts.
    """

    model_config = ConfigDict(extra="forbid")

    job_name: str | None = Field(
        default=None,
        description=(
            "Harbor job name. Defaults to the loop's deterministic '<candidate>-<dataset>', "
            "which is what makes the SDK's Harbor-valid result cache usable."
        ),
    )
    jobs_dir: Path = Field(
        default=Path("eval-and-optimize") / "results",
        description="Directory to store job results, resolved relative to the experiment directory.",
    )
    n_attempts: int = Field(default=1, ge=1, description="Number of attempts Harbor runs per task.")
    n_concurrent_trials: int = Field(
        default=os.cpu_count() or 4, ge=1, description="Maximum number of concurrent Harbor trials."
    )
    quiet: bool = Field(default=False, description="Suppress Harbor's trial progress display.")
    artifacts: list[str] = Field(default=[], description="Additional Harbor artifact sources to collect per trial.")
    trace_dir: str = Field(
        default=DEFAULT_TRACE_ARTIFACT_SOURCE,
        description=(
            "Container path of agent traces, collected into the trial's 'traces' artifact directory "
            "and published to each verifier as the TRACE_DIR environment variable."
        ),
    )
    trace_format: Literal["otlp", "atif"] = Field(
        default="otlp",
        description=(
            "Which collected trace artifact becomes the trial's primary trace. Both formats remain available in "
            "the trial resources."
        ),
    )
    max_retries: int = Field(default=0, ge=0, description="Harbor per-trial retries on transient failures.")
    timeout_multiplier: float | None = Field(default=None, description="Global Harbor timeout multiplier.")
    agent_timeout_multiplier: float | None = Field(default=None, description="Agent-phase timeout multiplier.")
    verifier_timeout_multiplier: float | None = Field(default=None, description="Verifier-phase timeout multiplier.")
    agent_setup_timeout_multiplier: float | None = Field(default=None, description="Agent-setup timeout multiplier.")
    environment_build_timeout_multiplier: float | None = Field(
        default=None, description="Environment-build timeout multiplier."
    )
    import_path: str = Field(
        default="harbor_wrapper:WrappedAgent",
        description="Harbor agent import path resolved inside the candidate directory.",
    )


class HarborRunnerOutcomeEvaluator(roles.OutcomeEvaluator):
    """Run Harbor through the SDK's ``HarborAgentTaskRunner`` and parse the job dir."""

    name = "harbor-runner"
    dataset_type = HarborDataset
    config_type = HarborRunnerConfig

    evaluator_type: EvaluatorType = "harbor-runner"

    def __init__(
        self,
        options: HarborRunnerConfig | None = None,
        experiment_dir: Path | None = None,
    ) -> None:
        super().__init__(options or HarborRunnerConfig(), experiment_dir=experiment_dir)

    async def _run(
        self,
        agent: Path,
        dataset: Dataset,
        options: EvaluatorConfig,
    ) -> Sequence[TrialResult]:
        if not isinstance(options, HarborRunnerConfig):
            raise TypeError("Options must be a HarborRunnerConfig")

        inputs = await resolve_harbor_run_inputs(agent, dataset, options, self.experiment_dir)
        harbor_dataset = inputs.dataset
        sdk_tasks = _sdk_tasks_for(harbor_dataset)

        runtime_config = HarborRuntimeConfig(
            jobs_dir=inputs.jobs_dir,
            job_name=inputs.job_name,
            agent_import_path=options.import_path,
            agent_dir=inputs.agent_path,
            n_attempts=options.n_attempts,
            n_concurrent_trials=options.n_concurrent_trials,
            quiet=options.quiet,
            force_rerun=options.force_rerun,
            artifacts=list(options.artifacts),
            trace_dir=options.trace_dir,
            max_retries=options.max_retries,
            timeout_multiplier=options.timeout_multiplier,
            agent_timeout_multiplier=options.agent_timeout_multiplier,
            verifier_timeout_multiplier=options.verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=options.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=options.environment_build_timeout_multiplier,
        )

        # Two different name spaces, and mixing them up produces either an empty
        # run or an empty cache:
        #   * Harbor's local-dataset `task_names` filter matches the task
        #     *directory* name, which is the Experimentalist task id.
        #   * `result.json` records `[task].name` from task.toml, which is what
        #     the SDK's tasks are keyed by and what its cache counts.
        runner = HarborAgentTaskRunner(
            config=runtime_config,
            dataset_path=inputs.dataset_path,
            task_names=[task.id for task in harbor_dataset.tasks],
        )
        # Called for its effect — running (or resuming) the Harbor job. The returned
        # trials are not the contract: the job dir is, and it is shared with
        # `harbor_native`, which is what keeps the two types equivalent. Note the
        # trials are not *lossy* — `metadata["reward_details"]` carries every verifier
        # reward — so the reason to ignore them is the shared source of truth, not
        # missing data.
        sdk_trials = await runner.run_tasks(list(sdk_tasks.values()))
        logger.debug("SDK Harbor runner returned %d trial(s) for job %s", len(sdk_trials), inputs.job_name)

        return trials_from_job_dir(inputs.job_dir, harbor_dataset.tasks, trace_format=options.trace_format)


def _sdk_tasks_for(dataset: HarborDataset) -> dict[str, AgentEvalTask]:
    """Map each selected dataset task onto the SDK task carrying its full Harbor name.

    The mapping is by task *directory*, never by name similarity: the SDK reads
    ``[task].name`` from the same ``task.toml`` Harbor will read, so matching on
    the directory guarantees the ids we hand the runner are exactly the
    ``task_name`` values Harbor writes into ``result.json``.

    Args:
        dataset: The (possibly subset) Harbor dataset being evaluated.

    Returns:
        dict[str, AgentEvalTask]: Selected tasks keyed by Experimentalist task id,
        in dataset order.

    Raises:
        ValueError: If the dataset has no resolvable source directory.
        HarborTaskNameError: If a selected task has no discovered counterpart, or
            if two selected tasks resolve to the same full Harbor name.
    """
    if dataset.source is None:
        raise ValueError("Harbor dataset source is required")
    dataset_path = local_path_from_uri(dataset.source.uri, context="Harbor dataset reference").resolve()
    discovered = discover_harbor_tasks(dataset_path)
    by_dir: dict[Path, AgentEvalTask] = {}
    for sdk_task in discovered:
        task_dir = sdk_task.metadata.get("harbor_task_dir")
        if isinstance(task_dir, str) and task_dir:
            by_dir[Path(task_dir).resolve()] = sdk_task

    selected: dict[str, AgentEvalTask] = {}
    full_names: dict[str, str] = {}
    for task in dataset.tasks:
        if not task.uri:
            raise HarborTaskNameError(f"Harbor task {task.id!r} has no URI, so its Harbor name cannot be resolved")
        task_dir = local_path_from_uri(task.uri, context="Harbor task reference").resolve()
        sdk_task = by_dir.get(task_dir)
        if sdk_task is None:
            raise HarborTaskNameError(
                f"Harbor task {task.id!r} at {task_dir} was not discovered under dataset {dataset_path}; "
                "the dataset directory and the task directories must agree"
            )
        if sdk_task.id in full_names:
            raise HarborTaskNameError(
                f"Harbor tasks {full_names[sdk_task.id]!r} and {task.id!r} both declare [task].name = "
                f"{sdk_task.id!r}; task names must be unique within a dataset"
            )
        full_names[sdk_task.id] = task.id
        selected[task.id] = sdk_task
    return selected


def harbor_task_names(dataset: HarborDataset) -> dict[str, str]:
    """Return ``{experimentalist_task_id: full_harbor_name}`` for a Harbor dataset.

    The readable view of the two-namespace translation ``_run`` depends on. Pass a
    ``dataset.subset(...)`` to scope it to selected tasks.

    Args:
        dataset: Harbor dataset whose source directory holds the tasks.

    Returns:
        dict[str, str]: Short Experimentalist task id to full Harbor ``[task].name``.
    """
    return {task_id: sdk_task.id for task_id, sdk_task in _sdk_tasks_for(dataset).items()}
