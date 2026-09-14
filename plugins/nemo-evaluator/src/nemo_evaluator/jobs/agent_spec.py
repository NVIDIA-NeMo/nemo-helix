# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec and target models for the agent-evaluation job.

These are the wire (submitter-facing) and canonical (resolved) DTOs that
:class:`~nemo_evaluator.jobs.agent_evaluate.AgentEvalJob` validates and runs,
plus the ``Target`` union describing what generates trials. They live in their
own module so the job and its compiler can both depend on them without importing
each other.
"""

from __future__ import annotations

from typing import Any, Literal, Self, TypeAlias

# Imported for their registration side effects: each module registers its bundle
# payload kind so MetricBundle payloads round-trip through validation.
import nemo_evaluator.shared.metric_bundles.cloudpickle  # noqa: F401
import nemo_evaluator.shared.metric_bundles.inline  # noqa: F401
from filesets import FilesetPathError, parse_fileset_ref
from nemo_evaluator.api.schemas import MetricInline, TaskInputs, TaskMetadataList, TasksetRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.metric_resolution import to_runtime_bundle, unresolved_model_refs
from nemo_evaluator.jobs.publication_spec import PublicationSpec
from nemo_evaluator.metric_refs import MetricRefOrInline
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_evaluator_sdk.values import Agent, Model, RunConfigOnline, RunConfigOnlineModel, SecretRef
from nemo_evaluator_sdk.values.agents import AgentBase
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ModelTarget(BaseModel):
    """Generate trials by calling a Model (OpenAI-compatible) endpoint.

    The prompt template *is* the request sent to the model, so it lives here with the endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["model"] = "model"
    model: Model = Field(description="The model endpoint to generate trials against.")
    prompt_template: str | dict[str, Any] | None = Field(
        default=None,
        description="How each task maps to the chat/completion request. Defaults to a single user "
        "message carrying the task prompt when omitted.",
    )
    params: RunConfigOnlineModel | None = Field(
        default=None, description="Optional online-inference parameters for trial generation."
    )


class AgentTarget(BaseModel):
    """Generate trials by calling a generic HTTP or NeMo Agent Toolkit target.

    The selected agent variant owns its request and response profile, so there is no
    separate prompt template here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["agent"] = "agent"
    agent: Agent = Field(description="The agent endpoint to generate trials against.")
    params: RunConfigOnline | None = Field(
        default=None, description="Optional online-inference parameters for trial generation."
    )


class FabricRunnerTarget(BaseModel):
    """Generate trials by driving an agent harness through the NeMo Fabric runtime.

    Fabric is harness-agnostic: the harness (Codex, Hermes, ...) is selected by the supplied
    config's ``harness.adapter_id`` and is never inferred from ``model``. ``model`` is applied as the
    config's default model when given.

    A run is described by exactly one complete ``config``. Fabric 0.1.0rc2 removed profile overlays,
    so the former ``profiles`` field is gone — fold any overlay you were passing into ``config``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fabric"] = "fabric"
    config: dict[str, Any] = Field(
        description="Inline NeMo Fabric agent config (an ``agent.yaml`` as a JSON-shaped mapping). Its "
        "``harness.adapter_id`` selects the harness, e.g. ``nvidia.fabric.codex`` for Codex.",
    )
    model: str | None = Field(
        default=None,
        description="Optional ``provider/model`` slug applied as the config's default model; the harness "
        "default is used when omitted.",
    )
    timeout_s: int = Field(default=600, ge=1, description="Per-task timeout for the Fabric run, in seconds.")
    capture_trajectory: bool = Field(
        default=True,
        description="Capture the agent trajectory as ATIF via NeMo Relay and attach it to trial evidence. "
        "Requires the NeMo Relay gateway in the run environment.",
    )


class HarborRunnerTarget(BaseModel):
    """Generate trials by driving a Harbor job through the SDK's :class:`HarborAgentTaskRunner`.

    Runs in *native* mode: Harbor builds and runs its own ``JobConfig`` (executing each task in a
    Docker environment, retrying, and writing a per-trial results tree), then the runtime adapts that
    tree into SDK trials. The dataset Harbor runs against is recovered from each task's
    ``harbor_dataset_path`` metadata, so it is not configured here. The runtime-only jobs directory is
    injected from the job's storage at run time; only the harness-selection and run knobs live here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["harbor"] = "harbor"
    agent_name: str | None = Field(
        default="oracle",
        description="Built-in Harbor agent to run (e.g. 'oracle'). Ignored when `agent_import_path` is set.",
    )
    agent_import_path: str | None = Field(
        default=None,
        description="Custom Harbor agent import path (e.g. 'harbor_wrapper:WrappedAgent'); overrides `agent_name`. "
        "The module must already be importable in the run environment.",
    )
    agent_model_name: str | None = Field(default=None, description="Optional model slug passed to the Harbor agent.")
    agent_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the Harbor agent's constructor, the equivalent of Harbor's "
        "`--ak key=value`. Not for secrets: Harbor persists them unredacted in the job dir's `config.json`; "
        "inject secrets through the environment instead.",
    )
    n_attempts: int = Field(default=1, ge=1, description="Number of attempts Harbor runs per task.")
    n_concurrent_trials: int = Field(default=4, ge=1, description="Maximum concurrent Harbor trials.")
    max_retries: int = Field(default=0, ge=0, description="Harbor per-trial retry attempts on transient failures.")
    artifacts: list[str] = Field(default_factory=list, description="Harbor artifact sources to collect per trial.")
    trace_dir: str | None = Field(
        default=None,
        description="Container path of agent traces to collect as the 'traces' artifact (e.g. '/app/traces').",
    )
    reward_key: str = Field(
        default="reward", description="Key read from Harbor's per-trial rewards mapping to score against."
    )


class GymRunnerTarget(BaseModel):
    """Generate trials by driving a NeMo Gym environment through the SDK's :class:`GymAgentTaskRunner`.

    The deployment chooses colocated execution in the Gym task container or a separate sandboxed
    Gym host. An environment FileSet requires the sandboxed path. The environment dataset is
    recovered from the tasks at run time — ``discover_gym_tasks`` records the source row data needed
    to materialize the selected tasks for rollout collection.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["gym"] = "gym"
    environment: FilesetRef | None = Field(
        default=None,
        description="Environment FileSet containing a native-v1 or wheels-v1 Gym package. "
        "The complete FileSet is staged read-only; file fragments are not supported.",
    )
    agent: str = Field(description="Agent name to collect rollouts with, e.g. 'simple_agent'.")
    agent_config: str | None = Field(
        default=None,
        description="Repo-relative built-in agent config. Required without an environment FileSet; "
        "with a FileSet it is used only when the package does not declare the selected agent instance.",
    )
    resources_server: str = Field(
        description="Resources-server (environment) name, e.g. 'mcqa' (--resources-server).",
    )
    model_type: str = Field(
        default="inference_provider",
        description="Model-type config (--model-type). `inference_provider` speaks OpenAI-compatible chat; "
        "`openai_model` uses the OpenAI Responses API.",
    )
    bind_resources_server: bool = Field(
        default=True,
        description="Auto-bind the agent's `resources_server.name` via a Hydra override. Set False for "
        "self-contained agents that already bind their own resources-server.",
    )
    hydra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters merged into Gym's Hydra config, as nested data — {'model': {'temperature': 0.7}} "
        "rather than pre-serialized Hydra strings — so a spec survives being sent as JSON. Flattened to "
        "Hydra's grammar at invocation, after the auto-derived resources-server binding. Distinct from "
        "`env_vars`: these configure the Gym environment, not the OS environment.",
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables set on the `gym` invocation. Some Gym environments are "
        "configurable only this way — `wmt_translation` reads `WMT_TRANSLATION_COMET_PY_CACHE` for its "
        "model-cache root and defaults to a container-only path — and a job spec has no ambient "
        "environment to inherit from, so whatever the environment needs has to travel in the spec.",
    )
    agent_ref_name: str | None = Field(
        default=None,
        description="Gym agent instance rollouts are routed to when running against a sandboxed host, "
        "stamped as each row's `agent_ref`. Defaults to `agent`. Set it when the environment's config "
        "defines the agent under a different name -- `mcqa` registers `mcqa_simple_agent`, and routing to "
        "`simple_agent` there does not answer.",
    )
    env_secrets: dict[str, SecretRef] = Field(
        default_factory=dict,
        description="Environment variables sourced from the secrets service, as {ENV_NAME: secret-ref}. "
        "The reference travels in the spec; the service resolves it into the job's environment at "
        "compile time, so no credential is stored on the spec or written to a run bundle. Use this "
        "rather than `env_vars` for anything secret -- a Gym model API key belongs here.",
    )
    num_repeats: int = Field(default=1, ge=1, description="Attempts per row; each attempt becomes one trial.")
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent rollouts for `gym eval run`.",
    )
    startup_timeout_s: float = Field(default=240.0, gt=0, description="Max wait for `gym env start` readiness.")
    collection_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Max wait for `gym eval run` collection; None = unbounded.",
    )
    shutdown_grace_s: float = Field(
        default=30.0,
        gt=0,
        description="Grace period for the Gym subprocess group to exit on SIGTERM before escalating to SIGKILL.",
    )
    reward_key: str = Field(default="reward", description="Key read from each rollout record.")

    @field_validator("environment")
    @classmethod
    def _require_whole_environment_fileset(cls, value: FilesetRef | None) -> FilesetRef | None:
        if value is None:
            return None
        try:
            _, _, file_path = parse_fileset_ref(value.root, workspace_fallback="_validation")
        except FilesetPathError as exc:
            raise ValueError(f"invalid environment FileSet reference: {value.root!r}") from exc
        if file_path:
            raise ValueError("environment FileSet references must not include a file fragment")
        return value

    @model_validator(mode="after")
    def _require_builtin_agent_config_without_environment(self) -> Self:
        if self.environment is None and self.agent_config is None:
            raise ValueError("The agent_config field is required when no environment FileSet is supplied")
        return self


#: The agent-runner slot of the target union — the spec-side mirror of ``AgentTaskRunner``, resolved
#: to a runtime at run time. ``kind``-discriminated; widen with more members as runners land.
AgentRunnerTarget: TypeAlias = FabricRunnerTarget | GymRunnerTarget | HarborRunnerTarget

#: What generates trials: a Model or Agent endpoint, or an agent runner. ``kind``-discriminated, and
#: the spec-level analog of the SDK's runtime ``AgentEvalTarget`` (Model | Agent | AgentTaskRunner).
Target: TypeAlias = ModelTarget | AgentTarget | AgentRunnerTarget


def target_agent_identity(target: Target | Model | AgentBase | None) -> tuple[str | None, str | None]:
    """``(agent_name, model_name)`` derivable from a target, for publishing to Intake.

    Only targets that carry a real name yield one — nothing here invents an identity, because a
    made-up agent name is worse than an explicit one the submitter had to supply. A ``ModelTarget``
    has a model but no agent; the runners other than Harbor name a harness, not an agent. Those
    cases return ``None`` and the spec must carry ``publication.intake.agent_name``.

    Accepts both unions: agent-eval passes its ``Target`` spec wrappers, while the dataset-driven
    eval's ``TargetSpec`` is the bare ``Model``/``Agent`` SDK value. Without the bare branches a row
    target falls through to ``(None, None)`` and publishes under an empty agent name.

    Distinct from ``result_persistence._agent_target_fields``, which flattens the same targets to
    ``(kind, name, url)`` filter traits and folds runner *models* into its ``name`` slot.
    """
    if isinstance(target, AgentTarget):
        return target.agent.name, None
    if isinstance(target, HarborRunnerTarget):
        return target.agent_import_path or target.agent_name, target.agent_model_name
    if isinstance(target, GymRunnerTarget):
        return target.agent, None
    if isinstance(target, ModelTarget):
        return None, target.model.name
    if isinstance(target, FabricRunnerTarget):
        return None, target.model
    # Bare SDK values, as carried by the dataset-driven eval spec.
    if isinstance(target, AgentBase):
        return target.name, None
    if isinstance(target, Model):
        return None, target.name
    return None, None


class _AgentEvalTaskCommon(BaseModel):
    """Fields shared by the submitter and canonical task DTOs (everything but ``metrics``).

    ``metrics`` differs between the two (refs allowed vs. fully resolved), so — as
    with ``EvaluateInputSpec``/``EvaluateSpec`` — the variants are siblings that add
    it, not a subtype pair (a mutable field can't be narrowed across inheritance).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable task identifier, unique within the task collection.")
    intent: str = Field(description="Human-readable description of the desired agent behavior.")
    inputs: TaskInputs = Field(default_factory=TaskInputs, description="Inputs supplied to the task.")
    reference: dict[str, Any] = Field(
        default_factory=dict,
        description="Grader-only ground truth (held-out tests, expected outputs, rubric data). Surfaced to "
        "metrics but never seeded into the agent's workspace or shown to the agent, so a metric can grade "
        "against artifacts the agent cannot influence.",
    )
    views: dict[str, SemanticView] = Field(
        default_factory=dict,
        description="Optional reporting views mapping this task's metric outputs into named semantic scores.",
    )
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the task.")


class AgentEvalTaskInput(_AgentEvalTaskCommon):
    """Submitter-facing task DTO: metrics may be inline bundles or stored-metric references."""

    metrics: list[MetricRefOrInline] = Field(
        default_factory=list,
        description="Metrics that score this task, inline and/or references to stored metrics.",
    )


class AgentEvalTaskSpec(_AgentEvalTaskCommon):
    """Canonical task DTO: metrics fully resolved to inline bundles, reconstructed at run time."""

    metrics: list[MetricInline] = Field(
        default_factory=list,
        description="Inline metric bundles that score this task; reconstructed to runtime metrics at run time.",
    )


class _AgentEvalSpecCommon(BaseModel):
    """Fields shared by the submitter input and canonical agent-eval specs (everything but ``tasks``)."""

    # ``oneOf`` mirrors the ``_require_exactly_one_trial_source`` validator into the OpenAPI schema, so
    # the generated contract (and clients) reject a target-less or both-supplied request instead of
    # only discovering it via a 422 at runtime. Each branch also excludes an explicit ``null`` (the
    # validator keys off non-null, not mere presence), so a request that sends ``"target": null``
    # alongside ``trials`` is accepted by the schema exactly as the runtime accepts it.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {"required": ["target"], "properties": {"target": {"not": {"type": "null"}}}},
                {"required": ["trials"], "properties": {"trials": {"not": {"type": "null"}}}},
            ]
        },
    )

    target: Target | None = Field(
        default=None,
        description="What generates trials online: a Model or Agent endpoint, or an agent runner (e.g. a "
        "Fabric harness). Endpoint targets carry their own request config (prompt template / inference "
        "params). Mutually exclusive with `trials`.",
    )
    trials: list[AgentEvalTrial] | None = Field(
        default=None,
        description="Precomputed trials to score directly (offline eval), instead of generating them from a "
        "`target`. Mutually exclusive with `target`.",
    )
    max_concurrent_tasks: int = Field(
        default=4,
        ge=1,
        description="Maximum number of tasks evaluated concurrently. Distinct from a target's "
        "`params.parallelism`, which bounds concurrent inference requests *within* trial generation.",
    )
    fail_fast: bool = Field(default=False, description="Stop the run on the first scoring failure when True.")
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Caller-supplied tags recorded on the run's metadata (e.g. benchmark, mode, backend).",
    )
    publication: PublicationSpec | None = Field(
        default=None,
        description="Where the completed run publishes its results, beyond its own result bundle. "
        "Omit to publish nowhere.",
    )

    @model_validator(mode="after")
    def _require_resolvable_publication_identity(self) -> Self:
        # Publishing needs an agent name, and only some targets carry one. Rejecting here makes it a
        # 422 on submit rather than a failure discovered after the evaluation has already run.
        intake = self.publication.intake if self.publication is not None else None
        if intake is None or intake.agent_name is not None:
            return self
        if target_agent_identity(self.target)[0] is None:
            source = "the precomputed `trials`" if self.target is None else f"a `{self.target.kind}` target"
            raise ValueError(
                f"`publication.intake.agent_name` is required: it cannot be derived from {source}. "
                "Supply the name the published trajectories should be recorded under."
            )
        return self

    @model_validator(mode="after")
    def _require_exactly_one_trial_source(self) -> Self:
        # The SDK evaluator requires exactly one of trials/target (one generates trials online, the
        # other scores precomputed ones); enforce it at the spec boundary so a target-less or
        # both-supplied spec is rejected at validation rather than failing inside the run.
        if (self.target is None) == (self.trials is None):
            raise ValueError(
                "provide exactly one of `target` (generate trials online) or `trials` (score precomputed trials)"
            )
        return self


class AgentEvalInputSpec(_AgentEvalSpecCommon):
    """Submitter-facing agent-evaluation input.

    ``tasks`` is either an inline list of tasks (whose metrics may be inline or references) or a
    :class:`TasksetRef` naming a stored taskset whose member tasks are loaded and expanded during spec
    resolution. Either way it hydrates to the canonical ``AgentEvalSpec.tasks`` list.
    """

    tasks: TasksetRef | list[AgentEvalTaskInput] = Field(
        description="Tasks to evaluate: an inline list (at least one) or a reference to a stored taskset.",
    )

    @model_validator(mode="after")
    def _reject_empty_inline_tasks(self) -> Self:
        # A TasksetRef is validated (and required non-empty) when it is expanded during resolution; an
        # inline list must carry at least one task, mirroring the canonical spec's ``min_length=1``.
        if isinstance(self.tasks, list) and not self.tasks:
            raise ValueError("provide at least one task, or a `tasks` taskset reference")
        return self


class AgentEvalSpec(_AgentEvalSpecCommon):
    """Canonical agent-evaluation spec: tasks with all metric references resolved to inline."""

    tasks: list[AgentEvalTaskSpec] = Field(min_length=1, description="Tasks to evaluate; at least one is required.")

    @model_validator(mode="after")
    def _reject_unresolved_metric_model_refs(self) -> Self:
        for task in self.tasks:
            unresolved = unresolved_model_refs([unbundle_metric(to_runtime_bundle(metric)) for metric in task.metrics])
            if unresolved:
                raise ValueError(
                    f"AgentEvalSpec task {task.id!r} metric models must be resolved before run: "
                    + ", ".join(unresolved)
                )
        return self
