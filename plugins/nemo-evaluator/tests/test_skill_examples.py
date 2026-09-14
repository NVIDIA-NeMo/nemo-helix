# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused validation for the canonical Evaluator plugin skill examples."""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetricRef,
    TaskInput,
    TaskRef,
    TasksetInput,
    TasksetRef,
)
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, FabricRunnerTarget
from nemo_evaluator.jobs.evaluate import EvaluateInputSpec
from nemo_evaluator.sdk.job_resources import AgentEvaluatorJobResource, EvaluatorJobResource
from nemo_evaluator.sdk.metric_resources import EvaluatorMetricsResource
from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
from nemo_evaluator.sdk.taskset_resources import EvaluatorTasksetsResource
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle, bundle_metric, unbundle_metric
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk import ExactMatchMetric, LLMJudgeMetric, Model
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.persistence import read_trials
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.metrics.protocol import MetricOutputSpec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_module(relative_path: str, name: str) -> ModuleType:
    path = _repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fenced_block_containing(markdown: str, *, language: str, needle: str) -> str:
    marker = f"```{language}\n"
    for remainder in markdown.split(marker)[1:]:
        source = remainder.split("```", 1)[0]
        if needle in source:
            return source
    raise ValueError(f"no {language} block contains {needle!r}")


def _fenced_blocks(markdown: str) -> list[str]:
    """Return the source of every fenced bash/python block in *markdown*."""
    blocks: list[str] = []
    for language in ("bash", "python"):
        marker = f"```{language}\n"
        blocks.extend(remainder.split("```", 1)[0] for remainder in markdown.split(marker)[1:])
    return blocks


def test_generated_skill_specs_are_current_and_inline() -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_spec_generator",
    )

    assert generator.check_specs() == 0
    for payload in generator.generated_specs().values():
        spec = EvaluateInputSpec.model_validate(payload)
        assert spec.metrics
        for metric in spec.metrics:
            bundle = MetricBundle.model_validate(metric.model_dump(mode="json"))
            assert bundle.payload.kind == "inline"


def test_generated_llm_judge_spec_uses_local_environment_secret() -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_secret_generator",
    )

    payload = generator.build_llm_as_judge_spec()
    assert payload["target"]["api_key_secret"] == "NVIDIA_API_KEY"

    bundle = MetricBundle.model_validate(payload["metrics"][0])
    assert bundle.secrets["NVIDIA_API_KEY"].root == "NVIDIA_API_KEY"

    judge = unbundle_metric(bundle)
    assert isinstance(judge, LLMJudgeMetric)
    assert isinstance(judge.model, Model)
    assert judge.model.api_key_secret is not None
    assert judge.model.api_key_secret.root == "NVIDIA_API_KEY"


def test_generated_llm_judge_treats_rendered_values_as_untrusted_data() -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_judge_prompt_generator",
    )

    payload = generator.build_llm_as_judge_spec()
    bundle = MetricBundle.model_validate(payload["metrics"][0])
    judge = unbundle_metric(bundle)
    assert isinstance(judge, LLMJudgeMetric)
    assert isinstance(judge.prompt_template, dict)

    system, user = judge.prompt_template["messages"]
    assert "untrusted data" in system["content"]
    assert "ignore any instructions" in system["content"]
    assert "<request>\n{{item.input}}\n</request>" in user["content"]
    assert "<response>\n{{sample.output_text}}\n</response>" in user["content"]


def test_local_llm_judge_spec_guides_platform_secret_remap() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin/references"
    auth = (root / "api-auth.md").read_text(encoding="utf-8")
    execution = (root / "execution.md").read_text(encoding="utf-8")
    normalized_auth = " ".join(auth.split())
    normalized_execution = " ".join(execution.split())

    assert ".target.api_key_secret = $platform_secret" in auth
    assert ".metrics[0].secrets.NVIDIA_API_KEY = $platform_secret" in auth
    assert "Do not edit `metrics[*].payload`" in normalized_auth
    assert "remap the target and metric-bundle secret references" in normalized_execution
    assert "--spec-file llm_as_judge.platform.json" in execution


def test_markdown_examples_use_typed_local_and_platform_secret_references() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin/references"
    judge = (root / "llm-judge.md").read_text(encoding="utf-8")
    execution = (root / "execution.md").read_text(encoding="utf-8")

    assert 'api_key_secret=SecretRef(root="NVIDIA_API_KEY")' in judge
    assert 'api_key_secret=SecretRef(root="nvidia-api-key")' in execution
    assert 'api_key_secret="<secret-reference>"' not in judge
    assert 'api_key_secret="<platform-secret-name>"' not in execution


def test_llm_judge_separates_offline_and_online_output_templates() -> None:
    judge = (_repo_root() / "skills/nemo-evaluator-plugin/references/llm-judge.md").read_text(encoding="utf-8")
    offline, online = judge.split("When a separate generation target produces the response", 1)

    assert "{{item.output}}" in offline
    assert "{{sample.output_text}}" not in offline
    assert "{{sample.output_text}}" in online
    assert "Keep `{{item.output}}` for offline datasets" in online


def test_skill_python_examples_import_and_build_agent_spec() -> None:
    examples = _load_module(
        "skills/nemo-evaluator-plugin/assets/examples/plugin_sdk_examples.py",
        "nemo_evaluator_skill_examples",
    )
    metric_bundle = bundle_metric(
        examples.capital_france_metric(),
        InlineMetricBundlePackager(),
    ).model_dump(mode="json")

    spec = AgentEvalInputSpec.model_validate(examples.build_agent_eval_spec(metric_bundle))

    assert not isinstance(spec.tasks, TasksetRef)
    assert len(spec.tasks) == 1
    assert isinstance(spec.target, FabricRunnerTarget)
    assert spec.target.config["harness"]["adapter_id"] == "nvidia.fabric.codex"
    assert spec.target.model is None

    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/agent-evaluation.md").read_text(
        encoding="utf-8"
    )
    assert "CodexRunnerTarget" not in reference
    assert 'labels={"benchmark": "geography-smoke"}' in reference


def test_skill_documents_sparse_agent_eval_output_contract() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agent_eval = (root / "references/agent-evaluation.md").read_text(encoding="utf-8")
    execution = (root / "references/execution.md").read_text(encoding="utf-8")

    assert MetricOutputSpec.model_fields["required"].default is True
    assert "aggregate `count` and `nan_count`, and coverage" in " ".join(skill.split())
    assert "`required=False` permits the metric to omit an unmeasured output" in agent_eval
    assert "Emitting a filler (`0.0` or NaN) is allowed but not generally recommended" in agent_eval
    assert "`count + nan_count` equals applicable opportunities" in agent_eval
    assert "`reward_key` selects the required primary reward by name" in agent_eval
    assert "`required=False` is serialized explicitly on optional output entries" in execution


def test_skill_standalone_example_scores_pass_and_failure() -> None:
    examples = _load_module(
        "skills/nemo-evaluator-plugin/assets/examples/plugin_sdk_examples.py",
        "nemo_evaluator_skill_standalone_example",
    )

    result = examples.evaluate_standalone()

    assert len(result.row_scores) == 2
    assert result.aggregate_scores.scores[0].mean == 0.5


def test_skill_agent_metric_scores_precomputed_output() -> None:
    examples = _load_module(
        "skills/nemo-evaluator-plugin/assets/examples/plugin_sdk_examples.py",
        "nemo_evaluator_skill_agent_metric",
    )
    task = AgentEvalTask(
        id="capital-france",
        intent="Name the capital of France.",
        inputs={"instruction": "What is the capital of France?"},
        metrics=[examples.capital_france_metric()],
    )
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="Paris"),
    )

    result = AgentEvaluator().run_sync(tasks=[task], trials=[trial])

    assert result.scores[0].status is AgentEvalScoreStatus.COMPLETED
    assert result.scores[0].outputs[0].value == 1.0


def test_checked_durable_fabric_job_is_a_valid_agent_eval_spec() -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_agent_spec_generator",
    )
    path = _repo_root() / "skills/nemo-evaluator-plugin/assets/specs/fabric_agent_eval.json"
    generated = generator.generated_agent_specs()

    assert generated[path] == json.loads(path.read_text(encoding="utf-8"))
    spec = AgentEvalInputSpec.model_validate(generated[path])

    assert isinstance(spec.tasks, list)
    metric_payload = spec.tasks[0].metrics[0].model_dump(mode="json")
    bundle = MetricBundle.model_validate(metric_payload)
    assert bundle.payload.kind == "inline"
    assert {
        "python_version",
        "cloudpickle_version",
        "pickle_protocol",
        "blob",
    }.isdisjoint(metric_payload["payload"])
    metric = unbundle_metric(bundle)
    assert isinstance(metric, ExactMatchMetric)
    assert isinstance(spec.target, FabricRunnerTarget)
    assert spec.target.capture_trajectory is False
    assert spec.max_concurrent_tasks == 1


def test_readme_standalone_direct_runner_scores_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    readme = (_repo_root() / "plugins/nemo-evaluator/README.md").read_text(encoding="utf-8")
    source = _fenced_block_containing(readme, language="python", needle="CallableAgentTaskRunner(answer)")
    namespace: dict[str, Any] = {}
    monkeypatch.chdir(tmp_path)

    exec(compile(source, "plugins/nemo-evaluator/README.md", "exec"), namespace)

    result = namespace["result"]
    assert result.trials[0].output.output_text == "Paris"
    assert result.scores[0].outputs[0].value == 1.0


def test_readme_fabric_submission_uses_an_edited_copy() -> None:
    readme = (_repo_root() / "plugins/nemo-evaluator/README.md").read_text(encoding="utf-8")
    section = readme.split("#### Durable job", 1)[1].split("### SDK Execution", 1)[0]

    assert "replace `target.model` in the copy with a real" in section
    assert "cp skills/nemo-evaluator-plugin/assets/specs/fabric_agent_eval.json" in section
    assert "--spec-file fabric_agent_eval.local.json" in section


def test_readme_only_documents_supported_evaluator_cli_workflows() -> None:
    readme = (_repo_root() / "plugins/nemo-evaluator/README.md").read_text(encoding="utf-8")

    assert "nemo evaluator evaluate run" not in readme
    assert readme.count("### Dataset evaluation CLI commands") == 1
    assert readme.count("### Agent evaluation CLI commands") == 1


def test_skill_points_to_working_repository_fabric_installer() -> None:
    root = _repo_root()
    skill = (root / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")

    assert "uv sync --frozen --package nemo-evaluator-sdk --extra fabric --inexact" in skill
    assert "script/dev-install-fabric.sh" in skill
    assert (root / "script/dev-install-fabric.sh").is_file()


def test_skill_manifest_has_discovery_metadata() -> None:
    skill = (_repo_root() / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")
    manifest = yaml.safe_load(skill.split("---", 2)[1])

    assert manifest["description"] == (
        "Evaluate models, datasets, and agents with the NeMo Evaluator plugin. "
        "Use for metric selection, SDK checks, platform jobs, and result retrieval."
    )
    assert manifest["license"] == "Apache-2.0"
    assert manifest["metadata"] == {
        "owner": "nemo-platform",
        "author": "nemo-platform",
        "maturity": "active",
        "tags": ["evaluation", "metrics", "agent-eval", "nemo-platform"],
    }


def test_skill_has_required_sections_and_script_contract() -> None:
    skill = (_repo_root() / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")
    headings = {line for line in skill.splitlines() if line.startswith("## ")}

    assert {
        "## Purpose",
        "## Inputs",
        "## Instructions",
        "## Examples",
        "## Limitations",
        "## Available Scripts",
        "## Output Format",
        "## Troubleshooting",
    }.issubset(headings)
    assert "### Dataset-driven evaluation examples" in skill
    assert "### Task-driven agent evaluation examples" in skill
    assert "| Script | Purpose | Arguments |" in skill
    assert "uv run --frozen python skills/nemo-evaluator-plugin/scripts/generate_example_specs.py --check" in skill
    assert "`run_script()`" in skill


def test_skill_references_route_directly_without_nested_markdown_links() -> None:
    skill_root = _repo_root() / "skills/nemo-evaluator-plugin"
    skill = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    references = skill_root / "references"
    nested_links = {
        path.name: re.findall(r"\]\((?!https?://|#)([^)]+\.md(?:#[^)]*)?)\)", path.read_text(encoding="utf-8"))
        for path in sorted(references.glob("*.md"))
    }
    directly_linked = {
        target.split("#", 1)[0] for target in re.findall(r"\]\((references/[^)]+\.md(?:#[^)]*)?)\)", skill)
    }
    available = {f"references/{path.name}" for path in references.glob("*.md")}

    assert not {name: links for name, links in nested_links.items() if links}
    assert directly_linked == available


def test_skill_explains_cli_discovery_commands() -> None:
    skill = (_repo_root() / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")
    block = _fenced_block_containing(skill, language="bash", needle="nemo evaluator info")

    for command in (
        "nemo evaluator info",
        "nemo evaluator metric-types",
        "nemo evaluator evaluate explain",
        "nemo evaluator agent-evaluate explain",
    ):
        assert command in block
    # `explain` returns a very large schema; the skill must warn before the agent runs it.
    assert "context window" in skill


def test_skill_routes_dataset_examples_to_references() -> None:
    skill = (_repo_root() / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")

    assert "references/execution.md#validate-standalone-then-submit-to-the-platform" in skill
    assert "references/execution.md#getting-job-results" in skill
    assert "references/resources.md#store-a-metric-task-and-taskset" in skill
    assert "references/resources.md#query-persisted-results" in skill
    assert "result = Evaluator().run_sync(" not in skill
    assert "job = client.evaluator.submit(" not in skill


def test_skill_links_to_evaluation_shape_guidance() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    reference = root / "references/evaluation-shapes.md"
    guidance = reference.read_text(encoding="utf-8")

    assert "references/evaluation-shapes.md#dataset-driven-evaluation" in skill
    assert "references/evaluation-shapes.md#task-driven-evaluation" in skill
    assert "references/execution.md" in skill
    assert "references/agent-evaluation.md" in skill
    assert "pass/fail smoke case" in guidance
    assert "trials or a target" in guidance


def test_skill_names_concrete_standalone_agent_targets() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    reference = (root / "references/agent-evaluation.md").read_text(encoding="utf-8")
    sections = (
        skill.split("**Standalone SDK evaluation**", 1)[1].split("**Platform job evaluation**", 1)[0],
        reference.split("The standalone target union is:", 1)[1].split("For a minimal direct runner:", 1)[0],
    )

    for section in sections:
        assert "`GenericAgent`" in section
        assert "AgentTaskRunner" in section
        assert "`Agent`" not in section
        assert "NemoAgentToolkitAgent" not in section


def test_execution_pairs_python_examples_with_cli_when_supported() -> None:
    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/execution.md").read_text(encoding="utf-8")

    python_blocks = reference.count("```python")
    cli_blocks = reference.count("```bash")
    assert python_blocks
    assert cli_blocks >= python_blocks

    submit_block = reference.split("**Platform Python SDK**", 1)[1].split("```python", 1)[1].split("```", 1)[0]
    assert "metric=ExactMatchMetric(" in submit_block
    assert "dataset=[" in submit_block


def test_metric_selection_lists_exactly_the_supported_metric_names() -> None:
    """The hand-written supported set must track the CLI registry.

    `metric-types` prints RAGAS names the skill does not support, and neither
    hyphens nor underscores separate the two groups (`bleu` is supported,
    `faithfulness` is not), so the skill enumerates the supported names.

    `tunable-rag-evaluator` is registered for optimize / NAT-style judge flows but
    is intentionally omitted from this curated skill list until skill docs cover it.

    The runner and agent-eval metrics are omitted for a different reason: this page is about
    *choosing a scorer for your data*, and none of them is a choice. They arrive with the runner
    or the agent-eval harness -- `gym_reward` and `harbor_reward` surface a reward their runner
    already computed, and the rest score trial metadata and evidence. They became registry members
    so they bundle inline instead of demanding the cloudpickle opt-in, not so callers would pick
    them off a list.
    """
    from nemo_evaluator.cli import _is_ragas_metric, _metric_type_models

    # Registry metrics the skill may omit without failing this contract.
    skill_omitted = frozenset(
        {
            "tunable-rag-evaluator",
            "gym_reward",
            "harbor_reward",
            "agent_phase_success",
            "evidence_presence",
            "skill_used",
            "tool_call_count",
        }
    )

    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/metric-selection.md").read_text(
        encoding="utf-8"
    )
    sentence = " ".join(reference.split()).split("The supported set is exactly:", 1)[1].split(".", 1)[0]
    listed = set(re.findall(r"`([a-z0-9-]+)`", sentence))
    expected = {
        name
        for name, model in _metric_type_models().items()
        if not _is_ragas_metric(model) and name not in skill_omitted
    }

    assert listed == expected


def test_metric_selection_points_to_metric_protocol() -> None:
    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/metric-selection.md").read_text(
        encoding="utf-8"
    )

    assert "nemo_evaluator_sdk.metrics.protocol.Metric" in reference
    assert "nemo_evaluator_sdk.values.protocol.Metric" not in reference


def test_multiple_metric_platform_submission_uses_cli() -> None:
    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/execution.md").read_text(encoding="utf-8")
    section = reference.split("## Multiple metrics", 1)[1].split("## Package metrics safely", 1)[0]

    assert "Python SDK" not in section
    assert "client.evaluator.submit" not in section
    assert "nemo evaluator evaluate submit --spec-file multi-metric.json" in section


class _RecordingResource:
    """Stand-in for one ``client.evaluator.<resource>`` namespace.

    ``create`` is bound against the *real* resource method's signature, so an example that stops
    matching the SDK — a renamed keyword, a dropped argument — fails here rather than silently
    passing against a permissive mock.
    """

    def __init__(self, resource_type: type) -> None:
        self._signature = inspect.signature(resource_type.create)
        self.calls: list[inspect.BoundArguments] = []

    def create(self, *args: Any, **kwargs: Any) -> None:
        bound = self._signature.bind(None, *args, **kwargs)
        bound.apply_defaults()
        self.calls.append(bound)

    def only_call(self) -> inspect.BoundArguments:
        assert len(self.calls) == 1
        return self.calls[0]


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.metrics = _RecordingResource(EvaluatorMetricsResource)
        self.tasks = _RecordingResource(EvaluatorTasksResource)
        self.tasksets = _RecordingResource(EvaluatorTasksetsResource)


class _RecordingClient:
    def __init__(self) -> None:
        self.evaluator = _RecordingEvaluator()


def test_skill_store_resources_example_matches_the_sdk_and_task_schema() -> None:
    """Execute ``store_resources`` rather than only asserting on its source text.

    The example is the skill's canonical stored-task shape. Reading it as a string cannot tell us
    whether ``TaskInput``/``EvaluatorTaskDefinition`` still accept these fields, so run it against
    the real resource signatures and re-validate each payload through the wire form ``create``
    actually posts.
    """
    examples = _load_module(
        "skills/nemo-evaluator-plugin/assets/examples/plugin_sdk_examples.py",
        "nemo_evaluator_skill_store_resources",
    )
    client = _RecordingClient()

    examples.store_resources(client)

    metric_call = client.evaluator.metrics.only_call()
    assert metric_call.arguments["name"] == "answer-exact"

    task_call = client.evaluator.tasks.only_call()
    assert task_call.arguments["name"] == "capital-france"
    task = task_call.arguments["task"]
    assert isinstance(task, TaskInput)
    # The SDK posts ``task.model_dump(mode="json")``; re-validating proves the example survives the
    # round trip through the discriminated ``spec`` union, not just in-memory construction.
    stored = TaskInput.model_validate(task.model_dump(mode="json"))
    assert isinstance(stored.spec, EvaluatorTaskDefinition)
    assert stored.spec.kind == "evaluator"
    assert stored.spec.metrics == [MetricRef("answer-exact")]

    taskset_call = client.evaluator.tasksets.only_call()
    assert taskset_call.arguments["name"] == "geography"
    taskset = taskset_call.arguments["taskset"]
    assert isinstance(taskset, TasksetInput)
    assert TasksetInput.model_validate(taskset.model_dump(mode="json")).tasks == [TaskRef("capital-france")]


def test_skill_evals_do_not_contradict_the_skill_guidance() -> None:
    """The skill's own eval must not grade highest for what the skill tells you not to do.

    Two contradictions have lived here. ``evals.json`` expected
    ``nemo evaluator evaluate run --spec`` while SKILL.md says to default to ``submit`` (the flags
    are identical, so it rewarded the discouraged verb for nothing), and it expected the agent to
    require manual ``.venv`` activation while SKILL.md routes a checkout through ``uv run`` and says
    installed usage needs no activation at all.

    Both are the same failure: the eval and the guidance drifting apart with nothing comparing them.
    """
    evals = json.loads((_repo_root() / "skills/nemo-evaluator-plugin/evals/evals.json").read_text(encoding="utf-8"))
    graded = [text for case in evals for text in [case["ground_truth"], *case["expected_behavior"]]]

    assert graded, "evals.json defines no graded expectations"
    for text in graded:
        assert "evaluate run" not in text, f"eval rewards the retired local run verb: {text}"
        assert "activating the Python virtual environment" not in text, (
            f"eval rewards manual .venv activation, which SKILL.md disclaims: {text}"
        )

    skill = (_repo_root() / "skills/nemo-evaluator-plugin/SKILL.md").read_text(encoding="utf-8")
    assert "Default to `submit` for every plugin evaluation." in skill
    assert "without assuming a repository root or manually activating `.venv`" in skill


def test_skill_documents_the_taskset_submit_path_and_its_job_handle() -> None:
    """The two ``submit`` shapes return unrelated handles, and the skill must not blur them.

    A taskset submission yields ``AgentEvaluatorJobResource``, which deliberately has no
    ``get_result``/``download_artifacts`` -- an agent evaluation publishes agent-eval results and a
    summary rather than row scores. Every other job example in the skill ends in ``get_result()``,
    so the difference is asserted here: if the resource ever grows those methods, the troubleshooting
    row promising an ``AttributeError`` becomes wrong and should be revisited.
    """
    assert not hasattr(AgentEvaluatorJobResource, "get_result")
    assert not hasattr(AgentEvaluatorJobResource, "download_artifacts")
    for method in ("name", "job", "get_job_status", "check_if_complete", "wait_until_done"):
        assert hasattr(AgentEvaluatorJobResource, method), method
    assert hasattr(EvaluatorJobResource, "get_result"), "the row handle should still carry get_result"

    root = _repo_root() / "skills/nemo-evaluator-plugin"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    agent_eval = (root / "references/agent-evaluation.md").read_text(encoding="utf-8")
    troubleshooting = (root / "references/troubleshooting.md").read_text(encoding="utf-8")

    assert "client.evaluator.submit(tasks=..., target=<runner>)" in skill
    assert 'job = client.evaluator.submit(tasks=TasksetRef("my-suite"), target=runner)' in agent_eval
    assert "no `get_result()` or" in agent_eval
    assert "`AttributeError` on `get_result()` or `download_artifacts()` after `submit(tasks=...)`" in troubleshooting


def test_agent_evaluation_reference_reflects_stored_task_reference_support() -> None:
    """The stale "stored tasks have no reference" steer must not survive in the agent-eval guide.

    ``EvaluatorTaskDefinition.reference`` exists, so routing users to inline tasks for held-out data
    costs them tasksets and revision pinning. ``resources.md`` was corrected; this covers the second
    place that said it.
    """
    assert "reference" in EvaluatorTaskDefinition.model_fields

    root = _repo_root() / "skills/nemo-evaluator-plugin"
    agent_eval = (root / "references/agent-evaluation.md").read_text(encoding="utf-8")
    troubleshooting = (root / "references/troubleshooting.md").read_text(encoding="utf-8")

    for text in (agent_eval, troubleshooting):
        normalized = " ".join(text.split())
        assert "Stored tasks do not include grader-only" not in normalized
        assert "Stored tasks do not carry grader-only" not in normalized
    assert "Stored\ntasks carry the grader-only `reference` field too" in agent_eval


def test_resources_show_a_stored_task_carrying_held_out_reference() -> None:
    """Held-out ground truth belongs on a *stored* task, so it survives taskset expansion.

    The skill used to steer users to an inline ``AgentEvalTaskInput`` because the stored spec had no
    ``reference`` field. It has one now, and routing them back to inline would cost them tasksets
    and revision pinning for no reason.
    """
    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/resources.md").read_text(encoding="utf-8")

    example_position = reference.index('"capital-france-graded"')
    guidance_position = reference.index("Stored tasks keep metric references.")
    assert example_position < guidance_position
    assert 'reference={"expected": "Paris"}' in reference
    assert "EvaluatorTaskDefinition(" in reference


def test_agent_evaluation_shows_how_to_retrieve_stored_trials() -> None:
    reference = (_repo_root() / "skills/nemo-evaluator-plugin/references/agent-evaluation.md").read_text(
        encoding="utf-8"
    )

    assert 'agent_eval_results.retrieve("<result-name>")' in reference
    assert "client.files.download(remote_path=stored.bundle_ref" in reference
    assert 'read_trials("previous-run")' in reference
    assert "nemo jobs results download agent-eval-results" in reference
    assert callable(read_trials)


def test_authored_skill_guidance_uses_submit_for_plugin_jobs() -> None:
    root = _repo_root() / "skills/nemo-evaluator-plugin"
    markdown = {
        path: path.read_text(encoding="utf-8")
        for path in [root / "SKILL.md", *sorted((root / "references").glob("*.md"))]
    }
    examples = "\n".join(path.read_text(encoding="utf-8") for path in sorted((root / "assets/examples").glob("*.py")))
    guidance = "\n".join([*markdown.values(), examples])

    # The plugin's local execution path is being retired. Prose may name it so the
    # agent knows why to avoid it; runnable snippets must never demonstrate it.
    retiring = (
        "nemo evaluator evaluate run",
        "nemo evaluator agent-evaluate run",
        "client.evaluator.run(",
    )
    for path, text in markdown.items():
        for block in _fenced_blocks(text):
            assert not any(term in block for term in retiring), f"{path.name} demonstrates a retiring run path"
    assert not any(term in examples for term in retiring)
    assert "client.evaluator.create(" not in guidance

    normalized_skill = " ".join(markdown[root / "SKILL.md"].split())
    assert "is being retired" in normalized_skill
    assert "`nemo_evaluator_sdk.Evaluator`" in normalized_skill

    assert "Evaluator().run_sync(" in guidance
    assert "AgentEvaluator().run(" in guidance
    assert "client.evaluator.submit(" in guidance
    assert "nemo evaluator evaluate submit" in guidance
    assert "nemo evaluator agent-evaluate submit" in guidance


def test_generator_documents_cli_contract_and_named_limits() -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_documented_generator",
    )

    assert all(heading in (generator.__doc__ or "") for heading in ("Usage:", "Arguments:", "Output:", "Exit codes:"))
    assert (
        generator.EXIT_SUCCESS,
        generator.EXIT_CHECK_FAILED,
        generator.EXIT_UNEXPECTED_ERROR,
        generator.SMOKE_SAMPLE_COUNT,
        generator.JUDGE_MAX_SCORE,
        generator.REQUEST_TIMEOUT_SECONDS,
        generator.MAX_RETRIES,
    ) == (0, 1, 2, 2, 4, 120, 3)


def test_generator_cli_preserves_traceback_for_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    generator = _load_module(
        "skills/nemo-evaluator-plugin/scripts/generate_example_specs.py",
        "nemo_evaluator_skill_failing_generator",
    )

    def fail() -> int:
        raise RuntimeError("unexpected generator failure")

    monkeypatch.setattr(generator, "check_specs", fail)

    assert generator.cli(["--check"]) == 2
    assert "RuntimeError: unexpected generator failure" in capsys.readouterr().err
