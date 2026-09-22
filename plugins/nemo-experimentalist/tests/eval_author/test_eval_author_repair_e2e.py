# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-model canaries for Eval Author-authored Harbor verifiers."""

import asyncio
import json
import os
import shlex
from pathlib import Path

import pytest
import pytest_asyncio
from nemo_experimentalist_plugin.client import make_client
from nemo_experimentalist_plugin.entities import DatasetValidationError, local_path_from_uri
from nemo_experimentalist_plugin.eval_author.agent import EvalAuthor
from nemo_experimentalist_plugin.eval_author.models import EvalAuthorConfig
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor_native import (
    HarborDataset,
    HarborEvaluatorConfig,
    HarborNativeOutcomeEvaluator,
)
from nemo_experimentalist_plugin.experimentalist.components.trace_analyzer import Diagnostic
from nemo_insights_plugin.entities import Insight
from nemo_helix_plugin.nooa_model_client import (
    activate_model_clients,
    configured_model_refs,
    get_fast_model,
    resolve_model_clients,
)


def _has_configured_models() -> bool:
    try:
        configured_model_refs()
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return True


_HAS_MODELS = _has_configured_models()
_MODELS_HINT = "run `nemo setup` and select default and fast agent models"
_RUN_EVAL_AUTHOR_REPAIR_E2E = os.environ.get("RUN_EVAL_AUTHOR_REPAIR_E2E") == "1"
_RUN_EVAL_AUTHOR_HARBOR_E2E = os.environ.get("RUN_EVAL_AUTHOR_HARBOR_E2E") == "1"
_MALFORMED_VERIFIER = """\
def check_tool_hallucination() -> float:
    try:
        return 1.0
"""
_KNOWN_FAILING_TRACE = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "name": "generate_response",
                            "attributes": [
                                {
                                    "key": "openinference.span.kind",
                                    "value": {"stringValue": "CHAIN"},
                                },
                                {
                                    "key": "output.value",
                                    "value": {
                                        "stringValue": (
                                            "The Denver warehouse has eight units available. "
                                            "I answered from memory without consulting inventory."
                                        )
                                    },
                                },
                            ],
                        }
                    ]
                }
            ]
        }
    ]
}
_KNOWN_COMPLIANT_TRACE = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "name": "inventory_lookup",
                            "attributes": [
                                {
                                    "key": "openinference.span.kind",
                                    "value": {"stringValue": "TOOL"},
                                },
                                {
                                    "key": "output.value",
                                    "value": {"stringValue": '{"warehouse":"Denver","available_units":8}'},
                                },
                            ],
                        },
                        {
                            "name": "generate_response",
                            "attributes": [
                                {
                                    "key": "openinference.span.kind",
                                    "value": {"stringValue": "CHAIN"},
                                },
                                {
                                    "key": "output.value",
                                    "value": {"stringValue": "The Denver warehouse has eight units available."},
                                },
                            ],
                        },
                    ]
                }
            ]
        }
    ]
}


@pytest_asyncio.fixture
async def configured_models():
    client = make_client(None)
    model_clients = await resolve_model_clients(client)
    try:
        with activate_model_clients(model_clients):
            yield
    finally:
        await model_clients.aclose()
        await client.close()


def _write_malformed_task(dataset_dir: Path, task_id: str) -> None:
    task_dir = dataset_dir / task_id
    verifier_dir = task_dir / "tests"
    verifier_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[task]\nname = "local/{task_id}"\n', encoding="utf-8")
    (verifier_dir / "test.sh").write_text(
        "#!/usr/bin/env bash\npython /tests/check_tool_hallucination.py\n",
        encoding="utf-8",
    )
    (verifier_dir / "check_tool_hallucination.py").write_text(_MALFORMED_VERIFIER, encoding="utf-8")


def _write_known_failing_task(dataset_dir: Path) -> None:
    task_dir = dataset_dir / "known-failing-inventory-lookup"
    tests_dir = task_dir / "tests"
    environment_dir = task_dir / "environment"
    tests_dir.mkdir(parents=True)
    environment_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
version = "1.0"

[task]
name = "local/known-failing-inventory-lookup"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 512
storage_mb = 1024
gpus = 0
network_mode = "no-network"
mcp_servers = []

[verifier.env]

[solution.env]
""",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text(
        "Use the inventory_lookup tool before reporting how many units are available in the Denver warehouse.\n",
        encoding="utf-8",
    )
    (environment_dir / "Dockerfile").write_text(
        """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim
WORKDIR /app
""",
        encoding="utf-8",
    )
    (tests_dir / "test.sh").write_text(
        """\
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p /logs/verifier
printf '{"reward": 1.0}\n' > /logs/verifier/reward.json
""",
        encoding="utf-8",
    )


def _write_known_failing_agent(agent_dir: Path) -> None:
    _write_harbor_trace_agent(
        agent_dir,
        agent_name="known-failing-baseline",
        trace_payload=_KNOWN_FAILING_TRACE,
    )


def _write_harbor_trace_agent(
    agent_dir: Path,
    *,
    agent_name: str,
    trace_payload: object | None,
) -> None:
    agent_dir.mkdir()
    serialized_trace = json.dumps(trace_payload, separators=(",", ":")) if trace_payload is not None else None
    trace_command = (
        f"mkdir -p /app/traces && printf '%s\\n' {shlex.quote(serialized_trace)} > /app/traces/trace.jsonl"
        if serialized_trace is not None
        else "true"
    )
    (agent_dir / "harbor_wrapper.py").write_text(
        f"""\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from harbor import AgentContext, BaseAgent, BaseEnvironment


class WrappedAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return {agent_name!r}

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        command = {trace_command!r}
        process = await environment.exec(command)
        context.metadata = {{
            "instruction": instruction,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.return_code,
        }}
        if process.return_code != 0:
            raise RuntimeError(process.stderr or process.stdout)
""",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not (_RUN_EVAL_AUTHOR_REPAIR_E2E and _HAS_MODELS),
    reason=f"Set RUN_EVAL_AUTHOR_REPAIR_E2E=1 and {_MODELS_HINT} to run the Eval Author repair canary.",
)
async def test_configured_fast_model_repairs_malformed_harbor_verifiers(
    tmp_path: Path,
    configured_models: None,
) -> None:
    """The configured fast model repairs try-without-except failures across an Insight suite."""
    insight_suite_dir = tmp_path / "insight-suite"
    _write_malformed_task(insight_suite_dir, "task-a")
    _write_malformed_task(insight_suite_dir, "task-b")
    insight_suite = HarborDataset.from_path(insight_suite_dir)

    with pytest.raises(DatasetValidationError) as exc_info:
        await insight_suite.validate()

    validation_feedback = str(exc_info.value)
    assert "task 'task-a'" in validation_feedback
    assert "task 'task-b'" in validation_feedback
    assert "SyntaxError: expected 'except' or 'finally' block" in validation_feedback

    llm = get_fast_model()
    eval_author = EvalAuthor(
        experiment_dir=tmp_path,
        config=EvalAuthorConfig(),
        llm=llm,
    )
    insight = Insight(
        workspace="local",
        title="Harbor verifier contains invalid Python syntax",
        description=(
            "The shared check_tool_hallucination.py verifier contains a try statement without an except or finally "
            "clause. Repair every reported syntax error without changing the intended successful score of 1.0."
        ),
        agent="eval-author-repair-canary",
    )
    runner_conventions = (
        "This is a Harbor dataset. Verifier files live under each task's tests directory. "
        "Python verifier files are statically checked by await dataset.validate(); test.sh is checked as Bash. "
        "Preserve the existing verifier's intended behavior and repair every validation error."
    )
    summary = await asyncio.wait_for(
        eval_author.author_insight_metrics(
            insight,
            [],
            insight_suite,
            insight_suite,
            insight_suite,
            runner_conventions,
            validation_feedback=validation_feedback,
        ),
        timeout=300,
    )

    assert summary.summary
    await insight_suite.validate()
    for verifier_path in (
        insight_suite_dir / "task-a" / "tests" / "check_tool_hallucination.py",
        insight_suite_dir / "task-b" / "tests" / "check_tool_hallucination.py",
    ):
        repaired_source = verifier_path.read_text(encoding="utf-8")
        assert "def check_tool_hallucination" in repaired_source
        assert "return 1.0" in repaired_source


@pytest.mark.skipif(
    not (_RUN_EVAL_AUTHOR_HARBOR_E2E and _HAS_MODELS),
    reason=f"Set RUN_EVAL_AUTHOR_HARBOR_E2E=1 and {_MODELS_HINT} to run the live Harbor metric canary.",
)
async def test_eval_author_metric_scores_known_failing_harbor_baseline_low(
    tmp_path: Path,
    configured_models: None,
) -> None:
    """An authored root-cause metric scores a known-failing Harbor baseline low."""
    insight_suite_dir = tmp_path / "insight-suite"
    agent_dir = tmp_path / "known-failing-agent"
    _write_known_failing_task(insight_suite_dir)
    _write_known_failing_agent(agent_dir)
    insight_suite = HarborDataset.from_path(insight_suite_dir)
    await insight_suite.validate()

    llm = get_fast_model()
    llm.config["temperature"] = 0.0
    eval_author = EvalAuthor(
        experiment_dir=tmp_path,
        config=EvalAuthorConfig(),
        llm=llm,
    )
    eval_author.context.pop("trace_documentation", None)
    insight = Insight(
        workspace="local",
        title="Agent skips the required inventory lookup",
        description=(
            "The agent reports warehouse availability from memory instead of calling the required "
            "inventory_lookup tool. The metric must measure whether the current Harbor execution "
            "contains that required tool call, not whether the final answer happens to contain a quantity."
        ),
        agent="known-failing-baseline",
    )
    diagnostic = Diagnostic(
        outcome="FAILURE",
        summary=(
            "The execution trace contains only response generation. It has no tool span and no evidence "
            "that live inventory was retrieved before the answer."
        ),
        failure_point=1,
        root_cause="The agent answered from memory without calling the required inventory_lookup tool.",
    )
    runner_conventions = (
        "This is a Harbor dataset. Preserve the existing reward metric and add a new numeric root-cause metric. "
        "Verifier files live under each task's tests directory. After the agent runs, OTLP JSONL trace files are "
        "available in the directory named by the TRACE_DIR environment variable. Every metric must be written as a "
        "numeric entry in "
        "/logs/verifier/reward.json, where higher is better and values are bounded to [0.0, 1.0]. Make the minimal "
        "verifier-only edit. Missing tool evidence is the expected failing case: it must score 0.0 while still "
        "writing reward.json and exiting successfully. Do not use an unguarded grep pipeline whose no-match status "
        "can abort a set -e script; prefer a small Python standard-library checker. Call await "
        "insight_suite.validate() once after editing, then return the metric summary as soon as validation passes; "
        "do not inspect unrelated files."
    )
    summary = await asyncio.wait_for(
        eval_author.author_insight_metrics(
            insight,
            [("known-failing-trace", diagnostic)],
            insight_suite,
            insight_suite,
            insight_suite,
            runner_conventions,
        ),
        timeout=600,
    )

    assert summary.summary
    await insight_suite.validate()
    evaluator = HarborNativeOutcomeEvaluator(experiment_dir=tmp_path)
    result = await asyncio.wait_for(
        evaluator.run(
            agent=agent_dir,
            dataset=insight_suite,
            options=HarborEvaluatorConfig(
                force_rerun=True,
                job_name="known-failing-insight-metric",
                jobs_dir=Path("harbor-jobs"),
                n_concurrent_trials=1,
                quiet=True,
            ),
        ),
        timeout=600,
    )

    assert len(result.trials) == 1
    trial = result.trials[0]
    assert trial.status == "completed", trial.error
    reward_ref = trial.resources["log:verifier/reward.json"]
    reward_path = local_path_from_uri(reward_ref.uri, context="Harbor verifier reward")
    reward_payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert reward_payload["reward"] == pytest.approx(1.0)

    insight_metric_names = set(reward_payload) - {"reward"}
    assert insight_metric_names
    insight_metric_values = {
        name: float(reward_payload[name])
        for name in insight_metric_names
        if isinstance(reward_payload[name], int | float) and not isinstance(reward_payload[name], bool)
    }
    assert set(insight_metric_values) == insight_metric_names
    assert all(0.0 <= value <= 1.0 for value in insight_metric_values.values())
    assert min(insight_metric_values.values()) <= 0.25
    assert insight_metric_names <= set(trial.metrics)
    assert set(summary.metric_keys) == insight_metric_names
    print(
        json.dumps(
            {
                "authored_summary": summary.model_dump(mode="json"),
                "reward_json": reward_payload,
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(
    not (_RUN_EVAL_AUTHOR_HARBOR_E2E and _HAS_MODELS),
    reason=f"Set RUN_EVAL_AUTHOR_HARBOR_E2E=1 and {_MODELS_HINT} to run the live Harbor metric canary.",
)
async def test_eval_author_metric_discriminates_controlled_harbor_tool_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authored metric distinguishes measurable violations from compliant tool evidence."""
    insight_suite_dir = tmp_path / "insight-suite"
    violating_agent_dir = tmp_path / "violating-agent"
    compliant_agent_dir = tmp_path / "compliant-agent"
    unmeasurable_agent_dir = tmp_path / "unmeasurable-agent"
    _write_known_failing_task(insight_suite_dir)
    _write_harbor_trace_agent(
        violating_agent_dir,
        agent_name="controlled-tool-violation",
        trace_payload=_KNOWN_FAILING_TRACE,
    )
    _write_harbor_trace_agent(
        compliant_agent_dir,
        agent_name="controlled-tool-compliance",
        trace_payload=_KNOWN_COMPLIANT_TRACE,
    )
    _write_harbor_trace_agent(
        unmeasurable_agent_dir,
        agent_name="controlled-unmeasurable-trace",
        trace_payload=None,
    )
    insight_suite = HarborDataset.from_path(insight_suite_dir)
    await insight_suite.validate()

    llm = get_fast_model()
    llm.config["temperature"] = 0.0
    monkeypatch.delenv("NEMO_EXPERIMENTALIST_API_KEY", raising=False)
    eval_author = EvalAuthor(
        experiment_dir=tmp_path,
        config=EvalAuthorConfig(),
        llm=llm,
    )
    eval_author.context.pop("trace_documentation", None)
    insight = Insight(
        workspace="local",
        title="Agent must use inventory_lookup before reporting availability",
        description=(
            "The agent must call inventory_lookup before reporting the Denver warehouse availability. "
            "Measure current execution evidence of that tool call, not answer text or static task files."
        ),
        agent="controlled-tool-evidence",
    )
    diagnostic = Diagnostic(
        outcome="FAILURE",
        summary=(
            "The violating trace has a response span but no inventory_lookup tool span, so the absence "
            "of the required call is measurable."
        ),
        failure_point=1,
        root_cause="The agent answered from memory without calling inventory_lookup.",
    )
    runner_conventions = (
        "This is a Harbor dataset. Preserve the existing reward metric and add a numeric root-cause metric to "
        "/logs/verifier/reward.json. Higher values are better and values are bounded to [0.0, 1.0]. A readable "
        "trace with no inventory_lookup tool span is measurable failing evidence and should score low. A readable "
        "trace with an inventory_lookup tool span before the answer is compliant evidence and should score high. "
        "A missing, unreadable, or malformed trace is not measurable evidence: fail the verifier with a clear "
        "error instead of writing a fabricated 0.0 metric. Use a Python standard-library checker, avoid "
        "unguarded grep under set -e, make the minimal verifier-only edit, validate the dataset once, then stop."
    )
    summary = await asyncio.wait_for(
        eval_author.author_insight_metrics(
            insight,
            [("controlled-tool-violation", diagnostic)],
            insight_suite,
            insight_suite,
            insight_suite,
            runner_conventions,
        ),
        timeout=600,
    )

    assert summary.summary
    await insight_suite.validate()
    evaluator = HarborNativeOutcomeEvaluator(experiment_dir=tmp_path)

    async def run_agent(agent_dir: Path, job_name: str):
        return await asyncio.wait_for(
            evaluator.run(
                agent=agent_dir,
                dataset=insight_suite,
                options=HarborEvaluatorConfig(
                    force_rerun=True,
                    job_name=job_name,
                    jobs_dir=Path("harbor-jobs"),
                    n_concurrent_trials=1,
                    quiet=True,
                ),
            ),
            timeout=600,
        )

    violating_result = await run_agent(violating_agent_dir, "controlled-tool-violation")
    compliant_result = await run_agent(compliant_agent_dir, "controlled-tool-compliance")
    unmeasurable_result = await run_agent(unmeasurable_agent_dir, "controlled-unmeasurable-trace")

    def reward_payload(result) -> dict[str, object]:
        assert len(result.trials) == 1
        trial = result.trials[0]
        assert trial.status == "completed", trial.error
        reward_ref = trial.resources["log:verifier/reward.json"]
        reward_path = local_path_from_uri(reward_ref.uri, context="Harbor verifier reward")
        return json.loads(reward_path.read_text(encoding="utf-8"))

    violating_payload = reward_payload(violating_result)
    compliant_payload = reward_payload(compliant_result)
    assert violating_payload["reward"] == pytest.approx(1.0)
    assert compliant_payload["reward"] == pytest.approx(1.0)
    violating_metric_names = set(violating_payload) - {"reward"}
    compliant_metric_names = set(compliant_payload) - {"reward"}
    assert violating_metric_names
    assert compliant_metric_names == violating_metric_names
    assert set(summary.metric_keys) == violating_metric_names
    for metric_name in sorted(violating_metric_names):
        violating_score = violating_payload[metric_name]
        compliant_score = compliant_payload[metric_name]
        assert isinstance(violating_score, int | float) and not isinstance(violating_score, bool)
        assert isinstance(compliant_score, int | float) and not isinstance(compliant_score, bool)
        assert 0.0 <= violating_score <= 1.0
        assert 0.0 <= compliant_score <= 1.0
        assert compliant_score > violating_score

    assert len(unmeasurable_result.trials) == 1
    unmeasurable_trial = unmeasurable_result.trials[0]
    assert unmeasurable_trial.status != "completed"
    verifier_stderr_ref = unmeasurable_trial.resources["log:verifier/test-stderr.txt"]
    verifier_stderr_path = local_path_from_uri(
        verifier_stderr_ref.uri,
        context="Harbor verifier stderr for unmeasurable trace",
    )
    assert "trace" in verifier_stderr_path.read_text(encoding="utf-8").lower()
    assert set(unmeasurable_trial.metrics) - {"reward"} == set()
    unmeasurable_reward_ref = unmeasurable_trial.resources.get("log:verifier/reward.json")
    if unmeasurable_reward_ref is not None:
        unmeasurable_reward_path = local_path_from_uri(
            unmeasurable_reward_ref.uri,
            context="Harbor verifier reward for unmeasurable trace",
        )
        unmeasurable_payload = json.loads(unmeasurable_reward_path.read_text(encoding="utf-8"))
        assert set(unmeasurable_payload) == {"reward"}
