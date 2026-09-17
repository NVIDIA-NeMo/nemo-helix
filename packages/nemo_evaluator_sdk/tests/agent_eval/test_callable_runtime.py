# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.callable_runtime import (
    CallableAgentTaskRunner,
    TrialDraft,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus, AgentOutput, TrialMeasurements


def _task(task_id: str) -> AgentEvalTask:
    return AgentEvalTask(id=task_id, intent="Answer.", inputs={"instruction": f"q-{task_id}"})


@pytest.mark.asyncio
async def test_callable_runtime_wraps_str_output_and_trialdraft_in_order() -> None:
    async def agent_fn(task: AgentEvalTask) -> TrialDraft | str:
        if task.id == "t2":
            return TrialDraft(
                output=AgentOutput(output_text="drafted"),
                metadata={"k": "v"},
                measurements=TrialMeasurements(prompt_tokens=4, completion_tokens=1),
            )
        return f"answer-{task.id}"

    runtime = CallableAgentTaskRunner(agent_fn, parallelism=2)
    trials = await runtime.run_tasks([_task("t1"), _task("t2")], config=AgentEvalRunConfig())

    assert [trial.task_id for trial in trials] == ["t1", "t2"]
    assert trials[0].id == "t1:trial"
    assert trials[0].status == AgentEvalTrialStatus.COMPLETED
    assert trials[0].output is not None and trials[0].output.output_text == "answer-t1"
    assert trials[1].output is not None and trials[1].output.output_text == "drafted"
    assert trials[1].metadata == {"k": "v"}
    assert trials[1].measurements.total_tokens == 5
    assert trials[0].measurements == TrialMeasurements()


@pytest.mark.asyncio
async def test_callable_runtime_captures_exception_as_failed_trial_but_rejects_bad_return() -> None:
    async def failing_fn(task: AgentEvalTask) -> str:
        raise RuntimeError("nope")

    failed = (await CallableAgentTaskRunner(failing_fn).run_tasks([_task("boom")]))[0]
    assert failed.status == AgentEvalTrialStatus.FAILED
    assert failed.output is None
    assert "RuntimeError: nope" in failed.metadata["error"]

    async def bad_return_fn(task: AgentEvalTask) -> str:
        return 123  # type: ignore[return-value]  # invalid agent output is a programming error, not a failed trial

    with pytest.raises(TypeError, match="TrialDraft, AgentOutput, or str"):
        await CallableAgentTaskRunner(bad_return_fn).run_tasks([_task("bad")])
