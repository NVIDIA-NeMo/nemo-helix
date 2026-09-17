# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gym runner: content-hash identity, instruction rendering, task
discovery, and the rollout→trial fan-out — verified against a captured mcqa bundle.

Gym mutates ``responses_create_params`` and copies only a fixed allowlist of row
keys onto results, so ``_ng_task_index`` is the only surviving join. The runner
*assigns* that index in a materialized dataset rather than inferring it; these
tests lock that behaviour in (including against the real captured output)."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import Counter, deque
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.gym import (
    GymAgentTaskRunner,
    GymRewardMetric,
    GymRuntimeConfig,
    discover_gym_tasks,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import (
    _flatten_overrides,
    hydra_scalar,
    model_call_capture_dir,
    selection_args,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.dataset import (
    _canonical_row_hash,
    _content_text,
    _render_instruction,
    materialize_dataset,
    source_datasets,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.process import (
    LOG_TAIL_LINES,
    drain_pumps,
    gym_executable,
    gym_invocation_env,
    pending_servers,
    pump_stream,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import NG_ATTEMPT_INDEX, NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nemo_evaluator_sdk.agent_eval.runtimes.gym.results import (
    _INPUT_TOKEN_KEYS,
    _TOKEN_USAGE_KEYS,
    _agent_never_ran,
    _aggregate_metrics_path_for,
    _usage_measurements,
    _warn_invalid_usage_counts,
    ensure_fresh_output,
    read_run_aggregations,
    require_full_coverage,
    trials_from_rollouts,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus, TrialMeasurements
from nemo_evaluator_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = FIXTURES / "gym_mcqa_example.jsonl"
ROLLOUTS = FIXTURES / "gym_mcqa_rollouts.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index_map(tasks: list, tmp_path: Path) -> tuple[list[str], dict[int, str]]:
    """Materialize ``tasks`` the way the runner does, returning (task ids in order, index->id map)."""
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    return [task.id for task in tasks], index_map


def test_canonical_row_hash_is_stable_and_ignores_runtime_fields() -> None:
    base = {"responses_create_params": {"input": "q"}, "expected_answer": "A"}
    digest = _canonical_row_hash(base)
    # runtime-injected index fields do not perturb the hash
    assert _canonical_row_hash({**base, "_ng_task_index": 3, "_ng_rollout_index": 1}) == digest
    # key order does not matter
    assert _canonical_row_hash({"expected_answer": "A", "responses_create_params": {"input": "q"}}) == digest
    # a real content change yields a new id
    assert _canonical_row_hash({**base, "expected_answer": "B"}) != digest


def test_render_instruction_from_string_messages_and_parts() -> None:
    assert _render_instruction({"input": "hello"}) == "hello"
    assert _render_instruction({"input": [{"role": "user", "content": "pick A or B"}]}) == "pick A or B"
    assert _render_instruction({"instructions": "be brief", "input": "q"}) == "be brief\n\nq"
    parts = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "part"}]}]}
    assert _render_instruction(parts) == "part"


def test_render_instruction_is_empty_rather_than_fatal_when_the_row_carries_no_prompt() -> None:
    # A whole class of Gym environments ships `{"input": []}` and materializes the prompt elsewhere.
    # Rendering must report "no prompt", not fail the dataset.
    assert _render_instruction({"input": ""}) == ""
    assert _render_instruction({"input": []}) == ""
    assert _render_instruction({}) == ""


@pytest.mark.parametrize(
    ("name", "row"),
    [
        # Shapes taken verbatim from the five affected environments' data/example.jsonl.
        ("gdpval", {"responses_create_params": {"input": []}, "task_id": "83d10b06", "prompt": "You are an auditor…"}),
        (
            "legal_agent_bench",
            {
                "agent_ref": {"name": "legal_agent_bench_harbor_agent", "type": "responses_api_agents"},
                "instance_id": "legal_agent_bench::corporate-ma__analyze-tsa-markup",
                "responses_create_params": {"input": [], "temperature": 1.0, "top_p": 0.95},
            },
        ),
        ("aviary", {"task_idx": 0, "responses_create_params": {"input": []}, "agent_ref": {"name": "gsm8k_aviary"}}),
        ("scicode", {"responses_create_params": {"input": []}, "problem_id": "10", "sub_steps": [{"step": "10.1"}]}),
        ("toolsandbox", {"task_idx": 0, "responses_create_params": {"input": []}}),
    ],
)
def test_discover_gym_tasks_ingests_rows_with_no_prompt(name: str, row: dict, tmp_path: Path) -> None:
    # AALGO-498: these environments could not be ingested at all — discovery raised before a run began.
    dataset = tmp_path / f"{name}.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    (task,) = discover_gym_tasks(dataset)

    # No instruction is *absent*, not empty: agent_prompt() must still refuse to run an agent on nothing.
    assert "instruction" not in task.inputs
    with pytest.raises(ValueError, match="has no instruction"):
        task.agent_prompt()
    # ...but everything the Gym path actually needs survives: the row round-trips whole.
    assert task.inputs["gym_row"] == row["responses_create_params"]
    assert task.metadata["gym_row_extras"] == {k: v for k, v in row.items() if k != "responses_create_params"}


def test_promptless_rows_still_hash_distinctly(tmp_path: Path) -> None:
    # Identity is the whole row, so rows differing only in a field the runner ignores (`task_idx` for
    # aviary/toolsandbox, `instance_id` for legal_agent_bench) must remain distinct tasks — otherwise a
    # promptless dataset would collapse into a single task.
    dataset = tmp_path / "toolsandbox.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"task_idx": idx, "responses_create_params": {"input": []}}) for idx in range(5)) + "\n",
        encoding="utf-8",
    )
    tasks = discover_gym_tasks(dataset)
    assert len({task.id for task in tasks}) == 5
    # and they re-materialize into the five rows Gym will read, each with its own index
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    rows = _rows(tmp_path / "gym_input.jsonl")
    assert [row["task_idx"] for row in rows] == [0, 1, 2, 3, 4]
    assert index_map == {index: task.id for index, task in enumerate(tasks)}


def test_discover_gym_tasks_reports_how_many_rows_had_no_prompt(tmp_path: Path, caplog) -> None:
    dataset = tmp_path / "mixed.jsonl"
    dataset.write_text(
        json.dumps({"responses_create_params": {"input": "answer me"}})
        + "\n"
        + json.dumps({"task_idx": 0, "responses_create_params": {"input": []}})
        + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO):
        tasks = discover_gym_tasks(dataset)
    assert [("instruction" in task.inputs) for task in tasks] == [True, False]
    assert "1 of 2 task(s)" in caplog.text
    assert "carry no prompt" in caplog.text


@pytest.mark.parametrize("row", [{"task_idx": 0}, {"responses_create_params": None}, {"responses_create_params": []}])
def test_discover_gym_tasks_still_requires_responses_create_params(row: dict, tmp_path: Path) -> None:
    # Gym indexes into this key unconditionally, so a row without a usable one must be rejected at
    # discovery rather than crashing `gym eval run` after the servers are up. All three non-mapping
    # shapes are covered deliberately: a check of `params is None` alone would pass the missing-key
    # case while letting a list through to fail deep inside Gym.
    dataset = tmp_path / "bad_params.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 1 .* has no 'responses_create_params' mapping"):
        discover_gym_tasks(dataset)


def test_discover_gym_tasks_from_example_fixture() -> None:
    tasks = discover_gym_tasks(EXAMPLE)
    assert len(tasks) == len(_rows(EXAMPLE))
    assert len({task.id for task in tasks}) == len(tasks)  # unique ids
    for task in tasks:
        assert len(task.id) == 64  # sha256 hex
        assert task.inputs["instruction"]  # non-empty instruction
        assert task.metadata["gym_dataset_path"].endswith("gym_mcqa_example.jsonl")
        assert [type(metric).__name__ for metric in task.metrics] == ["GymRewardMetric"]


def test_materialize_dataset_stamps_task_index_and_returns_total_map(tmp_path: Path) -> None:
    # The runner assigns _ng_task_index rather than inferring it: every task gets exactly one row,
    # stamped with its own index, and the returned map covers all of them.
    tasks = discover_gym_tasks(EXAMPLE)
    dest = tmp_path / "gym_input.jsonl"
    index_map = materialize_dataset(tasks, dest)
    rows = _rows(dest)
    assert len(rows) == len(tasks)
    assert [row[NG_TASK_INDEX] for row in rows] == list(range(len(tasks)))
    assert index_map == {index: task.id for index, task in enumerate(tasks)}
    # the full source row is carried through so Gym's verifier still sees its non-prompt fields
    assert all("responses_create_params" in row for row in rows)
    # Gym assigns the rollout index per attempt; ours must not be present to be honored/overridden
    assert all(NG_ROLLOUT_INDEX not in row for row in rows)


def test_materialize_dataset_subsets_to_requested_tasks(tmp_path: Path) -> None:
    # Running a subset must hand Gym only that subset — otherwise Gym rolls out (and bills for) rows
    # whose trials we would then discard.
    tasks = discover_gym_tasks(EXAMPLE)
    subset = [tasks[3], tasks[1]]
    index_map = materialize_dataset(subset, tmp_path / "gym_input.jsonl")
    assert index_map == {0: tasks[3].id, 1: tasks[1].id}
    assert len(_rows(tmp_path / "gym_input.jsonl")) == 2


def test_materialize_dataset_overrides_preassigned_task_index(tmp_path: Path) -> None:
    # Codex's example 1: a dataset that already carries out-of-order _ng_task_index values. Gym honors
    # a pre-stamped index, so leaving the source values in place would swap trials between tasks. Our
    # stamp must be authoritative, and the map must agree with it.
    dataset = tmp_path / "preindexed.jsonl"
    dataset.write_text(
        json.dumps({NG_TASK_INDEX: 1, "responses_create_params": {"input": "What is 2 + 2?"}})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 0, "responses_create_params": {"input": "What is France's capital?"}})
        + "\n",
        encoding="utf-8",
    )
    tasks = discover_gym_tasks(dataset)
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    rows = _rows(tmp_path / "gym_input.jsonl")
    assert [row[NG_TASK_INDEX] for row in rows] == [0, 1]  # source values discarded
    # index 0 is the math row (first in file order), index 1 the capital row
    assert "2 + 2" in json.dumps(rows[0]["responses_create_params"])
    assert index_map[0] == tasks[0].id and index_map[1] == tasks[1].id

    # ...and the join round-trips: Gym echoing our stamps attributes each rollout to the right task.
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 1, NG_ROLLOUT_INDEX: 0, "response": "Lyon", "reward": 0.0})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "response": "4", "reward": 1.0})
        + "\n",
        encoding="utf-8",
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)
    responses = {trial.task_id: (trial.output.response if trial.output else None) for trial in trials}
    rewards = {trial.task_id: trial.metadata["reward"] for trial in trials}
    assert responses[tasks[0].id] == "4"  # the math task got the math answer, not the capital task's
    assert rewards[tasks[0].id] == 1.0
    assert responses[tasks[1].id] == "Lyon"


def test_serialization_variant_rows_collapse_without_index_drift(tmp_path: Path) -> None:
    # Codex's example 2: two rows with identical JSON *values* but different serialization. Gym's own
    # fallback dedup keys off raw line text and would assign two indices; ours collapses them to one
    # task. Because we now stamp the index ourselves, Gym never gets the chance to disagree.
    dataset = tmp_path / "variants.jsonl"
    dataset.write_text(
        '{"responses_create_params":{"input":"Choose A"},"expected":"A"}\n'
        '{ "expected": "A", "responses_create_params": { "input": "Choose A" } }\n',
        encoding="utf-8",
    )
    tasks = discover_gym_tasks(dataset)
    assert len(tasks) == 1  # same content -> same task
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    assert index_map == {0: tasks[0].id}
    assert len(_rows(tmp_path / "gym_input.jsonl")) == 1  # Gym is handed one row, so it emits one index


def test_discover_gym_tasks_warns_on_duplicate_rows(tmp_path: Path, caplog) -> None:
    # Repeated attempts are num_repeats, not row duplication. Duplicates collapse (identity is content)
    # and must be reported rather than silently swallowed.
    dataset = tmp_path / "dupes.jsonl"
    row = json.dumps({"responses_create_params": {"input": "same question"}})
    dataset.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        tasks = discover_gym_tasks(dataset)
    assert len(tasks) == 1
    assert "duplicate row" in caplog.text
    assert "num_repeats" in caplog.text


def test_materialize_dataset_reports_empty_task_list(tmp_path: Path) -> None:
    # The empty case used to surface as a confusing "needs a dataset path" from the provenance helper.
    with pytest.raises(ValueError, match="no tasks to run"):
        materialize_dataset([], tmp_path / "gym_input.jsonl")


def test_materialize_dataset_rejects_duplicate_tasks(tmp_path: Path) -> None:
    tasks = discover_gym_tasks(EXAMPLE)
    with pytest.raises(ValueError, match="more than once"):
        materialize_dataset([tasks[0], tasks[0]], tmp_path / "gym_input.jsonl")


def test_materialize_dataset_requires_source_rows(tmp_path: Path) -> None:
    # Tasks not built by discover_gym_tasks can't be re-materialized; fail before launching Gym.
    tasks = discover_gym_tasks(EXAMPLE)
    no_extras = tasks[0].model_copy(update={"metadata": {"gym_dataset_path": str(EXAMPLE)}})
    with pytest.raises(ValueError, match="gym_row_extras"):
        materialize_dataset([no_extras], tmp_path / "gym_input.jsonl")
    # the other half of the split is equally required
    no_params = tasks[0].model_copy(update={"inputs": {k: v for k, v in tasks[0].inputs.items() if k != "gym_row"}})
    with pytest.raises(ValueError, match="gym_row"):
        materialize_dataset([no_params], tmp_path / "gym_input.jsonl")


def test_content_text_warns_on_unsupported_shape(caplog) -> None:
    # A populated content in an unexpected shape means prompt text is being dropped -> warn.
    with caplog.at_level(logging.WARNING):
        assert _content_text({"unexpected": "mapping"}) == ""
    assert "unsupported type" in caplog.text
    # a missing content is unremarkable and must stay quiet
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert _content_text(None) == ""
    assert caplog.text == ""


def test_trials_fan_out_one_per_attempt(tmp_path: Path) -> None:
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    # synthesize a bundle: the first 2 tasks (indices 0,1) x 2 attempts, joined via _ng_task_index
    records = [
        {NG_TASK_INDEX: index, NG_ROLLOUT_INDEX: attempt, "reward": float(attempt), "response": {"ok": True}}
        for index in (0, 1)
        for attempt in range(2)  # num_repeats = 2
    ]
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    trials = trials_from_rollouts(bundle, tasks, index_map)
    assert len(trials) == 4  # 2 tasks x 2 attempts
    assert len({trial.id for trial in trials}) == 4  # distinct trial ids
    assert {trial.task_id for trial in trials} == {ordered[0], ordered[1]}  # attributed via _ng_task_index
    assert set(Counter(trial.task_id for trial in trials).values()) == {2}  # 2 trials per task
    assert {trial.metadata["reward"] for trial in trials} == {0.0, 1.0}


def test_trials_raise_on_unstamped_task_index(tmp_path: Path) -> None:
    # Every row Gym read carried an index we stamped, so an index we never issued means Gym reassigned
    # them instead of honoring ours. Attributing it would be a silent mis-join -> fail loudly.
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(json.dumps({NG_TASK_INDEX: 999, "reward": 1.0}) + "\n", encoding="utf-8")  # out of range
    with pytest.raises(ValueError):
        trials_from_rollouts(bundle, tasks, index_map)


def test_failures_sidecar_yields_failed_trials(tmp_path: Path) -> None:
    # Gym writes failed attempts to a *_failures.jsonl sidecar, not rollouts.jsonl. They must surface
    # as FAILED trials (counted + diagnosed), not vanish.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps(
            {
                NG_TASK_INDEX: 0,
                NG_ROLLOUT_INDEX: 0,
                "reward": 1.0,
                "response": {"usage": {"input_tokens": 8, "output_tokens": 2}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(
        json.dumps(
            {
                NG_TASK_INDEX: 1,
                NG_ROLLOUT_INDEX: 0,
                "error": "boom",
                "response": {"usage": {"prompt_tokens": 4, "completion_tokens": 1}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)
    by_status = {trial.task_id: trial.status for trial in trials}
    assert by_status[ordered[0]] == AgentEvalTrialStatus.COMPLETED
    assert by_status[ordered[1]] == AgentEvalTrialStatus.FAILED
    failed = next(trial for trial in trials if trial.status == AgentEvalTrialStatus.FAILED)
    completed = next(trial for trial in trials if trial.status == AgentEvalTrialStatus.COMPLETED)
    assert completed.measurements.total_tokens == 10
    assert failed.measurements.total_tokens == 5
    assert failed.metadata["reward"] is None
    assert failed.metadata["gym_failure"] == "boom"


def test_ensure_fresh_output_rejects_reused_dir(tmp_path: Path) -> None:
    # Gym appends to the failures sidecar, so a directory already holding one must be refused
    # (rather than cleared) to avoid mixing runs / clobbering prior results.
    rollouts = tmp_path / "rollouts.jsonl"
    (tmp_path / "rollouts_failures.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_fresh_output(rollouts)


def test_ensure_fresh_output_allows_clean_dir(tmp_path: Path) -> None:
    ensure_fresh_output(tmp_path / "rollouts.jsonl")  # no prior artifacts -> no raise


def test_source_datasets_is_provenance_only_and_never_raises() -> None:
    # The dataset path no longer drives execution — Gym reads the materialized file, built from the
    # tasks themselves. So mixing datasets in one run is now legitimate (the materialized dataset is
    # the union) and this summary must not gate the run; it only labels a log line.
    tasks = discover_gym_tasks(EXAMPLE)
    assert "gym_mcqa_example.jsonl" in source_datasets(tasks)
    other = tasks[0].model_copy(update={"metadata": {**tasks[0].metadata, "gym_dataset_path": "/other/ds.jsonl"}})
    summary = source_datasets([*tasks, other])  # must not raise
    assert "gym_mcqa_example.jsonl" in summary and "/other/ds.jsonl" in summary
    # tasks with no stamped path at all are fine too — the run does not depend on it
    unstamped = tasks[0].model_copy(
        update={"metadata": {k: v for k, v in tasks[0].metadata.items() if k != "gym_dataset_path"}}
    )
    assert source_datasets([unstamped])


def test_materialize_dataset_reassembles_row_without_storing_params_twice(tmp_path: Path) -> None:
    # The task already carries responses_create_params under inputs['gym_row']; gym_row_extras must
    # hold only the *rest* of the row so the run bundle does not persist the same payload twice per task.
    tasks = discover_gym_tasks(EXAMPLE)
    extras = tasks[0].metadata["gym_row_extras"]
    assert "responses_create_params" not in extras  # not duplicated
    # ...yet materialization still reconstructs the complete source row Gym's verifier needs
    materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    row = _rows(tmp_path / "gym_input.jsonl")[0]
    source = _rows(EXAMPLE)[0]
    assert {k: v for k, v in row.items() if k != NG_TASK_INDEX} == source


@pytest.mark.asyncio
async def test_a_rollout_carries_an_otlp_trace_the_sdk_can_read(tmp_path: Path) -> None:
    # Gym records a result, not a trace, so the OTLP view is projected from the record. It earns its
    # place only if the SDK's own reader accepts it and finds the agent's answer.
    from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
    from nemo_evaluator_sdk.values.otlp import final_output_text

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"responses_create_params": {"input": "What is 2 + 2?"}}) + "\n", encoding="utf-8")
    tasks = discover_gym_tasks(dataset)
    index_map = materialize_dataset(tasks, tmp_path / "gym_input.jsonl")
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps(
            {
                NG_TASK_INDEX: 0,
                NG_ROLLOUT_INDEX: 0,
                "reward": 1.0,
                "responses_create_params": {"input": "What is 2 + 2?"},
                "response": {
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                    "output": [
                        {"type": "function_call", "call_id": "c1", "name": "calculator", "arguments": "{}"},
                        {"type": "function_call_output", "call_id": "c1", "output": "4"},
                        {"type": "message", "content": [{"type": "output_text", "text": "The answer is 4."}]},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trial = trials_from_rollouts(bundle, tasks, index_map)[0]

    assert trial.evidence is not None
    handle = await trial.evidence.trace(EVIDENCE_TRACE, format=EVIDENCE_FORMAT_OTLP)
    resource_spans = await handle.resource_spans()
    # The agent's answer, not the tool's "4": only AGENT/CHAIN/LLM spans are eligible.
    assert final_output_text(resource_spans) == "The answer is 4."
    kinds = [
        attribute.value.string_value
        for resource_span in resource_spans
        for scope_spans in resource_span.scope_spans
        for span in scope_spans.spans
        for attribute in span.attributes
        if attribute.key == "openinference.span.kind"
    ]
    assert kinds == ["AGENT", "LLM", "TOOL"]


def test_success_records_without_rollout_index_get_distinct_ids(tmp_path: Path) -> None:
    # Mirror of the failure-path guard: two successful attempts for one task lacking _ng_rollout_index
    # must not collapse to a single trial id.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, "reward": 1.0}) + "\n" + json.dumps({NG_TASK_INDEX: 0, "reward": 0.0}) + "\n",
        encoding="utf-8",
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)
    assert len(trials) == 2
    assert len({trial.id for trial in trials}) == 2  # distinct ids despite missing rollout index
    assert all(trial.task_id == ordered[0] for trial in trials)


@pytest.mark.asyncio
async def test_indexless_rollouts_of_one_task_get_distinct_trace_ids(tmp_path: Path) -> None:
    # Span identity is derived, and Intake's spans table replaces on it, so two attempts sharing
    # an id would store only the last. Gym does not always record _ng_rollout_index, which leaves
    # the trial id to carry the distinction.
    tasks = discover_gym_tasks(EXAMPLE)
    _, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, "reward": 1.0, "response": "first"})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 0, "reward": 0.0, "response": "second"})
        + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    roots = []
    for trial in trials:
        assert trial.evidence is not None
        spans = await (await trial.evidence.trace("trace", format="otlp")).resource_spans()
        roots.append(spans[0].scope_spans[0].spans[0])

    assert len(roots) == 2
    assert len({root.trace_id for root in roots}) == 2
    assert len({root.span_id for root in roots}) == 2


@pytest.mark.asyncio
async def test_retried_rollout_gets_ids_distinct_from_its_first_attempt(tmp_path: Path) -> None:
    # A retry reuses its task and rollout indices, so only _ng_attempt_index separates it from the
    # attempt it replaced. Without it both land on one trial id — and since trace and span ids are
    # derived from that id, Intake's replace-on-identity would keep only the last attempt's evidence.
    tasks = discover_gym_tasks(EXAMPLE)
    _, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, NG_ATTEMPT_INDEX: 0, "reward": 0.0, "response": "first"})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, NG_ATTEMPT_INDEX: 1, "reward": 1.0, "response": "retry"})
        + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    assert len(trials) == 2
    assert len({trial.id for trial in trials}) == 2
    # The suffix rule matches _capture_path, so the trial id and the capture key agree on the attempt.
    assert sorted(trial.id for trial in trials) == sorted([f"{trials[0].task_id}:0", f"{trials[0].task_id}:0-a1"])

    roots = []
    for trial in trials:
        assert trial.evidence is not None
        spans = await (await trial.evidence.trace("trace", format="otlp")).resource_spans()
        roots.append(spans[0].scope_spans[0].spans[0])
    assert len({root.trace_id for root in roots}) == 2
    assert len({root.span_id for root in roots}) == 2


def test_retry_in_rollouts_does_not_collide_with_its_failed_attempt(tmp_path: Path) -> None:
    # The two loops share an id space: Gym writes the abandoned attempt to the failures sidecar and
    # the retry to rollouts, both under the same task and rollout indices.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, NG_ATTEMPT_INDEX: 1, "reward": 1.0}) + "\n",
        encoding="utf-8",
    )
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, NG_ATTEMPT_INDEX: 0, "error": "boom"}) + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    assert len(trials) == 2
    assert len({trial.id for trial in trials}) == 2
    assert all(trial.task_id == ordered[0] for trial in trials)


def test_indexless_failures_get_distinct_trial_ids(tmp_path: Path) -> None:
    # Kill-shaped failures can lack _ng_rollout_index; two for one task must not collapse to one id.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text("", encoding="utf-8")  # no successes
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(
        json.dumps({NG_TASK_INDEX: 1, "error": "oom"})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 1, "error": "sigterm"})
        + "\n",
        encoding="utf-8",
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)
    assert len(trials) == 2
    assert len({trial.id for trial in trials}) == 2  # distinct ids despite missing rollout index
    assert all(trial.status == AgentEvalTrialStatus.FAILED for trial in trials)
    assert all(trial.task_id == ordered[1] for trial in trials)


def test_malformed_line_in_failures_sidecar_is_skipped(tmp_path: Path) -> None:
    # A truncated line in the abnormal-termination sidecar must not sink the good trials.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0}) + "\n", encoding="utf-8")
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(
        json.dumps({NG_TASK_INDEX: 1, NG_ROLLOUT_INDEX: 0, "error": "boom"}) + "\n{ truncated\n", encoding="utf-8"
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)  # must not raise
    statuses = Counter(trial.status for trial in trials)
    assert statuses[AgentEvalTrialStatus.COMPLETED] == 1
    assert statuses[AgentEvalTrialStatus.FAILED] == 1  # the valid failure survived; the bad line was skipped


def test_malformed_reward_is_recorded_unscored_not_crashing(tmp_path: Path) -> None:
    # A non-numeric reward must not sink the whole batch; the trial is recorded PARTIAL / unscored.
    tasks = discover_gym_tasks(EXAMPLE)
    ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": "not-a-number"}) + "\n", encoding="utf-8"
    )
    trials = trials_from_rollouts(bundle, tasks, index_map)
    assert len(trials) == 1
    assert trials[0].status == AgentEvalTrialStatus.PARTIAL
    assert trials[0].metadata["reward"] is None


def test_trials_from_real_captured_bundle(tmp_path: Path) -> None:
    # End-to-end join against the real mcqa rollout bundle (--limit 2 --num-repeats 2 → 4 records).
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    trials = trials_from_rollouts(ROLLOUTS, tasks, index_map)
    records = _rows(ROLLOUTS)
    assert len(trials) == len(records)  # one trial per rollout record
    assert set(Counter(trial.task_id for trial in trials).values()) == {2}  # 2 attempts per task
    assert all(trial.task_id in {task.id for task in tasks} for trial in trials)
    assert all(trial.metadata["reward"] is not None for trial in trials)  # rewards surfaced


@pytest.mark.asyncio
async def test_pump_stream_writes_file_and_mirrors_to_logger(tmp_path: Path, caplog) -> None:
    # Subprocess output must land on disk in full (the durable record) while the logger mirror lets a
    # caller surface it in their terminal through ordinary logging config. Chunked reads mean a line
    # split across reads must still be emitted once, whole.
    stream = asyncio.StreamReader()
    stream.feed_data(b"first line\nsecond ")
    stream.feed_data(b"line\nno trailing newline")
    stream.feed_eof()
    log_path = tmp_path / "gym_eval.stdout.log"
    tails: dict[str, deque] = {}
    with caplog.at_level(logging.DEBUG, logger="nemo_evaluator_sdk.agent_eval.runtimes.gym.process"):
        await pump_stream(stream, log_path, label="gym eval", tails=tails, key="stdout")
    assert log_path.read_text(encoding="utf-8") == "first line\nsecond line\nno trailing newline"
    assert list(tails["stdout"]) == ["first line", "second line", "no trailing newline"]
    assert "second line" in caplog.text  # mirrored at DEBUG, not printed unconditionally


@pytest.mark.asyncio
async def test_pump_stream_tail_is_bounded(tmp_path: Path) -> None:
    # The in-memory tail exists only to enrich a failure message; the file is the complete record, so
    # a long collection must not accumulate its whole transcript in memory.
    stream = asyncio.StreamReader()
    stream.feed_data(b"".join(f"line {n}\n".encode() for n in range(LOG_TAIL_LINES * 3)))
    stream.feed_eof()
    tails: dict[str, deque] = {}
    await pump_stream(stream, tmp_path / "out.log", label="gym eval", tails=tails, key="stdout")
    assert len(tails["stdout"]) == LOG_TAIL_LINES
    assert tails["stdout"][-1] == f"line {LOG_TAIL_LINES * 3 - 1}"  # kept the *last* lines
    assert len((tmp_path / "out.log").read_text(encoding="utf-8").splitlines()) == LOG_TAIL_LINES * 3


@pytest.mark.asyncio
async def test_drain_pumps_returns_when_pipe_never_reaches_eof(tmp_path: Path, caplog) -> None:
    # A pump ends only at pipe EOF, which needs every inheritor of the write end to close it. Ray
    # daemonizes gcs_server outside our process group (see terminate), so a survivor can hold the fd
    # open forever. Draining must therefore be bounded: teardown cannot block on a leaked grandchild.
    stream = asyncio.StreamReader()
    stream.feed_data(b"partial output\n")  # data, but deliberately no feed_eof()
    pump = asyncio.create_task(pump_stream(stream, tmp_path / "stuck.log", label="gym env start"))
    await asyncio.sleep(0)  # let the pump consume what's buffered and block on the next read

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(drain_pumps([pump], grace_s=0.05, what="gym env start"), timeout=5)

    assert pump.cancelled() or pump.done()  # abandoned, not awaited forever
    assert "did not finish" in caplog.text  # and the truncation is reported, not silent
    # whatever was read before the stall is still on disk — the file is not lost
    assert "partial output" in (tmp_path / "stuck.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_drain_pumps_awaits_normal_completion(tmp_path: Path, caplog) -> None:
    # The bounded drain must not truncate the common case: a pump that reaches EOF is awaited to
    # completion and produces no warning.
    stream = asyncio.StreamReader()
    stream.feed_data(b"all of it\n")
    stream.feed_eof()
    tails: dict[str, deque] = {}
    pump = asyncio.create_task(pump_stream(stream, tmp_path / "done.log", label="gym eval", tails=tails))
    with caplog.at_level(logging.WARNING):
        await drain_pumps([pump], grace_s=5, what="gym eval")
    assert pump.done() and not pump.cancelled()
    assert list(tails["stdout"]) == ["all of it"]
    assert "did not finish" not in caplog.text


def test_trials_reject_index_map_disagreeing_with_tasks(tmp_path: Path) -> None:
    # Defence in depth: the index map and the task list must describe the same run. If they diverge,
    # trials would be attributed to tasks the caller never asked to score. People make optimization
    # decisions from these numbers, so a disagreement is a hard error, never a best-effort join.
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        trials_from_rollouts(bundle, tasks[1:], index_map)  # map covers a task not in the list


def test_success_record_without_usable_index_is_counted_not_silently_dropped(tmp_path: Path, caplog) -> None:
    # A success record with no usable _ng_task_index can't be attributed, but dropping it silently is
    # the dangerous case: with num_repeats > 1 the task keeps its other trials, so require_full_coverage
    # won't fire and the task is simply scored on fewer attempts than were actually run. Count + warn,
    # matching what the failures loop already does.
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0})
        + "\n"
        + json.dumps({NG_ROLLOUT_INDEX: 1, "reward": 0.0})  # no _ng_task_index at all
        + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        trials = trials_from_rollouts(bundle, tasks, index_map)
    assert len(trials) == 1  # the unattributable record is still skipped
    assert "1 Gym rollout record(s)" in caplog.text  # ...but reported, with the count
    assert NG_TASK_INDEX in caplog.text
    assert str(bundle) in caplog.text


def test_odd_sidecar_index_is_counted_not_fatal(tmp_path: Path) -> None:
    # The failures sidecar is read tolerantly on purpose: it is written during abnormal termination,
    # so one odd record must not discard every already-parsed success. An unknown index there is
    # unattributable, not evidence of a bad join — and it cannot inflate a score, since failures carry
    # no reward and any task left with no trial still trips require_full_coverage.
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "reward": 1.0}) + "\n", encoding="utf-8")
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(json.dumps({NG_TASK_INDEX: 999, "error": "boom"}) + "\n", encoding="utf-8")

    trials = trials_from_rollouts(bundle, tasks, index_map)  # must not raise
    assert len(trials) == 1  # the good success survived
    assert trials[0].task_id == tasks[0].id


def test_odd_success_index_still_raises(tmp_path: Path) -> None:
    # The mirror of the above: in the *successes* file an unknown index means Gym reindexed the
    # dataset, so every reward attribution is suspect. That must stay fatal.
    tasks = discover_gym_tasks(EXAMPLE)
    _ordered, index_map = _index_map(tasks, tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(json.dumps({NG_TASK_INDEX: 999, NG_ROLLOUT_INDEX: 0, "reward": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        trials_from_rollouts(bundle, tasks, index_map)


def test_incomplete_collection_raises_with_pointers_to_the_evidence(tmp_path: Path) -> None:
    # AgentEvaluator already hard-fails on a task with no trial (evaluator._score_trials), deliberately:
    # an incomplete run must not read as a successful one with lower counts. Synthesizing FAILED trials
    # here would defeat that guard *and* skew the mean, since an unscored trial is excluded from it.
    # So keep failing — but fail from the runner, naming the evidence, instead of leaving the operator
    # with the evaluator's bare task-id list after paying the full collection cost.
    tasks = discover_gym_tasks(EXAMPLE)
    rollouts = tmp_path / "rollouts.jsonl"
    covered = [trial_task_id for trial_task_id in [tasks[0].id]]
    with pytest.raises(RuntimeError) as excinfo:
        require_full_coverage(tasks, covered_task_ids=set(covered), rollouts_path=rollouts)
    message = str(excinfo.value)
    assert str(rollouts) in message  # points at the bundle
    assert "rollouts_failures.jsonl" in message  # ...and the sidecar
    assert str(tmp_path) in message  # ...and the work dir holding gym_env.log / gym_eval.*.log
    assert str(len(tasks) - 1) in message  # says how many tasks went unrepresented


def test_full_coverage_passes_silently(tmp_path: Path) -> None:
    tasks = discover_gym_tasks(EXAMPLE)
    require_full_coverage(tasks, covered_task_ids={t.id for t in tasks}, rollouts_path=tmp_path / "rollouts.jsonl")


def test_reward_metric_shape() -> None:
    metric = GymRewardMetric()
    assert metric.type == "gym_reward"
    assert len(metric.output_spec()) == 1


@pytest.mark.asyncio
async def test_reward_metric_missing_reward_is_unscored_not_zero() -> None:
    metric = GymRewardMetric()
    scored = await metric.compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata={"reward": 1.0}))
    )
    assert scored.outputs[0].value == 1.0
    # A missing reward must be unscored (None -> nan, excluded from the mean), not a spurious 0.0.
    unscored = await metric.compute_scores(MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata={})))
    assert unscored.outputs[0].value is None


def test_real_fixture_carries_the_schema_the_runner_reads() -> None:
    # Contract guard: the captured mcqa rollout bundle has the fields the runner depends on.
    records = _rows(ROLLOUTS)
    assert records
    for record in records:
        for key in ("responses_create_params", "response", "reward", "_ng_task_index", "_ng_rollout_index"):
            assert key in record, key


def test_aggregate_metrics_sidecar_with_invalid_utf8_is_skipped_not_raised(tmp_path: Path) -> None:
    # UnicodeDecodeError is a ValueError, not an OSError, so it slipped past the parse handler and would
    # raise straight out of run_tasks -- discarding every trial from a collection that already succeeded.
    rollouts_path = tmp_path / "rollouts.jsonl"
    _aggregate_metrics_path_for(rollouts_path).write_bytes(b'[{"agent_ref": {"name": "a"}, "x": \xff}]')

    assert read_run_aggregations(rollouts_path) is None


# ---------------------------------------------------------------------------
# Config pre-flight: override serialization, selection args, `gym env validate`
# ---------------------------------------------------------------------------


def test_hydra_scalars_use_hydra_spellings_not_python_ones() -> None:
    # `str(None)` is "None" and `str(True)` is "True" — both of which Hydra reads back as *strings*,
    # silently setting the literal text instead of a null or a boolean.
    assert hydra_scalar(None) == "null"
    assert hydra_scalar(True) == "true"
    assert hydra_scalar(False) == "false"
    assert hydra_scalar([1, None]) == "[1,null]"
    assert hydra_scalar(0.7) == "0.7"


def test_hydra_strings_are_quoted_so_the_grammar_cannot_retype_them() -> None:
    # Hydra's grammar is typed: unquoted, "true" parses as a bool, "null" as None, "1.5" as a float,
    # and "a,b" as a *sweep* — so a string override would silently become something else.
    assert hydra_scalar("true") == "'true'"
    assert hydra_scalar("null") == "'null'"
    assert hydra_scalar("1.5") == "'1.5'"
    assert hydra_scalar("a,b") == "'a,b'"
    assert hydra_scalar("A[B") == "'A[B'"  # unquoted this does not parse at all
    assert hydra_scalar("") == "''"
    # Only the quote is escaped: Hydra does not decode `\\` inside a quoted value, so escaping
    # backslashes would double them.
    assert hydra_scalar("he'llo") == "'he\\'llo'"
    assert hydra_scalar("back\\slash") == "'back\\slash'"
    # Interpolation survives quoting — the override sets the text, OmegaConf resolves it on read.
    assert hydra_scalar("${policy_base_url}") == "'${policy_base_url}'"


def test_hydra_rejects_a_value_it_cannot_express() -> None:
    # A trailing backslash escapes the closing quote and leaves the value unterminated; there is no
    # spelling that avoids it, so say so rather than emit something unparseable.
    with pytest.raises(ValueError, match="ends with a backslash"):
        hydra_scalar("ends\\")


def test_hydra_dicts_nested_in_a_list_use_the_grammars_bare_keys() -> None:
    # A mapping reached through a list has no dotted path to flatten onto, so it has to be spelled
    # inline. `str()` would emit Python's repr — `{'b': 1}` — whose quoted keys Hydra's `dictKey`
    # rule has no form for and rejects outright.
    assert hydra_scalar({"b": 1}) == "{b:1}"
    assert hydra_scalar([{"b": 1}, {"c": "true"}]) == "[{b:1},{c:'true'}]"
    # The typed spellings hold at depth: values recurse through the same function.
    assert hydra_scalar({"b": None, "c": True, "d": "a,b"}) == "{b:null,c:true,d:'a,b'}"
    assert hydra_scalar({"b": {"c": [1, "x"]}}) == "{b:{c:[1,'x']}}"
    assert hydra_scalar({}) == "{}"


@pytest.mark.parametrize("key", ["true", "NULL", "1.5", "inf", "b:c", "b,c", "b}c", "b'c", "", 1])
def test_hydra_rejects_dict_keys_it_would_retype_or_fail_to_parse(key: object) -> None:
    # Keys are emitted bare because the grammar has no quoted form, which leaves them at the mercy
    # of the lexer: `{true:1}` keys on the boolean True and `{b:c:1}` does not parse. Neither is what
    # the caller wrote, and the second is not even loud, so refuse both.
    with pytest.raises(ValueError, match="dict key"):
        hydra_scalar([{key: 1}])


def test_flatten_overrides_produces_forcing_dotted_paths() -> None:
    flattened = _flatten_overrides({"a": {"b": {"c": 1}}, "d": "x"})
    # `++` not `+`: a bare `+` fails on a key the merged config already defines, which is precisely
    # the case an override exists for.
    assert flattened == ["++a.b.c=1", "++d='x'"]
    assert _flatten_overrides({}) == []


def test_flatten_overrides_keeps_an_empty_mapping_instead_of_dropping_it() -> None:
    # An empty mapping has no leaves to descend to, so recursing emits nothing and the override
    # vanishes — the caller asked to clear `a` and the run would silently keep the config's value.
    assert _flatten_overrides({"a": {}}) == ["++a={}"]


def test_flatten_overrides_serializes_a_list_of_dicts() -> None:
    assert _flatten_overrides({"a": {"b": [{"c": 1}]}}) == ["++a.b=[{c:1}]"]


def _config(**kwargs: object) -> GymRuntimeConfig:
    return GymRuntimeConfig(agent="simple_agent", agent_config="cfg.yaml", resources_server="mcqa", **kwargs)  # type: ignore[arg-type]


def test_selection_binds_the_resources_server_by_default(tmp_path: Path) -> None:
    selection = selection_args(_config(), tmp_path)
    assert "--resources-server" in selection and "mcqa" in selection
    assert "+simple_agent.responses_api_agents.simple_agent.resources_server.name=mcqa" in selection


def test_selection_omits_the_binding_when_the_caller_binds_it_themselves(tmp_path: Path) -> None:
    # gdpval registers its server as `gdpval_resources_server`, so the automatic binding is wrong for
    # it and the caller supplies their own. Emitting both would leave the config ambiguous.
    selection = selection_args(
        _config(
            bind_resources_server=False,
            hydra_params={
                "simple_agent": {"responses_api_agents": {"simple_agent": {"resources_server": {"name": "other"}}}}
            },
        ),
        tmp_path,
    )
    assert not [arg for arg in selection if arg.startswith("+simple_agent.")]
    assert "++simple_agent.responses_api_agents.simple_agent.resources_server.name='other'" in selection


@pytest.mark.parametrize(
    "awkward",
    [
        pytest.param("run with spaces", id="spaces"),
        pytest.param("run,with,commas", id="commas"),
        pytest.param("run[1]", id="brackets"),
    ],
)
def test_selection_quotes_a_work_dir_containing_hydra_grammar(tmp_path: Path, awkward: str) -> None:
    # Raised by review on #1295. `hydra.run.dir` is a path, and Hydra's grammar reads `,` as a sweep
    # separator and `[` as a list opener. Checked against Hydra's own parser: unquoted, `/tmp/a,b`
    # comes back as a ChoiceSweep and `/tmp/x[1]` raises OverrideParseException. Quoting round-trips
    # all three, so the value has to go through hydra_scalar like every other override.
    work_dir = tmp_path / awkward
    arg = next(a for a in selection_args(_config(), work_dir) if a.startswith("hydra.run.dir="))
    value = arg.removeprefix("hydra.run.dir=")
    assert value == hydra_scalar(str(work_dir / "gym_hydra"))
    assert value.startswith("'") and value.endswith("'")


def test_selection_redirects_hydra_output_under_the_run_work_dir(tmp_path: Path) -> None:
    # Gym writes `outputs/<date>/<time>/` relative to cwd, and the subprocesses inherit ours so Gym
    # can find env.yaml — so without this every run litters the caller's directory.
    selection = selection_args(_config(), tmp_path)
    assert f"hydra.run.dir={hydra_scalar(str(tmp_path / 'gym_hydra'))}" in selection


def _stub_gym(tmp_path: Path, *, exit_code: int, message: str) -> str:
    """A stand-in for the `gym` CLI that records its argv and exits how the test wants."""
    script = tmp_path / "stub-gym"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(tmp_path / 'argv.txt')!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        f"print({message!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


@pytest.mark.asyncio
async def test_validate_config_passes_the_selection_and_logs_the_report(tmp_path: Path) -> None:
    runner = GymAgentTaskRunner(config=_config())
    gym = _stub_gym(tmp_path, exit_code=0, message="Config is valid.")

    await runner._validate_config(gym, ["--resources-server", "mcqa"], {}, tmp_path)

    assert (tmp_path / "argv.txt").read_text().splitlines() == ["env", "validate", "--resources-server", "mcqa"]
    # Kept next to the run's other logs so a passing pre-flight is still auditable afterwards.
    assert (tmp_path / "gym_validate.log").read_text().strip() == "Config is valid."


@pytest.mark.asyncio
async def test_validate_config_raises_with_gyms_own_report(tmp_path: Path) -> None:
    # The whole point of the pre-flight: surface Gym's diagnosis before a Ray cluster and several
    # uvicorn servers start, rather than as a readiness timeout that says nothing about the cause.
    complaint = "Error: references resources_servers/'gdpval', which is not defined"
    runner = GymAgentTaskRunner(config=_config())
    gym = _stub_gym(tmp_path, exit_code=1, message=complaint)

    with pytest.raises(RuntimeError) as excinfo:
        await runner._validate_config(gym, [], {}, tmp_path)

    assert complaint in str(excinfo.value)
    assert "mcqa" in str(excinfo.value)


def test_invocation_env_inherits_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Gym honours ordinary variables a caller expects to carry over — proxies, HF_TOKEN — so the
    # parent environment is the base layer rather than being replaced.
    monkeypatch.setenv("A485_INHERITED", "from-parent")
    assert gym_invocation_env(_config())["A485_INHERITED"] == "from-parent"


def test_invocation_env_disables_rays_uv_hook_by_default() -> None:
    # Gym launches each server from its own subdir with its own .venv; Ray's uv-project replication
    # asserts the driver's cwd holds the pyproject and aborts startup.
    assert gym_invocation_env(_config())["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_env_vars_win_over_the_inherited_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # The point of the field: naming a variable in the run config makes it a property of the
    # environment being run, not of whoever launched the runner.
    monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", "/whatever-the-launcher-had")
    env = gym_invocation_env(_config(env_vars={"WMT_TRANSLATION_COMET_PY_CACHE": "/from-the-run-config"}))
    assert env["WMT_TRANSLATION_COMET_PY_CACHE"] == "/from-the-run-config"


def test_env_vars_can_restore_rays_uv_hook() -> None:
    # The Ray default exists to make Gym work, not as an invariant — someone debugging that hook
    # needs a way to put it back, so an explicit value has to outrank it.
    env = gym_invocation_env(_config(env_vars={"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "1"}))
    assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "1"


def test_gym_executable_reports_how_to_install_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="own environment"):
        gym_executable()


#: Verbatim from a real `legal_agent_bench` startup timeout (AALGO-485 coverage sweep). Gym polls
#: repeatedly, so the *last* line is the live one — earlier lines name servers that have since come
#: up, and reporting those would send the reader after servers that are fine.
_REAL_ENV_LOG = """\
Waiting for servers to spin up
Checking for HTTP server statuses.
0 / 4 servers ready. Waiting for servers to spin up: ['simple_agent', 'legal_agent_bench', \
'legal_agent_bench_harbor_agent', 'policy_model']
2 / 4 servers ready. Waiting for servers to spin up: ['legal_agent_bench', 'policy_model']
3 / 4 servers ready. Waiting for servers to spin up: ['legal_agent_bench']
"""


#: Verbatim shape of a `legal_agent_bench` rollout whose Harbor agent died before running: Gym
#: recorded a normal-looking rollout with reward 0.0 and wrote nothing to the failures sidecar.
_EMPTY_RESPONSE = {
    "id": "resp_f63c4c78",
    "model": "stub-model",
    "object": "response",
    "output": [],
    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
}
#: The same field for a rollout where the model really answered (mcqa).
_REAL_RESPONSE = {
    "id": "resp_aaa",
    "object": "response",
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "\\boxed{B}"}]}],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}


def _two_tasks_and_map(tmp_path: Path):
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        json.dumps({"responses_create_params": {"input": "first question"}})
        + "\n"
        + json.dumps({"responses_create_params": {"input": "second question"}})
        + "\n",
        encoding="utf-8",
    )
    tasks = discover_gym_tasks(dataset)
    return tasks, materialize_dataset(tasks, tmp_path / "gym_input.jsonl")


def test_token_usage_keys_match_the_openai_schemas_they_mirror() -> None:
    """The keys are *wire-format* field names, cross-checked here against their source of truth.

    Raised in review of #1295: could these come from Gym directly? No — this runtime never imports
    `nemo_gym` (it shells out to the CLI, and nemo-platform excludes Ray by constraint), and the
    names are not Gym's anyway. They are OpenAI's, and `openai` *is* a dependency here.

    Deliberately a test rather than deriving the tuple at import time. What we match is the JSON a
    Gym rollout record carries, which is stable regardless of how the `openai` package organises its
    Python classes — importing `openai.types.responses.response_usage` into the runtime to obtain
    five string literals would trade a stable thing for a fragile one, and break the whole runner if
    that module ever moves. This way a genuinely new token field fails one test instead.
    """
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses.response_usage import ResponseUsage

    upstream = {field for field in CompletionUsage.model_fields if field.endswith("_tokens")} | {
        field for field in ResponseUsage.model_fields if field.endswith("_tokens")
    }
    assert set(_TOKEN_USAGE_KEYS) == upstream, (
        "the token-count keys have drifted from the OpenAI usage schemas they mirror; a field added "
        "upstream means a real model call could be reported in a key this runtime does not read"
    )
    # The input-side subset is a semantic judgement neither schema encodes: these are the counts a
    # call cannot avoid spending, because it always sends a prompt.
    assert set(_INPUT_TOKEN_KEYS) < set(_TOKEN_USAGE_KEYS)
    assert set(_INPUT_TOKEN_KEYS) == {"total_tokens", "input_tokens", "prompt_tokens"}


def test_agent_never_ran_detects_the_empty_zero_token_rollout() -> None:
    assert _agent_never_ran({"response": _EMPTY_RESPONSE, "reward": 0.0}) is True


def test_agent_never_ran_is_false_when_the_model_actually_answered() -> None:
    assert _agent_never_ran({"response": _REAL_RESPONSE, "reward": 1.0}) is False


def test_agent_never_ran_is_false_for_a_legitimately_empty_answer() -> None:
    # A model may answer with nothing and earn 0.0. Tokens were still spent, so it ran — treating
    # this as a failure would reclassify real (bad) results as broken ones.
    spent = {"output": [], "usage": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120}}
    assert _agent_never_ran({"response": spent, "reward": 0.0}) is False


@pytest.mark.parametrize(
    "output",
    [
        pytest.param("the agent answered in plain text", id="string"),
        pytest.param({"text": "the agent answered"}, id="mapping"),
        pytest.param([{"type": "message"}], id="list"),
    ],
)
def test_agent_never_ran_is_false_for_any_non_empty_output_shape(output: object) -> None:
    # `output` is not contractually a list. An earlier version special-cased lists, so a plain-string
    # or single-mapping answer with zero reported tokens was misread as "the agent never ran" — which
    # fails a trial that ran fine, and fails the whole run when every trial looks like that.
    zero = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert _agent_never_ran({"response": {"output": output, "usage": zero}}) is False


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # Chat Completions vocabulary: 12 prompt tokens prove the model was called, even though the
        # Responses-API key names are absent. Reading only one vocabulary made this read as zero.
        pytest.param({"prompt_tokens": 12, "completion_tokens": 0}, False, id="chat-vocab-nonzero"),
        pytest.param({"prompt_tokens": 0, "completion_tokens": 0}, True, id="chat-vocab-zero"),
        pytest.param({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, True, id="responses-vocab-zero"),
        # Naming none of the recognized keys proves nothing either way, so do not reclassify.
        pytest.param({}, False, id="empty-usage-proves-nothing"),
        pytest.param({"cost_usd": 0.0}, False, id="unrecognized-keys-only"),
        # Output-side counts alone prove nothing: zero output tokens is what an empty answer looks
        # like. Only an input-side count can show the model was never reached, because a call always
        # sends a prompt. These three shapes each used to be misread as "never ran".
        pytest.param({"output_tokens": 0}, False, id="output-side-only"),
        pytest.param({"completion_tokens": 0}, False, id="completion-only"),
        pytest.param({"prompt_tokens": 12, "completion_tokens": 0}, False, id="real-call-empty-answer"),
    ],
)
def test_agent_never_ran_requires_stated_zero_usage(usage: dict, expected: bool) -> None:
    assert _agent_never_ran({"response": {"output": [], "usage": usage}}) is expected


def test_agent_never_ran_is_false_without_usage_to_corroborate() -> None:
    # No usage block means we cannot tell "never called" from "answered with nothing"; other runners
    # populate `response` differently and must not be reclassified on a guess.
    assert _agent_never_ran({"response": {"output": []}, "reward": 0.0}) is False
    assert _agent_never_ran({"response": "Lyon", "reward": 0.0}) is False


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}},
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cache_read_tokens": 3},
        ),
        (
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_creation_input_tokens": 1,
                "cache_read_input_tokens": 4,
            },
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cache_creation_tokens": 1,
                "cache_read_tokens": 4,
            },
        ),
        ({"input_tokens": 0, "output_tokens": 0}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        ({"input_tokens": True, "output_tokens": "2"}, {}),
    ],
)
def test_gym_usage_is_captured_as_typed_measurements(usage: dict, expected: dict) -> None:
    measurements = _usage_measurements({"response": {"usage": usage}})

    assert measurements.model_dump(exclude_none=True) == expected


def test_gym_usage_prefers_chat_aliases() -> None:
    measurements = _usage_measurements(
        {
            "response": {
                "usage": {
                    "prompt_tokens": 10,
                    "input_tokens": 1,
                    "completion_tokens": 2,
                    "output_tokens": 9,
                    "total_tokens": 999,
                }
            }
        }
    )

    assert measurements == TrialMeasurements(prompt_tokens=10, completion_tokens=2)


def test_gym_usage_measurements_empty_when_response_or_usage_is_unusable() -> None:
    assert _usage_measurements({}) == TrialMeasurements()
    assert _usage_measurements({"response": "Lyon"}) == TrialMeasurements()
    assert _usage_measurements({"response": {"usage": None}}) == TrialMeasurements()


def test_gym_usage_omits_ambiguous_prompt_cache(caplog: pytest.LogCaptureFixture) -> None:
    measurements = _usage_measurements(
        {
            NG_TASK_INDEX: 0,
            NG_ROLLOUT_INDEX: 1,
            "response": {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 2},
                    "cache_read_input_tokens": 2,
                }
            },
        }
    )

    assert measurements == TrialMeasurements(completion_tokens=4)
    assert "both inclusive cache details and separate cache fields" in caplog.text


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": None},
                "cache_read_input_tokens": 4,
            },
            TrialMeasurements(prompt_tokens=9, completion_tokens=2, cache_read_tokens=4),
            id="null-inclusive-placeholder",
        ),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 4},
                "cache_read_input_tokens": None,
            },
            TrialMeasurements(prompt_tokens=5, completion_tokens=2, cache_read_tokens=4),
            id="null-separate-placeholder",
        ),
    ],
)
def test_gym_usage_ignores_null_cache_format_placeholders(
    usage: dict,
    expected: TrialMeasurements,
) -> None:
    assert _usage_measurements({"response": {"usage": usage}}) == expected


def test_gym_warn_invalid_usage_counts_logs_malformed_fields(caplog: pytest.LogCaptureFixture) -> None:
    _warn_invalid_usage_counts(
        {NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 1},
        {"prompt_tokens": True, "prompt_tokens_details": {"cached_tokens": -1}},
    )

    assert "prompt_tokens=True" in caplog.text
    assert "prompt_tokens_details.cached_tokens=-1" in caplog.text


def test_missing_results_file_names_the_logs_instead_of_raising_filenotfound(tmp_path: Path) -> None:
    # Raised by review on #1295. `ensure_fresh_output` deletes any stale file before the run and
    # `_collect_rollouts` already raised on a non-zero exit, so an absent file here means Gym
    # reported success and wrote nothing. Opening it regardless gives a bare FileNotFoundError
    # naming a path the reader has no reason to recognise.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    missing = tmp_path / "rollouts.jsonl"
    assert not missing.exists()

    with pytest.raises(RuntimeError) as excinfo:
        trials_from_rollouts(missing, tasks, index_map)

    message = str(excinfo.value)
    assert "wrote no results" in message
    assert "gym_eval.stdout.log" in message
    assert "gym_env.log" in message


def test_all_rollouts_empty_fails_the_run_instead_of_reporting_zeros(tmp_path: Path) -> None:
    # The legal_agent_bench case: "completed" trials scoring 0.0 that measured nothing at all.
    # Two attempts per task, so trial and task counts differ and the message has to state both —
    # "4 trials" alone reads as four tasks to anyone reasoning in tasks.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        "\n".join(
            json.dumps({NG_TASK_INDEX: task, NG_ROLLOUT_INDEX: attempt, "response": _EMPTY_RESPONSE, "reward": 0.0})
            for task in (0, 1)
            for attempt in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        trials_from_rollouts(bundle, tasks, index_map)

    message = str(excinfo.value)
    assert "never called" in message
    # Counted in both units: 4 attempts across 2 tasks. Reporting only trials invites reading the
    # run as four times bigger than it was.
    assert "4 trial(s)" in message
    assert "2 task(s)" in message
    # Points at the log that carries the environment's own traceback, not just the results file.
    assert "gym_env.log" in message
    # Evaluator's vocabulary in the prose: "rollout" is Gym's word for an attempt, and this message
    # is read by people working in trials and tasks. Paths are exempt — `rollouts.jsonl` is the real
    # filename and has to be quotable verbatim to be useful.
    prose = " ".join(word for word in message.split() if "/" not in word)
    assert "rollout" not in prose.lower()


def test_an_unattributed_failure_prevents_the_all_empty_raise(tmp_path: Path) -> None:
    # Raised in review of #1295: the guard covered unattributed *successes* but not unattributed
    # *failures*. Both are invisible to len(trials) for the same reason — neither becomes a trial —
    # and a failure record is a diagnosis in its own right (an OOM-killed agent here). Claiming
    # "nothing was measured" while the environment recorded one is false.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "response": _EMPTY_RESPONSE, "reward": 0.0}) + "\n",
        encoding="utf-8",
    )
    # No NG_TASK_INDEX, so it is counted but never attributed to a task.
    (tmp_path / "rollouts_failures.jsonl").write_text(
        json.dumps({"error": "agent OOM-killed mid-episode"}) + "\n", encoding="utf-8"
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    assert [trial.status for trial in trials] == [AgentEvalTrialStatus.FAILED]


def test_an_unattributed_rollout_prevents_the_all_empty_raise(tmp_path: Path) -> None:
    # A record Gym collected but that carried no usable _ng_task_index never becomes a trial, so it
    # is invisible to len(trials). It is still a rollout, and it may have run perfectly well —
    # raising "nothing was measured" while one exists would be false.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "response": _EMPTY_RESPONSE, "reward": 0.0})
        + "\n"
        # no NG_TASK_INDEX: unattributable, but a real rollout with real token usage
        + json.dumps({NG_ROLLOUT_INDEX: 0, "response": _REAL_RESPONSE, "reward": 1.0})
        + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    assert [trial.status for trial in trials] == [AgentEvalTrialStatus.FAILED]


def test_startup_timeout_does_not_claim_an_unknown_server_when_gym_named_none(tmp_path: Path) -> None:
    # Gym can print a readiness line with an empty list. Reporting "waiting on: unknown" contradicts
    # the counts in the same sentence; say only what is known.
    message = GymAgentTaskRunner(config=_config())._startup_timeout_message(
        "0 / 4 servers ready. Waiting for servers to spin up: []", tmp_path / "gym_env.log"
    )
    assert "unknown" not in message
    assert "did not name which remained outstanding" in message


def test_empty_rollouts_alongside_a_sidecar_failure_do_not_raise(tmp_path: Path) -> None:
    # A run that also recorded a real failure is not "nothing ran": that trial carries its own
    # diagnosis (here a verifier timeout, from an attempt that may well have reached the model), and
    # raising would discard it in favour of a vaguer message — while also miscounting the run as
    # "All 1 rollout(s)" when two attempts happened.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "response": _EMPTY_RESPONSE, "reward": 0.0}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "rollouts_failures.jsonl").write_text(
        json.dumps({NG_TASK_INDEX: 1, NG_ROLLOUT_INDEX: 0, "error": "verifier timeout"}) + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    assert {trial.task_id for trial in trials} == {tasks[0].id, tasks[1].id}
    assert all(trial.status is AgentEvalTrialStatus.FAILED for trial in trials)
    # The sidecar's own diagnosis survives rather than being replaced by the empty-rollout message.
    sidecar = next(trial for trial in trials if trial.task_id == tasks[1].id)
    assert sidecar.metadata["gym_failure"] == "verifier timeout"


def test_some_rollouts_empty_fails_only_those_trials(tmp_path: Path) -> None:
    # A partially broken run still measured something; the trials that ran are worth keeping.
    tasks, index_map = _two_tasks_and_map(tmp_path)
    bundle = tmp_path / "rollouts.jsonl"
    bundle.write_text(
        json.dumps({NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 0, "response": _EMPTY_RESPONSE, "reward": 0.0})
        + "\n"
        + json.dumps({NG_TASK_INDEX: 1, NG_ROLLOUT_INDEX: 0, "response": _REAL_RESPONSE, "reward": 1.0})
        + "\n",
        encoding="utf-8",
    )

    trials = trials_from_rollouts(bundle, tasks, index_map)

    by_task = {trial.task_id: trial for trial in trials}
    empty = by_task[tasks[0].id]
    real = by_task[tasks[1].id]
    assert empty.status is AgentEvalTrialStatus.FAILED
    assert empty.measurements == TrialMeasurements(prompt_tokens=0, completion_tokens=0)
    # The reward Gym reported is withheld so it cannot be averaged in as a real 0.0...
    assert empty.metadata["reward"] is None
    # ...but the reason is recorded.
    assert "never called" in empty.metadata["gym_failure"]
    assert real.status is AgentEvalTrialStatus.COMPLETED
    assert real.measurements == TrialMeasurements(prompt_tokens=1, completion_tokens=1)
    assert real.metadata["reward"] == 1.0


def test_pending_servers_reports_the_last_poll_not_the_first() -> None:
    assert pending_servers(_REAL_ENV_LOG) == (3, 4, ("legal_agent_bench",))


def test_pending_servers_is_none_when_gym_never_reported_readiness() -> None:
    # Gym died during composition or dependency install; there is no pending set to report, which
    # must not be confused with "nothing pending".
    assert pending_servers("NeMo Gym is starting a new Ray cluster...\nerror: No solution found\n") is None


def test_startup_timeout_names_the_server_that_did_not_come_up(tmp_path: Path) -> None:
    # The bare "servers not ready" message sent the reader to a log dominated by the three servers
    # that started fine. Gym knew which one was outstanding the whole time.
    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent="simple_agent",
            agent_config="cfg.yaml",
            resources_server="legal_agent_bench",
            startup_timeout_s=240.0,
        )
    )

    message = runner._startup_timeout_message(_REAL_ENV_LOG, tmp_path / "gym_env.log")

    assert "legal_agent_bench" in message
    assert "3 of 4" in message
    # The knob to change, not just the fact that a limit was hit.
    assert "startup_timeout_s" in message
    assert str(tmp_path / "gym_env.log") in message


def test_startup_timeout_says_so_when_gym_never_reported_readiness(tmp_path: Path) -> None:
    message = GymAgentTaskRunner(config=_config())._startup_timeout_message(
        "NeMo Gym is starting a new Ray cluster...\n", tmp_path / "gym_env.log"
    )
    assert "never reported server readiness" in message
    # Still names the environment, which is the one thing the reader always needs.
    assert "mcqa" in message


@pytest.mark.asyncio
async def test_collection_failure_points_at_the_server_log_too(tmp_path: Path) -> None:
    # Observed on wmt_translation: `gym eval run` reports a bare HTTP 500 from the resources-server
    # while the traceback explaining it (PermissionError on a hardcoded /opt/Gym path) is only in
    # gym_env.log. Naming just the two eval logs sends the reader to the one place the cause is not.
    runner = GymAgentTaskRunner(config=_config())
    gym = _stub_gym(tmp_path, exit_code=1, message="ClientResponseError: 500, Internal Server Error")

    with pytest.raises(RuntimeError) as excinfo:
        await runner._collect_rollouts(gym, tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path, {})

    message = str(excinfo.value)
    assert str(tmp_path / "gym_env.log") in message
    assert str(tmp_path / "gym_eval.stdout.log") in message


@pytest.mark.asyncio
async def test_collection_redirects_hydra_output_under_the_run_work_dir(tmp_path: Path) -> None:
    # `gym eval run` does not go through selection_args, so it needs its own redirect — without it
    # every collection drops an `outputs/<date>/<time>/` into the caller's cwd.
    runner = GymAgentTaskRunner(config=_config())
    gym = _stub_gym(tmp_path, exit_code=0, message="done")

    await runner._collect_rollouts(gym, tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path, {})

    argv = (tmp_path / "argv.txt").read_text().splitlines()
    assert f"hydra.run.dir={hydra_scalar(str(tmp_path / 'gym_hydra'))}" in argv


def test_leftover_captures_refuse_a_second_run_into_the_same_dir(tmp_path: Path) -> None:
    # The sandboxed runner writes captures *before* `rollouts.jsonl`, so a run that died in between
    # leaves them with no file above to catch it. Capture names repeat across runs of one dataset,
    # so without this the next run attaches a previous run's timing to its own trials.
    captures = model_call_capture_dir(tmp_path)
    captures.mkdir(parents=True)
    (captures / "0-0.capture.jsonl").write_text('{"model_call_id": "old"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="already holds Gym rollout output"):
        ensure_fresh_output(tmp_path / "rollouts.jsonl")


def test_an_empty_capture_directory_does_not_refuse_a_run(tmp_path: Path) -> None:
    # The host creates the directory whether or not Gym writes into it, so its mere existence is
    # not evidence of a previous run.
    model_call_capture_dir(tmp_path).mkdir(parents=True)

    ensure_fresh_output(tmp_path / "rollouts.jsonl")
