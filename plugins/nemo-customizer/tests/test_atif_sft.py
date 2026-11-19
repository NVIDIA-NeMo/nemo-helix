# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nemo_customizer.atif_sft import (
    PassFilter,
    convert_trajectory,
    held_out_tasks,
    join_bundle_path,
    trace_ref,
)
from nhx.automodel.tasks.training.datasets.schemas import SFTPChatDatasetItemSchema

BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


def agent(
    message: str = "", *, calls: list[tuple[str, dict[str, Any], str | None]] | None = None, **extra: Any
) -> dict:
    """An ATIF agent step. Each call is (id, arguments, result); a None result means no observation."""
    step: dict[str, Any] = {"source": "agent", "message": message, **extra}
    if calls:
        step["tool_calls"] = [
            {"tool_call_id": cid, "function_name": "bash", "arguments": args} for cid, args, _ in calls
        ]
        results = [{"source_call_id": cid, "content": out} for cid, _, out in calls if out is not None]
        if results:
            step["observation"] = {"results": results}
    return step


def trajectory(*steps: dict, tools: list[dict] | None = None, **extra: Any) -> dict:
    document: dict[str, Any] = {"schema_version": "ATIF-v1.7", "steps": list(steps), **extra}
    if tools is not None:
        document["agent"] = {"tool_definitions": tools}
    return document


def system(text: str = "You are an agent.") -> dict:
    return {"source": "system", "message": text}


def user(text: str = "Fix the bug.") -> dict:
    return {"source": "user", "message": text}


def test_one_row_per_assistant_turn_with_growing_prefix() -> None:
    converted = convert_trajectory(
        trajectory(
            system(),
            user(),
            agent("look", calls=[("c1", {"command": "ls"}, "a.py")]),
            agent("fix", calls=[("c2", {"command": "sed -i s/a/b/ a.py"}, "ok")]),
            agent("done"),
            tools=[BASH],
        )
    )

    assert converted.skipped == []
    assert len(converted.rows) == 3
    first, second, third = (row["messages"] for row in converted.rows)
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "assistant"]
    assert [m["role"] for m in third] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    # Each row extends the previous one, and its last message is the turn it trains on.
    assert second[: len(first)] == first
    assert third[: len(second)] == second
    assert second[-1]["tool_calls"][0]["function"]["arguments"] == {"command": "sed -i s/a/b/ a.py"}
    assert third[-1] == {"role": "assistant", "content": "done"}
    # The tool result is context for the next turn, never part of the target.
    assert first[-1].get("tool_calls") and all(m["role"] != "tool" for m in first)
    assert second[3] == {"role": "tool", "content": "a.py"}
    assert all(row["tools"] == [BASH] for row in converted.rows)


def test_rows_validate_against_the_sft_chat_schema() -> None:
    converted = convert_trajectory(
        trajectory(
            user(),
            agent("plan", reasoning_content="think first", calls=[("c1", {"command": "ls"}, "a.py")]),
            agent("done"),
            tools=[BASH],
        )
    )
    assert len(converted.rows) == 2
    for row in converted.rows:
        SFTPChatDatasetItemSchema.model_validate(row)


def test_reasoning_is_its_own_assistant_message_before_the_reply() -> None:
    converted = convert_trajectory(trajectory(user(), agent("answer", reasoning_content="because"), tools=[BASH]))
    assert converted.rows[0]["messages"] == [
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "thinking": "because"},
        {"role": "assistant", "content": "answer"},
    ]


def test_tool_call_without_text_omits_empty_content() -> None:
    converted = convert_trajectory(trajectory(user(), agent("", calls=[("c1", {"command": "ls"}, None)]), tools=[BASH]))
    target = converted.rows[0]["messages"][-1]
    assert "content" not in target
    assert target["tool_calls"][0]["function"]["name"] == "bash"


def test_empty_agent_steps_are_not_turns() -> None:
    converted = convert_trajectory(trajectory(user(), agent(""), agent("   "), agent("real")))
    assert len(converted.rows) == 1
    assert converted.rows[0]["messages"][-1]["content"] == "real"


def test_results_are_matched_to_calls_by_id_and_emitted_in_call_order() -> None:
    step = agent(calls=[("a", {"command": "one"}, None), ("b", {"command": "two"}, None)])
    step["observation"] = {
        "results": [{"source_call_id": "b", "content": "second"}, {"source_call_id": "a", "content": "first"}]
    }
    converted = convert_trajectory(trajectory(user(), step, agent("done"), tools=[BASH]))
    tool_messages = [m["content"] for m in converted.rows[1]["messages"] if m["role"] == "tool"]
    assert tool_messages == ["first", "second"]


def test_results_without_ids_pair_by_position_and_unmatched_results_are_dropped() -> None:
    positional = agent(calls=[("a", {"command": "one"}, None)])
    positional["observation"] = {"results": [{"content": "only"}]}
    converted = convert_trajectory(trajectory(user(), positional, agent("done"), tools=[BASH]))
    assert [m["content"] for m in converted.rows[1]["messages"] if m["role"] == "tool"] == ["only"]

    mismatched = agent(calls=[("a", {"command": "one"}, None)])
    mismatched["observation"] = {"results": [{"content": "x"}, {"content": "y"}]}
    converted = convert_trajectory(trajectory(user(), mismatched, agent("done"), tools=[BASH]))
    assert all(m["role"] != "tool" for m in converted.rows[1]["messages"])


def test_non_text_tool_output_is_serialized() -> None:
    step = agent(calls=[("a", {"command": "one"}, None)])
    step["observation"] = {"results": [{"source_call_id": "a", "content": {"exit_code": 0}}]}
    converted = convert_trajectory(trajectory(user(), step, agent("done"), tools=[BASH]))
    assert converted.rows[1]["messages"][2] == {"role": "tool", "content": '{"exit_code": 0}'}


def test_orchestrator_wrapper_is_skipped_but_its_subagent_conversation_converts() -> None:
    # The shape a Fabric agent-eval bundle stores: the top-level step only delegates, and the real
    # prompt and reply live in a subagent trajectory.
    wrapper = trajectory(
        agent(calls=[("c1", {"name": "DeepAgent"}, "{}")]),
        subagent_trajectories=[trajectory(user("Is this phishing?"), agent("VERDICT: phishing"))],
    )
    converted = convert_trajectory(wrapper)
    assert converted.skipped == ["no_prompt"]
    assert [row["messages"] for row in converted.rows] == [
        [{"role": "user", "content": "Is this phishing?"}, {"role": "assistant", "content": "VERDICT: phishing"}]
    ]
    assert all("tools" not in row for row in converted.rows)


def test_calls_without_tool_definitions_are_skipped_not_invented() -> None:
    converted = convert_trajectory(trajectory(user(), agent(calls=[("a", {"command": "ls"}, "x")]), agent("done")))
    assert converted.rows == []
    assert converted.skipped == ["no_tool_definitions"]


def test_subagent_inherits_the_parents_tool_definitions() -> None:
    parent = trajectory(
        user(),
        agent("delegating"),
        tools=[BASH],
        subagent_trajectories=[trajectory(user("sub task"), agent(calls=[("a", {"command": "ls"}, "x")]), agent("ok"))],
    )
    converted = convert_trajectory(parent)
    assert converted.skipped == []
    assert len(converted.rows) == 3
    assert all(row["tools"] == [BASH] for row in converted.rows)


def test_tool_definitions_are_normalized() -> None:
    converted = convert_trajectory(
        trajectory(user(), agent("hi"), tools=[{"name": "ping"}, {"type": "function", "function": BASH["function"]}])
    )
    assert converted.rows[0]["tools"] == [
        {
            "type": "function",
            "function": {"name": "ping", "description": "", "parameters": {"type": "object", "properties": {}}},
        },
        BASH,
    ]
    SFTPChatDatasetItemSchema.model_validate(converted.rows[0])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [({"command": "ls"}, {"command": "ls"}), ('{"command": "ls"}', {"command": "ls"}), (None, {})],
)
def test_tool_call_arguments_become_dicts(arguments: Any, expected: dict) -> None:
    step = agent(calls=[("a", {}, None)])
    step["tool_calls"][0]["arguments"] = arguments
    converted = convert_trajectory(trajectory(user(), step, tools=[BASH]))
    assert converted.rows[0]["messages"][-1]["tool_calls"][0]["function"]["arguments"] == expected


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", 3])
def test_unusable_tool_call_arguments_skip_the_conversation(arguments: Any) -> None:
    step = agent(calls=[("a", {}, None)])
    step["tool_calls"][0]["arguments"] = arguments
    converted = convert_trajectory(trajectory(user(), step, agent("done"), tools=[BASH]))
    assert converted.rows == []
    assert converted.skipped == ["bad_tool_arguments"]


def test_copied_context_is_history_not_a_training_target() -> None:
    converted = convert_trajectory(
        trajectory(user(), agent("earlier", is_copied_context=True), agent("new")),
    )
    assert len(converted.rows) == 1
    assert converted.rows[0]["messages"][-1]["content"] == "new"
    assert converted.rows[0]["messages"][1] == {"role": "assistant", "content": "earlier"}


def test_agent_turn_before_any_prompt_has_nothing_to_condition_on() -> None:
    # The leading agent turn is dropped, so no row starts with an assistant message ahead of its prompt.
    converted = convert_trajectory(trajectory(agent("hello"), user("late prompt"), agent("reply")))
    assert converted.skipped == []
    assert [row["messages"] for row in converted.rows] == [
        [{"role": "user", "content": "late prompt"}, {"role": "assistant", "content": "reply"}]
    ]


@pytest.mark.parametrize(
    ("pass_filter", "scores", "expected"),
    [
        ({}, [], True),
        ({"metric_type": "harbor_reward"}, [], False),
        (
            {"metric_type": "harbor_reward", "output": "reward"},
            [{"status": "completed", "metric_type": "harbor_reward", "outputs": [{"name": "reward", "value": 1.0}]}],
            True,
        ),
        (
            {"metric_type": "harbor_reward", "output": "reward"},
            [{"status": "completed", "metric_type": "harbor_reward", "outputs": [{"name": "reward", "value": 0.0}]}],
            False,
        ),
        (
            {"output": "accuracy", "min_value": 0.5},
            [{"status": "completed", "metric_type": "llm-judge", "outputs": [{"name": "accuracy", "value": 0.75}]}],
            True,
        ),
        # A failed score row, a different metric, or a boolean never counts as passing.
        (
            {"metric_type": "harbor_reward"},
            [{"status": "failed", "metric_type": "harbor_reward", "outputs": [{"name": "reward", "value": 1.0}]}],
            False,
        ),
        (
            {"metric_type": "harbor_reward"},
            [{"status": "completed", "metric_type": "llm-judge", "outputs": [{"name": "reward", "value": 1.0}]}],
            False,
        ),
        (
            {"metric_type": "harbor_reward"},
            [{"status": "completed", "metric_type": "harbor_reward", "outputs": [{"name": "reward", "value": True}]}],
            False,
        ),
        # Every matched value has to clear the bar, not just one.
        (
            {"output": "reward"},
            [
                {"status": "completed", "metric_type": "a", "outputs": [{"name": "reward", "value": 1.0}]},
                {"status": "completed", "metric_type": "b", "outputs": [{"name": "reward", "value": 0.0}]},
            ],
            False,
        ),
    ],
)
def test_pass_filter(pass_filter: dict, scores: list[dict], expected: bool) -> None:
    assert PassFilter.model_validate(pass_filter).passes(scores) is expected


def test_pass_filter_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        PassFilter.model_validate({"metric": "harbor_reward"})


def test_trace_ref_prefers_the_named_atif_trace_and_falls_back() -> None:
    both = {
        "evidence": {
            "descriptors": {
                "relay_atif": {"kind": "atif", "ref": "relay.atif.json"},
                "trace": {"kind": "trace", "format": "otlp", "ref": "relay.jsonl"},
                "trace:atif": {"kind": "trace", "format": "atif", "ref": "named.atif.json"},
            }
        }
    }
    assert trace_ref(both) == "named.atif.json"
    del both["evidence"]["descriptors"]["trace:atif"]
    assert trace_ref(both) == "relay.atif.json"
    assert trace_ref({"evidence": {"descriptors": {"trace": {"kind": "trace", "format": "otlp", "ref": "x"}}}}) is None
    assert trace_ref({}) is None


@pytest.mark.parametrize("ref", ["../other/x.json", "/etc/passwd", "a/../../x.json"])
def test_join_bundle_path_refuses_to_leave_the_bundle(ref: str) -> None:
    with pytest.raises(ValueError, match="outside its bundle"):
        join_bundle_path("results/attempt/agent-eval-results", ref)


def test_join_bundle_path_resolves_inside_the_bundle_and_at_the_fileset_root() -> None:
    assert join_bundle_path("results/a/agent-eval-results", "evidence/t/x.json") == (
        "results/a/agent-eval-results/evidence/t/x.json"
    )
    assert join_bundle_path("", "evidence/t/x.json") == "evidence/t/x.json"
    with pytest.raises(ValueError):
        join_bundle_path("", "../x.json")


def test_held_out_tasks_are_stable_and_leave_training_data() -> None:
    tasks = {f"task-{i}" for i in range(20)}
    held = held_out_tasks(tasks, 0.1)
    assert held == held_out_tasks(set(tasks), 0.1)
    assert len(held) == 2 and held < tasks
    assert len(held_out_tasks({"a", "b"}, 0.9)) == 1  # one task always stays for training
    assert held_out_tasks({"only"}, 0.5) == set()
    assert held_out_tasks(tasks, 0) == set()
    assert len(held_out_tasks({"a", "b", "c"}, 0.01)) == 1  # a positive fraction holds out at least one
