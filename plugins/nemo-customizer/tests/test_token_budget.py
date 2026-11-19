# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from nemo_customizer.token_budget import Behavior, ChatTemplateCounter, MaxTokens, apply_budget


class CharCounter:
    """One token per character of the serialized row, so lengths are easy to reason about."""

    def __init__(self) -> None:
        self.calls = 0

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
        self.calls += 1
        return len(json.dumps(messages)) + len(json.dumps(tools or []))


def turn(
    number: int, size: int = 100, *, thinking: bool = False, result_size: int | None = None
) -> list[dict[str, Any]]:
    """An assistant turn with a tool call, then its tool result. `size` characters of padding for the call."""
    messages: list[dict[str, Any]] = []
    if thinking:
        messages.append({"role": "assistant", "thinking": f"t{number}" + "." * size})
    messages += [
        {
            "role": "assistant",
            "content": f"a{number}" + "." * size,
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"n": number}}}],
        },
        {"role": "tool", "content": f"r{number}" + "." * (size if result_size is None else result_size)},
    ]
    return messages


def chain(
    turns: int, size: int = 100, *, final_thinking: bool = False, result_size: int | None = None
) -> list[dict[str, Any]]:
    """Rows for a conversation of `turns` agent turns: row i trains on turn i with all earlier turns as context."""
    head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    rows = []
    for index in range(turns):
        context = [message for earlier in range(index) for message in turn(earlier, size, result_size=result_size)]
        target = turn(index, size, thinking=final_thinking and index == turns - 1)[:-1]  # no tool result yet
        rows.append(
            {"messages": [*head, *context, *target], "tools": [{"type": "function", "function": {"name": "bash"}}]}
        )
    return rows


def lengths(rows: list[dict[str, Any]]) -> list[int]:
    return [CharCounter().count(row["messages"], row.get("tools")) for row in rows]


def budget(behavior: Behavior, tokens: int) -> MaxTokens:
    return MaxTokens(behavior=behavior, tokens=tokens)


def test_tokens_accept_a_k_suffix_and_plain_numbers() -> None:
    assert MaxTokens(tokens="32k").tokens == 32768
    assert MaxTokens(tokens="32K").tokens == 32768
    assert MaxTokens(tokens="4096").tokens == 4096
    assert MaxTokens(tokens=8192).tokens == 8192
    assert MaxTokens(tokens=1).behavior == "drop-chain"  # the default policy
    for bad in ("many", "3.5k", "-1", 0, -5):
        with pytest.raises(ValueError):
            MaxTokens(tokens=bad)
    with pytest.raises(ValueError):
        MaxTokens.model_validate({"tokens": 10, "behaviour": "drop-chain"})
    with pytest.raises(ValueError):
        MaxTokens.model_validate({"tokens": 10, "behavior": "shrink"})


def test_a_disabled_limit_changes_nothing_and_measures_nothing() -> None:
    counter = CharCounter()
    chains = [chain(5)]
    outcome = apply_budget(chains, MaxTokens(enabled=False, tokens=1), counter)
    assert outcome.chains == chains
    assert counter.calls == 0


@pytest.mark.parametrize("behavior", ["drop-chain", "drop-longest", "truncate-earlier-turns"])
def test_chains_that_fit_pass_through_with_a_single_measurement(behavior: Behavior) -> None:
    counter = CharCounter()
    rows = chain(6)
    outcome = apply_budget([rows], budget(behavior, max(lengths(rows))), counter)
    assert outcome.chains == [rows]
    assert (outcome.dropped_chains, outcome.dropped_rows, outcome.truncated_rows) == (0, 0, 0)
    assert counter.calls == 1  # the longest row fits, so the rest are not measured


def test_drop_chain_drops_every_row_when_the_last_is_over() -> None:
    rows = chain(8)
    limit = lengths(rows)[-1] - 1
    assert sum(length <= limit for length in lengths(rows)) > 1  # earlier rows would have fit

    outcome = apply_budget([rows, chain(2)], budget("drop-chain", limit), CharCounter())

    assert outcome.chains == [chain(2)]  # the short chain survives whole
    assert (outcome.dropped_chains, outcome.dropped_rows, outcome.truncated_rows) == (1, 8, 0)


def test_drop_longest_keeps_the_rows_before_the_limit_and_drops_the_rest() -> None:
    rows = chain(8)
    measured = lengths(rows)
    limit = measured[4]  # rows 0..4 fit, 5..7 do not

    outcome = apply_budget([rows], budget("drop-longest", limit), CharCounter())

    assert outcome.chains == [rows[:5]]
    assert (outcome.dropped_chains, outcome.dropped_rows) == (0, 3)
    assert all(length <= limit for length in lengths(outcome.chains[0]))


def test_drop_longest_with_no_row_that_fits_drops_the_chain() -> None:
    outcome = apply_budget([chain(3)], budget("drop-longest", 10), CharCounter())
    assert outcome.chains == []
    assert (outcome.dropped_chains, outcome.dropped_rows) == (1, 3)


def test_truncate_cuts_the_oldest_middle_turns_and_keeps_the_prompt_and_the_target() -> None:
    rows = chain(8)
    limit = lengths(rows)[4]

    outcome = apply_budget([rows], budget("truncate-earlier-turns", limit), CharCounter())

    [kept] = outcome.chains
    assert len(kept) == 8  # every turn is still a training row
    assert outcome.truncated_rows == 3 and outcome.dropped_rows == 0
    assert kept[:5] == rows[:5]  # rows that fit are untouched
    assert all(length <= limit for length in lengths(kept))
    last = kept[-1]["messages"]
    assert last[:2] == rows[-1]["messages"][:2]  # system and task prompt stay
    assert last[-1] == rows[-1]["messages"][-1]  # the turn being trained on stays
    # The oldest turns went; the most recent context stayed.
    assert "a0" not in json.dumps(last) and "a6" in json.dumps(last)
    assert kept[-1]["tools"] == rows[-1]["tools"]


def test_truncate_removes_the_fewest_turns_that_make_the_row_fit() -> None:
    rows = chain(8)
    limit = lengths(rows)[4]
    outcome = apply_budget([rows], budget("truncate-earlier-turns", limit), CharCounter())
    row = outcome.chains[0][-1]
    # Cutting one fewer turn would not fit: put the oldest remaining turn back and it is over the limit.
    assert lengths([row])[0] <= limit
    restored_turns = [m for m in rows[-1]["messages"] if m not in row["messages"]]
    assert restored_turns  # something was cut
    one_back = {**row, "messages": [*row["messages"][:2], *restored_turns[-2:], *row["messages"][2:]]}
    assert lengths([one_back])[0] > limit


def test_truncate_never_leaves_a_tool_call_without_its_result() -> None:
    rows = chain(10)
    limit = lengths(rows)[3]
    outcome = apply_budget([rows], budget("truncate-earlier-turns", limit), CharCounter())
    for row in outcome.chains[0]:
        middle = row["messages"][2:-1]  # between the prompt and the target
        calls = sum(1 for m in middle if m["role"] == "assistant" and m.get("tool_calls"))
        results = sum(1 for m in middle if m["role"] == "tool")
        assert calls == results


def test_truncate_cuts_a_whole_turn_even_when_cutting_only_its_call_would_fit() -> None:
    # Big tool calls, tiny results: dropping just the oldest call message would be enough to fit, but it
    # would leave that call's result dangling at the head of the context. The turn goes as a unit.
    rows = chain(6, size=400, result_size=1)
    full = lengths(rows)[-1]
    oldest_call = len(json.dumps(rows[-1]["messages"][2]))
    limit = full - oldest_call // 2  # more than the result's size, less than the call's: one message would do

    outcome = apply_budget([rows], budget("truncate-earlier-turns", limit), CharCounter())

    cut_rows = [row for row in outcome.chains[0] if row not in rows]
    assert cut_rows
    for row in cut_rows:
        middle = row["messages"][2:-1]
        assert middle[0]["role"] == "assistant"  # never starts on a tool result whose call was cut
        assert sum(m["role"] == "assistant" for m in middle) == sum(m["role"] == "tool" for m in middle)
    assert all(length <= limit for length in lengths(outcome.chains[0]))


def test_truncate_keeps_a_thinking_message_with_the_target_it_belongs_to() -> None:
    rows = chain(6, final_thinking=True)
    limit = lengths(rows)[3]
    outcome = apply_budget([rows], budget("truncate-earlier-turns", limit), CharCounter())
    last = outcome.chains[0][-1]["messages"]
    assert last[-2]["thinking"].startswith("t5") and last[-1]["content"].startswith("a5")


def test_truncate_drops_a_row_that_cannot_fit_even_with_no_middle() -> None:
    rows = chain(4)
    floor = lengths([{**rows[-1], "messages": [*rows[-1]["messages"][:2], rows[-1]["messages"][-1]]}])[0]
    outcome = apply_budget([rows], budget("truncate-earlier-turns", floor - 1), CharCounter())
    assert outcome.chains == []
    assert outcome.dropped_chains == 1 and outcome.dropped_rows == 4


def test_truncate_with_only_the_target_left_to_cut_drops_the_row() -> None:
    # Row 0 has no middle turns to remove, so an over-long first row is dropped, not truncated.
    rows = chain(3, size=500)
    outcome = apply_budget([rows], budget("truncate-earlier-turns", lengths(rows)[0] - 1), CharCounter())
    assert outcome.chains == []
    assert outcome.dropped_rows == 3


def test_measuring_is_logarithmic_in_chain_length() -> None:
    rows = chain(64, size=20)
    counter = CharCounter()
    apply_budget([rows], budget("drop-longest", lengths(rows)[31]), counter)
    assert counter.calls <= 8  # one for the last row plus about log2(64) probes, not one per row


def test_a_chain_that_cannot_be_measured_is_dropped_and_counted_without_ending_the_run() -> None:
    class Flaky:
        def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
            if any(m.get("content") == "poison" for m in messages):
                raise ValueError("template exploded")
            return 5

    poisoned = chain(2)
    poisoned[-1]["messages"][1] = {"role": "user", "content": "poison"}
    healthy = chain(2)

    outcome = apply_budget([poisoned, healthy], budget("drop-chain", 10), Flaky())

    assert outcome.chains == [healthy]
    assert (outcome.render_errors, outcome.dropped_chains, outcome.dropped_rows) == (1, 1, 0)


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.rendered.append(kwargs)
        return " ".join(str(message.get("content", "")) for message in messages)

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def test_the_chat_template_counter_renders_without_a_generation_prompt_and_counts_tokens() -> None:
    tokenizer = FakeTokenizer()
    counter = ChatTemplateCounter(tokenizer)
    tools = [{"type": "function", "function": {"name": "bash"}}]

    assert (
        counter.count([{"role": "user", "content": "one two"}, {"role": "assistant", "content": "three"}], tools) == 3
    )
    assert tokenizer.rendered == [{"tools": tools, "tokenize": False, "add_generation_prompt": False}]
    counter.count([{"role": "user", "content": "x"}], None)
    assert tokenizer.rendered[-1]["tools"] is None
