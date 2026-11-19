# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Length budget for SFT rows.

Training does not truncate chat rows. Packing drops any row longer than the packed sequence size, one row
at a time and without a word of it in the job output. With one row per assistant turn that skews the
dataset: a chain's rows grow as the conversation does, so the late turns go first. This module applies the
limit up front, by a stated policy, and counts what it did.

A *chain* is the rows of one conversation, in turn order. Each row extends the one before it, so rendered
length never decreases along a chain (a chat template that strips earlier turns' reasoning can break that
by a few tokens). The policies lean on that to measure a chain with a handful of renders instead of one per
row. When the assumption is off by a few tokens the failure is mild: the packer drops the stray row, as it
would have anyway.

Behaviors, for a row over ``tokens``:

- ``drop-chain``: drop every row of its chain. The chain is judged by its last row, its longest.
- ``drop-longest``: drop each row that is over the limit, keep the shorter rows before it.
- ``truncate-earlier-turns``: keep the row but cut its oldest middle turns (assistant turn plus its tool
  results) until it fits. The system prompt, the task prompt and the turn being trained on always stay. A
  row that still does not fit with the middle gone is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

Behavior = Literal["drop-chain", "truncate-earlier-turns", "drop-longest"]
Row = dict[str, Any]


class MaxTokens(BaseModel):
    """Length limit for the rows of the SFT dataset.

    ``tokens`` takes an integer or a ``k`` suffix, where ``32k`` is 32768. Counts are the rendered chat
    template length under the student's tokenizer. Training adds a few tokens of its own, so leave a small
    margin below the training sequence length.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Turn the limit off without removing the block.")
    behavior: Behavior = Field(default="drop-chain", description="What to do with a row over the limit.")
    tokens: int = Field(gt=0, description="Largest allowed row, in tokens.")

    @field_validator("tokens", mode="before")
    @classmethod
    def _parse_k_suffix(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            if text.endswith("k") and text[:-1].isdigit():
                return int(text[:-1]) * 1024
            if text.isdigit():
                return int(text)
            raise ValueError(f"tokens {value!r} is not an integer or a number with a k suffix, like 32k")
        return value


class TokenCounter(Protocol):
    """Rendered length of a row under the student's chat template."""

    def count(self, messages: list[Row], tools: list[Row] | None) -> int: ...


class ChatTemplateCounter:
    """Counts tokens by rendering the tokenizer's chat template, as training does."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, messages: list[Row], tools: list[Row] | None) -> int:
        text = self._tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=False
        )
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class BudgetOutcome:
    """Chains that survive the budget, and what it cost."""

    chains: list[list[Row]]
    dropped_chains: int = 0
    dropped_rows: int = 0
    truncated_rows: int = 0
    render_errors: int = 0


def apply_budget(chains: list[list[Row]], budget: MaxTokens, counter: TokenCounter) -> BudgetOutcome:
    """Apply the budget to every chain. A disabled budget returns the chains untouched."""
    if not budget.enabled:
        return BudgetOutcome(chains=chains)
    outcome = BudgetOutcome(chains=[])
    for chain in chains:
        cost = BudgetOutcome(chains=[])  # per chain, merged only if the chain was measured completely
        try:
            kept = _fit_chain(chain, budget, counter, cost)
        except Exception:  # noqa: BLE001 - a template can raise anything; one bad chain must not end the job
            logger.warning("could not measure a chain; dropping it", exc_info=True)
            outcome.render_errors += 1
            outcome.dropped_chains += 1  # counted as a chain, not as rows over the limit: its length is unknown
            continue
        outcome.dropped_rows += cost.dropped_rows
        outcome.truncated_rows += cost.truncated_rows
        if kept:
            outcome.chains.append(kept)
        else:
            outcome.dropped_chains += 1
    return outcome


def _fit_chain(chain: list[Row], budget: MaxTokens, counter: TokenCounter, outcome: BudgetOutcome) -> list[Row]:
    limit = budget.tokens

    def length(row: Row) -> int:
        return counter.count(row["messages"], row.get("tools"))

    if length(chain[-1]) <= limit:
        return chain  # the longest row fits, so they all do
    if budget.behavior == "drop-chain":
        outcome.dropped_rows += len(chain)
        return []

    # Lengths do not decrease along a chain, so the rows over the limit are a suffix: find where it starts.
    low, high = 0, len(chain) - 1  # chain[high] is over the limit
    while low < high:
        middle = (low + high) // 2
        if length(chain[middle]) > limit:
            high = middle
        else:
            low = middle + 1
    kept = chain[:low]
    over = chain[low:]
    if budget.behavior == "drop-longest":
        outcome.dropped_rows += len(over)
        return kept
    for row in over:  # truncate-earlier-turns
        trimmed = _trim_earlier_turns(row, limit, length)
        if trimmed is None:
            outcome.dropped_rows += 1
        else:
            outcome.truncated_rows += 1
            kept.append(trimmed)
    return kept


def _trim_earlier_turns(row: Row, limit: int, length: Any) -> Row | None:
    """The row with its oldest middle turns removed until it fits, or ``None`` if nothing makes it fit."""
    messages = row["messages"]
    head_end = _head_end(messages)
    target_start = _target_start(messages)
    units = _units(messages[head_end:target_start])
    if not units:
        return None

    def without(count: int) -> Row:
        kept = [message for unit in units[count:] for message in unit]
        return {**row, "messages": [*messages[:head_end], *kept, *messages[target_start:]]}

    if length(without(len(units))) > limit:
        return None
    low, high = 1, len(units)  # without(high) fits; find the fewest turns that must go
    while low < high:
        middle = (low + high) // 2
        if length(without(middle)) <= limit:
            high = middle
        else:
            low = middle + 1
    return without(low)


def _head_end(messages: list[Row]) -> int:
    """Index after the prompt: the leading system messages and the first user message."""
    for index, message in enumerate(messages):
        if message["role"] == "user":
            return index + 1
    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        index += 1
    return index


def _is_thinking_only(message: Row) -> bool:
    return (
        message["role"] == "assistant"
        and "thinking" in message
        and "content" not in message
        and "tool_calls" not in message
    )


def _target_start(messages: list[Row]) -> int:
    """Index where the turn being trained on starts: its thinking message when it has one, else the reply."""
    start = len(messages) - 1
    if start >= 1 and _is_thinking_only(messages[start - 1]):
        start -= 1
    return start


def _units(middle: list[Row]) -> list[list[Row]]:
    """Whole turns: an assistant turn with its thinking and its tool results, or a mid-chain user message.

    Cutting by unit never separates a tool call from its result.
    """
    units: list[list[Row]] = []
    for message in middle:
        if units and (
            message["role"] == "tool" or (message["role"] == "assistant" and _is_thinking_only(units[-1][-1]))
        ):
            units[-1].append(message)
        else:
            units.append([message])
    return units
