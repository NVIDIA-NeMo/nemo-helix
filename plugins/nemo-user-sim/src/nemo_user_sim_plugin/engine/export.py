# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turns curated conversations into chat-format training records for the Helix post-training plugin."""

import json
from collections.abc import Iterable

import pandas as pd

METADATA_KEYS = ("trajectory_id", "persona_uuid", "probe_type", "locale")
TRAINED_ROLES = ("system", "user", "assistant", "tool")


def apply_review_feedback(curated: pd.DataFrame, feedback: dict[str, str], *, require_review: bool) -> pd.DataFrame:
    """Drop conversations a reviewer gave negative feedback; with ``require_review``, keep only positive ones."""
    verdicts = curated["trajectory_id"].map(feedback)
    keep = verdicts == "positive" if require_review else verdicts != "negative"
    return curated[keep].reset_index(drop=True)


def chat_messages(conversation_messages: str) -> list[dict]:
    """A UserSim row's messages in chat format, without reasoning traces.

    Every message keeps a string ``content``, empty on a turn that only calls tools: the post-training
    plugin rejects a message without one.
    """
    return [
        {
            "role": message["role"],
            "content": message.get("content") or "",
            **{key: message[key] for key in ("tool_calls", "tool_call_id") if message.get(key)},
        }
        for message in json.loads(conversation_messages)
        if message.get("role") in TRAINED_ROLES
    ]


def with_tool_steps(conversation_messages: str, turns: list[list[dict]]) -> str:
    """``conversation_messages`` with each agent turn's own tool steps put just before that turn's reply.

    ``turns`` holds one list per assistant reply, in order. A reply is an assistant message that doesn't call
    UserSim's tools.
    """
    merged: list[dict] = []
    reply = 0
    for message in json.loads(conversation_messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            if reply < len(turns):
                merged.extend(turns[reply])
            reply += 1
        merged.append(message)
    return json.dumps(merged, ensure_ascii=False)


def training_records(rows: pd.DataFrame) -> Iterable[dict]:
    """One ``{"messages": [...], "metadata": {...}}`` record per conversation."""
    for row in rows.to_dict("records"):
        yield {
            "messages": chat_messages(row["conversation_messages"]),
            "metadata": {key: row.get(key) for key in METADATA_KEYS},
        }
