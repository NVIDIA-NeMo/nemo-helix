# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
from nemo_user_sim_plugin.engine.export import apply_review_feedback, chat_messages, training_records
from nemo_user_sim_plugin.intake import review_feedback
from nemo_user_sim_plugin.jobs._common import read_result_frame

CURATED = pd.DataFrame({"trajectory_id": ["a", "b", "c"]})


@pytest.mark.parametrize(
    ("require_review", "kept"),
    [(False, ["a", "c"]), (True, ["a"])],
)
def test_reviewer_feedback_filters_curated_conversations(require_review, kept):
    feedback = {"a": "positive", "b": "negative"}
    assert list(apply_review_feedback(CURATED, feedback, require_review=require_review).trajectory_id) == kept


async def test_a_reviewers_latest_feedback_on_a_session_wins():
    class Intake:
        async def list_annotations(self, *, workspace, query_params):
            assert query_params["filter"] == {"kind": "feedback"}

            async def annotations():
                for session_id, value in [("a", "negative"), ("b", "positive"), ("a", "positive"), ("z", "negative")]:
                    yield SimpleNamespace(session_id=session_id, value=value)

            return annotations()

    assert await review_feedback(Intake(), "default", {"a", "b"}) == {"a": "positive", "b": "positive"}


def test_training_record_keeps_dialogue_and_drops_reasoning():
    messages = [
        {"role": "user", "content": "Freeze my card."},
        {"role": "assistant", "content": "Done.", "reasoning_content": "private", "tool_calls": None},
    ]
    row = pd.DataFrame([{"trajectory_id": "a", "locale": "en_US", "conversation_messages": json.dumps(messages)}])

    (record,) = training_records(row)

    assert record["messages"] == [
        {"role": "user", "content": "Freeze my card."},
        {"role": "assistant", "content": "Done."},
    ]
    assert record["metadata"]["trajectory_id"] == "a"


async def test_export_fails_rather_than_ignore_reviews_it_cant_read():
    class Intake:
        async def list_annotations(self, *, workspace, query_params):
            raise ConnectionError("intake down")

    with pytest.raises(ConnectionError):
        await review_feedback(Intake(), "default", {"a"})


def test_results_with_nested_columns_read_back(tmp_path):
    # UserSim's evaluations carry Arrow struct columns, whose pandas metadata pandas can't restore.
    persona = pa.struct([("first_name", pa.string()), ("age", pa.int64())])
    frame = pd.DataFrame({"persona": pd.Series([{"first_name": "Ana", "age": 34}], dtype=pd.ArrowDtype(persona))})
    frame.to_parquet(tmp_path / "evaluations.parquet", index=False)

    assert read_result_frame(tmp_path / "evaluations.parquet")["persona"][0]["first_name"] == "Ana"


def test_every_message_has_string_content_as_the_post_training_plugin_requires():
    messages = [
        {"role": "user", "content": "What's my balance?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "get_balance"}}]},
        {"role": "tool", "content": "$12", "tool_call_id": "1"},
    ]

    chat = chat_messages(json.dumps(messages))

    assert all(isinstance(message["content"], str) for message in chat)
    assert chat[1]["tool_calls"][0]["function"]["name"] == "get_balance"
