# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure tau2-sim → OTLP-span-dict transform."""

import json
import time
from typing import TypedDict

from evaluation.otlp_build import session_id_for, sim_to_spans

_SESSION = "tau2-airline-task7-t0"


class _CommonSpanArgs(TypedDict):
    agent_name: str
    agent_version: str
    session_id: str
    evaluation_name: str


def _spans(sim, *, include_rewards=False, task=None, agent_llm="openai/m"):
    return sim_to_spans(
        sim,
        agent_name="tau2-airline",
        agent_version="v1",
        session_id=_SESSION,
        evaluation_name="tau2-airline-20260626-000000-abcd",
        task=task,
        include_rewards=include_rewards,
        agent_llm=agent_llm,
    )


def _by_kind(spans, kind):
    return [s for s in spans if s["kind"] == kind]


def _root(spans):
    (root,) = _by_kind(spans, "AGENT")
    return root


def _basic_sim(messages, *, task_id="7", trial=0, reward=1.0):
    return {
        "task_id": task_id,
        "trial": trial,
        "messages": messages,
        "reward_info": {"reward": reward},
        "termination_reason": "done",
    }


def _telecom_sim():
    """A telecom sim with a user-driven device tool — the ATIF 422 regression case.

    The customer (user) runs a device tool: a ``role="user"`` message carrying a
    ``ToolCall`` with ``requestor="user"``, answered by a ``role="tool"`` result
    that is also ``requestor="user"``. ATIF-v1.7 (agent-only) rejected this; OTLP
    must ingest it faithfully.
    """
    return _basic_sim(
        [
            {"role": "user", "content": "my data isn't working"},
            {
                "role": "assistant",
                "content": "Let me check. Can you run a speed test?",
                "tool_calls": [{"id": "a1", "name": "get_line_status", "arguments": {"id": 7}}],
            },
            {"role": "tool", "tool_call_id": "a1", "content": "line ok", "requestor": "assistant"},
            {
                "role": "user",
                "content": "ok running it",
                "tool_calls": [{"id": "u1", "name": "run_speed_test", "arguments": {}, "requestor": "user"}],
            },
            {"role": "tool", "tool_call_id": "u1", "content": "12 Mbps", "requestor": "user"},
            {"role": "assistant", "content": "Your speed looks fine."},
        ],
        task_id="mobile_data",
    )


# --- timestamps (must land in the platform's retention window) -------------


def test_spans_timestamped_near_now_not_a_fixed_past_date():
    # Regression: a hardcoded past base (2023) was silently dropped by Intake's
    # span retention/TTL window, so OTLP spans never became queryable. Spans must
    # be stamped at ~ingest time. Default base = wall-clock now.
    before = time.time_ns()
    spans = _spans(_basic_sim([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]))
    starts = [s["start_ns"] for s in spans]
    assert min(starts) >= before
    assert max(s["end_ns"] for s in spans) <= time.time_ns() + 60 * 1_000_000_000


def test_base_ns_is_honored_when_provided():
    base = 1_900_000_000_000_000_000
    spans = sim_to_spans(
        _basic_sim([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]),
        agent_name="a",
        agent_version="v",
        session_id="s",
        evaluation_name="run-x",
        task=None,
        include_rewards=False,
        agent_llm="m",
        base_ns=base,
    )
    assert min(s["start_ns"] for s in spans) == base  # monotonic clock seeded from the provided base
    assert all(base <= s["start_ns"] < base + 60 * 1_000_000_000 for s in spans)  # tightly clustered


# --- session id (stable identity, reproduces the prior ATIF slug) ----------


def test_session_id_basic():
    sim = {"task_id": "7", "trial": 0}
    assert session_id_for(sim, evaluation_name="tau2-airline-x") == "tau2-airline-x-task7-t0"


def test_session_id_slugs_evaluation_and_task():
    sim = {"task_id": "set[1]/odd", "trial": 0}
    assert session_id_for(sim, evaluation_name="tau2/airline") == "tau2-airline-taskset-1-odd-t0"


def test_session_id_trial_none_defaults_to_zero():
    assert session_id_for({"task_id": "0", "trial": None}, evaluation_name="a").endswith("-t0")


def test_session_id_is_run_unique():
    sim = {"task_id": "7", "trial": 0}
    a = session_id_for(sim, evaluation_name="run-A")
    b = session_id_for(sim, evaluation_name="run-B")
    assert a != b  # different runs -> different session ids
    assert a == session_id_for(sim, evaluation_name="run-A")  # same run -> stable (twins correlate)


# --- AGENT root ------------------------------------------------------------


def test_exactly_one_agent_root_with_no_parent():
    spans = _spans(
        _basic_sim(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
    )
    root = _root(spans)
    assert root["name"] == "tau2-airline"
    assert root["parent_span_id"] is None
    assert root["attributes"]["openinference.span.kind"] == "AGENT"
    assert root["attributes"]["gen_ai.agent.name"] == "tau2-airline"
    assert root["attributes"]["gen_ai.agent.version"] == "v1"


def test_agent_root_input_first_user_output_last_agent():
    spans = _spans(
        _basic_sim(
            [
                {"role": "user", "content": "first-user"},
                {"role": "assistant", "content": "mid"},
                {"role": "user", "content": "second-user"},
                {"role": "assistant", "content": "last-agent"},
            ]
        )
    )
    root = _root(spans)
    assert root["attributes"]["input.value"] == "first-user"
    assert root["attributes"]["output.value"] == "last-agent"


# --- grouping --------------------------------------------------------------


def test_every_span_carries_conversation_and_session_id():
    spans = _spans(_telecom_sim())
    assert spans  # non-empty
    for s in spans:
        assert s["attributes"]["gen_ai.conversation.id"] == _SESSION
        assert s["attributes"]["session.id"] == _SESSION


def test_span_ids_unique_and_deterministic():
    sim = _telecom_sim()
    spans = _spans(sim)
    ids = [s["span_id"] for s in spans]
    assert len(ids) == len(set(ids))  # unique
    assert all(len(i) == 16 for i in ids)  # 8 bytes hex
    again = _spans(sim)
    assert [s["span_id"] for s in again] == ids  # pure / deterministic


# --- LLM turns -------------------------------------------------------------


def test_one_llm_turn_per_assistant_message():
    spans = _spans(
        _basic_sim(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "two"},
            ]
        )
    )
    llms = _by_kind(spans, "LLM")
    assert len(llms) == 2
    root = _root(spans)
    for llm in llms:
        assert llm["parent_span_id"] == root["span_id"]
        assert llm["name"].startswith("agent-")
        assert llm["attributes"]["gen_ai.request.model"] == "openai/m"


def test_llm_output_value_carries_content_and_tool_calls():
    spans = _spans(
        _basic_sim(
            [
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"x": 1}}],
                },
            ]
        )
    )
    (llm,) = _by_kind(spans, "LLM")
    assert llm["attributes"]["output.mime_type"] == "application/json"
    out = json.loads(llm["attributes"]["output.value"])
    assert out["content"] == "calling"
    assert out["tool_calls"][0]["name"] == "lookup"


# --- agent-driven TOOL -----------------------------------------------------


def test_agent_tool_is_child_of_its_llm_turn():
    spans = _spans(
        _basic_sim(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"x": 1}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "result-text"},
            ]
        )
    )
    (llm,) = _by_kind(spans, "LLM")
    (tool,) = _by_kind(spans, "TOOL")
    assert tool["parent_span_id"] == llm["span_id"]
    assert tool["name"] == "lookup"
    assert tool["attributes"]["gen_ai.tool.name"] == "lookup"
    assert tool["attributes"]["gen_ai.agent.name"] == "tau2-airline"  # agent-driven keeps agent name
    assert json.loads(tool["attributes"]["input.value"]) == {"x": 1}
    assert tool["attributes"]["output.value"] == "result-text"


# --- user-driven TOOL (the telecom 422 regression) -------------------------


def test_user_driven_tool_is_child_of_root_with_actor_and_no_agent_name():
    spans = _spans(_telecom_sim())
    user_tools = [s for s in _by_kind(spans, "TOOL") if s["attributes"].get("tau2.actor") == "user"]
    assert len(user_tools) == 1
    ut = user_tools[0]
    root = _root(spans)
    assert ut["parent_span_id"] == root["span_id"]  # parented to AGENT root, not an LLM turn
    assert ut["name"] == "run_speed_test"
    assert ut["attributes"]["gen_ai.tool.name"] == "run_speed_test"
    assert "gen_ai.agent.name" not in ut["attributes"]  # NOT the agent's action
    assert ut["attributes"]["output.value"] == "12 Mbps"


def test_agent_and_user_tools_distinguishable():
    spans = _spans(_telecom_sim())
    tools = _by_kind(spans, "TOOL")
    agent_tools = [t for t in tools if "gen_ai.agent.name" in t["attributes"]]
    user_tools = [t for t in tools if t["attributes"].get("tau2.actor") == "user"]
    assert {t["name"] for t in agent_tools} == {"get_line_status"}
    assert {t["name"] for t in user_tools} == {"run_speed_test"}
    # the agent tool hangs off an LLM turn; the user tool hangs off the root
    llm_ids = {s["span_id"] for s in _by_kind(spans, "LLM")}
    assert agent_tools[0]["parent_span_id"] in llm_ids
    assert user_tools[0]["parent_span_id"] == _root(spans)["span_id"]


# --- EVALUATOR / reward gating --------------------------------------------


def test_no_evaluator_span_in_realistic_mode():
    spans = _spans(_basic_sim([{"role": "user", "content": "hi"}]), include_rewards=False)
    assert _by_kind(spans, "EVALUATOR") == []


def test_evaluator_span_in_oracle_mode():
    spans = _spans(_basic_sim([{"role": "user", "content": "hi"}], reward=1.0), include_rewards=True)
    (ev,) = _by_kind(spans, "EVALUATOR")
    root = _root(spans)
    assert ev["name"] == "tau2.verifier"
    assert ev["parent_span_id"] == root["span_id"]
    assert ev["attributes"]["score"] == 1.0
    out = json.loads(ev["attributes"]["output.value"])
    assert out["score"] == 1.0


def test_no_evaluator_when_reward_absent_even_in_oracle_mode():
    sim = {"task_id": "0", "messages": [{"role": "user", "content": "hi"}]}  # no reward_info
    assert _by_kind(_spans(sim, include_rewards=True), "EVALUATOR") == []


# --- failed tool -> ERROR status ------------------------------------------


def test_failed_tool_call_marks_error_status():
    spans = _spans(
        _basic_sim(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "boom", "arguments": {}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "kaboom", "error": True},
            ]
        )
    )
    (tool,) = _by_kind(spans, "TOOL")
    assert tool["status"] == "ERROR"
    assert tool["status_message"]


# --- task gating (oracle-gating invariant) ---------------------------------


def test_task_evaluation_criteria_gated_by_include_rewards():
    task = {
        "description": {"purpose": "p"},
        "user_scenario": {"instructions": "book X"},
        "evaluation_criteria": {"actions": [{"name": "book_reservation"}]},
    }
    realistic = _root(_spans(_basic_sim([{"role": "user", "content": "hi"}]), include_rewards=False, task=task))
    oracle = _root(_spans(_basic_sim([{"role": "user", "content": "hi"}]), include_rewards=True, task=task))
    realistic_task = json.loads(realistic["attributes"]["tau2.task"])
    oracle_task = json.loads(oracle["attributes"]["tau2.task"])
    assert realistic_task["user_scenario"]["instructions"] == "book X"  # setup kept
    assert "evaluation_criteria" not in realistic_task  # gold withheld in realistic mode
    assert oracle_task["evaluation_criteria"]["actions"][0]["name"] == "book_reservation"


def test_every_span_carries_evaluation_and_test_case_tags():
    sim = _basic_sim([{"role": "user", "content": "hi"}], task_id="7")
    spans = sim_to_spans(
        sim,
        agent_name="tau2-airline",
        agent_version="v1",
        session_id=_SESSION,
        evaluation_name="tau2-airline-20260626-000000-abcd",
        include_rewards=True,
    )
    assert spans  # at least the AGENT root + EVALUATOR
    for s in spans:
        assert s["attributes"]["nemo.evaluation.name"] == "tau2-airline-20260626-000000-abcd"
        assert s["attributes"]["nemo.test_case.name"] == "7"


def test_tags_identical_across_realistic_and_oracle_twins():
    sim = _basic_sim([{"role": "user", "content": "hi"}], task_id="7")
    common: _CommonSpanArgs = {
        "agent_name": "tau2-airline",
        "agent_version": "v1",
        "session_id": _SESSION,
        "evaluation_name": "run-1",
    }
    realistic = sim_to_spans(sim, include_rewards=False, **common)
    oracle = sim_to_spans(sim, include_rewards=True, **common)

    def tag(s):
        return (s["attributes"]["nemo.evaluation.name"], s["attributes"]["nemo.test_case.name"])

    assert {tag(s) for s in realistic} == {("run-1", "7")}
    assert {tag(s) for s in oracle} == {("run-1", "7")}


# --- user-text CHAIN spans -------------------------------------------------


def test_user_text_spans_emitted_per_utterance():
    """Multi-turn sim emits CHAIN spans for each non-empty user utterance."""
    sim = _basic_sim(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
        ]
    )
    spans = _spans(sim, include_rewards=False)
    chain_spans = _by_kind(spans, "CHAIN")
    assert len(chain_spans) == 2
    # Names are user-<step_id>; steps are 1-indexed and increment per message
    names = {s["name"] for s in chain_spans}
    assert names == {"user-1", "user-3"}


def test_user_text_spans_parented_to_agent_root():
    sim = _basic_sim(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    spans = _spans(sim, include_rewards=False)
    root = _root(spans)
    (chain,) = _by_kind(spans, "CHAIN")
    assert chain["parent_span_id"] == root["span_id"]


def test_user_text_span_has_correct_attributes():
    sim = _basic_sim([{"role": "user", "content": "how are you?"}])
    spans = _spans(sim, include_rewards=False)
    (chain,) = _by_kind(spans, "CHAIN")
    assert chain["attributes"]["tau2.actor"] == "user"
    assert chain["attributes"]["output.value"] == "how are you?"
    assert chain["attributes"]["output.mime_type"] == "text/plain"
    assert chain["attributes"]["openinference.span.kind"] == "CHAIN"


def test_empty_content_user_message_yields_no_chain_span():
    """A user message with empty/None content must NOT produce a CHAIN span."""
    sim = _basic_sim(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": None},
        ]
    )
    spans = _spans(sim, include_rewards=False)
    assert _by_kind(spans, "CHAIN") == []


def test_user_text_spans_precede_llm_spans_in_timeline():
    """User utterance spans must start before subsequent agent LLM spans (correct interleaving)."""
    sim = _basic_sim(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "sure"},
        ]
    )
    spans = _spans(sim, include_rewards=False)
    llms = sorted(_by_kind(spans, "LLM"), key=lambda s: s["start_ns"])
    chains = sorted(_by_kind(spans, "CHAIN"), key=lambda s: s["start_ns"])
    # Each user utterance at step N precedes the next assistant at step N+1
    # chain[0] (user-1) < llm[0] (agent-2); chain[1] (user-3) < llm[1] (agent-4)
    assert chains[0]["start_ns"] < llms[0]["start_ns"]
    assert chains[1]["start_ns"] < llms[1]["start_ns"]


def test_user_text_spans_visible_in_both_twins():
    """User text is not gated by include_rewards — appears in realistic and oracle twins."""
    sim = _basic_sim([{"role": "user", "content": "check my bill"}])
    realistic = _spans(sim, include_rewards=False)
    oracle = _spans(sim, include_rewards=True)
    assert len(_by_kind(realistic, "CHAIN")) == 1
    assert len(_by_kind(oracle, "CHAIN")) == 1
