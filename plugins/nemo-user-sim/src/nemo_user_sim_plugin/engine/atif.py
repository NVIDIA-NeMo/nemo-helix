# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converts a finished UserSim row into an ATIF trajectory that Intake stores as one session."""

import json

ATIF_VERSION = "ATIF-v1.7"
PERSONA_KEYS = (
    "probe_type",
    "probe_variant",
    "locale",
    "persona_uuid",
    "persona_name",
    "persona_age_bin",
    "persona_education_level",
    "persona_occupation",
    "user_interaction_style",
)


SIMULATED_SUFFIX = " (user-sim)"


def simulated_name(agent_name: str) -> str:
    """The agent name simulated traces publish under, so Studio's Agent filter keeps them apart from real traffic."""
    return f"{agent_name}{SIMULATED_SUFFIX}"


def row_to_atif(row: dict, *, agent_name: str, agent_version: str, model_name: str, run: str | None = None) -> dict:
    """Map messages to ATIF steps; tool results ride on the agent step that called them.

    The session ID is UserSim's trajectory ID, so Studio annotations join back to the row. With ``run``,
    the conversation is tagged as a test case of that run's Intake Evaluation, which keeps simulations
    apart from live traffic.
    """
    messages = json.loads(row["conversation_messages"])
    results = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m.get("role") == "tool"}
    steps: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            continue
        step: dict = {"step_id": len(steps) + 1, "message": message.get("content") or ""}
        if role in ("user", "system"):
            step["source"] = role
        else:
            step["source"] = "agent"
            if calls := message.get("tool_calls"):
                step["tool_calls"] = [
                    {
                        "tool_call_id": call["id"],
                        "function_name": call["function"]["name"],
                        "arguments": json.loads(call["function"].get("arguments") or "{}"),
                    }
                    for call in calls
                ]
                step["observation"] = {
                    "results": [
                        {"source_call_id": call["id"], "content": results.get(call["id"], "")} for call in calls
                    ]
                }
        steps.append(step)
    return {
        "schema_version": ATIF_VERSION,
        "session_id": row["trajectory_id"],
        "trajectory_id": row["trajectory_id"],
        "agent": {"name": simulated_name(agent_name), "version": agent_version, "model_name": model_name},
        "steps": steps,
        "extra": {"usersim": {key: row.get(key) for key in PERSONA_KEYS}},
        **(
            {
                "evaluation_context": {
                    "evaluation_name": run,
                    "test_case_name": f"{row['persona_name']} · {row['probe_type']}",
                }
            }
            if run
            else {}
        ),
    }
