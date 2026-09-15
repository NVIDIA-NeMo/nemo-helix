# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF trajectory helpers for reading and uploading agent traces.

Kept separate from ``otlp.py``: the two formats share no representation, and
mixing an ATIF reader into a module named for OTLP would blur a clean boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from nemo_experimentalist_plugin.entities import ResourceRef
from nemo_platform_plugin.intake.types import AtifCreateParams

# ATIF carries agent identity on the trajectory; the Experimentalist carries it as
# OTLP-style span attributes. This maps one onto the other.
_AGENT_FIELD_BY_ATTR: tuple[tuple[str, str], ...] = (
    ("name", "gen_ai.agent.name"),
    ("version", "agent.version"),
    ("model_name", "gen_ai.request.model"),
)


def _load(ref: ResourceRef) -> dict[str, Any]:
    """Parse the ATIF trajectory a ResourceRef points at."""
    path = Path(urlparse(ref.uri).path)
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict):
        raise ValueError(f"ATIF trajectory is not a JSON object: {ref.uri}")
    return trajectory


def read_session_id(ref: ResourceRef) -> str:
    """Return the trajectory's session_id, which Intake uses as the trace id.

    Args:
        ref(ResourceRef): Resource reference to the ATIF trajectory file.

    Returns:
        str: The trajectory session id.
    """
    session_id = _load(ref).get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"No session_id found in {ref.uri}")
    return session_id


def build_ingest_payload(
    ref: ResourceRef,
    *,
    evaluation_name: str,
    task_id: str,
    agent_attrs: dict[str, str],
) -> AtifCreateParams:
    """Load a trajectory and stamp evaluation identity onto it for Intake ingest.

    Agent fields the producer already set are left alone; ``agent_attrs`` only
    fills blanks, so a well-behaved producer stays authoritative.

    ``extra.verifier_result`` is deliberately not injected: the backend already
    creates ``evaluator_results`` rows per trial metric, and letting Intake derive
    a second set from the trajectory would double-count them.

    Args:
        ref(ResourceRef): Resource reference to the ATIF trajectory file.
        evaluation_name(str): Evaluation name, used as ``evaluation_context.evaluation_name``.
        task_id(str): Test case name, used as ``evaluation_context.test_case_name``.
        agent_attrs(dict[str, str]): OTLP-style agent attributes used as fallbacks.

    Returns:
        AtifCreateParams: The ATIF ingest request body.
    """
    trajectory = _load(ref)
    trajectory["evaluation_context"] = {
        "evaluation_name": evaluation_name,
        "test_case_name": task_id,
    }
    agent = trajectory.get("agent")
    if agent is None:
        agent = {}
    elif not isinstance(agent, dict):
        raise ValueError(f"ATIF trajectory agent is not an object: {ref.uri}")
    for field, attr in _AGENT_FIELD_BY_ATTR:
        if not agent.get(field) and attr in agent_attrs:
            agent[field] = agent_attrs[attr]
    trajectory["agent"] = agent
    # The trajectory is unvalidated JSON, so this cast is an assertion, not a proof: the
    # checks cover only the keys AtifCreateParams marks required. Intake validates the rest.
    if not isinstance(trajectory.get("schema_version"), str):
        raise ValueError(f"ATIF trajectory has no schema_version: {ref.uri}")
    if not all(isinstance(agent.get(field), str) for field in ("name", "version")):
        raise ValueError(f"ATIF trajectory has no agent name and version: {ref.uri}")
    return cast(AtifCreateParams, trajectory)
