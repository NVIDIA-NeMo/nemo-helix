# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recovering a Fabric trial's answer from its trace evidence.

The Fabric runtime falls back to this when the harness ``RunResult`` carries no output payload.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.fabric._common import trace_answer_text
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import register_trace_evidence
from nemo_evaluator_sdk.values.evidence import EvidenceDescriptor

from packages.nemo_evaluator_sdk.tests.agent_eval._otlp_testkit import write_answer_trace


def _write_atif(evidence_dir: Path, *messages: str) -> Path:
    path = evidence_dir / "trajectory.atif.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [{"source": "agent", "message": message} for message in messages],
            }
        ),
        encoding="utf-8",
    )
    return path


def _descriptors(*, otlp: Path | None = None, atif: Path | None = None) -> dict[str, EvidenceDescriptor]:
    descriptors: dict[str, EvidenceDescriptor] = {}
    register_trace_evidence(descriptors, atif=atif, otlp=otlp)
    return descriptors


def test_the_otlp_answer_wins_over_the_atif_one(tmp_path: Path) -> None:
    descriptors = _descriptors(otlp=write_answer_trace(tmp_path, "42"), atif=_write_atif(tmp_path, "stale"))

    assert trace_answer_text(descriptors) == "42"


def test_atif_answers_when_the_otlp_trace_carries_none(tmp_path: Path) -> None:
    """A registered OTLP trace is not itself an answer, so both formats have to be consulted.

    Reading the primary ``trace`` key instead of the format-qualified ones would skip ATIF
    whenever an OTLP trace existed, which is the shape an agent emitting only tool spans produces.
    """
    otlp = write_answer_trace(tmp_path, "")
    descriptors = _descriptors(otlp=otlp, atif=_write_atif(tmp_path, "first", "final"))

    assert trace_answer_text(descriptors) == "final"


def test_an_unreadable_otlp_trace_falls_through_to_atif(tmp_path: Path) -> None:
    otlp = write_answer_trace(tmp_path, "42")
    otlp.write_text("{not json", encoding="utf-8")
    descriptors = _descriptors(otlp=otlp, atif=_write_atif(tmp_path, "final"))

    assert trace_answer_text(descriptors) == "final"


def test_a_whitespace_only_otlp_answer_falls_through_to_atif(tmp_path: Path) -> None:
    descriptors = _descriptors(otlp=write_answer_trace(tmp_path, "  \n "), atif=_write_atif(tmp_path, "final"))

    assert trace_answer_text(descriptors) == "final"


def test_a_whitespace_only_answer_is_no_answer(tmp_path: Path) -> None:
    """Blank is answerless, so a trial reports no answer rather than whitespace.

    The format readers hand back a trace's last agent text verbatim; a caller asking whether the
    agent answered at all has to apply the presence rule itself.
    """
    descriptors = _descriptors(otlp=write_answer_trace(tmp_path, " "), atif=_write_atif(tmp_path, "\t"))

    assert trace_answer_text(descriptors) is None


def test_a_trial_with_no_trace_evidence_has_no_answer() -> None:
    assert trace_answer_text(_descriptors()) is None


def test_an_agent_ending_on_an_empty_message_produced_no_answer(tmp_path: Path) -> None:
    descriptors = _descriptors(atif=_write_atif(tmp_path, "thinking out loud", ""))

    assert trace_answer_text(descriptors) is None
