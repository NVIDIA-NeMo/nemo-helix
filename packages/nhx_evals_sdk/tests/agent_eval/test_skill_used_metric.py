# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SkillUsedMetric (skill_present / skill_used)."""

from __future__ import annotations

from typing import Any

import pytest
from nhx_evals_sdk.agent_eval.metrics import SkillUsedMetric
from nhx_evals_sdk.execution.samples import build_metric_input
from nhx_evals_sdk.values.evidence import EVIDENCE_TRACE, CandidateEvidence, EvidenceDescriptor

_LOCATION = ".agents/skills/code-review"
_PROV = {
    "name": "code-review",
    "hash": "deadbeef",
    "mode": "codex_skills_dir",
    "adapter_id": "nvidia.fabric.codex",
    "location": _LOCATION,
}

_LOCATION_B = ".agents/skills/summarize"
_PROV_B = {
    "name": "summarize",
    "hash": "cafebabe",
    "mode": "codex_skills_dir",
    "adapter_id": "nvidia.fabric.codex",
    "location": _LOCATION_B,
}


def _atif(*, tool_path: str | None = None, message: str = "working") -> dict[str, Any]:
    step: dict[str, Any] = {"source": "agent", "message": message}
    if tool_path is not None:
        step["tool_calls"] = [{"function_name": "read_file", "arguments": {"path": tool_path}}]
    return {"schema_version": "ATIF-v1.7", "steps": [step]}


def _evidence(atif: dict[str, Any]) -> CandidateEvidence:
    return CandidateEvidence(descriptors={EVIDENCE_TRACE: EvidenceDescriptor(kind="trace", format="atif", data=atif)})


def _otlp_evidence(payload: str) -> CandidateEvidence:
    return CandidateEvidence(
        descriptors={
            EVIDENCE_TRACE: EvidenceDescriptor(
                kind="trace",
                format="otlp",
                data={
                    "resourceSpans": [
                        {
                            "scopeSpans": [
                                {"spans": [{"attributes": [{"key": "input.value", "value": {"stringValue": payload}}]}]}
                            ]
                        }
                    ]
                },
            )
        }
    )


async def _score(sample: dict[str, Any]) -> dict[str, Any]:
    result = await SkillUsedMetric().compute_scores(build_metric_input({}, sample, 0))
    return {output.name: output.value for output in result.outputs}


def test_output_spec_declares_two_booleans() -> None:
    specs = SkillUsedMetric().output_spec()
    assert [spec.name for spec in specs] == ["skill_present", "skill_used"]


@pytest.mark.asyncio
async def test_no_skill_present_both_false() -> None:
    assert await _score({}) == {"skill_present": False, "skill_used": False}


@pytest.mark.asyncio
async def test_present_and_used_when_trajectory_reads_the_skill() -> None:
    sample = {"skills": [_PROV], "evidence": _evidence(_atif(tool_path=f"{_LOCATION}/SKILL.md"))}
    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_present_not_used_when_trajectory_ignores_the_skill() -> None:
    sample = {"skills": [_PROV], "evidence": _evidence(_atif(tool_path="README.md"))}
    assert await _score(sample) == {"skill_present": True, "skill_used": False}


@pytest.mark.asyncio
async def test_present_and_used_when_only_an_otlp_trace_reads_the_skill() -> None:
    sample = {
        "skills": [_PROV],
        "evidence": _otlp_evidence(f'{{"path": "{_LOCATION}/SKILL.md"}}'),
    }
    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_otlp_answers_when_a_trial_carries_both_views() -> None:
    # OTLP is the richer view and the trial's primary trace, so it answers; ATIF fills the gap
    # for agents that emit no spans rather than overriding them.
    atif = EvidenceDescriptor(kind="trace", format="atif", data=_atif(tool_path="README.md"))
    otlp = EvidenceDescriptor(
        kind="trace",
        format="otlp",
        data={
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "attributes": [
                                        {"key": "output.value", "value": {"stringValue": f"{_LOCATION}/SKILL.md"}}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        },
    )
    sample = {
        "skills": [_PROV],
        "evidence": CandidateEvidence(descriptors={EVIDENCE_TRACE: otlp, "trace:atif": atif, "trace:otlp": otlp}),
    }

    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_a_malformed_atif_view_does_not_suppress_the_otlp_answer() -> None:
    # Discovery finds the ATIF file; only reading it reveals it is unusable. Answering False
    # there would let a broken file hide evidence the trial actually carries.
    broken = EvidenceDescriptor(
        kind="trace", format="atif", data={"schema_version": "ATIF-v1.7", "steps": "not-a-list"}
    )
    otlp = EvidenceDescriptor(
        kind="trace",
        format="otlp",
        data={
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "attributes": [
                                        {"key": "output.value", "value": {"stringValue": f"read {_LOCATION}/SKILL.md"}}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        },
    )
    sample = {
        "skills": [_PROV],
        "evidence": CandidateEvidence(descriptors={EVIDENCE_TRACE: broken, "trace:atif": broken, "trace:otlp": otlp}),
    }

    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_present_not_used_when_otlp_payload_ignores_the_skill() -> None:
    sample = {"skills": [_PROV], "evidence": _otlp_evidence('{"path": "README.md"}')}
    assert await _score(sample) == {"skill_present": True, "skill_used": False}


@pytest.mark.asyncio
async def test_present_not_used_without_a_trace() -> None:
    assert await _score({"skills": [_PROV]}) == {"skill_present": True, "skill_used": False}


@pytest.mark.asyncio
async def test_bare_name_mention_is_not_counted_as_used() -> None:
    # The skill *name* appears in a message, but not its staged location — not counted as used.
    sample = {"skills": [_PROV], "evidence": _evidence(_atif(message="starting the code-review task"))}
    assert (await _score(sample))["skill_used"] is False


# --- multi-skill ("skills" list) ---


@pytest.mark.asyncio
async def test_multi_skill_present_and_first_used() -> None:
    # "skill" is absent (multi-skill run); trajectory reads the first skill's location.
    sample = {
        "skills": [_PROV, _PROV_B],
        "evidence": _evidence(_atif(tool_path=f"{_LOCATION}/SKILL.md")),
    }
    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_multi_skill_present_and_second_used() -> None:
    # Trajectory reads the second skill's location — any-skill-used wins.
    sample = {
        "skills": [_PROV, _PROV_B],
        "evidence": _evidence(_atif(tool_path=f"{_LOCATION_B}/SKILL.md")),
    }
    assert await _score(sample) == {"skill_present": True, "skill_used": True}


@pytest.mark.asyncio
async def test_multi_skill_present_none_used() -> None:
    # Trajectory references neither skill's location.
    sample = {
        "skills": [_PROV, _PROV_B],
        "evidence": _evidence(_atif(tool_path="README.md")),
    }
    assert await _score(sample) == {"skill_present": True, "skill_used": False}


@pytest.mark.asyncio
async def test_empty_skills_list_scores_not_present() -> None:
    assert await _score({"skills": []}) == {"skill_present": False, "skill_used": False}


@pytest.mark.asyncio
async def test_a_trace_that_will_not_parse_scores_false_rather_than_guessing() -> None:
    # Read as OTLP like every other reader, so a malformed field refuses the trace rather than
    # being worked around. `_any_skill_used` then tries the ATIF view and, finding none, logs
    # which formats failed before scoring False -- a loud miss rather than a silent one.
    sample = {
        "skills": [_PROV],
        "evidence": CandidateEvidence(
            descriptors={
                EVIDENCE_TRACE: EvidenceDescriptor(
                    kind="trace",
                    format="otlp",
                    data={
                        "resourceSpans": [
                            {
                                "scopeSpans": [
                                    {
                                        "spans": [
                                            {
                                                "startTimeUnixNano": "not-a-number",
                                                "attributes": [
                                                    {
                                                        "key": "output.value",
                                                        "value": {"stringValue": f"{_LOCATION}/SKILL.md"},
                                                    }
                                                ],
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                )
            }
        ),
    }

    assert await _score(sample) == {"skill_present": True, "skill_used": False}
