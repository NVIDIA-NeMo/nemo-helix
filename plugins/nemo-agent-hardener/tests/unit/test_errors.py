# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the war-game failure taxonomy + classifiers (jobs.errors)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from nemo_agent_hardener_plugin.agent_resolver import AgentResolutionError
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_FILESET,
    CATEGORY_MANIFEST,
    CATEGORY_NETWORK,
    CATEGORY_SANDBOX,
    CATEGORY_UNEXPECTED,
    CATEGORY_VALIDATION_FAILED,
    CATEGORY_VICTIM_UNAVAILABLE,
    AgentHardenerRunError,
    classify_exception,
    classify_subprocess,
    read_run_error,
)

# A log from a run that ran the whole cycle but didn't pass validation (reaches the final summary).
# It also contains "openshell"/"victim" as normal content, which previously mislabeled it.
_COMPLETED_BUT_FAILED = (
    "sandbox ready\nopenshell-victim health ready\nAgent Hardener final log\n"
    "  round-0001: success=no\nValidator results:\n"
    "  - smart-benign-validator (benign): FAILED — 16/20 benign requests complied (0 refused, 4 errors)\n"
)


def test_agent_hardener_run_error_is_runtime_error_with_default_remediation() -> None:
    exc = AgentHardenerRunError(CATEGORY_FILESET, "boom")
    assert isinstance(exc, RuntimeError)  # keeps `pytest.raises(RuntimeError)` call sites working
    assert exc.category == CATEGORY_FILESET
    assert exc.remediation  # a default remediation is filled in from the category


def test_agent_hardener_run_error_explicit_remediation_wins() -> None:
    exc = AgentHardenerRunError(CATEGORY_FILESET, "boom", remediation="do X")
    assert exc.remediation == "do X"


def test_classify_exception_passes_through_typed_error() -> None:
    failure = classify_exception(AgentHardenerRunError(CATEGORY_SANDBOX, "sandbox down"))
    assert failure.category == CATEGORY_SANDBOX
    assert failure.message == "sandbox down"


def test_classify_exception_maps_agent_resolution_to_manifest() -> None:
    assert classify_exception(AgentResolutionError("no agent")).category == CATEGORY_MANIFEST


def test_classify_exception_maps_transport_error_to_network() -> None:
    assert classify_exception(httpx.ConnectError("refused")).category == CATEGORY_NETWORK
    assert classify_exception(ConnectionError("reset")).category == CATEGORY_NETWORK


def test_classify_exception_defaults_to_unexpected() -> None:
    failure = classify_exception(ValueError("weird"))
    assert failure.category == CATEGORY_UNEXPECTED
    assert "weird" in failure.message


def test_read_run_error_parses_structured_dump(tmp_path: Path) -> None:
    path = tmp_path / "run-error.json"
    path.write_text(
        json.dumps({"category": "sandbox", "message": "docker down", "remediation": "start docker", "stack": "..."}),
        encoding="utf-8",
    )
    failure = read_run_error(path)
    assert failure is not None
    assert (failure.category, failure.message, failure.remediation) == ("sandbox", "docker down", "start docker")


def test_read_run_error_missing_or_malformed_returns_none(tmp_path: Path) -> None:
    assert read_run_error(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert read_run_error(bad) is None


def test_read_run_error_defaults_unknown_category(tmp_path: Path) -> None:
    path = tmp_path / "run-error.json"
    path.write_text(json.dumps({"message": "something"}), encoding="utf-8")
    failure = read_run_error(path)
    assert failure is not None
    assert failure.category == CATEGORY_UNEXPECTED
    assert failure.message == "something"


def test_classify_subprocess_prefers_structured_error(tmp_path: Path) -> None:
    path = tmp_path / "run-error.json"
    path.write_text(json.dumps({"category": "victim_unavailable", "message": "victim died"}), encoding="utf-8")
    exc = classify_subprocess(1, "irrelevant log", read_run_error(path))
    assert exc.category == "victim_unavailable"
    assert str(exc) == "victim died"


def test_classify_subprocess_falls_back_to_log_heuristic() -> None:
    exc = classify_subprocess(1, "... could not reach the OpenShell gateway ...", None)
    assert exc.category == CATEGORY_SANDBOX


def test_classify_subprocess_unknown_log_is_unexpected() -> None:
    exc = classify_subprocess(3, "totally opaque output", None)
    assert exc.category == CATEGORY_UNEXPECTED
    assert "code 3" in str(exc)


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        ("connection refused by host", CATEGORY_NETWORK),
        ("the victim returned server disconnected", "victim_unavailable"),
        ("docker daemon not running", CATEGORY_SANDBOX),
        ("... attacker execution failed ...", "attacker_failed"),
        ("Attacker results:\n  attacker agent status: failed=1", "attacker_failed"),
    ],
)
def test_classify_subprocess_heuristic_cues(log: str, expected: str) -> None:
    assert classify_subprocess(1, log, None).category == expected


def test_completed_run_classifies_as_validation_failed_not_victim() -> None:
    # The reported bug: a run that finished the full cycle but failed validation was mislabeled
    # victim_unavailable (its log contains "openshell"/"victim") with a misleading "inspect the victim
    # log" message. It must now be validation_failed with a clear message.
    exc = classify_subprocess(1, _COMPLETED_BUT_FAILED, None)
    assert exc.category == CATEGORY_VALIDATION_FAILED
    assert "did not pass validation" in str(exc)
    assert "exited with code" not in str(exc)


def test_healthy_victim_phrase_alone_is_not_victim_unavailable() -> None:
    # "victim health ready" is a healthy line; with no failure cue and no completion marker it must
    # not be mistaken for a victim failure (the old broad "victim" cue did exactly that).
    assert classify_subprocess(1, "victim health ready\n", None).category == CATEGORY_UNEXPECTED


def test_genuine_victim_http_failure_still_victim_unavailable() -> None:
    # A real mid-run victim failure (never reaches the final summary) stays victim_unavailable.
    log = "sandbox ready\nOpenShell victim returned HTTP 422 - invalid model\n"
    assert classify_subprocess(1, log, None).category == CATEGORY_VICTIM_UNAVAILABLE


def test_read_run_error_surfaces_attacker_failed_from_agent_hardener(tmp_path: Path) -> None:
    # agent-hardener's AttackerError serializes category "attacker_failed" with its own remediation.
    path = tmp_path / "run-error.json"
    path.write_text(
        json.dumps(
            {
                "category": "attacker_failed",
                "message": "attacker(s) garak-agent-breaker did not complete: TimeoutError",
                "remediation": "raise garak.timeout_s or lower attack_intensity",
            }
        ),
        encoding="utf-8",
    )
    failure = read_run_error(path)
    assert failure is not None
    assert failure.category == "attacker_failed"
    assert "TimeoutError" in failure.message
