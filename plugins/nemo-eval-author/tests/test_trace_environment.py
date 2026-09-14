# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone trace-environment boundary helper."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from harbor.models.task.task import Task

_PLUGIN = Path(__file__).resolve().parents[1]
_SCRIPT = _PLUGIN / "skills" / "eval-author-trace-environment" / "scripts" / "trace_environment.py"
_SUMMARY_SCHEMA = "nemo.eval_author.trace_environment_summary.v2"
_CANDIDATE_SCHEMA = "nemo.eval_author.trace_environment_candidate.v2"
_VALIDATION_SCHEMA = "nemo.eval_author.trace_environment_validation.v5"


def _run(*args: str) -> tuple[int, dict[str, Any]]:
    result = subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, check=False)
    assert result.stdout, result.stderr
    return result.returncode, json.loads(result.stdout)


def _atif(*, image_only: bool = False) -> dict[str, Any]:
    message: str | list[dict[str, Any]]
    if image_only:
        message = [{"type": "image", "source": {"media_type": "image/png", "path": "screen.png"}}]
    else:
        message = "Repair the fixture for jane@example.com from /home/jane/project."
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "trajectory_id": "trace-1",
        "agent": {"name": "coding-agent", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": message},
            {
                "step_id": 2,
                "source": "agent",
                "message": "Calling 10.2.3.4 for the fixture.",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "fixture.read",
                        "arguments": {"phone": "+1 (303) 555-0119"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "call-1", "content": "Result for 123-45-6789"}]},
            },
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _workspace(tmp_path: Path, *, image_only: bool = False) -> tuple[Path, Path]:
    root = tmp_path / ".eval-author" / "trace-environments"
    code, result = _run("init", "--root", str(root), "--task-id", "repair-fixture")
    assert code == 0, result
    task_dir = Path(result["task_dir"])
    source = tmp_path / "source.atif.json"
    _write_json(source, _atif(image_only=image_only))
    code, result = _run(
        "prepare",
        "--task-dir",
        str(task_dir),
        "--atif",
        str(source),
        "--source-kind",
        "atif",
    )
    assert code == 0, result
    return task_dir, source


def _candidate(
    task_dir: Path,
    *,
    status: str = "candidate",
    ground_truth: dict[str, Any] | None = None,
    software_requirements: list[dict[str, Any]] | None = None,
) -> None:
    if ground_truth is None:
        ground_truth = {
            "availability": "absent",
            "use": "none",
            "artifacts": [],
            "absence_reason": "No distinct reference artifact was recorded.",
        }
    if software_requirements is None:
        software_requirements = []
    if status == "candidate":
        payload = {
            "schema": _CANDIDATE_SCHEMA,
            "status": "candidate",
            "decision_basis": "safe_atif_only",
            "instruction": "Repair the local fixture.",
            "requirements": [{"description": "The fixture check passes.", "evidence_steps": [1, 2]}],
            "verification_mode": "execution",
            "evidence_steps": [1, 2],
            "uncertainties": [],
            "reason_codes": [],
            "ground_truth": ground_truth,
            "software_requirements": software_requirements,
        }
    else:
        payload = {
            "schema": _CANDIDATE_SCHEMA,
            "status": "no_candidate",
            "decision_basis": "safe_atif_only",
            "instruction": None,
            "requirements": [],
            "verification_mode": None,
            "evidence_steps": [1],
            "uncertainties": ["The expected outcome is absent."],
            "reason_codes": ["missing_outcome"],
            "ground_truth": ground_truth,
            "software_requirements": software_requirements,
        }
    _write_json(task_dir / "candidate.json", payload)


@pytest.mark.parametrize("status", ["candidate", "no_candidate"])
def test_check_candidate_is_read_only_before_construction(tmp_path: Path, status: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status=status)
    before = {
        path.relative_to(task_dir): (path.read_bytes(), path.stat().st_mode)
        for path in task_dir.rglob("*")
        if path.is_file()
    }

    code, result = _run("check-candidate", "--task-dir", str(task_dir))

    assert code == 0, result
    assert result["scope"] == "candidate_metadata"
    assert result["execution_verified"] is False
    assert result["status"] == status
    assert not (task_dir / "task/task.toml").exists()
    assert json.loads((task_dir / "summary.json").read_text())["status"] == "pending"
    assert before == {
        path.relative_to(task_dir): (path.read_bytes(), path.stat().st_mode)
        for path in task_dir.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "runtime"),
        ("license", "Apache-2.0"),
        ("availability", "not_installed"),
        ("category", []),
        ("license", {}),
        ("availability", []),
    ],
)
def test_check_candidate_rejects_unknown_software_enum_without_repair(tmp_path: Path, field: str, value: Any) -> None:
    task_dir, _ = _workspace(tmp_path)
    software = _software(required=False, availability="unknown")
    software[field] = value
    _candidate(task_dir, software_requirements=[software])
    before = (task_dir / "candidate.json").read_bytes()

    code, result = _run("check-candidate", "--task-dir", str(task_dir))

    assert code == 1
    assert f"software_requirements[0].{field} is not recognized" in result["error"]
    assert (task_dir / "candidate.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [("availability", []), ("use", {})])
def test_check_candidate_rejects_invalid_ground_truth_enum(tmp_path: Path, field: str, value: Any) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "candidate.json"
    candidate = json.loads(path.read_text())
    candidate["ground_truth"][field] = value
    _write_json(path, candidate)
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert f"ground_truth.{field} is not recognized" in result["error"]


def test_check_candidate_rejects_unknown_evidence_step(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "candidate.json"
    candidate = json.loads(path.read_text())
    candidate["evidence_steps"] = [999]
    _write_json(path, candidate)

    code, result = _run("check-candidate", "--task-dir", str(task_dir))

    assert code == 1
    assert "evidence_steps" in result["error"]


def test_check_candidate_rejects_changed_safe_trace(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "safe/trace.atif.json"
    safe = json.loads(path.read_text())
    safe["steps"][0]["message"] = "Changed evidence"
    _write_json(path, safe)

    code, result = _run("check-candidate", "--task-dir", str(task_dir))

    assert code == 1
    assert "safe ATIF digest or size changed" in result["error"]


@pytest.mark.parametrize("status", ["pending", [], None])
def test_check_candidate_rejects_invalid_status(tmp_path: Path, status: Any) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "candidate.json"
    candidate = json.loads(path.read_text())
    candidate["status"] = status
    _write_json(path, candidate)
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert "candidate.status" in result["error"]


def test_check_candidate_requires_prepared_evidence(tmp_path: Path) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"
    code, result = _run("init", "--root", str(root), "--task-id", "repair-fixture")
    assert code == 0, result
    task_dir = Path(result["task_dir"])
    _candidate(task_dir)
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert "prepare ATIF evidence" in result["error"]


def test_check_candidate_rejects_summary_fields_without_moving_them(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status="no_candidate")
    path = task_dir / "candidate.json"
    candidate = json.loads(path.read_text())
    candidate["did_not_work"] = ["Construction note for the summary."]
    _write_json(path, candidate)
    before = path.read_bytes()
    summary_before = (task_dir / "summary.json").read_bytes()
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert "candidate fields do not match" in result["error"]
    assert path.read_bytes() == before
    assert (task_dir / "summary.json").read_bytes() == summary_before


def test_check_candidate_rejects_oversized_safe_trace_before_loading(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "safe/trace.atif.json"
    with path.open("wb") as stream:
        stream.truncate(25 * 1024 * 1024 + 1)
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert "safe ATIF exceeds" in result["error"]


def test_check_candidate_rejects_symlinked_safe_trace(tmp_path: Path) -> None:
    task_dir, source = _workspace(tmp_path)
    _candidate(task_dir)
    path = task_dir / "safe/trace.atif.json"
    path.unlink()
    path.symlink_to(source)
    code, result = _run("check-candidate", "--task-dir", str(task_dir))
    assert code == 1
    assert "safe ATIF must be a retained regular file" in result["error"]


def _review_privacy(task_dir: Path, *, reviewer_kind: str = "agent") -> None:
    code, result = _run(
        "review-privacy",
        "--task-dir",
        str(task_dir),
        "--reviewer-kind",
        reviewer_kind,
        "--note",
        "Reviewed every safe ATIF string and all contextual audit findings.",
    )
    assert code == 0, result


def _record_reproducibility(task_dir: Path) -> None:
    (task_dir / "reproducibility.json").unlink(missing_ok=True)
    code, result = _run("record-reproducibility", "--task-dir", str(task_dir))
    assert code == 0, result


def _review_publication(task_dir: Path, *, reviewer_kind: str = "agent") -> dict[str, Any]:
    code, preview = _run("prepare-publication", "--task-dir", str(task_dir))
    assert code == 0, preview
    code, result = _run(
        "review-publication",
        "--task-dir",
        str(task_dir),
        "--preview-dir",
        preview["preview_dir"],
        "--sha256",
        preview["sha256"],
        "--reviewer-kind",
        reviewer_kind,
        "--note",
        "Reviewed all files in this synthetic publication preview.",
    )
    assert code == 0, result
    return preview


def _ready_environment(
    task_dir: Path,
    *,
    mode: str = "separate",
    nop_reward: float = 0.0,
    oracle_reward: float = 1.0,
    oracle_exception: object | None = None,
    record_validation: bool = True,
) -> None:
    task = task_dir / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    verifier = f'\n[verifier]\nenvironment_mode = "{mode}"\n'
    if mode == "separate":
        verifier += 'network_mode = "no-network"\n\n[verifier.environment]\nnetwork_mode = "no-network"\n'
    (task / "task.toml").write_text(
        f'schema_version = "1.1"\n{verifier}\n[environment]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Repair the fixture.\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task / "README.md").write_text(
        """# Repair fixture

## Difficulty explanation

The task requires identifying and repairing an invalid local fixture.

## Environment and software requirements

The task uses the tools installed by its container image and has no proprietary dependencies.

## Ground-truth provenance

The expected state is derived from the retained trace evidence cited in the candidate record.

## Solution explanation

The reference solution replaces the invalid fixture with the required state.

## Verification explanation

The test checks the resulting fixture state without inspecting the agent's implementation.

## Relevant experience

The human reviewer confirmed that this fixture accurately represents the recorded workflow.
""",
        encoding="utf-8",
    )
    (task / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    if mode == "separate":
        (task / "tests" / "Dockerfile").write_text(
            "FROM scratch\nCOPY test.sh /tests/test.sh\n",
            encoding="utf-8",
        )
    (task / "solution" / "solve.sh").write_text("#!/usr/bin/env bash\ntouch repaired\n", encoding="utf-8")
    checksum = Task(task).checksum
    if mode == "separate":
        _record_reproducibility(task_dir)
    control_source = task_dir / "private" / "negative_agent.py"
    control_source.write_text("# Synthetic incomplete repair implementation.\n", encoding="utf-8")
    for job_number, arm, reward, exception in (
        (1, "nop-1", nop_reward, None),
        (2, "nop-2", nop_reward, None),
        (3, "oracle-1", oracle_reward, oracle_exception),
        (4, "oracle-2", oracle_reward, oracle_exception),
        (5, "negative-1", 0.0, None),
    ):
        proof_arm = arm.rsplit("-", 1)[0]
        if mode == "separate":
            extra = (
                [
                    "--negative-agent",
                    "negative_agent:IncompleteRepair",
                    "--negative-source",
                    "private/negative_agent.py",
                    "--negative-rationale",
                    "Repair only one of the required fixture records.",
                ]
                if proof_arm == "negative"
                else []
            )
            code, result = _run(
                "record-run-inputs",
                "--task-dir",
                str(task_dir),
                "--job-dir",
                f"private/jobs/{arm}",
                "--arm",
                proof_arm,
                *extra,
            )
            assert code == 0, result
        trial = task_dir / "private" / "jobs" / arm / "task__trial"
        trial.mkdir(parents=True)
        _write_json(
            trial / "result.json",
            {
                "task_checksum": checksum,
                "verifier_environment_mode": mode,
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": exception,
                "config": {
                    "job_id": f"job-{job_number}",
                    "agent": {"name": proof_arm if proof_arm != "negative" else "negative_agent:IncompleteRepair"},
                },
            },
        )
    if record_validation:
        code, result = _run(
            "record-validation",
            "--task-dir",
            str(task_dir),
            "--nop-job-dir",
            "private/jobs/nop-1",
            "--nop-job-dir",
            "private/jobs/nop-2",
            "--oracle-job-dir",
            "private/jobs/oracle-1",
            "--oracle-job-dir",
            "private/jobs/oracle-2",
            "--negative-job-dir",
            "private/jobs/negative-1",
            "--harbor-version",
            "0.21.0",
        )
        assert code == 0, result


def test_init_creates_private_gitignored_workspace(tmp_path: Path) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"

    code, result = _run("init", "--root", str(root), "--task-id", "repair-fixture")

    assert code == 0, result
    task_dir = root / "repair-fixture"
    assert result == {"gitignored": True, "status": "pending", "task_dir": str(task_dir)}
    assert (root / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == _SUMMARY_SCHEMA
    assert summary["status"] == "pending"
    if os.name == "posix":
        assert root.stat().st_mode & 0o777 == 0o700
        assert task_dir.stat().st_mode & 0o777 == 0o700
        assert (task_dir / "summary.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("symlink", [False, True])
def test_init_preserves_unexpected_gitignore(tmp_path: Path, symlink: bool) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"
    root.mkdir(parents=True)
    ignore = root / ".gitignore"
    original = "*\n!.gitignore\n" if symlink else "existing-user-rule\n"
    target = tmp_path / "existing-ignore" if symlink else ignore
    target.write_text(original, encoding="utf-8")
    if symlink:
        ignore.symlink_to(target)

    code, result = _run("init", "--root", str(root), "--task-id", "repair-fixture")

    assert code == 1
    assert "workspace .gitignore" in result["error"]
    assert target.read_text(encoding="utf-8") == original
    assert ignore.is_symlink() is symlink
    assert not (root / "repair-fixture").exists()


def test_init_refuses_to_write_outside_eval_author(tmp_path: Path) -> None:
    code, result = _run("init", "--root", str(tmp_path / "elsewhere"), "--task-id", "repair-fixture")

    assert code == 1
    assert "under a .eval-author directory" in result["error"]
    assert not (tmp_path / "elsewhere").exists()


def test_prepare_preserves_original_and_writes_text_only_redacted_atif(tmp_path: Path) -> None:
    task_dir, source = _workspace(tmp_path)

    private = task_dir / "private" / "source.atif.json"
    safe = task_dir / "safe" / "trace.atif.json"
    privacy = json.loads((task_dir / "safe" / "privacy.json").read_text(encoding="utf-8"))
    safe_text = safe.read_text(encoding="utf-8")
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))

    assert private.read_bytes() == source.read_bytes()
    assert "jane@example.com" not in safe_text
    assert "/home/jane" not in safe_text
    assert "10.2.3.4" not in safe_text
    assert "+1 (303) 555-0119" not in safe_text
    assert "123-45-6789" not in safe_text
    assert "session-1" not in safe_text
    assert "call-1" not in safe_text
    assert privacy["contextual_review_required"] is True
    assert privacy["contextual_review_complete"] is False
    assert set(privacy["deterministic_redactions"]) >= {
        "email",
        "home_path",
        "identifier",
        "ipv4",
        "phone",
        "ssn",
    }
    assert summary["source"]["original_sha256"] == f"sha256:{hashlib.sha256(private.read_bytes()).hexdigest()}"
    assert summary["source"]["safe_sha256"] == f"sha256:{hashlib.sha256(safe.read_bytes()).hexdigest()}"
    assert (task_dir / "private" / "canonical.atif.json").is_file()
    assert (task_dir / "private" / "privacy-audit.json").is_file()


def test_prepare_reports_non_object_observation(tmp_path: Path) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"
    code, result = _run("init", "--root", str(root), "--task-id", "repair-fixture")
    assert code == 0, result
    task_dir = Path(result["task_dir"])
    source = tmp_path / "source.atif.json"
    payload = _atif()
    payload["steps"][1]["observation"] = "not-an-object"
    _write_json(source, payload)

    code, result = _run(
        "prepare",
        "--task-dir",
        str(task_dir),
        "--atif",
        str(source),
        "--source-kind",
        "atif",
    )

    assert code == 1
    assert result["error"] == "trajectory.steps[1].observation must be an object"


def test_image_only_instruction_blocks_candidate_but_can_be_no_candidate(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path, image_only=True)
    _candidate(task_dir)
    _review_privacy(task_dir)

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )

    assert code == 1
    assert "unresolved non-text evidence" in result["error"]
    privacy = json.loads((task_dir / "safe" / "privacy.json").read_text(encoding="utf-8"))
    assert privacy["contextual_review_complete"] is True

    _candidate(task_dir, status="no_candidate")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")
    assert code == 0, result


def test_no_candidate_summary_records_reasons_and_lessons(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status="no_candidate")

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "no_candidate",
        "--worked-well",
        "The instruction was recoverable.",
        "--did-not-work",
        "The expected outcome was absent.",
    )

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "no_candidate"
    assert summary["reasons"] == ["missing_outcome"]
    assert summary["worked_well"] == ["The instruction was recoverable."]
    assert summary["did_not_work"] == ["The expected outcome was absent."]
    assert "`no_candidate`" in (task_dir / "summary.md").read_text(encoding="utf-8")
    code, check = _run("check", "--task-dir", str(task_dir))
    assert code == 0, check
    assert check["valid"] is True


def test_summary_captures_ground_truth_and_proprietary_software(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    expected = task_dir / "private" / "ground-truth" / "expected.json"
    _write_json(expected, {"fixture": "repaired"})
    expected.chmod(0o600)
    _candidate(
        task_dir,
        status="no_candidate",
        ground_truth={
            "availability": "available",
            "use": "comparison_only",
            "artifacts": [
                {
                    "kind": "expected_output",
                    "path": "private/ground-truth/expected.json",
                    "sha256": f"sha256:{hashlib.sha256(expected.read_bytes()).hexdigest()}",
                    "provenance": {
                        "kind": "external",
                        "step_ids": [],
                        "uri": "https://example.test/fixtures/expected.json",
                        "revision": "abc123",
                        "source_id": None,
                    },
                    "notes": "The trace records this expected fixture state.",
                }
            ],
            "absence_reason": None,
        },
        software_requirements=[
            {
                "name": "ExampleCAD",
                "category": "desktop_application",
                "required": True,
                "version": "2026",
                "license": "proprietary",
                "availability": "unavailable",
                "redistributable": False,
                "provenance": {
                    "kind": "atif_step",
                    "step_ids": [1, 2],
                    "uri": None,
                    "revision": None,
                    "source_id": None,
                },
                "notes": "The workflow requires its native file format and runtime.",
            }
        ],
    )

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate"]["ground_truth"]["availability"] == "available"
    assert summary["candidate"]["ground_truth"]["artifacts"][0]["kind"] == "expected_output"
    assert summary["candidate"]["software_requirements"][0]["license"] == "proprietary"
    markdown = (task_dir / "summary.md").read_text(encoding="utf-8")
    assert "Retained artifacts: 1" in markdown
    assert "ExampleCAD (desktop_application)" in markdown
    assert "license=proprietary" in markdown
    assert "availability=unavailable" in markdown


def test_candidate_rejects_required_unavailable_software(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(
        task_dir,
        software_requirements=[
            {
                "name": "ExampleCAD",
                "category": "desktop_application",
                "required": True,
                "version": None,
                "license": "commercial",
                "availability": "unavailable",
                "redistributable": False,
                "provenance": {
                    "kind": "atif_step",
                    "step_ids": [1],
                    "uri": None,
                    "revision": None,
                    "source_id": None,
                },
                "notes": "No licensed runtime is available in the task environment.",
            }
        ],
    )

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )

    assert code == 1
    assert "candidate requires unavailable software: ExampleCAD" in result["error"]


def _software(*, required: bool, availability: str) -> dict[str, Any]:
    return {
        "name": "ExampleRuntime",
        "category": "cli",
        "required": required,
        "version": None,
        "license": "open_source",
        "availability": availability,
        "redistributable": True,
        "provenance": {"kind": "atif_step", "step_ids": [1], "uri": None, "revision": None, "source_id": None},
        "notes": "Runtime availability recorded from the task evidence.",
    }


@pytest.mark.parametrize(
    "required,availability,nop_reward,record_validation,expected",
    [
        (True, "unknown", 0.0, True, "unproven"),
        (False, "unknown", 0.0, True, "ready"),
        (True, "available", 0.0, True, "ready"),
        (True, "installable", 0.0, True, "ready"),
        (True, "unknown", 1.0, True, "failed"),
        (True, "unknown", 0.0, False, "unproven"),
    ],
)
def test_software_availability_gates_readiness(
    tmp_path: Path, required: bool, availability: str, nop_reward: float, record_validation: bool, expected: str
) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, software_requirements=[_software(required=required, availability=availability)])
    _ready_environment(task_dir, nop_reward=nop_reward, record_validation=record_validation)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate", "--human-reviewed")
    assert code == 0, result
    assert result["environment_status"] == expected
    code, result = _run("check", "--task-dir", str(task_dir))
    assert code == 0, result
    _review_publication(task_dir)
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result
    assert json.loads((output / "result.json").read_text())["environment"]["status"] == expected


def test_check_rejects_ready_claim_with_unknown_required_software(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, software_requirements=[_software(required=True, availability="unknown")])
    _ready_environment(task_dir)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate", "--human-reviewed")
    assert code == 0, result
    summary_path = task_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["environment"]["status"] = "ready"
    _write_json(summary_path, summary)
    markdown = task_dir / "summary.md"
    markdown.write_text(markdown.read_text().replace("Environment: `unproven`", "Environment: `ready`"))
    code, result = _run("check", "--task-dir", str(task_dir))
    assert code == 1
    assert any("software" in error for error in result["errors"])
    code, result = _run("prepare-publication", "--task-dir", str(task_dir))
    assert code == 1
    assert "software" in result["error"]


def test_ground_truth_digest_must_match_retained_artifact(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    expected = task_dir / "private" / "ground-truth" / "expected.json"
    _write_json(expected, {"fixture": "repaired"})
    expected.chmod(0o600)
    _candidate(
        task_dir,
        status="no_candidate",
        ground_truth={
            "availability": "available",
            "use": "verification",
            "artifacts": [
                {
                    "kind": "expected_output",
                    "path": "private/ground-truth/expected.json",
                    "sha256": f"sha256:{'0' * 64}",
                    "provenance": {
                        "kind": "atif_step",
                        "step_ids": [2],
                        "uri": None,
                        "revision": None,
                        "source_id": None,
                    },
                    "notes": "The trace records this expected fixture state.",
                }
            ],
            "absence_reason": None,
        },
    )

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")

    assert code == 1
    assert "sha256 does not match the retained artifact" in result["error"]


def test_ground_truth_path_cannot_escape_private_directory(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    outside = task_dir / "private" / "outside.json"
    _write_json(outside, {"fixture": "repaired"})
    outside.chmod(0o600)
    _candidate(
        task_dir,
        status="no_candidate",
        ground_truth={
            "availability": "available",
            "use": "verification",
            "artifacts": [
                {
                    "kind": "expected_output",
                    "path": "private/ground-truth/../outside.json",
                    "sha256": f"sha256:{hashlib.sha256(outside.read_bytes()).hexdigest()}",
                    "provenance": {
                        "kind": "atif_step",
                        "step_ids": [2],
                        "uri": None,
                        "revision": None,
                        "source_id": None,
                    },
                    "notes": "The trace records this expected fixture state.",
                }
            ],
            "absence_reason": None,
        },
    )

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")

    assert code == 1
    assert "path escapes private/ground-truth/" in result["error"]


def test_ready_candidate_requires_privacy_review_and_both_harbor_arms(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )
    assert code == 1
    assert "review-privacy" in result["error"]

    _review_privacy(task_dir, reviewer_kind="human")
    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
        "--human-reviewed",
        "--worked-well",
        "NOP and Oracle produced the required rewards.",
    )

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "candidate"
    assert summary["environment"] == {
        "path": "task",
        "status": "ready",
        "technical_status": "passed",
        "review_status": "human_reviewed",
        "validation": "validation.json",
        "verifier_environment_mode": "separate",
        "isolation_status": "isolated",
    }
    assert summary["privacy"]["contextual_review_complete"] is True
    code, check = _run("check", "--task-dir", str(task_dir))
    assert code == 0, check


def test_ready_candidate_rejects_wrong_arm_reward(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)
    validation_path = task_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["runs"]["nop"][0]["reward"] = 1
    _write_json(validation_path, validation)
    _review_privacy(task_dir)

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )

    assert code == 1
    assert "differs from the retained Harbor result evidence" in result["error"]


def test_ready_candidate_requires_reviewer_facing_readme(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)
    _review_privacy(task_dir)
    (task_dir / "task" / "README.md").unlink()

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )

    assert code == 1
    assert "candidate environment is missing: task/README.md" in result["error"]


def test_ready_candidate_requires_substantive_readme_sections(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)
    _review_privacy(task_dir)
    readme = task_dir / "task" / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "The test checks the resulting fixture state without inspecting the agent's implementation.", ""
        ),
        encoding="utf-8",
    )

    code, result = _run(
        "finalize",
        "--task-dir",
        str(task_dir),
        "--status",
        "candidate",
    )

    assert code == 1
    assert "Verification explanation" in result["error"]


def test_reproducibility_records_complete_task_tree_and_portability(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)

    report = json.loads((task_dir / "reproducibility.json").read_text(encoding="utf-8"))

    assert report["schema"] == "nemo.eval_author.trace_environment_reproducibility.v3"
    assert report["task_tree_sha256"].startswith("sha256:")
    assert report["file_count"] == 7
    assert report["portability"]["state"] == "image_pinned_recipe"
    assert report["portability"]["dependency_closure"] == "unverified"
    assert report["network"] == {"agent": "no-network", "verifier": "no-network"}
    assert report["contamination"]["passed"] is True


def test_reproducibility_digest_covers_executable_bits(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    _review_privacy(task_dir)
    solve = task_dir / "task" / "solution" / "solve.sh"
    solve.chmod(solve.stat().st_mode | stat.S_IXUSR)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "reproducibility.json differs from the current task tree" in result["error"]


def test_reproducibility_retains_failed_contamination_evidence(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    target = task_dir / "task/environment/.git/config"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('[remote "origin"]\n', encoding="utf-8")
    (task_dir / "reproducibility.json").unlink()

    code, result = _run("record-reproducibility", "--task-dir", str(task_dir))

    assert code == 1
    assert result["contamination_findings"] == 1
    report = json.loads((task_dir / "reproducibility.json").read_text(encoding="utf-8"))
    assert report["contamination"]["passed"] is False
    assert report["contamination"]["findings"][0]["code"] == "git_metadata_in_agent_context"


def test_candidate_requires_no_network_agent_environment(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            '[environment]\nnetwork_mode = "no-network"', '[environment]\nnetwork_mode = "public"'
        ),
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "[environment].network_mode must be no-network" in result["error"]


def test_reproducibility_rejects_solution_file_in_agent_context(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task" / "environment" / "answer.sh").write_bytes(
        (task_dir / "task" / "solution" / "solve.sh").read_bytes()
    )
    (task_dir / "reproducibility.json").unlink()

    code, result = _run("record-reproducibility", "--task-dir", str(task_dir))

    assert code == 1
    assert result["contamination_findings"] == 1
    report = json.loads((task_dir / "reproducibility.json").read_text(encoding="utf-8"))
    assert report["contamination"]["findings"][0]["code"] == "solution_in_agent_context"


def test_validation_requires_independent_repeat_and_negative_jobs(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)

    code, result = _run(
        "record-validation",
        "--task-dir",
        str(task_dir),
        "--nop-job-dir",
        "private/jobs/nop-1",
        "--oracle-job-dir",
        "private/jobs/oracle-1",
        "--negative-job-dir",
        "private/jobs/negative-1",
        "--harbor-version",
        "0.21.0",
    )

    assert code == 1
    assert "at least 2 independent nop Harbor jobs" in result["error"]


def test_validation_requires_distinct_harbor_job_ids(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    duplicate = task_dir / "private" / "jobs" / "nop-2" / "task__trial" / "result.json"
    payload = json.loads(duplicate.read_text(encoding="utf-8"))
    payload["config"]["job_id"] = "job-1"
    _write_json(duplicate, payload)

    code, result = _run(
        "record-validation",
        "--task-dir",
        str(task_dir),
        "--nop-job-dir",
        "private/jobs/nop-1",
        "--nop-job-dir",
        "private/jobs/nop-2",
        "--oracle-job-dir",
        "private/jobs/oracle-1",
        "--oracle-job-dir",
        "private/jobs/oracle-2",
        "--negative-job-dir",
        "private/jobs/negative-1",
        "--harbor-version",
        "0.21.0",
    )

    assert code == 1
    assert "config.job_id values must be distinct" in result["error"]


def test_check_detects_changed_safe_evidence(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status="no_candidate")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")
    assert code == 0, result
    with (task_dir / "safe" / "trace.atif.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    code, result = _run("check", "--task-dir", str(task_dir))

    assert code == 1
    assert result["valid"] is False
    assert "safe_path is missing or its digest changed" in result["errors"]


def test_prepare_narrowly_repairs_provider_redaction_quote(tmp_path: Path) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"
    code, result = _run("init", "--root", str(root), "--task-id", "repair-quote")
    assert code == 0, result
    source = tmp_path / "broken.atif.json"
    source.write_text(
        '{"schema_version":"ATIF-v1.7","agent":{"name":"agent"},"steps":'
        '[{"step_id":1,"source":"user","message":"prefix \\"PASSWORD=<redacted>" suffix"}]}',
        encoding="utf-8",
    )

    code, result = _run(
        "prepare",
        "--task-dir",
        result["task_dir"],
        "--atif",
        str(source),
        "--source-kind",
        "atif",
    )

    assert code == 0, result
    assert result["normalization_count"] == 1
    summary = json.loads((root / "repair-quote" / "summary.json").read_text(encoding="utf-8"))
    assert summary["source"]["normalizations"][0]["kind"] == "escape_provider_redaction_placeholder_quote"
    assert (root / "repair-quote" / "private" / "source.atif.json").read_bytes() == source.read_bytes()


def test_prepare_bounds_string_encoded_images_and_audits_context(tmp_path: Path) -> None:
    root = tmp_path / ".eval-author" / "trace-environments"
    code, result = _run("init", "--root", str(root), "--task-id", "bound-image")
    assert code == 0, result
    task_dir = Path(result["task_dir"])
    source = tmp_path / "image.atif.json"
    payload = _atif()
    payload["steps"][0]["message"] = [
        {
            "type": "text",
            "text": "Use https://api.example.test/path for Example Labs at 123 Main Street and John Smith.",
        },
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "A" * 100_001},
        },
    ]
    payload["steps"][1]["observation"]["results"][0]["content"] = (
        'before {"type":"image","source":{"type":"base64","media_type":"image/png","data":"DATA"}} after'
    )
    payload["steps"][1]["message"] = "Call service.default.svc.cluster.local"
    _write_json(source, payload)

    code, result = _run(
        "prepare",
        "--task-dir",
        str(task_dir),
        "--atif",
        str(source),
        "--source-kind",
        "atif",
    )

    assert code == 0, result
    canonical = json.loads((task_dir / "private" / "canonical.atif.json").read_text(encoding="utf-8"))
    assert canonical["steps"][0]["message"][1]["source"]["data"] == ""
    assert canonical["steps"][1]["observation"]["results"][0]["content"][1]["source"]["data"] == ""
    safe_text = (task_dir / "safe" / "trace.atif.json").read_text(encoding="utf-8")
    assert "svc.cluster.local" not in safe_text
    audit = json.loads((task_dir / "private" / "privacy-audit.json").read_text(encoding="utf-8"))
    assert audit["url_hosts"] == ["api.example.test"]
    assert {finding["kind"] for finding in audit["candidate_findings"]} >= {
        "organization",
        "person_name",
        "street_address",
    }


@pytest.mark.parametrize("encoded_size", [12, 100_001])
@pytest.mark.parametrize("location", ["observation", "text_part", "extra", "message"])
def test_prepare_omits_repeated_image_metadata_without_losing_prose(
    tmp_path: Path, encoded_size: int, location: str
) -> None:
    code, initialized = _run("init", "--root", str(tmp_path / ".eval-author"), "--task-id", "image-metadata")
    assert code == 0, initialized
    task_dir = Path(initialized["task_dir"])
    payload = _atif()
    metadata = {"type": "image", "file": {"base64": "A" * encoded_size, "caption": "synthetic diagram"}}
    embedded = "before [metadata] " + json.dumps(metadata) + " after"
    step = payload["steps"][1]
    result = step["observation"]["results"][0]
    if location == "observation":
        result["content"] = embedded
    elif location == "text_part":
        result["content"] = [{"type": "text", "text": embedded}]
    elif location == "extra":
        result["extra"] = {"image_metadata": metadata}
    else:
        step["message"] = embedded
    source = tmp_path / "source.atif.json"
    _write_json(source, payload)
    code, report = _run("prepare", "--task-dir", str(task_dir), "--atif", str(source), "--source-kind", "atif")
    assert code == 0, report
    assert (task_dir / "private/source.atif.json").read_bytes() == source.read_bytes()
    for artifact in ("private/canonical.atif.json", "safe/trace.atif.json"):
        text = (task_dir / artifact).read_text()
        assert "A" * encoded_size not in text
        assert "synthetic diagram" in text
        if location != "extra":
            assert "before [metadata]" in text and " after" in text
    summary = json.loads((task_dir / "summary.json").read_text())
    operation = summary["source"]["normalizations"][0]
    assert operation["metadata_field_count"] == 1
    assert operation["metadata_omitted_characters"] == encoded_size


def test_shared_verifier_is_rejected(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, mode="shared", record_validation=False)
    _review_privacy(task_dir, reviewer_kind="human")

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate", "--human-reviewed")

    assert code == 1
    assert "must explicitly set [verifier].environment_mode to separate" in result["error"]


def test_candidate_requires_explicit_verifier_environment_mode(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace('environment_mode = "separate"\n', ""),
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "must explicitly set [verifier].environment_mode to separate" in result["error"]


def test_candidate_requires_explicit_verifier_environment(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            '\n[verifier.environment]\nnetwork_mode = "no-network"\n',
            "\n",
        ),
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "must contain an explicit [verifier.environment] table" in result["error"]


def test_verifier_environment_must_be_no_network(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8").replace(
            '[verifier.environment]\nnetwork_mode = "no-network"\n',
            '[verifier.environment]\nnetwork_mode = "public"\n',
        ),
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "[verifier.environment].network_mode must be no-network" in result["error"]


def test_separate_verifier_can_inherit_image_without_tests_dockerfile(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task" / "tests" / "Dockerfile").unlink()
    _record_reproducibility(task_dir)
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["environment"]["status"] == "unproven"
    assert summary["environment"]["isolation_status"] == "isolated"


def test_step_verifier_cannot_override_separate_mode(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8")
        + '\n[[steps]]\nname = "grade"\n[steps.verifier]\nenvironment_mode = "shared"\n',
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "step 1 verifier must not override separate mode" in result["error"]


def test_step_verifier_environment_must_be_no_network(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8")
        + '\n[[steps]]\nname = "grade"\n[steps.verifier.environment]\nnetwork_mode = "public"\n',
        encoding="utf-8",
    )
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 1
    assert "[steps.verifier.environment] for step 1 must set network_mode to no-network" in result["error"]


def test_step_can_define_a_separate_no_network_verifier_environment(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    task_toml = task_dir / "task" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8")
        + '\n[[steps]]\nname = "grade"\n[steps.verifier]\nenvironment_mode = "separate"\n'
        + '[steps.verifier.environment]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    _record_reproducibility(task_dir)
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["environment"]["status"] == "unproven"
    assert summary["environment"]["isolation_status"] == "isolated"


def test_failed_harbor_proof_is_validated_and_attached(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, oracle_reward=0.0, oracle_exception={"type": "RuntimeError"})
    _review_privacy(task_dir)

    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")

    assert code == 0, result
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["environment"]["status"] == "failed"
    assert summary["environment"]["technical_status"] == "failed"
    assert summary["environment"]["validation"] == "validation.json"
    code, check = _run("check", "--task-dir", str(task_dir))
    assert code == 0, check


def test_export_uses_a_strict_publication_whitelist(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)
    _review_privacy(task_dir, reviewer_kind="human")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate", "--human-reviewed")
    assert code == 0, result
    output = tmp_path / "product"

    _review_publication(task_dir)
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))

    assert code == 0, result
    assert (output / "candidate.json").is_file()
    assert (output / "reproducibility.json").is_file()
    assert (output / "result.json").is_file()
    assert (output / "task").is_dir()
    assert not (output / "validation.json").exists()
    assert not (output / "safe").exists()
    assert not (output / "private").exists()
    product = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert product["schema"] == "nemo.eval_author.trace_environment_product.v3"
    assert product["reproducibility"]["contamination_passed"] is True
    assert product["technical_validation"]["minimum_runs"] == {"negative": 1, "nop": 2, "oracle": 2}
    assert product["technical_validation"]["distinct_jobs"] is True
    assert product["technical_validation"]["container_freshness"] == "unverified"
    assert "fresh_jobs" not in product["technical_validation"]
    assert product["reproducibility"]["dependency_closure"] == "unverified"


@pytest.mark.parametrize("status", ["candidate", "no_candidate"])
def test_export_requires_publication_review_even_after_trace_review(tmp_path: Path, status: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status=status)
    if status == "candidate":
        _ready_environment(task_dir, record_validation=False)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", status)
    assert code == 0, result
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 1
    assert "review-publication" in result["error"]
    assert not output.exists()


@pytest.mark.parametrize("reviewer_kind", ["agent", "human"])
def test_no_candidate_publication_reviews_exact_product(tmp_path: Path, reviewer_kind: str) -> None:
    task_dir, _ = _workspace(tmp_path, image_only=True)
    _candidate(task_dir, status="no_candidate")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")
    assert code == 0, result
    preview = _review_publication(task_dir, reviewer_kind=reviewer_kind)
    preview_dir = Path(preview["preview_dir"])
    assert preview["file_count"] == 2
    assert stat.S_IMODE(preview_dir.stat().st_mode) == 0o700
    receipt = task_dir / "private/publication-review.json"
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert json.loads(receipt.read_text())["reviewer_kind"] == reviewer_kind
    code, second = _run("prepare-publication", "--task-dir", str(task_dir))
    assert code == 0, second
    assert second == preview
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result
    assert result["files"] == ["candidate.json", "result.json"]
    for path in output.iterdir():
        assert path.read_bytes() == (preview_dir / path.name).read_bytes()
    product = json.loads((output / "result.json").read_text())
    assert product["status"] == "no_candidate"
    assert product["privacy"]["contextual_review_complete"] is False


@pytest.mark.parametrize("change", ["candidate", "task", "manifest", "executable", "empty_directory"])
def test_publication_review_is_invalidated_by_changed_export(tmp_path: Path, change: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")
    assert code == 0, result
    preview = _review_publication(task_dir)
    if change in {"candidate", "manifest"}:
        path = task_dir / ("candidate.json" if change == "candidate" else "reproducibility.json")
        # Equivalent JSON still changes the exact bytes approved for publication.
        path.write_text(path.read_text() + "\n")
    elif change == "task":
        path = task_dir / "task/instruction.md"
        path.write_text(path.read_text() + "Keep the repair local.\n")
        _record_reproducibility(task_dir)
    elif change == "executable":
        (task_dir / "task/tests/test.sh").chmod(0o744)
        _record_reproducibility(task_dir)
    else:
        (task_dir / "task/empty").mkdir()
        _record_reproducibility(task_dir)
    code, result = _run("check", "--task-dir", str(task_dir))
    assert code == 0, result
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 1
    assert "publication review is stale" in result["error"]
    assert not output.exists()
    code, result = _run(
        "review-publication",
        "--task-dir",
        str(task_dir),
        "--preview-dir",
        preview["preview_dir"],
        "--sha256",
        preview["sha256"],
        "--reviewer-kind",
        "agent",
        "--note",
        "Old preview",
    )
    assert code == 1
    assert "preview is stale" in result["error"]
    updated = _review_publication(task_dir)
    assert updated["sha256"] != preview["sha256"]
    assert Path(preview["preview_dir"]).is_dir()
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result


def test_publication_review_binds_derived_result_json(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status="no_candidate")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")
    assert code == 0, result
    _review_publication(task_dir)
    # This changes only the public result, not candidate.json or the safe trace.
    _review_privacy(task_dir)
    code, result = _run("check", "--task-dir", str(task_dir))
    assert code == 0, result
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 1
    assert "publication review is stale" in result["error"]
    assert not output.exists()


@pytest.mark.parametrize("change", ["bytes", "mode", "symlink", "extra_file", "missing_file"])
def test_publication_review_rejects_tampered_preview(tmp_path: Path, change: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status="no_candidate")
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "no_candidate")
    assert code == 0, result
    code, preview = _run("prepare-publication", "--task-dir", str(task_dir))
    assert code == 0, preview
    path = Path(preview["preview_dir"]) / "candidate.json"
    if change == "bytes":
        path.write_text(path.read_text() + "\n")
    elif change == "mode":
        path.chmod(0o744)
    elif change == "symlink":
        path.unlink()
        path.symlink_to(task_dir / "candidate.json")
    elif change == "extra_file":
        (path.parent / "extra.txt").write_text("Unreviewed internal project detail.")
    else:
        path.unlink()
    code, result = _run(
        "review-publication",
        "--task-dir",
        str(task_dir),
        "--preview-dir",
        preview["preview_dir"],
        "--sha256",
        preview["sha256"],
        "--reviewer-kind",
        "agent",
        "--note",
        "Changed preview",
    )
    assert code == 1
    assert "digest" in result["error"] or "symlink" in result["error"]
    assert not (task_dir / "private/publication-review.json").exists()


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="extended attributes require OS support")
@pytest.mark.parametrize("status", ["candidate", "no_candidate"])
def test_publication_omits_source_extended_attributes(tmp_path: Path, status: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir, status=status)
    if status == "candidate":
        _ready_environment(task_dir, record_validation=False)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", status)
    assert code == 0, result
    sources = [task_dir / "candidate.json"]
    if status == "candidate":
        sources.extend([task_dir / "reproducibility.json", task_dir / "task", task_dir / "task/tests/test.sh"])
    attribute = "user.fixture_note"
    for path in sources:
        os.setxattr(path, attribute, b"fixture metadata before review")
    preview = _review_publication(task_dir)
    preview_dir = Path(preview["preview_dir"])
    for path in sources:
        assert attribute not in os.listxattr(preview_dir / path.relative_to(task_dir))
        os.setxattr(path, attribute, b"fixture metadata changed after review")
    output = tmp_path / "product"
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result
    for path in sources:
        assert os.getxattr(path, attribute) == b"fixture metadata changed after review"
        assert attribute not in os.listxattr(output / path.relative_to(task_dir))
    for path in output.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (preview_dir / path.relative_to(output)).read_bytes()


@pytest.mark.parametrize("executable_bits", [0, stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH, 0o111])
def test_export_preserves_executable_bits_and_task_digest(tmp_path: Path, executable_bits: int) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    scripts = (Path("tests/test.sh"), Path("solution/solve.sh"))
    for relative in scripts:
        (task_dir / "task" / relative).chmod(0o640 | executable_bits)
    _record_reproducibility(task_dir)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")
    assert code == 0, result
    output = tmp_path / "product"
    _review_publication(task_dir)
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result

    for relative in scripts:
        exported = output / "task" / relative
        assert stat.S_IMODE(exported.stat().st_mode) == 0o644 | executable_bits
        assert exported.read_bytes() == (task_dir / "task" / relative).read_bytes()
    if executable_bits & stat.S_IXUSR:
        execution = subprocess.run([str(output / "task/tests/test.sh")], capture_output=True, check=False)
        assert execution.returncode == 0, execution.stderr

    # Independently rescan the actual exported tree through the public command.
    code, result = _run("init", "--root", str(tmp_path / ".eval-author/imported"), "--task-id", "imported-task")
    assert code == 0, result
    imported = Path(result["task_dir"])
    shutil.copytree(output / "task", imported / "task")
    _record_reproducibility(imported)
    expected = json.loads((task_dir / "reproducibility.json").read_text())
    actual = json.loads((imported / "reproducibility.json").read_text())
    published = json.loads((output / "reproducibility.json").read_text())
    product = json.loads((output / "result.json").read_text())
    assert actual == published == expected
    assert product["reproducibility"]["task_tree_sha256"] == actual["task_tree_sha256"]


@pytest.mark.parametrize(
    "field,value", [("container_freshness", "verified"), ("distinct_jobs", False), ("fresh_jobs", True)]
)
def test_validation_rejects_unsupported_job_evidence_claims(tmp_path: Path, field: str, value: object) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir)
    path = task_dir / "validation.json"
    validation = json.loads(path.read_text())
    assert validation["distinct_jobs"] is True
    assert validation["container_freshness"] == "unverified"
    assert "fresh_jobs" not in validation
    validation[field] = value
    _write_json(path, validation)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")
    assert code == 1
    assert "versioned contract" in result["error"]


def test_image_pinning_does_not_claim_dependency_closure(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _candidate(task_dir)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task/environment/Dockerfile").write_text(
        "FROM example.invalid/runtime@sha256:" + "a" * 64 + "\nRUN pip install example-package\n"
    )
    _record_reproducibility(task_dir)
    _review_privacy(task_dir)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")
    assert code == 0, result
    output = tmp_path / "product"
    _review_publication(task_dir)
    code, result = _run("export", "--task-dir", str(task_dir), "--output-dir", str(output))
    assert code == 0, result
    manifest = json.loads((output / "reproducibility.json").read_text())
    product = json.loads((output / "result.json").read_text())
    assert manifest["portability"]["state"] == "image_pinned_recipe"
    assert manifest["portability"]["dependency_closure"] == "unverified"
    assert product["reproducibility"]["portability_state"] == "image_pinned_recipe"
    assert product["reproducibility"]["dependency_closure"] == "unverified"


def test_batch_prepare_is_resumable_and_reports_full_denominator(tmp_path: Path) -> None:
    source_one = tmp_path / "one.json"
    source_two = tmp_path / "two.json"
    _write_json(source_one, _atif())
    _write_json(source_two, _atif())
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": "nemo.eval_author.trace_environment_batch.v1",
            "members": [
                {"task_id": "task-one", "atif": "one.json", "source_kind": "atif"},
                {"task_id": "task-two", "atif": "two.json", "source_kind": "atif"},
            ],
        },
    )
    root = tmp_path / ".eval-author" / "trace-environments"

    code, first = _run("batch-prepare", "--root", str(root), "--manifest", str(manifest))
    assert code == 0, first
    assert first["denominator"] == 2
    assert first["counts"] == {"prepared": 2}
    code, second = _run("batch-prepare", "--root", str(root), "--manifest", str(manifest))
    assert code == 0, second
    assert second["counts"] == {"existing": 2}
    code, status = _run("batch-status", "--root", str(root), "--manifest", str(manifest))
    assert code == 0, status
    assert status["denominator"] == 2
    assert status["counts"] == {"pending_prepared": 2}


def test_batch_status_reports_malformed_workspace_without_aborting(tmp_path: Path) -> None:
    source_one = tmp_path / "one.json"
    source_two = tmp_path / "two.json"
    _write_json(source_one, _atif())
    _write_json(source_two, _atif())
    manifest = tmp_path / "manifest.json"
    manifest_payload = {
        "schema": "nemo.eval_author.trace_environment_batch.v1",
        "members": [
            {"task_id": "task-one", "atif": "one.json", "source_kind": "atif"},
            {"task_id": "task-two", "atif": "two.json", "source_kind": "atif"},
        ],
    }
    _write_json(manifest, manifest_payload)
    root = tmp_path / ".eval-author" / "trace-environments"
    code, prepared = _run("batch-prepare", "--root", str(root), "--manifest", str(manifest))
    assert code == 0, prepared
    malformed_summary = root / "task-two" / "summary.json"
    summary = json.loads(malformed_summary.read_text(encoding="utf-8"))
    summary["unexpected"] = True
    _write_json(malformed_summary, summary)
    manifest_payload["members"].append({"task_id": "task-three", "atif": "missing.json", "source_kind": "atif"})
    _write_json(manifest, manifest_payload)

    code, status = _run("batch-status", "--root", str(root), "--manifest", str(manifest))

    assert code == 1, status
    assert status["denominator"] == 3
    assert status["counts"] == {"invalid": 1, "missing": 1, "pending_prepared": 1}
    assert status["valid"] is False
    assert status["members"] == [
        {"task_id": "task-one", "status": "pending_prepared"},
        {
            "task_id": "task-two",
            "status": "invalid",
            "error": "summary fields do not match the versioned contract",
        },
        {"task_id": "task-three", "status": "missing"},
    ]


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "source.kind",
        "privacy.reviewer_kind",
        "environment.status",
        "environment.technical_status",
        "environment.review_status",
        "environment.validation",
        "environment.verifier_environment_mode",
        "environment.isolation_status",
    ],
)
@pytest.mark.parametrize("bad_value", [[], {}])
def test_batch_status_contains_invalid_enum_types(tmp_path: Path, field: str, bad_value: object) -> None:
    task_dir, source = _workspace(tmp_path)
    path = task_dir / "summary.json"
    summary = json.loads(path.read_text())
    target = summary
    keys = field.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = bad_value
    _write_json(path, summary)
    code, result = _run("init", "--root", str(task_dir.parent), "--task-id", "healthy")
    assert code == 0, result
    manifest = tmp_path / "batch.json"
    _write_json(
        manifest,
        {
            "schema": "nemo.eval_author.trace_environment_batch.v1",
            "members": [
                {"task_id": name, "atif": str(source), "source_kind": "atif"}
                for name in (task_dir.name, "healthy", "missing")
            ],
        },
    )
    code, report = _run("batch-status", "--root", str(task_dir.parent), "--manifest", str(manifest))
    assert code == 1
    assert report["denominator"] == 3
    assert report["valid"] is False
    assert report["counts"] == {"invalid": 1, "pending_unprepared": 1, "missing": 1}
    assert [row["task_id"] for row in report["members"]] == [task_dir.name, "healthy", "missing"]


def _record_existing_jobs(task_dir: Path) -> tuple[int, dict[str, Any]]:
    args = []
    for arm, count in (("nop", 2), ("oracle", 2), ("negative", 1)):
        for index in range(1, count + 1):
            args.extend([f"--{arm}-job-dir", f"private/jobs/{arm}-{index}"])
    return _run("record-validation", "--task-dir", str(task_dir), *args, "--harbor-version", "0.21.0")


@pytest.mark.parametrize("change", ["verifier", "executable", "instruction", "build_input", "new_file"])
def test_stale_jobs_cannot_prove_a_changed_task(tmp_path: Path, change: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    verifier = task_dir / "task/tests/test.sh"
    if change == "verifier":
        verifier.write_text("#!/bin/sh\nexit 99\n")
    elif change == "executable":
        verifier.chmod(verifier.stat().st_mode ^ stat.S_IXUSR)
    elif change == "instruction":
        (task_dir / "task/instruction.md").write_text("Repair a different fixture.\n")
    elif change == "build_input":
        (task_dir / "task/environment/Dockerfile").write_text("FROM scratch\nENV FIXTURE_REVISION=2\n")
    else:
        (task_dir / "task/environment/input.txt").write_text("New initial state.\n")
    _record_reproducibility(task_dir)
    code, result = _record_existing_jobs(task_dir)
    assert code == 1
    assert "pre-run task snapshot differs" in result["error"]
    assert not (task_dir / "validation.json").exists()


def test_results_must_match_current_harbor_checksum(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    for path in (task_dir / "private/jobs").glob("*/task__*/result.json"):
        result = json.loads(path.read_text())
        result["task_checksum"] = "f" * 64
        _write_json(path, result)
    code, result = _record_existing_jobs(task_dir)
    assert code == 1
    assert "Harbor result task checksum differs" in result["error"]


def test_run_inputs_cannot_be_attached_after_a_job_exists(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    code, result = _run(
        "record-run-inputs", "--task-dir", str(task_dir), "--job-dir", "private/jobs/nop-1", "--arm", "nop"
    )
    assert code == 1
    assert "before Harbor creates" in result["error"]


def test_missing_pre_run_inputs_rejects_historical_proof(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    for path in (task_dir / "private/run-inputs").glob("*.json"):
        path.unlink()
    code, result = _record_existing_jobs(task_dir)
    assert code == 1
    assert "record-run-inputs is required" in result["error"]


@pytest.mark.parametrize(
    "arm,agent",
    [
        ("nop-1", None),
        ("oracle-1", None),
        ("oracle-1", {"name": "nop"}),
        ("nop-1", {"name": "oracle"}),
        ("negative-1", {"name": "nop"}),
        ("negative-1", {"name": "oracle"}),
        ("negative-1", {"import_path": "harbor.agents.nop:NopAgent"}),
        ("negative-1", {"import_path": "other_agent:WrongControl"}),
        ("nop-1", {"name": "nop", "import_path": "other_agent:WrongControl"}),
    ],
)
def test_wrong_proof_agent_is_rejected(tmp_path: Path, arm: str, agent: object) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    path = task_dir / f"private/jobs/{arm}/task__trial/result.json"
    result = json.loads(path.read_text())
    result["config"]["agent"] = agent
    _write_json(path, result)
    code, result = _record_existing_jobs(task_dir)
    assert code == 1
    assert "agent" in result["error"]


def test_import_path_agent_identity_is_supported(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    identities = {
        "nop": "harbor.agents.nop:NopAgent",
        "oracle": "harbor.agents.oracle:OracleAgent",
        "negative": "negative_agent:IncompleteRepair",
    }
    for path in (task_dir / "private/jobs").glob("*/task__*/result.json"):
        arm = path.parents[1].name.rsplit("-", 1)[0]
        result = json.loads(path.read_text())
        result["config"]["agent"] = {"name": None, "import_path": identities[arm]}
        _write_json(path, result)
    code, result = _record_existing_jobs(task_dir)
    assert code == 0, result
    assert result["passed"] is True


def test_negative_source_mutation_invalidates_proof(tmp_path: Path) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "private/negative_agent.py").write_text("# Changed implementation\n")
    code, result = _record_existing_jobs(task_dir)
    assert code == 1
    assert "negative control source" in result["error"]


@pytest.mark.parametrize("agent", ["nop", "oracle", "harbor.agents.nop:NopAgent", "harbor.agents.oracle:OracleAgent"])
def test_builtin_agent_cannot_be_declared_negative(tmp_path: Path, agent: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    code, result = _run(
        "record-run-inputs",
        "--task-dir",
        str(task_dir),
        "--job-dir",
        "private/jobs/new-negative",
        "--arm",
        "negative",
        "--negative-agent",
        agent,
        "--negative-source",
        "private/negative_agent.py",
        "--negative-rationale",
        "Incomplete repair",
    )
    assert code == 1
    assert "distinct from NOP and Oracle" in result["error"]


@pytest.mark.parametrize(
    "copy_source,pinned",
    [
        ("example.invalid/tool:latest", False),
        ("example.invalid/tool@sha256:" + "a" * 64, True),
        ("builder", True),
        ("0", True),
        ("${TOOLS_IMAGE}", False),
    ],
)
def test_portability_tracks_external_copy_sources(tmp_path: Path, copy_source: str, pinned: bool) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task/environment/Dockerfile").write_text(
        f'FROM scratch AS builder\nFROM scratch\nCOPY --from="{copy_source}" /tool /tool\n'
    )
    _record_reproducibility(task_dir)
    report = json.loads((task_dir / "reproducibility.json").read_text())
    assert report["portability"]["state"] == ("image_pinned_recipe" if pinned else "local_only")
    copies = [image for image in report["portability"]["container_images"] if image["instruction"] == "copy"]
    assert len(copies) == 1
    assert copies[0]["reference"] == copy_source
    assert copies[0]["internal_stage"] == (copy_source in ("builder", "0"))


@pytest.mark.parametrize(
    "source,pinned,internal",
    [
        ("example.invalid/tool:latest", False, False),
        ("example.invalid/tool@sha256:" + "a" * 64, True, False),
        ("builder", True, True),
        ("0", True, True),
        ("${TOOLS_IMAGE}", False, False),
    ],
)
def test_portability_tracks_multiple_run_mounts(tmp_path: Path, source: str, pinned: bool, internal: bool) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task/environment/Dockerfile").write_text(
        "FROM scratch AS builder\nFROM scratch\n"
        "RUN --mount=type=bind,from=builder,target=/local \\\n"
        f"    --mount=type=bind,from={source},target=/tool true\n"
    )
    _record_reproducibility(task_dir)
    report = json.loads((task_dir / "reproducibility.json").read_text())
    assert report["portability"]["state"] == ("image_pinned_recipe" if pinned else "local_only")
    mounts = [image for image in report["portability"]["container_images"] if image["instruction"] == "run_mount"]
    assert [(item["reference"], item["internal_stage"]) for item in mounts] == [("builder", True), (source, internal)]


@pytest.mark.parametrize(
    "reference,pinned",
    [("docker/dockerfile:1", False), ("docker/dockerfile@sha256:" + "a" * 64, True), ("${FRONTEND}", False)],
)
@pytest.mark.parametrize("location", ["environment", "tests"])
def test_portability_inventories_dockerfile_frontend(
    tmp_path: Path, reference: str, pinned: bool, location: str
) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / f"task/{location}/Dockerfile").write_text(f"# syntax={reference}\nFROM scratch\n")
    _record_reproducibility(task_dir)
    report = json.loads((task_dir / "reproducibility.json").read_text())
    assert report["portability"]["state"] == ("image_pinned_recipe" if pinned else "local_only")
    frontends = [item for item in report["portability"]["container_images"] if item["instruction"] == "frontend"]
    assert len(frontends) == 1
    assert frontends[0]["reference"] == reference
    assert frontends[0]["immutable"] is pinned
    assert frontends[0]["internal_stage"] is False


@pytest.mark.parametrize("prefix", ["\n", "# ordinary comment\n", "FROM scratch\n"])
def test_portability_ignores_inactive_syntax_comments(tmp_path: Path, prefix: str) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    (task_dir / "task/environment/Dockerfile").write_text(prefix + "# syntax=docker/dockerfile:1\nFROM scratch\n")
    _record_reproducibility(task_dir)
    report = json.loads((task_dir / "reproducibility.json").read_text())
    assert report["portability"]["state"] == "image_pinned_recipe"
    assert not any(item["instruction"] == "frontend" for item in report["portability"]["container_images"])


@pytest.mark.parametrize(
    "location", ["environment", "verifier.environment", "steps.environment", "steps.verifier.environment"]
)
@pytest.mark.parametrize("pinned", [True, False])
def test_portability_tracks_configured_images(tmp_path: Path, location: str, pinned: bool) -> None:
    task_dir, _ = _workspace(tmp_path)
    _ready_environment(task_dir, record_validation=False)
    path = task_dir / "task/task.toml"
    config = path.read_text()
    image = "example.invalid/runtime@sha256:" + "b" * 64 if pinned else "example.invalid/runtime:latest"
    if location.startswith("steps."):
        config += f'\n[[steps]]\nname = "grade"\n[{location}]\nnetwork_mode = "no-network"\ndocker_image = "{image}"\n'
    else:
        config = config.replace(f"[{location}]", f'[{location}]\ndocker_image = "{image}"')
    path.write_text(config)
    _record_reproducibility(task_dir)
    report = json.loads((task_dir / "reproducibility.json").read_text())
    expected = (
        "immutable_image" if pinned and location == "environment" else "image_pinned_recipe" if pinned else "local_only"
    )
    assert report["portability"]["state"] == expected
    assert report["portability"]["configured_images"][0]["reference"] == image


@pytest.mark.parametrize("task_id", ["Has-Caps", "has spaces", "../escape", ""])
def test_init_rejects_unsafe_task_ids(tmp_path: Path, task_id: str) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "init", "--root", str(tmp_path / "root"), "--task-id", task_id],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
