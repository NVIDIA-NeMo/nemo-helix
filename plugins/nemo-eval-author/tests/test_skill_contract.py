# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the bundled Eval Author skills.

``eval-author`` is the core skill: it owns the standard, the vocabulary, the
boundaries, and the routing. ``eval-author-discover``, ``eval-author-audit``,
and ``eval-author-task-create`` bundle scripts. ``eval-author-inspect-trace``
uses the provider's supported commands. Sub-flows defer the standard to the core.

These tests are that enforcement:

- The frontmatter of each skill carries every field ``docs/contributing/skills-spec.mdx`` requires.
- The core routes to every sub-flow, and each sub-flow points back at the core
  rather than restating the standard, which is how the two would drift.
- Only the sub-flow can execute. The core routes, so it gets no Bash.
- The bundled scripts depend on Harbor and nothing else. Harbor is acceptable
  because a repository holding Harbor evaluations has Harbor by construction; a
  NeMo import would not be, and that is the boundary these tests defend.
- ``_ladder.py`` stays out of module scope in ``discover.py``, so a repository
  without Harbor gets an inventory instead of an ImportError.
- No bundled directory is named after a provider package. ``scripts/harbor/``
  would be importable as ``harbor``, which makes ``find_spec`` succeed on a
  machine with no Harbor and the probe claim an install that is not there.
- Discovery reports a valid suite runnable and names the rung a broken one fails.
- Discovery scripts write no files, and audit scripts write only requested audit
  artifacts under ``.eval-author/``.

These tests compare the skill against this repository, never against Harbor's
rules. The skill reimplements no Harbor rule: it asks Harbor for every verdict,
so a Harbor change that tightens a rule flows through without a test change here.

Nothing here imports the platform, for the same reason the bundled scripts do not:
these skills are copied into someone else's repository and have to stand alone. The
tests that execute Harbor-backed behavior skip when Harbor is absent, so the whole
file still runs against nothing but pytest, PyYAML, and jsonschema.
"""

import ast
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import types
from collections.abc import Callable
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
import yaml

_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
_CORE_DIR = _SKILLS_DIR / "eval-author"
_DISCOVER_DIR = _SKILLS_DIR / "eval-author-discover"
_AUDIT_DIR = _SKILLS_DIR / "eval-author-audit"
_TASK_CREATE_DIR = _SKILLS_DIR / "eval-author-task-create"
_INSPECT_DIR = _SKILLS_DIR / "eval-author-inspect-trace"
_MLFLOW_TO_ATIF_DIR = _SKILLS_DIR / "mlflow-to-atif"
_TRACE_ENVIRONMENT_DIR = _SKILLS_DIR / "eval-author-trace-environment"
_SKILL_DIRS = (
    _CORE_DIR,
    _DISCOVER_DIR,
    _AUDIT_DIR,
    _TASK_CREATE_DIR,
    _INSPECT_DIR,
    _MLFLOW_TO_ATIF_DIR,
    _TRACE_ENVIRONMENT_DIR,
)
_SUB_FLOW_DIRS = (_DISCOVER_DIR, _AUDIT_DIR, _TASK_CREATE_DIR, _INSPECT_DIR, _TRACE_ENVIRONMENT_DIR)
_DISCOVER_SCRIPTS_DIR = _DISCOVER_DIR / "scripts"
_AUDIT_SPEC_DIR = _AUDIT_DIR / "scripts" / "audit_spec"
_TASK_CREATE_SCRIPTS_DIR = _TASK_CREATE_DIR / "scripts"
_MLFLOW_TO_ATIF_SCRIPTS_DIR = _MLFLOW_TO_ATIF_DIR / "scripts"
_TRACE_ENVIRONMENT_SCRIPTS_DIR = _TRACE_ENVIRONMENT_DIR / "scripts"
_SCRIPT_DIRS = (
    _DISCOVER_SCRIPTS_DIR,
    _AUDIT_SPEC_DIR,
    _TASK_CREATE_SCRIPTS_DIR,
    _MLFLOW_TO_ATIF_SCRIPTS_DIR,
    _TRACE_ENVIRONMENT_SCRIPTS_DIR,
)
_DISCOVER = _DISCOVER_SCRIPTS_DIR / "discover.py"
_DISCOVER_RENDER_REPORT = _DISCOVER_SCRIPTS_DIR / "render_report.py"
_LADDER = _DISCOVER_SCRIPTS_DIR / "providers" / "harbor" / "_ladder.py"
_AUDIT_VALIDATE = _AUDIT_SPEC_DIR / "validate.py"
_AUDIT_GENERATE = _AUDIT_SPEC_DIR / "generate.py"
_AUDIT_MEASURE = _AUDIT_SPEC_DIR / "measure.py"
_AUDIT_REPORT = _AUDIT_SPEC_DIR / "report.py"
_AUDIT_TEMPLATE = _AUDIT_DIR / "templates" / "audit.md"
_AUDIT_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit.schema.json"
_AUDIT_COVERAGE_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_coverage.schema.json"
_AUDIT_COVERAGE_REPORT_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_coverage_report.schema.json"
_AUDIT_CAPABILITY_JUDGMENTS_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_capability_judgments.schema.json"
_AUDIT_CAPABILITIES_DETAILS_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_capabilities_details.schema.json"
_AUDIT_FAILURE_CASE_JUDGMENTS_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_failure_case_judgments.schema.json"
_AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_failure_cases_details.schema.json"
_AUDIT_TOOL_CALLS_DETAILS_JSON_SCHEMA = _AUDIT_DIR / "schemas" / "audit_tool_calls_details.schema.json"
_AUDIT_TOOL_CALLS_COVERAGE_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "tool_calls.coverage.json"
_AUDIT_CAPABILITIES_COVERAGE_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "capabilities.coverage.json"
_AUDIT_CAPABILITY_JUDGMENTS_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "capability_judgments.json"
_AUDIT_FAILURE_CASES_COVERAGE_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "failure_cases.coverage.json"
_AUDIT_FAILURE_CASE_JUDGMENTS_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "failure_case_judgments.json"
_AUDIT_COVERAGE_REPORT_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "coverage_report.json"
_AUDIT_TOOL_CALLS_DETAILS_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "tool_calls.details.json"
_AUDIT_CAPABILITIES_DETAILS_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "capabilities.details.json"
_AUDIT_FAILURE_CASES_DETAILS_EXAMPLE = _AUDIT_DIR / "examples" / "schemas" / "failure_cases.details.json"
_MLFLOW_TO_ATIF = _MLFLOW_TO_ATIF_SCRIPTS_DIR / "convert_mlflow_to_atif.py"
_TRACE_ENVIRONMENT = _TRACE_ENVIRONMENT_SCRIPTS_DIR / "trace_environment.py"

_REQUIRED_FRONTMATTER = (
    "name",
    "description",
    "triggers",
    "not-for",
    "compatibility",
    "maturity",
    "license",
    "user-invocable",
    "allowed-tools",
)
_MAX_BODY_LINES = 500

# Matches the core skill name but not a longer name that starts with it, so a
# reference to eval-author-discover cannot pass for a reference to eval-author.
_CORE_REFERENCE = re.compile(rf"\b{re.escape(_CORE_DIR.name)}\b(?!-)")

# Only tests that execute Harbor-backed behavior need Harbor. Skipping rather than
# failing is what lets this file run wherever the skills themselves run.
_needs_harbor = pytest.mark.skipif(find_spec("harbor") is None, reason="Harbor is not installed")

# Root reads a mode-000 file regardless, so the failure these tests stage cannot
# happen there and the tests would pass without proving anything.
_needs_unreadable_files = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="this user can read a file whatever its mode, so unreadability cannot be staged",
)

# Bundled scripts may name only these third-party roots. Anything else would
# become a hidden dependency for repositories that copy the skill.
_PERMITTED_THIRD_PARTY = frozenset({"harbor", "jsonschema", "pydantic", "yaml"})


def _not_for_names(frontmatter: dict) -> set[str]:
    """Return the skill names in not-for, dropping the parenthetical reason."""
    return {entry.split("(", 1)[0].strip() for entry in frontmatter["not-for"]}


def _frontmatter_and_body(skill_dir: Path) -> tuple[dict, str]:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{skill_dir.name}/SKILL.md must open with YAML frontmatter"
    _, frontmatter, body = text.split("---\n", 2)
    return yaml.safe_load(frontmatter), body


def _bundled_scripts(scripts_dir: Path | None = None) -> list[Path]:
    """Return every bundled module, including the ones under providers/."""
    roots = _SCRIPT_DIRS if scripts_dir is None else (scripts_dir,)
    return sorted(path for root in roots for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _local_roots(scripts_dir: Path) -> set[str]:
    """Return the top-level names a bundled module can import from scripts/."""
    return {path.stem if path.is_file() else path.name for path in scripts_dir.iterdir()} | {
        path.stem for path in _bundled_scripts(scripts_dir)
    }


def _imported_roots(path: Path) -> set[str]:
    """Return the root package of every import in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module and not node.level else []
        else:
            continue
        roots.update(name.split(".")[0] for name in names)
    return roots


def _run_discover(repo: Path, *args: str, with_harbor: bool = True) -> tuple[int, dict]:
    """Run discover.py as the skill documents it, and parse its JSON.

    ``with_harbor=False`` passes ``-S``, which drops site-packages from the path
    so Harbor and PyYAML are both unimportable. That reproduces a customer
    repository with no Harbor install without needing a second interpreter.
    """
    command = [sys.executable, *([] if with_harbor else ["-S"]), str(_DISCOVER), "--repo", str(repo), *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.stdout, f"discover.py printed nothing; stderr:\n{result.stderr}"
    return result.returncode, json.loads(result.stdout)


def _run_json_script(
    script: Path,
    *args: str,
    python_args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> tuple[int, dict, str]:
    """Run an audit script that emits JSON on stdout."""
    result = subprocess.run(
        [sys.executable, *python_args, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.stdout, f"{script.name} printed nothing; stderr:\n{result.stderr}"
    return result.returncode, json.loads(result.stdout), result.stderr


def _import_audit_measure(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a fresh copy of measure.py so monkeypatches stay local to one test."""
    monkeypatch.syspath_prepend(str(_AUDIT_SPEC_DIR))
    sys.modules.pop("measure", None)
    return importlib.import_module("measure")


def _import_discover_render_report(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the discover Markdown renderer as a standalone script module."""
    monkeypatch.syspath_prepend(str(_DISCOVER_SCRIPTS_DIR))
    sys.modules.pop("render_report", None)
    return importlib.import_module("render_report")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"


def _audit_payload(path: Path) -> dict:
    block = re.search(
        r"<!-- BEGIN:nemo-eval-author-audit:v1 -->\s*```yaml\n(?P<body>.*?)\n```\s*"
        r"<!-- END:nemo-eval-author-audit:v1 -->",
        path.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert block is not None
    return yaml.safe_load(block.group("body"))


def _assert_literal_block_scalar(text: str, field: str, first_line: str) -> None:
    pattern = rf"(?:^|\n)(?:\s*-\s+|\s+){re.escape(field)}: \|[-+]?\n\s+{re.escape(first_line)}"
    assert re.search(pattern, text), text


def _template_payload() -> dict:
    block = re.search(
        r"<!-- BEGIN:nemo-eval-author-audit:v1 -->\s*```yaml\n(?P<body>.*?)\n```\s*"
        r"<!-- END:nemo-eval-author-audit:v1 -->",
        _AUDIT_TEMPLATE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert block is not None
    return yaml.safe_load(block.group("body"))


def _write_audit_items(path: Path, items: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"items": items}), encoding="utf-8")


def _write_atif_trace(
    path: Path,
    *,
    tool_calls: list[str] | None = None,
    embedded_tool_calls: list[str] | None = None,
    session_id: str | None = "session-1",
    trajectory_id: str | None = "root-trajectory",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "example-agent", "version": "1.0.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Help me recover my account."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "I will inspect the account.",
                "tool_calls": [
                    {"tool_call_id": f"root-call-{index}", "function_name": name, "arguments": {}}
                    for index, name in enumerate(tool_calls or [], start=1)
                ],
            },
        ],
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if trajectory_id is not None:
        payload["trajectory_id"] = trajectory_id
    if embedded_tool_calls:
        payload["subagent_trajectories"] = [
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "session-1",
                "trajectory_id": "sub-trajectory",
                "agent": {"name": "helper-agent", "version": "1.0.0"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "message": "Looking up the account.",
                        "tool_calls": [
                            {"tool_call_id": f"sub-call-{index}", "function_name": name, "arguments": {}}
                            for index, name in enumerate(embedded_tool_calls, start=1)
                        ],
                    }
                ],
            }
        ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _measurement_dir(out_dir: Path, task_id: str, run_id: str, method: str = "tool_calls") -> Path:
    return out_dir / f"task={quote(task_id, safe='')}" / f"run={quote(run_id, safe='')}" / method


def _audit_item_counts(payload: dict) -> dict[str, int]:
    return {
        kind: sum(1 for item in payload["items"] if item["kind"] == kind)
        for kind in ("capability", "failure_case", "tool")
    }


def _write_coverage(
    path: Path,
    *,
    audit: Path,
    covered: list[str],
    item_kind: str = "tool",
    method: str = "tool_calls",
    task_id: str = "account-recovery",
    run_id: str = "trial-001",
) -> dict:
    audit_payload = _audit_payload(audit)
    item_counts = _audit_item_counts(audit_payload)
    payload = {
        "schema": "nemo.eval_author.audit_coverage.v1",
        "audit": {
            "path": str(audit),
            "schema": audit_payload["schema"],
            "agent": audit_payload["agent"],
            "status": audit_payload["status"],
            "item_count": len(audit_payload["items"]),
        },
        "subject": {
            "trace": str(path.parent / "trajectory.json"),
            "trace_format": "atif",
            "task_id": task_id,
            "run_id": run_id,
        },
        "method": {"name": method},
        "item_kind": item_kind,
        "item_kind_count": item_counts[item_kind],
        "covered": covered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _ticket_tool_item() -> dict:
    return {
        "kind": "tool",
        "name": "ticket.create",
        "description": "Creates a support ticket for issues requiring human follow-up.",
        "expected_use": "Used when self-service resolution cannot proceed.",
        "expected_failure_behavior": "If ticket creation fails, the agent explains the failure and avoids duplicates.",
        "evidence_required": [
            {
                "kind": "tool_call",
                "tool": "ticket.create",
                "description": "Trace shows a ticket.create call for an escalation.",
            }
        ],
    }


def _write_audit(tmp_path: Path, transform: Callable[[str], str] | None = None) -> Path:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos\n\n## Tools\n\n- customer.lookup\n", encoding="utf-8")

    audit_dir = tmp_path / ".eval-author"
    audit_dir.mkdir()
    audit = audit_dir / "audit.md"
    text = _AUDIT_TEMPLATE.read_text(encoding="utf-8").replace(
        'sha256: "sha256:<replace-with-64-hex-digest>"',
        f"sha256: {_digest(ethos)}",
    )
    if transform is not None:
        text = transform(text)
    audit.write_text(text, encoding="utf-8")
    return audit


def _without_user_intent_evidence(text: str) -> str:
    """Make the template capability coverable by deterministic tool-call evidence only."""
    return text.replace(
        "      - kind: user_intent\n        description: User is trying to recover account access.\n",
        "",
        1,
    )


def _write_capability_judgments(
    path: Path,
    *,
    trace: Path | None = None,
    capability: str = "account_recovery",
    evidence_index: int = 0,
    kind: str = "user_intent",
    description: str = "User is trying to recover account access.",
    status: str = "satisfied",
    confidence: str = "high",
) -> None:
    """Write a skill-authored capability judgment sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "nemo.eval_author.audit_capability_judgments.v1",
                "trace_sha256": _digest(trace) if trace is not None else "sha256:" + ("0" * 64),
                "judged_by": "eval-author-audit skill",
                "judgments": [
                    {
                        "capability": capability,
                        "evidence_index": evidence_index,
                        "kind": kind,
                        "description": description,
                        "status": status,
                        "confidence": confidence,
                        "rationale": "The trace shows the user asking for help recovering account access.",
                        "supporting_trace_refs": ["$.steps[0].message"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_failure_case_judgments(
    path: Path,
    *,
    trace: Path | None = None,
    failure_case: str = "account_recovery_unverified_identity",
    trigger_description: str = "User requests account recovery but cannot satisfy identity verification.",
    output_status: str = "satisfied",
    include_output: bool = True,
) -> None:
    """Write skill-authored judgments for the template failure case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    judgments = [
        {
            "failure_case": failure_case,
            "evidence_index": 0,
            "kind": "user_intent",
            "description": trigger_description,
            "status": "satisfied",
            "confidence": "high",
            "rationale": "The trace shows recovery blocked by failed identity verification.",
            "supporting_trace_refs": ["$.steps[0].message"],
        }
    ]
    if include_output:
        judgments.append(
            {
                "failure_case": failure_case,
                "evidence_index": 1,
                "kind": "output",
                "description": (
                    "Agent refuses direct recovery, explains the allowed path, and does not disclose passwords "
                    "or reset tokens."
                ),
                "status": output_status,
                "confidence": "high",
                "rationale": "The agent refuses direct recovery and exposes no credential data.",
                "supporting_trace_refs": ["$.steps[1].message"],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "nemo.eval_author.audit_failure_case_judgments.v1",
                "trace_sha256": _digest(trace) if trace is not None else "sha256:" + ("0" * 64),
                "judged_by": "eval-author-audit skill",
                "judgments": judgments,
            }
        ),
        encoding="utf-8",
    )


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a bundled script and return the completed process."""
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, check=False)


def _write_task(task_dir: Path, *, name: str = "smoke/generated") -> None:
    """Write a Harbor task that mirrors the canonical prebuilt-image layout.

    ``environment/`` has to exist even though the image is prebuilt, because
    ``Task.is_valid_dir`` checks for the directory before reading the config.
    """
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        "\n[task]\n"
        f'name = "{name}"\n'
        'authors = [{ name = "NVIDIA" }]\n'
        "\n[agent]\ntimeout_sec = 120.0\n"
        "\n[verifier]\ntimeout_sec = 60.0\n"
        '\n[environment]\ndocker_image = "smoke-agent-env:latest"\ncpus = 1\nmemory_mb = 1024\n',
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Look up the total hours for Ada.\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")


def _load_mlflow_to_atif() -> Any:
    """Load the standalone converter so boundary behavior can be tested without a child process."""
    spec = spec_from_file_location("test_convert_mlflow_to_atif", _MLFLOW_TO_ATIF)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mlflow_export() -> dict[str, Any]:
    """Return a small synthetic MLflow export with one retriever child."""
    return {
        "traces": [
            {
                "info": {
                    "trace_id": "tr-fixture",
                    "trace_metadata": {
                        "mlflow.traceInputs": '{"question":"Where is Paris?"}',
                        "mlflow.traceOutputs": '{"answer":"France"}',
                    },
                    "assessments": [
                        {
                            "assessment_id": "assessment-1",
                            "assessment_name": "correctness",
                            "feedback": {"value": 1.0},
                        }
                    ],
                },
                "data": {
                    "spans": [
                        {
                            "trace_id": "tr-fixture",
                            "span_id": "root-span",
                            "parent_span_id": None,
                            "name": "answer",
                            "start_time_unix_nano": 1_000_000_000,
                            "end_time_unix_nano": 3_000_000_000,
                            "status": {"code": "STATUS_CODE_OK"},
                            "attributes": {
                                "mlflow.spanType": '"CHAIN"',
                                "mlflow.spanInputs": '{"question":"Where is Paris?"}',
                                "mlflow.spanOutputs": '{"answer":"France"}',
                            },
                        },
                        {
                            "trace_id": "tr-fixture",
                            "span_id": "tool-span",
                            "parent_span_id": "root-span",
                            "name": "lookup",
                            "start_time_unix_nano": 2_000_000_000,
                            "end_time_unix_nano": 2_500_000_000,
                            "status": {"code": "STATUS_CODE_OK"},
                            "attributes": {
                                "mlflow.spanType": '"RETRIEVER"',
                                "mlflow.spanInputs": '{"query":"Paris"}',
                                "mlflow.spanOutputs": '{"documents":["Paris is in France."]}',
                            },
                        },
                    ]
                },
            }
        ]
    }


@pytest.fixture
def suite(tmp_path: Path) -> Path:
    """Build a repository with one valid task and one Harbor job config."""
    _write_task(tmp_path / "dataset" / "task-one")
    (tmp_path / "harbor-job.yaml").write_text(
        "job_name: fixture\ndatasets:\n  - path: ./dataset\nagents:\n  - name: oracle\n",
        encoding="utf-8",
    )
    return tmp_path


def _tree_state(root: Path) -> dict[str, bytes | None]:
    """Map every path under root to its bytes, so an in-place edit is visible.

    Comparing names alone cannot catch a script that rewrote a file that was
    already there, which is the promise this is here to hold.
    """
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in root.rglob("*")
    }


def _named(report: dict, name: str) -> dict:
    """Return one check by name, failing loudly when the ladder never ran it."""
    found = next((check for check in report["checks"] if check["name"] == name), None)
    assert found is not None, f"no {name!r} check in report; ran {[c['name'] for c in report['checks']]}"
    return found


@pytest.mark.parametrize("skill_dir", _SKILL_DIRS, ids=lambda path: path.name)
def test_frontmatter_carries_every_required_field(skill_dir: Path) -> None:
    frontmatter, _ = _frontmatter_and_body(skill_dir)
    missing = [field for field in _REQUIRED_FRONTMATTER if field not in frontmatter]
    assert not missing, f"{skill_dir.name} frontmatter is missing {missing}"
    assert frontmatter["name"] == skill_dir.name, "frontmatter name must match the skill directory name"
    assert frontmatter["maturity"] in {"alpha", "beta", "active", "deprecated"}
    assert isinstance(frontmatter["user-invocable"], bool)
    assert len(frontmatter["triggers"]) >= 3, "the routing audit needs at least three trigger phrases"
    assert len(frontmatter["not-for"]) >= 2, "not-for needs at least two sibling skills to prevent collisions"


@pytest.mark.parametrize("skill_dir", _SKILL_DIRS, ids=lambda path: path.name)
def test_no_skill_can_edit_what_it_did_not_write(skill_dir: Path) -> None:
    """Eval Author creates its own report and changes nothing that was already there.

    ``Write`` covers sub-flow artifacts under ``.eval-author/``. ``Edit`` would let
    it rewrite files that predate it, which is the permission customers declined to
    grant and the reason these ship as skills.
    """
    frontmatter, _ = _frontmatter_and_body(skill_dir)
    tools = set(frontmatter["allowed-tools"])
    assert not {"Edit", "MultiEdit", "NotebookEdit"} & tools, (
        f"{skill_dir.name} edits nothing that predates it, so {sorted(tools)} is too broad"
    )


def test_the_core_routes_and_the_sub_flow_executes() -> None:
    """The core only picks a sub-flow, so it neither runs nor saves anything."""
    core_tools = set(_frontmatter_and_body(_CORE_DIR)[0]["allowed-tools"])
    discover_tools = set(_frontmatter_and_body(_DISCOVER_DIR)[0]["allowed-tools"])
    audit_tools = set(_frontmatter_and_body(_AUDIT_DIR)[0]["allowed-tools"])
    task_create_tools = set(_frontmatter_and_body(_TASK_CREATE_DIR)[0]["allowed-tools"])
    inspect_tools = set(_frontmatter_and_body(_INSPECT_DIR)[0]["allowed-tools"])
    trace_environment_tools = set(_frontmatter_and_body(_TRACE_ENVIRONMENT_DIR)[0]["allowed-tools"])

    assert not {"Bash", "Write"} & core_tools, f"the core routes and explains; {sorted(core_tools)} is too broad"
    assert {"Bash", "Write"} <= discover_tools, (
        f"{_DISCOVER_DIR.name} runs a script and saves a report; it has {sorted(discover_tools)}"
    )
    assert {"Bash", "Write"} <= audit_tools, (
        f"{_AUDIT_DIR.name} generates and validates audit files; it has {sorted(audit_tools)}"
    )
    assert {"Bash", "Write"} <= task_create_tools, (
        f"{_TASK_CREATE_DIR.name} generates and proves task drafts; it has {sorted(task_create_tools)}"
    )
    assert {"Bash", "Write"} <= inspect_tools, (
        f"{_INSPECT_DIR.name} runs provider commands and saves a report; it has {sorted(inspect_tools)}"
    )
    assert {"Bash", "Write"} <= trace_environment_tools, (
        f"{_TRACE_ENVIRONMENT_DIR.name} prepares and verifies task artifacts; it has {sorted(trace_environment_tools)}"
    )


def test_the_core_names_every_sub_flow() -> None:
    """A sub-flow the core never mentions cannot be routed to."""
    _, body = _frontmatter_and_body(_CORE_DIR)
    for skill_dir in _SUB_FLOW_DIRS:
        assert skill_dir.name in body, f"the core does not route to {skill_dir.name}"


def test_inspect_flow_is_reached_only_through_eval_author() -> None:
    """Generic Intake questions must not match the inspect-trace sub-flow.

    ``nemo-intake`` already owns instrumentation, ingest, and query. If this
    sub-flow stays user-invocable and repeats those phrases, an agent with both
    skills loaded cannot tell which one to start.
    """
    inspect_frontmatter, inspect_body = _frontmatter_and_body(_INSPECT_DIR)
    core_frontmatter, _ = _frontmatter_and_body(_CORE_DIR)
    overlapping = (
        "inspect this agent trace",
        "what happened in this agent trace",
        "explain this production agent run",
        "did this trace succeed",
        "why did this trace fail",
        "inspecting agent runs",
    )

    assert inspect_frontmatter["user-invocable"] is False
    assert "eval-author has routed" in inspect_frontmatter["description"]
    assert "nemo-intake" in inspect_frontmatter["description"]
    assert "nemo-intake" in _not_for_names(inspect_frontmatter)
    assert "nemo-intake" in _not_for_names(core_frontmatter)
    assert "what happened in this agent trace" in core_frontmatter["triggers"]
    for phrase in overlapping:
        assert phrase not in inspect_frontmatter["triggers"]
        assert phrase not in inspect_frontmatter["description"]
    assert "after `eval-author` selects it" in inspect_body


def test_inspect_flow_is_only_a_cli_driven_skill() -> None:
    _, body = _frontmatter_and_body(_INSPECT_DIR)

    assert {path.name for path in _INSPECT_DIR.iterdir()} == {"SKILL.md"}
    for command in (
        "nemo intake traces",
        "nemo intake spans",
        "nemo intake evaluator-results",
    ):
        assert command in body


def test_inspect_flow_resolves_the_cli_without_changing_the_environment() -> None:
    """The trace reader must handle installed and source-checkout CLI invocations."""
    _, body = _frontmatter_and_body(_INSPECT_DIR)

    candidates = (
        "caller-supplied invocation",
        "command -v nemo",
        ".venv/bin/nemo",
        "uv run --no-sync nemo",
    )
    positions = [body.index(candidate) for candidate in candidates]
    assert positions == sorted(positions), "CLI candidates must appear in priority order"
    assert "Use the resolved invocation for every command" in body


@pytest.mark.parametrize("skill_dir", _SUB_FLOW_DIRS, ids=lambda path: path.name)
def test_each_sub_flow_defers_to_the_core(skill_dir: Path) -> None:
    """A sub-flow points at the core rather than restating the standard itself.

    Restating it is how the two drift: the copy in the sub-flow gets edited and the
    core keeps saying something else. The match rejects a bare prefix, so naming
    ``eval-author-discover`` does not count as pointing at ``eval-author``.
    """
    frontmatter, body = _frontmatter_and_body(skill_dir)
    assert _CORE_REFERENCE.search(body), f"{skill_dir.name} must point at {_CORE_DIR.name} for the standard"
    assert _CORE_DIR.name in _not_for_names(frontmatter), (
        f"{skill_dir.name} must name {_CORE_DIR.name} in not-for so the router can tell them apart"
    )


@pytest.mark.parametrize("skill_dir", _SKILL_DIRS, ids=lambda path: path.name)
def test_skill_body_stays_within_the_line_budget(skill_dir: Path) -> None:
    _, body = _frontmatter_and_body(skill_dir)
    line_count = len(body.splitlines())
    assert line_count <= _MAX_BODY_LINES, (
        f"{skill_dir.name} body is {line_count} lines, over the {_MAX_BODY_LINES} budget"
    )


def test_every_bundled_path_the_skill_names_exists() -> None:
    _, body = _frontmatter_and_body(_DISCOVER_DIR)
    for relative in (
        "scripts/discover.py",
        "scripts/render_report.py",
        "scripts/_checks.py",
        "scripts/providers/harbor/_probe.py",
        "scripts/providers/harbor/_inventory.py",
        "scripts/providers/harbor/_ladder.py",
    ):
        assert relative in body, f"SKILL.md no longer documents {relative}"
        assert (_DISCOVER_DIR / relative).exists(), f"SKILL.md names {relative}, which is missing on disk"


def test_discover_saved_report_is_human_friendly_before_json_evidence() -> None:
    """The saved discovery report should lead with the user verdict, not raw JSON."""
    _, body = _frontmatter_and_body(_DISCOVER_DIR)
    required_guidance = (
        "The saved report must be useful to a human first, and auditable second",
        "does this repo have evals, and how do I run",
        "render_report.py --summary",
        "basis of the final assistant reply",
        "Configs",
        "Advisories",
        "Evidence JSON",
        "preserves the original stdout JSON",
    )
    for phrase in required_guidance:
        assert phrase in body, f"discover report guidance is missing {phrase!r}"

    assert body.index("--summary") < body.index("Evidence JSON")


def test_discover_unproven_outcome_has_no_run_command(suite: Path) -> None:
    """Missing Harbor produces the 'could not prove readiness' report path."""
    code, report = _run_discover(suite, with_harbor=False)

    assert code == 1
    assert report["proven"] is False
    assert report["runnable"] is False
    assert report["run_command"] is None
    assert _named(report, "harbor")["status"] == "fail"


@_needs_harbor
def test_discover_not_ready_outcome_names_required_failures(suite: Path) -> None:
    """A proven but broken suite produces the 'not ready to run' report path."""
    (suite / "harbor-job.yaml").write_text(
        "job_name: fixture\ndatasets:\n  - path: ./no-such-dataset\nagents:\n  - name: oracle\n",
        encoding="utf-8",
    )

    code, report = _run_discover(suite)

    assert code == 1
    assert report["proven"] is True
    assert report["runnable"] is False
    assert report["run_command"] is None
    assert report["configs"][0]["runnable"] is False
    assert _named(report, "resolution")["status"] == "fail"


@_needs_harbor
def test_discover_ready_outcome_with_one_config_includes_run_command(suite: Path) -> None:
    """One runnable config produces the direct 'run this command' report path."""
    code, report = _run_discover(suite)

    backend = _named(report, "backend")
    if backend["status"] != "pass":
        pytest.skip(f"no environment backend available: {backend['message']}")
    assert code == 0
    assert report["proven"] is True
    assert report["runnable"] is True
    assert report["configs"][0]["runnable"] is True
    assert report["run_command"] == f"cd {suite} && harbor job start -c harbor-job.yaml"


@_needs_harbor
def test_discover_ready_outcome_with_multiple_configs_omits_run_command(suite: Path) -> None:
    """Multiple runnable configs ask the user to choose instead of inventing one command."""
    (suite / "second-job.yaml").write_text(
        "job_name: second\ndatasets:\n  - path: ./dataset\nagents:\n  - name: oracle\n",
        encoding="utf-8",
    )

    code, report = _run_discover(suite)

    backend = _named(report, "backend")
    if backend["status"] != "pass":
        pytest.skip(f"no environment backend available: {backend['message']}")
    assert code == 0
    assert report["proven"] is True
    assert report["runnable"] is True
    assert [config["runnable"] for config in report["configs"]] == [True, True]
    assert [config["path"] for config in report["configs"]] == ["harbor-job.yaml", "second-job.yaml"]
    assert report["run_command"] is None


def _discovery_report_fixture(
    *,
    proven: bool,
    runnable: bool,
    configs: list[dict[str, Any]],
    run_command: str | None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo_root": "/repo",
        "provider": "harbor",
        "proven": proven,
        "runnable": runnable,
        "runtime": {
            "python": "3.13.2",
            "harbor_importable": proven,
            "harbor_version": "0.21.0" if proven else None,
            "harbor_cli": "/repo/.venv/bin/harbor",
        },
        "configs": configs,
        "dataset_paths": ["dataset"],
        "task_count": 1,
        "ethos_path": None,
        "fingerprint": "sha256:abc123",
        "input_file_count": 3,
        "discovered_at": "2026-09-10T00:00:00+00:00",
        "checks": checks or [],
        "run_command": run_command,
    }


def _config_fixture(*, path: str, runnable: bool, checks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "name": path,
        "path": path,
        "runnable": runnable,
        "required_env_vars": [],
        "checks": checks or [],
    }


def _check_fixture(
    name: str,
    *,
    status: str,
    severity: str,
    message: str,
    hint: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "group": "validation",
        "status": status,
        "severity": severity,
        "message": message,
        "hint": hint,
        "proven": True,
    }


def test_discover_report_renderer_handles_unproven_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    harbor = _check_fixture(
        "harbor",
        status="fail",
        severity="required",
        message="Harbor is not importable, so nothing in this report is proven.",
        hint="Install Harbor into the interpreter running this script, then run discovery again.",
    )
    report = _discovery_report_fixture(
        proven=False,
        runnable=False,
        configs=[_config_fixture(path="harbor-job.yaml", runnable=False)],
        run_command=None,
        checks=[harbor],
    )

    markdown = renderer.render_report(report)

    assert "readiness has not been checked" in markdown
    assert "| `harbor-job.yaml` | Not checked | Not checked |" in markdown
    assert "Use the Python environment for this suite" in markdown
    assert "No required failures" not in markdown
    assert "## Evidence JSON" in markdown


def test_discover_report_renderer_handles_not_ready_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    resolution = _check_fixture(
        "resolution",
        status="fail",
        severity="required",
        message="Harbor could not resolve the job: FileNotFoundError: ./no-such-dataset",
        hint="This error occurs before Harbor starts a container.",
    )
    report = _discovery_report_fixture(
        proven=True,
        runnable=False,
        configs=[_config_fixture(path="harbor-job.yaml", runnable=False, checks=[resolution])],
        run_command=None,
        checks=[resolution],
    )

    markdown = renderer.render_report(report)

    assert "none of the 1 configurations is ready" in markdown
    assert "Check the dataset paths and job settings" in markdown
    assert "`resolution`: Harbor could not resolve the job: FileNotFoundError: ./no-such-dataset" in markdown
    assert "Hint: This error occurs before Harbor starts a container." in markdown


def test_discover_report_renderer_handles_single_ready_config(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    report = _discovery_report_fixture(
        proven=True,
        runnable=True,
        configs=[_config_fixture(path="harbor-job.yaml", runnable=True)],
        run_command="cd /repo && harbor job start -c harbor-job.yaml",
    )

    markdown = renderer.render_report(report)

    assert "This repo has Harbor evals, and they are ready to run." in markdown
    assert "| `harbor-job.yaml` | Ready |" in markdown
    assert "Run the evals with:" in markdown
    assert "```bash\ncd /repo && harbor job start -c harbor-job.yaml\n```" in markdown
    assert "No required failures." in markdown


def test_discover_report_renderer_handles_multiple_ready_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    report = _discovery_report_fixture(
        proven=True,
        runnable=True,
        configs=[
            _config_fixture(path="harbor-job.yaml", runnable=True),
            _config_fixture(path="second-job.yaml", runnable=True),
        ],
        run_command=None,
    )

    markdown = renderer.render_report(report)

    assert "This repo has Harbor evals, and they are ready to run." in markdown
    assert "Each discovered config is ready. Pick the config you want to run" in markdown
    assert "| `harbor-job.yaml` | Ready |" in markdown
    assert "| `second-job.yaml` | Ready |" in markdown


@pytest.mark.parametrize("proven", [True, False])
def test_discover_report_renderer_empty_repo(monkeypatch: pytest.MonkeyPatch, proven: bool) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    report = _discovery_report_fixture(proven=proven, runnable=False, configs=[], run_command=None)
    report.update(task_count=0, dataset_paths=[])
    summary = renderer.render_summary(report)
    assert summary.startswith("I did not find Harbor evals")
    assert "other kinds of evaluations" in summary
    assert "has Harbor evals" not in summary
    assert "Python" not in summary


def test_discover_report_renderer_tasks_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    report = _discovery_report_fixture(proven=True, runnable=False, configs=[], run_command=None)
    assert "task or dataset files, but no run configuration" in renderer.render_summary(report)


@pytest.mark.parametrize("error", ["Not a directory: /missing", "Discovery needs Python 3.11 or later"])
def test_discover_report_renderer_errors(monkeypatch: pytest.MonkeyPatch, error: str) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    report = {"error": error, "hint": "Correct the input and try again."}
    markdown = renderer.render_report(report)
    assert "I could not inspect this repository." in markdown
    assert error in markdown
    assert report["hint"] in markdown
    assert "possible Harbor evals" not in markdown
    assert "Harbor installed" not in markdown
    assert "## Configs" not in markdown


@pytest.mark.parametrize("ready_count", [0, 1])
@pytest.mark.parametrize(
    "docker_error",
    [
        "SystemExit: Docker daemon is not running. Please start Docker and try again.",
        "PermissionError: permission denied accessing Docker socket",
        "Cannot connect to the Docker daemon",
    ],
)
def test_discover_report_renderer_shared_docker_blocker(
    monkeypatch: pytest.MonkeyPatch,
    ready_count: int,
    docker_error: str,
) -> None:
    renderer = _import_discover_render_report(monkeypatch)
    failure = _check_fixture(
        "backend",
        status="fail",
        severity="required",
        message=f"Environment backend docker is not ready: {docker_error}",
    )
    configs = [
        _config_fixture(path=f"config-{i}.yaml", runnable=i < ready_count, checks=[] if i < ready_count else [failure])
        for i in range(5)
    ]
    report = _discovery_report_fixture(
        proven=True, runnable=False, configs=configs, run_command=None, checks=[failure] * (5 - ready_count)
    )
    summary = renderer.render_summary(report)
    assert summary.count("Check Docker access") == 1
    assert "Start Docker only if it is confirmed stopped" in summary
    assert "Docker is stopped" not in summary
    assert "same environment" in summary
    assert "backend" not in summary
    assert "SystemExit" not in summary
    if ready_count:
        assert "1 of 5 configurations are ready" in summary
        assert "`config-0.yaml`" in summary
    else:
        assert "could not verify readiness" in summary
    markdown = renderer.render_report(report)
    before_evidence = markdown.split("## Evidence JSON")[0]
    assert before_evidence.count(failure["message"]) == 1
    for config in configs[ready_count:]:
        assert config["path"] in before_evidence.split("## Diagnostic Details")[1]
    assert summary in markdown


def test_discover_docker_access_retry_guidance() -> None:
    _, body = _frontmatter_and_body(_DISCOVER_DIR)
    assert "retry `docker info`" in body
    assert "rerun the full discovery command" in body
    assert "Do not bypass a denied request" in body
    assert "`docker info` alone does not prove eval readiness" in body


def test_discover_report_renderer_cli_summary_and_evidence(tmp_path: Path) -> None:
    report = _discovery_report_fixture(
        proven=True,
        runnable=True,
        configs=[_config_fixture(path="job.yaml", runnable=True)],
        run_command="cd /repo && harbor job start -c job.yaml",
    )
    source = json.dumps(report, separators=(",", ":")) + "\n"
    path = tmp_path / "discovery.json"
    path.write_text(source)
    summary = subprocess.run(
        [sys.executable, str(_DISCOVER_RENDER_REPORT), "--summary", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    saved = subprocess.run(
        [sys.executable, str(_DISCOVER_RENDER_REPORT), str(path)], capture_output=True, text=True, check=True
    ).stdout
    assert summary.strip() in saved
    assert report["run_command"] in summary
    assert "Evidence JSON" not in summary
    assert source in saved


def test_task_create_script_the_skill_names_exists() -> None:
    _, body = _frontmatter_and_body(_TASK_CREATE_DIR)
    relative = "scripts/task_pipeline.py"
    assert relative in body
    assert (_TASK_CREATE_DIR / relative).is_file()


def test_mlflow_to_atif_script_the_skill_names_exists() -> None:
    _, body = _frontmatter_and_body(_MLFLOW_TO_ATIF_DIR)
    relative = "scripts/convert_mlflow_to_atif.py"
    assert relative in body
    assert (_MLFLOW_TO_ATIF_DIR / relative).is_file()


def test_trace_environment_script_the_skill_names_exists() -> None:
    _, body = _frontmatter_and_body(_TRACE_ENVIRONMENT_DIR)
    relative = "scripts/trace_environment.py"
    assert relative in body
    assert (_TRACE_ENVIRONMENT_DIR / relative).is_file()


def test_trace_environment_records_ground_truth_and_software_constraints() -> None:
    _, body = _frontmatter_and_body(_TRACE_ENVIRONMENT_DIR)

    for phrase in (
        "ground_truth",
        "software_requirements",
        "proprietary",
        "redistributable",
        "private/ground-truth/",
    ):
        assert phrase in body


def test_mlflow_to_atif_converts_export_to_private_v17_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(_mlflow_export()), encoding="utf-8")
    output = tmp_path / "atif"

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--agent-name",
        "fixture-agent",
        "--agent-version",
        "1.0.0",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["converted"] == 1
    assert summary["schema_version"] == "ATIF-v1.7"
    assert summary["validated_with_harbor"] is False
    target = Path(summary["files"][0])
    trajectory = json.loads(target.read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["trajectory_id"] == "tr-fixture"
    assert trajectory["agent"] == {"name": "fixture-agent", "version": "1.0.0"}
    assert [step["source"] for step in trajectory["steps"]] == ["user", "agent", "agent"]
    assert trajectory["steps"][0]["extra"]["mlflow_to_atif"]["canonical_role"] == "human_instruction"
    tool_step = trajectory["steps"][1]
    assert tool_step["llm_call_count"] == 0
    assert tool_step["tool_calls"][0] == {
        "tool_call_id": "tool-span",
        "function_name": "lookup",
        "arguments": {"query": "Paris"},
        "extra": {"mlflow": {"span_type": "RETRIEVER"}},
    }
    assert tool_step["observation"]["results"][0]["source_call_id"] == "tool-span"
    assert trajectory["steps"][2]["message"] == "France"
    assert trajectory["extra"]["mlflow"]["info"]["assessments"][0]["assessment_id"] == "assessment-1"
    assert [span["span_id"] for span in trajectory["extra"]["mlflow"]["spans"]] == ["root-span", "tool-span"]
    assert trajectory["extra"]["mlflow_to_atif"]["loss_codes"] == [
        "mlflow_span_tree_linearized",
        "orchestration_parent_not_emitted_as_step",
    ]
    assert trajectory["extra"]["mlflow_to_atif"]["normalization_codes"] == []
    if os.name == "posix":
        assert output.stat().st_mode & 0o777 == 0o700
        assert target.stat().st_mode & 0o777 == 0o600


def test_mlflow_to_atif_normalizes_parent_bounded_microsecond_event_times() -> None:
    converter = _load_mlflow_to_atif()
    payload = _mlflow_export()
    event_time_micros = 1_788_000_000_000_006
    event_time_nanos = event_time_micros * 1_000
    root, tool = payload["traces"][0]["data"]["spans"]
    root["start_time_unix_nano"] = event_time_nanos - 2_000
    root["end_time_unix_nano"] = event_time_nanos + 2_000
    tool["start_time_unix_nano"] = event_time_nanos - 1_000
    tool["end_time_unix_nano"] = event_time_nanos + 1_000
    tool["events"] = [{"name": "completed", "time_unix_nano": event_time_micros}]

    (trajectory,) = converter.convert_export(
        payload,
        agent_name="fixture-agent",
        agent_version="1.0.0",
    )

    preserved_tool = trajectory["extra"]["mlflow"]["spans"][1]
    assert preserved_tool["events"][0]["time_unix_nano"] == event_time_nanos
    assert trajectory["steps"][1]["extra"]["mlflow"]["events"][0]["time_unix_nano"] == event_time_nanos
    assert trajectory["extra"]["mlflow_to_atif"]["normalization_codes"] == [
        "event_time_microseconds_normalized_to_nanoseconds"
    ]


def test_mlflow_to_atif_does_not_normalize_event_time_outside_parent_bounds() -> None:
    converter = _load_mlflow_to_atif()
    payload = _mlflow_export()
    event_time = 1_788_000_000_000_006
    payload["traces"][0]["data"]["spans"][1]["events"] = [{"name": "unbounded", "time_unix_nano": event_time}]

    (trajectory,) = converter.convert_export(
        payload,
        agent_name="fixture-agent",
        agent_version="1.0.0",
    )

    preserved_tool = trajectory["extra"]["mlflow"]["spans"][1]
    assert preserved_tool["events"][0]["time_unix_nano"] == event_time
    assert trajectory["extra"]["mlflow_to_atif"]["normalization_codes"] == []


def test_mlflow_to_atif_accepts_one_bare_trace_to_dict_value(tmp_path: Path) -> None:
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(_mlflow_export()["traces"][0]), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["converted"] == 1


def test_mlflow_to_atif_uses_request_preview_after_empty_input_placeholders() -> None:
    converter = _load_mlflow_to_atif()
    payload = _mlflow_export()
    info = payload["traces"][0]["info"]
    info["request_preview"] = "Where is Paris?"
    info["trace_metadata"]["mlflow.traceInputs"] = "{}"
    payload["traces"][0]["data"]["spans"][0]["attributes"]["mlflow.spanInputs"] = "[]"

    (trajectory,) = converter.convert_export(
        payload,
        agent_name="fixture-agent",
        agent_version="1.0.0",
    )

    assert trajectory["steps"][0]["message"] == "Where is Paris?"


def test_mlflow_to_atif_uses_later_populated_root_input_attribute() -> None:
    converter = _load_mlflow_to_atif()
    payload = _mlflow_export()
    root_attributes = payload["traces"][0]["data"]["spans"][0]["attributes"]
    root_attributes["mlflow.spanInputs"] = "{}"
    root_attributes["input.value"] = "Use the populated root input."

    (trajectory,) = converter.convert_export(
        payload,
        agent_name="fixture-agent",
        agent_version="1.0.0",
    )

    assert trajectory["steps"][0]["message"] == "Use the populated root input."


def test_mlflow_to_atif_matches_current_mlflow_external_and_native_trace_ids(tmp_path: Path) -> None:
    payload = _mlflow_export()
    external_trace_id = "tr-06429b3cbfa8ac4aee6168aedf62e3fe"
    payload["traces"][0]["info"]["trace_id"] = external_trace_id
    for span in payload["traces"][0]["data"]["spans"]:
        span["trace_id"] = "BkKbPL+orEruYWiu32Lj/g=="
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 0, result.stderr
    target = Path(json.loads(result.stdout)["files"][0])
    assert json.loads(target.read_text(encoding="utf-8"))["trajectory_id"] == external_trace_id


@pytest.mark.parametrize(
    "trace_input",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "valid"}, {"content": "missing a role"}]},
    ],
)
def test_mlflow_to_atif_rejects_empty_or_malformed_chat_input(tmp_path: Path, trace_input: dict[str, Any]) -> None:
    payload = _mlflow_export()
    payload["traces"][0]["data"]["spans"][0]["attributes"]["mlflow.spanInputs"] = json.dumps(trace_input)
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 1
    assert "empty or malformed message list" in result.stderr
    assert not (tmp_path / "atif").exists()


@pytest.mark.parametrize(
    ("output_mode", "expected_state", "expected_loss_code"),
    [
        ("missing", "missing", "missing_tool_output_rendered_as_empty_string"),
        ("null", "null", "null_tool_output_rendered_as_empty_string"),
    ],
)
def test_mlflow_to_atif_distinguishes_missing_and_null_tool_outputs(
    tmp_path: Path,
    output_mode: str,
    expected_state: str,
    expected_loss_code: str,
) -> None:
    payload = _mlflow_export()
    attributes = payload["traces"][0]["data"]["spans"][1]["attributes"]
    if output_mode == "missing":
        attributes.pop("mlflow.spanOutputs")
    else:
        attributes["mlflow.spanOutputs"] = "null"
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 0, result.stderr
    target = Path(json.loads(result.stdout)["files"][0])
    trajectory = json.loads(target.read_text(encoding="utf-8"))
    tool_result = trajectory["steps"][1]["observation"]["results"][0]
    assert tool_result["content"] == ""
    assert tool_result["extra"]["mlflow"]["output_state"] == expected_state
    assert expected_loss_code in trajectory["extra"]["mlflow_to_atif"]["loss_codes"]


def test_mlflow_to_atif_rejects_incomplete_paginated_export(tmp_path: Path) -> None:
    payload = _mlflow_export()
    payload["next_page_token"] = "fetch-another-page"
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 1
    assert "fetch all pages" in result.stderr
    assert not (tmp_path / "atif").exists()


@pytest.mark.parametrize(
    ("parent_ids", "expected_error"),
    [
        ((None, "missing-span"), "unresolved parent span ID"),
        (("tool-span", "root-span"), "parent graph contains a cycle"),
    ],
)
def test_mlflow_to_atif_rejects_invalid_parent_graphs(
    tmp_path: Path,
    parent_ids: tuple[str | None, str],
    expected_error: str,
) -> None:
    payload = _mlflow_export()
    spans = payload["traces"][0]["data"]["spans"]
    for span, parent_id in zip(spans, parent_ids, strict=True):
        span["parent_span_id"] = parent_id
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 1
    assert expected_error in result.stderr
    assert not (tmp_path / "atif").exists()


def test_mlflow_to_atif_rejects_info_and_span_trace_id_mismatch(tmp_path: Path) -> None:
    payload = _mlflow_export()
    for span in payload["traces"][0]["data"]["spans"]:
        span["trace_id"] = "a-different-trace"
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 1
    assert "info trace ID does not match its spans" in result.stderr
    assert not (tmp_path / "atif").exists()


def test_mlflow_to_atif_requires_recoverable_human_input(tmp_path: Path) -> None:
    payload = _mlflow_export()
    payload["traces"][0]["info"]["trace_metadata"].pop("mlflow.traceInputs")
    payload["traces"][0]["data"]["spans"][0]["attributes"].pop("mlflow.spanInputs")
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    assert result.returncode == 1
    assert "no recoverable root input" in result.stderr
    assert not (tmp_path / "atif").exists()


def test_mlflow_to_atif_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(_mlflow_export()), encoding="utf-8")
    args = (
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
    )

    first = _run_script(_MLFLOW_TO_ATIF, *args)
    second = _run_script(_MLFLOW_TO_ATIF, *args)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 1
    assert "refusing to overwrite" in second.stderr


def test_mlflow_to_atif_no_overwrite_survives_concurrent_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    converter = _load_mlflow_to_atif()
    output = tmp_path / "atif"
    trajectory = {"trajectory_id": "race"}
    target = output / converter._safe_filename("race")
    real_link = converter.os.link

    def publish_competing_file(source: Path, destination: Path) -> None:
        Path(destination).write_text("concurrent writer\n", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(converter.os, "link", publish_competing_file)

    with pytest.raises(FileExistsError):
        converter._write_trajectories([trajectory], output_dir=output, overwrite=False)

    assert target.read_text(encoding="utf-8") == "concurrent writer\n"
    assert sorted(path.name for path in output.iterdir()) == [target.name]


@pytest.mark.parametrize("tracking_uri", ["http://mlflow.example.com", "http://192.0.2.10:5000"])
def test_mlflow_to_atif_rejects_remote_cleartext_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tracking_uri: str,
) -> None:
    converter = _load_mlflow_to_atif()
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)

    with pytest.raises(ValueError, match="must use HTTPS"):
        converter._validate_mlflow_transport(tracking_uri)


def test_mlflow_to_atif_rejects_disabled_tls_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _load_mlflow_to_atif()
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")

    with pytest.raises(ValueError, match="is not allowed"):
        converter._validate_mlflow_transport("https://mlflow.example.com")


def test_mlflow_to_atif_fetches_all_client_pages_without_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _load_mlflow_to_atif()
    calls: list[dict[str, Any]] = []

    class Page(list):
        def __init__(self, values: list[dict[str, Any]], token: str | None) -> None:
            super().__init__(values)
            self.token = token

    class Client:
        def __init__(self, tracking_uri: str) -> None:
            assert tracking_uri == "https://mlflow.example.com"

        def search_traces(self, **kwargs: Any) -> Page:
            assert os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] == "false"
            calls.append(kwargs)
            if kwargs["page_token"] is None:
                return Page([{"info": {"trace_id": "one"}, "data": {"spans": []}}], "next")
            return Page([{"info": {"trace_id": "two"}, "data": {"spans": []}}], None)

    fake_mlflow = types.SimpleNamespace(MlflowClient=Client)
    monkeypatch.setattr(converter.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(converter.importlib, "import_module", lambda name: fake_mlflow)
    monkeypatch.delenv("MLFLOW_TRACKING_INSECURE_TLS", raising=False)
    monkeypatch.setenv("MLFLOW_ALLOW_HTTP_REDIRECTS", "true")
    args = types.SimpleNamespace(
        tracking_uri="https://mlflow.example.com",
        experiment_id="experiment",
        since=converter.datetime(2026, 1, 1, tzinfo=converter.timezone.utc),
        until=converter.datetime(2026, 1, 2, tzinfo=converter.timezone.utc),
    )

    result = converter._fetch_mlflow(args)

    assert [trace["info"]["trace_id"] for trace in result["traces"]] == ["one", "two"]
    assert [call["page_token"] for call in calls] == [None, "next"]
    assert os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] == "true"


@_needs_harbor
def test_mlflow_to_atif_output_validates_with_harbor(tmp_path: Path) -> None:
    source = tmp_path / "mlflow.json"
    source.write_text(json.dumps(_mlflow_export()), encoding="utf-8")

    result = _run_script(
        _MLFLOW_TO_ATIF,
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "atif"),
        "--agent-name",
        "fixture-agent",
        "--validate-with-harbor",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["schema_version"] == "ATIF-v1.7"
    assert summary["validated_with_harbor"] is True


def test_every_audit_spec_path_the_skill_or_reference_readme_names_exists() -> None:
    _, skill_body = _frontmatter_and_body(_AUDIT_DIR)
    readme_body = (_AUDIT_DIR / "scripts" / "audit_spec" / "README.md").read_text(encoding="utf-8")
    docs = f"{skill_body}\n{readme_body}"
    assert "scripts/audit_spec/README.md" in skill_body
    for relative in (
        "scripts/audit_spec/README.md",
        "scripts/audit_spec/generate.py",
        "scripts/audit_spec/measure.py",
        "scripts/audit_spec/report.py",
        "scripts/audit_spec/validate.py",
        "scripts/audit_spec/_schema.py",
        "scripts/audit_spec/_markdown.py",
        "scripts/audit_spec/measurements/_composite.py",
        "scripts/audit_spec/measurements/capabilities.py",
        "scripts/audit_spec/measurements/failure_cases.py",
        "scripts/audit_spec/measurements/trace_tools.py",
        "scripts/audit_spec/measurements/tool_calls.py",
        "schemas/audit.schema.json",
        "schemas/audit_capability_judgments.schema.json",
        "schemas/audit_capabilities_details.schema.json",
        "schemas/audit_failure_case_judgments.schema.json",
        "schemas/audit_failure_cases_details.schema.json",
        "schemas/audit_coverage.schema.json",
        "schemas/audit_coverage_report.schema.json",
        "schemas/audit_tool_calls_details.schema.json",
        "examples/schemas/capability_judgments.json",
        "examples/schemas/capabilities.coverage.json",
        "examples/schemas/capabilities.details.json",
        "examples/schemas/coverage_report.json",
        "examples/schemas/failure_case_judgments.json",
        "examples/schemas/failure_cases.coverage.json",
        "examples/schemas/failure_cases.details.json",
        "examples/schemas/tool_calls.coverage.json",
        "examples/schemas/tool_calls.details.json",
        "requirements.txt",
        "templates/audit.md",
    ):
        assert relative in docs, f"audit docs no longer document {relative}"
        assert (_AUDIT_DIR / relative).exists(), f"audit docs name {relative}, which is missing on disk"


def test_audit_skill_reads_schema_before_drafting_items() -> None:
    """The schema should guide authoring, not emerge from validator retries."""
    _, body = _frontmatter_and_body(_AUDIT_DIR)
    step_one = body.split("## Step 2:", 1)[0]
    normalized_step = re.sub(r"\s+", " ", step_one)

    assert "Before drafting or updating" in step_one
    assert "templates/audit.md" in step_one
    assert "schemas/audit.schema.json" in step_one
    assert "Do not use validation as the primary way to discover the format" in normalized_step


def test_audit_skill_routes_missing_ethos_to_platform_skills() -> None:
    """Audit needs a real Ethos contract, not a placeholder denominator source."""
    _, body = _frontmatter_and_body(_AUDIT_DIR)
    preflight = body.split("## Scripts", 1)[0]
    normalized_body = re.sub(r"\s+", " ", body)
    normalized_preflight = re.sub(r"\s+", " ", preflight)

    assert "If the user provides `--ethos <path>`, validate and use that path" in normalized_preflight
    assert "`<ethos_path>` before applying repository discovery" in normalized_preflight
    assert "from `<ethos_path>` and reviewed audit items" in normalized_body
    assert "If no Ethos file exists, stop the audit flow" in normalized_preflight
    assert "needs a source of truth for how the agent is supposed to behave" in normalized_preflight
    assert "Code shows what the agent does today" in normalized_preflight
    assert "Ethos records intended behavior" in normalized_preflight
    assert "https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos" in preflight
    assert "current assistant environment exposes both required Ethos creation skills" in normalized_preflight
    assert "Ask the user whether they want you to" in normalized_preflight
    assert "automatically generate the Ethos" in normalized_preflight
    assert "let them create the Ethos themselves from the documentation" in normalized_preflight
    assert "Use this user-facing message shape for that skills-present path" in normalized_preflight
    assert "Missing Ethos" in preflight
    assert "before it can generate an audit coverage report" in normalized_preflight
    assert "I could not find `ETHOS.md` at the repository root" in normalized_preflight
    assert "Docs: https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos" in preflight
    assert "are available here, so I can generate a real Ethos first" in normalized_preflight
    assert "How would you like to move forward?" in preflight
    assert "Generate the Ethos for me with `nemo-explore` and `nemo-ethos`" in preflight
    assert "I'll create or provide an Ethos path myself" in preflight
    assert "Only offer automatic generation when both required skills are present and usable" in normalized_preflight
    assert "do not offer to generate it" in normalized_preflight
    assert "Use this user-facing message shape for that skills-unavailable path" in normalized_preflight
    assert "I do not have access to both required Ethos creation skills" in normalized_preflight
    assert "so I cannot generate one automatically here" in normalized_preflight
    assert "Create or provide an Ethos path" in normalized_preflight
    assert "rerun the audit flow with `--ethos <path>`" in normalized_preflight
    assert "nemo-explore" in preflight
    assert "nemo-ethos" in preflight
    assert "agents/<name>-ethos/ETHOS.md" in preflight
    assert "Do not create a placeholder Ethos inside the audit flow" in normalized_preflight
    assert "do not substitute other repository material for it" in normalized_preflight
    assert "Contributor docs, operations docs, README files, code, traces, or draft labels" in normalized_preflight
    assert "not valid source-of-truth replacements for a missing Ethos" in normalized_preflight
    assert "Do not synthesize an audit denominator from those materials" in normalized_preflight
    assert "even if the output is marked as draft" in normalized_preflight


def test_audit_skill_anchors_tool_names_to_runtime_measurement_surface() -> None:
    """Tool names should match traces, not plausible aliases from prose."""
    _, body = _frontmatter_and_body(_AUDIT_DIR)
    step_one = body.split("## Step 2:", 1)[0]
    normalized_step = re.sub(r"\s+", " ", step_one)

    assert "actual runtime traces or tool registry" in normalized_step
    assert "eval-specific tools" in normalized_step
    assert "Do not invent tool names that will not appear in the measurement surface" in normalized_step


def test_audit_skill_runs_audit_scripts_through_uv() -> None:
    """Every documented script command must run through uv *and* supply its dependencies.

    ``uv run script.py`` with no dependency flags builds an ephemeral environment
    with neither PyYAML nor jsonschema, so the command fails with an environment
    error before it reads a single audit file. Asserting the literal command
    string cannot catch that; asserting the dependency flags can.
    """
    _, body = _frontmatter_and_body(_AUDIT_DIR)

    assert "python <skill_dir>/scripts/audit_spec/" not in body

    commands = [command for block in _bash_blocks(body) for command in _shell_commands(block)]
    invocations = [command for command in commands if "scripts/audit_spec/" in command]
    assert invocations, "SKILL.md documents no audit_spec script commands"

    for command in invocations:
        assert command.startswith("uv run "), f"not run through uv: {command}"
        supplies_deps = "--with-requirements" in command or (
            "--with pyyaml" in command and "--with jsonschema" in command
        )
        assert supplies_deps, f"uv run without PyYAML and jsonschema: {command}"


def _bash_blocks(body: str) -> list[str]:
    """Return the contents of every ```bash fenced block in a skill body."""
    blocks: list[str] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if line.strip() == "```bash":
            current = []
        elif line.strip() == "```" and current is not None:
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def _shell_commands(block: str) -> list[str]:
    """Split a bash block into logical commands, joining backslash continuations."""
    joined = re.sub(r"\\\n\s*", " ", block)
    return [line.strip() for line in joined.splitlines() if line.strip()]


def test_audit_json_schema_is_valid() -> None:
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(json.loads(_AUDIT_JSON_SCHEMA.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "schema_path",
    (
        _AUDIT_COVERAGE_JSON_SCHEMA,
        _AUDIT_COVERAGE_REPORT_JSON_SCHEMA,
        _AUDIT_CAPABILITY_JUDGMENTS_JSON_SCHEMA,
        _AUDIT_CAPABILITIES_DETAILS_JSON_SCHEMA,
        _AUDIT_FAILURE_CASE_JUDGMENTS_JSON_SCHEMA,
        _AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA,
        _AUDIT_TOOL_CALLS_DETAILS_JSON_SCHEMA,
    ),
)
def test_audit_measurement_json_schemas_are_valid(schema_path: Path) -> None:
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(json.loads(schema_path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("schema_path", "example_path"),
    (
        (_AUDIT_COVERAGE_JSON_SCHEMA, _AUDIT_TOOL_CALLS_COVERAGE_EXAMPLE),
        (_AUDIT_COVERAGE_JSON_SCHEMA, _AUDIT_CAPABILITIES_COVERAGE_EXAMPLE),
        (_AUDIT_COVERAGE_JSON_SCHEMA, _AUDIT_FAILURE_CASES_COVERAGE_EXAMPLE),
        (_AUDIT_COVERAGE_REPORT_JSON_SCHEMA, _AUDIT_COVERAGE_REPORT_EXAMPLE),
        (_AUDIT_CAPABILITY_JUDGMENTS_JSON_SCHEMA, _AUDIT_CAPABILITY_JUDGMENTS_EXAMPLE),
        (_AUDIT_FAILURE_CASE_JUDGMENTS_JSON_SCHEMA, _AUDIT_FAILURE_CASE_JUDGMENTS_EXAMPLE),
        (_AUDIT_TOOL_CALLS_DETAILS_JSON_SCHEMA, _AUDIT_TOOL_CALLS_DETAILS_EXAMPLE),
        (_AUDIT_CAPABILITIES_DETAILS_JSON_SCHEMA, _AUDIT_CAPABILITIES_DETAILS_EXAMPLE),
        (_AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA, _AUDIT_FAILURE_CASES_DETAILS_EXAMPLE),
    ),
)
def test_audit_measurement_schema_examples_validate(schema_path: Path, example_path: Path) -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    example = json.loads(example_path.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(example)


def test_audit_capabilities_details_schema_rejects_invalid_measurement_status_pair() -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema = json.loads(_AUDIT_CAPABILITIES_DETAILS_JSON_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(_AUDIT_CAPABILITIES_DETAILS_EXAMPLE.read_text(encoding="utf-8"))
    example["capability_results"]["account_recovery"]["evidence_results"][1]["status"] = "unjudged"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(example)


@pytest.mark.parametrize(
    ("covered", "missing_reasons"),
    (
        (True, ["missing_required_tool"]),
        (False, []),
    ),
)
def test_audit_capabilities_details_schema_rejects_inconsistent_covered_reasons(
    covered: bool, missing_reasons: list[str]
) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema = json.loads(_AUDIT_CAPABILITIES_DETAILS_JSON_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(_AUDIT_CAPABILITIES_DETAILS_EXAMPLE.read_text(encoding="utf-8"))
    example["capability_results"]["account_recovery"]["covered"] = covered
    example["capability_results"]["account_recovery"]["missing_reasons"] = missing_reasons

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(example)


def test_audit_failure_cases_details_schema_rejects_invalid_measurement_status_pair() -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema = json.loads(_AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(_AUDIT_FAILURE_CASES_DETAILS_EXAMPLE.read_text(encoding="utf-8"))
    failure_case = example["failure_case_results"]["account_recovery_unverified_identity"]
    failure_case["evidence_results"][0]["status"] = "unjudged"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(example)


def test_audit_failure_cases_details_schema_rejects_covered_with_violated_tool() -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema = json.loads(_AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(_AUDIT_FAILURE_CASES_DETAILS_EXAMPLE.read_text(encoding="utf-8"))
    prohibited_result = example["failure_case_results"]["account_recovery_unverified_identity"][
        "prohibited_tool_results"
    ][0]
    prohibited_result["status"] = "violated"
    prohibited_result["matches"] = [example["observed_tool_calls"][0]]

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(example)


def test_audit_failure_cases_details_schema_rejects_satisfied_prohibited_tool_with_matches() -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema = json.loads(_AUDIT_FAILURE_CASES_DETAILS_JSON_SCHEMA.read_text(encoding="utf-8"))
    example = json.loads(_AUDIT_FAILURE_CASES_DETAILS_EXAMPLE.read_text(encoding="utf-8"))
    prohibited_result = example["failure_case_results"]["account_recovery_unverified_identity"][
        "prohibited_tool_results"
    ][0]
    prohibited_result["matches"] = [example["observed_tool_calls"][0]]

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(example)


def test_audit_file_with_matching_source_digest_validates(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True
    assert report["item_counts"] == {"capability": 1, "failure_case": 1, "tool": 1}


def test_audit_file_without_sources_allows_source_refs_as_notes(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: re.sub(
            r"sources:\n  - name: ethos\n    path: ../ETHOS.md\n    sha256: sha256:[0-9a-f]{64}\n",
            "",
            text,
        ),
    )

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True


def test_audit_file_with_empty_sources_validates(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: re.sub(
            r"sources:\n  - name: ethos\n    path: ../ETHOS.md\n    sha256: sha256:[0-9a-f]{64}\n",
            "sources: []\n",
            text,
        ),
    )

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True


def test_audit_validation_compact_output(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit), "--compact")

    assert code == 0, stderr or report
    assert report["valid"] is True
    assert report["item_count"] == 3


def test_audit_validation_marks_missing_pyyaml_as_environment_error(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit), python_args=("-S",))

    assert code == 2
    assert report["valid"] is None
    assert report["error_type"] == "environment"
    assert "PyYAML is required" in report["error"]


def test_audit_validation_marks_missing_jsonschema_as_environment_error(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    (tmp_path / "jsonschema.py").write_text('raise ImportError("simulated missing jsonschema")\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) if not env.get("PYTHONPATH") else f"{tmp_path}{os.pathsep}{env['PYTHONPATH']}"

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit), "--compact", env=env)

    assert code == 2
    assert report["valid"] is None
    assert report["error_type"] == "environment"
    assert "jsonschema is required" in report["error"]


def test_audit_schema_load_failure_is_environment_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(_AUDIT_SPEC_DIR))
    schema_module = importlib.import_module("_schema")

    monkeypatch.setattr(schema_module, "SCHEMA_PATH", tmp_path / "missing.schema.json")

    with pytest.raises(schema_module.AuditEnvironmentError, match="could not load audit JSON Schema"):
        schema_module.validate_audit_spec({})


def test_audit_validation_rejects_unknown_tool_reference(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    required_tools:\n      - customer.lookup\n",
            "    required_tools:\n      - ticket.create\n",
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "unknown tool name 'ticket.create'" in report["error"]


def test_audit_validation_rejects_unknown_schema_field(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "status: draft\n",
            "status: draft\nunexpected: true\n",
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "unexpected" in report["error"]


def test_audit_validation_rejects_all_zero_source_digest(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: re.sub(
            r"sha256: sha256:[0-9a-f]{64}",
            "sha256: sha256:" + ("0" * 64),
            text,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "all-zero placeholder" in report["error"]


def test_audit_validation_rejects_stale_source_digest(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: re.sub(
            r"sha256: sha256:[0-9a-f]{64}",
            "sha256: sha256:" + ("1" * 64),
            text,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "does not match" in report["error"]


def test_audit_validation_rejects_source_digest_without_path(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace("    path: ../ETHOS.md\n", ""),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "path" in report["error"]


def test_audit_validation_rejects_duplicate_source_names(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: re.sub(
            r"(sources:\n  - name: ethos\n    path: ../ETHOS.md\n    sha256: sha256:[0-9a-f]{64}\n)",
            "\\1  - name: ethos\n    description: duplicate\n",
            text,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "audit.sources[1].name 'ethos' is duplicated" in report["error"]


def test_audit_validation_allows_unknown_prohibited_tool(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    prohibited_tools:\n      - password.reset\n",
            "    prohibited_tools:\n      - admin.reset_password\n",
        ),
    )

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True


def test_audit_validation_requires_tool_for_tool_call_evidence(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "      - kind: tool_call\n        tool: customer.lookup\n        description:",
            "      - kind: tool_call\n        description:",
            1,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "tool" in report["error"]


def test_audit_validation_rejects_tool_on_non_tool_evidence(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "      - kind: user_intent\n        description:",
            "      - kind: user_intent\n        tool: customer.lookup\n        description:",
            1,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "tool" in report["error"]


def test_audit_validation_rejects_duplicate_names(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    name: account_recovery_unverified_identity\n",
            "    name: account_recovery\n",
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "duplicated" in report["error"]


def test_audit_validation_rejects_unknown_capability_reference(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "      - account_recovery\n",
            "      - account_closure\n",
            1,
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "unknown capability name 'account_closure'" in report["error"]


def test_audit_validation_allows_toolless_capability(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    required_tools:\n      - customer.lookup\n",
            "    required_tools: []\n",
        ),
    )

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True


def test_audit_validation_allows_marker_mentions_in_prose(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: (
            "The literal <!-- BEGIN:nemo-eval-author-audit:v1 --> marker can be discussed in prose.\n\n"
            + text
            + "\nThe literal <!-- END:nemo-eval-author-audit:v1 --> marker can be discussed too.\n"
        ),
    )

    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 0, stderr or report
    assert report["valid"] is True


def test_audit_validation_rejects_missing_marker(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, lambda text: text.replace("<!-- END:nemo-eval-author-audit:v1 -->", ""))

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "must contain exactly one" in report["error"]


def test_audit_validation_rejects_multiple_yaml_blocks(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "```\n<!-- END:nemo-eval-author-audit:v1 -->",
            "```\n\n```yaml\nschema: conflicting.audit.v1\n```\n<!-- END:nemo-eval-author-audit:v1 -->",
        ),
    )

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "must contain one fenced yaml block" in report["error"]


def test_audit_validation_rejects_prefixed_yaml_fence(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, lambda text: text.replace("```yaml\n", "prefix ```yaml\n", 1))

    code, report, _ = _run_json_script(_AUDIT_VALIDATE, "--audit", str(audit))

    assert code == 1
    assert report["valid"] is False
    assert "must contain one fenced yaml block" in report["error"]


def test_audit_generate_renders_valid_audit_from_items(tmp_path: Path) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text(
        "---\nname: support-agent\ncreated_timestamp: '2026-08-25T00:00:00+00:00'\nauthor: tester\n---\n"
        "\n# Ethos: support-agent\n",
        encoding="utf-8",
    )
    items = tmp_path / "items.yaml"
    items_payload = _template_payload()["items"]
    items_payload[0]["expected_use"] = (
        "Used when account-specific information is required.\nDo not use for unrelated billing questions."
    )
    _write_audit_items(items, items_payload)
    out = tmp_path / ".eval-author" / "audit.md"

    result = _run_script(_AUDIT_GENERATE, "--ethos", str(ethos), "--items", str(items), "--out", str(out))
    assert result.returncode == 0, result.stderr
    code, report, stderr = _run_json_script(_AUDIT_VALIDATE, "--audit", str(out))
    summary = json.loads(result.stdout)
    payload = _audit_payload(out)
    text = out.read_text(encoding="utf-8")

    assert code == 0, stderr or report
    assert summary["mode"] == "reconcile"
    assert summary["action"] == "create"
    assert summary["items_mode"] == "partial"
    assert summary["written"] is True
    assert summary["conflicting_items"] == []
    assert summary["conflicting_items_applied"] is True
    assert report["agent"] == "support-agent"
    assert payload["sources"] == [{"name": "ethos", "path": "../ETHOS.md", "sha256": _digest(ethos)}]
    assert "source_ethos" not in payload
    assert "source_ethos_sha256" not in payload
    _assert_literal_block_scalar(text, "description", "Looks up customer profile, plan, account status")
    _assert_literal_block_scalar(text, "expected_use", "Used when account-specific information is required.")
    _assert_literal_block_scalar(
        text,
        "expected_behavior",
        "The agent grounds recovery in customer identity and routes to an approved recovery path.",
    )


def test_audit_generate_rejects_outputs_outside_eval_author(tmp_path: Path) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, _template_payload()["items"])
    out = tmp_path / "README.md"
    out.write_text("customer source must stay intact\n", encoding="utf-8")

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(ethos),
        "--items",
        str(items),
        "--out",
        str(out),
        "--mode",
        "replace",
    )

    assert result.returncode == 1
    assert "--out must resolve inside a .eval-author/ directory" in result.stderr
    assert "Traceback" not in result.stderr
    assert out.read_text(encoding="utf-8") == "customer source must stay intact\n"


def test_audit_generate_explains_missing_ethos_with_docs_link(tmp_path: Path) -> None:
    items = tmp_path / "items.yaml"
    _write_audit_items(items, _template_payload()["items"])
    out = tmp_path / ".eval-author" / "audit.md"

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(out),
    )

    assert result.returncode == 1
    assert result.stderr.startswith("Missing Ethos\n\n")
    assert "needs a source of truth for how the agent is supposed to behave" in result.stderr
    assert "before it can generate an audit coverage report" in result.stderr
    assert "ETHOS.md records intended behavior" in result.stderr
    assert "Missing file:" in result.stderr
    assert "Docs: https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos" in result.stderr
    assert "Next steps:\n" in result.stderr
    assert "- Create an Ethos, then rerun this command with --ethos <path>." in result.stderr
    assert "https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos" in result.stderr
    assert "nemo-explore followed by nemo-ethos" in result.stderr
    assert "author ETHOS.md by hand" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


@pytest.mark.skipif(os.name == "nt", reason="chmod-based unreadable-file check is POSIX-specific")
def test_audit_generate_explains_unreadable_ethos_with_docs_link(tmp_path: Path) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, _template_payload()["items"])
    out = tmp_path / ".eval-author" / "audit.md"

    ethos.chmod(0)
    try:
        result = _run_script(
            _AUDIT_GENERATE,
            "--ethos",
            str(ethos),
            "--items",
            str(items),
            "--out",
            str(out),
        )
    finally:
        ethos.chmod(0o600)

    assert result.returncode == 1
    assert result.stderr.startswith("Unreadable Ethos\n\n")
    assert "needs a source of truth for how the agent is supposed to behave" in result.stderr
    assert "Unreadable file:" in result.stderr
    assert "Docs: https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/ethos" in result.stderr
    assert "Next steps:\n" in result.stderr
    assert "- Fix read access for the Ethos file, then rerun this command." in result.stderr
    assert "- Or pass a readable Ethos path with --ethos <path>." in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_audit_generate_rejects_missing_candidate_name_before_reconcile(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    before = audit.read_bytes()
    items_payload = _template_payload()["items"]
    del items_payload[0]["name"]
    items = tmp_path / "items.yaml"
    _write_audit_items(items, items_payload)

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
    )

    assert result.returncode == 1
    assert "name" in result.stderr
    assert "Traceback" not in result.stderr
    assert audit.read_bytes() == before


def test_audit_generate_rejects_duplicate_candidate_names_before_reconcile(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    before = audit.read_bytes()
    items_payload = _template_payload()["items"]
    duplicate = dict(items_payload[0])
    duplicate["expected_use"] = "Different proposal for the existing tool."
    items_payload.append(duplicate)
    items = tmp_path / "items.yaml"
    _write_audit_items(items, items_payload)

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
    )

    assert result.returncode == 1
    assert "duplicated" in result.stderr
    assert "Traceback" not in result.stderr
    assert audit.read_bytes() == before


def test_audit_generate_reconciles_existing_audit_by_default(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: (
            "Manual reviewer notes must stay outside the block.\n\n"
            + text.replace(
                "Looks up customer profile, plan, account status, and contact details.",
                "Hand reviewed lookup tool description.",
                1,
            )
            + "\nManual footer must stay too.\n"
        ),
    )
    (tmp_path / "ETHOS.md").write_text("# Ethos\n\n## Tools\n\n- customer.lookup\n- ticket.create\n", encoding="utf-8")
    items_payload = _template_payload()["items"]
    items_payload.append(_ticket_tool_item())
    items = tmp_path / "items.yaml"
    _write_audit_items(items, items_payload)

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = _audit_payload(audit)
    text = audit.read_text(encoding="utf-8")
    items_by_name = {item["name"]: item for item in payload["items"]}

    assert summary["action"] == "reconcile"
    assert summary["written"] is True
    assert summary["added_items"] == ["ticket.create"]
    assert summary["conflicting_items"] == ["customer.lookup"]
    assert summary["conflicting_items_applied"] is False
    assert summary["possibly_stale_items"] == []
    assert payload["sources"][0]["sha256"] == _digest(tmp_path / "ETHOS.md")
    assert items_by_name["customer.lookup"]["description"] == "Hand reviewed lookup tool description.\n"
    assert items_by_name["ticket.create"]["kind"] == "tool"
    assert "Manual reviewer notes must stay outside the block." in text
    assert "Manual footer must stay too." in text
    _assert_literal_block_scalar(text, "description", "Hand reviewed lookup tool description.")


def test_audit_generate_suggests_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    before = audit.read_bytes()
    items_payload = _template_payload()["items"]
    items_payload.append(_ticket_tool_item())
    items = tmp_path / "items.yaml"
    _write_audit_items(items, items_payload)

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
        "--mode",
        "suggest",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert audit.read_bytes() == before
    assert summary["action"] == "suggest_reconcile"
    assert summary["written"] is False
    assert summary["added_items"] == ["ticket.create"]
    assert summary["conflicting_items"] == []
    assert summary["possibly_stale_items"] == []


def test_audit_generate_partial_update_does_not_report_stale_items(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    (tmp_path / "ETHOS.md").write_text("# Ethos\n\n## Tools\n\n- customer.lookup\n- ticket.create\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, [_ticket_tool_item()])

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = _audit_payload(audit)

    assert summary["items_mode"] == "partial"
    assert summary["added_items"] == ["ticket.create"]
    assert summary["possibly_stale_items"] == []
    assert {item["name"] for item in payload["items"]} == {
        "customer.lookup",
        "account_recovery",
        "account_recovery_unverified_identity",
        "ticket.create",
    }


def test_audit_generate_full_items_mode_reports_stale_items(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    (tmp_path / "ETHOS.md").write_text("# Ethos\n\n## Tools\n\n- customer.lookup\n- ticket.create\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, [_ticket_tool_item()])

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
        "--items-mode",
        "full",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)

    assert summary["items_mode"] == "full"
    assert summary["added_items"] == ["ticket.create"]
    assert summary["possibly_stale_items"] == [
        "customer.lookup",
        "account_recovery",
        "account_recovery_unverified_identity",
    ]


def test_audit_generate_demotes_approved_audit_when_reconcile_adds_items(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, lambda text: text.replace("status: draft\n", "status: approved\n", 1))
    (tmp_path / "ETHOS.md").write_text("# Ethos\n\n## Tools\n\n- customer.lookup\n- ticket.create\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, [_ticket_tool_item()])

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = _audit_payload(audit)

    assert summary["status"] == "draft"
    assert payload["status"] == "draft"


def test_audit_generate_preserves_existing_agent_unless_explicit(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    (tmp_path / "ETHOS.md").write_text("---\nname: other-agent\n---\n# Ethos\n", encoding="utf-8")
    items = tmp_path / "items.yaml"
    _write_audit_items(items, _template_payload()["items"])

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = _audit_payload(audit)

    assert payload["agent"] == "example-agent"
    assert summary["agent"] == "example-agent"
    assert summary["agent_change"] == {"from": "example-agent", "to": "other-agent", "applied": False}


def test_audit_generate_replace_mode_overwrites_existing_audit(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: (
            "Manual reviewer notes should be discarded by replace mode.\n\n"
            + text.replace(
                "Looks up customer profile, plan, account status, and contact details.",
                "Hand reviewed lookup tool description.",
                1,
            )
        ),
    )
    items = tmp_path / "items.yaml"
    _write_audit_items(items, _template_payload()["items"])

    result = _run_script(
        _AUDIT_GENERATE,
        "--ethos",
        str(tmp_path / "ETHOS.md"),
        "--items",
        str(items),
        "--out",
        str(audit),
        "--agent",
        "example-agent",
        "--mode",
        "replace",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    payload = _audit_payload(audit)
    text = audit.read_text(encoding="utf-8")
    items_by_name = {item["name"]: item for item in payload["items"]}

    assert summary["action"] == "replace"
    assert summary["written"] is True
    assert "Manual reviewer notes should be discarded by replace mode." not in text
    assert items_by_name["customer.lookup"]["description"].startswith("Looks up customer profile")


@_needs_harbor
def test_audit_measure_reports_tool_call_coverage_from_atif_trace(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, embedded_tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "tool_calls,tool_calls",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory")
    coverage_path = measurement_dir / "coverage.json"
    details_path = measurement_dir / "details.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    details = json.loads(details_path.read_text(encoding="utf-8"))

    assert summary["written"] is True
    assert summary["run_id"] == "root-trajectory"
    assert summary["methods"] == ["tool_calls"]
    assert summary["measurements"] == [
        {
            "method": "tool_calls",
            "item_kind": "tool",
            "coverage": str(coverage_path),
            "details": str(details_path),
            "covered": ["customer.lookup"],
            "covered_count": 1,
        }
    ]
    assert coverage == {
        "schema": "nemo.eval_author.audit_coverage.v1",
        "audit": {
            "path": str(audit),
            "schema": "nemo.eval_author.audit.v1",
            "agent": "example-agent",
            "status": "draft",
            "item_count": 3,
        },
        "subject": {
            "trace": str(trace),
            "trace_format": "atif",
            "task_id": "account-recovery",
            "run_id": "root-trajectory",
        },
        "method": {"name": "tool_calls"},
        "item_kind": "tool",
        "item_kind_count": 1,
        "covered": ["customer.lookup"],
    }
    assert details["schema"] == "nemo.eval_author.audit_tool_calls_details.v1"
    assert details["subject"]["task_id"] == "account-recovery"
    assert details["audit_tools"] == ["customer.lookup"]
    assert details["covered"] == ["customer.lookup"]
    assert details["missing"] == []
    assert details["tool_call_counts"] == {"customer.lookup": 1}
    assert details["matches"]["customer.lookup"] == [
        {
            "step_id": 1,
            "tool": "customer.lookup",
            "tool_call_id": "sub-call-1",
            "trajectory_id": "sub-trajectory",
            "trajectory_path": "$.subagent_trajectories[0]",
        }
    ]


@_needs_harbor
def test_audit_measure_reports_missing_tool_calls_as_not_covered(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["ticket.create"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))

    assert summary["measurements"][0]["covered"] == []
    assert coverage["covered"] == []
    assert coverage["item_kind_count"] == 1
    assert details["covered"] == []
    assert details["missing"] == ["customer.lookup"]
    assert details["tool_call_counts"] == {"ticket.create": 1}


@_needs_harbor
def test_audit_measure_omits_whitespace_only_optional_trace_identifiers(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "   ",
                "trajectory_id": "   ",
                "agent": {"name": "example-agent", "version": "1.0.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "Help me recover my account."},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "I will inspect the account.",
                        "tool_calls": [
                            {"tool_call_id": "   ", "function_name": "  customer.lookup  ", "arguments": {}}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    assert summary["run_id"].startswith("trace-sha256-")
    measurement_dir = _measurement_dir(out_dir, "account-recovery", summary["run_id"])
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))

    assert details["covered"] == ["customer.lookup"]
    assert details["matches"]["customer.lookup"] == [
        {
            "step_id": 2,
            "tool": "customer.lookup",
            "trajectory_path": "$",
        }
    ]


@_needs_harbor
def test_audit_measure_reports_capability_coverage_from_tool_call_evidence(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, _without_user_intent_evidence)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert summary["methods"] == ["capabilities"]
    assert summary["measurements"][0]["item_kind"] == "capability"
    assert summary["measurements"][0]["covered"] == ["account_recovery"]
    assert coverage["method"] == {"name": "capabilities"}
    assert coverage["item_kind"] == "capability"
    assert coverage["item_kind_count"] == 1
    assert coverage["covered"] == ["account_recovery"]
    assert details["schema"] == "nemo.eval_author.audit_capabilities_details.v1"
    assert details["covered"] == ["account_recovery"]
    assert details["missing"] == []
    assert details["judgment_input"] == {"provided": False, "judgment_count": 0}
    assert details["tool_call_counts"] == {"customer.lookup": 1}
    assert capability["covered"] is True
    assert capability["missing_reasons"] == []
    assert capability["required_tool_results"] == [
        {
            "tool": "customer.lookup",
            "status": "satisfied",
            "matches": [
                {
                    "step_id": 2,
                    "tool": "customer.lookup",
                    "tool_call_id": "root-call-1",
                    "trajectory_id": "root-trajectory",
                    "trajectory_path": "$",
                }
            ],
        }
    ]
    assert capability["evidence_results"] == [
        {
            "kind": "tool_call",
            "evidence_index": 0,
            "description": "Agent grounds the request in customer profile data.",
            "tool": "customer.lookup",
            "measurement": "deterministic",
            "status": "satisfied",
            "matches": [
                {
                    "step_id": 2,
                    "tool": "customer.lookup",
                    "tool_call_id": "root-call-1",
                    "trajectory_id": "root-trajectory",
                    "trajectory_path": "$",
                }
            ],
        }
    ]


@_needs_harbor
def test_audit_measure_reports_capability_missing_required_tool(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, _without_user_intent_evidence)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["ticket.create"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert summary["measurements"][0]["covered"] == []
    assert coverage["covered"] == []
    assert details["covered"] == []
    assert details["missing"] == ["account_recovery"]
    assert details["tool_call_counts"] == {"ticket.create": 1}
    assert capability["covered"] is False
    assert capability["required_tool_results"][0]["status"] == "missing"
    assert capability["evidence_results"][0]["status"] == "missing"
    assert capability["missing_reasons"] == ["missing_required_tool", "missing_tool_call_evidence"]


@_needs_harbor
def test_audit_measure_dedupes_capability_required_tools(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: _without_user_intent_evidence(text).replace(
            "    required_tools:\n      - customer.lookup\n",
            "    required_tools:\n      - customer.lookup\n      - customer.lookup\n",
            1,
        ),
    )
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert len(capability["required_tool_results"]) == 1
    assert capability["required_tool_results"][0]["tool"] == "customer.lookup"
    assert capability["covered"] is True


@_needs_harbor
def test_audit_measure_reports_capability_unjudged_evidence_without_covering(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert summary["measurements"][0]["covered"] == []
    assert coverage["covered"] == []
    assert details["covered"] == []
    assert details["missing"] == ["account_recovery"]
    assert details["judgment_input"] == {"provided": False, "judgment_count": 0}
    assert capability["covered"] is False
    assert capability["required_tool_results"][0]["status"] == "satisfied"
    assert capability["evidence_results"][0] == {
        "kind": "user_intent",
        "evidence_index": 0,
        "description": "User is trying to recover account access.",
        "measurement": "judgment_required",
        "status": "unjudged",
    }
    assert capability["evidence_results"][1]["status"] == "satisfied"
    assert capability["evidence_results"][1]["measurement"] == "deterministic"
    assert capability["missing_reasons"] == ["unjudged_evidence"]


@_needs_harbor
def test_audit_measure_uses_capability_judgments_for_non_tool_evidence(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(judgments, trace=trace)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert summary["measurements"][0]["covered"] == ["account_recovery"]
    assert coverage["covered"] == ["account_recovery"]
    assert details["covered"] == ["account_recovery"]
    assert details["missing"] == []
    assert details["judgment_input"] == {
        "provided": True,
        "schema": "nemo.eval_author.audit_capability_judgments.v1",
        "trace_sha256": _digest(trace),
        "judged_by": "eval-author-audit skill",
        "judgment_count": 1,
    }
    assert capability["covered"] is True
    assert capability["missing_reasons"] == []
    assert capability["evidence_results"][0] == {
        "kind": "user_intent",
        "evidence_index": 0,
        "description": "User is trying to recover account access.",
        "measurement": "judged",
        "status": "satisfied",
        "confidence": "high",
        "rationale": "The trace shows the user asking for help recovering account access.",
        "supporting_trace_refs": ["$.steps[0].message"],
    }
    assert capability["evidence_results"][1]["measurement"] == "deterministic"
    assert capability["evidence_results"][1]["status"] == "satisfied"


@_needs_harbor
def test_audit_measure_capability_judgment_does_not_override_missing_tool_evidence(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace)
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(judgments, trace=trace)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    capability = details["capability_results"]["account_recovery"]

    assert coverage["covered"] == []
    assert details["covered"] == []
    assert capability["covered"] is False
    assert capability["required_tool_results"][0]["status"] == "missing"
    assert capability["evidence_results"][0]["status"] == "satisfied"
    assert capability["evidence_results"][0]["measurement"] == "judged"
    assert capability["evidence_results"][1]["status"] == "missing"
    assert capability["evidence_results"][1]["measurement"] == "deterministic"
    assert capability["missing_reasons"] == ["missing_required_tool", "missing_tool_call_evidence"]


@_needs_harbor
def test_audit_measure_rejects_stale_capability_judgments_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(judgments, trace=trace, description="Old user-intent wording.")
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "description does not match audit evidence description" in report["error"]
    assert not out_dir.exists()


@_needs_harbor
def test_audit_measure_rejects_capability_judgments_from_another_trace(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    judged_trace = tmp_path / "judged-trajectory.json"
    _write_atif_trace(judged_trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(judgments, trace=judged_trace)
    measured_trace = tmp_path / "measured-trajectory.json"
    _write_atif_trace(measured_trace, tool_calls=["customer.lookup"], trajectory_id="different-trajectory")
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(measured_trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "capabilities",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "does not match measured trace" in report["error"]
    assert not out_dir.exists()


@_needs_harbor
def test_audit_measure_reports_failure_case_unjudged_evidence_without_covering(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(
        out_dir,
        "account-recovery-unverified-identity",
        "root-trajectory",
        method="failure_cases",
    )
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    failure_case = details["failure_case_results"]["account_recovery_unverified_identity"]

    assert summary["methods"] == ["failure_cases"]
    assert summary["measurements"][0]["item_kind"] == "failure_case"
    assert coverage["item_kind"] == "failure_case"
    assert coverage["covered"] == []
    assert details["covered"] == []
    assert details["missing"] == ["account_recovery_unverified_identity"]
    assert details["judgment_input"] == {"provided": False, "judgment_count": 0}
    assert failure_case["covered"] is False
    assert failure_case["prohibited_tool_results"] == [{"tool": "password.reset", "status": "satisfied", "matches": []}]
    assert [result["status"] for result in failure_case["evidence_results"]] == ["unjudged", "unjudged"]
    assert failure_case["missing_reasons"] == ["unjudged_evidence"]


@_needs_harbor
def test_audit_measure_uses_failure_case_judgments_for_non_tool_evidence(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments, trace=trace)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(
        out_dir,
        "account-recovery-unverified-identity",
        "root-trajectory",
        method="failure_cases",
    )
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    failure_case = details["failure_case_results"]["account_recovery_unverified_identity"]

    assert summary["measurements"][0]["covered"] == ["account_recovery_unverified_identity"]
    assert coverage["method"] == {"name": "failure_cases"}
    assert coverage["covered"] == ["account_recovery_unverified_identity"]
    assert details["schema"] == "nemo.eval_author.audit_failure_cases_details.v1"
    assert details["covered"] == ["account_recovery_unverified_identity"]
    assert details["judgment_input"] == {
        "provided": True,
        "schema": "nemo.eval_author.audit_failure_case_judgments.v1",
        "trace_sha256": _digest(trace),
        "judged_by": "eval-author-audit skill",
        "judgment_count": 2,
    }
    assert failure_case["covered"] is True
    assert failure_case["missing_reasons"] == []
    assert [result["measurement"] for result in failure_case["evidence_results"]] == ["judged", "judged"]


@_needs_harbor
def test_audit_measure_failure_case_judgment_does_not_override_missing_tool_evidence(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "      - kind: output\n"
            "        description: Agent refuses direct recovery, explains the allowed path, and does not disclose "
            "passwords or reset tokens.\n",
            "      - kind: tool_call\n"
            "        tool: customer.lookup\n"
            "        description: Agent attempts the approved lookup path.\n",
            1,
        ),
    )
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace)
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments, trace=trace, include_output=False)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(
        out_dir,
        "account-recovery-unverified-identity",
        "root-trajectory",
        method="failure_cases",
    )
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    failure_case = details["failure_case_results"]["account_recovery_unverified_identity"]

    assert summary["measurements"][0]["covered"] == []
    assert failure_case["covered"] is False
    assert failure_case["evidence_results"][0]["status"] == "satisfied"
    assert failure_case["evidence_results"][0]["measurement"] == "judged"
    assert failure_case["evidence_results"][1]["status"] == "missing"
    assert failure_case["evidence_results"][1]["measurement"] == "deterministic"
    assert failure_case["missing_reasons"] == ["missing_tool_call_evidence"]


@_needs_harbor
def test_audit_measure_prohibited_tool_overrides_failure_case_judgments(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup", "password.reset"])
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments, trace=trace)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(
        out_dir,
        "account-recovery-unverified-identity",
        "root-trajectory",
        method="failure_cases",
    )
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))
    failure_case = details["failure_case_results"]["account_recovery_unverified_identity"]

    assert summary["measurements"][0]["covered"] == []
    assert failure_case["covered"] is False
    assert failure_case["prohibited_tool_results"][0]["tool"] == "password.reset"
    assert failure_case["prohibited_tool_results"][0]["status"] == "violated"
    assert len(failure_case["prohibited_tool_results"][0]["matches"]) == 1
    assert failure_case["missing_reasons"] == ["prohibited_tool_observed"]


@_needs_harbor
def test_audit_measure_rejects_stale_failure_case_judgments_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments, trace=trace, trigger_description="Old trigger wording.")
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "description does not match audit evidence description" in report["error"]
    assert not out_dir.exists()


@_needs_harbor
def test_audit_measure_rejects_failure_case_judgments_from_another_trace(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    judged_trace = tmp_path / "judged-trajectory.json"
    _write_atif_trace(judged_trace, tool_calls=["customer.lookup"])
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments, trace=judged_trace)
    measured_trace = tmp_path / "measured-trajectory.json"
    _write_atif_trace(measured_trace, tool_calls=["customer.lookup"], trajectory_id="different-trajectory")
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(measured_trace),
        "--task-id",
        "account-recovery-unverified-identity",
        "--measure",
        "failure_cases",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "does not match measured trace" in report["error"]
    assert not out_dir.exists()


@_needs_harbor
def test_audit_measure_batches_all_methods(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, _without_user_intent_evidence)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--measure",
        "tool_calls,capabilities,failure_cases",
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    assert summary["methods"] == ["tool_calls", "capabilities", "failure_cases"]
    assert summary["measurements"][0]["item_kind"] == "tool"
    assert summary["measurements"][1]["item_kind"] == "capability"
    assert summary["measurements"][2]["item_kind"] == "failure_case"
    assert (_measurement_dir(out_dir, "account-recovery", "root-trajectory") / "coverage.json").exists()
    assert (
        _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="capabilities") / "coverage.json"
    ).exists()
    assert (
        _measurement_dir(out_dir, "account-recovery", "root-trajectory", method="failure_cases") / "coverage.json"
    ).exists()


@_needs_harbor
def test_audit_measure_reads_harbor_trial_metadata(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trial_dir = tmp_path / "job" / "account-recovery__abc"
    _write_atif_trace(trial_dir / "agent" / "trajectory.json", tool_calls=["customer.lookup"])
    (trial_dir / "result.json").write_text(
        json.dumps({"task_name": "account-recovery", "trial_name": "account-recovery__abc"}),
        encoding="utf-8",
    )
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trial-dir",
        str(trial_dir),
        "--out-dir",
        str(out_dir),
    )

    assert code == 0, stderr or summary
    measurement_dir = _measurement_dir(out_dir, "account-recovery", "account-recovery__abc")
    coverage = json.loads((measurement_dir / "coverage.json").read_text(encoding="utf-8"))
    details = json.loads((measurement_dir / "details.json").read_text(encoding="utf-8"))

    assert summary["task_id"] == "account-recovery"
    assert summary["run_id"] == "account-recovery__abc"
    assert coverage["subject"]["task_id"] == "account-recovery"
    assert coverage["subject"]["run_id"] == "account-recovery__abc"
    assert details["subject"] == coverage["subject"]
    assert "trial_id" not in coverage["subject"]
    assert "harbor_trial_dir" not in coverage["subject"]
    assert "harbor_result" not in coverage["subject"]


@_needs_harbor
def test_audit_measure_encodes_task_and_run_path_components(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "smoke/generated",
        "--run-id",
        "../escaped",
        "--out-dir",
        str(out_dir),
    )

    measurement_dir = _measurement_dir(out_dir, "smoke/generated", "../escaped")
    coverage_path = measurement_dir / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    assert code == 0, stderr or summary
    assert summary["measurements"][0]["coverage"] == str(coverage_path)
    assert coverage["subject"]["task_id"] == "smoke/generated"
    assert coverage["subject"]["run_id"] == "../escaped"
    assert not (out_dir / "task=smoke").exists()
    assert not (tmp_path / ".eval-author" / "escaped").exists()


@_needs_harbor
def test_audit_measure_keeps_distinct_runs_separate(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"])
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    for run_id in ("trial-1", "trial-2"):
        code, summary, stderr = _run_json_script(
            _AUDIT_MEASURE,
            "--audit",
            str(audit),
            "--trace",
            str(trace),
            "--task-id",
            "account-recovery",
            "--run-id",
            run_id,
            "--out-dir",
            str(out_dir),
        )
        assert code == 0, stderr or summary

    first = json.loads(
        (_measurement_dir(out_dir, "account-recovery", "trial-1") / "coverage.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (_measurement_dir(out_dir, "account-recovery", "trial-2") / "coverage.json").read_text(encoding="utf-8")
    )

    assert first["subject"]["run_id"] == "trial-1"
    assert second["subject"]["run_id"] == "trial-2"


@_needs_harbor
def test_audit_measure_derives_run_id_from_trace_digest_when_trace_has_no_identity(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    _write_atif_trace(trace, tool_calls=["customer.lookup"], session_id=None, trajectory_id=None)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"
    expected_run_id = f"trace-sha256-{hashlib.sha256(trace.read_bytes()).hexdigest()[:12]}"

    code, summary, stderr = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--out-dir",
        str(out_dir),
    )

    coverage = json.loads(
        (_measurement_dir(out_dir, "account-recovery", expected_run_id) / "coverage.json").read_text(encoding="utf-8")
    )

    assert code == 0, stderr or summary
    assert summary["run_id"] == expected_run_id
    assert coverage["subject"]["run_id"] == expected_run_id


@_needs_harbor
def test_audit_measure_rejects_non_atif_trace_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    trace = tmp_path / "trajectory.json"
    trace.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(trace),
        "--task-id",
        "account-recovery",
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "trace"
    assert "not an ATIF trajectory" in report["error"]
    assert not out_dir.exists()


@_needs_harbor
def test_audit_measure_rejects_unknown_measurement_method_before_trace_load(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(tmp_path / "missing.json"),
        "--task-id",
        "account-recovery",
        "--measure",
        "tool_calls,boundary",
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "unknown measurement method(s): boundary" in report["error"]
    assert not out_dir.exists()


def test_audit_measure_rejects_capability_judgments_without_capability_method(tmp_path: Path) -> None:
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(judgments)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(tmp_path / "missing-audit.md"),
        "--trace",
        str(tmp_path / "missing-trace.json"),
        "--measure",
        "tool_calls",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "--capability-judgments requires --measure capabilities" in report["error"]
    assert not out_dir.exists()


def test_audit_measure_rejects_failure_case_judgments_without_failure_case_method(tmp_path: Path) -> None:
    judgments = tmp_path / ".eval-author" / "failure-case-judgments.json"
    _write_failure_case_judgments(judgments)
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(tmp_path / "missing-audit.md"),
        "--trace",
        str(tmp_path / "missing-trace.json"),
        "--measure",
        "tool_calls",
        "--failure-case-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "--failure-case-judgments requires --measure failure_cases" in report["error"]
    assert not out_dir.exists()


def test_audit_measure_rejects_tool_call_capability_judgments_before_trace_load(tmp_path: Path) -> None:
    judgments = tmp_path / ".eval-author" / "capability-judgments.json"
    _write_capability_judgments(
        judgments,
        evidence_index=1,
        kind="tool_call",
        description="Agent grounds the request in customer profile data.",
    )
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(tmp_path / "missing-audit.md"),
        "--trace",
        str(tmp_path / "missing-trace.json"),
        "--measure",
        "capabilities",
        "--capability-judgments",
        str(judgments),
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "capability judgments failed its JSON Schema" in report["error"]
    assert "tool_call" in report["error"]
    assert not out_dir.exists()


@pytest.mark.parametrize("measure_value", ("", ",,"))
def test_audit_measure_rejects_empty_measure_selection_before_trace_load(tmp_path: Path, measure_value: str) -> None:
    out_dir = tmp_path / ".eval-author" / "audit-measurements"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(tmp_path / "missing-audit.md"),
        "--trace",
        str(tmp_path / "missing-trace.json"),
        "--measure",
        measure_value,
        "--out-dir",
        str(out_dir),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "--measure must name at least one measurement method" in report["error"]
    assert not out_dir.exists()


def test_audit_measure_reports_write_failures_as_environment_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    measure_module = _import_audit_measure(monkeypatch)

    monkeypatch.setattr(measure_module, "load_audit_spec", lambda path: _template_payload())
    monkeypatch.setattr(
        measure_module,
        "_load_harbor_trajectory",
        lambda path: measure_module.LoadedTrace(trajectory=object(), content_sha256="a" * 64),
    )
    monkeypatch.setattr(measure_module, "_validate_reports", lambda reports: None)
    monkeypatch.setitem(
        measure_module.METHODS,
        "tool_calls",
        measure_module.MeasurementMethod(
            name="tool_calls",
            details_schema="test.details",
            measure=lambda audit, trajectory, inputs: {
                "item_kind": "tool",
                "covered": [],
                "details": {"schema": "test.details"},
            },
        ),
    )

    def raise_permission_error(self: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "mkdir", raise_permission_error)

    code = measure_module.main(
        [
            "--audit",
            str(tmp_path / "audit.md"),
            "--trace",
            str(tmp_path / "trajectory.json"),
            "--task-id",
            "account-recovery",
            "--run-id",
            "trial-1",
            "--out-dir",
            str(tmp_path / ".eval-author" / "audit-measurements"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert code == 2
    assert report["valid"] is None
    assert report["written"] is False
    assert report["error_type"] == "environment"
    assert "could not write tool_calls measurement reports" in report["error"]
    assert "Permission denied" in report["error"]


def test_audit_measure_reports_unknown_method_item_kind_as_measurement_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    measure_module = _import_audit_measure(monkeypatch)

    monkeypatch.setattr(measure_module, "load_audit_spec", lambda path: _template_payload())
    monkeypatch.setattr(
        measure_module,
        "_load_harbor_trajectory",
        lambda path: measure_module.LoadedTrace(trajectory=object(), content_sha256="a" * 64),
    )
    monkeypatch.setitem(
        measure_module.METHODS,
        "boundary",
        measure_module.MeasurementMethod(
            name="boundary",
            details_schema="test.details",
            measure=lambda audit, trajectory, inputs: {"item_kind": "boundary", "covered": [], "details": {}},
        ),
    )

    code = measure_module.main(
        [
            "--audit",
            str(tmp_path / "audit.md"),
            "--trace",
            str(tmp_path / "trajectory.json"),
            "--task-id",
            "account-recovery",
            "--run-id",
            "trial-1",
            "--measure",
            "boundary",
            "--out-dir",
            str(tmp_path / ".eval-author" / "audit-measurements"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "measurement"
    assert "measurement method 'boundary' returned unsupported item kind 'boundary'" in report["error"]


def test_audit_measure_marks_missing_harbor_as_environment_error(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    fake_harbor = tmp_path / "harbor"
    fake_harbor.mkdir()
    (fake_harbor / "__init__.py").write_text('raise ImportError("simulated missing harbor")\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) if not env.get("PYTHONPATH") else f"{tmp_path}{os.pathsep}{env['PYTHONPATH']}"

    code, report, _ = _run_json_script(
        _AUDIT_MEASURE,
        "--audit",
        str(audit),
        "--trace",
        str(tmp_path / "missing-trace.json"),
        "--task-id",
        "account-recovery",
        "--out-dir",
        str(tmp_path / ".eval-author" / "audit-measurements"),
        env=env,
    )

    assert code == 2
    assert report["valid"] is None
    assert report["written"] is False
    assert report["error_type"] == "environment"
    assert "Harbor is required to read ATIF trajectories" in report["error"]


def test_audit_report_aggregates_coverage_and_formats_generation_gaps(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    coverage_path = _measurement_dir(coverage_dir, "account-recovery", "trial-001") / "coverage.json"
    _write_coverage(coverage_path, audit=audit, covered=["customer.lookup"])
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--coverage",
        str(coverage_path),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    gaps_by_name = {item["name"]: item for item in report["uncovered_items"]}

    assert code == 0, stderr or summary
    assert summary["written"] is True
    assert summary["coverage_input_count"] == 1
    assert summary["warning_count"] == 0
    assert summary["warnings"] == []
    assert summary["measured_kinds"] == ["tool"]
    assert summary["covered_count"] == 1
    assert summary["uncovered_count"] == 2
    assert summary["uncovered"] == ["account_recovery", "account_recovery_unverified_identity"]
    assert report["measured_kinds"] == ["tool"]
    assert report["warnings"] == []
    assert report["audit"]["item_counts"] == {"capability": 1, "failure_case": 1, "tool": 1}
    assert report["coverage"]["overall"] == {"item_count": 3, "covered_count": 1, "uncovered_count": 2}
    assert report["coverage"]["by_kind"] == {
        "capability": {"item_count": 1, "covered_count": 0, "uncovered_count": 1},
        "failure_case": {"item_count": 1, "covered_count": 0, "uncovered_count": 1},
        "tool": {"item_count": 1, "covered_count": 1, "uncovered_count": 0},
    }
    assert report["covered"] == ["customer.lookup"]
    assert report["uncovered"] == ["account_recovery", "account_recovery_unverified_identity"]
    assert report["input_reports"] == [
        {
            "path": str(coverage_path),
            "method": "tool_calls",
            "item_kind": "tool",
            "item_kind_count": 1,
            "subject": {
                "trace": str(coverage_path.parent / "trajectory.json"),
                "trace_format": "atif",
                "task_id": "account-recovery",
                "run_id": "trial-001",
            },
            "covered": ["customer.lookup"],
            "covered_count": 1,
        }
    ]
    assert gaps_by_name["account_recovery"]["reason"] == "not_measured_by_any_method"
    assert gaps_by_name["account_recovery"]["generation"]["needed_tools"] == ["customer.lookup"]
    assert "Exercise capability account_recovery" in gaps_by_name["account_recovery"]["generation"]["focus"]
    assert gaps_by_name["account_recovery"]["audit_item"]["kind"] == "capability"
    assert gaps_by_name["account_recovery_unverified_identity"]["reason"] == "not_measured_by_any_method"
    assert gaps_by_name["account_recovery_unverified_identity"]["generation"]["needed_tools"] == ["customer.lookup"]
    assert (
        "identity is not verified" in gaps_by_name["account_recovery_unverified_identity"]["audit_item"]["description"]
    )


def test_audit_report_aggregates_capability_coverage(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    tool_coverage_path = _measurement_dir(coverage_dir, "account-recovery", "trial-001") / "coverage.json"
    capability_coverage_path = (
        _measurement_dir(coverage_dir, "account-recovery", "trial-001", method="capabilities") / "coverage.json"
    )
    _write_coverage(tool_coverage_path, audit=audit, covered=["customer.lookup"])
    _write_coverage(
        capability_coverage_path,
        audit=audit,
        item_kind="capability",
        method="capabilities",
        covered=["account_recovery"],
    )
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    gaps_by_name = {item["name"]: item for item in report["uncovered_items"]}

    assert code == 0, stderr or summary
    assert summary["measured_kinds"] == ["capability", "tool"]
    assert summary["covered_count"] == 2
    assert summary["uncovered"] == ["account_recovery_unverified_identity"]
    assert report["covered"] == ["customer.lookup", "account_recovery"]
    assert report["coverage"]["by_kind"]["capability"] == {
        "item_count": 1,
        "covered_count": 1,
        "uncovered_count": 0,
    }
    assert "account_recovery" not in gaps_by_name
    assert gaps_by_name["account_recovery_unverified_identity"]["reason"] == "not_measured_by_any_method"


def test_audit_report_aggregates_failure_case_coverage(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    failure_coverage_path = (
        _measurement_dir(
            coverage_dir,
            "account-recovery-unverified-identity",
            "trial-001",
            method="failure_cases",
        )
        / "coverage.json"
    )
    _write_coverage(
        failure_coverage_path,
        audit=audit,
        item_kind="failure_case",
        method="failure_cases",
        covered=["account_recovery_unverified_identity"],
    )
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))

    assert code == 0, stderr or summary
    assert summary["measured_kinds"] == ["failure_case"]
    assert summary["covered_count"] == 1
    assert report["covered"] == ["account_recovery_unverified_identity"]
    assert report["coverage"]["by_kind"]["failure_case"] == {
        "item_count": 1,
        "covered_count": 1,
        "uncovered_count": 0,
    }


def test_audit_report_dedupes_generation_needed_tools(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    required_tools:\n      - customer.lookup\n",
            "    required_tools:\n      - customer.lookup\n      - customer.lookup\n",
            1,
        ),
    )
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    _write_coverage(coverage_dir / "coverage.json", audit=audit, covered=["customer.lookup"])
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    gaps_by_name = {item["name"]: item for item in report["uncovered_items"]}

    assert code == 0, stderr or summary
    assert gaps_by_name["account_recovery"]["generation"]["needed_tools"] == ["customer.lookup"]


def test_audit_report_failure_case_empty_expected_tools_suppresses_fallback(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace(
            "    expected_tools:\n      - customer.lookup\n    prohibited_tools:\n      - password.reset\n",
            "    expected_tools: []\n    prohibited_tools:\n      - password.reset\n",
            1,
        ),
    )
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    _write_coverage(coverage_dir / "coverage.json", audit=audit, covered=["customer.lookup"])
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    gaps_by_name = {item["name"]: item for item in report["uncovered_items"]}

    assert code == 0, stderr or summary
    assert gaps_by_name["account_recovery_unverified_identity"]["generation"]["needed_tools"] == []


def test_audit_report_failure_case_needed_tools_prefer_expected_and_filter_prohibited(tmp_path: Path) -> None:
    ticket_yaml = yaml.safe_dump([_ticket_tool_item()], sort_keys=False)
    ticket_block = "\n".join(f"  {line}" for line in ticket_yaml.splitlines())
    audit = _write_audit(
        tmp_path,
        lambda text: text.replace("\n  - kind: capability\n", f"\n{ticket_block}\n\n  - kind: capability\n", 1).replace(
            "    expected_tools:\n      - customer.lookup\n    prohibited_tools:\n      - password.reset\n",
            "    expected_tools:\n      - ticket.create\n    prohibited_tools:\n      - ticket.create\n",
            1,
        ),
    )
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    _write_coverage(coverage_dir / "coverage.json", audit=audit, covered=["customer.lookup"])
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, summary, stderr = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    gaps_by_name = {item["name"]: item for item in report["uncovered_items"]}

    assert code == 0, stderr or summary
    assert gaps_by_name["account_recovery_unverified_identity"]["generation"]["needed_tools"] == []
    assert ".:" not in gaps_by_name["account_recovery_unverified_identity"]["generation"]["focus"]


def test_audit_report_rejects_stale_coverage_denominator_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_path = tmp_path / ".eval-author" / "audit-measurements" / "coverage.json"
    coverage = _write_coverage(coverage_path, audit=audit, covered=["customer.lookup"])
    coverage["audit"]["item_count"] = 99
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, report, _ = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage",
        str(coverage_path),
        "--out",
        str(out),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "coverage_input"
    assert "audit.item_count 99 does not match current audit 3" in report["error"]
    assert not out.exists()


def test_audit_report_warns_on_mismatched_audit_status(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_path = tmp_path / ".eval-author" / "audit-measurements" / "coverage.json"
    _write_coverage(coverage_path, audit=audit, covered=["customer.lookup"])
    audit.write_text(audit.read_text(encoding="utf-8").replace("status: draft\n", "status: approved\n", 1))
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, report, _ = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage",
        str(coverage_path),
        "--out",
        str(out),
    )

    aggregate = json.loads(out.read_text(encoding="utf-8"))
    expected_warning = {
        "kind": "audit_status_mismatch",
        "coverage": str(coverage_path),
        "input_status": "draft",
        "current_status": "approved",
    }

    assert code == 0
    assert report["valid"] is True
    assert report["written"] is True
    assert report["warning_count"] == 1
    assert report["warnings"] == [expected_warning]
    assert aggregate["warnings"] == [expected_warning]


def test_audit_report_rejects_unknown_covered_item_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_path = tmp_path / ".eval-author" / "audit-measurements" / "coverage.json"
    _write_coverage(coverage_path, audit=audit, covered=["admin.reset_password"])
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, report, _ = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage",
        str(coverage_path),
        "--out",
        str(out),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "coverage_input"
    assert "covered item 'admin.reset_password' is not in the current audit" in report["error"]
    assert not out.exists()


def test_audit_report_rejects_empty_coverage_set_without_writing(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    coverage_dir = tmp_path / ".eval-author" / "audit-measurements"
    coverage_dir.mkdir(parents=True)
    out = tmp_path / ".eval-author" / "audit-coverage-report.json"

    code, report, _ = _run_json_script(
        _AUDIT_REPORT,
        "--audit",
        str(audit),
        "--coverage-dir",
        str(coverage_dir),
        "--out",
        str(out),
    )

    assert code == 1
    assert report["valid"] is True
    assert report["written"] is False
    assert report["error_type"] == "coverage_input"
    assert "no coverage.json files found" in report["error"]
    assert not out.exists()


def test_bundled_scripts_never_import_the_platform() -> None:
    """The boundary that makes the skill copyable: Harbor is fine, NeMo is not."""
    offenders: dict[str, set[str]] = {}
    for scripts_dir in _SCRIPT_DIRS:
        permitted = _local_roots(scripts_dir) | sys.stdlib_module_names | _PERMITTED_THIRD_PARTY
        for path in _bundled_scripts(scripts_dir):
            found = {root for root in _imported_roots(path) if root not in permitted}
            if found:
                offenders[path.relative_to(scripts_dir).as_posix()] = found
    assert not offenders, (
        "Bundled scripts may import the standard library, a sibling, or "
        f"{sorted(_PERMITTED_THIRD_PARTY)}, found: "
        + "; ".join(f"{filename} imports {sorted(names)}" for filename, names in sorted(offenders.items()))
    )


def test_no_bundled_directory_is_named_after_a_provider_package() -> None:
    """The trap that makes a literal ``scripts/harbor/`` unusable.

    A directory on the path named ``harbor`` is importable as a namespace package.
    On a machine with no Harbor installed that makes ``find_spec("harbor")``
    succeed, so the probe reports an install that is not there and the ladder then
    fails on import. Provider code sits one level down for exactly this reason.
    """
    for scripts_dir in _SCRIPT_DIRS:
        collisions = _local_roots(scripts_dir) & _PERMITTED_THIRD_PARTY
        assert not collisions, (
            f"{sorted(collisions)} shadows a package the skill imports; "
            "keep it under scripts/providers/ rather than directly in scripts/"
        )


def test_only_the_ladder_imports_harbor() -> None:
    """Every other discover module must keep working when Harbor is absent."""
    assert "harbor" in _imported_roots(_LADDER), "the ladder is the Harbor boundary and must import Harbor"
    for path in _bundled_scripts(_DISCOVER_SCRIPTS_DIR):
        if path == _LADDER:
            continue
        assert "harbor" not in _imported_roots(path), (
            f"{path.relative_to(_DISCOVER_SCRIPTS_DIR)} imports Harbor; move that call behind the probe"
        )


def test_discover_defers_the_ladder_import_to_call_time() -> None:
    """A module-scope ladder import would break every Harbor-free repository."""
    tree = ast.parse(_DISCOVER.read_text(encoding="utf-8"), filename=str(_DISCOVER))
    module_scope: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_scope.update(alias.name for alias in node.names)
            if node.module:
                module_scope.add(node.module)
    named = sorted(name for name in module_scope if "_ladder" in name)
    assert not named, f"discover.py imports {named} at module scope; move it inside a function, after the probe"


@_needs_harbor
def test_discover_proves_a_valid_suite_runnable(suite: Path) -> None:
    code, report = _run_discover(suite)

    assert report["proven"] is True, f"Harbor must judge this report; runtime was {report['runtime']}"
    assert report["task_count"] == 1
    assert report["dataset_paths"] == ["dataset"]
    assert len(report["configs"]) == 1

    for name in ("schema", "resolution", "tasks", "coverage", "credentials", "agent"):
        check = _named(report, name)
        assert check["status"] == "pass", f"{name} failed: {check['message']}"
        assert check["proven"] is True

    backend = _named(report, "backend")
    if backend["status"] != "pass":
        pytest.skip(f"no environment backend available: {backend['message']}")
    assert code == 0
    assert report["runnable"] is True
    assert report["run_command"] == f"cd {suite} && harbor job start -c harbor-job.yaml"


@_needs_harbor
def test_discover_names_the_rung_a_broken_config_fails(suite: Path) -> None:
    """A dataset path that does not exist must fail resolution, not schema."""
    (suite / "harbor-job.yaml").write_text(
        "job_name: fixture\ndatasets:\n  - path: ./no-such-dataset\nagents:\n  - name: oracle\n",
        encoding="utf-8",
    )

    code, report = _run_discover(suite)

    assert code == 1
    assert report["runnable"] is False
    assert _named(report, "schema")["status"] == "pass", "the shape is valid; only the path is wrong"
    resolution = _named(report, "resolution")
    assert resolution["status"] == "fail"
    assert "no-such-dataset" in resolution["message"]


@_needs_harbor
def test_discover_names_an_unknown_agent(suite: Path) -> None:
    (suite / "harbor-job.yaml").write_text(
        "job_name: fixture\ndatasets:\n  - path: ./dataset\nagents:\n  - name: no-such-agent\n",
        encoding="utf-8",
    )

    code, report = _run_discover(suite)

    assert code == 1
    agent = _named(report, "agent")
    assert agent["status"] == "fail"
    assert "no-such-agent" in agent["message"]


@_needs_harbor
def test_discover_reports_required_host_variables(suite: Path) -> None:
    task_toml = suite / "dataset" / "task-one" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8") + '\n[environment.env]\nACME_API_KEY = "${ACME_API_KEY}"\n',
        encoding="utf-8",
    )

    _, report = _run_discover(suite)

    credentials = _named(report, "credentials")
    assert credentials["status"] == "pass"
    assert "ACME_API_KEY" in credentials["message"]
    assert [item["name"] for item in report["configs"][0]["required_env_vars"]] == ["ACME_API_KEY"]


@_needs_harbor
def test_discover_reports_a_task_harbor_silently_dropped(suite: Path) -> None:
    """Harbor skips an unparseable task without raising, which coverage must catch."""
    broken = suite / "dataset" / "task-two"
    _write_task(broken)
    (broken / "task.toml").write_text("this is not valid toml = = =\n", encoding="utf-8")

    code, report = _run_discover(suite)

    assert code == 1
    coverage = _named(report, "coverage")
    assert coverage["status"] == "fail"
    assert "task-two" in coverage["message"]


def test_discover_marks_every_finding_unproven_without_harbor(suite: Path) -> None:
    """The promise that keeps an inventory from reading as evidence."""
    code, report = _run_discover(suite, with_harbor=False)

    assert code == 1, "no Harbor means no proof, so discovery cannot report success"
    assert report["proven"] is False
    assert report["runnable"] is False
    assert report["run_command"] is None
    assert report["runtime"]["harbor_importable"] is False

    harbor_check = _named(report, "harbor")
    assert harbor_check["status"] == "fail"
    assert harbor_check["severity"] == "required"
    assert harbor_check["hint"], "a missing Harbor must tell the user what to do"

    observed = [check for check in report["checks"] if check["name"] != "harbor"]
    assert observed, "an unproven inventory is still worth reporting"
    assert all(check["proven"] is False for check in observed), (
        f"these findings claim proof without Harbor: {[c['name'] for c in observed if c['proven']]}"
    )


def test_discover_finds_configs_without_pyyaml(suite: Path) -> None:
    """Without PyYAML the fallback still finds YAML configs, and says it cannot read them."""
    _, report = _run_discover(suite, with_harbor=False)

    assert [config["path"] for config in report["configs"]] == ["harbor-job.yaml"]
    parse = _named(report, "config-parse")
    assert parse["status"] == "fail"
    assert "harbor-job.yaml" in parse["message"]


def test_discovery_writes_no_files(suite: Path) -> None:
    """The scripts report and the agent saves, which is what makes the skill safe to run."""
    before = _tree_state(suite)

    _run_discover(suite)

    assert _tree_state(suite) == before


def test_discover_reports_a_config_pyyaml_cannot_parse(suite: Path) -> None:
    """Broken YAML is a finding, not a crash.

    A tab cannot start a YAML token, so PyYAML raises ``yaml.YAMLError``, which is
    not a ``ValueError``. Left uncaught it escaped the scan and took the whole run
    down with a traceback and no report at all. The file still names Harbor work,
    so it is reported through the same fallback that runs when PyYAML is absent.
    """
    (suite / "broken.yaml").write_text("datasets:\n\t- path: ./tabbed\n", encoding="utf-8")

    _, report = _run_discover(suite)

    assert "harbor-job.yaml" in [config["path"] for config in report["configs"]], (
        "one broken file must not hide the configs around it"
    )
    parse = _named(report, "config-parse")
    assert parse["status"] == "fail"
    assert "broken.yaml" in parse["message"]
    assert parse["hint"], "an unreadable config must tell the user what to do"


def test_discover_ignores_a_yaml_file_that_declares_no_harbor_work(suite: Path) -> None:
    """Broken YAML that names no datasets or tasks is somebody else's file."""
    (suite / "docker-compose.yaml").write_text("services:\n\t- broken\n", encoding="utf-8")

    _, report = _run_discover(suite)

    assert [config["path"] for config in report["configs"]] == ["harbor-job.yaml"]


@_needs_unreadable_files
def test_discover_reports_an_unreadable_ethos(suite: Path) -> None:
    """ETHOS.md is advisory, so failing to read it must not cost the whole report."""
    ethos = suite / "ETHOS.md"
    ethos.write_text("# doctrine\n", encoding="utf-8")
    ethos.chmod(0o000)
    try:
        _, report = _run_discover(suite, with_harbor=False)
    finally:
        ethos.chmod(0o644)

    assert report["ethos_path"] is None, "an unread file defines no doctrine"
    check = _named(report, "ethos")
    assert check["status"] == "warn"
    assert check["severity"] == "advisory"


@_needs_unreadable_files
def test_discover_keeps_an_unreadable_dataset_file_out_of_the_fingerprint(suite: Path) -> None:
    """One unreadable file must neither abort the fingerprint nor silently join it."""
    _, baseline = _run_discover(suite, with_harbor=False)

    blocked = suite / "dataset" / "task-one" / "blocked.bin"
    blocked.write_bytes(b"payload")
    blocked.chmod(0o000)
    try:
        _, report = _run_discover(suite, with_harbor=False)
    finally:
        blocked.chmod(0o644)

    assert report["input_file_count"] == baseline["input_file_count"]
    assert report["fingerprint"] == baseline["fingerprint"]


def test_discover_fails_with_a_hint_when_the_path_is_missing(tmp_path: Path) -> None:
    code, report = _run_discover(tmp_path / "nowhere")
    assert code == 1
    assert "error" in report
    assert report["hint"]
