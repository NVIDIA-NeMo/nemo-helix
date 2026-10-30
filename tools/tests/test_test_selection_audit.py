# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools.ci.test_selection_audit import (
    AuditFinding,
    AuditReport,
    _python_files,
    audit_repository,
    parse_change_action,
    parse_workflow_jobs,
    render_json,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_files_excludes_local_environments(tmp_path: Path) -> None:
    source = tmp_path / "test_example.py"
    source.write_text("import example\n", encoding="utf-8")
    for directory in (".venv", ".flox"):
        cached = tmp_path / "nested" / directory / "cache" / "dependency.py"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"\xff")

    assert _python_files(tmp_path) == (source,)


def test_change_action_resolves_output_aliases() -> None:
    action = parse_change_action((REPO_ROOT / ".github/actions/changes/action.yaml").read_text(encoding="utf-8"))

    assert action.output_filters["test"] == ("tests",)
    assert action.output_filters["scaled-evals"] == ("scaled-evals",)
    assert "deps" in action.output_filters["cpu-smoke"]
    assert "docker-scripts" in action.output_filters["cpu-smoke"]


def test_workflow_parser_finds_gated_test_commands() -> None:
    jobs = parse_workflow_jobs((REPO_ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))

    assert jobs["python-unit-test-scaled-evals"].referenced_outputs == ("deps", "scaled-evals")
    scaled_evals = jobs["python-unit-test-scaled-evals"]
    assert scaled_evals.run_blocks[0] == (
        "uv run --frozen --group scaled-evals pytest plugins/_temporary-scaled-evals/tests -v"
    )
    # The per-runner Harbor step runs from the nhx-sandbox package, which owns harbor_opensandbox.
    assert scaled_evals.run_block_working_directories == ("", "packages/nhx_sandbox")
    assert jobs["deployments-openshell-tests"].referenced_outputs == ("deps", "deployments-openshell")
    assert jobs["deployments-openshell-tests"].run_blocks == ("make test-deployments-openshell",)


def test_pytest_paths_resolve_against_the_step_working_directory() -> None:
    workflow = """\
jobs:
  plugin-tests:
    name: Plugin tests
    if: needs.changes.outputs.plugin == 'true'
    steps:
      - name: Run plugin tests
        working-directory: plugins/example
        run: |
          python -m pytest tests/test_one.py
      - name: Run root tests
        run: pytest tests/test_two.py
"""
    job = parse_workflow_jobs(workflow)["plugin-tests"]

    assert job.run_block_working_directories == ("plugins/example", "")


def test_a_run_line_inside_a_block_scalar_is_content_not_a_step() -> None:
    workflow = """\
jobs:
  smoke:
    name: Smoke
    if: needs.changes.outputs.plugin == 'true'
    steps:
      - name: Run smoke
        working-directory: plugins/example
        run: |
          make check
          run: smoke
"""
    job = parse_workflow_jobs(workflow)["smoke"]

    assert job.run_blocks == ("make check\nrun: smoke",)
    assert job.run_block_working_directories == ("plugins/example",)


def test_audit_report_is_clean_after_filter_updates() -> None:
    report = audit_repository(REPO_ROOT)

    assert report.findings == ()


def test_markdown_report_reads_like_agent_handoff() -> None:
    report = _agent_handoff_fixture()
    markdown = render_markdown(report)

    assert markdown.startswith("# CI test selection audit agent handoff")
    assert "Use this report as a work queue" in markdown
    assert "### Do this: run this job when root dependencies change" in markdown
    assert "### Do this: run this job when the Make target changes" not in markdown
    assert ".github/actions/changes/action.yaml" not in markdown
    assert ".github/workflows/ci.yaml" not in markdown
    assert "Instruction for the agent:" in markdown
    assert "Example broad-output gate:" in markdown
    assert "needs.changes.outputs.deps == 'true'" in markdown
    assert "needs.changes.outputs.scaled-evals == 'true'" in markdown
    assert "### Open question:" in markdown
    assert "Question for the agent:" in markdown
    assert "| Job | Confidence | Missing path |" not in markdown


def test_json_report_includes_agent_instruction_fields() -> None:
    report = _agent_handoff_fixture()
    json_report = render_json(report)

    assert '"agent_instructions"' in json_report
    assert '"agent_kind": "apply"' in json_report
    assert '"agent_kind": "investigate"' in json_report
    assert '"agent_instruction"' in json_report
    assert '"agent_example_fix"' in json_report


def _agent_handoff_fixture() -> AuditReport:
    return AuditReport(
        findings=(
            AuditFinding(
                job_id="python-unit-test-scaled-evals",
                job_name="Python unit tests (scaled-evals plugin)",
                filter_outputs=("scaled-evals",),
                filter_ids=("scaled-evals",),
                missing_path="pyproject.toml",
                reason="uv command reads root dependency declarations",
                confidence="high",
                suggestion="pyproject.toml",
            ),
            AuditFinding(
                job_id="python-unit-test-scaled-evals",
                job_name="Python unit tests (scaled-evals plugin)",
                filter_outputs=("scaled-evals",),
                filter_ids=("scaled-evals",),
                missing_path="packages/nemo_helix_plugin/src/nemo_helix_plugin/authz.py",
                reason="`plugins/_temporary-scaled-evals/tests` imports `nemo_helix_plugin.authz`",
                confidence="medium",
                suggestion="packages/nemo_helix_plugin/src/nemo_helix_plugin/authz.py",
            ),
        ),
        audited_jobs=("python-unit-test-scaled-evals",),
        skipped_jobs=(),
    )
