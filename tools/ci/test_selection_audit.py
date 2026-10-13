#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report dependency-sensitive CI jobs whose path filters may be too narrow."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import shlex
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Confidence = Literal["high", "medium", "low"]
InstructionKind = Literal["apply", "investigate"]

_OUTPUT_REF_RE = re.compile(r"(?:needs\.changes|steps\.filter)\.outputs\.([A-Za-z0-9_-]+)")
_WORKFLOW_OUTPUT_REF_RE = re.compile(r"needs\.changes\.outputs\.([A-Za-z0-9_-]+)")
_JOB_START_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$")
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_VERSION_OPERATOR_RE = re.compile(r"\s*(?:===|==|~=|!=|<=|>=|<|>|=)")
_CONFIDENCE_RANK: Mapping[Confidence, int] = {"low": 0, "medium": 1, "high": 2}
_REPORT_HEADER_TEMPLATE = """# CI test selection audit agent handoff

Use this report as a work queue for improving path-gated CI jobs. Apply high-confidence instructions first; they are likely false-negative risks. Treat medium-confidence entries as open questions: investigate the test purpose, job cost, and overlapping coverage before changing filters.

Audited jobs: {audited_jobs}
Skipped gated non-test jobs: {skipped_jobs}
"""
_JOB_SECTION_TEMPLATE = """## {job_id}

Job name: {job_name}
Current gate output(s): {filter_outputs}
Current filter id(s): {filter_ids}
"""
_HIGH_CONFIDENCE_TEMPLATE = """### Do this: {title}

Confidence: high

Instruction for the agent:
{instruction}

Evidence:
{evidence}
{example_fix}
"""
_MEDIUM_CONFIDENCE_TEMPLATE = """### Open question: {title}

Confidence: medium

Question for the agent:
{instruction}

Evidence:
{evidence}
{example_fix}
"""
_LOW_CONFIDENCE_TEMPLATE = """### Investigate carefully: {title}

Confidence: low

Question for the agent:
{instruction}

Evidence:
{evidence}
{example_fix}
"""
_BROAD_OUTPUT_EXAMPLE_TEMPLATE = """Example broad-output gate:

```yaml
if: >
  !cancelled() &&
  (needs.changes.outputs.deps == 'true' || needs.changes.outputs.{primary_output} == 'true')
```

Prefer the existing `deps` output for root dependency files instead of duplicating `pyproject.toml` and `uv.lock` into every narrow filter.
"""


@dataclass(frozen=True)
class ChangeFilter:
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class ChangeAction:
    filters: Mapping[str, ChangeFilter]
    output_filters: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class WorkflowJob:
    job_id: str
    name: str
    referenced_outputs: tuple[str, ...]
    run_blocks: tuple[str, ...]
    raw_block: str


@dataclass(frozen=True)
class WorkspacePackage:
    name: str
    root: Path
    pyproject: Path
    import_roots: tuple[str, ...]


@dataclass(frozen=True)
class RepoModel:
    root: Path
    packages_by_name: Mapping[str, WorkspacePackage]
    packages_by_import: Mapping[str, tuple[WorkspacePackage, ...]]
    dependency_groups: Mapping[str, tuple[str, ...]]
    make_targets: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class InferredPath:
    path: str
    reason: str
    confidence: Confidence


@dataclass(frozen=True)
class AuditFinding:
    job_id: str
    job_name: str
    filter_outputs: tuple[str, ...]
    filter_ids: tuple[str, ...]
    missing_path: str
    reason: str
    confidence: Confidence
    suggestion: str


@dataclass(frozen=True)
class AgentInstruction:
    finding: AuditFinding
    kind: InstructionKind
    title: str
    instruction: str
    evidence: tuple[str, ...]
    example_fix: str


@dataclass(frozen=True)
class AuditReport:
    findings: tuple[AuditFinding, ...]
    audited_jobs: tuple[str, ...]
    skipped_jobs: tuple[str, ...]


def parse_change_action(text: str) -> ChangeAction:
    return ChangeAction(filters=parse_filters(text), output_filters=parse_output_filters(text))


def parse_filters(text: str) -> Mapping[str, ChangeFilter]:
    lines = text.splitlines()
    filters_start = -1
    filters_indent = 0
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)filters:\s*\|\s*(?:#.*)?$", line)
        if match is not None:
            filters_start = index + 1
            filters_indent = len(match.group(1))
            break

    if filters_start < 0:
        return {}

    current_name = ""
    current_patterns: list[str] = []
    parsed: dict[str, ChangeFilter] = {}
    filter_indent = -1

    for line in lines[filters_start:]:
        if not line.strip():
            continue
        indent = _leading_spaces(line)
        if indent <= filters_indent:
            break
        stripped = line.strip()
        if filter_indent < 0 and stripped.endswith(":") and not stripped.startswith("-"):
            filter_indent = indent
        if indent == filter_indent and stripped.endswith(":") and not stripped.startswith("-"):
            if current_name:
                parsed[current_name] = ChangeFilter(name=current_name, patterns=tuple(current_patterns))
            current_name = stripped[:-1].strip()
            current_patterns = []
            continue
        if current_name and stripped.startswith("-"):
            pattern = _strip_yaml_scalar(stripped[1:].strip())
            if pattern:
                current_patterns.append(pattern)

    if current_name:
        parsed[current_name] = ChangeFilter(name=current_name, patterns=tuple(current_patterns))
    return parsed


def parse_output_filters(text: str) -> Mapping[str, tuple[str, ...]]:
    lines = text.splitlines()
    in_outputs = False
    current_output = ""
    output_indent = -1
    refs_by_output: dict[str, list[str]] = {}

    for line in lines:
        if not in_outputs:
            if re.match(r"^outputs:\s*(?:#.*)?$", line):
                in_outputs = True
            continue

        if line and _leading_spaces(line) == 0 and not line.startswith(" "):
            break
        if not line.strip():
            continue

        indent = _leading_spaces(line)
        stripped = line.strip()
        if output_indent < 0 and stripped.endswith(":") and not stripped.startswith("-"):
            output_indent = indent
        if indent == output_indent and stripped.endswith(":") and not stripped.startswith("-"):
            current_output = stripped[:-1].strip()
            refs_by_output.setdefault(current_output, [])
            continue
        if current_output and "value:" in stripped:
            refs_by_output[current_output].extend(_unique(_OUTPUT_REF_RE.findall(stripped)))

    return {name: tuple(_unique(refs)) for name, refs in refs_by_output.items() if refs}


def parse_workflow_jobs(text: str) -> Mapping[str, WorkflowJob]:
    lines = text.splitlines()
    jobs_start = -1
    for index, line in enumerate(lines):
        if line == "jobs:":
            jobs_start = index + 1
            break
    if jobs_start < 0:
        return {}

    starts: list[tuple[str, int]] = []
    for index in range(jobs_start, len(lines)):
        match = _JOB_START_RE.match(lines[index])
        if match is not None:
            starts.append((match.group(1), index))

    parsed: dict[str, WorkflowJob] = {}
    for start_index, (job_id, line_index) in enumerate(starts):
        next_index = starts[start_index + 1][1] if start_index + 1 < len(starts) else len(lines)
        block_lines = lines[line_index:next_index]
        raw_block = "\n".join(block_lines)
        parsed[job_id] = WorkflowJob(
            job_id=job_id,
            name=_parse_job_name(block_lines),
            referenced_outputs=tuple(_unique(_WORKFLOW_OUTPUT_REF_RE.findall(raw_block))),
            run_blocks=tuple(_parse_run_blocks(block_lines)),
            raw_block=raw_block,
        )
    return parsed


def load_repo_model(repo_root: Path) -> RepoModel:
    root_pyproject = _load_toml(repo_root / "pyproject.toml")
    packages = _load_workspace_packages(repo_root, root_pyproject)
    return RepoModel(
        root=repo_root,
        packages_by_name={_normalize_package_name(package.name): package for package in packages},
        packages_by_import=_packages_by_import_root(packages),
        dependency_groups=_load_dependency_groups(root_pyproject),
        make_targets=parse_make_targets((repo_root / "Makefile").read_text(encoding="utf-8")),
    )


def parse_make_targets(text: str) -> Mapping[str, tuple[str, ...]]:
    lines = text.splitlines()
    targets: dict[str, tuple[str, ...]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^([A-Za-z0-9_./-]+)\s*:[^=]?.*$", line)
        if match is None or match.group(1) == ".PHONY":
            index += 1
            continue
        target_name = match.group(1)
        recipe: list[str] = []
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if next_line.startswith("\t") or (next_line.startswith(" ") and next_line.strip()):
                recipe.append(_normalize_recipe_line(next_line))
                index += 1
                continue
            if not next_line.strip():
                index += 1
                continue
            break
        if recipe:
            targets[target_name] = tuple(recipe)
    return targets


def audit_repository(repo_root: Path, include_all_gated_jobs: bool = False) -> AuditReport:
    action = parse_change_action((repo_root / ".github/actions/changes/action.yaml").read_text(encoding="utf-8"))
    jobs = parse_workflow_jobs((repo_root / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))
    repo = load_repo_model(repo_root)

    findings: list[AuditFinding] = []
    audited_jobs: list[str] = []
    skipped_jobs: list[str] = []

    for job in jobs.values():
        if not job.referenced_outputs:
            continue
        if not include_all_gated_jobs and not _is_test_like_job(job):
            skipped_jobs.append(job.job_id)
            continue
        filter_ids = _resolve_filter_ids(job.referenced_outputs, action)
        if not filter_ids:
            skipped_jobs.append(job.job_id)
            continue
        audited_jobs.append(job.job_id)
        inferred_paths = infer_job_paths(job, repo)
        selected_filters = tuple(action.filters[filter_id] for filter_id in filter_ids if filter_id in action.filters)
        for inferred in inferred_paths:
            if _path_is_covered(inferred.path, selected_filters):
                continue
            findings.append(
                AuditFinding(
                    job_id=job.job_id,
                    job_name=job.name,
                    filter_outputs=job.referenced_outputs,
                    filter_ids=filter_ids,
                    missing_path=inferred.path,
                    reason=inferred.reason,
                    confidence=inferred.confidence,
                    suggestion=_suggest_pattern(inferred.path),
                )
            )

    return AuditReport(
        findings=tuple(sorted(findings, key=lambda item: (item.job_id, item.missing_path, item.reason))),
        audited_jobs=tuple(audited_jobs),
        skipped_jobs=tuple(skipped_jobs),
    )


def infer_job_paths(job: WorkflowJob, repo: RepoModel) -> tuple[InferredPath, ...]:
    collector = _PathCollector()

    for run_block in job.run_blocks:
        _infer_run_block(run_block, repo, collector, active_make_targets=())

    test_paths = tuple(path for path in collector.paths if _path_points_to_python_tests(repo.root, path))
    for test_path in test_paths:
        _infer_python_import_closure(repo, test_path, collector)

    return collector.as_tuple()


def render_markdown(report: AuditReport) -> str:
    lines = [
        _REPORT_HEADER_TEMPLATE.format(
            audited_jobs=len(report.audited_jobs),
            skipped_jobs=len(report.skipped_jobs),
        ).rstrip()
    ]
    if not report.findings:
        lines.append("\nNo missing path-filter coverage was found.")
        return "\n".join(lines)

    for job_id, instructions in _agent_instructions_by_job(report):
        first = instructions[0].finding
        lines.append("")
        lines.append(
            _JOB_SECTION_TEMPLATE.format(
                job_id=job_id,
                job_name=first.job_name or "(unnamed)",
                filter_outputs=", ".join(f"`{output}`" for output in first.filter_outputs),
                filter_ids=", ".join(f"`{filter_id}`" for filter_id in first.filter_ids),
            ).rstrip()
        )
        for instruction in instructions:
            lines.append("")
            lines.append(_render_instruction(instruction).rstrip())
    return "\n".join(lines)


def render_json(report: AuditReport) -> str:
    instructions = [instruction for _, items in _agent_instructions_by_job(report) for instruction in items]
    instructions_by_finding = {instruction.finding: instruction for instruction in instructions}
    payload: Mapping[str, object] = {
        "audited_jobs": report.audited_jobs,
        "skipped_jobs": report.skipped_jobs,
        "findings": [
            {
                "job_id": finding.job_id,
                "job_name": finding.job_name,
                "filter_outputs": finding.filter_outputs,
                "filter_ids": finding.filter_ids,
                "missing_path": finding.missing_path,
                "reason": finding.reason,
                "confidence": finding.confidence,
                "suggestion": finding.suggestion,
                "agent_kind": instructions_by_finding[finding].kind,
                "agent_title": instructions_by_finding[finding].title,
                "agent_instruction": instructions_by_finding[finding].instruction,
                "agent_evidence": instructions_by_finding[finding].evidence,
                "agent_example_fix": instructions_by_finding[finding].example_fix,
            }
            for finding in report.findings
        ],
        "agent_instructions": [
            {
                "job_id": instruction.finding.job_id,
                "job_name": instruction.finding.job_name,
                "confidence": instruction.finding.confidence,
                "kind": instruction.kind,
                "title": instruction.title,
                "instruction": instruction.instruction,
                "evidence": instruction.evidence,
                "example_fix": instruction.example_fix,
                "missing_path": instruction.finding.missing_path,
                "suggestion": instruction.finding.suggestion,
            }
            for instruction in instructions
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _agent_instructions_by_job(report: AuditReport) -> tuple[tuple[str, tuple[AgentInstruction, ...]], ...]:
    instructions_by_job: dict[str, list[AgentInstruction]] = {}
    for finding in report.findings:
        instructions_by_job.setdefault(finding.job_id, []).append(_agent_instruction_for_finding(finding))

    grouped: list[tuple[str, tuple[AgentInstruction, ...]]] = []
    for job_id, instructions in instructions_by_job.items():
        grouped.append((job_id, tuple(sorted(instructions, key=_instruction_sort_key))))
    return tuple(grouped)


def _instruction_sort_key(instruction: AgentInstruction) -> tuple[int, str]:
    confidence_order: Mapping[Confidence, int] = {"high": 0, "medium": 1, "low": 2}
    return (confidence_order[instruction.finding.confidence], instruction.finding.missing_path)


def _agent_instruction_for_finding(finding: AuditFinding) -> AgentInstruction:
    evidence = (
        f"Missing path: `{finding.missing_path}`",
        f"Why it matters: {finding.reason}",
        f"Current filter id(s): {', '.join(f'`{filter_id}`' for filter_id in finding.filter_ids)}",
        f"Suggested path pattern: `{finding.suggestion}`",
    )
    if finding.confidence == "high":
        return AgentInstruction(
            finding=finding,
            kind="apply",
            title=_high_confidence_title(finding),
            instruction=_high_confidence_instruction(finding),
            evidence=evidence,
            example_fix=_example_fix(finding),
        )
    return AgentInstruction(
        finding=finding,
        kind="investigate",
        title=_medium_confidence_title(finding),
        instruction=_medium_confidence_question(finding),
        evidence=evidence,
        example_fix=_example_fix(finding),
    )


def _render_instruction(instruction: AgentInstruction) -> str:
    finding = instruction.finding
    template = _HIGH_CONFIDENCE_TEMPLATE
    if finding.confidence == "medium":
        template = _MEDIUM_CONFIDENCE_TEMPLATE
    elif finding.confidence == "low":
        template = _LOW_CONFIDENCE_TEMPLATE
    return template.format(
        title=instruction.title,
        instruction=_indent_block(instruction.instruction),
        evidence=_bullet_list(instruction.evidence),
        example_fix=instruction.example_fix,
    )


def _high_confidence_title(finding: AuditFinding) -> str:
    if finding.missing_path in {"pyproject.toml", "uv.lock"}:
        return "run this job when root dependencies change"
    if finding.missing_path == "Makefile":
        return "run this job when the Make target changes"
    if "/tests/" in finding.missing_path or finding.missing_path.endswith("_test.py"):
        return "run this job when its explicit test file changes"
    if finding.missing_path.endswith("pyproject.toml"):
        return "run this job when its workspace package metadata changes"
    return "add missing high-confidence path coverage"


def _high_confidence_instruction(finding: AuditFinding) -> str:
    primary_output = _primary_filter_output(finding)
    if finding.missing_path in {"pyproject.toml", "uv.lock"}:
        return (
            f"Update `{finding.job_id}` so it also runs when dependency files change. Prefer broad output `deps` "
            f"alongside `{primary_output}` instead of adding `{finding.missing_path}` directly to the narrow "
            "filter, unless that job intentionally must ignore global dependency changes."
        )
    if finding.missing_path == "Makefile":
        return (
            f"Update filter `{_primary_filter_id(finding)}` so changes to `Makefile` trigger `{finding.job_id}`. "
            "This job delegates to a Make target, so Makefile edits can change the command without touching the "
            "plugin or service paths."
        )
    return (
        f"Update filter `{_primary_filter_id(finding)}` so `{finding.job_id}` runs when "
        f"`{finding.suggestion}` changes. This is high confidence because the job directly consumes that path."
    )


def _medium_confidence_title(finding: AuditFinding) -> str:
    if finding.missing_path.startswith(".github/"):
        return "decide whether change-filter edits should trigger this job"
    return "decide whether this imported path should trigger this job"


def _medium_confidence_question(finding: AuditFinding) -> str:
    if finding.missing_path.startswith(".github/"):
        return (
            f"Should `{finding.job_id}` run when `{finding.missing_path}` changes? If edits to the change-filter "
            "definition can accidentally stop this narrow job from running, add self-coverage to the filter. If another "
            "required validation already catches filter-definition edits, document that and leave this unchanged."
        )
    return (
        f"Should changes to `{finding.missing_path}` invalidate `{finding.job_id}`? Check whether the normal unit "
        "suite already covers this dependency under the same installed extras/environment. If not, broaden the "
        f"`{_primary_filter_id(finding)}` filter or use a package-level pattern that covers this import closure."
    )


def _example_fix(finding: AuditFinding) -> str:
    if finding.missing_path in {"pyproject.toml", "uv.lock"} and "deps" not in finding.filter_outputs:
        return "\n" + _BROAD_OUTPUT_EXAMPLE_TEMPLATE.format(primary_output=_primary_filter_output(finding))
    if finding.confidence == "high":
        return (
            f"\nExample filter addition:\n\n```yaml\n{_primary_filter_id(finding)}:\n  - '{finding.suggestion}'\n```\n"
        )
    if _should_show_package_broadening_example(finding):
        package_root = _package_root_pattern(finding.missing_path)
        if package_root:
            return (
                "\nExample broader filter option, if the investigation confirms this job owns that closure:\n\n"
                "```yaml\n"
                f"{_primary_filter_id(finding)}:\n"
                f"  - '{package_root}'\n"
                "```\n"
            )
    return ""


def _primary_filter_output(finding: AuditFinding) -> str:
    for output in finding.filter_outputs:
        if output != "deps":
            return output
    return finding.filter_outputs[0] if finding.filter_outputs else "CHANGE_OUTPUT"


def _primary_filter_id(finding: AuditFinding) -> str:
    for filter_id in finding.filter_ids:
        if filter_id != "deps":
            return filter_id
    return finding.filter_ids[0] if finding.filter_ids else "FILTER_ID"


def _should_show_package_broadening_example(finding: AuditFinding) -> bool:
    return finding.confidence == "medium" and finding.missing_path.startswith(("packages/", "plugins/", "services/"))


def _package_root_pattern(path: str) -> str:
    parts = path.split("/")
    if len(parts) < 3:
        return ""
    if parts[0] in {"packages", "plugins"}:
        return "/".join(parts[:2]) + "/**"
    if parts[0] == "services":
        if parts[1] == "core" and len(parts) >= 3:
            return "/".join(parts[:3]) + "/**"
        return "/".join(parts[:2]) + "/**"
    return ""


def _bullet_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _indent_block(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to audit. Defaults to the current directory.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report format. Defaults to markdown.",
    )
    parser.add_argument(
        "--all-gated-jobs",
        action="store_true",
        help="Audit every job gated by the changes action, not just test-like jobs.",
    )
    args = parser.parse_args(argv)

    report = audit_repository(args.repo_root.resolve(), include_all_gated_jobs=args.all_gated_jobs)
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report) + "\n")
    return 0


class _PathCollector:
    def __init__(self) -> None:
        self._paths: dict[str, InferredPath] = {}

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self._paths)

    def add(self, path: str, reason: str, confidence: Confidence) -> None:
        normalized = _normalize_repo_path(path)
        existing = self._paths.get(normalized)
        if existing is None or _CONFIDENCE_RANK[confidence] > _CONFIDENCE_RANK[existing.confidence]:
            self._paths[normalized] = InferredPath(path=normalized, reason=reason, confidence=confidence)

    def as_tuple(self) -> tuple[InferredPath, ...]:
        return tuple(sorted(self._paths.values(), key=lambda item: (item.path, item.reason)))


def _infer_run_block(
    run_block: str,
    repo: RepoModel,
    collector: _PathCollector,
    active_make_targets: tuple[str, ...],
) -> None:
    for command in _logical_shell_lines(run_block):
        tokens = _split_shell(command)
        if not tokens:
            continue
        _infer_tokens(tokens, repo, collector, active_make_targets)


def _infer_tokens(
    tokens: Sequence[str],
    repo: RepoModel,
    collector: _PathCollector,
    active_make_targets: tuple[str, ...],
) -> None:
    for index, token in enumerate(tokens):
        if token == "uv":
            collector.add("pyproject.toml", "uv command reads root dependency declarations", "high")
            collector.add("uv.lock", "uv command installs locked dependency versions", "high")
            _infer_uv_options(tokens[index + 1 :], repo, collector)
        if token == "pytest":
            _infer_pytest_paths(tokens[index + 1 :], repo, collector)
        if token == "make":
            _infer_make_targets(tokens[index + 1 :], repo, collector, active_make_targets)
        if token in {"python", "python3"}:
            _infer_python_module(tokens[index + 1 :], repo, collector)


def _infer_uv_options(tokens: Sequence[str], repo: RepoModel, collector: _PathCollector) -> None:
    packages = _option_values(tokens, "--package")
    groups = _option_values(tokens, "--group") + _option_values(tokens, "--only-group")
    extras = _option_values(tokens, "--extra")

    for group in groups:
        for dependency in repo.dependency_groups.get(group, ()):
            package = repo.packages_by_name.get(_normalize_package_name(dependency))
            if package is not None:
                collector.add(
                    _relpath(repo.root, package.pyproject),
                    f"uv dependency group `{group}` includes `{dependency}`",
                    "high",
                )

    for package_name in packages:
        package = repo.packages_by_name.get(_normalize_package_name(package_name))
        if package is not None:
            collector.add(
                _relpath(repo.root, package.pyproject), f"uv installs workspace package `{package.name}`", "high"
            )

    if extras and packages:
        for package_name in packages:
            package = repo.packages_by_name.get(_normalize_package_name(package_name))
            if package is not None:
                collector.add(
                    _relpath(repo.root, package.pyproject),
                    f"uv extra(s) {', '.join(f'`{extra}`' for extra in extras)} are declared by `{package.name}`",
                    "high",
                )

    for index, token in enumerate(tokens):
        if token == "pytest":
            _infer_pytest_paths(tokens[index + 1 :], repo, collector)
        if token in {"python", "python3"}:
            _infer_python_module(tokens[index + 1 :], repo, collector)
        if _looks_like_python_script(token, repo.root):
            collector.add(token, "uv runs this Python script directly", "high")


def _infer_pytest_paths(tokens: Sequence[str], repo: RepoModel, collector: _PathCollector) -> None:
    skip_next = False
    for token in tokens:
        if token in {"&&", "||", ";"}:
            break
        if skip_next:
            skip_next = False
            continue
        if token in {"-k", "-m", "--ignore", "--ignore-glob", "--deselect", "--rootdir"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        pytest_path = _pytest_path_token(token)
        if _looks_like_repo_path(pytest_path, repo.root):
            collector.add(pytest_path, "pytest collects this test path", "high")


def _infer_make_targets(
    tokens: Sequence[str],
    repo: RepoModel,
    collector: _PathCollector,
    active_make_targets: tuple[str, ...],
) -> None:
    targets = [token for token in tokens if _looks_like_make_target(token)]
    for target in targets:
        if target not in repo.make_targets or target in active_make_targets:
            continue
        recipe = "\n".join(repo.make_targets[target])
        _infer_run_block(recipe, repo, collector, (*active_make_targets, target))


def _infer_python_module(tokens: Sequence[str], repo: RepoModel, collector: _PathCollector) -> None:
    for index, token in enumerate(tokens):
        if token != "-m" or index + 1 >= len(tokens):
            continue
        module = tokens[index + 1]
        inferred = _module_to_repo_path(module, repo)
        if inferred:
            collector.add(inferred, f"python executes module `{module}`", "high")


def _infer_python_import_closure(repo: RepoModel, test_path: str, collector: _PathCollector) -> None:
    root_path = repo.root / test_path
    files = _python_files(root_path)
    for python_file in files:
        try:
            tree = ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))
        except SyntaxError:
            continue
        for module in _iter_imported_modules(tree):
            inferred = _module_to_repo_path(module, repo)
            if inferred:
                collector.add(inferred, f"`{test_path}` imports `{module}`", "medium")


def _iter_imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module
        elif isinstance(node, ast.Call):
            module = _dynamic_import_module(node)
            if module:
                yield module


def _dynamic_import_module(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "importorskip" and _is_name(node.func.value, "pytest"):
            return _first_string_arg(node)
        if node.func.attr == "import_module" and _is_name(node.func.value, "importlib"):
            return _first_string_arg(node)
    return ""


def _first_string_arg(node: ast.Call) -> str:
    if not node.args:
        return ""
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return ""


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _module_to_repo_path(module: str, repo: RepoModel) -> str:
    parts = module.split(".")
    if not parts:
        return ""
    packages = repo.packages_by_import.get(parts[0], ())
    if not packages:
        return ""
    for package in packages:
        source_root = package.root / "src"
        module_path = source_root.joinpath(*parts)
        if module_path.is_dir():
            return _relpath(repo.root, module_path) + "/**"
        module_file = module_path.with_suffix(".py")
        if module_file.is_file():
            return _relpath(repo.root, module_file)
    for package in packages:
        source_root = package.root / "src"
        top_level = source_root / parts[0]
        if top_level.is_dir():
            return _relpath(repo.root, top_level) + "/**"
        top_level_file = top_level.with_suffix(".py")
        if top_level_file.is_file():
            return _relpath(repo.root, top_level_file)
    return ""


def _packages_by_import_root(packages: Sequence[WorkspacePackage]) -> Mapping[str, tuple[WorkspacePackage, ...]]:
    grouped: dict[str, list[WorkspacePackage]] = {}
    for package in packages:
        for import_root in package.import_roots:
            grouped.setdefault(import_root, []).append(package)
    return {import_root: tuple(group) for import_root, group in grouped.items()}


def _load_workspace_packages(repo_root: Path, root_pyproject: Mapping[str, object]) -> tuple[WorkspacePackage, ...]:
    tool_table = _mapping(root_pyproject.get("tool"))
    uv_table = _mapping(tool_table.get("uv"))
    workspace_table = _mapping(uv_table.get("workspace"))
    members = _string_sequence(workspace_table.get("members"))

    packages: list[WorkspacePackage] = []
    for member in members:
        package_root = repo_root / member.rstrip("/")
        pyproject = package_root / "pyproject.toml"
        if not pyproject.is_file():
            continue
        package_pyproject = _load_toml(pyproject)
        project_table = _mapping(package_pyproject.get("project"))
        name = _string_value(project_table.get("name"))
        if not name:
            continue
        import_roots = _discover_import_roots(package_root)
        packages.append(
            WorkspacePackage(
                name=name,
                root=package_root,
                pyproject=pyproject,
                import_roots=import_roots,
            )
        )
    return tuple(packages)


def _discover_import_roots(package_root: Path) -> tuple[str, ...]:
    src_dir = package_root / "src"
    if not src_dir.is_dir():
        return ()
    roots: list[str] = []
    for child in src_dir.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir() and _is_python_identifier(child.name):
            roots.append(child.name)
        elif child.suffix == ".py" and _is_python_identifier(child.stem):
            roots.append(child.stem)
    return tuple(sorted(roots))


def _load_dependency_groups(root_pyproject: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    groups_table = _mapping(root_pyproject.get("dependency-groups"))
    cache: dict[str, tuple[str, ...]] = {}
    for group_name in groups_table:
        cache[group_name] = _load_dependency_group(group_name, groups_table, active_groups=(), cache=cache)
    return cache


def _load_dependency_group(
    group_name: str,
    groups_table: Mapping[str, object],
    active_groups: tuple[str, ...],
    cache: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    cached = cache.get(group_name)
    if cached is not None:
        return cached
    if group_name in active_groups:
        return ()

    dependencies: list[str] = []
    for item in _object_sequence(groups_table.get(group_name)):
        if isinstance(item, str):
            name = _dependency_project_name(item)
            if name:
                dependencies.append(name)
        elif isinstance(item, Mapping):
            include_group = _string_value(item.get("include-group"))
            if include_group:
                dependencies.extend(
                    _load_dependency_group(include_group, groups_table, (*active_groups, group_name), cache)
                )

    result = tuple(_unique(dependencies))
    cache[group_name] = result
    return result


def _load_toml(path: Path) -> Mapping[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_job_name(block_lines: Sequence[str]) -> str:
    for line in block_lines:
        match = re.match(r"^    name:\s*(.+?)\s*$", line)
        if match is not None:
            return _strip_yaml_scalar(match.group(1))
    return ""


def _parse_run_blocks(block_lines: Sequence[str]) -> list[str]:
    blocks: list[str] = []
    index = 0
    while index < len(block_lines):
        line = block_lines[index]
        match = re.match(r"^(\s*)run:\s*(.*)$", line)
        if match is None:
            index += 1
            continue
        indent = len(match.group(1))
        rest = match.group(2).strip()
        if rest in {"|", ">"}:
            collected: list[str] = []
            index += 1
            while index < len(block_lines):
                candidate = block_lines[index]
                if candidate.strip() and _leading_spaces(candidate) <= indent:
                    break
                collected.append(candidate[indent + 2 :] if len(candidate) >= indent + 2 else "")
                index += 1
            blocks.append("\n".join(collected))
            continue
        blocks.append(_strip_yaml_scalar(rest))
        index += 1
    return blocks


def _logical_shell_lines(block: str) -> Iterable[str]:
    pending = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            line = line[1:].strip()
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        if pending:
            line = pending + line
            pending = ""
        yield _normalize_recipe_line(line)
    if pending:
        yield _normalize_recipe_line(pending.strip())


def _split_shell(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return tuple(command.split())


def _option_values(tokens: Sequence[str], option_name: str) -> tuple[str, ...]:
    values: list[str] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token == option_name and index + 1 < len(tokens):
            values.append(tokens[index + 1])
            skip_next = True
            continue
        prefix = option_name + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
    return tuple(values)


def _resolve_filter_ids(outputs: Sequence[str], action: ChangeAction) -> tuple[str, ...]:
    filter_ids: list[str] = []
    for output in outputs:
        refs = action.output_filters.get(output, (output,))
        for ref in refs:
            if ref in action.filters:
                filter_ids.append(ref)
    return tuple(_unique(filter_ids))


def _path_is_covered(path: str, filters: Sequence[ChangeFilter]) -> bool:
    normalized = _normalize_repo_path(path)
    for change_filter in filters:
        for pattern in change_filter.patterns:
            if _pattern_matches(normalized, _normalize_repo_path(pattern)):
                return True
    return False


def _pattern_matches(path: str, pattern: str) -> bool:
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path == prefix or path.startswith(prefix + "/")
    if path.endswith("/**"):
        prefix = path[:-3]
        return fnmatch.fnmatchcase(prefix, pattern) or fnmatch.fnmatchcase(prefix + "/", pattern)
    return False


def _suggest_pattern(path: str) -> str:
    if path.endswith("/**"):
        return path
    return path


def _is_test_like_job(job: WorkflowJob) -> bool:
    text = (job.job_id + "\n" + job.name + "\n" + "\n".join(job.run_blocks)).lower()
    return "pytest" in text or "make test" in text or "benchmark" in text or " test" in text or "smoke" in text


def _path_points_to_python_tests(repo_root: Path, path: str) -> bool:
    if path.endswith("/**"):
        return False
    candidate = repo_root / path
    if candidate.is_file():
        return candidate.suffix == ".py"
    return candidate.is_dir()


def _python_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,) if path.suffix == ".py" else ()
    if not path.is_dir():
        return ()
    return tuple(sorted(child for child in path.rglob("*.py") if ".venv" not in child.parts))


def _looks_like_repo_path(token: str, repo_root: Path) -> bool:
    cleaned = _normalize_repo_path(token)
    if not cleaned or cleaned.startswith("$"):
        return False
    if "/" in cleaned or cleaned.endswith(".py"):
        return True
    return (repo_root / cleaned).exists()


def _pytest_path_token(token: str) -> str:
    return token.split("::", maxsplit=1)[0]


def _looks_like_python_script(token: str, repo_root: Path) -> bool:
    return token.endswith(".py") and (repo_root / token).is_file()


def _looks_like_make_target(token: str) -> bool:
    if token.startswith("-") or "=" in token or token in {"&&", "||", ";"}:
        return False
    return bool(re.match(r"^[A-Za-z0-9_./-]+$", token))


def _normalize_recipe_line(line: str) -> str:
    normalized = line.strip()
    normalized = normalized.replace("$(UV)", "uv")
    normalized = normalized.replace("${UV}", "uv")
    normalized = normalized.replace("$(MAKE)", "make")
    normalized = re.sub(r"\$\([A-Za-z0-9_]+\)", "", normalized)
    return normalized


def _dependency_project_name(requirement: str) -> str:
    requirement = requirement.split(";", maxsplit=1)[0].strip()
    requirement = requirement.split("[", maxsplit=1)[0].strip()
    requirement = _VERSION_OPERATOR_RE.split(requirement, maxsplit=1)[0].strip()
    match = _PROJECT_NAME_RE.match(requirement)
    return match.group(0) if match is not None else ""


def _normalize_package_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip().strip("'\"")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/") if not normalized.endswith("/**") else normalized


def _relpath(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _strip_yaml_scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    if " #" in stripped:
        return stripped.split(" #", maxsplit=1)[0].strip()
    return stripped


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _object_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(value)


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _is_python_identifier(value: str) -> bool:
    return value.isidentifier()


if __name__ == "__main__":
    raise SystemExit(main())
