# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic skill documentation and source-readiness checks."""

import ast
import re
from typing import Literal

import yaml
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.metrics.protocol import MetricDiagnostic, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.skill_inputs import artifact_contents
from nhx_evals_sdk.metrics.skills import (
    _MARKDOWN,
    SkillReferenceIntegrityMetric,
    SkillStructureMetric,
    _snapshot,
    _split_document,
    _unmeasured,
)
from nhx_evals_sdk.values.metrics import MetricBase
from pydantic import Field

_TEXT_SUFFIXES = (".md", ".mdc", ".py", ".sh", ".bash", ".toml", ".yaml", ".yml", ".json", ".txt", ".js", ".ts", ".tsx")
_NESTING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)


def _headings(body: str) -> set[str]:
    tokens = _MARKDOWN.parse(body)
    return {
        tokens[index + 1].content.casefold() for index, token in enumerate(tokens[:-1]) if token.type == "heading_open"
    }


def _readiness(checks: dict[str, bool]) -> MetricResult:
    return MetricResult(
        outputs=[
            MetricOutput(name="score", value=sum(checks.values()) / len(checks)),
            MetricOutput(name="defect_count", value=sum(not passed for passed in checks.values())),
        ],
        diagnostics=[
            MetricDiagnostic(message="Readiness check was not satisfied.", details={"rule": rule})
            for rule, passed in checks.items()
            if not passed
        ],
    )


class _ReadinessMetric(MetricBase):
    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("score", required=False),
            MetricOutputSpec.discrete_score("defect_count", required=False),
        ]


class SkillStaticCorrectnessMetric(_ReadinessMetric):
    """Check entry structure, documented examples, and local reference validity."""

    type: Literal[MetricType.SKILL_STATIC_CORRECTNESS] = MetricType.SKILL_STATIC_CORRECTNESS

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        snapshot = _snapshot(input)
        if snapshot is None or snapshot.document is None:
            return _unmeasured("missing_document", "The original SKILL.md is unavailable.")
        structure = await SkillStructureMetric().compute_scores(input)
        references = await SkillReferenceIntegrityMetric().compute_scores(input)
        structure_values = {output.name: output.value for output in structure.outputs}
        reference_values = {output.name: output.value for output in references.outputs}
        _, body = _split_document(snapshot.document)
        blocks = _MARKDOWN.parse(body)
        return _readiness(
            {
                "structure": bool(structure_values["conforms"]),
                "examples": any(block.type == "fence" for block in blocks)
                or any("example" in heading for heading in _headings(body)),
                "references": reference_values["broken"] == 0,
            }
        )


class SkillDescriptionQualityMetric(_ReadinessMetric):
    """Measure explicit trigger context, descriptive length, and specific language."""

    type: Literal[MetricType.SKILL_DESCRIPTION_QUALITY] = MetricType.SKILL_DESCRIPTION_QUALITY
    minimum_characters: int = Field(default=20, gt=0, strict=True)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        snapshot = _snapshot(input)
        if snapshot is None or snapshot.document is None:
            return _unmeasured("missing_document", "The original SKILL.md is unavailable.")
        header, _ = _split_document(snapshot.document)
        try:
            fields = yaml.safe_load(header) if header is not None else {}
        except (yaml.YAMLError, RecursionError):
            fields = {}
        description = fields.get("description", "") if isinstance(fields, dict) else ""
        description = description if isinstance(description, str) else ""
        words = set(re.findall(r"\b\w+\b", description.casefold()))
        return _readiness(
            {
                "descriptive_length": len(description.strip()) >= self.minimum_characters,
                "trigger_context": bool(words & {"use", "when", "for", "helps", "allows"}),
                "specific_language": bool(words) and not words & {"something", "things", "stuff", "various", "general"},
            }
        )


class SkillReliabilityReadinessMetric(_ReadinessMetric):
    """Check documented operational guidance and explicitly required I/O contracts."""

    type: Literal[MetricType.SKILL_RELIABILITY_READINESS] = MetricType.SKILL_RELIABILITY_READINESS
    requires_inputs: bool = Field(default=False, strict=True)
    requires_output_contract: bool = Field(default=False, strict=True)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        snapshot = _snapshot(input)
        if snapshot is None or snapshot.document is None:
            return _unmeasured("missing_document", "The original SKILL.md is unavailable.")
        _, body = _split_document(snapshot.document)
        headings = _headings(body)
        checks = {
            "prerequisites": any("prerequisite" in heading or "requirement" in heading for heading in headings),
            "error_handling": any("error" in heading or "troubleshoot" in heading for heading in headings),
            "limitations": any("limitation" in heading or "constraint" in heading for heading in headings),
        }
        if self.requires_inputs:
            checks["inputs"] = any("input" in heading or "argument" in heading for heading in headings)
        if self.requires_output_contract:
            checks["output_contract"] = any("output" in heading for heading in headings)
        return _readiness(checks)


class SkillScriptQualityMetric(MetricBase):
    """Report advisory findings from Python source."""

    type: Literal[MetricType.SKILL_SCRIPT_QUALITY] = MetricType.SKILL_SCRIPT_QUALITY
    max_nesting: int = Field(default=6, ge=1, strict=True)

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.discrete_score("scripts_checked", required=False),
            MetricOutputSpec.discrete_score("defect_count", required=False),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        snapshot = _snapshot(input)
        if snapshot is None:
            return _unmeasured("missing_snapshot", "The original skill snapshot is unavailable.")
        scripts = [path for path in snapshot.files if path.startswith("scripts/") and path.endswith(".py")]
        contents = artifact_contents(input, package=True) if scripts else {}
        if contents is None or any(path not in contents for path in scripts):
            return _unmeasured("missing_script_source", "Original source is required for every Python script.")
        findings: list[MetricDiagnostic] = []
        for path in scripts:
            for rule in _script_findings(contents[path], self.max_nesting):
                findings.append(
                    MetricDiagnostic(message="Advisory Python script finding.", details={"file": path, "rule": rule})
                )
        return MetricResult(
            outputs=[
                MetricOutput(name="scripts_checked", value=len(scripts)),
                MetricOutput(name="defect_count", value=len(findings)),
            ],
            diagnostics=findings,
        )


def _script_findings(source: str, max_nesting: int) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ["syntax_error"]
    nodes = list(ast.walk(tree))
    findings = []
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in nodes):
        findings.append("flat_script")
    first_line = source.partition("\n")[0]
    if not first_line.startswith("#!") or not first_line[2:].strip():
        findings.append("missing_shebang")
    imports = {alias.name.split(".")[0] for node in nodes if isinstance(node, ast.Import) for alias in node.names} | {
        node.module.split(".")[0] for node in nodes if isinstance(node, ast.ImportFrom) and node.module
    }
    if not imports & {"argparse", "click", "typer"} and not any(isinstance(node, ast.Raise) for node in nodes):
        findings.append("no_input_validation")
    docstring = ast.get_docstring(tree) or ""
    if not all(
        re.search(rf"(?im)^\s*{section}\s*:", docstring)
        for section in ("usage", "arguments?", "outputs?", r"exit\s+codes?")
    ):
        findings.append("missing_script_contract")
    if any(
        isinstance(node, ast.Constant) and type(node.value) in {int, float} and node.value not in {-1, 0, 1, 2}
        for node in nodes
    ):
        findings.append("magic_numbers")
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    depth = 0
    while pending:
        node, level = pending.pop()
        level += isinstance(node, _NESTING)
        depth = max(depth, level)
        pending.extend((child, level) for child in ast.iter_child_nodes(node))
    if depth > max_nesting:
        findings.append("deep_nesting")
    for node in nodes:
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"exit", "sys.exit", "SystemExit"} and node.args:
                value = _literal_integer(node.args[0])
                if value is not None and value not in {0, 1, 2}:
                    findings.append("nonstandard_exit_code")
            if name == "print" and node.args and isinstance(node.args[0], ast.Constant):
                text = node.args[0].value
                stderr = any(
                    keyword.arg == "file" and ast.unparse(keyword.value) == "sys.stderr" for keyword in node.keywords
                )
                if isinstance(text, str) and re.match(r"(?i)\s*(error|usage|invalid)\b", text) and not stderr:
                    findings.append("diagnostic_to_stdout")
    return sorted(set(findings))


def _literal_integer(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_integer(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value) if value is not None else None
    return None


class SkillUnicodeSafetyMetric(MetricBase):
    """Count hidden Unicode controls with explicit document or package scope."""

    type: Literal[MetricType.SKILL_UNICODE_SAFETY] = MetricType.SKILL_UNICODE_SAFETY
    scope: Literal["document", "package"] = "document"
    include_spaces: bool = Field(default=False, strict=True)

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.discrete_score("finding_count", required=False),
            MetricOutputSpec.discrete_score("files_checked", required=False),
            MetricOutputSpec.continuous_score("clean", required=False),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        snapshot = _snapshot(input)
        contents = artifact_contents(input, package=self.scope == "package")
        if snapshot is None or contents is None:
            return _unmeasured("missing_content", "Original text is unavailable for the selected scope.")
        expected = (
            [path for path in snapshot.files if path.endswith(_TEXT_SUFFIXES)]
            if self.scope == "package"
            else ["SKILL.md"]
        )
        if not expected:
            return _unmeasured("no_text_files", "There are no text files in the selected scope.")
        if any(path not in contents for path in expected):
            return _unmeasured("incomplete_content", "Every in-scope text file must be available.")
        findings = []
        for path in expected:
            for offset, char in enumerate(contents[path]):
                category = _unicode_category(ord(char), self.include_spaces)
                if category is None or (offset == 0 and char == "\ufeff"):
                    continue
                findings.append(
                    MetricDiagnostic(
                        message="Hidden Unicode control detected.",
                        details={
                            "file": path,
                            "offset": offset,
                            "codepoint": f"U+{ord(char):04X}",
                            "category": category,
                        },
                    )
                )
        return MetricResult(
            outputs=[
                MetricOutput(name="finding_count", value=len(findings)),
                MetricOutput(name="files_checked", value=len(expected)),
                MetricOutput(name="clean", value=float(not findings)),
            ],
            diagnostics=findings,
        )


def _unicode_category(code: int, spaces: bool) -> str | None:
    if 0xE0000 <= code <= 0xE007F:
        return "unicode_tags"
    if code in {0x061C, 0x200E, 0x200F} or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
        return "bidi_controls"
    if code in {0x034F, 0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}:
        return "zero_width"
    if 0x2061 <= code <= 0x2064 or 0x206A <= code <= 0x206F:
        return "invisible_controls"
    if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        return "variation_selectors"
    if spaces and (code in {0x00A0, 0x00AD, 0x3000} or 0x2000 <= code <= 0x200A):
        return "confusable_spaces"
    return None
