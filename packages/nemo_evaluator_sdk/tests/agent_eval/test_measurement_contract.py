# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MEASUREMENT_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "runtime_sec",
    "duration_ms",
    "cost_usd",
}


def test_production_consumers_do_not_read_measurements_from_trial_metadata() -> None:
    root = Path(__file__).resolve().parents[4]
    scan_roots = (
        root / "packages/nemo_evaluator_sdk/src",
        root / "plugins/nemo-evaluator/src",
        root / "packages/nemo_evaluator_sdk/examples",
    )
    violations: list[str] = []

    for scan_root in scan_roots:
        for path in scan_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(
                f"{path.relative_to(root)}:{line}: {key}" for line, key in _measurement_metadata_reads(tree)
            )

    assert violations == [], "durable measurements must be read from trial.measurements:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param('t.metadata["prompt_tokens"]', (1, "prompt_tokens"), id="renamed-receiver"),
        pytest.param('trials[0].metadata.get("runtime_sec")', (1, "runtime_sec"), id="indexed-receiver"),
        pytest.param('md = trial.metadata\nmd.pop("cost_usd")', (2, "cost_usd"), id="metadata-alias"),
        pytest.param(
            'md = trial.metadata\nmeasurements = md\nmeasurements["total_tokens"]',
            (3, "total_tokens"),
            id="chained-metadata-alias",
        ),
        pytest.param(
            'trial.metadata.setdefault("completion_tokens", 0)',
            (1, "completion_tokens"),
            id="setdefault-read",
        ),
        pytest.param(
            '"cache_read_tokens" in trial.metadata',
            (1, "cache_read_tokens"),
            id="membership-test",
        ),
        pytest.param(
            '"cache_creation_tokens" not in trial.metadata',
            (1, "cache_creation_tokens"),
            id="negative-membership-test",
        ),
    ],
)
def test_tripwire_rejects_common_metadata_reads(source: str, expected: tuple[int, str]) -> None:
    assert _measurement_metadata_reads(ast.parse(source)) == [expected]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('usage["prompt_tokens"]', id="raw-usage-mapping"),
        pytest.param('trial.metadata["reward"]', id="non-measurement-metadata"),
        pytest.param(
            'md = trial.metadata\ndef consumer():\n    md = usage\n    return md["prompt_tokens"]',
            id="alias-does-not-leak-into-function-scope",
        ),
    ],
)
def test_tripwire_allows_non_trial_measurement_reads(source: str) -> None:
    assert _measurement_metadata_reads(ast.parse(source)) == []


def _measurement_metadata_reads(tree: ast.AST) -> list[tuple[int, str]]:
    visitor = _MeasurementMetadataReadVisitor()
    visitor.visit(tree)
    return visitor.reads


class _MeasurementMetadataReadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: list[tuple[int, str]] = []
        self._metadata_aliases: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_metadata_mapping(node.value):
            self._record(node, _string_constant(node.slice))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and self._is_metadata_mapping(node.func.value)
            and node.args
        ):
            self._record(node, _string_constant(node.args[0]))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(operator, ast.In | ast.NotIn) and self._is_metadata_mapping(comparator):
                self._record(node, _string_constant(left))
            left = comparator
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        is_metadata_mapping = self._is_metadata_mapping(node.value)
        self.visit(node.value)
        for target in node.targets:
            self._bind_alias(target, is_metadata_mapping=is_metadata_mapping)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            is_metadata_mapping = self._is_metadata_mapping(node.value)
            self.visit(node.value)
            self._bind_alias(node.target, is_metadata_mapping=is_metadata_mapping)
        self.visit(node.annotation)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        is_metadata_mapping = self._is_metadata_mapping(node.value)
        self.visit(node.value)
        self._bind_alias(node.target, is_metadata_mapping=is_metadata_mapping)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_scoped(node)

    def _visit_scoped(self, node: ast.AST) -> None:
        outer_aliases = self._metadata_aliases
        self._metadata_aliases = set()
        self.generic_visit(node)
        self._metadata_aliases = outer_aliases

    def _is_metadata_mapping(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "metadata") or (
            isinstance(node, ast.Name) and node.id in self._metadata_aliases
        )

    def _bind_alias(self, target: ast.AST, *, is_metadata_mapping: bool) -> None:
        if not isinstance(target, ast.Name):
            return
        if is_metadata_mapping:
            self._metadata_aliases.add(target.id)
        else:
            self._metadata_aliases.discard(target.id)

    def _record(self, node: ast.expr, key: str | None) -> None:
        if key is not None and key in _MEASUREMENT_KEYS:
            self.reads.append((node.lineno, key))


def _string_constant(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
