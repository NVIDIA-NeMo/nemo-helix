# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute unchanged shared Rego assertions with native OPA, never a lowering layer.

OPA's own parser discovers rules; its test runner reports every indexed case.
Expected parameter manifests prevent empty/partially enumerated tests from
passing vacuously, including a newly added failing row in a literal cases table.
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nhx.core.auth.app.embedded_pdp.opa_binary import resolve_opa_binary

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "services/core/auth/src/nhx/core/auth/app"
PARAMETER_KEYS = {
    "data.authz_test.test_workspace_deletion_requires_admin": {
        "viewer@test.com",
        "editor@test.com",
        "admin@test.com",
        "platform-admin@test.com",
        "other-admin@test.com",
        "outsider@test.com",
    },
    "data.authz_test.test_editor_retains_workspace_update_and_resource_delete": {
        "/apis/entities/v2/workspaces/test-ws",
        "/apis/models/v2/workspaces/test-ws/models/test-model",
    },
}


@dataclass(frozen=True)
class PolicyTest:
    path: Path
    rule: str
    parameterized: bool = False
    parameter_domain: frozenset[str] | None = None

    @property
    def id(self) -> str:
        return f"{self.path.name}::{self.rule}"


def _resolve_binary(binary_path: str | None = None) -> Path:
    return resolve_opa_binary(binary_path if binary_path is not None else os.environ.get("NHX_AUTH_OPA_BINARY_PATH"))


def _command(binary: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=120, check=False)


def discover(app: Path = APP, binary: Path | None = None) -> list[PolicyTest]:
    binary = binary or _resolve_binary()
    tests = {}
    for path in sorted((app / "policy_tests").glob("*.rego")):
        result = _command(binary, "parse", "--format=json", str(path))
        if result.returncode:
            raise ValueError(f"OPA cannot parse {path}: {result.stderr}")
        module = json.loads(result.stdout)
        package = ".".join(part["value"] for part in module["package"]["path"])
        for rule in module.get("rules", []):
            head = rule["head"]
            reference = head.get("ref", [])
            name = head.get("name") or reference[0]["value"]
            if name.startswith("todo_"):
                raise ValueError(f"Skipped policy test is forbidden: {path}: {name}")
            if not name.startswith("test_"):
                continue
            parameterized = len(reference) > 1
            if head.get("args") or len(reference) > 2 or (parameterized and reference[1]["type"] != "var"):
                raise ValueError(f"Unsupported test declaration: {path}: {name}")
            domain = None
            for expression in rule.get("body", []):
                terms = expression.get("terms")
                if (
                    isinstance(terms, list)
                    and len(terms) == 3
                    and terms[1] == {"type": "var", "value": "cases"}
                    and terms[2].get("type") == "object"
                ):
                    fields = terms[2]["value"]
                    if all(key["type"] == "string" for key, _ in fields):
                        domain = frozenset(key["value"] for key, _ in fields)
            full_name = f"{package}.{name}"
            candidate = PolicyTest(path, full_name, parameterized, domain)
            if full_name in tests:
                raise ValueError(f"Duplicate test declaration: {full_name}")
            tests[full_name] = candidate
    if not tests:
        raise ValueError("No policy tests discovered")
    return list(tests.values())


class CorpusRunner:
    def __init__(
        self,
        app: Path = APP,
        parameter_keys: dict[str, set[str]] | None = None,
        binary: Path | None = None,
    ):
        self.app = app
        self.binary = binary or _resolve_binary()
        self.parameter_keys = PARAMETER_KEYS if parameter_keys is None else parameter_keys
        self.tests = discover(app, self.binary)
        self._reports = None
        self._returncode = None

    def _load_reports(self) -> dict[str, dict]:
        if self._reports is None:
            result = _command(
                self.binary,
                "test",
                "--format=json",
                "--fail-on-empty",
                str(self.app / "policies"),
                str(self.app / "policy_tests"),
                str(self.app.parent / "assets/static-authz.yaml"),
            )
            try:
                report = json.loads(result.stdout)
            except ValueError as exc:
                raise AssertionError(f"OPA test produced no JSON report: {result.stderr}") from exc
            if not isinstance(report, list) or not report:
                raise AssertionError("OPA returned zero policy tests")
            reports = {}
            for item in report:
                key = f"{item['package']}.{item['name']}"
                if key in reports:
                    raise AssertionError(f"Duplicate OPA test result: {key}")
                reports[key] = item
            expected = {test.rule for test in self.tests}
            if set(reports) != expected:
                raise AssertionError(f"OPA report rule set mismatch: expected {expected}, got {set(reports)}")
            self._reports = reports
            self._returncode = result.returncode
        return self._reports

    def run(self, test: PolicyTest) -> int:
        if test.parameterized:
            if test.rule not in self.parameter_keys:
                raise ValueError(f"Add exhaustive expected parameter keys for {test.rule}")
            if test.parameter_domain is not None and self.parameter_keys[test.rule] != test.parameter_domain:
                raise ValueError(f"Expected parameter keys do not match cases table for {test.rule}")
        report = self._load_reports()[test.rule]
        if report.get("fail") or report.get("error") or report.get("skip"):
            raise AssertionError(f"{test.id}: failed/skipped assertion: {report}")
        if not test.parameterized:
            if report.get("sub_results"):
                raise AssertionError(f"Unexpected indexed results for {test.id}")
            return 1
        expected = self.parameter_keys[test.rule]
        results = report.get("sub_results")
        if not expected or not isinstance(results, dict) or set(results) != expected:
            raise AssertionError(f"{test.id}: expected parameter keys {expected!r}, got {results!r}")
        for key, child in results.items():
            if child.get("fail") or child.get("error") or child.get("skip") or child.get("sub_results"):
                raise AssertionError(f"{test.id}[{key}]: failed/skipped/unsupported indexed assertion: {child}")
        return len(expected)

    def run_all(self, match: str = "") -> tuple[int, int]:
        tests = [test for test in self.tests if match in test.id]
        if not tests:
            raise ValueError("No matching tests")
        cases = sum(self.run(test) for test in tests)
        if self._returncode != 0:
            raise AssertionError(f"OPA test exited with status {self._returncode}")
        if cases <= 0:
            raise AssertionError("OPA returned zero test cases")
        return len(tests), cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="", help="Validate tests whose ID contains this string")
    parser.add_argument("--opa-binary-path", help="Provisioned compatible native OPA executable")
    args = parser.parse_args()
    runner = CorpusRunner(binary=_resolve_binary(args.opa_binary_path))
    rules, cases = runner.run_all(args.match)
    print(f"Passed {rules} Rego policy rules ({cases} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
