# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared authorization corpus natively in Regorus, including `with`.

Discovery uses Regorus's parsed AST, not a second Rego parser. Indexed tests
require an exhaustive expected-key manifest: a missing failed case must never
be mistaken for a passing test. Policies, fixtures, and assertions are loaded
unchanged; no source rewriting or external implementation of `with` is used.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

# Upstream re-exports its PyO3 extension dynamically and ships no Engine stub.
from regorus import Engine  # ty: ignore[unresolved-import]

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


def reference_root(reference: dict) -> str:
    while "Var" not in reference:
        reference = next(iter(reference.values()))["refr"]
    return reference["Var"]["value"]


def discover(app: Path = APP) -> list[PolicyTest]:
    parser = Engine()
    packages = {}
    for path in sorted((app / "policy_tests").glob("*.rego")):
        packages[str(path)] = parser.add_policy_from_file(str(path))
    tests = {}
    for module in json.loads(parser.get_ast_as_json()):
        path = module["source"]["file"]
        for rule in module["ast"]["rules"]:
            if "Spec" in rule:
                head = rule["Spec"]["head"]
                reference = next(iter(head.values()))["refr"]
            else:
                head = {}
                reference = rule["Default"]["refr"]
            name = reference_root(reference)
            if not name.startswith("test_"):
                continue
            parameterized = "RefBrack" in reference
            if (head and "Compr" not in head) or not (
                "Var" in reference
                or (
                    parameterized and "Var" in reference["RefBrack"]["refr"] and "Var" in reference["RefBrack"]["index"]
                )
            ):
                raise ValueError(f"Unsupported test declaration: {path}: {name}")
            domain = None
            if parameterized:
                # Pin manifests to literal `cases` tables too: adding a failing
                # row must not pass merely because its key is absent in output.
                first = rule["Spec"]["bodies"][0]["query"]["stmts"][0]["literal"]
                assignment = first.get("Expr", {}).get("expr", {}).get("AssignExpr", {})
                if assignment.get("lhs", {}).get("Var", {}).get("value") == "cases":
                    fields = assignment.get("rhs", {}).get("Object", {}).get("fields")
                    if fields is not None and all("String" in field[1] for field in fields):
                        domain = frozenset(field[1]["String"]["value"] for field in fields)
            full_name = f"{packages[path]}.{name}"
            tests[full_name] = PolicyTest(Path(path), full_name, parameterized, domain)
    if not tests:
        raise ValueError("No policy tests discovered")
    return list(tests.values())


class CorpusRunner:
    def __init__(self, app: Path = APP, parameter_keys: dict[str, set[str]] | None = None):
        self.parameter_keys = PARAMETER_KEYS if parameter_keys is None else parameter_keys
        self.engine = Engine()
        for directory in ("policies", "policy_tests"):
            for path in sorted((app / directory).glob("*.rego")):
                self.engine.add_policy_from_file(str(path))
        data = yaml.safe_load((app.parent / "assets/static-authz.yaml").read_text())
        self.engine.add_data_json(json.dumps(data, allow_nan=False))
        self.engine.set_input_json("{}")

    def run(self, test: PolicyTest) -> int:
        value = json.loads(self.engine.eval_rule_as_json(test.rule))
        if not test.parameterized:
            if value is not True:
                raise AssertionError(f"{test.id}: expected true, got {value!r}")
            return 1
        if test.rule not in self.parameter_keys:
            raise ValueError(f"Add exhaustive expected parameter keys for {test.rule}")
        expected = self.parameter_keys[test.rule]
        if test.parameter_domain is not None and expected != test.parameter_domain:
            raise ValueError(f"Expected parameter keys do not match cases table for {test.rule}")
        if not expected or not isinstance(value, (list, dict)) or set(value) != expected:
            raise AssertionError(f"{test.id}: expected parameter keys {expected!r}, got {value!r}")
        if isinstance(value, dict) and any(result is not True for result in value.values()):
            raise AssertionError(f"{test.id}: failed parameterized assertions: {value!r}")
        return len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="", help="Run tests whose ID contains this string")
    args = parser.parse_args()
    tests = [test for test in discover() if args.match in test.id]
    if not tests:
        parser.error("No matching tests")
    runner = CorpusRunner()
    cases = 0
    for test in tests:
        cases += runner.run(test)
        print(f"PASS {test.id}", flush=True)
    print(f"Passed {len(tests)} Rego policy rules ({cases} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
