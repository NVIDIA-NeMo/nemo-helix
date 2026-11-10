# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the shared Rego assertions with the embedded evaluator."""

import json
from pathlib import Path

import pytest
import yaml
from regorus import Engine  # ty: ignore[unresolved-import]

APP = Path(__file__).resolve().parents[1] / "src/nhx/core/auth/app"


def policy_tests() -> list[str]:
    """Discover test rules from Regorus's AST, including undefined/failing rules."""
    parser = Engine()
    packages = {
        str(path): parser.add_policy_from_file(str(path)) for path in sorted((APP / "policy_tests").glob("*.rego"))
    }
    tests = set()
    for module in json.loads(parser.get_ast_as_json()):
        for rule in module["ast"]["rules"]:
            reference = next(iter(rule["Spec"]["head"].values()))["refr"] if "Spec" in rule else rule["Default"]["refr"]
            root = reference
            while "Var" not in root:
                root = next(iter(root.values()))["refr"]
            name = root["Var"]["value"]
            if name.startswith("test_"):
                # Tests must be booleans; use `every` for tables so failing rows
                # cannot disappear from the result of a set-valued test rule.
                assert "Var" in reference, f"Use a boolean test rule: {name}"
                tests.add(f"{packages[module['source']['file']]}.{name}")
    assert tests, "No Rego tests discovered"
    return sorted(tests)


@pytest.fixture(scope="module")
def engine():
    evaluator = Engine()
    for directory in ("policies", "policy_tests"):
        for path in sorted((APP / directory).glob("*.rego")):
            evaluator.add_policy_from_file(str(path))
    evaluator.add_data_json(json.dumps(yaml.safe_load((APP.parent / "assets/static-authz.yaml").read_text())))
    evaluator.set_input_json("{}")
    return evaluator


@pytest.mark.parametrize("rule", policy_tests())
def test_shared_rego_policy_corpus(engine, rule):
    assert json.loads(engine.eval_rule_as_json(rule)) is True
