# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the authorization Rego corpus with regopy, without native `with`.

rego-cpp 1.5.2 cannot compile transitive rule/builtin calls under `with`.
The corpus uses only input and base-data overrides, which this runner implements
by evaluating each expression in an interpreter with those documents installed.
The original assertions and fixture expressions remain Rego, not Python ports.
This deliberately supports a bounded test grammar: conjunctive test bodies,
ordinary Rego expressions, and input/base-data `with` modifiers. Unsupported
constructs fail instead of silently skipping tests. Production policies are
loaded unchanged.
"""

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from regopy import Interpreter
from regopy.node import Node
from regopy.rego_shared import NodeKind, rego_get_error, rego_node_get, rego_node_size

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "services/core/auth/src/nhx/core/auth/app"


def clean(source: str) -> str:
    """Remove comments, without changing strings or line boundaries."""
    return re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', lambda m: "" if m[0].startswith("#") else m[0], source)


def boundaries(source: str):
    """Yield character positions outside strings and balanced containers."""
    depth = 0
    quoted = escaped = False
    for index, char in enumerate(source):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced Rego expression")
        elif depth == 0:
            yield index
    if quoted or depth:
        raise ValueError("Unbalanced Rego expression")


def statements(body: str) -> list[str]:
    starts = [0]
    for index in boundaries(body):
        if body[index] in "\n;" and not body[index + 1 :].lstrip().startswith("with "):
            starts.append(index + 1)
    starts.append(len(body) + 1)
    return [part for a, b in zip(starts, starts[1:]) if (part := body[a : b - 1].strip())]


@dataclass(frozen=True)
class Binding:
    value: object
    term: str
    json_compatible: bool


def binding(node: Node) -> Binding:
    """Round-trip Rego sets as sets, not the arrays in regopy's JSON output."""
    if node.kind in (NodeKind.Term, NodeKind.Scalar):
        return binding(Node(rego_node_get(node._impl, 0)))
    children = [binding(Node(rego_node_get(node._impl, i))) for i in range(rego_node_size(node._impl))]
    compatible = all(child.json_compatible for child in children)
    if node.kind == NodeKind.Set:
        term = "{" + ",".join(child.term for child in children) + "}" if children else "set()"
        return Binding(json.loads(node.json()), term, False)
    if node.kind == NodeKind.Array:
        return Binding(json.loads(node.json()), "[" + ",".join(child.term for child in children) + "]", compatible)
    if node.kind == NodeKind.ObjectItem:
        return Binding(
            None, children[0].term + ":" + children[1].term, compatible and isinstance(children[0].value, str)
        )
    if node.kind == NodeKind.Object:
        return Binding(json.loads(node.json()), "{" + ",".join(child.term for child in children) + "}", compatible)
    return Binding(json.loads(node.json()), node.json(), True)


@dataclass(frozen=True)
class PolicyTest:
    path: Path
    name: str
    package: str
    fixtures: str
    body: str
    parameter: str | None = None

    @property
    def id(self) -> str:
        return f"{self.path.name}::{self.name}"


def discover(app: Path = APP) -> list[PolicyTest]:
    tests = []
    for path in sorted((app / "policy_tests").glob("*.rego")):
        source = clean(path.read_text())
        matches = list(re.finditer(r"^[ \t]*(test_\w+)(?:\[(\w+)\])?\s+if\s*\{", source, re.MULTILINE))
        if len(matches) != len(re.findall(r"^[ \t]*test_\w+", source, re.MULTILINE)):
            raise ValueError(f"Unsupported test declaration: {path}")
        spans = []
        for match in matches:
            start = match.end()
            # A wrapping array lets the scanner find the test body's closing brace.
            tail = source[start:]
            depth = 1
            quoted = escaped = False
            for index, char in enumerate(tail):
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if not depth:
                        spans.append((match.start(), start, start + index, match[1], match[2]))
                        break
            else:
                raise ValueError(f"Unclosed test in {path}")
        fixtures = source
        for begin, _, end, _, _ in reversed(spans):
            fixtures = fixtures[:begin] + fixtures[end + 1 :]
        package = re.search(r"^package ([\w.]+)", fixtures, re.MULTILINE)
        if not package:
            raise ValueError(f"Missing package: {path}")
        for _, start, end, name, parameter in spans:
            tests.append(PolicyTest(path, name, package[1], fixtures, source[start:end], parameter))
    if not tests:
        raise ValueError("No policy tests discovered")
    return tests


class CorpusRunner:
    def __init__(self, app: Path = APP):
        self.modules = [(p.name, p.read_text()) for p in sorted((app / "policies").glob("*.rego"))]
        self.data = yaml.safe_load((app.parent / "assets/static-authz.yaml").read_text())
        self.fixtures = {test.path.name: (test.package, test.fixtures) for test in discover(app)}

    def interpreter(self, test: PolicyTest, data: dict, input_doc: object) -> Interpreter:
        interpreter = Interpreter()
        for name, source in self.modules:
            interpreter.add_module(name, source)
        for name, (_, source) in self.fixtures.items():
            interpreter.add_module(name, source)
        interpreter.add_data(data)
        # Input(dict) uses ctypes string conversion that truncates embedded NULs.
        interpreter.set_input_term(json.dumps(input_doc, allow_nan=False))
        return interpreter

    @staticmethod
    def query(interpreter: Interpreter, expression: str, bindings: dict) -> list[dict]:
        prefix = "; ".join(f"{name} := {value.term}" for name, value in bindings.items())
        query = f"{prefix}; {expression}" if prefix else expression
        output = interpreter.query(f"{query}; compat_success := true")
        # Never stringify/repr unsuccessful native Output objects: regopy 1.5.2
        # can segfault in rego_output_string, including during pytest reporting.
        if not output.ok():
            raise RuntimeError(f"regopy query failed: {rego_get_error(interpreter._impl)}; query={expression}")
        if str(output) == "undefined":
            return []
        return [
            {key: binding(output.binding(key, index)) for key in output[index].bindings if key != "compat_success"}
            for index in range(len(output))
        ]

    def run(self, test: PolicyTest) -> None:
        if re.search(r"\bwith\b", re.sub(r'"(?:\\.|[^"\\])*"', "", test.fixtures)):
            raise ValueError(f"Unsupported with in fixture helper: {test.path}")
        aliases = {}
        for _, source in self.modules:
            package = re.search(r"^package ([\w.]+)", source, re.MULTILINE)
            if package and package[1] == test.package:
                for name in re.findall(r"^(\w+)\s*(?:\(|:=|if\b|contains\b)", source, re.MULTILINE):
                    aliases[name] = f"data.{test.package}.{name}"
        for path, alias in re.findall(r"^import (data\.[\w.]+)(?: as (\w+))?", test.fixtures, re.MULTILINE):
            aliases[alias or path.split(".")[-1]] = path
        for package, source in self.fixtures.values():
            if package == test.package:
                for name in re.findall(r"^(\w+)\s*:=", source, re.MULTILINE):
                    aliases[name] = f"data.{test.package}.{name}"
        # Test-local variables shadow package fixture documents.
        for name in re.findall(r"\b(\w+)\s*:=", test.body):
            aliases.pop(name, None)

        def qualify(source: str) -> str:
            def replace(match):
                text = match[0]
                if text.startswith('"'):
                    return text
                head, *tail = text.split(".")
                return ".".join([aliases.get(head, head), *tail])

            return re.sub(r'"(?:\\.|[^"\\])*"|\b[a-zA-Z_]\w*(?:\.\w+)*', replace, source)

        base = self.interpreter(test, self.data, {})
        states = [{}]
        expressions = statements(test.body)
        if not expressions:
            raise ValueError(f"Empty test body: {test.id}")
        for original in expressions:
            statement = qualify(original)
            positions = set(boundaries(statement))
            modifiers = [m for m in re.finditer(r"\bwith\s+([\w.]+)\s+as\s+", statement) if m.start() in positions]
            expression = statement[: modifiers[0].start()].strip() if modifiers else statement
            if re.search(r"\bwith\b", re.sub(r'"(?:\\.|[^"\\])*"', "", expression)):
                raise ValueError(f"Nested with is unsupported: {statement}")
            next_states = []
            for bindings in states:
                data = copy.deepcopy(self.data)
                input_doc = {}
                for index, modifier in enumerate(modifiers):
                    end = modifiers[index + 1].start() if index + 1 < len(modifiers) else len(statement)
                    rhs = statement[modifier.end() : end].strip()
                    values = self.query(base, f"compat_value := {rhs}", bindings)
                    if len(values) != 1 or "compat_value" not in values[0]:
                        raise ValueError(f"With replacement must have one defined value: {rhs}")
                    replacement = values[0]["compat_value"]
                    if not replacement.json_compatible:
                        raise ValueError(f"With replacement must be JSON-compatible: {rhs}")
                    value = replacement.value
                    target = modifier[1]
                    if target == "input":
                        input_doc = value
                    elif target.startswith("data.authz.") and target.split(".")[2] in {
                        "roles",
                        "principals",
                        "endpoints",
                        "workspaces",
                        "config",
                        "scope_families",
                    }:
                        parts = target.split(".")[1:]
                        node = data
                        for part in parts[:-1]:
                            node = node.setdefault(part, {})
                        node[parts[-1]] = value
                    else:
                        raise ValueError(f"Unsupported with target: {target}")
                interpreter = self.interpreter(test, data, input_doc) if modifiers else base
                rows = self.query(interpreter, expression, bindings)
                if test.parameter and not rows:
                    raise AssertionError(f"{test.id}: parameterized case {bindings} failed: {statement}")
                next_states.extend(rows)
            states = next_states
            if not states:
                raise AssertionError(f"{test.id}: assertion undefined/false: {statement}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="", help="Run tests whose ID contains this string")
    args = parser.parse_args()
    runner = CorpusRunner()
    tests = [test for test in discover() if args.match in test.id]
    if not tests:
        parser.error("No matching tests")
    for test in tests:
        runner.run(test)
        print(f"PASS {test.id}", flush=True)
    print(f"Passed {len(tests)} Rego policy tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
