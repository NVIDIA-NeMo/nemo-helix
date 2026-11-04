# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every shared Rego authorization assertion against the embedded runtime.

The test-only runner installs input/base-data `with` overrides externally because
regopy 1.5.2's native compiler fails on transitive calls under `with`. Policy
sources and expected assertions are not translated or replaced.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[4] / "script/test_regopy_policies.py"
SPEC = importlib.util.spec_from_file_location("regopy_policy_corpus", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corpus
SPEC.loader.exec_module(corpus)


@pytest.fixture(scope="module")
def runner():
    return corpus.CorpusRunner()


@pytest.mark.parametrize("case", corpus.discover(), ids=lambda case: case.id)
def test_shared_rego_policy_corpus(runner, case):
    runner.run(case)


def synthetic(tmp_path, source):
    app = tmp_path / "app"
    (app / "policies").mkdir(parents=True)
    (app / "policy_tests").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/static-authz.yaml").write_text("authz: {config: {flag: false}}\n")
    (app / "policy_tests/synthetic.rego").write_text(source)
    return corpus.CorpusRunner(app), corpus.discover(app)[0]


@pytest.mark.parametrize(
    "body",
    [
        "items := {1, 2}\nitems == {1, 2}",
        'nested := {"items": {1, 2}}\nnested.items == {1, 2}',
        "items := set()\nitems == set()",
        "mock := true\nmock",
        'text := "with input as # not a comment"\ntext == "with input as # not a comment"',
        "data.authz.config.flag with data.authz.config.flag as true\nnot data.authz.config.flag",
        'input.flag with input as {"flag": true}\nnot input.flag',
        r'input["key\u0000suffix"] == "value\u0000suffix" with input as {"key\u0000suffix": "value\u0000suffix"}',
        f'input.number == {2**100} with input as {{"number": {2**100}}}',
    ],
)
def test_runner_preserves_rego_values_and_scoping(tmp_path, body):
    evaluator, case = synthetic(tmp_path, f"package synthetic\nmock := false\ntest_case if {{\n{body}\n}}")
    evaluator.run(case)


@pytest.mark.parametrize("body", ["false", "1 == 2", "data.missing", "not true"])
def test_runner_rejects_false_or_undefined_assertion(tmp_path, body):
    evaluator, case = synthetic(tmp_path, f"package synthetic\ntest_case if {{\n{body}\n}}")
    with pytest.raises(AssertionError):
        evaluator.run(case)


@pytest.mark.parametrize("indent", [" ", "  ", "\t"])
def test_runner_discovers_and_executes_indented_failures(tmp_path, indent):
    evaluator, first = synthetic(
        tmp_path,
        f"package synthetic\ntest_passing if {{ true }}\n{indent}test_failing if {{ false }}",
    )
    cases = corpus.discover(first.path.parent.parent)
    assert [case.name for case in cases] == ["test_passing", "test_failing"]
    evaluator.run(cases[0])
    with pytest.raises(AssertionError):
        evaluator.run(cases[1])


def test_runner_checks_every_parameterized_case(tmp_path):
    evaluator, case = synthetic(
        tmp_path,
        "package synthetic\ntest_case[key] if {\n"
        'cases := {"passing": true, "failing": false}\n'
        "some key, expected in cases\nexpected\n}",
    )
    with pytest.raises(AssertionError, match="parameterized case"):
        evaluator.run(case)


def test_runner_rejects_nested_with(tmp_path):
    evaluator, case = synthetic(
        tmp_path,
        'package synthetic\ntest_case if {\nitems := [x | x := input.x with input as {"x": 1}]\n}',
    )
    with pytest.raises(ValueError, match="Nested with"):
        evaluator.run(case)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("package synthetic\n", "No policy tests"),
        ("package synthetic\ntest_case(x) if { true }", "Unsupported test declaration"),
        ("package synthetic\n  test_case(x) if { true }", "Unsupported test declaration"),
        ("package synthetic\ntest_case[key.id] if { true }", "Unsupported test declaration"),
        ("package synthetic\ntest_case if {", "Unclosed test"),
    ],
)
def test_runner_discovery_rejects_unsupported_or_missing_tests(tmp_path, source, message):
    with pytest.raises(ValueError, match=message):
        synthetic(tmp_path, source)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "Empty test body"),
        ("true with data.unrelated as true", "Unsupported with target"),
        ("true with data.authz.allow as true", "Unsupported with target"),
        ("true with input as {1, 2}", "JSON-compatible"),
    ],
)
def test_runner_rejects_unsupported_execution(tmp_path, body, message):
    evaluator, case = synthetic(tmp_path, f"package synthetic\ntest_case if {{\n{body}\n}}")
    with pytest.raises(ValueError, match=message):
        evaluator.run(case)


def test_runner_never_formats_failed_native_output(monkeypatch):
    class FailedOutput:
        def ok(self):
            return False

        def __str__(self):
            raise AssertionError("Native error output must not be formatted")

        __repr__ = __str__

    class FailedInterpreter:
        _impl = None

        def query(self, query):
            return FailedOutput()

    monkeypatch.setattr(corpus, "rego_get_error", lambda impl: "native diagnostic")
    with pytest.raises(RuntimeError, match="native diagnostic"):
        corpus.CorpusRunner.query(FailedInterpreter(), "true", {})
