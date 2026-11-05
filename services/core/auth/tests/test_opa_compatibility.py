# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OPA CLI execution of every shared Rego authorization assertion."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[4] / "script/test_opa_policies.py"
SPEC = importlib.util.spec_from_file_location("opa_policy_corpus", RUNNER_PATH)
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


def synthetic(tmp_path, source, parameter_keys=None):
    app = tmp_path / "app"
    (app / "policies").mkdir(parents=True)
    (app / "policy_tests").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/static-authz.yaml").write_text("authz: {config: {flag: false}}\n")
    (app / "policy_tests/synthetic.rego").write_text(source)
    return corpus.CorpusRunner(app, parameter_keys), corpus.discover(app)


@pytest.mark.parametrize(
    "body",
    [
        "items := {1, 2}\nitems == {1, 2}",
        'nested := {"items": {1, 2}}\nnested.items == {1, 2}',
        "items := set()\ncount(items) == 0",
        "mock := true\nmock",
        'text := "with input as # not a comment"\ntext == "with input as # not a comment"',
        "data.authz.config.flag with data.authz.config.flag as true\nnot data.authz.config.flag",
        'input.flag with input as {"flag": true}\nnot input.flag',
        r'input["key\u0000suffix"] == "value\u0000suffix" with input as {"key\u0000suffix": "value\u0000suffix"}',
        f'input.number == {2**100} with input as {{"number": {2**100}}}',
        'items := [x | x := input.x with input as {"x": 1}]\nitems == [1]',
        'helper with input as {"flag": true}',
    ],
)
def test_native_rego_values_and_with_scoping(tmp_path, body):
    evaluator, cases = synthetic(
        tmp_path,
        f"package synthetic\nmock := false\nhelper if {{ input.flag }}\ntest_case if {{\n{body}\n}}",
    )
    evaluator.run(cases[0])


@pytest.mark.parametrize("body", ["false", "1 == 2", "data.missing", "not true"])
def test_runner_rejects_false_or_undefined_assertion(tmp_path, body):
    evaluator, cases = synthetic(tmp_path, f"package synthetic\ntest_case if {{\n{body}\n}}")
    with pytest.raises(AssertionError):
        evaluator.run(cases[0])


@pytest.mark.parametrize("value", ["false", "null", "0", "{}", '"<undefined>"'])
def test_runner_requires_true_not_truthiness(tmp_path, value):
    evaluator, cases = synthetic(tmp_path, f"package synthetic\ntest_case := {value}")
    with pytest.raises(AssertionError):
        evaluator.run(cases[0])


@pytest.mark.parametrize("indent", [" ", "  ", "\t"])
def test_native_discovery_executes_indented_failures(tmp_path, indent):
    evaluator, cases = synthetic(
        tmp_path,
        f"package synthetic\ntest_passing if {{ true }}\n{indent}test_failing if {{ false }}",
    )
    assert [case.rule for case in cases] == ["data.synthetic.test_passing", "data.synthetic.test_failing"]
    evaluator.run(cases[0])
    with pytest.raises(AssertionError):
        evaluator.run(cases[1])


@pytest.mark.parametrize("expected", [False, True])
def test_runner_checks_every_parameterized_case(tmp_path, expected):
    evaluator, cases = synthetic(
        tmp_path,
        "package synthetic\ntest_case[key] if {\n"
        f'cases := {{"first": true, "second": {str(expected).lower()}}}\n'
        "some key, expected in cases\nexpected\n}",
        {"data.synthetic.test_case": {"first", "second"}},
    )
    if expected:
        assert evaluator.run(cases[0]) == 2
    else:
        with pytest.raises(AssertionError, match="failed/skipped|expected parameter keys"):
            evaluator.run(cases[0])


def test_runner_detects_new_failing_parameter_not_in_manifest(tmp_path):
    evaluator, cases = synthetic(
        tmp_path,
        "package synthetic\ntest_case[key] if {\n"
        'cases := {"old": true, "new": false}\n'
        "some key, expected in cases\nexpected\n}",
        {"data.synthetic.test_case": {"old"}},
    )
    with pytest.raises(ValueError, match="do not match cases table"):
        evaluator.run(cases[0])


def test_runner_requires_parameter_manifest(tmp_path):
    evaluator, cases = synthetic(tmp_path, 'package synthetic\ntest_case[key] if { key := "a" }', {})
    with pytest.raises(ValueError, match="exhaustive expected parameter keys"):
        evaluator.run(cases[0])


@pytest.mark.parametrize(
    "source",
    [
        "package synthetic\n",
        'package synthetic\nfixture := "test_not_a_rule if { true }"\n',
        "package synthetic\n# test_not_a_rule if { true }\n",
    ],
)
def test_native_discovery_does_not_invent_tests(tmp_path, source):
    with pytest.raises(ValueError, match="No policy tests"):
        synthetic(tmp_path, source)


@pytest.mark.parametrize("indent", ["", "  "])
def test_runner_rejects_unsupported_function_tests(tmp_path, indent):
    with pytest.raises(ValueError, match="Unsupported test declaration"):
        synthetic(tmp_path, f"package synthetic\n{indent}test_case(x) if {{ true }}")


def test_corpus_parameter_manifests_match_discovery():
    discovered = {case.rule: case for case in corpus.discover() if case.parameterized}
    assert set(discovered) == set(corpus.PARAMETER_KEYS)
    for rule, case in discovered.items():
        assert case.parameter_domain is not None
        assert case.parameter_domain == corpus.PARAMETER_KEYS[rule]


def test_complete_corpus_has_positive_rule_and_parameter_case_counts(runner):
    rules, cases = runner.run_all()
    assert rules == len(corpus.discover()) > 0
    assert cases == sum(len(corpus.PARAMETER_KEYS[test.rule]) if test.parameterized else 1 for test in runner.tests)
    assert cases > rules


@pytest.mark.parametrize("name", ["todo_case", "todo_test_case"])
def test_runner_forbids_skipped_tests(tmp_path, name):
    with pytest.raises(ValueError, match="Skipped policy test is forbidden"):
        synthetic(tmp_path, f"package synthetic\n{name} if {{ true }}")


@pytest.mark.parametrize("report", [[], None, {}, [{"package": "data.synthetic", "name": "test_missing"}]])
def test_runner_rejects_empty_or_incomplete_reports(tmp_path, monkeypatch, report):
    evaluator, cases = synthetic(tmp_path, "package synthetic\ntest_case if { true }")
    monkeypatch.setattr(
        corpus, "_command", lambda *args: subprocess.CompletedProcess(args, 0, stdout=json.dumps(report))
    )
    with pytest.raises(AssertionError):
        evaluator.run(cases[0])


@pytest.mark.parametrize("flag", ["fail", "error", "skip"])
def test_runner_rejects_reported_failure_even_with_zero_exit(tmp_path, monkeypatch, flag):
    evaluator, cases = synthetic(tmp_path, "package synthetic\ntest_case if { true }")
    report = [{"package": "data.synthetic", "name": "test_case", flag: True}]
    monkeypatch.setattr(
        corpus, "_command", lambda *args: subprocess.CompletedProcess(args, 0, stdout=json.dumps(report))
    )
    with pytest.raises(AssertionError, match="failed/skipped"):
        evaluator.run_all()


def test_runner_rejects_nonzero_exit_despite_passing_report(tmp_path, monkeypatch):
    evaluator, _ = synthetic(tmp_path, "package synthetic\ntest_case if { true }")
    report = [{"package": "data.synthetic", "name": "test_case"}]
    monkeypatch.setattr(
        corpus, "_command", lambda *args: subprocess.CompletedProcess(args, 2, stdout=json.dumps(report))
    )
    with pytest.raises(AssertionError, match="exited with status"):
        evaluator.run_all()


def test_runner_rejects_empty_parameter_domain(tmp_path):
    evaluator, cases = synthetic(
        tmp_path,
        "package synthetic\ntest_case[key] if { cases := {}; some key, expected in cases; expected }",
        {"data.synthetic.test_case": set()},
    )
    with pytest.raises(AssertionError):
        evaluator.run(cases[0])


def test_match_cannot_pass_without_tests(runner):
    with pytest.raises(ValueError, match="No matching tests"):
        runner.run_all("does-not-exist")
