# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the shared-patch guard sees every patch API, and only refuses shared targets."""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each test patches a stdlib or third-party module or class a different way; all must be refused.
REFUSED = """
import contextlib
import time
import types
import unittest.mock as mock
from unittest.mock import patch

import httpx
import pytest

local = types.SimpleNamespace(time=time)


def noop(*args, **kwargs): ...


def test_monkeypatch_object(monkeypatch):
    monkeypatch.setattr(local.time, "sleep", noop)

def test_monkeypatch_dotted(monkeypatch):
    monkeypatch.setattr("httpx.Client.send", noop)

def test_monkeypatch_context():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", noop)

def test_monkeypatch_delattr(monkeypatch):
    monkeypatch.delattr(time, "sleep")

def test_monkeypatch_setitem_module_dict(monkeypatch):
    monkeypatch.setitem(time.__dict__, "sleep", noop)

@patch("time.sleep")
def test_patch_decorator(sleep): ...

@patch.object(httpx.AsyncClient, "send")
async def test_patch_object_async_decorator(send): ...

@patch("time.sleep")
@patch("time.monotonic")
def test_stacked_decorators(monotonic, sleep): ...

def test_patch_context_manager_alias():
    with patch("test_refused.local.time.sleep"): ...

def test_patch_dotted_fstring():
    name = "sleep"
    with patch(f"time.{name}"): ...

def test_patch_object_context_manager():
    with mock.patch.object(httpx.Client, "send"): ...

def test_patch_multiple():
    with patch.multiple(time, sleep=noop): ...

def test_patcher_start():
    patcher = patch("time.sleep")
    patcher.start()
    patcher.stop()

def test_exit_stack():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("time.sleep"))

def test_mocker(mocker):
    mocker.patch("httpx.get")

def test_mocker_object(mocker):
    mocker.patch.object(httpx.Client, "send")

def test_patch_dict_module_dict():
    with patch.dict(time.__dict__, {"sleep": noop}): ...

@pytest.fixture
def patched_clock(monkeypatch):
    monkeypatch.setattr(time, "monotonic", noop)

def test_fixture(patched_clock): ...
"""

ALLOWED = """
import os
import tempfile
import time
import types
from unittest.mock import patch

import httpx
import pytest

import faketool


def noop(*args, **kwargs): ...


def test_instance(monkeypatch):
    client = httpx.Client()
    monkeypatch.setattr(client, "send", noop)
    client.close()

def test_module_under_test_binding(monkeypatch):
    module = types.SimpleNamespace(sleep=time.sleep, time=time)
    monkeypatch.setattr(module, "sleep", noop)
    monkeypatch.setattr(module, "time", types.SimpleNamespace(sleep=noop))

def test_first_party_module(monkeypatch):
    import helpers
    monkeypatch.setattr(helpers, "VALUE", 2)
    with patch("helpers.VALUE", 3): ...

def test_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARED_PATCH_GUARD_PROBE", "1")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with patch.dict(os.environ, {"SHARED_PATCH_GUARD_PROBE": "2"}): ...

def test_third_party_tool_patching_for_the_test():
    with faketool.mock_sleep():
        assert time.sleep is not None

def test_no_op_and_created_attributes(monkeypatch):
    monkeypatch.setattr(time, "sleep", time.sleep)
    monkeypatch.setattr(httpx, "SHARED_PATCH_GUARD_PROBE", 1, raising=False)
    with patch("time.SHARED_PATCH_GUARD_PROBE", create=True): ...

def test_arguments_and_test_helpers(monkeypatch):
    import typer.testing
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(typer.testing.CliRunner, "invoke", noop)

@pytest.mark.allow_shared_patch("time.sleep")
def test_marker_names_the_patch(monkeypatch):
    monkeypatch.setattr(time, "sleep", noop)

@pytest.mark.allow_shared_patch
def test_marker_without_names_allows_any(monkeypatch):
    monkeypatch.setattr(time, "monotonic", noop)
    monkeypatch.setattr(httpx.Client, "send", noop)
"""

FAKE_TOOL = """
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def mock_sleep():
    with patch("time.sleep"):
        yield
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeini(f"[pytest]\npythonpath = {REPO_ROOT} {pytester.path} site-packages\nasyncio_mode = auto\n")
    pytester.makepyfile(test_refused=REFUSED, test_allowed=ALLOWED, helpers="VALUE = 1\n")
    # A path under site-packages makes faketool third-party, like respx patching on a test's behalf.
    tool = pytester.path / "site-packages" / "faketool.py"
    tool.parent.mkdir()
    tool.write_text(FAKE_TOOL)
    return pytester


def _run(project: pytest.Pytester, *args: str) -> pytest.RunResult:
    return project.runpytest_subprocess("-p", "tests.shared_patch_guard", "-p", "no:cacheprovider", "-rA", *args)


def test_strict_mode_refuses_every_patch_api(project: pytest.Pytester) -> None:
    result = _run(project, "--shared-patch-guard=strict", "test_refused.py")
    refused = REFUSED.count("\ndef test_") + REFUSED.count("\nasync def test_")
    result.assert_outcomes(failed=refused - 1, errors=1)
    assert result.stdout.str().count("SharedPatchError") >= refused


def test_strict_mode_allows_what_is_not_shared(project: pytest.Pytester) -> None:
    result = _run(project, "--shared-patch-guard=strict", "test_allowed.py")
    result.assert_outcomes(passed=ALLOWED.count("\ndef test_"))


def test_strict_is_the_default_mode(project: pytest.Pytester) -> None:
    result = _run(project, "test_refused.py", "-k", "decorator and not async and not stacked")
    result.assert_outcomes(failed=1)


def test_report_mode_records_without_refusing(project: pytest.Pytester) -> None:
    report = project.path / "patches.jsonl"
    result = _run(project, "--shared-patch-guard=report", f"--shared-patch-report={report}")
    result.assert_outcomes(
        passed=REFUSED.count("\ndef test_") + REFUSED.count("\nasync def test_") + ALLOWED.count("\ndef test_")
    )
    records = [json.loads(line) for line in report.read_text().splitlines()]
    refused = {record["key"] for record in records if record["decision"] == "refused"}
    assert "test_refused.py time.sleep" in refused
    assert "test_refused.py httpx.Client.send" in refused
    assert "test_refused.py httpx.AsyncClient.send" in refused
    assert {record["decision"] for record in records if "faketool" in record["requester"]} == {"tool"}
    assert {record["decision"] for record in records if record["test"] and "marker" in record["test"]} == {"marker"}


def test_integration_tests_are_reported_not_refused(project: pytest.Pytester) -> None:
    project.makeconftest(
        "import pytest\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        item.add_marker(pytest.mark.integration)\n"
    )
    project.makeini(
        f"[pytest]\npythonpath = {REPO_ROOT} {project.path} site-packages\nasyncio_mode = auto\n"
        "markers =\n    integration: integration test\n"
    )
    result = _run(project, "--shared-patch-guard=strict", "test_refused.py")
    result.assert_outcomes(passed=REFUSED.count("\ndef test_") + REFUSED.count("\nasync def test_"))


def test_first_party_packages_installed_as_a_wheel(project: pytest.Pytester) -> None:
    installed = project.path / "site-packages" / "ourwheel.py"
    installed.write_text("import time\n\ndef wait():\n    time.sleep(0)\n")
    project.makepyfile(
        test_wheel="import ourwheel\n\ndef test_patch_wheel_module(monkeypatch):\n"
        "    monkeypatch.setattr(ourwheel, 'wait', lambda: None)\n"
    )
    result = _run(project, "--shared-patch-guard=strict", "test_wheel.py")
    result.assert_outcomes(failed=1)
    result = _run(project, "--shared-patch-guard=strict", "-o", "shared_patch_first_party=ourwheel", "test_wheel.py")
    result.assert_outcomes(passed=1)


def test_off_mode_installs_nothing(project: pytest.Pytester) -> None:
    result = _run(project, "--shared-patch-guard=off", "test_refused.py")
    result.assert_outcomes(passed=REFUSED.count("\ndef test_") + REFUSED.count("\nasync def test_"))
