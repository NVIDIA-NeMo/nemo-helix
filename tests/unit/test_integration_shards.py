# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real pytest collection with complementary integration shards."""

import json
from pathlib import Path

import pytest

from tests.integration_shards import integration_shard

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("services/core/models/tests/integration/test_api.py", "services"),
        ("services/intake/tests/test_ingest.py", "services"),
        ("services/new-service/tests/test_api.py", "services"),
        ("services/core/secrets/tests/integration/test_api.py", "platform"),
        ("services/core/secrets-other/tests/test_api.py", "services"),
        ("plugins/nemo-evals/tests/integration/test_job.py", "platform"),
        ("packages/nemo_helix_ext/tests/local/test_daemon_lifecycle.py", "platform"),
        ("tests/auth/integration/test_jobs_auth.py", "platform"),
        ("agents/new-agent/tests/test_api.py", "platform"),
    ],
)
def test_package_assignment(path: str, expected: str) -> None:
    assert integration_shard(Path(path)) == expected


@pytest.fixture
def shard_suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeini(
        "[pytest]\naddopts = --import-mode=importlib\n"
        "markers =\n    integration: integration test\n    xdist_group: resource group\n"
    )
    pytester.makeconftest(
        "import json\n"
        "from pathlib import Path\n"
        "pytest_plugins = ['tests.integration_shards']\n"
        "def pytest_collection_finish(session):\n"
        "    selected = {item.nodeid: [mark.name for mark in item.iter_markers()] for item in session.items}\n"
        "    Path('selected.json').write_text(json.dumps(selected))\n"
    )
    for directory, filename in [
        ("services/core/models/tests", "test_models.py"),
        ("services/core/secrets/tests", "test_secrets.py"),
        ("plugins/nemo-evals/tests", "test_evals.py"),
        ("tests/integration", "test_auth.py"),
    ]:
        path = pytester.path / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import pytest\n"
            "@pytest.mark.integration\n"
            "@pytest.mark.xdist_group('shared-service-pool')\n"
            "def test_integration(): pass\n"
        )
    pytester.makepyfile(test_unit="def test_unit(): pass")
    return pytester


def test_shards_are_disjoint_complete_and_preserve_markers(shard_suite: pytest.Pytester) -> None:
    selections = {}
    for shard in (None, "services", "platform"):
        args = ["-m", "integration", "--collect-only", "-q"]
        if shard:
            args.append(f"--integration-shard={shard}")
        result = shard_suite.runpytest(*args)
        assert result.ret == pytest.ExitCode.OK
        selections[shard] = json.loads((shard_suite.path / "selected.json").read_text())
    baseline = selections[None]
    services = selections["services"]
    platform = selections["platform"]
    assert len(baseline) == 4
    assert len(services) == 1
    assert len(platform) == 3
    assert services.keys().isdisjoint(platform)
    assert services | platform == baseline
    assert all("xdist_group" in markers for markers in baseline.values())


@pytest.mark.parametrize(("shard", "count"), [("services", 1), ("platform", 3)])
def test_shards_execute_with_xdist_resource_grouping(shard_suite: pytest.Pytester, shard: str, count: int) -> None:
    shard_suite.runpytest(
        "-m", "integration", f"--integration-shard={shard}", "-n", "2", "--dist", "loadgroup", "-q"
    ).assert_outcomes(passed=count)


def test_default_collection_is_unchanged(shard_suite: pytest.Pytester) -> None:
    shard_suite.runpytest("-q").assert_outcomes(passed=5)


def test_sharding_rejects_non_integration_collection(shard_suite: pytest.Pytester) -> None:
    result = shard_suite.runpytest("--integration-shard=services", "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*requires integration-only collection*"])
