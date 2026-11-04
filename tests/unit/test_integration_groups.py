# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that integration grouping neither drops nor duplicates collected tests."""

import json

import pytest

pytest_plugins = ["pytester"]


def test_groups_preserve_integration_collection(pytester: pytest.Pytester) -> None:
    pytester.makeini(
        "[pytest]\naddopts = --import-mode=importlib\n"
        "markers =\n    integration: integration test\n    xdist_group: resource group\n"
    )
    pytester.makeconftest(
        "import json\n"
        "from pathlib import Path\n"
        "pytest_plugins = ['tests.integration_groups']\n"
        "def pytest_collection_finish(session):\n"
        "    selected = {item.nodeid: [mark.name for mark in item.iter_markers()] for item in session.items}\n"
        "    Path('selected.json').write_text(json.dumps(selected))\n"
    )
    # In-process pytester shares sys.modules with the outer suite. Use unique
    # probe names so already-imported real test modules cannot replace these.
    for directory, filename in [
        ("services/core/models/tests", "test_group_probe_models.py"),
        ("services/core/secrets/tests", "test_group_probe_secrets.py"),
        ("plugins/nemo-evals/tests", "test_group_probe_evals.py"),
        ("tests/integration", "test_group_probe_auth.py"),
    ]:
        path = pytester.path / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import pytest\n"
            "@pytest.mark.integration\n"
            "@pytest.mark.xdist_group('shared-service-pool')\n"
            "def test_integration(): pass\n"
        )
    pytester.makepyfile(test_group_probe_unit="def test_unit(): pass")
    selections = {}
    for group in (None, "services", "platform"):
        args = ["-m", "integration", "--collect-only", "-q"]
        if group:
            args.append(f"--integration-group={group}")
        assert pytester.runpytest(*args).ret == pytest.ExitCode.OK
        selections[group] = json.loads((pytester.path / "selected.json").read_text())
    baseline = selections[None]
    services = selections["services"]
    platform = selections["platform"]
    assert len(baseline) == 4
    assert services and platform
    assert services.keys().isdisjoint(platform)
    assert services | platform == baseline
    assert all("xdist_group" in markers for markers in baseline.values())
