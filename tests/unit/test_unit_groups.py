# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that unit groups neither drop nor duplicate collected tests, nor import other groups."""

import json

import pytest

pytest_plugins = ["pytester"]


def _collect(pytester: pytest.Pytester, *args: str) -> tuple[list[str], set[str]]:
    imports = pytester.path / "imported.txt"
    imports.write_text("")
    assert pytester.runpytest("--collect-only", "-q", *args).ret == pytest.ExitCode.OK
    return json.loads((pytester.path / "selected.json").read_text()), set(imports.read_text().split())


def test_groups_partition_collection_without_importing_other_groups(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\naddopts = --import-mode=importlib\n")
    pytester.makeconftest(
        "import json\n"
        "from pathlib import Path\n"
        "pytest_plugins = ['tests.unit_groups']\n"
        "def pytest_collection_finish(session):\n"
        "    Path('selected.json').write_text(json.dumps([i.nodeid for i in session.items]))\n"
    )
    # Unique probe names: in-process pytester shares sys.modules with the outer suite.
    for n, directory in enumerate(4 * ["services/core/jobs", "plugins/nemo-evals", "packages/nhx_common"]):
        path = pytester.path / directory / "tests" / f"test_unit_group_probe_{n}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"open({str(pytester.path / 'imported.txt')!r}, 'a').write('{path.stem} ')\n"
            "def test_a(): pass\ndef test_b(): pass\n"
        )

    baseline, all_imported = _collect(pytester)
    groups = [_collect(pytester, f"--unit-group={group}") for group in ("core", "plugins", "misc")]

    assert len(baseline) == 24 and len(all_imported) == 12
    assert sorted(sum((items for items, _ in groups), [])) == sorted(baseline)
    for items, imported in groups:
        assert len(items) == 8
        assert imported == {item.split("::")[0].rsplit("/", 1)[-1].removesuffix(".py") for item in items}


def test_unknown_group_is_a_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("pytest_plugins = ['tests.unit_groups']\n")
    assert pytester.runpytest("--unit-group=other").ret == pytest.ExitCode.USAGE_ERROR
