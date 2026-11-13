# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that test shards neither drop nor duplicate collected tests, nor import other shards."""

import json

import pytest

pytest_plugins = ["pytester"]


def _collect(pytester: pytest.Pytester, *args: str) -> tuple[list[str], set[str]]:
    imports = pytester.path / "imported.txt"
    imports.write_text("")
    assert pytester.runpytest("--collect-only", "-q", *args).ret == pytest.ExitCode.OK
    return json.loads((pytester.path / "selected.json").read_text()), set(imports.read_text().split())


def test_shards_partition_collection_without_importing_other_shards(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\naddopts = --import-mode=importlib\n")
    pytester.makeconftest(
        "import json\n"
        "from pathlib import Path\n"
        "pytest_plugins = ['tests.shards']\n"
        "def pytest_collection_finish(session):\n"
        "    Path('selected.json').write_text(json.dumps([i.nodeid for i in session.items]))\n"
    )
    # Unique probe names: in-process pytester shares sys.modules with the outer suite.
    for n in range(12):
        path = pytester.path / f"pkg{n % 3}" / "tests" / f"test_shard_probe_{n}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"open({str(pytester.path / 'imported.txt')!r}, 'a').write('{path.stem} ')\n"
            "def test_a(): pass\ndef test_b(): pass\n"
        )

    baseline, all_imported = _collect(pytester)
    shards = [_collect(pytester, f"--test-shard={i}/3") for i in (1, 2, 3)]

    assert len(baseline) == 24 and len(all_imported) == 12
    assert sorted(sum((items for items, _ in shards), [])) == sorted(baseline)
    for items, imported in shards:
        assert items
        assert imported == {item.split("::")[0].rsplit("/", 1)[-1].removesuffix(".py") for item in items}


@pytest.mark.parametrize("value", ["0/3", "4/3", "x/3", "3"])
def test_invalid_shard_is_a_usage_error(pytester: pytest.Pytester, value: str) -> None:
    pytester.makeconftest("pytest_plugins = ['tests.shards']\n")
    assert pytester.runpytest(f"--test-shard={value}").ret == pytest.ExitCode.USAGE_ERROR
