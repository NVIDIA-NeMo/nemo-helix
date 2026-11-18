# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every ``nemo.jobs`` entry-point key must be ``<plugin>.<NemoJob.name>``.

The job's ``name`` is user-facing: it is the submit command (``nemo <plugin> <name>``), the default
route segment (``/jobs/<name>``) and, with the plugin, the job type that ``nemo <plugin> list-jobs
--type`` and ``nemo jobs list --type`` accept. The key's plugin prefix decides which ``nemo
<plugin>`` group the job joins. A key whose suffix differs from ``name`` would give one job two
names, so this reads every manifest that declares jobs and checks each key against its class.

Manifests are read as data rather than through ``importlib.metadata``: the bundled ``nemo-helix``
wrapper re-declares plugin entry points under the same keys, and ``entry_points()`` keeps only one
of two same-named entries, so a bad key in either copy could hide behind the other.
"""

import importlib
import importlib.util
import tomllib
from pathlib import Path

import pytest
from nemo_helix_plugin.job import NemoJob

ROOT = Path(__file__).resolve().parents[2]

_MANIFESTS = sorted(
    path
    for pattern in ("plugins/*/pyproject.toml", "packages/*/pyproject.toml", "services/*/pyproject.toml")
    for path in ROOT.glob(pattern)
)


def _job_entry_points() -> list[tuple[str, str, str]]:
    entries = []
    for manifest in _MANIFESTS:
        with manifest.open("rb") as file:
            project = tomllib.load(file).get("project", {})
        for key, target in project.get("entry-points", {}).get("nemo.jobs", {}).items():
            entries.append((str(manifest.relative_to(ROOT)), key, target))
    return entries


_ENTRIES = _job_entry_points()


def test_job_entry_points_are_found() -> None:
    assert any(key == "data-designer.create" for _, key, _ in _ENTRIES)
    assert any(manifest == "packages/nemo_helix/pyproject.toml" for manifest, _, _ in _ENTRIES)


@pytest.mark.parametrize(
    ("manifest", "key", "target"), _ENTRIES, ids=[f"{manifest}:{key}" for manifest, key, _ in _ENTRIES]
)
def test_job_entry_point_key_is_plugin_and_job_name(manifest: str, key: str, target: str) -> None:
    module_name, _, attribute = target.partition(":")
    if importlib.util.find_spec(module_name.partition(".")[0]) is None:
        pytest.skip(f"{module_name} is not installed")
    value = getattr(importlib.import_module(module_name), attribute)
    if not (isinstance(value, type) and issubclass(value, NemoJob)):
        return  # Post-run actions and other non-job values share the group; they have no job name.

    plugin, _, job_name = key.partition(".")
    assert plugin, f"{manifest}: {key!r} has no '<plugin>.' prefix"
    assert job_name == value.name, (
        f"{manifest}: {key!r} registers {target} whose name is {value.name!r}; "
        f"register it as '{plugin}.{value.name}' or rename the job"
    )
