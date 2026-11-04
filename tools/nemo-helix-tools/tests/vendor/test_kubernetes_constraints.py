# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep release wheel dependencies compatible with Jobs' deserialize API."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("services/core/jobs/pyproject.toml", None),
        ("packages/nhx_common/pyproject.toml", None),
        ("packages/nemo_helix/pyproject.toml", "jobs-service"),
        ("packages/nemo_helix/pyproject.toml", "nhx-common"),
    ],
)
def test_kubernetes_dependency_excludes_incompatible_deserialize_api(path: str, extra: str | None) -> None:
    project = tomllib.loads((ROOT / path).read_text())["project"]
    dependencies = project["optional-dependencies"][extra] if extra else project["dependencies"]
    requirement = next(Requirement(dep) for dep in dependencies if Requirement(dep).name == "kubernetes")
    assert "30.1.0" in requirement.specifier
    assert "36.0.3" in requirement.specifier
    assert "37.0.0" not in requirement.specifier
    assert "37.0.1" not in requirement.specifier
    assert "38.0.0" not in requirement.specifier
