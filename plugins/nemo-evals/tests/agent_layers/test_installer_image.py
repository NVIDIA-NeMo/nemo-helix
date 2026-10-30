# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent installer image installs the Harbor and NeMo Fabric the workspace locks, so layers install as trials do."""

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCKERFILE = _ROOT / "docker" / "Dockerfile.nhx-agent-installer"


def _locked(package: str) -> str:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    (version,) = {entry["version"] for entry in lock["package"] if entry["name"] == package}
    return version


@pytest.mark.parametrize(
    ("argument", "package"), [("HARBOR_VERSION", "harbor"), ("NEMO_FABRIC_VERSION", "nemo-fabric")]
)
def test_the_installer_pins_what_uv_lock_locks(argument: str, package: str) -> None:
    match = re.search(rf"^ARG {argument}=(\S+)$", _DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)

    assert match is not None, f"{_DOCKERFILE.name} has no ARG {argument}"
    assert match[1] == _locked(package), f"bump ARG {argument} in {_DOCKERFILE.name} to uv.lock's {package}"
