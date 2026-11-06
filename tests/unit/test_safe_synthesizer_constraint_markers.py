# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Dockerfile pin validator with both architecture markers."""

import builtins
import importlib.metadata
from collections.abc import Mapping, Sequence, Set
from pathlib import Path, PurePath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.markers import EvaluateContext, Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def _exec_validator(validator: str, *, machine: str, torchao: str, version: Mock) -> None:
    """Run the validator with its imports resolved to fakes, inside the validator only."""

    class _MarkerOnMachine(Marker):
        def evaluate(
            self,
            environment: Mapping[str, str | Set[str]] | None = None,
            context: EvaluateContext = "metadata",
        ) -> bool:
            return super().evaluate({**(environment or {}), "platform_machine": machine}, context)

    def requirement(line: str) -> Requirement:
        parsed = Requirement(line)
        if parsed.marker is not None:
            parsed.marker = _MarkerOnMachine(str(parsed.marker))
        return parsed

    class _ConstraintsPath:
        def __init__(self, filename: str) -> None:
            self.name = PurePath(filename).name

        def read_text(self) -> str:
            return torchao if self.name == "safe-synthesizer-constraints.txt" else ""

    fakes = {
        "importlib.metadata": SimpleNamespace(version=version),
        "pathlib": SimpleNamespace(Path=_ConstraintsPath),
        "packaging.requirements": SimpleNamespace(Requirement=requirement),
    }
    real_import = builtins.__import__

    def validator_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name in fakes:
            return fakes[name]
        return real_import(name, globals, locals, fromlist, level)

    code = compile(validator, "<dockerfile-pin-validator>", "exec")
    exec(code, {"__builtins__": {**vars(builtins), "__import__": validator_import}})


@pytest.mark.parametrize(
    ("machine", "installed", "error"),
    [
        ("aarch64", None, None),
        ("x86_64", "0.17.0+cu129", None),
        ("x86_64", None, importlib.metadata.PackageNotFoundError),
        ("x86_64", "0.16.0+cu129", RuntimeError),
    ],
)
def test_torchao_constraint_validation(
    machine: str,
    installed: str | None,
    error: type[Exception] | None,
) -> None:
    dockerfile = (ROOT / "docker/Dockerfile.nhx-safe-synthesizer-tasks").read_text()
    validator = dockerfile.split("# Fail the image build if a later sync changes the pinned training versions.", 1)[1]
    validator = validator.split("RUN python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    constraints = (ROOT / "plugins/nemo-safe-synthesizer/constraints.txt").read_text()
    torchao = next(line for line in constraints.splitlines() if line.startswith("torchao=="))

    lookup = Mock(return_value=installed)
    if installed is None:
        lookup.side_effect = importlib.metadata.PackageNotFoundError("torchao")

    if error is None:
        _exec_validator(validator, machine=machine, torchao=torchao, version=lookup)
    else:
        with pytest.raises(error):
            _exec_validator(validator, machine=machine, torchao=torchao, version=lookup)

    if machine == "aarch64":
        lookup.assert_not_called()
    else:
        lookup.assert_called_once_with("torchao")
