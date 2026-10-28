# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Dockerfile pin validator with both architecture markers."""

import importlib.metadata
from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging import markers

ROOT = Path(__file__).resolve().parents[2]


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
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    installed: str | None,
    error: type[Exception] | None,
) -> None:
    dockerfile = (ROOT / "docker/Dockerfile.nhx-safe-synthesizer-tasks").read_text()
    validator = dockerfile.split("# Fail the image build if a later sync changes the pinned training versions.", 1)[1]
    validator = validator.split("RUN python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    constraints = (ROOT / "plugins/nemo-safe-synthesizer/constraints.txt").read_text()
    torchao = next(line for line in constraints.splitlines() if line.startswith("torchao=="))

    environment = {**markers.default_environment(), "platform_machine": machine}
    monkeypatch.setattr(markers, "default_environment", lambda: environment.copy())
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: torchao if path.name == "safe-synthesizer-constraints.txt" else "",
    )
    lookup = Mock(return_value=installed)
    if installed is None:
        lookup.side_effect = importlib.metadata.PackageNotFoundError("torchao")
    monkeypatch.setattr(importlib.metadata, "version", lookup)

    if error is None:
        exec(compile(validator, "<dockerfile-pin-validator>", "exec"), {})
    else:
        with pytest.raises(error):
            exec(compile(validator, "<dockerfile-pin-validator>", "exec"), {})

    if machine == "aarch64":
        lookup.assert_not_called()
    else:
        lookup.assert_called_once_with("torchao")
