# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPO environment package validation for each supported format."""

from pathlib import Path

import pytest
import yaml
from nhx.rl.tasks.environment.allowlist import IMAGE_ADAPTER_ALLOWLIST
from nhx.rl.tasks.environment.validate import (
    MANIFEST_FILENAME,
    EnvironmentPackageValidationError,
    load_manifest,
    validate_package_layout,
)

NATIVE_CONFIG = "responses_api_models/vllm_model/configs/policy_model.yaml"
CONFIGS_CONFIG = "configs/policy_model.yaml"
ADAPTER_AGENT = sorted(IMAGE_ADAPTER_ALLOWLIST)[0]


def _write_package(root: Path, manifest: dict, files: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return root


def _validate(root: Path) -> None:
    validate_package_layout(root, load_manifest(root))


def _native(**overrides: object) -> dict:
    return {"format": "native-v1", "config_paths": [NATIVE_CONFIG], "metadata": {"name": "env"}} | overrides


def _wheels(**overrides: object) -> dict:
    return {"format": "wheels-v1", "config_paths": [CONFIGS_CONFIG], "metadata": {"name": "env"}} | overrides


def _adapter(**overrides: object) -> dict:
    return {
        "format": "adapter-wheels-v1",
        "config_paths": [CONFIGS_CONFIG],
        "metadata": {"name": "env"},
        "adapter": {"agent": ADAPTER_AGENT},
    } | overrides


@pytest.mark.parametrize(
    ("manifest", "files"),
    [
        pytest.param(_native(), [NATIVE_CONFIG], id="native-v1"),
        pytest.param(_wheels(), [CONFIGS_CONFIG, "wheels/env-0.1.0-py3-none-any.whl"], id="wheels-v1"),
        pytest.param(_adapter(), [CONFIGS_CONFIG, "wheels/env-0.1.0-py3-none-any.whl"], id="adapter-wheels-v1"),
    ],
)
def test_well_formed_package_validates(tmp_path: Path, manifest: dict, files: list[str]) -> None:
    _validate(_write_package(tmp_path / "env", manifest, files))


@pytest.mark.parametrize(
    ("manifest", "files", "error"),
    [
        pytest.param(
            _native(config_paths=[CONFIGS_CONFIG]),
            [CONFIGS_CONFIG],
            "native-v1 config_paths must be under",
            id="native-v1-config-outside-gym-server",
        ),
        pytest.param(
            _native(),
            [],
            "config_paths reference files that are not in the package",
            id="native-v1-missing-config",
        ),
        pytest.param(
            _wheels(),
            [CONFIGS_CONFIG],
            "must carry a non-empty wheels/ directory",
            id="wheels-v1-without-wheels",
        ),
        pytest.param(
            _wheels(),
            [CONFIGS_CONFIG, "wheels/env-0.1.0.tar.gz"],
            "Non-wheel files in wheels/",
            id="wheels-v1-non-wheel-in-wheels",
        ),
        pytest.param(
            _adapter(adapter={"agent": "not-in-the-training-image"}),
            [CONFIGS_CONFIG, "wheels/env-0.1.0-py3-none-any.whl"],
            "is not built into the training image",
            id="adapter-wheels-v1-unknown-agent",
        ),
        pytest.param(
            _adapter(config_paths=[NATIVE_CONFIG]),
            [NATIVE_CONFIG, "wheels/env-0.1.0-py3-none-any.whl"],
            "should live under configs/",
            id="adapter-wheels-v1-config-outside-configs",
        ),
        pytest.param(
            _native(),
            [NATIVE_CONFIG, "data/train.jsonl"],
            "Prompt JSONL must not live in the environment package",
            id="dataset-rows-in-package",
        ),
    ],
)
def test_malformed_package_is_rejected(tmp_path: Path, manifest: dict, files: list[str], error: str) -> None:
    root = _write_package(tmp_path / "env", manifest, files)
    with pytest.raises(EnvironmentPackageValidationError, match=error):
        _validate(root)
