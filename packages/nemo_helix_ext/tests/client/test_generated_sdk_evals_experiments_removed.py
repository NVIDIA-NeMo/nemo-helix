# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "nemo_helix.resources.evaluations",
        "nemo_helix.types.evaluations",
        "nemo_helix.resources.experiments",
        "nemo_helix.types.experiments",
    ],
)
def test_generated_evals_experiments_sdk_modules_are_removed(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None
