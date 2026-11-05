# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure authorization uses the reviewed published wheel, not an unrelated binding."""

import importlib
import importlib.metadata
from pathlib import Path

from nhx.core.auth.app.embedded_pdp import rego_engine


def test_engine_comes_from_pinned_gordon_distribution():
    distribution = importlib.metadata.distribution("gordon-to-regorus")
    assert distribution.version == "0.13.0"
    assert distribution.files is not None
    installed_files = {Path(str(distribution.locate_file(file))).resolve() for file in distribution.files}
    regorus = importlib.import_module("regorus")
    assert regorus.__file__ is not None
    assert Path(regorus.__file__).resolve() in installed_files
    assert getattr(rego_engine, "Engine") is regorus.Engine
