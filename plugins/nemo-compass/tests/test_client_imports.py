# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys

import pytest


def test_typed_client_import_does_not_load_analysis_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in [
        "nemo_compass_plugin.client",
        "nemo_compass_plugin.endpoints",
        "nemo_compass_plugin.types",
        "nemo_compass_plugin.analysis.run",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("nemo_compass_plugin.client")

    assert "nemo_compass_plugin.analysis.run" not in sys.modules
