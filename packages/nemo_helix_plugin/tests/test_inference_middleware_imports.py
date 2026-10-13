# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import boundary tests for the inference middleware public base interface."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_importing_middleware_base_does_not_load_heavy_sdk_modules() -> None:
    code = textwrap.dedent(
        """
        import sys

        from nemo_helix_plugin.inference_middleware import NemoInferenceMiddleware

        assert NemoInferenceMiddleware.__name__ == "NemoInferenceMiddleware"
        loaded = {"openai", "anthropic", "pydantic"} & set(sys.modules)
        assert loaded == set(), loaded
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
