# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline regression checks for tutorial examples; no jobs or downloads."""

import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from nemo_safe_synthesizer_plugin.sdk.job_builder import SafeSynthesizerJobBuilder

TUTORIALS = Path(__file__).resolve().parents[2] / "docs/safe-synthesizer/tutorials"


def python_block_after(page: str, heading: str) -> str:
    section = (TUTORIALS / page).read_text().split(heading, 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match is not None
    return match[1]


def test_small_table_example_is_pii_only() -> None:
    source = python_block_after("safe-synthesizer-101.mdx", '**Error: "Dataset must have at least 200')
    namespace = {
        "SafeSynthesizerJobBuilder": SafeSynthesizerJobBuilder,
        "client": Mock(),
        "df": pd.DataFrame({"name": ["Example"] * 150}),
        "provider_name": "system/example",
    }
    exec(compile(source, "<pii-only-tutorial>", "exec"), namespace)
    builder = namespace["builder"]
    # Bypass uploading: inspect the real builder's spec without any API calls.
    builder._data_source_path = "default/example#input.csv"
    config = builder._build_job_spec()["config"]
    assert config["enable_synthesis"] is False
    assert config["enable_replace_pii"] is True
    assert config["replace_pii"]["globals"]["classify"]["classify_model_provider"] == "system/example"
    namespace["client"].assert_not_called()
    assert namespace["client"].mock_calls == []


def test_delta_uses_final_sample_and_documented_exponent(monkeypatch: pytest.MonkeyPatch) -> None:
    source = python_block_after("differential-privacy.mdx", "## Load and Prepare Data")
    download = Mock(return_value="/unused")
    monkeypatch.setitem(sys.modules, "kagglehub", Mock(dataset_download=download))
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({"row": range(30000)}))
    namespace = {"pd": pd}
    exec(compile(source, "<delta-tutorial>", "exec"), namespace)
    assert len(namespace["df"]) == 26250
    assert namespace["recommended_delta"] == pytest.approx(1 / 26250**1.2)
