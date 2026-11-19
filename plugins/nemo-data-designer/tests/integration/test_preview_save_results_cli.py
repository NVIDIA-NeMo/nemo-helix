# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved preview artifacts through ``nemo data-designer preview --save-results``."""

from pathlib import Path

import nemo_data_designer_plugin.testing.utils as u
import pandas as pd
import pytest
from nhx.testing import ClientContext

pytestmark = pytest.mark.integration


def _write_preview_config(tmp_path: Path) -> Path:
    return u.write_config_file(
        tmp_path,
        f"""
import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[
        dd.ModelConfig(
            alias="text", model={u.ENABLED_MODEL_NAME!r},
            provider={u.OPEN_PROVIDER_NAME!r},
        )
    ])
    builder.add_column(dd.SamplerColumnConfig(
        name="label",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=['Hello, "reader"']),
    ))
    builder.add_column(dd.SamplerColumnConfig(
        name="status",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["ready"]),
    ))
    return builder
""",
    )


def test_preview_saves_csv_alongside_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_context: ClientContext
) -> None:
    config_path = _write_preview_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    artifact_path = Path("saved")

    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "3", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code == 0, result.output
    results_dir = u.find_single_preview_results_dir(artifact_path)
    csv_path = results_dir / "dataset.csv"
    assert csv_path.is_file()
    csv_dataset = pd.read_csv(csv_path)
    parquet_dataset = u.read_saved_preview_dataset(results_dir)
    expected = pd.DataFrame({"label": ['Hello, "reader"'] * 3, "status": ["ready"] * 3})
    pd.testing.assert_frame_equal(csv_dataset, expected)
    pd.testing.assert_frame_equal(csv_dataset, parquet_dataset, check_dtype=False)
    assert f"CSV path: {csv_path}" in result.output
    assert "Preview complete" in result.output


def test_preview_save_results_reports_blocked_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_context: ClientContext
) -> None:
    config_path = _write_preview_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    artifact_path = Path("blocked")
    artifact_path.write_text("This is a file, not a directory.", encoding="utf-8")

    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "3", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code == 1, result.output
    assert result.exception is not None
    assert "blocked" in str(result.exception)
    assert "CSV path:" not in result.output
    assert "Results path:" not in result.output
    assert artifact_path.read_text(encoding="utf-8") == "This is a file, not a directory."
    assert not list(tmp_path.rglob("dataset.csv"))
    assert not list(tmp_path.rglob("dataset.parquet"))
