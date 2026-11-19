# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved preview artifacts and save failures through the public CLI."""

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
        dd.ModelConfig(alias="text", model={u.ENABLED_MODEL_NAME!r},
                       provider={u.OPEN_PROVIDER_NAME!r})
    ])
    builder.add_column(dd.SamplerColumnConfig(
        name="message",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=['He said "hello", café']),
    ))
    return builder
""",
    )


def test_preview_saves_csv_next_to_parquet(tmp_path: Path, client_context: ClientContext) -> None:
    config_path = _write_preview_config(tmp_path)
    artifact_path = tmp_path / "saved preview"

    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "3", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code == 0, result.output
    results_dir = u.find_single_preview_results_dir(artifact_path)
    csv_path = results_dir / "dataset.csv"
    csv_dataset = pd.read_csv(csv_path).convert_dtypes(dtype_backend="pyarrow")
    parquet_dataset = u.read_saved_preview_dataset(results_dir)
    expected = pd.DataFrame({"message": ['He said "hello", café'] * 3}).convert_dtypes(dtype_backend="pyarrow")
    pd.testing.assert_frame_equal(csv_dataset, expected)
    pd.testing.assert_frame_equal(csv_dataset, parquet_dataset, check_dtype=False)
    # Rich may wrap the path, so compare without display whitespace.
    output = "".join(result.output.split())
    assert "CSVpath:" + "".join(str(csv_path).split()) in output
    assert "Preview complete" in result.output


def test_preview_save_failure_is_reported(tmp_path: Path, client_context: ClientContext) -> None:
    config_path = _write_preview_config(tmp_path)
    artifact_path = tmp_path / "blocked"
    artifact_path.write_text("existing file", encoding="utf-8")

    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "3", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, NotADirectoryError)
    assert "Not a directory" in str(result.exception)
    assert "CSV path:" not in result.output
    assert artifact_path.read_text(encoding="utf-8") == "existing file"
    assert not list(tmp_path.rglob("dataset.csv"))
    assert not list(tmp_path.rglob("dataset.parquet"))
