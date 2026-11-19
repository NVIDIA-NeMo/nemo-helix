# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved preview artifacts through the in-process Data Designer CLI."""

import csv
from pathlib import Path

import nemo_data_designer_plugin.testing.utils as u
import pytest
from nhx.testing import ClientContext

pytestmark = pytest.mark.integration

SAMPLE_VALUE = 'café, "quoted"\nsecond line'


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return u.write_config_file(
        tmp_path,
        f"""
import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(
        dd.SamplerColumnConfig(
            name="text",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[{SAMPLE_VALUE!r}]),
        )
    )
    return builder
""",
    )


def test_save_preview_exports_csv_and_reports_its_path(
    tmp_path: Path, config_path: Path, client_context: ClientContext
) -> None:
    artifact_path = tmp_path / "saved"
    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "3", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code == 0, result.output
    results_dir = u.find_single_preview_results_dir(artifact_path)
    csv_path = results_dir / "dataset.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == ["text"]
        records = list(reader)

    assert records == [{"text": SAMPLE_VALUE}] * 3
    assert records == u.read_saved_preview_dataset(results_dir).to_dict(orient="records")
    assert "CSV path:" in result.output
    # Rich can wrap a long path across multiple terminal lines.
    assert "".join(str(csv_path).split()) in "".join(result.output.split())
    assert "Preview complete" in result.output


def test_save_preview_rejects_a_file_as_artifact_directory(
    tmp_path: Path, config_path: Path, client_context: ClientContext
) -> None:
    artifact_path = tmp_path / "not-a-directory"
    artifact_path.write_text("keep this file", encoding="utf-8")
    result = u.invoke_cli(
        ["preview", str(config_path), "--num-records", "1", "--save-results", "--artifact-path", str(artifact_path)],
        client_context,
    )

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, NotADirectoryError)
    assert "Not a directory" in str(result.exception)
    assert "CSV path:" not in result.output
    assert "Results path:" not in result.output
    assert artifact_path.read_text(encoding="utf-8") == "keep this file"
