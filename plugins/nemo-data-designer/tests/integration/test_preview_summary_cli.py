# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the completion summary of ``nemo data-designer preview``."""

from pathlib import Path

import nemo_data_designer_plugin.testing.utils as u
import pytest
from nemo_data_designer_plugin.config import get_config
from nhx.testing import ClientContext

pytestmark = pytest.mark.integration


@pytest.fixture
def preview_config(tmp_path: Path) -> Path:
    return u.write_config_file(
        tmp_path,
        f"""
import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text", model={u.ENABLED_MODEL_NAME!r},
                provider={u.OPEN_PROVIDER_NAME!r},
            )
        ]
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="subject",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["math"]),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(name="description", expr="Subject: {{{{ subject }}}}")
    )
    return builder
""",
    )


def test_preview_summary_reports_record_and_column_counts(preview_config: Path, client_context: ClientContext) -> None:
    result = u.invoke_cli(["preview", str(preview_config), "--num-records", "3", "--non-interactive"], client_context)

    assert result.exit_code == 0, result.output
    assert "Preview complete — 3 record(s) generated, 2 column(s)" in result.output
    assert "subject" in result.output
    assert "description" in result.output
    assert "Subject: math" in result.output


def test_rejected_preview_does_not_print_success_summary(preview_config: Path, client_context: ClientContext) -> None:
    too_many_records = get_config().preview_num_records.max + 1
    result = u.invoke_cli(["preview", str(preview_config), "--num-records", str(too_many_records)], client_context)

    assert result.exit_code != 0, result.output
    assert "Max num records" in result.output
    assert "Preview complete —" not in result.output
    assert "traceback" not in result.output.lower()
