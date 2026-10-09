# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``app/task_config.py`` request models."""

from __future__ import annotations

from pathlib import Path

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest, PreviewRequest
from nemo_anonymizer_plugin.config import DEFAULT_MAX_PREVIEW_NUM_RECORDS


def _config() -> AnonymizerConfig:
    return AnonymizerConfig(replace=Redact())


def test_preview_request_accepts_http_source() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
        num_records=5,
    )
    assert req.data.source == "https://example.com/x.csv"


def test_anonymizer_request_accepts_local_path(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    csv.write_text("text\nhello\n")
    req = AnonymizerRequest(
        config=_config(),
        data=AnonymizerInputSpec(source=str(csv), text_column="text"),
    )
    assert req.data.source == str(csv)


def test_anonymizer_request_accepts_fileset_source_without_local_path_validation() -> None:
    req = AnonymizerRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="team-a/pii-inputs#data/input.parquet", text_column="text"),
    )

    assert req.data.source == "team-a/pii-inputs#data/input.parquet"


def test_preview_num_records_advertises_the_default_maximum() -> None:
    schema = PreviewRequest.model_json_schema()["properties"]["num_records"]

    assert schema["minimum"] == 1
    assert schema["maximum"] == DEFAULT_MAX_PREVIEW_NUM_RECORDS
    assert schema["default"] == DEFAULT_MAX_PREVIEW_NUM_RECORDS


def test_preview_num_records_leaves_the_ceiling_to_the_configured_max() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
        num_records=DEFAULT_MAX_PREVIEW_NUM_RECORDS + 1,
    )

    assert req.num_records == DEFAULT_MAX_PREVIEW_NUM_RECORDS + 1


def test_request_dump_preserves_replace_discriminator() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
    )

    dumped = req.model_dump(mode="json")

    assert dumped["config"]["replace"]["kind"] == "redact"
